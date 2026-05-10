"""
train_human_brain.py
--------------------
Three-stage training pipeline for Tutorial 2 (human cortical development,
missing modality inference).

Monitor keys are hardcoded here — NOT in the notebook — to prevent
misconfiguration:
  - Stages 1a, 1b, 2 : DREG_PRE.validation_step  logs "val_elbo_loss"
  - Stage 3           : DREG_DYN.validation_step  logs "val_elbo_loss_d"
"""

import os
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import mmvelo_multi_cond.dataset_all_modality as _dm_mod
from mmvelo_multi_cond.dataset_all_modality import (
    MultiomeHumanBrainDataModule_MissingModalityPrediction,
    DynDataModule_Smooth_MissingModalityPrediction,
)
from mmvelo_multi_cond.models_missingmodality_all_adv_modadv import (
    DREG_PRE, DREG_DYN, EarlyStoppingWithWarmup,
)
from mmvelo_multi_cond.utils import fit_beta_gamma_scale, get_filter_idx_raw


# ── Internal helpers ──────────────────────────────────────────────────────────

def _cat(result, idx):
    return torch.cat([r[idx].cpu() for r in result]).numpy()


def _gpu_flag(device):
    return 1 if device == "cuda" else 0


# ── Data loading ──────────────────────────────────────────────────────────────

def make_datamodule(data_dir="data/human_brain", batch_size=128,
                    min_counts_genes=10, min_counts_peaks=10):
    """
    Create MultiomeHumanBrainDataModule_MissingModalityPrediction,
    redirecting data loading to the local tutorial data directory.
    """
    import scanpy as sc

    def _load_local():
        adata_rna  = sc.read_h5ad(os.path.join(data_dir, "joint_rna_adata.h5ad"))
        adata_atac = sc.read_h5ad(os.path.join(data_dir, "joint_atac_adata.h5ad"))
        return adata_rna, adata_atac

    # Monkeypatch the hardcoded path loader BEFORE importing the DataModule
    _dm_mod.load_greenleaf_missingmodal_data = _load_local

    dm = MultiomeHumanBrainDataModule_MissingModalityPrediction(
        batch_size=batch_size,
        min_counts_genes=min_counts_genes,
        min_counts_peaks=min_counts_peaks,
        batch_sub=False,
        filter_outliers=False,
        pretrain_multi=True,
    )
    print(f"RNA dim   : {dm.rna_dim}")
    print(f"ATAC dim  : {dm.atac_dim}")
    print(f"# cells   : {dm.adata_r.shape[0]}")
    print(f"# cond    : {dm.num_cat}")
    return dm


# ── Stage 1a: Multiome-only pretraining ──────────────────────────────────────

def run_stage1a(dm, run_path, params, device="cuda"):
    """
    Train DREG_PRE on multiome-only cells (Stage 1a).

    Returns
    -------
    model : trained DREG_PRE (in-memory, to be passed directly to run_stage1b
            for a warm-start Stage 1b — matching the reference training script).
            Returns None if the checkpoint already exists.
    """
    ckpt = os.path.join(run_path, "checkpoint_pre_multi.ckpt")
    p = params

    if not os.path.exists(ckpt):
        print("=" * 60)
        print("Stage 1a: Multiome-only pretraining (DREG_PRE)")
        print("=" * 60)

        model = DREG_PRE(
            rna_dim=dm.rna_dim, atac_dim=dm.atac_dim,
            r_h1_dim=p["r_h1dim"], r_h2_dim=p["r_h2dim"],
            a_h1_dim=p["a_h1dim"], a_h2_dim=p["a_h2dim"],
            z_dim=p["zdim"], d_h_dim=p["d_h_dim"],
            cat_dim=dm.num_cat,
            l_prior_r=dm.l_prior_r, l_prior_a=dm.l_prior_a,
            lr=p["lr"], z_learnable=p.get("z_learnable", True),
            d_coeff=p.get("d_coeff", 0.01),
            warmup=p["warmup"],
            llik_scaling=p.get("llik_scaling", True),
            pretrain_multi_end=False,
        )
        model.set_norm_mat(dm)
        model.set_retain_gene_idx(dm)

        logger = TensorBoardLogger(run_path + "/", name="stage1a")
        trainer = pl.Trainer(
            gpus=_gpu_flag(device),
            max_epochs=p["pre_num_epochs"],
            logger=logger,
            callbacks=[
                # DREG_PRE.validation_step logs "val_elbo_loss"
                EarlyStoppingWithWarmup(
                    monitor="val_elbo_loss", mode="min",
                    patience=p["patience"], warmup=p["warmup"], verbose=True,
                ),
                ModelCheckpoint(
                    dirpath=run_path, filename="checkpoint_pre_multi",
                    monitor="val_elbo_loss", save_top_k=1,
                ),
            ],
        )
        trainer.fit(model, dm)
        print("Stage 1a complete.")
        return model  # return in-memory model for warm-start Stage 1b
    else:
        print(f"Stage 1a checkpoint found: {ckpt}")
        return None  # caller should load from checkpoint in run_stage1b


# ── Stage 1b: All modalities ─────────────────────────────────────────────────

def run_stage1b(dm, run_path, params, device="cuda", model=None):
    """
    Fine-tune DREG_PRE on all modalities (Stage 1b) and collect predictions.

    Parameters
    ----------
    model : DREG_PRE returned by run_stage1a, or None.
        If provided, training continues warm from this model object — exactly
        matching the reference script (train_multi_cond_missing_all.py) which
        sets model.pretrain_multi_end=True on the in-memory model without
        reloading the checkpoint.
        If None (checkpoint already existed), loads from checkpoint_pre_multi.ckpt.

    Stores obsm["latent"], layers["rec_s"], layers["rec_u"], layers["s_raw"],
    layers["u_raw"], obsm["lr"] in dm.adata_r, and layers["rec_a"], obsm["la"]
    in dm.adata_a.

    Returns
    -------
    dm : updated in-place
    """
    pre_multi_ckpt = os.path.join(run_path, "checkpoint_pre_multi.ckpt")
    ckpt           = os.path.join(run_path, "checkpoint_pre.ckpt")
    p = params

    # Switch DataModule to all-modality mode
    dm.pretrain_multi_end()

    if not os.path.exists(ckpt):
        print("=" * 60)
        print("Stage 1b: All modalities (DREG_PRE)")
        print("=" * 60)

        if model is not None:
            # Warm-start: continue from Stage 1a in-memory model.
            # Matches reference: dm.pretrain_multi_end() + model.pretrain_multi_end=True
            # followed by a fresh trainer — no checkpoint reload.
            model.pretrain_multi_end = True
            model.set_norm_mat(dm)
            model.set_retain_gene_idx(dm)
        else:
            # Cold-start fallback: load Stage 1a checkpoint.
            model = DREG_PRE.load_from_checkpoint(
                pre_multi_ckpt,
                rna_dim=dm.rna_dim, atac_dim=dm.atac_dim,
                r_h1_dim=p["r_h1dim"], r_h2_dim=p["r_h2dim"],
                a_h1_dim=p["a_h1dim"], a_h2_dim=p["a_h2dim"],
                z_dim=p["zdim"], d_h_dim=p["d_h_dim"],
                cat_dim=dm.num_cat,
                l_prior_r=dm.l_prior_r, l_prior_a=dm.l_prior_a,
                lr=p["lr"], z_learnable=p.get("z_learnable", True),
                d_coeff=p.get("d_coeff", 0.01),
                warmup=p.get("warmup_post_pre", 1),
                llik_scaling=p.get("llik_scaling", True),
                pretrain_multi_end=True, strict=False,
            )
            model.set_norm_mat(dm)
            model.set_retain_gene_idx(dm)

        logger = TensorBoardLogger(run_path + "/", name="stage1b")
        trainer = pl.Trainer(
            gpus=_gpu_flag(device),
            max_epochs=p["num_epochs"],
            logger=logger,
            callbacks=[
                # DREG_PRE.validation_step logs "val_elbo_loss"
                EarlyStoppingWithWarmup(
                    monitor="val_elbo_loss", mode="min",
                    patience=p["patience"],
                    warmup=p.get("warmup_post_pre", 1), verbose=True,
                ),
                ModelCheckpoint(
                    dirpath=run_path, filename="checkpoint_pre",
                    monitor="val_elbo_loss", save_top_k=1,
                ),
            ],
        )
        trainer.fit(model, dm)
        print("Stage 1b complete.")

    # ── Prediction (always from best checkpoint_pre) ───────────────────────────
    print("Running Stage 1b prediction on all cells...")
    model_load = DREG_PRE.load_from_checkpoint(
        ckpt,
        rna_dim=dm.rna_dim, atac_dim=dm.atac_dim,
        r_h1_dim=p["r_h1dim"], r_h2_dim=p["r_h2dim"],
        a_h1_dim=p["a_h1dim"], a_h2_dim=p["a_h2dim"],
        z_dim=p["zdim"], d_h_dim=p["d_h_dim"],
        cat_dim=dm.num_cat,
        l_prior_r=dm.l_prior_r, l_prior_a=dm.l_prior_a,
        lr=p["lr"], z_learnable=p.get("z_learnable", True),
        d_coeff=p.get("d_coeff", 0.01),
        warmup=p.get("warmup_post_pre", 1),
        llik_scaling=p.get("llik_scaling", True),
        pretrain_multi_end=True, strict=False,
    )
    model_load.set_norm_mat(dm)
    model_load.set_retain_gene_idx(dm)

    trainer_pred = pl.Trainer(gpus=_gpu_flag(device), logger=False)
    result = trainer_pred.predict(model_load, dm.all_dataloader())

    # indices: [0]=z [1]=rec_s [2]=rec_u [3]=rec_a [4]=lr [5]=la [6]=s_raw [7]=u_raw
    zs = _cat(result, 0)
    dm.adata_r.obsm["latent"]   = zs
    dm.adata_r.layers["rec_s"]  = _cat(result, 1)
    dm.adata_r.layers["rec_u"]  = _cat(result, 2)
    dm.adata_r.obsm["lr"]       = _cat(result, 4)
    dm.adata_r.layers["s_raw"]  = _cat(result, 6)
    dm.adata_r.layers["u_raw"]  = _cat(result, 7)
    dm.adata_a.obsm["latent"]   = zs
    dm.adata_a.layers["rec_a"]  = _cat(result, 3)
    dm.adata_a.obsm["la"]       = _cat(result, 5)

    print(f"Latent shape: {zs.shape}")
    return dm


# ── Stage 2: Smoothed Profile Reconstruction ─────────────────────────────────

def run_stage2(dm, run_path, params, device="cuda"):
    """
    Build DynDataModule_Smooth_MissingModalityPrediction, fine-tune DREG_PRE
    on smoothed profiles, and run prediction.

    Returns
    -------
    dm_s : DynDataModule_Smooth_MissingModalityPrediction, with s_raw/u_raw/a_raw attached
    """
    pre_ckpt = os.path.join(run_path, "checkpoint_pre.ckpt")
    ckpt     = os.path.join(run_path, "checkpoint_pre_sec.ckpt")
    p = params

    dm_s = DynDataModule_Smooth_MissingModalityPrediction(
        dm, n_neighbors=p["n_neighbors"], modality_wise_smoothing=False,
    )
    print("DynDataModule_Smooth_MissingModalityPrediction ready.")

    if not os.path.exists(ckpt):
        print("=" * 60)
        print("Stage 2: Smoothed Profile Reconstruction (DREG_PRE)")
        print("=" * 60)

        model = DREG_PRE.load_from_checkpoint(
            pre_ckpt,
            rna_dim=dm.rna_dim, atac_dim=dm.atac_dim,
            r_h1_dim=p["r_h1dim"], r_h2_dim=p["r_h2dim"],
            a_h1_dim=p["a_h1dim"], a_h2_dim=p["a_h2dim"],
            z_dim=p["zdim"], d_h_dim=p["d_h_dim"],
            cat_dim=dm_s.num_cat,
            l_prior_r=dm.l_prior_r, l_prior_a=dm.l_prior_a,
            lr=p.get("lr_moment", 1e-2),
            z_learnable=p.get("z_learnable", True),
            d_coeff=p.get("d_coeff", 0.01),
            warmup=p["warmup"],
            llik_scaling=p.get("llik_scaling", True),
            pretrain_first_end=True, strict=False,
        )
        model.set_norm_mat(dm_s)
        model.set_retain_gene_idx(dm_s)

        logger = TensorBoardLogger(run_path + "/", name="stage2")
        trainer = pl.Trainer(
            gpus=_gpu_flag(device),
            max_epochs=p["num_epochs"],
            logger=logger,
            callbacks=[
                # DREG_PRE.validation_step logs "val_elbo_loss"
                EarlyStoppingWithWarmup(
                    monitor="val_elbo_loss", mode="min",
                    patience=p["patience"], warmup=p["warmup"], verbose=True,
                ),
                ModelCheckpoint(
                    dirpath=run_path, filename="checkpoint_pre_sec",
                    monitor="val_elbo_loss", save_top_k=1,
                ),
            ],
        )
        trainer.fit(model, dm_s)
        print("Stage 2 complete.")

    # ── Prediction ────────────────────────────────────────────────────────────
    print("Running Stage 2 prediction on all cells...")
    model_load = DREG_PRE.load_from_checkpoint(
        ckpt,
        rna_dim=dm.rna_dim, atac_dim=dm.atac_dim,
        r_h1_dim=p["r_h1dim"], r_h2_dim=p["r_h2dim"],
        a_h1_dim=p["a_h1dim"], a_h2_dim=p["a_h2dim"],
        z_dim=p["zdim"], d_h_dim=p["d_h_dim"],
        cat_dim=dm_s.num_cat,
        l_prior_r=dm.l_prior_r, l_prior_a=dm.l_prior_a,
        lr=p.get("lr_moment", 1e-2),
        z_learnable=p.get("z_learnable", True),
        d_coeff=p.get("d_coeff", 0.01),
        warmup=p["warmup"],
        llik_scaling=p.get("llik_scaling", True),
        pretrain_first_end=True, strict=False,
    )
    model_load.set_norm_mat(dm_s)
    model_load.set_retain_gene_idx(dm_s)

    trainer_pred = pl.Trainer(gpus=_gpu_flag(device), logger=False)
    result = trainer_pred.predict(model_load, dm_s.all_dataloader())

    dm_s.adata_r.layers["s_raw"] = _cat(result, 6)
    dm_s.adata_r.layers["u_raw"] = _cat(result, 7)
    dm_s.adata_a.layers["a_raw"] = _cat(result, 8)
    return dm_s


# ── Stage 3: Cell State Dynamics ─────────────────────────────────────────────

def run_stage3(dm, dm_s, run_path, params, device="cuda"):
    """
    Train DREG_DYN (Stage 3) and run prediction.

    IMPORTANT: DREG_DYN.validation_step logs "val_elbo_loss_d".
    Both callbacks are hardcoded to monitor "val_elbo_loss_d" here.

    Returns
    -------
    dm_s : updated in-place with velocity layers attached
    """
    pre_sec_ckpt = os.path.join(run_path, "checkpoint_pre_sec.ckpt")
    ckpt         = os.path.join(run_path, "checkpoint.ckpt")
    p = params

    filter_idx = get_filter_idx_raw(
        dm_s,
        test_threshold=-1,
        su_threshold=p.get("su_corr", -1),
        su_ratio=p.get("su_ratio", 50),
        min_counts_su=p.get("min_counts_su", 20),
    )
    dm_s.adata_r.var["estimated_genes"] = (filter_idx.numpy() == 1)
    print(f"Genes for dynamics: {int(filter_idx.sum())} / {dm_s.rna_dim}")

    if not os.path.exists(ckpt):
        print("=" * 60)
        print("Stage 3: Cell State Dynamics (DREG_DYN)")
        print("=" * 60)

        model = DREG_DYN.load_from_checkpoint(
            pre_sec_ckpt,
            rna_dim=dm.rna_dim, atac_dim=dm.atac_dim,
            r_h1_dim=p["r_h1dim"], r_h2_dim=p["r_h2dim"],
            a_h1_dim=p["a_h1dim"], a_h2_dim=p["a_h2dim"],
            z_dim=p["zdim"], d_h_dim=p["d_h_dim"],
            cat_dim=dm.num_cat,
            l_prior_r=dm.l_prior_r, l_prior_a=dm.l_prior_a,
            lr=p["lr_dyn"],
            z_learnable=p.get("z_learnable", True),
            d_coeff=p.get("d_coeff", 0.01),
            filter_idx=filter_idx.to(device), strict=False,
        )
        model.set_norm_mat(dm_s)
        model.log_beta, model.log_gamma = fit_beta_gamma_scale(dm_s, su_scale=False)
        model.set_beta_gamma_ss_ratio()
        model.set_filter_idx(filter_idx)
        model.set_grad_for_training()
        model.set_retain_gene_idx(dm_s)

        logger = TensorBoardLogger(run_path + "/", name="stage3")
        trainer = pl.Trainer(
            gpus=_gpu_flag(device),
            max_epochs=p["num_epochs"],
            logger=logger,
            callbacks=[
                # DREG_DYN.validation_step logs "val_elbo_loss_d"
                # Using "val_elbo_loss" here causes MisconfigurationException
                EarlyStoppingWithWarmup(
                    monitor="val_elbo_loss_d", mode="min",
                    patience=p["patience"],
                    warmup=p.get("warmup_dyn", 0), verbose=True,
                ),
                ModelCheckpoint(
                    dirpath=run_path, filename="checkpoint",
                    monitor="val_elbo_loss_d", save_top_k=1,
                ),
            ],
        )
        trainer.fit(model, dm_s)
        print("Stage 3 complete.")

    # ── Prediction ────────────────────────────────────────────────────────────
    print("Running Stage 3 prediction on all cells...")
    model_load = DREG_DYN.load_from_checkpoint(
        ckpt,
        rna_dim=dm.rna_dim, atac_dim=dm.atac_dim,
        r_h1_dim=p["r_h1dim"], r_h2_dim=p["r_h2dim"],
        a_h1_dim=p["a_h1dim"], a_h2_dim=p["a_h2dim"],
        z_dim=p["zdim"], d_h_dim=p["d_h_dim"],
        cat_dim=dm.num_cat,
        l_prior_r=dm.l_prior_r, l_prior_a=dm.l_prior_a,
        lr=p["lr_dyn"],
        z_learnable=p.get("z_learnable", True),
        d_coeff=p.get("d_coeff", 0.01),
        filter_idx=filter_idx.to(device), strict=False,
    )
    model_load.set_norm_mat(dm_s)
    model_load.set_filter_idx(filter_idx)
    model_load.set_retain_gene_idx(dm_s)
    # compute_dadt must be set AFTER training, only for prediction
    model_load.compute_dadt = True

    trainer_pred = pl.Trainer(gpus=_gpu_flag(device), logger=False)
    # Do NOT pass ckpt_path: the trainer would reload with strict=True, rejecting
    # "log_gamma_beta_ss" added by set_beta_gamma_ss_ratio() during training.
    result = trainer_pred.predict(model_load, dm_s.all_dataloader())

    # indices: [0]=z [1]=d [2]=dsdt [3]=dsdt_obs [4]=d_var [5]=dadt [6]=dudt
    dm_s.adata_r.obsm["latent"]     = _cat(result, 0)
    dm_s.adata_r.obsm["dynamics"]   = _cat(result, 1)
    dm_s.adata_r.layers["dsdt"]     = _cat(result, 2)
    dm_s.adata_r.layers["dsdt_obs"] = _cat(result, 3)
    dm_s.adata_r.obsm["d_var"]      = _cat(result, 4) ** 2
    dm_s.adata_a.layers["dadt"]     = _cat(result, 5)
    dm_s.adata_r.layers["dudt"]     = _cat(result, 6)

    print(f"dsdt shape : {dm_s.adata_r.layers['dsdt'].shape}")
    print(f"dadt shape : {dm_s.adata_a.layers['dadt'].shape}")
    return dm_s


# ── Save ─────────────────────────────────────────────────────────────────────

def save_results(dm_s, run_path):
    dm_s.adata_r.write_loom(
        os.path.join(run_path, "adata_rna.loom"), write_obsm_varm=True)
    dm_s.adata_a.write_loom(
        os.path.join(run_path, "adata_atac.loom"), write_obsm_varm=True)
    print(f"Results saved to {run_path}/")
