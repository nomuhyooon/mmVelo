"""
train_mouse_brain.py
--------------------
Three-stage training pipeline for Tutorial 1 (embryonic mouse brain).

Monitor keys are hardcoded here — NOT in the notebook — to prevent
misconfiguration:
  - Stage 1 & 2 : DREG_PRE.validation_step  logs "val_elbo_loss"
  - Stage 3      : DREG_DYN.validation_step  logs "val_elbo_loss_d"
"""

import os
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from mmvelo_multi.models import DREG_PRE, DREG_DYN, EarlyStoppingWithWarmup
from mmvelo_multi.dataset import DynDataModule_Smooth
from mmvelo_multi.utils import fit_beta_gamma_scale, get_filter_idx_raw


# ── Internal helpers ──────────────────────────────────────────────────────────

def _cat(result, idx):
    return torch.cat([r[idx].cpu() for r in result]).numpy()


def _gpu_flag(device):
    return 1 if device == "cuda" else 0


# ── Stage 1: Cell State Inference ─────────────────────────────────────────────

def run_stage1(dm, run_path, params, device="cuda"):
    """
    Train DREG_PRE (Stage 1) and run prediction on all cells.

    Stores the following in dm.adata_r / dm.adata_a:
      obsm["latent"], layers["rec_s"], layers["rec_u"], layers["s_raw"],
      layers["u_raw"], obsm["lr"], obsm["la"], layers["rec_a"]

    Parameters
    ----------
    dm        : TutorialBrainDataModule
    run_path  : str  — directory for checkpoints and TensorBoard logs
    params    : dict — must contain: zdim, r_h1dim, r_h2dim, a_h1dim, a_h2dim,
                       d_h_dim, lr, num_epochs, patience, warmup
    device    : "cuda" or "cpu"

    Returns
    -------
    dm        : updated in-place (layers/obsm attached)
    """
    ckpt = os.path.join(run_path, "checkpoint_pre.ckpt")
    p = params

    # ── Training (skipped if checkpoint exists) ───────────────────────────────
    if not os.path.exists(ckpt):
        print("=" * 60)
        print("Stage 1: Cell State Inference (DREG_PRE)")
        print("=" * 60)

        model = DREG_PRE(
            rna_dim=dm.rna_dim, atac_dim=dm.atac_dim,
            r_h1_dim=p["r_h1dim"], r_h2_dim=p["r_h2dim"],
            a_h1_dim=p["a_h1dim"], a_h2_dim=p["a_h2dim"],
            z_dim=p["zdim"], d_h_dim=p["d_h_dim"],
            l_prior_r=dm.l_prior_r, l_prior_a=dm.l_prior_a,
            lr=p["lr"], z_learnable=True, d_coeff=1e-2,
            warmup=p["warmup"], llik_scaling=True,
        )
        model.set_norm_mat(dm)
        model.set_retain_gene_idx(dm)

        logger = TensorBoardLogger(run_path, name="stage1")
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
                    dirpath=run_path, filename="checkpoint_pre",
                    monitor="val_elbo_loss", save_top_k=1,
                ),
            ],
        )
        trainer.fit(model=model, datamodule=dm)
        print("Stage 1 training complete.")

    # ── Prediction ────────────────────────────────────────────────────────────
    print("Running Stage 1 prediction on all cells...")
    model_load = DREG_PRE.load_from_checkpoint(
        ckpt,
        rna_dim=dm.rna_dim, atac_dim=dm.atac_dim,
        r_h1_dim=p["r_h1dim"], r_h2_dim=p["r_h2dim"],
        a_h1_dim=p["a_h1dim"], a_h2_dim=p["a_h2dim"],
        z_dim=p["zdim"], d_h_dim=p["d_h_dim"],
        l_prior_r=dm.l_prior_r, l_prior_a=dm.l_prior_a,
        lr=p["lr"], z_learnable=True, d_coeff=1e-2,
        warmup=p["warmup"], llik_scaling=True, strict=False,
    )
    model_load.set_norm_mat(dm)
    model_load.set_retain_gene_idx(dm)

    trainer_pred = pl.Trainer(gpus=_gpu_flag(device), logger=False)
    result = trainer_pred.predict(model_load, dm.all_dataloader())

    # indices: [0]=z [1]=rec_s [2]=rec_u [3]=rec_a [4]=lr [5]=la [6]=s_raw [7]=u_raw
    dm.adata_r.obsm["latent"]   = _cat(result, 0)
    dm.adata_r.layers["rec_s"]  = _cat(result, 1)
    dm.adata_r.layers["rec_u"]  = _cat(result, 2)
    dm.adata_a.layers["rec_a"]  = _cat(result, 3)
    dm.adata_r.obsm["lr"]       = _cat(result, 4).reshape(-1, 1)
    dm.adata_a.obsm["la"]       = _cat(result, 5).reshape(-1, 1)
    dm.adata_r.layers["s_raw"]  = _cat(result, 6)
    dm.adata_r.layers["u_raw"]  = _cat(result, 7)

    print(f"Latent shape: {dm.adata_r.obsm['latent'].shape}")
    return dm


# ── Stage 2: Smoothed Profile Reconstruction ──────────────────────────────────

def build_smooth_dm(dm, params):
    """Create DynDataModule_Smooth from a Stage-1-fitted DataModule."""
    dm_s = DynDataModule_Smooth(dm, batch_size=params["batch_size"],
                                n_neighbors=params["n_neighbors"])
    print("DynDataModule_Smooth ready.")
    return dm_s


def run_stage2(dm, dm_s, run_path, params, device="cuda"):
    """
    Fine-tune DREG_PRE on smoothed profiles (Stage 2) and run prediction.

    Stores layers["s_raw"], layers["u_raw"] in dm_s.adata_r
    and layers["a_raw"] in dm_s.adata_a.

    Returns
    -------
    dm_s : updated in-place
    """
    pre_ckpt = os.path.join(run_path, "checkpoint_pre.ckpt")
    ckpt     = os.path.join(run_path, "checkpoint_pre_sec.ckpt")
    p = params

    if not os.path.exists(ckpt):
        print("=" * 60)
        print("Stage 2: Smoothed Profile Reconstruction (DREG_PRE fine-tune)")
        print("=" * 60)

        model = DREG_PRE.load_from_checkpoint(
            pre_ckpt,
            rna_dim=dm_s.rna_dim, atac_dim=dm_s.atac_dim,
            r_h1_dim=p["r_h1dim"], r_h2_dim=p["r_h2dim"],
            a_h1_dim=p["a_h1dim"], a_h2_dim=p["a_h2dim"],
            z_dim=p["zdim"], d_h_dim=p["d_h_dim"],
            l_prior_r=dm.l_prior_r, l_prior_a=dm.l_prior_a,
            lr=p["lr"], z_learnable=True, d_coeff=1e-2,
            warmup=p["warmup"], llik_scaling=True,
            pretrain_first_end=True, strict=False,
        )
        model.set_norm_mat(dm_s)
        model.set_retain_gene_idx(dm_s)

        logger = TensorBoardLogger(run_path, name="stage2")
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
        trainer.fit(model=model, datamodule=dm_s)
        print("Stage 2 training complete.")

    # ── Prediction ────────────────────────────────────────────────────────────
    print("Running Stage 2 prediction on all cells...")
    model_load = DREG_PRE.load_from_checkpoint(
        ckpt,
        rna_dim=dm_s.rna_dim, atac_dim=dm_s.atac_dim,
        r_h1_dim=p["r_h1dim"], r_h2_dim=p["r_h2dim"],
        a_h1_dim=p["a_h1dim"], a_h2_dim=p["a_h2dim"],
        z_dim=p["zdim"], d_h_dim=p["d_h_dim"],
        l_prior_r=dm.l_prior_r, l_prior_a=dm.l_prior_a,
        lr=p["lr"], z_learnable=True, d_coeff=1e-2,
        warmup=p["warmup"], llik_scaling=True,
        pretrain_first_end=True, strict=False,
    )
    model_load.set_norm_mat(dm_s)
    model_load.set_retain_gene_idx(dm_s)

    trainer_pred = pl.Trainer(gpus=_gpu_flag(device), logger=False)
    result = trainer_pred.predict(model_load, dm_s.all_dataloader())

    # indices: [6]=s_raw [7]=u_raw [8]=a_raw
    dm_s.adata_r.layers["s_raw"] = _cat(result, 6)
    dm_s.adata_r.layers["u_raw"] = _cat(result, 7)
    dm_s.adata_a.layers["a_raw"] = _cat(result, 8)

    print(f"s_raw shape: {dm_s.adata_r.layers['s_raw'].shape}")
    return dm_s


# ── Stage 3: Cell State Dynamics ─────────────────────────────────────────────

def run_stage3(dm, dm_s, run_path, params, device="cuda"):
    """
    Train DREG_DYN (Stage 3) and run prediction.

    IMPORTANT: DREG_DYN.validation_step logs "val_elbo_loss_d".
    Both EarlyStoppingWithWarmup and ModelCheckpoint are hardcoded
    to monitor "val_elbo_loss_d" here.  Do NOT change this.

    Stores obsm["latent"], obsm["dynamics"], layers["dsdt"],
    layers["dsdt_obs"], layers["dudt"] in dm_s.adata_r and
    layers["dadt"] in dm_s.adata_a.

    Returns
    -------
    dm_s : updated in-place
    """
    pre_sec_ckpt = os.path.join(run_path, "checkpoint_pre_sec.ckpt")
    ckpt         = os.path.join(run_path, "checkpoint.ckpt")
    p = params

    # Gene filter
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
            rna_dim=dm_s.rna_dim, atac_dim=dm_s.atac_dim,
            r_h1_dim=p["r_h1dim"], r_h2_dim=p["r_h2dim"],
            a_h1_dim=p["a_h1dim"], a_h2_dim=p["a_h2dim"],
            z_dim=p["zdim"], d_h_dim=p["d_h_dim"],
            l_prior_r=dm.l_prior_r, l_prior_a=dm.l_prior_a,
            lr=p["lr_dyn"], z_learnable=True, d_coeff=1e-2,
            warmup=p["warmup_dyn"],
            filter_idx=filter_idx.to(device), strict=False,
        )
        model.set_norm_mat(dm_s)
        model.log_beta, model.log_gamma = fit_beta_gamma_scale(dm_s, su_scale=False)
        model.set_beta_gamma_ss_ratio()
        model.set_filter_idx(filter_idx)
        model.set_grad_for_training()
        model.set_retain_gene_idx(dm_s)

        logger = TensorBoardLogger(run_path, name="stage3")
        trainer = pl.Trainer(
            gpus=_gpu_flag(device),
            max_epochs=p["num_epochs"],
            logger=logger,
            callbacks=[
                # DREG_DYN.validation_step logs "val_elbo_loss_d"
                # Using "val_elbo_loss" here causes MisconfigurationException
                EarlyStoppingWithWarmup(
                    monitor="val_elbo_loss_d", mode="min",
                    patience=p["patience"], warmup=p["warmup_dyn"], verbose=True,
                ),
                ModelCheckpoint(
                    dirpath=run_path, filename="checkpoint",
                    monitor="val_elbo_loss_d", save_top_k=1,
                ),
            ],
        )
        trainer.fit(model=model, datamodule=dm_s)
        print("Stage 3 training complete.")

    # ── Prediction ────────────────────────────────────────────────────────────
    print("Running Stage 3 prediction on all cells...")
    model_load = DREG_DYN.load_from_checkpoint(
        ckpt,
        rna_dim=dm_s.rna_dim, atac_dim=dm_s.atac_dim,
        r_h1_dim=p["r_h1dim"], r_h2_dim=p["r_h2dim"],
        a_h1_dim=p["a_h1dim"], a_h2_dim=p["a_h2dim"],
        z_dim=p["zdim"], d_h_dim=p["d_h_dim"],
        l_prior_r=dm.l_prior_r, l_prior_a=dm.l_prior_a,
        lr=p["lr_dyn"], z_learnable=True, d_coeff=1e-2,
        warmup=p["warmup_dyn"],
        filter_idx=filter_idx.to(device), strict=False,
    )
    model_load.set_norm_mat(dm_s)
    model_load.set_filter_idx(filter_idx)
    model_load.set_retain_gene_idx(dm_s)
    # compute_dadt must be True AFTER training, only for prediction
    model_load.compute_dadt = True

    trainer_pred = pl.Trainer(gpus=_gpu_flag(device), logger=False)
    # Do NOT pass ckpt_path here: the trainer would reload the checkpoint with
    # strict=True, which rejects "log_gamma_beta_ss" added by set_beta_gamma_ss_ratio().
    # We already loaded the model via load_from_checkpoint above.
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


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_results(dm_s, run_path):
    """Save AnnData loom files and normalisation matrices."""
    dm_s.adata_r.write_loom(
        os.path.join(run_path, "adata_rna.loom"), write_obsm_varm=True)
    dm_s.adata_a.write_loom(
        os.path.join(run_path, "adata_atac.loom"), write_obsm_varm=True)
    import numpy as np
    np.savetxt(os.path.join(run_path, "norm_mat_s.txt"), dm_s.norm_mat_r[0])
    np.savetxt(os.path.join(run_path, "norm_mat_u.txt"), dm_s.norm_mat_r[1])
    np.savetxt(os.path.join(run_path, "norm_mat_a.txt"), dm_s.norm_mat_a)
    print(f"Results saved to {run_path}/")
