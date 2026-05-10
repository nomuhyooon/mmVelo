"""
Ablation study for mmVelo:
  1. With vs. without KNN smoothing (zdim = 10)
  2. Multiple latent dimensionalities: zdim = 5, 10, 30, 50 (with smoothing)

For each condition the script:
  - Trains mmVelo from scratch (skips a stage if a checkpoint already exists)
  - Generates a latent-space streamline plot using the shared UMAP / cluster labels
    from the baseline run (smooth_z10)
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse
import scipy.stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
import scvelo as scv
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

sys.path.append("/home/nomura/Proj/mmvelo/src")
from mmvelo_multi.dataset import MultiomeBrainDataModule_Pre, DynDataModule_Smooth
from mmvelo_multi.utils import (
    fit_beta_gamma, fit_beta_gamma_scale, get_filter_idx_raw,
)
from mmvelo_multi.models import DREG_PRE, DREG_DYN, EarlyStoppingWithWarmup
from mmvelo_multi.streamlineplot import velocity_graph

# ============================================================
# Paths and ablation conditions
# ============================================================

BASE_RUN_PATH = (
    "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN"
    "/2023-05-07T15:19:02_s43_k100_for_analysis"
)
ANNDATA_DIR = BASE_RUN_PATH + "/downstream_analysis/result/anndata"
ABLATION_OUTPUT_DIR = (
    BASE_RUN_PATH + "/downstream_analysis/result/ablation"
)

# Each entry: name (used for directory), zdim, use smoothing
ABLATION_CONDITIONS = [
    {"name": "smooth_z5",    "zdim":  5, "smooth": True},
    {"name": "smooth_z10",   "zdim": 10, "smooth": True},   # baseline
    {"name": "smooth_z30",   "zdim": 30, "smooth": True},
    {"name": "smooth_z50",   "zdim": 50, "smooth": True},
    {"name": "nosmooth_z10", "zdim": 10, "smooth": False},
]

# Hyperparameters matching the baseline run
DEFAULT_PARAMS = dict(
    seed=43,
    r_h1dim=128, r_h2dim=64, a_h1dim=128, a_h2dim=64,
    d_h_dim=64,
    z_learnable=True, d_coeff=1e-2,
    num_epochs=1000,
    lr=1e-4, lr_moment=1e-2, lr_dyn=1e-4,
    su_corr=-1, su_ratio=50, min_counts_su=20,
    llik_scaling=True, patience=30,
    n_neighbors=100,
    warmup=30, warmup_dyn=10,
    batch_size=128,
    n_genes=3000, n_peaks=20000,
    min_counts_genes=10, min_counts_peaks=10,
)

# ============================================================
# DynDataModule without smoothing
# ============================================================

class DynDataModule_NoSmooth(DynDataModule_Smooth):
    """Identical to DynDataModule_Smooth but skips KNN-based moment smoothing.

    Instead of computing smoothed moments (Ms, Mu, Ma) via sc.pp.moments,
    the log-normalised counts are used directly as the moment matrices.
    """

    def calc_moments(self):
        s = self.adata_r.layers["spliced"]
        u = self.adata_r.layers["unspliced"]
        a = self.adata_a.X

        self.adata_r.layers["Ms"] = (
            s.toarray() if scipy.sparse.issparse(s) else np.array(s)
        ).astype(np.float32)
        self.adata_r.layers["Mu"] = (
            u.toarray() if scipy.sparse.issparse(u) else np.array(u)
        ).astype(np.float32)
        self.adata_a.layers["Ma"] = (
            a.toarray() if scipy.sparse.issparse(a) else np.array(a)
        ).astype(np.float32)


# ============================================================
# Training helpers
# ============================================================

def _collect_list(result, idx):
    return torch.cat([r[idx].cpu() for r in result]).numpy()


def run_training(condition, params, run_dir):
    """Run the three-stage mmVelo training pipeline for one ablation condition.

    Stages are skipped if the corresponding checkpoint already exists.
    Returns (adata_r, adata_a) with all inferred quantities populated.
    """
    zdim = condition["zdim"]
    smooth = condition["smooth"]
    os.makedirs(run_dir, exist_ok=True)

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(params["seed"])
    np.random.seed(params["seed"])

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=run_dir + "/")

    # ------------------------------------------------------------------
    # Load raw data
    # ------------------------------------------------------------------
    print(f"  Loading DataModule …")
    dm = MultiomeBrainDataModule_Pre(
        batch_size=params["batch_size"],
        n_top_genes=params["n_genes"],
        n_top_peaks=params["n_peaks"],
        min_counts_genes=params["min_counts_genes"],
        min_counts_peaks=params["min_counts_peaks"],
    )

    # ------------------------------------------------------------------
    # Stage 1: pre-training (encoder / decoder)
    # ------------------------------------------------------------------
    pre_ckpt = run_dir + "/checkpoint_pre.ckpt"
    trainer_pre = pl.Trainer(
        gpus=1, max_epochs=params["num_epochs"], logger=tb_logger,
        callbacks=[
            EarlyStoppingWithWarmup(
                monitor="val_elbo_loss", mode="min",
                patience=params["patience"], warmup=params["warmup"],
                verbose=True,
            ),
            ModelCheckpoint(
                dirpath=run_dir, filename="checkpoint_pre",
                monitor="val_elbo_loss", save_top_k=1,
            ),
        ],
    )

    if not os.path.exists(pre_ckpt):
        print(f"  Stage 1: pre-training …")
        model_pre = DREG_PRE(
            dm.rna_dim, dm.atac_dim,
            params["r_h1dim"], params["r_h2dim"],
            params["a_h1dim"], params["a_h2dim"],
            zdim, params["d_h_dim"],
            dm.l_prior_r, dm.l_prior_a, params["lr"],
            z_learnable=params["z_learnable"],
            d_coeff=params["d_coeff"],
            warmup=params["warmup"],
            llik_scaling=params["llik_scaling"],
        )
        model_pre.set_norm_mat(dm)
        model_pre.set_retain_gene_idx(dm)
        trainer_pre.fit(model=model_pre, datamodule=dm)
    else:
        print(f"  Stage 1: checkpoint found, skipping training.")

    print(f"  Stage 1: running inference …")
    model_pre_infer = DREG_PRE(
        dm.rna_dim, dm.atac_dim,
        params["r_h1dim"], params["r_h2dim"],
        params["a_h1dim"], params["a_h2dim"],
        zdim, params["d_h_dim"],
        dm.l_prior_r, dm.l_prior_a, params["lr"],
        z_learnable=params["z_learnable"],
        d_coeff=params["d_coeff"],
        warmup=params["warmup"],
        llik_scaling=params["llik_scaling"],
    )
    model_pre_infer.set_norm_mat(dm)
    model_pre_infer.set_retain_gene_idx(dm)
    result = trainer_pre.predict(
        model=model_pre_infer,
        dataloaders=dm.all_dataloader(),
        ckpt_path=pre_ckpt,
    )
    dm.adata_r.obsm["latent"]  = _collect_list(result, 0)
    dm.adata_r.layers["rec_s"] = _collect_list(result, 1)
    dm.adata_r.layers["rec_u"] = _collect_list(result, 2)
    dm.adata_r.obsm["lr"]      = _collect_list(result, 4)
    dm.adata_r.layers["s_raw"] = _collect_list(result, 6)
    dm.adata_r.layers["u_raw"] = _collect_list(result, 7)
    dm.adata_a.obsm["latent"]  = dm.adata_r.obsm["latent"]
    dm.adata_a.layers["rec_a"] = _collect_list(result, 3)
    dm.adata_a.obsm["la"]      = _collect_list(result, 5)
    del result

    print(f"  Stage 1: plotting reconstruction correlations …")
    plot_pretrain_corr(dm, run_dir + "/corr")

    sc.pp.neighbors(dm.adata_r, n_neighbors=15, use_rep="latent")

    # ------------------------------------------------------------------
    # Stage 2: second pre-training with (or without) smoothing
    # ------------------------------------------------------------------
    DataModuleClass = DynDataModule_Smooth if smooth else DynDataModule_NoSmooth
    dm_s = DataModuleClass(dm, n_neighbors=params["n_neighbors"])

    pre_sec_ckpt = run_dir + "/checkpoint_pre_sec.ckpt"
    trainer_sec = pl.Trainer(
        gpus=1, max_epochs=params["num_epochs"], logger=tb_logger,
        callbacks=[
            EarlyStoppingWithWarmup(
                monitor="val_elbo_loss", mode="min",
                patience=params["patience"], warmup=params["warmup"],
                verbose=True,
            ),
            ModelCheckpoint(
                dirpath=run_dir, filename="checkpoint_pre_sec",
                monitor="val_elbo_loss", save_top_k=1,
            ),
        ],
    )

    if not os.path.exists(pre_sec_ckpt):
        print(f"  Stage 2: pre-training (second) …")
        model_sec = DREG_PRE.load_from_checkpoint(
            pre_ckpt,
            rna_dim=dm.rna_dim, atac_dim=dm.atac_dim,
            r_h1_dim=params["r_h1dim"], r_h2_dim=params["r_h2dim"],
            a_h1_dim=params["a_h1dim"], a_h2_dim=params["a_h2dim"],
            z_dim=zdim, d_h_dim=params["d_h_dim"],
            l_prior_r=dm.l_prior_r, l_prior_a=dm.l_prior_a,
            lr=params["lr_moment"],
            z_learnable=params["z_learnable"],
            d_coeff=params["d_coeff"],
            warmup=params["warmup"],
            llik_scaling=params["llik_scaling"],
            pretrain_first_end=True, strict=False,
        )
        model_sec.set_norm_mat(dm_s)
        model_sec.set_retain_gene_idx(dm_s)
        trainer_sec.fit(model=model_sec, datamodule=dm_s)
    else:
        print(f"  Stage 2: checkpoint found, skipping training.")

    print(f"  Stage 2: running inference …")
    model_sec_infer = DREG_PRE(
        dm.rna_dim, dm.atac_dim,
        params["r_h1dim"], params["r_h2dim"],
        params["a_h1dim"], params["a_h2dim"],
        zdim, params["d_h_dim"],
        dm.l_prior_r, dm.l_prior_a, params["lr_moment"],
        z_learnable=params["z_learnable"],
        d_coeff=params["d_coeff"],
        warmup=params["warmup"],
        llik_scaling=params["llik_scaling"],
        pretrain_first_end=True,
    )
    model_sec_infer.set_norm_mat(dm_s)
    model_sec_infer.set_retain_gene_idx(dm_s)
    result = trainer_sec.predict(
        model=model_sec_infer,
        dataloaders=dm_s.all_dataloader(),
        ckpt_path=pre_sec_ckpt,
    )
    dm_s.adata_r.layers["s_raw"] = _collect_list(result, 6)
    dm_s.adata_r.layers["u_raw"] = _collect_list(result, 7)
    dm_s.adata_a.layers["a_raw"] = _collect_list(result, 8)
    del result

    filter_idx = get_filter_idx_raw(
        dm_s,
        test_threshold=-1,
        su_threshold=params["su_corr"],
        su_ratio=params["su_ratio"],
        min_counts_su=params["min_counts_su"],
    )
    dm_s.adata_r.var["estimated_genes"] = (filter_idx.numpy() == 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Stage 3: dynamics training
    # ------------------------------------------------------------------
    dyn_ckpt = run_dir + "/checkpoint.ckpt"
    trainer_dyn = pl.Trainer(
        gpus=1, max_epochs=params["num_epochs"], logger=tb_logger,
        callbacks=[
            EarlyStoppingWithWarmup(
                monitor="val_elbo_loss_d", mode="min",
                patience=params["patience"],
                warmup=params["warmup_dyn"],
                verbose=True,
            ),
            ModelCheckpoint(
                dirpath=run_dir, filename="checkpoint",
                monitor="val_elbo_loss_d", save_top_k=1,
            ),
        ],
    )

    if not os.path.exists(dyn_ckpt):
        print(f"  Stage 3: dynamics training …")
        model_dyn = DREG_DYN.load_from_checkpoint(
            pre_sec_ckpt,
            rna_dim=dm.rna_dim, atac_dim=dm.atac_dim,
            r_h1_dim=params["r_h1dim"], r_h2_dim=params["r_h2dim"],
            a_h1_dim=params["a_h1dim"], a_h2_dim=params["a_h2dim"],
            z_dim=zdim, d_h_dim=params["d_h_dim"],
            l_prior_r=dm.l_prior_r, l_prior_a=dm.l_prior_a,
            lr=params["lr_dyn"],
            z_learnable=params["z_learnable"],
            d_coeff=params["d_coeff"],
            filter_idx=filter_idx.to(device),
            strict=False,
        )
        model_dyn.set_norm_mat(dm_s)
        model_dyn.log_gamma_beta = fit_beta_gamma(dm_s)
        model_dyn.log_beta, model_dyn.log_gamma = fit_beta_gamma_scale(
            dm_s, su_scale=False
        )
        model_dyn.set_beta_gamma_ss_ratio()
        model_dyn.set_filter_idx(filter_idx)
        model_dyn.set_grad_for_training()
        model_dyn.set_retain_gene_idx(dm_s)
        trainer_dyn.fit(model=model_dyn, datamodule=dm_s)
    else:
        print(f"  Stage 3: checkpoint found, skipping training.")

    print(f"  Stage 3: running inference …")
    model_dyn_infer = DREG_DYN(
        dm.rna_dim, dm.atac_dim,
        params["r_h1dim"], params["r_h2dim"],
        params["a_h1dim"], params["a_h2dim"],
        zdim, params["d_h_dim"],
        dm.l_prior_r, dm.l_prior_a, params["lr_dyn"],
        z_learnable=params["z_learnable"],
        d_coeff=params["d_coeff"],
    )
    model_dyn_infer.set_norm_mat(dm_s)
    model_dyn_infer.set_retain_gene_idx(dm_s)
    model_dyn_infer.set_filter_idx(filter_idx.to(device))
    # Register log_gamma_beta (Parameter) and log_gamma_beta_ss (buffer)
    # so the checkpoint state_dict keys match. Values will be overwritten on load.
    model_dyn_infer.log_gamma_beta = fit_beta_gamma(dm_s)
    model_dyn_infer.set_beta_gamma_ss_ratio()
    model_dyn_infer.compute_dadt = True

    result = trainer_dyn.predict(
        model=model_dyn_infer,
        dataloaders=dm_s.all_dataloader(),
        ckpt_path=dyn_ckpt,
    )
    dm_s.adata_r.obsm["latent"]    = _collect_list(result, 0)
    dm_s.adata_r.obsm["dynamics"]  = _collect_list(result, 1)
    dm_s.adata_r.layers["dsdt"]    = _collect_list(result, 2)
    dm_s.adata_r.layers["dsdt_obs"]= _collect_list(result, 3)
    dm_s.adata_r.obsm["d_var"]     = _collect_list(result, 4) ** 2
    dm_s.adata_a.layers["dadt"]    = _collect_list(result, 5)
    dm_s.adata_r.layers["dudt"]    = _collect_list(result, 6)
    del result

    return dm_s.adata_r, dm_s.adata_a


# ============================================================
# Streamline plot
# ============================================================

def make_streamline_plot(adata_r, adata_a, save_dir, clusters, umap_coord,
                         n_neighbors=100):
    """Generate latent-space streamline plot and save PNG files."""
    os.makedirs(save_dir, exist_ok=True)

    adata_r.obsm["X_umap"] = umap_coord
    adata_r.obs["clusters"] = clusters

    sc.pp.neighbors(adata_r, n_neighbors=n_neighbors, use_rep="latent")
    adata_a.uns["neighbors"]      = adata_r.uns["neighbors"]
    adata_a.obsp["distances"]     = adata_r.obsp["distances"]
    adata_a.obsp["connectivities"]= adata_r.obsp["connectivities"]

    if scipy.sparse.issparse(adata_r.layers["dsdt"]):
        adata_r.layers["dsdt"] = adata_r.layers["dsdt"].toarray()

    # Build latent-space AnnData for streamline computation
    adata_z = ad.AnnData(X=adata_r.obsm["latent"])
    adata_z.obs_names            = adata_r.obs_names
    adata_z.obsm["latent"]       = adata_r.obsm["latent"]
    adata_z.layers["latent"]     = adata_r.obsm["latent"]
    adata_z.layers["dynamics"]   = adata_r.obsm["dynamics"]
    adata_z.obsm["X_umap"]       = adata_r.obsm["X_umap"]
    adata_z.obsp["distances"]    = adata_r.obsp["distances"]
    adata_z.obsp["connectivities"] = adata_r.obsp["connectivities"]
    adata_z.uns["neighbors"]     = adata_r.uns["neighbors"]

    velocity_graph(adata_z, vkey="dynamics", xkey="latent", n_jobs=16)
    scv.tl.velocity_embedding(adata_z, basis="umap", vkey="dynamics")

    adata_z.obs["clusters"] = adata_r.obs["clusters"].values
    if "clusters_colors" in adata_r.uns:
        adata_z.uns["clusters_colors"] = adata_r.uns["clusters_colors"]

    scv.pl.velocity_embedding_stream(
        adata_z, vkey="dynamics", color="clusters",
        save=save_dir + "/dzdt_streamline_clusters_tanh.png",
        title="", dpi=300, legend_loc="none",
    )
    scv.pl.velocity_embedding_stream(
        adata_z, vkey="dynamics", color="clusters",
        save=save_dir + "/dzdt_streamline_clusters_tanh_with_legend.png",
        title="", dpi=300, legend_loc="right margin",
    )
    print(f"  Streamline plots saved to: {save_dir}")


# ============================================================
# Gene-wise / peak-wise reconstruction correlation
# ============================================================

def _to_dense(mat):
    """Convert sparse matrix to dense ndarray if necessary."""
    if scipy.sparse.issparse(mat):
        return mat.toarray()
    return np.array(mat)


def colwise_pearsonr(r_c, r_ld):
    """Column-wise Pearson correlation between two 2-D arrays."""
    return np.array([
        scipy.stats.pearsonr(r_c[:, i], r_ld[:, i])[0]
        for i in range(r_c.shape[1])
    ])


def plot_pretrain_corr(dm, save_dir):
    """Plot gene-wise (spliced, unspliced) and peak-wise (ATAC) Pearson
    correlation between observed counts and pre-training reconstruction.

    Format matches plot_genewise_corr_for_fig / plot_peakwise_corr_for_fig
    in mmvelo_multi/utils.py:
      - figsize (5, 5), dpi=300
      - blue = train, red = test
      - s=1 for genes, s=0.03 for peaks
      - x-axis: log10 total count
      - y-axis: pearson correlation
    """
    os.makedirs(save_dir, exist_ok=True)
    train_idx = dm.idx["train"]
    test_idx  = dm.idx["test"]

    # ── Spliced ──────────────────────────────────────────────────────────
    obs_s = _to_dense(dm.adata_r.layers["spliced"])
    rec_s = dm.adata_r.layers["rec_s"]
    log_count_s  = np.log10(obs_s.sum(axis=0))
    train_corr_s = colwise_pearsonr(obs_s[train_idx], rec_s[train_idx])
    test_corr_s  = colwise_pearsonr(obs_s[test_idx],  rec_s[test_idx])

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    ax.scatter(log_count_s, train_corr_s, color="blue", s=1, label="train")
    ax.scatter(log_count_s, test_corr_s,  color="red",  s=1, label="test")
    ax.set_xlabel("log10 total count")
    ax.set_ylabel("pearson correlation")
    ax.legend()
    ax.set_title("spliced")
    fig.savefig(save_dir + "/train_test_corr_s_for_fig.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    # ── Unspliced ─────────────────────────────────────────────────────────
    obs_u = _to_dense(dm.adata_r.layers["unspliced"])
    rec_u = dm.adata_r.layers["rec_u"]
    log_count_u  = np.log10(obs_u.sum(axis=0))
    train_corr_u = colwise_pearsonr(obs_u[train_idx], rec_u[train_idx])
    test_corr_u  = colwise_pearsonr(obs_u[test_idx],  rec_u[test_idx])

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    ax.scatter(log_count_u, train_corr_u, color="blue", s=1, label="train")
    ax.scatter(log_count_u, test_corr_u,  color="red",  s=1, label="test")
    ax.set_xlabel("log10 total count")
    ax.set_ylabel("pearson correlation")
    ax.legend()
    ax.set_title("unspliced")
    fig.savefig(save_dir + "/train_test_corr_u_for_fig.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    # ── ATAC ──────────────────────────────────────────────────────────────
    obs_a = _to_dense(dm.adata_a.X)
    rec_a = dm.adata_a.layers["rec_a"]
    log_count_a  = np.log10(obs_a.sum(axis=0))
    train_corr_a = colwise_pearsonr(obs_a[train_idx], rec_a[train_idx])
    test_corr_a  = colwise_pearsonr(obs_a[test_idx],  rec_a[test_idx])

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    ax.scatter(log_count_a, train_corr_a, color="blue", s=0.03, label="train")
    ax.scatter(log_count_a, test_corr_a,  color="red",  s=0.03, label="test")
    ax.set_xlabel("log10 total count")
    ax.set_ylabel("pearson correlation")
    ax.legend()
    ax.set_title("atac")
    fig.savefig(save_dir + "/train_test_corr_a_for_fig.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    print(f"  Reconstruction correlation plots saved to: {save_dir}")


# ============================================================
# Velocity Consistency Score (VCS)
# ============================================================

def compute_vcs(x_mat, v_mat, pseudotime, n_bins=10):
    """Compute Velocity Consistency Score for each feature.

    Parameters
    ----------
    x_mat : array-like, shape (n_cells, n_features)
        Observed values (e.g. Ms, Mu, Ma).
    v_mat : array-like, shape (n_cells, n_features)
        Inferred velocity (e.g. dsdt, dudt, dadt).
    pseudotime : np.ndarray, shape (n_cells,)
        Pseudotime values; used for equal-frequency binning.
    n_bins : int
        Number of pseudotime bins T.

    Returns
    -------
    vcs : np.ndarray, shape (n_features,)
        VCS_j for each feature j.
    """
    x_mat = _to_dense(x_mat).astype(np.float32)
    v_mat = _to_dense(v_mat).astype(np.float32)

    # Equal-frequency binning: split sorted cell indices into n_bins groups
    order = np.argsort(pseudotime)
    bins = np.array_split(order, n_bins)   # list of length n_bins

    # Mean expression per bin: shape (n_bins, n_features)
    bin_means = np.stack([x_mat[idx].mean(axis=0) for idx in bins], axis=0)

    # ΔX̄_{t,j} = X̄_{t+1,j} - X̄_{t,j}: shape (n_bins-1, n_features)
    delta = bin_means[1:] - bin_means[:-1]

    # VCS_j = Σ_{t=1}^{T-1} ΔX̄_{t,j} * mean_c∈C_t[ sgn(v_{c,j}) ]
    vcs = np.zeros(x_mat.shape[1], dtype=np.float32)
    for t, idx in enumerate(bins[:-1]):
        mean_sgn = np.sign(v_mat[idx]).mean(axis=0)   # shape (n_features,)
        vcs += delta[t] * mean_sgn

    return vcs


def compute_vcs_all(adata_r, adata_a, pseudotime, n_bins=10):
    """Compute VCS for spliced, unspliced, and ATAC modalities.

    Returns a dict with keys 'spliced', 'unspliced', 'atac', each mapping
    to a 1-D numpy array of VCS values.
    """
    return {
        "spliced":   compute_vcs(adata_r.layers["Ms"],  adata_r.layers["dsdt"],
                                  pseudotime, n_bins),
        "unspliced": compute_vcs(adata_r.layers["Mu"],  adata_r.layers["dudt"],
                                  pseudotime, n_bins),
        "atac":      compute_vcs(adata_a.layers["Ma"],  adata_a.layers["dadt"],
                                  pseudotime, n_bins),
    }


def plot_vcs_boxplots(vcs_results, save_path):
    """Box-plot comparison of VCS distributions across ablation conditions.

    Parameters
    ----------
    vcs_results : dict  {condition_name -> {modality -> np.ndarray}}
    save_path : str
        Output PNG file path.
    """
    modalities   = ["spliced", "unspliced", "atac"]
    mod_labels   = ["Spliced (dsdt)", "Unspliced (dudt)", "ATAC (dadt)"]
    cond_names   = list(vcs_results.keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, mod, mod_label in zip(axes, modalities, mod_labels):
        data   = [vcs_results[c][mod] for c in cond_names]
        bp = ax.boxplot(data, labels=cond_names, patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.5),
                        showfliers=False)
        colors = plt.cm.tab10(np.linspace(0, 1, len(cond_names)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(mod_label)
        ax.set_ylabel("VCS")
        ax.set_xticklabels(cond_names, rotation=30, ha="right")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    fig.suptitle("Velocity Consistency Score — ablation comparison", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"VCS boxplot saved to: {save_path}")


# ============================================================
# Main
# ============================================================

os.chdir("/home/nomura/Proj/mmvelo")
os.makedirs(ABLATION_OUTPUT_DIR, exist_ok=True)
plt.rcParams["font.family"] = "sans-serif"
np.random.seed(DEFAULT_PARAMS["seed"])

# Reference UMAP coordinates, cluster labels, and pseudotime from the baseline run
umap_coord = pd.read_csv(
    ANNDATA_DIR + "/umap_coordinate.tsv", sep="\t", header=None
).to_numpy()
clusters = pd.read_json(
    ANNDATA_DIR + "/cell_clusters.json", typ="series"
).astype("category")
pseudotime = pd.read_csv(
    ANNDATA_DIR + "/pseudotime.tsv", sep="\t", header=None
)[0].to_numpy()

vcs_results = {}   # {condition_name -> {modality -> np.ndarray}}

for condition in ABLATION_CONDITIONS:
    print(f"\n{'='*60}")
    print(f"Condition: {condition['name']}  "
          f"(zdim={condition['zdim']}, smooth={condition['smooth']})")
    print(f"{'='*60}")

    # The baseline (smooth_z10) re-uses the existing run directory so that
    # pre-existing checkpoints are picked up automatically.
    if condition["name"] == "smooth_z10":
        run_dir = BASE_RUN_PATH
    else:
        run_dir = ABLATION_OUTPUT_DIR + "/" + condition["name"]

    save_dir = ABLATION_OUTPUT_DIR + "/" + condition["name"] + "_streamline"

    adata_r, adata_a = run_training(condition, DEFAULT_PARAMS, run_dir)
    make_streamline_plot(adata_r, adata_a, save_dir, clusters, umap_coord)

    print(f"  Computing VCS …")
    vcs_results[condition["name"]] = compute_vcs_all(
        adata_r, adata_a, pseudotime, n_bins=10
    )

# Save per-condition VCS arrays as TSVs and plot boxplots
vcs_out_dir = ABLATION_OUTPUT_DIR + "/vcs"
os.makedirs(vcs_out_dir, exist_ok=True)

for cname, scores in vcs_results.items():
    for mod, arr in scores.items():
        pd.Series(arr).to_csv(
            f"{vcs_out_dir}/vcs_{cname}_{mod}.tsv",
            sep="\t", header=False, index=False,
        )

plot_vcs_boxplots(vcs_results, ABLATION_OUTPUT_DIR + "/vcs_boxplot.png")

print("\nAblation analysis complete.")
print(f"Results saved under: {ABLATION_OUTPUT_DIR}")
