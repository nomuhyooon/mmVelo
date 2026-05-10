"""
06_velocity_uncertainty.py

訓練済み DREG_DYN モデルから RNA velocity / chromatin velocity の
不確実性を3層に分解して定量し、結果を loom ファイルおよび UMAP プロットとして保存する。

3層の分解:
  1. State-estimation uncertainty  : q(z|a,s,u) から z をサンプリング (d を posterior mean 固定)
  2. Dynamics fluctuation uncertainty : q(d|z_hat) から d をサンプリング (z を posterior mean 固定)
  3. Transition ambiguity          : z, d をともにサンプリングし kNN 遷移重みの方向分散を評価

前提: 00_get_inferred_data.py が実行済みで
  downstream_analysis/result/anndata/ に以下が存在すること:
    - adata_rna.loom  (layers: spliced_count, unspliced_count, Ms, Mu;
                       var: estimated_genes; obsm: X_umap, latent)
    - adata_atac.loom (layers: atac_count, Ma)
    - norm_mat_s.txt, norm_mat_u.txt, norm_mat_a.txt

出力:
  downstream_analysis/result/uncertainty/
    - adata_rna_uncertainty.loom
        layers : vel_mean               (N, rna_dim)  RNA velocity 平均
        obsm   : z_hat, d_hat           (N, z_dim)
        obs    : u_state, u_dyn, u_trans, u_trans_z
    - adata_atac_uncertainty.loom
        layers : vel_mean               (N, atac_dim) ATAC velocity 平均
        obs    : u_state, u_dyn, u_trans
    - velocity_uncertainty_umap.png
"""

import os
import sys
import json
import numpy as np
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.append("/home/nomura/Proj/mmvelo/src")
from mmvelo_multi.models import DREG_DYN
from mmvelo_multi.dataset import SHARESeqHFDataModule_Pre
from mmvelo_multi.velocity_uncertainty import (
    compute_velocity_uncertainty,
    save_velocity_uncertainty,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
os.chdir("/home/nomura/Proj/mmvelo")

runPath     = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis"
anndata_dir = runPath + "/downstream_analysis/result/anndata"
save_dir    = runPath + "/downstream_analysis/result/uncertainty"
os.makedirs(save_dir, exist_ok=True)

# ── Params ─────────────────────────────────────────────────────────────────────
with open(runPath + "/params.json") as f:
    params = json.load(f)

torch.manual_seed(params["seed"])
np.random.seed(params["seed"])

N_SAMPLES   = 200   # 各層のサンプリング回数
N_NEIGHBORS = 100    # Transition ambiguity 用 kNN 近傍数
TAU         = None  # bandwidth; None → 近傍距離の中央値を自動設定
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# ── Load pre-processed AnnData ─────────────────────────────────────────────────
print("Loading pre-processed AnnData...")
adata_r = sc.read_loom(anndata_dir + "/adata_rna.loom",
                        obs_names="obs_names", var_names="var_names")
adata_a = sc.read_loom(anndata_dir + "/adata_atac.loom",
                        obs_names="obs_names", var_names="var_names")

# ── Build DataLoader ────────────────────────────────────────────────────────────
def _to_dense(x):
    if hasattr(x, "toarray"):
        return x.toarray().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


class _DenseDataSet(torch.utils.data.Dataset):
    def __init__(self, s, u, a, ms, mu, ma):
        self.data = [s, u, a, ms, mu, ma]

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        return tuple(torch.tensor(x[idx], dtype=torch.float32) for x in self.data)


dataset = _DenseDataSet(
    s  = _to_dense(adata_r.layers["spliced_count"]),
    u  = _to_dense(adata_r.layers["unspliced_count"]),
    a  = _to_dense(adata_a.layers["atac_count"]),
    ms = _to_dense(adata_r.layers["Ms"]),
    mu = _to_dense(adata_r.layers["Mu"]),
    ma = _to_dense(adata_a.layers["Ma"]),
)
dl = torch.utils.data.DataLoader(
    dataset, batch_size=params["batch_size"],
    shuffle=False, num_workers=2, pin_memory=True,
)

# ── DataModule (rna_dim / atac_dim / l_prior の取得に使用) ────────────────────
print("Loading DataModule (for model dimensions)...")
dm_raw = SHARESeqHFDataModule_Pre(
    batch_size        = params["batch_size"],
    n_top_genes       = params["n_genes"],
    n_top_peaks       = params["n_peaks"],
    min_counts_genes  = params["min_counts_genes"],
    min_counts_peaks  = params["min_counts_peaks"],
)

# ── filter_idx: 00_get_inferred_data.py が var["estimated_genes"] として保存済み
estimated  = adata_r.var["estimated_genes"]
filter_idx = torch.tensor(estimated.values.astype(np.float32))

# ── Load DREG_DYN ──────────────────────────────────────────────────────────────
print("Loading DREG_DYN from checkpoint.ckpt ...")
model = DREG_DYN.load_from_checkpoint(
    runPath + "/checkpoint.ckpt",
    rna_dim     = dm_raw.rna_dim,
    atac_dim    = dm_raw.atac_dim,
    r_h1_dim    = params["r_h1dim"],
    r_h2_dim    = params["r_h2dim"],
    a_h1_dim    = params["a_h1dim"],
    a_h2_dim    = params["a_h2dim"],
    z_dim       = params["zdim"],
    d_h_dim     = params["d_h_dim"],
    l_prior_r   = dm_raw.l_prior_r,
    l_prior_a   = dm_raw.l_prior_a,
    lr          = params["lr_dyn"],
    z_learnable = params["z_learnable"],
    d_coeff     = params["d_coeff"],
    filter_idx  = filter_idx.to(device),
    strict      = False,
)

# norm_mat はバッファだが strict=False でスキップされるため、保存済み .txt から設定
norm_mat_s = torch.tensor(np.loadtxt(anndata_dir + "/norm_mat_s.txt"), dtype=torch.float32)
norm_mat_u = torch.tensor(np.loadtxt(anndata_dir + "/norm_mat_u.txt"), dtype=torch.float32)
norm_mat_a = torch.tensor(np.loadtxt(anndata_dir + "/norm_mat_a.txt"), dtype=torch.float32)

model.register_buffer("norm_mat_s",      norm_mat_s)
model.register_buffer("norm_mat_u",      norm_mat_u)
model.register_buffer("norm_mat_a",      norm_mat_a)
model.register_buffer("retain_gene_idx", torch.ones(dm_raw.rna_dim))
model.register_buffer("filter_idx",      filter_idx)

model = model.to(device)
model.eval()

# ── Compute 3-layer velocity uncertainty ────────────────────────────────────────
print(f"Computing 3-layer velocity uncertainty "
      f"(n_samples={N_SAMPLES}, n_neighbors={N_NEIGHBORS}, tau={TAU})...")
results = compute_velocity_uncertainty(
    model, dl,
    n_samples   = N_SAMPLES,
    n_neighbors = N_NEIGHBORS,
    tau         = TAU,
    device      = device,
)

# ── 結果サマリーの出力 ─────────────────────────────────────────────────────────
print("\n── Uncertainty summary (mean ± std across cells) ──")
for key in ["u_state_rna", "u_state_atac", "u_dyn_rna", "u_dyn_atac",
            "u_trans_z", "u_trans_rna", "u_trans_atac"]:
    v = results[key]
    print(f"  {key:20s}: {v.mean():.4f} ± {v.std():.4f}  "
          f"[min={v.min():.4f}, max={v.max():.4f}]")

# ── AnnData に格納して .loom 保存 ───────────────────────────────────────────────
class _MockDM:
    def __init__(self, adata_r, adata_a):
        self.adata_r = adata_r
        self.adata_a = adata_a

mock_dm = _MockDM(adata_r, adata_a)
save_velocity_uncertainty(
    mock_dm, results,
    save_dir      = save_dir,
    filename_rna  = "adata_rna_uncertainty.loom",
    filename_atac = "adata_atac_uncertainty.loom",
)

# ── UMAP 可視化 ─────────────────────────────────────────────────────────────────
print("Plotting UMAP...")

# X_umap が保存されていない場合のフォールバック
if "X_umap" not in adata_r.obsm:
    sc.pp.neighbors(adata_r, use_rep="latent", n_neighbors=15)
    sc.tl.umap(adata_r)

# ATAC の obs 値を RNA AnnData にコピー (UMAP は RNA 空間で計算)
for col in ["u_state", "u_dyn", "u_trans"]:
    adata_r.obs[f"atac_{col}"] = adata_a.obs[col].values

# 描画設定: 3列 × 2行 (RNA / ATAC × 3層)
plot_keys = [
    ("u_state",      "Reds",   "State-estimation uncertainty (RNA)"),
    ("u_dyn",        "Reds",   "Dynamics fluctuation uncertainty (RNA)"),
    ("u_trans",      "Reds",   "Transition ambiguity (RNA)"),
    ("atac_u_state", "Blues",  "State-estimation uncertainty (ATAC)"),
    ("atac_u_dyn",   "Blues",  "Dynamics fluctuation uncertainty (ATAC)"),
    ("atac_u_trans", "Blues",  "Transition ambiguity (ATAC)"),
]

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
for ax, (key, cmap, title) in zip(axes.flat, plot_keys):
    sc.pl.umap(adata_r, color=key,
               cmap=cmap, title=title,
               ax=ax, show=False, vmax="p95")

plt.tight_layout()
umap_path = save_dir + "/velocity_uncertainty_umap.png"
plt.savefig(umap_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {umap_path}")

# ── Latent-space transition ambiguity の UMAP (単独) ───────────────────────────
fig2, ax2 = plt.subplots(1, 1, figsize=(6, 5))
sc.pl.umap(adata_r, color="u_trans_z",
           cmap="Purples", title="Transition ambiguity (latent space)",
           ax=ax2, show=False, vmax="p95")
plt.tight_layout()
umap_z_path = save_dir + "/transition_ambiguity_latent_umap.png"
plt.savefig(umap_z_path, dpi=300, bbox_inches="tight")
plt.close(fig2)
print(f"Saved: {umap_z_path}")

print(f"\nAll results saved to: {save_dir}")
print("Done.")
