"""
07_uncertainty_vs_pseudotime.py

on-manifold fluctuation (u_dyn_par) と off-manifold instability (u_dyn_perp) を
pseudotime に沿って散布図 + LOWESS トレンド曲線で可視化する。

対象クラスター: TAC, HS-TAC, Hair Shaft-Cuticle/Cortex, Medulla
RNA / ATAC それぞれ独立したプロットを生成する。
y 軸上限はプロット対象データの 95 パーセンタイル値とする。

出力 (save_dir/):
  RNA:
    uncertainty_vs_pseudotime_rna_combined.png
    uncertainty_vs_pseudotime_rna_on_manifold.png
    uncertainty_vs_pseudotime_rna_off_manifold.png
  ATAC:
    uncertainty_vs_pseudotime_atac_combined.png
    uncertainty_vs_pseudotime_atac_on_manifold.png
    uncertainty_vs_pseudotime_atac_off_manifold.png
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

# ── Paths ──────────────────────────────────────────────────────────────────────
anndata_dir = (
    "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/"
    "2023-08-03T13:31:54_nb_k50_for_analysis/"
    "downstream_analysis/result/anndata"
)
save_dir = (
    "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/"
    "2023-08-03T13:31:54_nb_k50_for_analysis/"
    "downstream_analysis/result/off_manifold_mod"
)
os.makedirs(save_dir, exist_ok=True)

# ── Load off-manifold AnnData ──────────────────────────────────────────────────
print("Loading off-manifold AnnData (RNA & ATAC)...")
adata_rna  = sc.read_loom(save_dir + "/adata_rna_offmanifold_mod.loom",
                           obs_names="obs_names", var_names="var_names")
adata_atac = sc.read_loom(save_dir + "/adata_atac_offmanifold_mod.loom",
                           obs_names="obs_names", var_names="var_names")

# ── Load pseudotime & refined_clusters ────────────────────────────────────────
obs_names    = pd.read_csv(anndata_dir + "/obs_names.txt",
                           header=None)[0].values
pseudotime   = pd.read_csv(anndata_dir + "/pseudotime.tsv",
                           sep="\t", header=None)[0].values
ref_clusters = pd.read_csv(anndata_dir + "/refined_clusters.tsv",
                           sep="\t", header=None)[0].values

assert len(obs_names) == len(adata_rna), (
    f"obs_names length mismatch: {len(obs_names)} vs {len(adata_rna)}")

for adata in [adata_rna, adata_atac]:
    adata.obs["pseudotime"]   = pseudotime
    adata.obs["ref_clusters"] = ref_clusters

# ── Cluster color map ─────────────────────────────────────────────────────────
adata_rna_orig = sc.read_loom(
    anndata_dir + "/adata_rna.loom",
    obs_names="obs_names", var_names="var_names",
)
cluster_color_map = {}
if "ref_clusters_colors" in adata_rna_orig.uns:
    cats = list(adata_rna_orig.obs["ref_clusters"].cat.categories)
    cols = list(adata_rna_orig.uns["ref_clusters_colors"])
    cluster_color_map = dict(zip(cats, cols))
    print("Cluster colors from adata_rna.loom:", cluster_color_map)
else:
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    all_clusters   = sorted(adata_rna.obs["ref_clusters"].unique())
    cluster_color_map = {c: default_colors[i % len(default_colors)]
                         for i, c in enumerate(all_clusters)}
    print("Fallback colors:", cluster_color_map)

# ── Settings ───────────────────────────────────────────────────────────────────
TARGET_CLUSTERS = ["TAC", "HS-TAC", "Hair Shaft-Cuticle/Cortex", "Medulla"]
ALPHA_SCATTER   = 0.35
POINT_SIZE      = 8
DPI             = 300
LOWESS_FRAC     = 0.25   # LOWESS の平滑化幅 (0–1)
TREND_LW        = 1.2

# ── Helper functions ───────────────────────────────────────────────────────────

def build_df(adata):
    """対象クラスターの pseudotime / on- / off-manifold を DataFrame に整理."""
    df = adata.obs[["pseudotime", "u_dyn_par", "u_dyn_perp", "ref_clusters"]].copy()
    df.columns = ["pseudotime", "on_manifold", "off_manifold", "ref_clusters"]
    df = df[df["ref_clusters"].isin(TARGET_CLUSTERS)].copy()
    df = df.dropna(subset=["pseudotime", "on_manifold", "off_manifold"])
    return df


def add_lowess(ax, x, y, color, lw=TREND_LW, frac=LOWESS_FRAC, linestyle="-"):
    """LOWESS トレンド曲線を描画する。"""
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    smoothed = lowess(ys, xs, frac=frac, return_sorted=True)
    ax.plot(smoothed[:, 0], smoothed[:, 1],
            color=color, linewidth=lw, linestyle=linestyle, zorder=5)


def _ylim_upper(values_list):
    """複数配列をまとめた 95 パーセンタイルを返す。"""
    all_vals = np.concatenate([v for v in values_list if len(v) > 0])
    return np.nanpercentile(all_vals, 95)


def plot_single(ax, df, y_col, ylabel, title, ylim_top, clst_color_map):
    """1 軸分の散布図 + 全細胞で1本の LOWESS トレンドを描画する。"""
    for clst in TARGET_CLUSTERS:
        sub = df[df["ref_clusters"] == clst]
        if sub.empty:
            continue
        color = clst_color_map.get(clst, "gray")
        ax.scatter(sub["pseudotime"], sub[y_col],
                   c=color, alpha=ALPHA_SCATTER, s=POINT_SIZE,
                   linewidths=0, label=clst)

    # 全細胞で1本のトレンド曲線
    add_lowess(ax, df["pseudotime"].values, df[y_col].values,
               color="black", lw=TREND_LW)

    ax.set_xlabel("Pseudotime", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(bottom=0, top=ylim_top)
    ax.legend(markerscale=2, fontsize=9, framealpha=0.7,
              loc="upper left", bbox_to_anchor=(1.01, 1))


def plot_combined(ax, df, ylim_on, ylim_off, clst_color_map):
    """on-manifold (○, 左軸) と off-manifold (△, 右軸) を双軸で描画する。
    トレンド曲線は全細胞で on/off 各1本。"""
    ax2 = ax.twinx()

    for clst in TARGET_CLUSTERS:
        sub = df[df["ref_clusters"] == clst]
        if sub.empty:
            continue
        color = clst_color_map.get(clst, "gray")
        ax.scatter(sub["pseudotime"], sub["on_manifold"],
                   c=color, alpha=ALPHA_SCATTER, s=POINT_SIZE,
                   linewidths=0, marker="o", label=clst)
        ax2.scatter(sub["pseudotime"], sub["off_manifold"],
                    c=color, alpha=ALPHA_SCATTER, s=POINT_SIZE,
                    linewidths=0, marker="^")

    # 全細胞で on/off 各1本のトレンド曲線
    add_lowess(ax,  df["pseudotime"].values, df["on_manifold"].values,
               color="black", lw=TREND_LW, linestyle="-")
    add_lowess(ax2, df["pseudotime"].values, df["off_manifold"].values,
               color="black", lw=TREND_LW, linestyle="--")

    ax.set_ylim(bottom=0, top=ylim_on)
    ax2.set_ylim(bottom=0, top=ylim_off)

    ax.set_xlabel("Pseudotime", fontsize=12)
    ax.set_ylabel("On-manifold fluctuation",  fontsize=12)
    ax2.set_ylabel("Off-manifold instability", fontsize=12)

    # 凡例: クラスター色 + on/off マーカー + トレンド線
    handles_clst = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=clst_color_map.get(c, "gray"),
                   markersize=7, label=c)
        for c in TARGET_CLUSTERS
        if c in df["ref_clusters"].values
    ]
    handles_type = [
        plt.Line2D([0], [0], marker="o", color="gray",
                   markersize=7, linewidth=0,
                   label="on-manifold fluctuation (○, left)"),
        plt.Line2D([0], [0], marker="^", color="gray",
                   markersize=7, linewidth=0,
                   label="off-manifold instability (△, right)"),
    ]
    handles_trend = [
        plt.Line2D([0], [0], color="black", lw=TREND_LW,
                   linestyle="-",  label="trend: on-manifold"),
        plt.Line2D([0], [0], color="black", lw=TREND_LW,
                   linestyle="--", label="trend: off-manifold"),
    ]
    ax.legend(handles=handles_clst + handles_type + handles_trend,
              fontsize=9, framealpha=0.7,
              loc="upper left", bbox_to_anchor=(1.28, 1))


def save_fig(fig, path, tight=True):
    if tight:
        plt.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ── Main loop: RNA & ATAC ──────────────────────────────────────────────────────
for modality, adata in [("rna", adata_rna), ("atac", adata_atac)]:
    print(f"\n── {modality.upper()} ──")
    df = build_df(adata)
    print(f"  Cells after filtering: {len(df)}")
    print(df["ref_clusters"].value_counts().to_string())

    # y 軸上限 (95 パーセンタイル)
    ylim_on  = _ylim_upper([
        df.loc[df["ref_clusters"] == c, "on_manifold"].values
        for c in TARGET_CLUSTERS
    ])
    ylim_off = _ylim_upper([
        df.loc[df["ref_clusters"] == c, "off_manifold"].values
        for c in TARGET_CLUSTERS
    ])
    # (1) Combined
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_combined(ax, df, ylim_on=ylim_on, ylim_off=ylim_off,
                  clst_color_map=cluster_color_map)
    ax.set_title(f"On/Off-manifold uncertainty vs Pseudotime ({modality.upper()})", fontsize=13)
    # 右 y 軸と legend が重ならないようにプロット領域を左寄せに調整
    fig.subplots_adjust(right=0.58)
    save_fig(fig, f"{save_dir}/uncertainty_vs_pseudotime_{modality}_combined.png", tight=False)

    # (2) On-manifold only
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_single(ax, df, "on_manifold",
                ylabel="On-manifold fluctuation",
                title=f"On-manifold fluctuation vs Pseudotime ({modality.upper()})",
                ylim_top=ylim_on,
                clst_color_map=cluster_color_map)
    save_fig(fig, f"{save_dir}/uncertainty_vs_pseudotime_{modality}_on_manifold.png")

    # (3) Off-manifold only
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_single(ax, df, "off_manifold",
                ylabel="Off-manifold instability",
                title=f"Off-manifold instability vs Pseudotime ({modality.upper()})",
                ylim_top=ylim_off,
                clst_color_map=cluster_color_map)
    save_fig(fig, f"{save_dir}/uncertainty_vs_pseudotime_{modality}_off_manifold.png")

print("\nDone.")
