"""
Supplementary figure: average chromatin velocity of each peak cluster projected onto UMAP.

For each peak cluster (leiden), computes the mean dadt across peaks in that cluster
per cell, then visualizes the result on the cell-state UMAP.
This helps relate peak-level clusters to cell-state structure.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scanpy as sc

# ── output directory ──────────────────────────────────────────────────────────
out_dir = (
    "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/"
    "2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/"
    "result/clusterwise_mean_velocity"
)
os.makedirs(out_dir, exist_ok=True)

# ── load data ─────────────────────────────────────────────────────────────────
anndata_dir = (
    "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/"
    "2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/"
    "result/anndata"
)

adata_r = sc.read_loom(
    anndata_dir + "/adata_rna.loom", obs_names="obs_names", var_names="var_names"
)
adata_a = sc.read_loom(
    anndata_dir + "/adata_atac.loom", obs_names="obs_names", var_names="var_names"
)

adata_r.obsm["X_umap"] = pd.read_csv(
    anndata_dir + "/umap_coordinate.tsv", sep="\t", header=None
).to_numpy()
adata_r.obs["clusters"] = (
    pd.read_json(anndata_dir + "/cell_clusters.json", typ="series")
    .astype("category")
)

# ── load peak cluster assignments ─────────────────────────────────────────────
dadt_clust_dir = (
    "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/"
    "2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/"
    "result/dadt_clustering"
)
adata_peak = sc.read_loom(dadt_clust_dir + "/adata_dadt_cluster.loom")
adata_peak.obs_names = adata_peak.obs["obs_names"]

# Apply the same cluster-label remapping used in 09_dadt_clustering_heatmap_for_fig.py
# so that cluster numbers are consistent across the paper
leiden_remap = {"0": 0, "1": 4, "2": 2, "3": 6, "4": 7, "5": 5, "6": 1, "7": 8, "8": 3}
adata_peak.obs["cluster_id"] = (
    adata_peak.obs["leiden"].map(leiden_remap).astype(int)
)

# ── build peak-index lookup aligned with adata_a.var_names ───────────────────
# adata_a.var_names == adata_peak.obs_names (same 25 071 peaks, same order)
peak_to_cluster = pd.Series(
    adata_peak.obs["cluster_id"].values, index=adata_peak.obs_names
)
# reindex to guarantee same order as adata_a columns
peak_to_cluster = peak_to_cluster.reindex(adata_a.var_names)

cluster_ids = sorted(peak_to_cluster.dropna().unique().astype(int))

# ── chromatin velocity matrix: cells × peaks ─────────────────────────────────
dadt = adata_a.layers["dadt"]          # scipy sparse (3735 cells × 25071 peaks)
umap_coords = adata_r.obsm["X_umap"]  # (3735 cells × 2)
cell_clusters = adata_r.obs["clusters"]

# ── helper: UMAP scatter colored by a continuous value ───────────────────────
def plot_umap_velocity(
    umap, values, title, filepath,
    cmap="coolwarm", vmin=None, vmax=None, s=4, dpi=300,
    show_colorbar=True
):
    """Scatter UMAP colored by `values` (one float per cell)."""
    if vmin is None or vmax is None:
        absmax = np.nanpercentile(np.abs(values), 99)
        vmin, vmax = -absmax, absmax

    fig, ax = plt.subplots(figsize=(4, 4))
    sc_plot = ax.scatter(
        umap[:, 0], umap[:, 1],
        c=values, cmap=cmap, vmin=vmin, vmax=vmax,
        s=s, linewidths=0, rasterized=True
    )
    if show_colorbar:
        cbar = fig.colorbar(sc_plot, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("mean dadt", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("UMAP 1", fontsize=8)
    ax.set_ylabel("UMAP 2", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_umap_velocity_blank(
    umap, values, filepath,
    cmap="coolwarm", vmin=None, vmax=None, s=4, dpi=300
):
    """Axes-off version for figure panels."""
    if vmin is None or vmax is None:
        absmax = np.nanpercentile(np.abs(values), 99)
        vmin, vmax = -absmax, absmax

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(
        umap[:, 0], umap[:, 1],
        c=values, cmap=cmap, vmin=vmin, vmax=vmax,
        s=s, linewidths=0, rasterized=True
    )
    ax.axis("off")
    fig.tight_layout()
    plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close()


# ── per-cluster mean velocity ─────────────────────────────────────────────────
# Collect mean velocities to determine a shared color scale across panels
mean_velocities = {}

for clst in cluster_ids:
    peak_mask = (peak_to_cluster == clst).values
    # mean over peaks in this cluster for each cell
    clst_dadt = dadt[:, peak_mask]          # (n_cells × n_clst_peaks)
    mean_vel = np.asarray(clst_dadt.mean(axis=1)).ravel()  # (n_cells,)
    mean_velocities[clst] = mean_vel

# shared symmetric color scale using 99th percentile over all clusters
all_vals = np.concatenate(list(mean_velocities.values()))
absmax_global = np.nanpercentile(np.abs(all_vals), 99)
vmin_global, vmax_global = -absmax_global, absmax_global

# ── plot individual cluster panels ───────────────────────────────────────────
for clst, mean_vel in mean_velocities.items():
    n_peaks = int((peak_to_cluster == clst).sum())

    # labeled version
    plot_umap_velocity(
        umap_coords, mean_vel,
        title=f"Peak cluster {clst}  (n={n_peaks})",
        filepath=os.path.join(out_dir, f"umap_mean_velocity_cluster_{clst}.png"),
        vmin=vmin_global, vmax=vmax_global
    )
    # axes-off version
    plot_umap_velocity_blank(
        umap_coords, mean_vel,
        filepath=os.path.join(out_dir, f"umap_mean_velocity_cluster_{clst}_blank.png"),
        vmin=vmin_global, vmax=vmax_global
    )
    print(f"Cluster {clst}: {n_peaks} peaks, "
          f"mean vel range [{mean_vel.min():.3f}, {mean_vel.max():.3f}]")

# ── multi-panel figure (all clusters in one image) ───────────────────────────
n_clusters = len(cluster_ids)
ncols = 3
nrows = int(np.ceil(n_clusters / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
axes = axes.ravel()

for i, clst in enumerate(cluster_ids):
    mean_vel = mean_velocities[clst]
    n_peaks = int((peak_to_cluster == clst).sum())
    ax = axes[i]
    sc_plot = ax.scatter(
        umap_coords[:, 0], umap_coords[:, 1],
        c=mean_vel, cmap="coolwarm",
        vmin=vmin_global, vmax=vmax_global,
        s=3, linewidths=0, rasterized=True
    )
    ax.set_title(f"Cluster {clst}  (n={n_peaks})", fontsize=9)
    ax.set_xlabel("UMAP 1", fontsize=7)
    ax.set_ylabel("UMAP 2", fontsize=7)
    ax.tick_params(labelsize=6)
    fig.colorbar(sc_plot, ax=ax, fraction=0.046, pad=0.04).set_label(
        "mean dadt", fontsize=6
    )

# hide unused panels
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

fig.suptitle(
    "Average chromatin velocity per peak cluster projected onto UMAP",
    fontsize=11, y=1.01
)
fig.tight_layout()
plt.savefig(
    os.path.join(out_dir, "umap_mean_velocity_all_clusters.png"),
    dpi=300, bbox_inches="tight"
)
plt.close()
print("Saved multi-panel figure.")

# ── reference panel: cell-type clusters ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
sc.pl.umap(adata_r, color="clusters", ax=ax, show=False, title="Cell clusters (reference)")
fig.tight_layout()
plt.savefig(
    os.path.join(out_dir, "umap_cell_clusters_reference.png"),
    dpi=300, bbox_inches="tight"
)
plt.close()
print("Saved reference cell-cluster UMAP.")
print(f"\nAll outputs written to: {out_dir}")
