"""
viz_human_brain.py
------------------
Visualization helpers for Tutorial 2 (human cortical development, missing
modality inference).  Keeping complex boilerplate out of the notebook.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ad
import scanpy as sc
import scvelo as scv


# ── Load / split helpers ──────────────────────────────────────────────────────

def load_results_and_split(run_path, data_dir="data/human_brain"):
    """
    Reload saved loom files, attach cluster labels, filter to included cells,
    and split by modality.

    Returns
    -------
    adata_r, adata_a,
    adata_r_a2r, adata_r_r2r,   # RNA velocity subsets
    adata_a_r2a, adata_a_a2a    # ATAC velocity subsets
    """
    adata_r = sc.read_loom(
        f"{run_path}/adata_rna.loom", obs_names="obs_names", var_names="var_names")
    adata_a = sc.read_loom(
        f"{run_path}/adata_atac.loom", obs_names="obs_names", var_names="var_names")

    cluster_df = pd.read_csv(
        f"{data_dir}/cluster_annotation_refined.txt", sep="\t", index_col=0)
    adata_r.obs["ref_clusters"] = cluster_df.iloc[:, 0].reindex(adata_r.obs_names)
    adata_a.obs["ref_clusters"] = adata_r.obs["ref_clusters"].values
    adata_a.obsm["X_umap"]     = adata_r.obsm["X_umap"]

    cells_included = pd.read_csv(
        f"{data_dir}/cells_included.txt", sep="\t", header=None)[0].tolist()
    adata_r = adata_r[cells_included, :]
    adata_a = adata_a[cells_included, :]

    print(f"Cells after filtering : {adata_r.shape[0]}")
    print(f"Modality breakdown    : {adata_r.obs['modality'].value_counts().to_dict()}")

    adata_r_a2r = adata_r[adata_r.obs["modality"] == "atac", :]
    adata_r_r2r = adata_r[adata_r.obs["modality"].isin(["rna", "multiome"]), :]
    adata_a_r2a = adata_a[adata_a.obs["modality"] == "rna", :]
    adata_a_a2a = adata_a[adata_a.obs["modality"].isin(["atac", "multiome"]), :]

    print(f"\natac2rna cells  : {adata_r_a2r.shape[0]}")
    print(f"rna2rna  cells  : {adata_r_r2r.shape[0]}")
    print(f"rna2atac cells  : {adata_a_r2a.shape[0]}")
    print(f"atac2atac cells : {adata_a_a2a.shape[0]}")

    return adata_r, adata_a, adata_r_a2r, adata_r_r2r, adata_a_r2a, adata_a_a2a


def plot_streamline(adata, vkey, xkey, title, n_neighbors=30, n_jobs=4):
    """
    Build velocity graph and draw a streamline plot.

    Parameters
    ----------
    adata      : AnnData with obsm["latent"], obsm["X_umap"], obs["ref_clusters"]
    vkey       : velocity layer key (e.g. "dsdt" or "dadt")
    xkey       : count layer key   (e.g. "s_raw" or "a_raw")
    title      : plot title
    n_neighbors: for kNN graph construction
    """
    from mmvelo_multi_cond.streamlineplot import velocity_graph

    if hasattr(adata.layers[vkey], "toarray"):
        adata.layers[vkey] = adata.layers[vkey].toarray()

    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="latent")
    velocity_graph(adata, vkey=vkey, xkey=xkey, n_jobs=n_jobs)
    scv.tl.velocity_embedding(adata, basis="umap", vkey=vkey)

    fig, ax = plt.subplots(figsize=(7, 6))
    scv.pl.velocity_embedding_stream(
        adata, vkey=vkey, color="ref_clusters", ax=ax,
        title=title, dpi=150, legend_loc="right margin", show=False,
    )
    plt.tight_layout()
    plt.show()


# ── ExN lineage helpers ───────────────────────────────────────────────────────

def filter_exn_lineage(adata_r, adata_a, data_dir="data/human_brain",
                       exn_cell_types=None):
    """
    Subset to excitatory neuron lineage, attach DPT pseudotime, sort by pseudotime.

    Returns
    -------
    adata_r_exn, adata_a_exn  (sorted by pseudotime)
    """
    if exn_cell_types is None:
        exn_cell_types = ["nIPC/GluN", "GluN"]

    mask = adata_r.obs["ref_clusters"].isin(exn_cell_types)
    adata_r_exn = adata_r[mask, :].copy()
    adata_a_exn = adata_a[mask, :].copy()

    print(f"ExN lineage cells : {adata_r_exn.shape[0]}")
    print(adata_r_exn.obs["ref_clusters"].value_counts().to_string())

    dpt_df = pd.read_csv(
        f"{data_dir}/dpt_pseudotime.tsv", sep="\t", header=None, index_col=0)
    adata_r_exn.obs["dpt_pseudotime"] = dpt_df.iloc[:, 0].reindex(adata_r_exn.obs_names)
    adata_a_exn.obs["dpt_pseudotime"] = adata_r_exn.obs["dpt_pseudotime"].values

    order = np.argsort(adata_r_exn.obs["dpt_pseudotime"].values)
    adata_r_exn = adata_r_exn[order, :]
    adata_a_exn = adata_a_exn[order, :]

    print(f"\nPseudotime range : [{adata_r_exn.obs['dpt_pseudotime'].min():.4f}, "
          f"{adata_r_exn.obs['dpt_pseudotime'].max():.4f}]")
    return adata_r_exn, adata_a_exn


def trim_to_overlap(adata_a, adata_b, pt_key="dpt_pseudotime"):
    """
    Trim two AnnData objects to their overlapping pseudotime range.

    Returns trimmed (adata_a, adata_b).
    """
    min_pt = max(adata_a.obs[pt_key].min(), adata_b.obs[pt_key].min())
    max_pt = min(adata_a.obs[pt_key].max(), adata_b.obs[pt_key].max())
    print(f"Overlapping pseudotime range: [{min_pt:.4f}, {max_pt:.4f}]")

    def _trim(ad_):
        return ad_[
            (ad_.obs[pt_key] >= min_pt) & (ad_.obs[pt_key] <= max_pt), :]

    return _trim(adata_a), _trim(adata_b)


def cluster_atac_peaks(adata_a_exn, resolution=0.6):
    """
    Cluster ATAC peaks by cosine similarity of chromatin velocity (dadt).

    Returns
    -------
    clusters_sorted : pd.Series  (peak -> cluster, sorted by cluster id)
    row_colors      : list of RGBA tuples for clustermap row annotation
    make_col_annots : callable(pdt_series) -> (col_annots, col_colors)
                      builds MultiIndex col annotations for pseudotime coloring
    """
    dadt = adata_a_exn.layers["dadt"]
    if hasattr(dadt, "toarray"):
        dadt = dadt.toarray()

    dadt_norm = dadt / (np.std(dadt, axis=0) + 1e-8)

    adata_bin = ad.AnnData(X=dadt_norm.T)
    adata_bin.obs_names = pd.Index(adata_a_exn.var_names.to_numpy())

    sc.pp.neighbors(adata_bin, n_neighbors=30, metric="cosine", n_pcs=None, use_rep="X")
    sc.tl.leiden(adata_bin, resolution=resolution)
    sc.tl.umap(adata_bin)

    n_clusters = len(adata_bin.obs["leiden"].cat.categories)
    print(f"Number of ATAC peak clusters: {n_clusters}")

    fig, ax = plt.subplots(figsize=(7, 6))
    sc.pl.umap(
        adata_bin, color="leiden", ax=ax, show=False, legend_loc="right margin",
        title=f"ATAC peak clustering (chromatin velocity)\n"
              f"{n_clusters} clusters, resolution={resolution}",
    )
    plt.tight_layout()
    plt.show()

    clusters = adata_bin.obs["leiden"]
    sort_idx = np.argsort(clusters.values)
    clusters_sorted = clusters.iloc[sort_idx]

    cmap = plt.get_cmap("tab20")
    cluster_colors = [cmap(i / max(n_clusters - 1, 1)) for i in range(n_clusters)]
    row_colors = [cluster_colors[int(i)] for i in clusters_sorted]

    def make_col_annots(pdt_series):
        pdt_values = pdt_series.values
        col_annots = pd.MultiIndex.from_tuples(
            list(zip(pdt_values)), names=["pseudotime"])
        pdt_labels = col_annots.get_level_values("pseudotime")
        unique_labels = np.sort(np.unique(pdt_labels))
        pdt_lut = dict(zip(unique_labels,
                           sns.color_palette("viridis", len(unique_labels))))
        col_colors = pd.Series(pdt_labels, index=col_annots).map(pdt_lut)
        return col_annots, pd.concat([col_colors], axis=1)

    print("Peak clustering complete.")
    return clusters_sorted, row_colors, make_col_annots


# ── Heatmap helpers ───────────────────────────────────────────────────────────

def plot_dadt_heatmap(adata_exn, clusters_sorted, row_colors, make_col_annots, title):
    """
    Chromatin velocity (dadt) heatmap.

    Peaks sorted by Leiden cluster, cells sorted by pseudotime.

    Parameters
    ----------
    adata_exn       : ExN-filtered ATAC AnnData, cells sorted by pseudotime
    clusters_sorted : pd.Series from cluster_atac_peaks()
    row_colors      : list of RGBA tuples from cluster_atac_peaks()
    make_col_annots : callable from cluster_atac_peaks()
    title           : plot title
    """
    col_annots, col_colors = make_col_annots(adata_exn.obs["dpt_pseudotime"])

    dadt = adata_exn[:, clusters_sorted.index].layers["dadt"]
    if hasattr(dadt, "toarray"):
        dadt = dadt.toarray()
    dadt = dadt.T  # (n_peaks, n_cells)
    dadt = dadt / (np.std(dadt, axis=1, keepdims=True) + 1e-8)

    g = sns.clustermap(
        pd.DataFrame(dadt, columns=col_annots),
        cmap="coolwarm", xticklabels=False, yticklabels=False,
        row_cluster=False, col_cluster=False,
        row_colors=row_colors, col_colors=col_colors,
        vmin=-3, vmax=3, figsize=(12, 8),
    )
    g.fig.suptitle(title, y=1.01, fontsize=12)
    plt.show()


def plot_araw_heatmap(adata_a2a_exn, clusters_sorted, row_colors, make_col_annots, title):
    """
    Smoothed ATAC accessibility (a_raw) heatmap, z-scored per peak.

    Parameters
    ----------
    adata_a2a_exn   : ExN ATAC-only + Multiome AnnData, sorted by pseudotime
    clusters_sorted : pd.Series from cluster_atac_peaks()
    row_colors      : list of RGBA tuples from cluster_atac_peaks()
    make_col_annots : callable from cluster_atac_peaks()
    title           : plot title
    """
    import scipy.stats

    col_annots, col_colors = make_col_annots(adata_a2a_exn.obs["dpt_pseudotime"])

    a_raw = adata_a2a_exn[:, clusters_sorted.index].layers["a_raw"]
    if hasattr(a_raw, "toarray"):
        a_raw = a_raw.toarray()
    a_raw_z = np.clip(scipy.stats.zscore(a_raw, axis=0).T, -3, 3)  # (n_peaks, n_cells)

    g = sns.clustermap(
        pd.DataFrame(a_raw_z, columns=col_annots),
        cmap="viridis", xticklabels=False, yticklabels=False,
        row_cluster=False, col_cluster=False,
        row_colors=row_colors, col_colors=col_colors,
        vmin=-3, vmax=3, figsize=(12, 8),
    )
    g.fig.suptitle(title, y=1.01, fontsize=12)
    plt.show()
