import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
import scvelo as scv
from scipy.io import mmwrite, mmread
import seaborn as sns
import scipy

np.random.seed(42)

dir_path = "/home/nomura/Proj/mmvelo/experiments/Greenleaf_multiome_Cond_merged_all_missing/2024-01-19T11:22:03_barch_128/downstream_analysis/result/anndata"
adata_r = sc.read_loom(dir_path + "/adata_rna.loom", obs_names="obs_names", var_names="var_names")
adata_a = sc.read_loom(dir_path + "/adata_atac.loom", obs_names="obs_names", var_names="var_names")

# make dir
dir_path = "/home/nomura/Proj/mmvelo/experiments/Greenleaf_multiome_Cond_merged_all_missing/2024-01-19T11:22:03_barch_128/downstream_analysis/result/heatmap"
if not os.path.exists(dir_path):
    os.mkdir(dir_path)
    
adata_r.obs["ref_clusters"] = pd.read_csv("/home/nomura/Proj/mmvelo/experiments/Greenleaf_multiome_Cond_merged_all_missing/2024-01-19T11:22:03_barch_128/downstream_analysis/result/annotation/cluster_annotation_refined.txt",
                                            sep='\t', index_col=0)
adata_a.obs["ref_clusters"] = adata_r.obs["ref_clusters"]
adata_a.obsm["X_umap"] = adata_r.obsm["X_umap"]
cells_included = pd.read_csv("/home/nomura/Proj/mmvelo/experiments/Greenleaf_multiome_Cond_merged_all_missing/2024-01-19T11:22:03_barch_128/downstream_analysis/result/anndata/cells_included.txt",
            sep="\t", header=None)[0].to_list()
adata_r = adata_r[cells_included, :]
adata_a = adata_a[cells_included, :]

def plot_umap(adata, dir_name, n_neighbors=30, min_dist=0.2, cluster_name="clusters", 
              fig_name="umap.png", legend_loc="right margin", color_map=None):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    sc.pl.umap(adata, return_fig=True, color=cluster_name, legend_loc=legend_loc, color_map=color_map)
    plt.savefig(dir_name+"/"+fig_name, bbox_inches='tight')
    plt.close(fig)    
save_dir =dir_path
plot_umap(adata_r, save_dir, cluster_name="ref_clusters", fig_name="umap_clusters_refined.png")

# confine to ExN lineage
exn_list = []
for cell_cluster in adata_r.obs["ref_clusters"]:
    exn_list.append((cell_cluster in ["nIPC/GluN", 'GluN']))
adata_r = adata_r[exn_list, :] # 17490
adata_a = adata_a[exn_list, :] # 17490

pd.read_csv("/home/nomura/Proj/mmvelo/experiments/Greenleaf_multiome_Cond_merged_all_missing/2024-01-19T11:22:03_barch_128/downstream_analysis/result/annotation" + "/dpt_pseudotime.tsv",
            sep="\t", header=None, index_col=0)
adata_r.obs["dpt_pseudotime"] = pd.read_csv("/home/nomura/Proj/mmvelo/experiments/Greenleaf_multiome_Cond_merged_all_missing/2024-01-19T11:22:03_barch_128/downstream_analysis/result/annotation" + "/dpt_pseudotime.tsv",
                                            sep="\t", header=None, index_col=0)
adata_a.obs["dpt_pseudotime"] = adata_r.obs["dpt_pseudotime"]

# order cells by dpt pseudotime
cell_order_dpt = np.argsort(adata_r.obs["dpt_pseudotime"])
adata_r, adata_a = adata_r[cell_order_dpt, :], adata_a[cell_order_dpt, :]

# RNA
#adata_r_a2r = adata_r[adata_r.obs["modality"] == "atac", :] # 6223 cells
#adata_r_r2r = adata_r[(adata_r.obs["modality"] == "rna") | (adata_r.obs["modality"] == "multiome"), :] # 11267 cells
# ATAC
#adata_a_r2a = adata_a[adata_a.obs["modality"] == "rna", :] # 5893 cells
#adata_a_a2a = adata_a[(adata_a.obs["modality"] == "atac") | (adata_a.obs["modality"] == "multiome"), :] # 11597 cells

# heatmap
## ATAC
save_dir = dir_path + "/atac_heatmap"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

a_raw = adata_a.layers["a_raw"].toarray()
dadt = adata_a.layers["dadt"].toarray()
var_name = adata_a.var_names.to_numpy()
dadt = dadt / np.std(dadt, axis=0)
a_raw = scipy.stats.zscore(a_raw)
resolution = 0.6
adata_bin = ad.AnnData(X=dadt.T)
adata_bin.layers["rec_x"] = a_raw.T
adata_bin.obs_names = var_name
sc.pp.neighbors(adata_bin, n_neighbors=30, metric="cosine", n_pcs=None, use_rep="X")
sc.tl.leiden(adata_bin, resolution=resolution)
sc.tl.umap(adata_bin)
fig, ax = plt.subplots(figsize=(5, 5))
sc.pl.umap(adata_bin, color="leiden")
fig.tight_layout()
plt.savefig(save_dir + "/umap_leiden_resolution_{}.png".format(resolution), bbox_inches='tight', dpi=300)
plt.close()

def generate_cluster_colors(num_clusters):
    cmap = plt.get_cmap('tab20')
    cluster_colors = [cmap(i) for i in np.linspace(0, 1, num_clusters)]
    return cluster_colors

clusters = adata_bin.obs["leiden"]
num_clusters = len(adata_bin.obs["leiden"].cat.categories)
cluster_colors = generate_cluster_colors(num_clusters)
sorted_dadt = dadt[:, np.argsort(clusters)]
sorted_var_name = var_name[np.argsort(clusters)]
sorted_dadt = sorted_dadt.T
sorted_x_raw = a_raw[:, np.argsort(clusters)]
sorted_x_raw = sorted_x_raw.T
clusters = clusters[np.argsort(clusters)]
row_colors = [cluster_colors[int(i)] for i in clusters]

# set columns colors
col_annots = [list(adata_r.obs["dpt_pseudotime"])]
col_annots = list(zip(*col_annots))
col_annots = pd.MultiIndex.from_tuples(col_annots, names=["pseudotime"])
pdt_labels = col_annots.get_level_values("pseudotime")
pdt_pal = sns.color_palette("viridis", pdt_labels.unique().size)
pdt_lut = dict(zip(pdt_labels.unique().sort_values(), pdt_pal))
pdt_colors = pd.Series(pdt_labels, index=col_annots).map(pdt_lut)
clusters_pdt_colors = pd.concat([pdt_colors], axis=1)

sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(5, 5))
sns.clustermap(pd.DataFrame(sorted_x_raw, columns=col_annots),
               cmap='viridis', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, 
               row_colors=row_colors, col_colors=clusters_pdt_colors,
               vmin=-3, vmax=3)
fig.tight_layout()
plt.savefig(save_dir + "/heatmap_clustering_binned_dx_leiden_{}_x.png".format(resolution), bbox_inches='tight')
plt.close()

sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(5, 5))
sns.clustermap(pd.DataFrame(sorted_dadt, columns=col_annots),
               cmap='coolwarm', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, 
               row_colors=row_colors, col_colors=clusters_pdt_colors,
               vmin=-3, vmax=3)
plt.gca().axis("off")
fig.tight_layout()
plt.savefig(save_dir + "/heatmap_clustering_binned_dx_leiden_{}.png".format(resolution), bbox_inches='tight')
plt.close()


# rna 2 atac
## first, determine the min/max pseudotime for plot
adata_r2a = adata_a[adata_a.obs["modality"] == "rna"]
adata_a2a = adata_a[(adata_a.obs["modality"] == "atac") | (adata_a.obs["modality"] == "multiome"), :]
max_pseudotime = min(adata_r2a.obs["dpt_pseudotime"].max(),
                     adata_a2a.obs["dpt_pseudotime"].max())
min_pseudotime = max(adata_r2a.obs["dpt_pseudotime"].min(),
                     adata_a2a.obs["dpt_pseudotime"].min())
adata_r2a = adata_r2a[(adata_r2a.obs["dpt_pseudotime"] >= min_pseudotime) & (adata_r2a.obs["dpt_pseudotime"] <= max_pseudotime), :]
adata_a2a = adata_a2a[(adata_a2a.obs["dpt_pseudotime"] >= min_pseudotime) & (adata_a2a.obs["dpt_pseudotime"] <= max_pseudotime), :]

col_annots = [adata_r2a.obs["dpt_pseudotime"].to_list()]
col_annots = list(zip(*col_annots))
col_annots = pd.MultiIndex.from_tuples(col_annots, names=["pseudotime"])
pdt_labels = col_annots.get_level_values("pseudotime")
pdt_pal = sns.color_palette("viridis", pdt_labels.unique().size)
pdt_lut = dict(zip(pdt_labels.unique().sort_values(), pdt_pal))
pdt_colors = pd.Series(pdt_labels, index=col_annots).map(pdt_lut)
clusters_pdt_colors = pd.concat([pdt_colors], axis=1)
sorted_dadt = adata_r2a[:, clusters.index].layers["dadt"].toarray().T
sorted_dadt = sorted_dadt / np.std(sorted_dadt, axis=1).reshape(-1, 1)

sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(5, 5))
sns.clustermap(pd.DataFrame(sorted_dadt, columns=col_annots),
               cmap='coolwarm', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, 
               row_colors=row_colors,
               col_colors=clusters_pdt_colors,
               vmin=-3, vmax=3)
plt.gca().axis("off")
fig.tight_layout()
plt.savefig(save_dir + "/heatmap_rna2atac_binned_dx_leiden_{}.png".format(resolution), bbox_inches='tight')
plt.close()
del adata_r2a

## atac 2 atac
col_annots = [adata_a2a.obs["dpt_pseudotime"].to_list()]
col_annots = list(zip(*col_annots))
col_annots = pd.MultiIndex.from_tuples(col_annots, names=["pseudotime"])
pdt_labels = col_annots.get_level_values("pseudotime")
pdt_pal = sns.color_palette("viridis", pdt_labels.unique().size)
pdt_lut = dict(zip(pdt_labels.unique().sort_values(), pdt_pal))
pdt_colors = pd.Series(pdt_labels, index=col_annots).map(pdt_lut)
clusters_pdt_colors = pd.concat([pdt_colors], axis=1)
sorted_dadt = adata_a2a[:, clusters.index].layers["dadt"].toarray().T
sorted_dadt = sorted_dadt / np.std(sorted_dadt, axis=1).reshape(-1, 1)

sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(5, 5))
sns.clustermap(pd.DataFrame(sorted_dadt, columns=col_annots),
               cmap='coolwarm', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, 
               row_colors=row_colors,
               col_colors=clusters_pdt_colors,
               vmin=-3, vmax=3)
plt.gca().axis("off")
fig.tight_layout()
plt.savefig(save_dir + "/heatmap_atac2atac_binned_dx_leiden_{}.png".format(resolution), bbox_inches='tight')
plt.close()
del sorted_dadt

sorted_x_raw = adata_a2a[:, clusters.index].layers["a_raw"].toarray()
sorted_x_raw = scipy.stats.zscore(sorted_x_raw).T
sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(5, 5))
sns.clustermap(pd.DataFrame(sorted_x_raw, columns=col_annots),
               cmap='viridis', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, 
               row_colors=row_colors, col_colors=clusters_pdt_colors,
               vmin=-3, vmax=3)
fig.tight_layout()
plt.savefig(save_dir + "/heatmap_atac2atac_binned_x_leiden_{}.png".format(resolution), bbox_inches='tight')
plt.close()
del sorted_x_raw
del adata_a2a

#########
# spliced
save_dir = dir_path + "/spliced_heatmap"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

x_raw = adata_r.layers["s_raw"].toarray()
dxdt = adata_r.layers["dsdt"].toarray()
var_name = adata_r.var_names.to_numpy()
dxdt = dxdt / np.std(dxdt, axis=0)
x_raw = scipy.stats.zscore(x_raw)
resolution = 0.4
adata_bin = ad.AnnData(X=dxdt.T)
adata_bin.layers["rec_x"] = x_raw.T
adata_bin.obs_names = var_name
sc.pp.neighbors(adata_bin, n_neighbors=30, metric="cosine", n_pcs=None, use_rep="X")
sc.tl.leiden(adata_bin, resolution=resolution)
sc.tl.umap(adata_bin)
fig, ax = plt.subplots(figsize=(5, 5))
sc.pl.umap(adata_bin, color="leiden")
fig.tight_layout()
plt.savefig(save_dir + "/umap_spliced_leiden_resolution_{}.png".format(resolution), bbox_inches='tight', dpi=300)
plt.close()

def generate_cluster_colors(num_clusters):
    cmap = plt.get_cmap('tab20')
    cluster_colors = [cmap(i) for i in np.linspace(0, 1, num_clusters)]
    return cluster_colors

clusters = adata_bin.obs["leiden"]
num_clusters = len(adata_bin.obs["leiden"].cat.categories)
cluster_colors = generate_cluster_colors(num_clusters)
sorted_dxdt = dxdt[:, np.argsort(clusters)]
sorted_var_name = var_name[np.argsort(clusters)]
sorted_dxdt = sorted_dxdt.T
sorted_x_raw = x_raw[:, np.argsort(clusters)]
sorted_x_raw = sorted_x_raw.T
clusters = clusters[np.argsort(clusters)]
row_colors = [cluster_colors[int(i)] for i in clusters]

# set columns colors
col_annots = [list(adata_r.obs["dpt_pseudotime"])]
col_annots = list(zip(*col_annots))
col_annots = pd.MultiIndex.from_tuples(col_annots, names=["pseudotime"])
pdt_labels = col_annots.get_level_values("pseudotime")
pdt_pal = sns.color_palette("viridis", pdt_labels.unique().size)
pdt_lut = dict(zip(pdt_labels.unique().sort_values(), pdt_pal))
pdt_colors = pd.Series(pdt_labels, index=col_annots).map(pdt_lut)
clusters_pdt_colors = pd.concat([pdt_colors], axis=1)

sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(5, 5))
sns.clustermap(pd.DataFrame(sorted_x_raw, columns=col_annots),
               cmap='viridis', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, 
               row_colors=row_colors, col_colors=clusters_pdt_colors,
               vmin=-3, vmax=3)
fig.tight_layout()
plt.savefig(save_dir + "/spliced_heatmap_clustering_binned_dx_leiden_{}_x.png".format(resolution), bbox_inches='tight')
plt.close()

sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(5, 5))
sns.clustermap(pd.DataFrame(sorted_dxdt, columns=col_annots),
               cmap='coolwarm', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, 
               row_colors=row_colors, col_colors=clusters_pdt_colors,
               vmin=-3, vmax=3)
plt.gca().axis("off")
fig.tight_layout()
plt.savefig(save_dir + "/spliced_heatmap_clustering_binned_dx_leiden_{}.png".format(resolution), bbox_inches='tight')
plt.close()

del dxdt, sorted_dxdt, x_raw, sorted_x_raw

# atac 2 rna
## first, determine the min/max pseudotime for plot
adata_a2r = adata_r[adata_r.obs["modality"] == "atac"]
adata_r2r = adata_r[(adata_r.obs["modality"] == "rna") | (adata_a.obs["modality"] == "multiome"), :]
max_pseudotime = min(adata_a2r.obs["dpt_pseudotime"].max(),
                     adata_r2r.obs["dpt_pseudotime"].max())
min_pseudotime = max(adata_a2r.obs["dpt_pseudotime"].min(),
                     adata_r2r.obs["dpt_pseudotime"].min())
adata_a2r = adata_a2r[(adata_a2r.obs["dpt_pseudotime"] >= min_pseudotime) & (adata_a2r.obs["dpt_pseudotime"] <= max_pseudotime), :]
adata_r2r = adata_r2r[(adata_r2r.obs["dpt_pseudotime"] >= min_pseudotime) & (adata_r2r.obs["dpt_pseudotime"] <= max_pseudotime), :]

col_annots = [adata_a2r.obs["dpt_pseudotime"].to_list()]
col_annots = list(zip(*col_annots))
col_annots = pd.MultiIndex.from_tuples(col_annots, names=["pseudotime"])
pdt_labels = col_annots.get_level_values("pseudotime")
pdt_pal = sns.color_palette("viridis", pdt_labels.unique().size)
pdt_lut = dict(zip(pdt_labels.unique().sort_values(), pdt_pal))
pdt_colors = pd.Series(pdt_labels, index=col_annots).map(pdt_lut)
clusters_pdt_colors = pd.concat([pdt_colors], axis=1)
sorted_dxdt = adata_a2r[:, clusters.index].layers["dsdt"].toarray().T
sorted_dxdt = sorted_dxdt / np.std(sorted_dxdt, axis=1).reshape(-1, 1)

sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(5, 5))
sns.clustermap(pd.DataFrame(sorted_dxdt, columns=col_annots),
               cmap='coolwarm', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, 
               row_colors=row_colors,
               col_colors=clusters_pdt_colors,
               vmin=-3, vmax=3)
plt.gca().axis("off")
fig.tight_layout()
plt.savefig(save_dir + "/heatmap_atac2rna_binned_dx_leiden_{}.png".format(resolution), bbox_inches='tight')
plt.close()
del adata_a2r

## rna 2 rna
col_annots = [adata_r2r.obs["dpt_pseudotime"].to_list()]
col_annots = list(zip(*col_annots))
col_annots = pd.MultiIndex.from_tuples(col_annots, names=["pseudotime"])
pdt_labels = col_annots.get_level_values("pseudotime")
pdt_pal = sns.color_palette("viridis", pdt_labels.unique().size)
pdt_lut = dict(zip(pdt_labels.unique().sort_values(), pdt_pal))
pdt_colors = pd.Series(pdt_labels, index=col_annots).map(pdt_lut)
clusters_pdt_colors = pd.concat([pdt_colors], axis=1)
sorted_dxdt = adata_r2r[:, clusters.index].layers["dsdt"].toarray().T
sorted_dxdt = sorted_dxdt / np.std(sorted_dxdt, axis=1).reshape(-1, 1)

sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(5, 5))
sns.clustermap(pd.DataFrame(sorted_dxdt, columns=col_annots),
               cmap='coolwarm', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, 
               row_colors=row_colors,
               col_colors=clusters_pdt_colors,
               vmin=-3, vmax=3)
plt.gca().axis("off")
fig.tight_layout()
plt.savefig(save_dir + "/heatmap_rna2rna_binned_dx_leiden_{}.png".format(resolution), bbox_inches='tight')
plt.close()
del sorted_dxdt

sorted_x_raw = adata_r2r[:, clusters.index].layers["s_raw"].toarray()
sorted_x_raw = scipy.stats.zscore(sorted_x_raw).T
sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(5, 5))
sns.clustermap(pd.DataFrame(sorted_x_raw, columns=col_annots),
               cmap='viridis', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, 
               row_colors=row_colors, col_colors=clusters_pdt_colors,
               vmin=-3, vmax=3)
fig.tight_layout()
plt.savefig(save_dir + "/heatmap_rna2rna_binned_x_leiden_{}.png".format(resolution), bbox_inches='tight')
plt.close()
del sorted_x_raw
del adata_r2r


#########
# unspliced
save_dir = dir_path + "/unspliced_heatmap"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

x_raw = adata_r.layers["u_raw"].toarray()
dxdt = adata_r.layers["dudt"].toarray()
var_name = adata_r.var_names.to_numpy()
dxdt = dxdt / np.std(dxdt, axis=0)
x_raw = scipy.stats.zscore(x_raw)
resolution = 0.4
adata_bin = ad.AnnData(X=dxdt.T)
adata_bin.layers["rec_x"] = x_raw.T
adata_bin.obs_names = var_name
sc.pp.neighbors(adata_bin, n_neighbors=30, metric="cosine", n_pcs=None, use_rep="X")
sc.tl.leiden(adata_bin, resolution=resolution)
sc.tl.umap(adata_bin)
fig, ax = plt.subplots(figsize=(5, 5))
sc.pl.umap(adata_bin, color="leiden")
fig.tight_layout()
plt.savefig(save_dir + "/umap_unspliced_leiden_resolution_{}.png".format(resolution), bbox_inches='tight', dpi=300)
plt.close()

def generate_cluster_colors(num_clusters):
    cmap = plt.get_cmap('tab20')
    cluster_colors = [cmap(i) for i in np.linspace(0, 1, num_clusters)]
    return cluster_colors

clusters = adata_bin.obs["leiden"]
num_clusters = len(adata_bin.obs["leiden"].cat.categories)
cluster_colors = generate_cluster_colors(num_clusters)
sorted_dxdt = dxdt[:, np.argsort(clusters)]
sorted_var_name = var_name[np.argsort(clusters)]
sorted_dxdt = sorted_dxdt.T
sorted_x_raw = x_raw[:, np.argsort(clusters)]
sorted_x_raw = sorted_x_raw.T
clusters = clusters[np.argsort(clusters)]
row_colors = [cluster_colors[int(i)] for i in clusters]

# set columns colors
col_annots = [list(adata_r.obs["dpt_pseudotime"])]
col_annots = list(zip(*col_annots))
col_annots = pd.MultiIndex.from_tuples(col_annots, names=["pseudotime"])
pdt_labels = col_annots.get_level_values("pseudotime")
pdt_pal = sns.color_palette("viridis", pdt_labels.unique().size)
pdt_lut = dict(zip(pdt_labels.unique().sort_values(), pdt_pal))
pdt_colors = pd.Series(pdt_labels, index=col_annots).map(pdt_lut)
clusters_pdt_colors = pd.concat([pdt_colors], axis=1)

sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(5, 5))
sns.clustermap(pd.DataFrame(sorted_x_raw, columns=col_annots),
               cmap='viridis', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, 
               row_colors=row_colors, col_colors=clusters_pdt_colors,
               vmin=-3, vmax=3)
fig.tight_layout()
plt.savefig(save_dir + "/unspliced_heatmap_clustering_binned_dx_leiden_{}_x.png".format(resolution), bbox_inches='tight')
plt.close()

sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(5, 5))
sns.clustermap(pd.DataFrame(sorted_dxdt, columns=col_annots),
               cmap='coolwarm', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, 
               row_colors=row_colors, col_colors=clusters_pdt_colors,
               vmin=-3, vmax=3)
plt.gca().axis("off")
fig.tight_layout()
plt.savefig(save_dir + "/unspliced_heatmap_clustering_binned_dx_leiden_{}.png".format(resolution), bbox_inches='tight')
plt.close()

del dxdt, sorted_dxdt, x_raw, sorted_x_raw

# atac 2 rna
## first, determine the min/max pseudotime for plot
adata_a2r = adata_r[adata_r.obs["modality"] == "atac"]
adata_r2r = adata_r[(adata_r.obs["modality"] == "rna") | (adata_a.obs["modality"] == "multiome"), :]
max_pseudotime = min(adata_a2r.obs["dpt_pseudotime"].max(),
                     adata_r2r.obs["dpt_pseudotime"].max())
min_pseudotime = max(adata_a2r.obs["dpt_pseudotime"].min(),
                     adata_r2r.obs["dpt_pseudotime"].min())
adata_a2r = adata_a2r[(adata_a2r.obs["dpt_pseudotime"] >= min_pseudotime) & (adata_a2r.obs["dpt_pseudotime"] <= max_pseudotime), :]
adata_r2r = adata_r2r[(adata_r2r.obs["dpt_pseudotime"] >= min_pseudotime) & (adata_r2r.obs["dpt_pseudotime"] <= max_pseudotime), :]

col_annots = [adata_a2r.obs["dpt_pseudotime"].to_list()]
col_annots = list(zip(*col_annots))
col_annots = pd.MultiIndex.from_tuples(col_annots, names=["pseudotime"])
pdt_labels = col_annots.get_level_values("pseudotime")
pdt_pal = sns.color_palette("viridis", pdt_labels.unique().size)
pdt_lut = dict(zip(pdt_labels.unique().sort_values(), pdt_pal))
pdt_colors = pd.Series(pdt_labels, index=col_annots).map(pdt_lut)
clusters_pdt_colors = pd.concat([pdt_colors], axis=1)
sorted_dxdt = adata_a2r[:, clusters.index].layers["dudt"].toarray().T
sorted_dxdt = sorted_dxdt / np.std(sorted_dxdt, axis=1).reshape(-1, 1)

sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(5, 5))
sns.clustermap(pd.DataFrame(sorted_dxdt, columns=col_annots),
               cmap='coolwarm', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, 
               row_colors=row_colors,
               col_colors=clusters_pdt_colors,
               vmin=-3, vmax=3)
plt.gca().axis("off")
fig.tight_layout()
plt.savefig(save_dir + "/heatmap_atac2rna_binned_dx_leiden_{}.png".format(resolution), bbox_inches='tight')
plt.close()
del adata_a2r

## rna 2 rna
col_annots = [adata_r2r.obs["dpt_pseudotime"].to_list()]
col_annots = list(zip(*col_annots))
col_annots = pd.MultiIndex.from_tuples(col_annots, names=["pseudotime"])
pdt_labels = col_annots.get_level_values("pseudotime")
pdt_pal = sns.color_palette("viridis", pdt_labels.unique().size)
pdt_lut = dict(zip(pdt_labels.unique().sort_values(), pdt_pal))
pdt_colors = pd.Series(pdt_labels, index=col_annots).map(pdt_lut)
clusters_pdt_colors = pd.concat([pdt_colors], axis=1)
sorted_dxdt = adata_r2r[:, clusters.index].layers["dudt"].toarray().T
sorted_dxdt = sorted_dxdt / np.std(sorted_dxdt, axis=1).reshape(-1, 1)

sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(5, 5))
sns.clustermap(pd.DataFrame(sorted_dxdt, columns=col_annots),
               cmap='coolwarm', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, 
               row_colors=row_colors,
               col_colors=clusters_pdt_colors,
               vmin=-3, vmax=3)
plt.gca().axis("off")
fig.tight_layout()
plt.savefig(save_dir + "/heatmap_rna2rna_binned_dx_leiden_{}.png".format(resolution), bbox_inches='tight')
plt.close()
del sorted_dxdt

sorted_x_raw = adata_r2r[:, clusters.index].layers["u_raw"].toarray()
sorted_x_raw = scipy.stats.zscore(sorted_x_raw).T
sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(5, 5))
sns.clustermap(pd.DataFrame(sorted_x_raw, columns=col_annots),
               cmap='viridis', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, 
               row_colors=row_colors, col_colors=clusters_pdt_colors,
               vmin=-3, vmax=3)
fig.tight_layout()
plt.savefig(save_dir + "/heatmap_rna2rna_binned_x_leiden_{}.png".format(resolution), bbox_inches='tight')
plt.close()
del sorted_x_raw
del adata_r2r