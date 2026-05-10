import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
from scipy.io import mmwrite, mmread
import seaborn as sns
import scipy

np.random.seed(42)


# load anndata
dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/anndata"
adata_r = sc.read_loom(dir_path + "/adata_rna.loom", obs_names="obs_names", var_names="var_names")
adata_a = sc.read_loom(dir_path + "/adata_atac.loom", obs_names="obs_names", var_names="var_names")
pseudotime = pd.read_csv(dir_path+"/pseudotime.tsv", sep="\t", header=None)[0].to_numpy()
adata_r.obs["pseudotime"] = pseudotime

# load motif 
dir_path = "/home/nomura/Proj/mmvelo/data/share_seq_hf_/motif_score"
motif_bool_mat = mmread(dir_path + "/motif_bool.mtx") # 25000 1205
motif_ids = pd.read_csv(dir_path+"/motif_ids.tsv", sep="\t", header=None)[0].to_numpy()
motif_names = pd.read_csv(dir_path+"/motif_names.tsv", sep="\t", header=None)[0].to_numpy()
adata_a.var_names

# make dir
dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/motif_clustering_dE"
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

# motif count
motif_match_mat = adata_a.layers["a_raw"].toarray() @ motif_bool_mat.toarray()
expected_c = adata_a.layers["a_raw"].toarray().sum(1).reshape(-1, 1) @ \
            adata_a.layers["a_raw"].toarray().sum(0).reshape(1, -1) / \
            adata_a.layers["a_raw"].toarray().sum()
motif_match_mat_expected_c = expected_c @ motif_bool_mat.toarray()
motif_match_mat = (motif_match_mat - motif_match_mat_expected_c) / motif_match_mat_expected_c


#motif_match_mat = adata_a.layers["a_raw"].toarray() @ motif_bool_mat.toarray()
#motif_match_mat = (motif_match_mat - np.mean(motif_match_mat, axis=0)) / np.std(motif_match_mat, axis=0)

adata_motif = ad.AnnData(X=motif_match_mat)
adata_motif.obs["clusters"] = adata_r.obs["clusters"].to_numpy()
adata_motif.obs["pseudotime"] = adata_r.obs["pseudotime"].to_numpy()
adata_motif.obs_names = adata_r.obs_names
adata_motif.var_names = motif_names
sc.pp.neighbors(adata_motif, n_neighbors=10, metric="correlation", n_pcs=None, use_rep="X")
sc.tl.umap(adata_motif)

dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/motif_clustering_dE"
fig, ax = plt.subplots(figsize=(7, 5))
sc.pl.umap(adata_motif, color=["pseudotime", "clusters", ])
fig.tight_layout()
plt.savefig(dir_path + "/umap_motif_cell.png", bbox_inches='tight', dpi=300)
plt.close()


# motif velocity
X = adata_a.layers["a_raw"].toarray()
dX =  adata_a.layers["dadt"].toarray()
M = motif_bool_mat.toarray()
E = expected_c
# dE/dt
dE = (adata_a.layers["a_raw"].toarray().sum() * \
    (adata_a.layers["dadt"].toarray().sum(1).reshape(-1, 1) @ \
    adata_a.layers["a_raw"].toarray().sum(0).reshape(1, -1) + \
    adata_a.layers["a_raw"].toarray().sum(1).reshape(-1, 1) @ \
    adata_a.layers["dadt"].toarray().sum(0).reshape(1, -1)) + \
        
    adata_a.layers["dadt"].toarray().sum() * \
        (adata_a.layers["a_raw"].toarray().sum(1).reshape(-1, 1) @ \
        adata_a.layers["a_raw"].toarray().sum(0).reshape(1, -1))) / \
            (adata_a.layers["a_raw"].toarray().sum()**2)
            
dZdt = ((dX @ M - dE @ M) * (E @ M) - (X @ M - E @ M) * (dE @ M)) / \
        (E @ M) ** 2
print(dZdt.min(), dZdt.max())

"""
motif_match_mat = adata_a.layers["dadt"].toarray() @ motif_bool_mat.toarray()
expected_d = adata_a.layers["dadt"].toarray().sum(1).reshape(-1, 1) @ \
            adata_a.layers["dadt"].toarray().sum(0).reshape(1, -1) / \
            adata_a.layers["dadt"].toarray().sum()
motif_match_mat_expected_d = expected_d @ motif_bool_mat.toarray()
#motif_match_mat = motif_match_mat - motif_match_mat_expected_d
motif_match_mat = motif_match_mat / motif_match_mat_expected_c
"""

adata_d_motif = ad.AnnData(X=dZdt)
adata_d_motif.obs["clusters"] = adata_r.obs["clusters"].to_numpy()
adata_d_motif.obs["pseudotime"] = adata_r.obs["pseudotime"].to_numpy()
adata_d_motif.obs_names = adata_r.obs_names
adata_d_motif.var_names = motif_names
sc.pp.neighbors(adata_d_motif, n_neighbors=10, metric="cosine", n_pcs=None, use_rep="X")
sc.tl.umap(adata_d_motif)

dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/motif_clustering_dE"
fig, ax = plt.subplots(figsize=(7, 5))
sc.pl.umap(adata_d_motif, color=["pseudotime", "clusters", ])
fig.tight_layout()
plt.savefig(dir_path + "/umap_d_motif_cell.png", bbox_inches='tight', dpi=300)
plt.close()

# show differential motifs
save_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/motif_clustering_dE/dev_diff_motifs"
if not os.path.exists(save_path):
    os.mkdir(save_path)
    
diff_motifs = np.argsort(-1 * (adata_motif.X > 0.1).sum(0))[:100]
for i, idx in enumerate(diff_motifs):
    motif_name = adata_motif.var_names[idx]
    X = adata_motif[:, idx].X.reshape(-1)
    dX = adata_d_motif[:, idx].X.reshape(-1)
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(3.0 * 2, 3.0), dpi=100)
    
    cbar0 = axes[0].scatter(x = adata_r.obsm["X_umap"][:,0], y = adata_r.obsm["X_umap"][:,1], s = 1,
                    #c = X, vmin = -1, vmax = 1, cmap="viridis")
                    c = X, cmap="viridis")
    axes[0].set_xlabel("UMAP1")
    axes[0].set_ylabel("UMAP2")
    axes[0].set_title("{} motif activity".format(motif_name))
    fig.colorbar(cbar0, ax=axes[0])

    cbar1 = axes[1].scatter(x = adata_r.obsm["X_umap"][:,0], y = adata_r.obsm["X_umap"][:,1], s = 1,
                    c = dX, 
                    vmin = -np.max(np.abs(dX)), vmax = np.max(np.abs(dX)) ,cmap="coolwarm")
                    #cmap="coolwarm")
    axes[1].set_xlabel("UMAP1")
    axes[1].set_ylabel("UMAP2")
    axes[1].set_title("{} motif velocity".format(motif_name))
    fig.colorbar(cbar1, ax=axes[1])
    fig.tight_layout()
    plt.savefig(save_path + "/{}_motif_activity_velocity.png".format(motif_name),
                bbox_inches='tight', dpi=100)
    plt.close("all")
    
    
X = adata_a.layers["a_raw"].toarray().sum(1)
dX = adata_a.layers["dadt"].toarray().sum(1)
X = (X - np.mean(X)) / np.std(X)
dX = (dX - np.mean(dX)) / np.std(dX)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(3.0 * 2, 3.0), dpi=100)    
cbar0 = axes[0].scatter(x = adata_r.obsm["X_umap"][:,0], y = adata_r.obsm["X_umap"][:,1], s = 1,
                c = X, cmap="viridis")
axes[0].set_xlabel("UMAP1")
axes[0].set_ylabel("UMAP2")
axes[0].set_title("peak total accessiblity".format(motif_name))
fig.colorbar(cbar0, ax=axes[0])

cbar1 = axes[1].scatter(x = adata_r.obsm["X_umap"][:,0], y = adata_r.obsm["X_umap"][:,1], s = 1,
                c = dX, cmap="viridis")
axes[1].set_xlabel("UMAP1")
axes[1].set_ylabel("UMAP2")
axes[1].set_title("peak total velocity".format(motif_name))
fig.colorbar(cbar1, ax=axes[1])
fig.tight_layout()
plt.savefig(save_path + "/peak_total_activity_velocity.png".format(motif_name),
            bbox_inches='tight', dpi=100)
plt.close("all")
    
motif_exist_idx = []
motif_exist_name = []
tf_exist_name = []
for i, motif_name in enumerate(motif_names):
    for gene_symbol in adata_r.var_names:
        if gene_symbol in ["F3","Tfcp2", "Maf", "Prdm1", "Rbpj", "Prdm1"]:
            continue
        else:
            #if gene_symbol in motif_name:
            if (gene_symbol in motif_name) or (gene_symbol.upper() in motif_name):
                print(i, gene_symbol, motif_name)
                motif_exist_idx.append(i)
                motif_exist_name.append(motif_name)
                tf_exist_name.append(gene_symbol)

#overlap
#1 Fap TFAP2A
#104 Tfcp2 Tfcp2l1
#122 Nf1 HNF1B
#144 Tfcp2 Tfcp2l1
#207 Fap TFAP2C
#247 Maf Bach1::Mafk
#546 Maf Mafb
#839 Prdm1 Prdm15
#843 Rbpj Rbpjl
#1100 Prdm1 Prdm14
#1143 Maf Mafg
len(motif_exist_idx) # 55
len(set(motif_exist_idx)) # 55
len(set(motif_exist_name)) # 47
len(set(tf_exist_name)) # 45



save_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/motif_clustering_dE/dev_diff_motifs_exist_TF_z_score"
if not os.path.exists(save_path):
    os.mkdir(save_path)
    
for i, idx in enumerate(motif_exist_idx):
    if idx > 200:
        break
    motif_name = adata_motif.var_names[idx]
    #X = adata_motif[:, idx].X.reshape(-1)
    #dX = adata_d_motif[:, idx].X.reshape(-1)
    X = scipy.stats.zscore(adata_motif[:, idx].X.reshape(-1).toarray())
    dX = scipy.stats.zscore(adata_d_motif[:, idx].X.reshape(-1).toarray())
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(3.0 * 2, 3.0), dpi=100)
    
    cbar0 = axes[0].scatter(x = adata_r.obsm["X_umap"][:,0], y = adata_r.obsm["X_umap"][:,1], s = 1,
                    #c = X, vmin = -1, vmax = 1, cmap="viridis")
                    c = X, cmap="viridis")
    axes[0].set_xlabel("UMAP1")
    axes[0].set_ylabel("UMAP2")
    axes[0].set_title("{} motif activity".format(motif_name))
    fig.colorbar(cbar0, ax=axes[0])

    cbar1 = axes[1].scatter(x = adata_r.obsm["X_umap"][:,0], y = adata_r.obsm["X_umap"][:,1], s = 1,
                    c = dX, 
                    vmin = -np.max(np.abs(dX)), vmax = np.max(np.abs(dX)) ,cmap="coolwarm")
                    #cmap="coolwarm")
    axes[1].set_xlabel("UMAP1")
    axes[1].set_ylabel("UMAP2")
    axes[1].set_title("{} motif velocity".format(motif_name))
    fig.colorbar(cbar1, ax=axes[1])
    fig.tight_layout()
    plt.savefig(save_path + "/{}_motif_activity_velocity.png".format(motif_name),
                bbox_inches='tight', dpi=100)
    plt.close("all")

save_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/motif_clustering_dE/dev_diff_motifs_exist_TF"
if not os.path.exists(save_path):
    os.mkdir(save_path)
    
for i, idx in enumerate(motif_exist_idx):
    if idx > 200:
        break
    motif_name = adata_motif.var_names[idx]
    X = adata_motif[:, idx].X.reshape(-1)
    dX = adata_d_motif[:, idx].X.reshape(-1)
    #X = scipy.stats.zscore(adata_motif[:, idx].X.reshape(-1).toarray())
    #dX = scipy.stats.zscore(adata_d_motif[:, idx].X.reshape(-1).toarray())
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(3.0 * 2, 3.0), dpi=100)
    
    cbar0 = axes[0].scatter(x = adata_r.obsm["X_umap"][:,0], y = adata_r.obsm["X_umap"][:,1], s = 1,
                    #c = X, vmin = -1, vmax = 1, cmap="viridis")
                    c = X, cmap="viridis")
    axes[0].set_xlabel("UMAP1")
    axes[0].set_ylabel("UMAP2")
    axes[0].set_title("{} motif activity".format(motif_name))
    fig.colorbar(cbar0, ax=axes[0])

    cbar1 = axes[1].scatter(x = adata_r.obsm["X_umap"][:,0], y = adata_r.obsm["X_umap"][:,1], s = 1,
                    c = dX, 
                    vmin = -np.max(np.abs(dX)), vmax = np.max(np.abs(dX)) ,cmap="coolwarm")
                    #cmap="coolwarm")
    axes[1].set_xlabel("UMAP1")
    axes[1].set_ylabel("UMAP2")
    axes[1].set_title("{} motif velocity".format(motif_name))
    fig.colorbar(cbar1, ax=axes[1])
    fig.tight_layout()
    plt.savefig(save_path + "/{}_motif_activity_velocity.png".format(motif_name),
                bbox_inches='tight', dpi=100)
    plt.close("all")


adata_moitf_exist = adata_motif[:, motif_exist_idx]
adata_moitf_exist.obs["clusters"] = adata_r.obs["clusters"].to_numpy()
adata_moitf_exist.obs["pseudotime"] = adata_r.obs["pseudotime"].to_numpy()
adata_moitf_exist.obs_names = adata_r.obs_names
sc.pp.neighbors(adata_moitf_exist, n_neighbors=10, metric="correlation", n_pcs=None, use_rep="X")
sc.tl.umap(adata_moitf_exist)

dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/motif_clustering_dE"
fig, ax = plt.subplots(figsize=(7, 5))
sc.pl.umap(adata_moitf_exist, color=["pseudotime", "clusters", ])
fig.tight_layout()
plt.savefig(dir_path + "/umap_motif_exist_cell.png", bbox_inches='tight', dpi=300)
plt.close()


adata_d_moitf_exist = adata_d_motif[:, motif_exist_idx]
adata_d_moitf_exist.obs["clusters"] = adata_r.obs["clusters"].to_numpy()
adata_d_moitf_exist.obs["pseudotime"] = adata_r.obs["pseudotime"].to_numpy()
adata_d_moitf_exist.obs_names = adata_r.obs_names
sc.pp.neighbors(adata_d_moitf_exist, n_neighbors=10, metric="cosine", n_pcs=None, use_rep="X")
sc.tl.umap(adata_d_moitf_exist)

dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/motif_clustering_dE"
fig, ax = plt.subplots(figsize=(7, 5))
sc.pl.umap(adata_d_moitf_exist, color=["pseudotime", "clusters", ])
fig.tight_layout()
plt.savefig(dir_path + "/umap_d_motif_exist_cell.png", bbox_inches='tight', dpi=300)
plt.close()



# cluster-wise pseudotime
# maybe this should be done with TAC-subclustering i.e. IRS-TAC, HS-CC TAC
set(adata_r.obs["clusters"])
cell_clusters = ['TAC', 'Inner Root Sheath', 'Hair Shaft-Cuticle/Cortex', 'Medulla']
sorted_cell_barcode = []
for cell_cluster in cell_clusters:
    cell_cluster
    adata_r_cluster = adata_r[adata_r.obs["clusters"] == cell_cluster, :]
    sorted_cells = list(adata_r_cluster.obs_names[np.argsort(adata_r_cluster.obs["pseudotime"])])
    sorted_cell_barcode.append(sorted_cells)
sorted_cell_barcode = sum(sorted_cell_barcode,[])
adata_r = adata_r[sorted_cell_barcode, :]
adata_a = adata_a[sorted_cell_barcode, :]
adata_moitf_exist = adata_moitf_exist[sorted_cell_barcode, :]
adata_d_moitf_exist = adata_d_moitf_exist[sorted_cell_barcode, :]

adata_r[adata_r.obs["clusters"] == "Hair Shaft-Cuticle/Cortex", :].obs["pseudotime"].max()
adata_r[adata_r.obs["clusters"] == "Medulla", :].obs["pseudotime"].max()
adata_r[adata_r.obs["clusters"] == "Inner Root Sheath", :].obs["pseudotime"].max()

motif_raw = adata_moitf_exist.X.toarray()
motif_dt = adata_d_moitf_exist.X.toarray()

var_name = adata_moitf_exist.var_names.to_numpy()
#motif_dt = motif_dt / np.std(motif_dt, axis=0)
motif_dt = scipy.stats.zscore(motif_dt)
motif_raw = scipy.stats.zscore(motif_raw)

"""
# not ordering by clusters
a_raw = adata_a.layers["a_raw"].toarray()[np.argsort(adata_r.obs["pseudotime"]), :]
dadt = adata_a.layers["dadt"].toarray()[np.argsort(adata_r.obs["pseudotime"]), :]
"""
"""


"""

# clustering
resolution = 1.0
adata_bin = ad.AnnData(X=motif_dt.T)
adata_bin.layers["rec_x"] = motif_raw.T
adata_bin.obs_names = var_name
sc.pp.neighbors(adata_bin, n_neighbors=15, metric="correlation", n_pcs=None, use_rep="X")
sc.tl.leiden(adata_bin, resolution=resolution)
sc.tl.umap(adata_bin)

dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/motif_clustering_dE"
fig, ax = plt.subplots(figsize=(5, 5))
sc.pl.umap(adata_bin, color="leiden")
fig.tight_layout()
plt.savefig(dir_path + "/umap_leiden_resolution_{}.png".format(resolution), bbox_inches='tight', dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(5, 5))
sc.pl.umap(adata_bin, color="leiden")
plt.gca().axis("off")
plt.title("")
fig.tight_layout()
plt.savefig(dir_path + "/umap_leiden_resolution_{}_blank.png".format(resolution), bbox_inches='tight', dpi=300)
plt.close()

def generate_cluster_colors(num_clusters):
    cmap = plt.get_cmap('tab20')
    cluster_colors = [cmap(i) for i in np.linspace(0, 1, num_clusters)]
    return cluster_colors

clusters = adata_bin.obs["leiden"]
num_clusters = len(adata_bin.obs["leiden"].cat.categories)
cluster_colors = generate_cluster_colors(num_clusters)

sorted_dadt = motif_dt[:, np.argsort(clusters)]
sorted_var_name = var_name[np.argsort(clusters)]


sorted_dadt = sorted_dadt.T
sorted_x_raw = motif_raw[:, np.argsort(clusters)]
sorted_x_raw = sorted_x_raw.T
clusters = clusters[np.argsort(clusters)]
row_colors = [cluster_colors[int(i)] for i in clusters]


# set columns colors
col_annots = [list(adata_r.obs["clusters"]), list(adata_r.obs["pseudotime"])]
col_annots = list(zip(*col_annots))
col_annots = pd.MultiIndex.from_tuples(col_annots, names=["clusters", "pseudotime"])

sc.pl.umap(adata_r, color="clusters")
plt.close("all")
clusters_labels = col_annots.get_level_values("clusters")
clusters_pal = [matplotlib.colors.to_rgb(hex) for hex in adata_r.uns["clusters_colors"]]
clusters_lut = dict(zip(map(str, clusters_labels.unique()), clusters_pal))
clusters_colors = pd.Series(clusters_labels, index=col_annots).map(clusters_lut) 

pdt_labels = col_annots.get_level_values("pseudotime")
pdt_pal = sns.color_palette("viridis", pdt_labels.unique().size)
pdt_lut = dict(zip(pdt_labels.unique().sort_values(), pdt_pal))
pdt_colors = pd.Series(pdt_labels, index=col_annots).map(pdt_lut)
clusters_pdt_colors = pd.concat([clusters_colors, pdt_colors], axis=1)



sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(5, 5))
sns.clustermap(pd.DataFrame(sorted_dadt, columns=col_annots),
               cmap='coolwarm', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, 
               row_colors=row_colors, col_colors=clusters_pdt_colors,
)
               #vmin=-3, vmax=3)
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_clustering_binned_dx_leiden_{}.png".format(resolution), bbox_inches='tight')
plt.close("all")

sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(5, 5))
sns.clustermap(pd.DataFrame(sorted_dadt, columns=col_annots),
               cmap='coolwarm', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, 
               row_colors=row_colors, col_colors=clusters_pdt_colors,
)
               #vmin=-3, vmax=3)
plt.gca().axis("off")
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_clustering_binned_dx_leiden_{}_blank.png".format(resolution), bbox_inches='tight', dpi=300)
plt.close()


sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(5, 5))
sns.clustermap(pd.DataFrame(sorted_x_raw, columns=col_annots),
               cmap='viridis', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, 
               row_colors=row_colors, col_colors=clusters_pdt_colors,
               vmin=-4, vmax=4)
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_clustering_binned_dx_leiden_{}_x.png".format(resolution), bbox_inches='tight')
plt.close()

sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(5, 5))
sns.clustermap(pd.DataFrame(sorted_x_raw, columns=col_annots),
               cmap='viridis', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, 
               row_colors=row_colors, col_colors=clusters_pdt_colors,
               vmin=-4, vmax=4)
plt.gca().axis("off")
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_clustering_binned_dx_leiden_{}_x_blank.png".format(resolution), bbox_inches='tight', dpi=300)
plt.close()



# save adata to perform motif enrichment analysis
dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/dadt_clustering"
adata_bin.write_loom(dir_path+"/adata_dadt_cluster.loom", write_obsm_varm=True)


# read adata
## change environment to scenicplus
import scanpy as sc
import pycistarget
import pyranges as pr
import pickle
import statsmodels.api as sm

dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/dadt_clustering"
adata_peak = sc.read_loom(dir_path+"/adata_dadt_cluster.loom")
adata_peak.obs_names = adata_peak.obs["obs_names"]
peaks = list(adata_peak.obs_names)

region_sets = dict()
for clst in set(adata_peak.obs["leiden"]):
    clst_peaks = adata_peak.obs_names[adata_peak.obs["leiden"] == clst]
    clst_peaks = pr.from_dict({"Chromosome": [peak.split(':')[0] for peak in clst_peaks],
                      "Start": [int(peak.split(':')[1].split('-')[0]) for peak in clst_peaks],
                      "End": [int(peak.split(':')[1].split('-')[1]) for peak in clst_peaks]})
    key = f'{clst}'
    region_sets[key] = clst_peaks


from pycistarget.motif_enrichment_cistarget import *
cistarget_dict = run_cistarget(ctx_db = '/home/nomura/Proj/mmvelo/pycistarget/mm10_screen_v10_clust.regions_vs_motifs.rankings.feather',
                                                      region_sets = region_sets,
                                                      specie = 'mus_musculus',
                                                      auc_threshold = 0.005,
                                                      nes_threshold = 3.0,
                                                      rank_threshold = 0.05,
                                                      annotation = ['Direct_annot', 'Orthology_annot'],
                                                      annotation_version = 'v10nr_clust',
                                                      path_to_motif_annotations = '/home/nomura/Proj/mmvelo/pycistarget/motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl',
                                                      n_cpu = 4,
                                                      #_temp_dir='/scratch/leuven/313/vsc31305/ray_spill'
                                                      )

dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/dadt_clustering"
with open(dir_path + '/cisTarget_dict.pkl', 'wb') as f:
  pickle.dump(cistarget_dict, f)

infile = open(dir_path + '/cisTarget_dict.pkl', 'rb')
cistarget_dict = pickle.load(infile)
infile.close()

#cistarget_results(cistarget_dict, name='0')
for clst in set(adata_peak.obs["leiden"]):
    out_file = dir_path + f'/cluster_{clst}_motif_enricment.html'
    cistarget_dict[clst].motif_enrichment.to_html(open(out_file, 'w'), escape=False, col_space=80)






adata_r_exn
lowess = sm.nonparametric.lowess
frac = 300 / adata_r_exn.n_obs # use 10 neighbor cells for lowess regression
p_time = np.linspace(0, 1, adata_r_exn.n_obs)

dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/dadt_clustering"
for clst in set(adata_peak.obs["leiden"]):
    adata_peak_clst = adata_peak[adata_peak.obs["leiden"] == clst, :]
    peak_clst_dadt = adata_peak_clst.X.toarray().mean(0)

    p_time = np.linspace(0, 1, peak_clst_dadt.shape[0])
    smooth_peak_dadt = lowess(peak_clst_dadt, p_time, frac=frac, return_sorted=False)
    
    norm_peak_clst_dadt = peak_clst_dadt / np.max(np.abs(smooth_peak_dadt))
    norm_smooth_peak_dadt = smooth_peak_dadt / np.max(np.abs(smooth_peak_dadt))
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(p_time, norm_smooth_peak_dadt, color="blue", linewidth=3)
    ax.scatter(p_time, norm_peak_clst_dadt, s=0.1, color="blue")
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('pseudotime')
    ax.set_ylabel('mean velocity')
    ax.set_title(f'Mean velocity in cluster {clst}')
    fig.tight_layout()
    plt.savefig(dir_path + f"/clst_{clst}_mean_vel.png", bbox_inches='tight', dpi=300)
    plt.close()

    tf_motifs = cistarget_dict[clst].motif_enrichment

    tf_lists = list(cistarget_dict[clst].motif_enrichment["Direct_annot"][:5]) + \
            list(cistarget_dict[clst].motif_enrichment["Orthology_annot"][:5])

    tf_list = []
    for i in range(len(tf_lists)):
        if isinstance(tf_lists[i], float):
            continue
        tf_lists[i] = tf_lists[i].split(", ")
        tf_list.append(tf_lists[i])
    tf_list = sum(tf_list, [])

    exists_tf_list = []
    for tf in tf_list:
        if tf in adata_r.var_names:
            exists_tf_list.append(tf)


    dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/dadt_clustering"
    file_path = dir_path + f"/tf_in_cluster_{clst}"
    if not os.path.exists(file_path):
        os.mkdir(file_path)

    for tf in exists_tf_list:
        argsort = np.argsort(adata_r_exn.obs["pseudotime"])
        p_time = np.linspace(0, 1, adata_r_exn.n_obs)
        tf_s = adata_r_exn[argsort, tf].layers["s_raw"].toarray().reshape(-1)
        tf_s = tf_s / np.std(tf_s)
        smooth_tf_s = lowess(tf_s, p_time, frac=frac, return_sorted=False)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(p_time, tf_s, s=0.1, color = "red")
        ax.plot(p_time, smooth_tf_s, color="red", linewidth=3)

        ax.scatter(p_time, peak_clst_dadt, s=0.1, color="blue")
        ax.plot(p_time, smooth_peak_dadt, color="blue", linewidth=3)
        ax.set_xlabel('pseudotime')
        ax.set_ylabel('mean expr/vel')
        ax.set_title(f'{tf}  mean expr vs {clst} mean peak vel')
        fig.tight_layout()
        plt.savefig(file_path + f"/{tf}_mean_expr.png", bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(figsize=(5, 5))
        norm_tf_s = tf_s / max(smooth_tf_s)
        norm_smooth_tf_s = smooth_tf_s / max(smooth_tf_s)
        norm_peak_clst_dadt = peak_clst_dadt / np.max(np.abs(smooth_peak_dadt))
        norm_smooth_peak_dadt = smooth_peak_dadt / np.max(np.abs(smooth_peak_dadt))

        ax.scatter(p_time, norm_tf_s, s=0.1, color="red")
        ax.plot(p_time, norm_smooth_tf_s, color="red", linewidth=3)
        ax.scatter(p_time, norm_peak_clst_dadt, s=0.1, color="blue")
        ax.plot(p_time, norm_smooth_peak_dadt, color="blue", linewidth=3)
        ax.set_ylim(-1.5, 1.5)

        ax.set_xlabel('pseudotime')
        ax.set_ylabel('norm mean expr/vel')
        ax.set_title(f'{tf}  mean expr vs {clst} mean peak vel')
        fig.tight_layout()
        plt.savefig(file_path + f"/{tf}_mean_expr_norm.png", bbox_inches='tight')
        plt.close()

        # for fig
        fig, ax = plt.subplots(figsize=(5, 5))

        ax.scatter(p_time, norm_tf_s, s=0.1, color="red")
        ax.plot(p_time, norm_smooth_tf_s, color="red", linewidth=3)
        ax.scatter(p_time, norm_peak_clst_dadt, s=0.1, color="blue")
        ax.plot(p_time, norm_smooth_peak_dadt, color="blue", linewidth=3)
        ax.set_ylim(-1.5, 1.5)

        ax.set_xlabel('pseudotime')
        ax.set_ylabel('norm mean expr/vel')
        plt.gca().axis("off")
        fig.tight_layout()
        plt.savefig(file_path + f"/{tf}_mean_expr_norm_blank.png", bbox_inches='tight', dpi=300)
        plt.close()
