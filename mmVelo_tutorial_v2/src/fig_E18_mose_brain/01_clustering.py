import os
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import scanpy as sc
import scvelo as scv


np.random.seed(42)

# read anndata
save_dir = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/anndata"
adata_r = sc.read_loom(save_dir + "/adata_rna.loom", obs_names="obs_names", var_names="var_names")

# UMAP
def embed_z(adata, n_neighbors=30, min_dist=0.2, densmap=False):
    z_mat = adata.obsm["latent"]
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, densmap=densmap)
    z_embed = reducer.fit_transform(z_mat)
    adata.obsm["X_umap"] = z_embed
    return z_embed

def plot_umap(adata, dir_name, embedding=False, n_neighbors=30, min_dist=0.2, cluster_name="clusters", 
              fig_name=None, legend_loc="right margin"):
    if embedding:
        adata.obsm["X_umap"] = embed_z(adata, n_neighbors=n_neighbors, min_dist=min_dist)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    sc.pl.umap(adata, return_fig=True, color=cluster_name, legend_loc=legend_loc)
    if fig_name is None:
        plt.savefig(dir_name+"/umap_" + cluster_name + ".png", bbox_inches='tight')
    else:
        plt.savefig(dir_name+"/"+fig_name, bbox_inches='tight')
    plt.close(fig)


save_dir = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/clustering"
os.mkdir(save_dir)
plot_umap(adata_r, save_dir, embedding=True, n_neighbors=30, min_dist=0.2, cluster_name=None, fig_name="umap.png")

# save UMAP coordinate
save_dir = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/anndata"
pd.DataFrame(adata_r.obsm["X_umap"]).to_csv(save_dir+"/umap_coordinate.tsv", sep="\t", header=None, index=None)

# visualize marker gene expression
def plot_expr_umap(adata, dir_name, color=None, size=None, fig_name=None):
    num_plots = len(color)
    fig, ax = plt.subplots(1, num_plots, figsize=(3, 3 * num_plots))
    sc.pl.umap(adata, return_fig=True, color=color, size=size)
    plt.savefig(dir_name+"/"+fig_name)
    plt.close(fig)

save_dir = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/clustering/marker_gene"
os.mkdir(save_dir)
plot_expr_umap(adata_r, save_dir, color=["Cdk1", "Top2a"], fig_name="Expr_RGcyc_nIPC.png", size=100)
plot_expr_umap(adata_r, save_dir, color=["Eomes", "Elavl2", "Elavl4", "Tbr1"], fig_name="Expr_nIPC.png", size=100)
plot_expr_umap(adata_r, save_dir, color=["Vim", "Acot1", "Id3", "Gja1"], fig_name="Expr_Astro.png", size=100)
plot_expr_umap(adata_r, save_dir, color=["Wnt8b", "Fam210b", "Emx1",  "Vim", "Pax6", "Slc1a3"], fig_name="Expr_RG.png", size=100)
plot_expr_umap(adata_r, save_dir, color=["Satb2", "Lmo4"], fig_name="Expr_L_Upper.png", size=100)
plot_expr_umap(adata_r, save_dir, color=["Tle4", "Foxp2"], fig_name="Expr_L_Deeper.png", size=100)
plot_expr_umap(adata_r, save_dir, color=["Elavl4", "Neurod2", "Rbfox1"], fig_name="Expr_Intermed_to_Neuron.png", size=100)
plot_expr_umap(adata_r, save_dir, color=["Crym", "Lmo4", "Zbtb20"], fig_name="Expr_Ependymal.png", size=100)
plot_expr_umap(adata_r, save_dir, color=["Slc1a3", "Sox9", "Fgfr3", "Lpar1"], fig_name="Expr_RG_Astro_OPC.png", size=100)
plot_expr_umap(adata_r, save_dir, color=["Prox1", "Dlx5", "Meis2", "Mki67", "Neurod6"], fig_name="Expr_sub.png", size=100)

# calculate cell cycle score. Tirosh et al., 2016
# https://scrapbox.io/read-matome/cell_cycle_score
S_genes = ["Mcm5", "Pcna", "Tyms", "Tyms", "Fen1", "Mcm2", "Rrm1"]
G2M_genes = ["Mki67", "Top2a", "Cdk1", "Hmgb2", "Nusap1","Birc5", "Tpx2"]
sc.tl.score_genes_cell_cycle(adata_r, s_genes=S_genes, g2m_genes=G2M_genes)

gcc_result = scv.tl.score_genes_cell_cycle(adata_r, copy=True)
plot_expr_umap(adata_r, save_dir, color=["S_score", "G2M_score", "phase"], fig_name="cell_cycle.png", size=100)
plot_expr_umap(adata_r, save_dir, color=["G2M_score"], fig_name="cell_cycle_g2m.png", size=100)
plot_expr_umap(adata_r, save_dir, color=["S_score"], fig_name="cell_cycle_s.png", size=100)

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
scv.pl.scatter(gcc_result, color='G2M_score - S_score', size=80, frameon=True, title="Cell cycle score")
plt.savefig(save_dir+"/umap_cellcycle.png", bbox_inches='tight')
plt.close(fig)


# compute neighbors and do clustering
save_dir = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/clustering"
sc.pp.neighbors(adata_r, n_neighbors=15, use_rep="latent")
sc.tl.leiden(adata_r, resolution=0.8, key_added="leiden")
plot_umap(adata_r, save_dir, embedding=False, cluster_name="leiden",
          fig_name="umap_leiden_on_data.png", legend_loc="on data")

old_to_new = {
    "0"  : "V-SVZ", 
    "1"  : "Deeper layer",
    "2"  : "RG, Astro, OPC",
    "3"  : "Upper layer",
    "4"  : "V-SVZ",
    "5"  : "Upper layer",
    "6"  : "Ependymal cells",
    "7"  : "IPC",
    "8"  : "Upper layer",
    "9"  : "Subplate",
    "10": "Upper layer",
    "11" : "RG, Astro, OPC",
    "12" : "RG, Astro, OPC"
}

# save clustering results
adata_r.obs["clusters"] = adata_r.obs["leiden"].map(old_to_new).astype('category')
plot_umap(adata_r, save_dir, embedding=False, cluster_name="clusters", fig_name="umap_clusters.png")
plot_umap(adata_r, save_dir, embedding=False, cluster_name="clusters",
          fig_name="umap_clusters_on_data.png", legend_loc="on data")

save_dir = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/anndata"
adata_r.obs["clusters"].to_json(save_dir + "/cell_clusters.json")

# check whether data can be loaded
pd.read_json(save_dir + "/cell_clusters.json", typ="series")
adata_r.obs["clusters"]

clusters = pd.read_json(save_dir + "/cell_clusters.json", typ="series")
adata_r.obs["clusters"] = clusters.astype("category")
save_dir = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/clustering"
plot_umap(adata_r, save_dir, embedding=False, cluster_name="clusters", fig_name="umap_clusters.png")