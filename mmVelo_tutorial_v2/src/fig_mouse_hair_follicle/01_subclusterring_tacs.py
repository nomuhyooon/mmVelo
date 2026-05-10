import os
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

# peak-gene linkage matrix
#dir_path = "/home/nomura/Proj/mmvelo/data/10x_multiome_brain_repre_wo_IN/analysis/feature_linkage/peak_gene_linkage.mtx"
#peak_gene_linkage = mmread(dir_path).toarray()

# load anndata
dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/anndata"
adata_r = sc.read_loom(dir_path + "/adata_rna.loom", obs_names="obs_names", var_names="var_names")
adata_a = sc.read_loom(dir_path + "/adata_atac.loom", obs_names="obs_names", var_names="var_names")

# make dir
dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/TAC_subclustering"
if not os.path.exists(dir_path):
    os.mkdir(dir_path)
    
def plot_umap(adata, dir_name, n_neighbors=30, min_dist=0.2, cluster_name="clusters", 
              fig_name="umap.png", legend_loc="right margin", color_map=None):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    sc.pl.umap(adata, return_fig=True, color=cluster_name, legend_loc=legend_loc, color_map=color_map)
    plt.savefig(dir_name+"/"+fig_name, bbox_inches='tight')
    plt.close(fig)
    
save_dir = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/TAC_subclustering"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
plot_umap(adata_r, save_dir, cluster_name="clusters", fig_name="umap_clusters.png")

adata_sub = ad.AnnData(adata_r.X.copy())
adata_sub.obs_names, adata_sub.var_names = adata_r.obs_names, adata_r.var_names
adata_sub.obs["clusters"] = adata_r.obs["clusters"]
adata_sub.uns["clusters_colors"] = adata_r.uns["clusters_colors"]
adata_sub.obsm["latent"] = adata_r.obsm["latent"]

sc.pp.neighbors(adata_sub, n_neighbors=15, use_rep="latent")
sc.tl.umap(adata_sub, min_dist=0.05)
plot_umap(adata_sub, save_dir, cluster_name="clusters", fig_name="umap_clusters_latent.png")

import sys
sys.path.append("/home/nomura/Proj/mmvelo/src")
from mmvelo_multi.streamlineplot import velocity_graph
adata_z = ad.AnnData(X=adata_r.obsm["latent"])
adata_z.obs_names = adata_r.obs_names
adata_z.obsm["latent"] = adata_r.obsm["latent"]
adata_z.layers["latent"] = adata_r.obsm["latent"]
adata_z.layers["dynamics"] = adata_r.obsm["dynamics"]    
adata_z.obsm["X_umap"] = adata_sub.obsm["X_umap"]
adata_z.obsp["distances"] = adata_r.obsp["distances"]
adata_z.obsp["connectivities"] = adata_r.obsp["connectivities"]
adata_z.uns["neighbors"] = adata_r.uns["neighbors"]
velocity_graph(adata_z, vkey="dynamics", xkey="latent", n_jobs=16)
scv.tl.velocity_embedding(adata_z, basis="umap", vkey="dynamics")
adata_z.obs["clusters"] = adata_r.obs["clusters"]
adata_z.uns["clusters"+"_colors"] = adata_r.uns["clusters"+"_colors"]
scv.pl.velocity_embedding_grid(adata_z, vkey="dynamics", color="clusters", save=save_dir+"/dzdt_grid_" + "clusters" +"_tanh.png", title="dzdt", dpi=300)
scv.pl.velocity_embedding_stream(adata_z, vkey="dynamics", color="clusters",save=save_dir+"/dzdt_streamline_" + "clusters" + "_tanh_with_legend_on_data.png", title="",
                                    dpi=300, legend_loc="on data", min_mass=0)

adata_tac = adata_sub[adata_sub.obs["clusters"] == "TAC", :]
sc.pp.neighbors(adata_tac, n_neighbors=30, use_rep="latent", metric="correlation")
sc.tl.leiden(adata_tac, resolution=0.2)
plot_umap(adata_tac, save_dir, cluster_name="leiden", fig_name="umap_leiden_tac.png")

tac_sub_dict = {"0": "HS-TAC", "1": "TAC", "2": "IRS-TAC", "3": "IRS-TAC"}
adata_tac.obs["TAC_subclusters"] = (adata_tac.obs["leiden"]
    .map(lambda x: tac_sub_dict.get(x, x))
    .astype("category"))

cell_clst_dict_except_tac = dict()
for clst in adata_sub.obs.clusters.cat.categories.to_list()[:3]:
    cell_name = adata_sub[adata_sub.obs["clusters"] == clst, :].obs_names.tolist()
    for cell in cell_name:
        cell_clst_dict_except_tac[cell] = clst
cell_clst_dict_except_tac.update(adata_tac.obs["TAC_subclusters"].to_dict())

refined_clusters = list()
for cell in adata_sub.obs_names:
    refined_clusters.append(cell_clst_dict_except_tac.get(cell))
adata_sub.obs["ref_clusters"] = pd.Categorical(refined_clusters)
adata_r.obs["ref_clusters"] = pd.Categorical(refined_clusters)

plot_umap(adata_sub, save_dir, cluster_name="ref_clusters", fig_name="umap_z_refined_clusters.png")
plot_umap(adata_r, save_dir, cluster_name="ref_clusters", fig_name="umap_refined_clusters.png")

# save
dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/anndata"
file_name = "/refined_clusters.tsv"
pd.DataFrame(adata_r.obs["ref_clusters"]).to_csv(dir_path+file_name, sep="\t", header=False, index=False)

# check
pd.read_csv(dir_path+file_name, sep="\t", header=None)[0]
adata_r.obs["ref_clusters"]