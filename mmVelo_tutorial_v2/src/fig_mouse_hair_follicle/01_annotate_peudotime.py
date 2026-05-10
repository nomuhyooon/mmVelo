import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
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
dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/dadt_clustering"
if not os.path.exists(dir_path):
    os.mkdir(dir_path)
    
def plot_umap(adata, dir_name, n_neighbors=30, min_dist=0.2, cluster_name="clusters", 
              fig_name="umap.png", legend_loc="right margin", color_map=None):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    sc.pl.umap(adata, return_fig=True, color=cluster_name, legend_loc=legend_loc, color_map=color_map)
    plt.savefig(dir_name+"/"+fig_name, bbox_inches='tight')
    plt.close(fig)
    
save_dir = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/pseudotime"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
plot_umap(adata_r, save_dir, cluster_name="clusters", fig_name="umap_clusters.png")


sc.pp.neighbors(adata_r, n_neighbors=30, use_rep="latent")
sc.tl.diffmap(adata_r, random_state=1, n_comps=10)

def plot_diff_map(adata, dir_name, basis="diffmap", components=[1,2], color=None, size=None, fig_name=None):
    num_plots = len(color)
    fig, ax = plt.subplots(1, num_plots, figsize=(3, 3 * num_plots))
    sc.pl.scatter(adata, basis=basis, components=components,
                  color=color, size=size)
    plt.savefig(dir_name+"/"+fig_name, bbox_inches='tight')
    plt.close(fig)

plot_diff_map(adata_r, save_dir, components=[0,1], color="clusters",
              fig_name = "diff_comp_0_1.png")
plot_diff_map(adata_r, save_dir, components=[2,3], color="clusters",
              fig_name = "diff_comp_2_3.png")
plot_diff_map(adata_r, save_dir, components=[4,5], color="clusters",
              fig_name = "diff_comp_4_5.png")
plot_diff_map(adata_r, save_dir, components=[6,7], color="clusters",
              fig_name = "diff_comp_6_7.png")
plot_diff_map(adata_r, save_dir, components=[8,9], color="clusters",
              fig_name = "diff_comp_8_9.png")

# compute pseudotime
root_ixs = adata_r.obsm["X_diffmap"][:, 2].argmax()
adata_r.uns["iroot"] = root_ixs
sc.tl.dpt(adata_r)
plot_umap(adata_r, save_dir, cluster_name="dpt_pseudotime", fig_name="umap_dpt_pseudotime.png")

# compare with DC2
diff_comp = adata_r.obsm["X_diffmap"][:, 2]
diff_comp = diff_comp - min(diff_comp)
diff_comp = diff_comp / max(diff_comp)
diff_comp = -1 * diff_comp + 1
adata_r.obs["diff_comp"] = diff_comp
plot_umap(adata_r, save_dir, cluster_name="diff_comp", fig_name="umap_diff_comp_2.png")

# use dpt_pseudotime as a pseudotime axis.
# save diffusion maps, dpt_pseudotime
adata_r.obsm["X_diffmap"]
adata_r.obs["dpt_pseudotime"]


dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/anndata"
file_name = "/pseudotime.tsv"
pd.DataFrame(adata_r.obs["dpt_pseudotime"]).to_csv(dir_path+file_name, sep="\t", header=False, index=False)

# check
pd.read_csv(dir_path+file_name, sep="\t", header=None)[0]
adata_r.obs["dpt_pseudotime"]

dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/anndata"
file_name = "/diffusion_map.tsv"
pd.DataFrame(adata_r.obsm["X_diffmap"]).to_csv(dir_path+file_name, sep="\t", header=False, index=False)

# check
pd.read_csv(dir_path+file_name, sep="\t", header=None)
adata_r.obsm["X_diffmap"]