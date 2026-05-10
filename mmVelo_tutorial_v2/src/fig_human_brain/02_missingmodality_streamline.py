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
dir_path = "/home/nomura/Proj/mmvelo/experiments/Greenleaf_multiome_Cond_merged_all_missing/2024-01-19T11:22:03_barch_128/downstream_analysis/result/missing_streamline"
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

adata_r_a2r = adata_r[adata_r.obs["modality"] == "atac", :] # 7407 cells
adata_a_r2a = adata_a[adata_a.obs["modality"] == "rna", :] # 8065 cells


adata_r_r2r = adata_r[(adata_r.obs["modality"] == "rna") | (adata_r.obs["modality"] == "multiome"), :] # 14925 cells
adata_a_a2a = adata_a[(adata_a.obs["modality"] == "atac") | (adata_a.obs["modality"] == "multiome"), :] # 14267 cells

sys.path.append("/home/nomura/Proj/mmvelo/src")
from mmvelo_multi.streamlineplot import velocity_graph

save_dir = dir_path + "/atac2rna"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# dsdt
adata_r_a2r.layers["dsdt"] = adata_r_a2r.layers["dsdt"].toarray()
sc.pp.neighbors(adata_r_a2r, n_neighbors=30, use_rep="latent")
velocity_graph(adata_r_a2r, vkey="dsdt", xkey="s_raw", n_jobs=16)
scv.tl.velocity_embedding(adata_r_a2r, basis="umap", vkey="dsdt")
scv.pl.velocity_embedding_grid(adata_r_a2r, vkey="dsdt", color="ref_clusters", save=save_dir+"/dsdt_grid_" + "clusters" +"_tanh.png", title="dsdt", dpi=300)
scv.pl.velocity_embedding_stream(adata_r_a2r, vkey="dsdt", color="ref_clusters", save=save_dir+"/dsdt_streamline_" + "clusters" + "_tanh_with_legend.png",
                                    title="", dpi=300, legend_loc="right margin", )
scv.pl.velocity_embedding_stream(adata_r_a2r, vkey="dsdt", color="ref_clusters", save=save_dir+"/dsdt_streamline_" + "clusters" + "_tanh.png",
                                    title="", dpi=300, legend_loc="none", )
scv.pl.velocity_embedding_stream(adata_r_a2r, vkey="dsdt", color="ref_clusters",save=save_dir+"/dsdt_streamline_" + "clusters" + "_tanh_with_legend_on_data.png", title="",
                                    dpi=300, legend_loc="on data", )

fig, ax = plt.subplots(figsize=(8, 6))
sc.pl.umap(adata_r, ax=ax, show=False, size=0.1, alpha=0.)
sc.pl.umap(adata_r_r2r, ax=ax, show=False, size=50, alpha=0.06)
scv.pl.velocity_embedding_stream(adata_r_a2r, vkey="dsdt", color="ref_clusters", ax=ax,
                                    title="", dpi=300, legend_loc="none", show=False, alpha=0.8)
plt.savefig(save_dir + "/background_dsdt_streamline.png", bbox_inches='tight', dpi=300)
plt.close("all")


# dudt
adata_r_a2r.layers["dudt"] = adata_r_a2r.layers["dudt"].toarray()
velocity_graph(adata_r_a2r, vkey="dudt", xkey="u_raw", n_jobs=16)
scv.tl.velocity_embedding(adata_r_a2r, basis="umap", vkey="dudt")
scv.pl.velocity_embedding_grid(adata_r_a2r, vkey="dudt", color="ref_clusters", save=save_dir+"/dudt_grid_" + "clusters" +"_tanh.png", title="dudt", dpi=300)
scv.pl.velocity_embedding_stream(adata_r_a2r, vkey="dudt", color="ref_clusters", save=save_dir+"/dudt_streamline_" + "clusters" + "_tanh_with_legend.png",
                                    title="", dpi=300, legend_loc="right margin", )
scv.pl.velocity_embedding_stream(adata_r_a2r, vkey="dudt", color="ref_clusters", save=save_dir+"/dudt_streamline_" + "clusters" + "_tanh.png",
                                    title="", dpi=300, legend_loc="none", )
scv.pl.velocity_embedding_stream(adata_r_a2r, vkey="dudt", color="ref_clusters",save=save_dir+"/dudt_streamline_" + "clusters" + "_tanh_with_legend_on_data.png", title="",
                                    dpi=300, legend_loc="on data", )

fig, ax = plt.subplots(figsize=(8, 6))
sc.pl.umap(adata_r, ax=ax, show=False, size=0.1, alpha=0.)
sc.pl.umap(adata_r_r2r, ax=ax, show=False, size=50, alpha=0.06)
scv.pl.velocity_embedding_stream(adata_r_a2r, vkey="dudt", color="ref_clusters", ax=ax,
                                    title="", dpi=300, legend_loc="none", show=False, alpha=0.8)
plt.savefig(save_dir + "/background_dudt_streamline.png", bbox_inches='tight', dpi=300)
plt.close("all")



save_dir = dir_path + "/rna2rna"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# dsdt
adata_r_r2r.layers["dsdt"] = adata_r_r2r.layers["dsdt"].toarray()
sc.pp.neighbors(adata_r_r2r, n_neighbors=30, use_rep="latent")
velocity_graph(adata_r_r2r, vkey="dsdt", xkey="s_raw", n_jobs=16)
scv.tl.velocity_embedding(adata_r_r2r, basis="umap", vkey="dsdt")
scv.pl.velocity_embedding_grid(adata_r_r2r, vkey="dsdt", color="ref_clusters", save=save_dir+"/dsdt_grid_" + "clusters" +"_tanh.png", title="dsdt", dpi=300)
scv.pl.velocity_embedding_stream(adata_r_r2r, vkey="dsdt", color="ref_clusters", save=save_dir+"/dsdt_streamline_" + "clusters" + "_tanh_with_legend.png",
                                    title="", dpi=300, legend_loc="right margin", )
scv.pl.velocity_embedding_stream(adata_r_r2r, vkey="dsdt", color="ref_clusters", save=save_dir+"/dsdt_streamline_" + "clusters" + "_tanh.png",
                                    title="", dpi=300, legend_loc="none", )
scv.pl.velocity_embedding_stream(adata_r_r2r, vkey="dsdt", color="ref_clusters",save=save_dir+"/dsdt_streamline_" + "clusters" + "_tanh_with_legend_on_data.png", title="",
                                    dpi=300, legend_loc="on data", )

fig, ax = plt.subplots(figsize=(8, 6))
sc.pl.umap(adata_r, ax=ax, show=False, size=0.1, alpha=0.)
sc.pl.umap(adata_r_a2r, ax=ax, show=False, size=50, alpha=0.06)
scv.pl.velocity_embedding_stream(adata_r_r2r, vkey="dsdt", color="ref_clusters", ax=ax,
                                    title="", dpi=300, legend_loc="none", show=False, alpha=0.8)
plt.savefig(save_dir + "/background_dsdt_streamline.png", bbox_inches='tight', dpi=300)
plt.close("all")

# dudt
adata_r_r2r.layers["dudt"] = adata_r_r2r.layers["dudt"].toarray()
velocity_graph(adata_r_r2r, vkey="dudt", xkey="u_raw", n_jobs=16)
scv.tl.velocity_embedding(adata_r_r2r, basis="umap", vkey="dudt")
scv.pl.velocity_embedding_grid(adata_r_r2r, vkey="dudt", color="ref_clusters", save=save_dir+"/dudt_grid_" + "clusters" +"_tanh.png", title="dudt", dpi=300)
scv.pl.velocity_embedding_stream(adata_r_r2r, vkey="dudt", color="ref_clusters", save=save_dir+"/dudt_streamline_" + "clusters" + "_tanh_with_legend.png",
                                    title="", dpi=300, legend_loc="right margin", )
scv.pl.velocity_embedding_stream(adata_r_r2r, vkey="dudt", color="ref_clusters", save=save_dir+"/dudt_streamline_" + "clusters" + "_tanh.png",
                                    title="", dpi=300, legend_loc="none", )
scv.pl.velocity_embedding_stream(adata_r_r2r, vkey="dudt", color="ref_clusters",save=save_dir+"/dudt_streamline_" + "clusters" + "_tanh_with_legend_on_data.png", title="",
                                    dpi=300, legend_loc="on data", )

fig, ax = plt.subplots(figsize=(8, 6))
sc.pl.umap(adata_r, ax=ax, show=False, size=0.1, alpha=0.)
sc.pl.umap(adata_r_a2r, ax=ax, show=False, size=50, alpha=0.06)
scv.pl.velocity_embedding_stream(adata_r_r2r, vkey="dudt", color="ref_clusters", ax=ax,
                                    title="", dpi=300, legend_loc="none", show=False, alpha=0.8)
plt.savefig(save_dir + "/background_dudt_streamline.png", bbox_inches='tight', dpi=300)
plt.close("all")


save_dir = dir_path + "/rna2atac"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
# dadt
adata_a_r2a.layers["dadt"] = adata_a_r2a.layers["dadt"].toarray()
sc.pp.neighbors(adata_a_r2a, n_neighbors=30, use_rep="latent")
velocity_graph(adata_a_r2a, vkey="dadt", xkey="a_raw", n_jobs=16)
scv.tl.velocity_embedding(adata_a_r2a, basis="umap", vkey="dadt")
scv.pl.velocity_embedding_grid(adata_a_r2a, vkey="dadt", color="ref_clusters", save=save_dir+"/dadt_grid_" + "clusters" +"_tanh.png", title="dadt", dpi=300)
scv.pl.velocity_embedding_stream(adata_a_r2a, vkey="dadt", color="ref_clusters", save=save_dir+"/dadt_streamline_" + "clusters" + "_tanh_with_legend.png",
                                    title="", dpi=300, legend_loc="right margin", )
scv.pl.velocity_embedding_stream(adata_a_r2a, vkey="dadt", color="ref_clusters", save=save_dir+"/dadt_streamline_" + "clusters" + "_tanh.png",
                                    title="", dpi=300, legend_loc="none", )
scv.pl.velocity_embedding_stream(adata_a_r2a, vkey="dadt", color="ref_clusters",save=save_dir+"/dadt_streamline_" + "clusters" + "_tanh_with_legend_on_data.png", title="",
                                    dpi=300, legend_loc="on data", )

fig, ax = plt.subplots(figsize=(8, 6))
sc.pl.umap(adata_r, ax=ax, show=False, size=0.1, alpha=0.)
sc.pl.umap(adata_a_a2a, ax=ax, show=False, size=50, alpha=0.06)
scv.pl.velocity_embedding_stream(adata_a_r2a, vkey="dadt", color="ref_clusters", ax=ax,
                                    title="", dpi=300, legend_loc="none", show=False, alpha=0.8)
plt.savefig(save_dir + "/background_dadt_streamline.png", bbox_inches='tight', dpi=300)
plt.close("all")

    
    
save_dir = dir_path + "/atac2atac"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
adata_a_a2a.layers["dadt"] = adata_a_a2a.layers["dadt"].toarray()
sc.pp.neighbors(adata_a_a2a, n_neighbors=30, use_rep="latent")
velocity_graph(adata_a_a2a, vkey="dadt", xkey="a_raw", n_jobs=16)
scv.tl.velocity_embedding(adata_a_a2a, basis="umap", vkey="dadt")
scv.pl.velocity_embedding_grid(adata_a_a2a, vkey="dadt", color="ref_clusters", save=save_dir+"/dadt_grid_" + "clusters" +"_tanh.png", title="dadt", dpi=300)
scv.pl.velocity_embedding_stream(adata_a_a2a, vkey="dadt", color="ref_clusters", save=save_dir+"/dadt_streamline_" + "clusters" + "_tanh_with_legend.png",
                                    title="", dpi=300, legend_loc="right margin", )
scv.pl.velocity_embedding_stream(adata_a_a2a, vkey="dadt", color="ref_clusters", save=save_dir+"/dadt_streamline_" + "clusters" + "_tanh.png",
                                    title="", dpi=300, legend_loc="none", )
scv.pl.velocity_embedding_stream(adata_a_a2a, vkey="dadt", color="ref_clusters",save=save_dir+"/dadt_streamline_" + "clusters" + "_tanh_with_legend_on_data.png", title="",
                                    dpi=300, legend_loc="on data", )

fig, ax = plt.subplots(figsize=(8, 6))
sc.pl.umap(adata_r, ax=ax, show=False, size=0.1, alpha=0.)
sc.pl.umap(adata_a_r2a, ax=ax, show=False, size=50, alpha=0.06)
scv.pl.velocity_embedding_stream(adata_a_a2a, vkey="dadt", color="ref_clusters", ax=ax,
                                    title="", dpi=300, legend_loc="none", show=False, alpha=0.8)
plt.savefig(save_dir + "/background_dadt_streamline.png", bbox_inches='tight', dpi=300)
plt.close("all")







