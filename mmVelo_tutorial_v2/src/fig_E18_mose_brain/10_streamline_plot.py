import os
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import matplotlib
import anndata as ad
import scanpy as sc
import scvelo as scv
import scanpy.external as sce
from scipy.io import mmwrite, mmread
import scipy
import sys
sys.path.append("/home/nomura/Proj/mmvelo/src")
from mmvelo_multi.streamlineplot import velocity_graph

np.random.seed(42)

# load anndata
dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/anndata"
adata_r = sc.read_loom(dir_path + "/adata_rna.loom", obs_names="obs_names", var_names="var_names")
adata_a = sc.read_loom(dir_path + "/adata_atac.loom", obs_names="obs_names", var_names="var_names")

adata_r.obsm["X_umap"]  = pd.read_csv(dir_path + "/umap_coordinate.tsv", sep="\t", header=None).to_numpy()
adata_r.obs["clusters"] = pd.read_json(dir_path + "/cell_clusters.json", typ="series").astype("category")
adata_r.obs["pseudotime"] = pd.read_csv(dir_path + "/pseudotime.tsv", sep="\t", header=None)[0].to_numpy()

# make dir
dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/streamline_plot"
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

plt.rcParams['font.family'] = 'sans-serif'

adata_r
sc.pp.neighbors(adata_r, n_neighbors=100, use_rep="latent")
adata_a.uns["neighbors"] = adata_r.uns["neighbors"]
adata_a.obsp["distances"] = adata_r.obsp["distances"]
adata_a.obsp["connectivities"] = adata_r.obsp["connectivities"]

adata_r.layers["dsdt"] = adata_r.layers["dsdt"].toarray()
adata_r.layers["s_raw"] = adata_r.layers["s_raw"].toarray()
adata_a.layers["dadt"] = adata_a.layers["dadt"].toarray()
adata_a.layers["a_raw"] = adata_a.layers["a_raw"].toarray()


def plot_vec_embed_tanh(adata_r, adata_a, dir_path, latent=False, spliced=False, dsdt_obs=False, ss_model=False, dadt=False, color=None):
    if latent:
        adata_z = ad.AnnData(X=adata_r.obsm["latent"])
        adata_z.obs_names = adata_r.obs_names
        adata_z.obsm["latent"] = adata_r.obsm["latent"]
        adata_z.layers["latent"] = adata_r.obsm["latent"]
        adata_z.layers["dynamics"] = adata_r.obsm["dynamics"]    
        adata_z.obsm["X_umap"] = adata_r.obsm["X_umap"]
        adata_z.obsp["distances"] = adata_r.obsp["distances"]
        adata_z.obsp["connectivities"] = adata_r.obsp["connectivities"]
        adata_z.uns["neighbors"] = adata_r.uns["neighbors"]
        velocity_graph(adata_z, vkey="dynamics", xkey="latent", n_jobs=16)
        scv.tl.velocity_embedding(adata_z, basis="umap", vkey="dynamics")
        adata_z.obs[color] = adata_r.obs[color]
        #adata_z.uns[color+"_colors"] = adata_r.uns[color+"_colors"]
        scv.pl.velocity_embedding_grid(adata_z, vkey="dynamics", color=color, save=dir_path+"/dzdt_grid_" + color +"_tanh.png", title="", dpi=300)
        scv.pl.velocity_embedding_stream(adata_z, vkey="dynamics", color=color,save=dir_path+"/dzdt_streamline_" + color + "_tanh.png", title="", dpi=300,
                                         legend_loc="none")
        scv.pl.velocity_embedding_stream(adata_z, vkey="dynamics", color=color,save=dir_path+"/dzdt_streamline_" + color + "_tanh_with_legend.png", title="", dpi=300,
                                         legend_loc="right margin")
    
    if spliced:
        velocity_graph(adata_r, vkey="dsdt", xkey="s_raw", n_jobs=16)
        scv.tl.velocity_embedding(adata_r, basis="umap", vkey="dsdt")
        scv.pl.velocity_embedding_grid(adata_r, vkey="dsdt", color=color, save=dir_path+"/dsdt_grid_" + color +"_tanh.png", dpi=300)
        scv.pl.velocity_embedding_stream(adata_r, vkey="dsdt", color=color, save=dir_path+"/dsdt_streamline_" + color + "_tanh.png",
                                         title="", dpi=300, legend_loc="none")
    

    if dadt:
        adata_a.obsp["distances"] = adata_r.obsp["distances"]
        adata_a.obsp["connectivities"] = adata_r.obsp["connectivities"]
        adata_a.uns["neighbors"] = adata_r.uns["neighbors"]
        adata_a.obsm["X_umap"] = adata_r.obsm["X_umap"]
        velocity_graph(adata_a, vkey="dadt", xkey="a_raw", n_jobs=16)
        scv.tl.velocity_embedding(adata_a, basis="umap", vkey="dadt")
        adata_a.obs[color] = adata_r.obs[color]
        adata_a.uns[color + "_colors"] = adata_r.uns[color + "_colors"]
        scv.pl.velocity_embedding_grid(adata_a, vkey="dadt", color=color, save=dir_path+"/dadt_grid_" + color +"_tanh.png", title="", dpi=300)
        scv.pl.velocity_embedding_stream(adata_a, vkey="dadt", color=color, save=dir_path+"/dadt_streamline_" + color + "_tanh.png", title="",
                                         legend_loc="none", dpi=300)

plot_vec_embed_tanh(adata_r, adata_a, dir_path, latent=True, color="clusters")
plot_vec_embed_tanh(adata_r, adata_a, dir_path, spliced=True, color="clusters")
plot_vec_embed_tanh(adata_r, adata_a, dir_path, dadt=True, color="clusters")

