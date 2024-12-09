import os
import json
import numpy as np
import sys

import scanpy as sc
import scvelo as scv
import anndata as ad

import torch
from torch.nn.parameter import Parameter

import umap
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append("./src")
from streamlineplot import velocity_graph


# for umap
def embed_z(dm, n_neighbors=30, min_dist=0.2, densmap=False):
    z_mat = dm.adata_r.obsm["latent"]
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, densmap=densmap)
    z_embed = reducer.fit_transform(z_mat)
    dm.adata_r.obsm["X_umap"] = z_embed
    return z_embed


def plot_umap(dm, exp_name, embedding=False, n_neighbors=30, min_dist=0.2, cluster_name="clusters", title=None):
    if embedding:
        dm.adata_r.obsm["X_umap"] = embed_z(dm, n_neighbors=n_neighbors, min_dist=min_dist)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    sc.pl.umap(dm.adata_r, return_fig=True, color=cluster_name)
    if title is None:
        plt.savefig(exp_name+"/umap_" + cluster_name + ".png", bbox_inches='tight')
    else:
        plt.savefig(exp_name+"/"+title, bbox_inches='tight')
    plt.close(fig)


def plot_vec_embed(dm, exp_name, latent=False, spliced=False, dsdt_obs=False, ss_model=False, dadt=False, color=None):
    if latent:
        dm.adata_z = ad.AnnData(X=dm.adata_r.obsm["latent"])
        dm.adata_z.obs_names = dm.adata_r.obs_names
        dm.adata_z.obsm["latent"] = dm.adata_r.obsm["latent"]
        dm.adata_z.layers["latent"] = dm.adata_r.obsm["latent"]
        dm.adata_z.layers["dynamics"] = dm.adata_r.obsm["dynamics"]    
        dm.adata_z.obsm["X_umap"] = dm.adata_r.obsm["X_umap"]
        dm.adata_z.obsp["distances"] = dm.adata_r.obsp["distances"]
        dm.adata_z.obsp["connectivities"] = dm.adata_r.obsp["connectivities"]
        dm.adata_z.uns["neighbors"] = dm.adata_r.uns["neighbors"]
        scv.tl.velocity_graph(dm.adata_z, vkey="dynamics", xkey="latent", n_jobs=16)
        scv.tl.velocity_embedding(dm.adata_z, basis="umap", vkey="dynamics")
        if color is None:
            scv.pl.velocity_embedding_grid(dm.adata_z, vkey="dynamics", save=exp_name+"/dzdt_grid.png", title="dzdt", dpi=300)
            scv.pl.velocity_embedding_stream(dm.adata_z, vkey="dynamics", save=exp_name+"/dzdt_streamline.png", title="dzdt", min_mass=0)
        else:
            dm.adata_z.obs[color] = dm.adata_r.obs[color]
            dm.adata_z.uns[color+"_colors"] = dm.adata_r.uns[color+"_colors"]
            scv.pl.velocity_embedding_grid(dm.adata_z, vkey="dynamics", color=color, save=exp_name+"/dzdt_grid_" + color +".png", title="dzdt", dpi=300)
            scv.pl.velocity_embedding_stream(dm.adata_z, vkey="dynamics", color=color,save=exp_name+"/dzdt_streamline_" + color + ".png", title="dzdt", min_mass=0)
    
    if spliced:
        scv.tl.velocity_graph(dm.adata_r, vkey="dsdt", xkey="s_raw", n_jobs=16)
        scv.tl.velocity_embedding(dm.adata_r, basis="umap", vkey="dsdt")
        if color is None:
            scv.pl.velocity_embedding_grid(dm.adata_r, vkey="dsdt",  save=exp_name+"/dsdt_grid.png", title="dsdt", dpi=300)
            scv.pl.velocity_embedding_stream(dm.adata_r, vkey="dsdt", save=exp_name+"/dsdt_streamline.png", title="dsdt", min_mass=0)
        else:
            scv.pl.velocity_embedding_grid(dm.adata_r, vkey="dsdt", color=color, save=exp_name+"/dsdt_grid_" + color +".png", title="dsdt", dpi=300)
            scv.pl.velocity_embedding_stream(dm.adata_r, vkey="dsdt", color=color, save=exp_name+"/dsdt_streamline_" + color + ".png", title="dsdt", min_mass=0)
    if dsdt_obs:
        
        scv.tl.velocity_graph(dm.adata_r, vkey="dsdt_obs", xkey="s_raw", n_jobs=16)
        scv.tl.velocity_embedding(dm.adata_r, basis="umap", vkey="dsdt_obs")
        scv.pl.velocity_embedding_grid(dm.adata_r, vkey="dsdt_obs", color=color, save=exp_name+"/dsdt_obs_grid_" + color +".png", title="dsdt_obs", dpi=300)
        scv.pl.velocity_embedding_stream(dm.adata_r, vkey="dsdt_obs", color=color, save=exp_name+"/dsdt_obs_streamline_" + color + ".png", title="dsdt_obs", min_mass=0)

    if ss_model:
        scv.tl.velocity(dm.adata_r, vkey="velocity_ss", mode="deterministic", )
        scv.tl.velocity_graph(dm.adata_r, vkey="velocity_ss", xkey="Ms", n_jobs=16)
        scv.tl.velocity_embedding(dm.adata_r, basis="umap", vkey="velocity_ss")
        scv.pl.velocity_embedding_grid(dm.adata_r, vkey="velocity_ss", color=color, save=exp_name+"/dsdt_ss_grid_" + color +".png", title="dsdt_ss", dpi=300)
        scv.pl.velocity_embedding_stream(dm.adata_r, vkey="velocity_ss", color=color, save=exp_name+"/dsdt_ss_streamline_" + color + ".png", title="dsdt_ss", min_mass=0)

    if dadt:
        dm.adata_a.obsp["distances"] = dm.adata_r.obsp["distances"]
        dm.adata_a.obsp["connectivities"] = dm.adata_r.obsp["connectivities"]
        dm.adata_a.uns["neighbors"] = dm.adata_r.uns["neighbors"]
        dm.adata_a.obsm["X_umap"] = dm.adata_r.obsm["X_umap"]
        scv.tl.velocity_graph(dm.adata_a, vkey="dadt", xkey="a_raw", n_jobs=16)
        scv.tl.velocity_embedding(dm.adata_a, basis="umap", vkey="dadt")
        if color is None:
            scv.pl.velocity_embedding_grid(dm.adata_a, vkey="dadt",  save=exp_name+"/dadt_grid.png", title="dadt", dpi=300)
            scv.pl.velocity_embedding_stream(dm.adata_a, vkey="dadt", save=exp_name+"/dadt_streamline.png", title="dadt", min_mass=0)
        else:
            dm.adata_a.obs[color] = dm.adata_r.obs[color]
            dm.adata_a.uns[color + "_colors"] = dm.adata_r.uns[color + "_colors"]
            scv.pl.velocity_embedding_grid(dm.adata_a, vkey="dadt", color=color, save=exp_name+"/dadt_grid_" + color +".png", title="dadt", dpi=300)
            scv.pl.velocity_embedding_stream(dm.adata_a, vkey="dadt", color=color, save=exp_name+"/dadt_streamline_" + color + ".png", title="dadt", min_mass=0)

def plot_filtered_vec_embed(dm, exp_name, filter_idx, spliced=False, dsdt_obs=False, ss_model=False, color=None):
    filter_idx = filter_idx.numpy()
    adata_r = dm.adata_r[:, filter_idx == 1]
    
    if spliced:
        velocity_graph(adata_r, vkey="dsdt", xkey="s_raw", n_jobs=16)
        scv.tl.velocity_embedding(adata_r, basis="umap", vkey="dsdt")
        if color is None:
            scv.pl.velocity_embedding_grid(adata_r, vkey="dsdt",  save=exp_name+"/dsdt_grid_filtered_.png", title="dsdt", dpi=300)
            scv.pl.velocity_embedding_stream(adata_r, vkey="dsdt", save=exp_name+"/dsdt_streamline_filtered_.png", title="dsdt")
        else:
            scv.pl.velocity_embedding_grid(adata_r, vkey="dsdt", color=color, save=exp_name+"/dsdt_grid_filtered_" + color +".png", title="dsdt", dpi=300)
            scv.pl.velocity_embedding_stream(adata_r, vkey="dsdt", color=color, save=exp_name+"/dsdt_streamline_filtered_" + color + ".png", title="dsdt")
    if dsdt_obs:
        velocity_graph(adata_r, vkey="dsdt_obs", xkey="s_raw", n_jobs=16)
        scv.tl.velocity_embedding(adata_r, basis="umap", vkey="dsdt_obs")
        scv.pl.velocity_embedding_grid(adata_r, vkey="dsdt_obs", color=color, save=exp_name+"/dsdt_obs_grid_filtered_" + color +".png", title="dsdt_obs", dpi=300)
        scv.pl.velocity_embedding_stream(adata_r, vkey="dsdt_obs", color=color, save=exp_name+"/dsdt_obs_streamline_filtered_" + color + ".png", title="dsdt_obs")

    if ss_model:
        scv.tl.velocity(adata_r, vkey="velocity_ss", mode="deterministic", )
        velocity_graph(adata_r, vkey="velocity_ss", xkey="Ms", n_jobs=16)
        scv.tl.velocity_embedding(adata_r, basis="umap", vkey="velocity_ss")
        scv.pl.velocity_embedding_grid(adata_r, vkey="velocity_ss", color=color, save=exp_name+"/dsdt_ss_grid_filtered_" + color +".png", title="dsdt_ss", dpi=300)
        scv.pl.velocity_embedding_stream(adata_r, vkey="velocity_ss", color=color, save=exp_name+"/dsdt_ss_streamline_filtered_" + color + ".png", title="dsdt_ss")

def plot_vec_embed_tanh(dm, exp_name, latent=False, spliced=False, dsdt_obs=False, ss_model=False, dadt=False, color=None):
    if latent:
        dm.adata_z = ad.AnnData(X=dm.adata_r.obsm["latent"])
        dm.adata_z.obs_names = dm.adata_r.obs_names
        dm.adata_z.obsm["latent"] = dm.adata_r.obsm["latent"]
        dm.adata_z.layers["latent"] = dm.adata_r.obsm["latent"]
        dm.adata_z.layers["dynamics"] = dm.adata_r.obsm["dynamics"]    
        dm.adata_z.obsm["X_umap"] = dm.adata_r.obsm["X_umap"]
        dm.adata_z.obsp["distances"] = dm.adata_r.obsp["distances"]
        dm.adata_z.obsp["connectivities"] = dm.adata_r.obsp["connectivities"]
        dm.adata_z.uns["neighbors"] = dm.adata_r.uns["neighbors"]
        velocity_graph(dm.adata_z, vkey="dynamics", xkey="latent", n_jobs=16)
        scv.tl.velocity_embedding(dm.adata_z, basis="umap", vkey="dynamics")
        if color is None:
            scv.pl.velocity_embedding_grid(dm.adata_z, vkey="dynamics", save=exp_name+"/dzdt_grid_tanh.png", title="dzdt", dpi=300)
            scv.pl.velocity_embedding_stream(dm.adata_z, vkey="dynamics", save=exp_name+"/dzdt_streamline_tanh.png", title="dzdt")
        else:
            dm.adata_z.obs[color] = dm.adata_r.obs[color]
            dm.adata_z.uns[color+"_colors"] = dm.adata_r.uns[color+"_colors"]
            scv.pl.velocity_embedding_grid(dm.adata_z, vkey="dynamics", color=color, save=exp_name+"/dzdt_grid_" + color +"_tanh.png", title="dzdt", dpi=300)
            scv.pl.velocity_embedding_stream(dm.adata_z, vkey="dynamics", color=color,save=exp_name+"/dzdt_streamline_" + color + "_tanh.png", title="dzdt")
    
    if spliced:
        velocity_graph(dm.adata_r, vkey="dsdt", xkey="s_raw", n_jobs=16)
        scv.tl.velocity_embedding(dm.adata_r, basis="umap", vkey="dsdt")
        if color is None:
            scv.pl.velocity_embedding_grid(dm.adata_r, vkey="dsdt",  save=exp_name+"/dsdt_grid_tanh.png", title="dsdt", dpi=300)
            scv.pl.velocity_embedding_stream(dm.adata_r, vkey="dsdt", save=exp_name+"/dsdt_streamline_tanh.png", title="dsdt")
        else:
            scv.pl.velocity_embedding_grid(dm.adata_r, vkey="dsdt", color=color, save=exp_name+"/dsdt_grid_" + color +"_tanh.png", title="dsdt", dpi=300)
            scv.pl.velocity_embedding_stream(dm.adata_r, vkey="dsdt", color=color, save=exp_name+"/dsdt_streamline_" + color + "_tanh.png", title="dsdt")
    if dsdt_obs:
        velocity_graph(dm.adata_r, vkey="dsdt_obs", xkey="s_raw", n_jobs=16)
        scv.tl.velocity_embedding(dm.adata_r, basis="umap", vkey="dsdt_obs")
        scv.pl.velocity_embedding_grid(dm.adata_r, vkey="dsdt_obs", color=color, save=exp_name+"/dsdt_obs_grid_" + color +"_tanh.png", title="dsdt_obs", dpi=300)
        scv.pl.velocity_embedding_stream(dm.adata_r, vkey="dsdt_obs", color=color, save=exp_name+"/dsdt_obs_streamline_" + color + "_tanh.png", title="dsdt_obs")

    if ss_model:
        scv.tl.velocity(dm.adata_r, vkey="velocity_ss", mode="deterministic", )
        velocity_graph(dm.adata_r, vkey="velocity_ss", xkey="Ms", n_jobs=16)
        scv.tl.velocity_embedding(dm.adata_r, basis="umap", vkey="velocity_ss")
        scv.pl.velocity_embedding_grid(dm.adata_r, vkey="velocity_ss", color=color, save=exp_name+"/dsdt_ss_grid_" + color +"_tanh.png", title="dsdt_ss", dpi=300)
        scv.pl.velocity_embedding_stream(dm.adata_r, vkey="velocity_ss", color=color, save=exp_name+"/dsdt_ss_streamline_" + color + "_tanh.png", title="dsdt_ss")

    if dadt:
        dm.adata_a.obsp["distances"] = dm.adata_r.obsp["distances"]
        dm.adata_a.obsp["connectivities"] = dm.adata_r.obsp["connectivities"]
        dm.adata_a.uns["neighbors"] = dm.adata_r.uns["neighbors"]
        dm.adata_a.obsm["X_umap"] = dm.adata_r.obsm["X_umap"]
        velocity_graph(dm.adata_a, vkey="dadt", xkey="a_raw", n_jobs=16)
        scv.tl.velocity_embedding(dm.adata_a, basis="umap", vkey="dadt")
        if color is None:
            scv.pl.velocity_embedding_grid(dm.adata_a, vkey="dadt",  save=exp_name+"/dadt_grid_tanh.png", title="dadt", dpi=300)
            scv.pl.velocity_embedding_stream(dm.adata_a, vkey="dadt", save=exp_name+"/dadt_streamline_tanh.png", title="dadt")
        else:
            dm.adata_a.obs[color] = dm.adata_r.obs[color]
            dm.adata_a.uns[color + "_colors"] = dm.adata_r.uns[color + "_colors"]
            scv.pl.velocity_embedding_grid(dm.adata_a, vkey="dadt", color=color, save=exp_name+"/dadt_grid_" + color +"_tanh.png", title="dadt", dpi=300)
            scv.pl.velocity_embedding_stream(dm.adata_a, vkey="dadt", color=color, save=exp_name+"/dadt_streamline_" + color + "_tanh.png", title="dadt")

# calculate gene-wise correlation
def colwise_pearsonr(r_c, r_ld):
    val = np.array([scipy.stats.pearsonr(r_c[:, i], r_ld[:, i])[0] for i in range(r_c.shape[1])])
    return(val)

def plot_genewise_corr(dm, exp_name, spliced=False, unspliced=False, ms=False, mu=False):
    if spliced:
        genewise_log_count = np.log10(np.sum(dm.adata_r.layers["spliced"].toarray(), axis=0))
        train_idx = dm.idx["train"]
        test_idx = dm.idx["test"]
        train_corr = colwise_pearsonr(dm.adata_r.layers["spliced"].toarray()[train_idx, :],
                                     dm.adata_r.layers["rec_s"][train_idx, :])
        test_corr = colwise_pearsonr(dm.adata_r.layers["spliced"].toarray()[test_idx, :],
                                     dm.adata_r.layers["rec_s"][test_idx, :])
    
        fig, ax = plt.subplots(1, 1, figsize=(5, 5 * 1))
        ax.scatter(genewise_log_count, train_corr, color="blue", s=1, label="train")
        ax.scatter(genewise_log_count, test_corr, color="red", s=1, label="test")
        plt.xlabel("log10 total count")
        plt.ylabel("pearson corr")
        ax.legend()
        plt.title("gene-wise correlation between observed and predicted, spliced")
        plt.savefig(exp_name+"/train_test_corr_s.png");plt.close("all")

    if unspliced:
        genewise_log_count = np.log10(np.sum(dm.adata_r.layers["unspliced"].toarray(), axis=0))
        train_idx = dm.idx["train"]
        test_idx = dm.idx["test"]
        train_corr = colwise_pearsonr(dm.adata_r.layers["unspliced"].toarray()[train_idx, :],
                                     dm.adata_r.layers["rec_u"][train_idx, :])
        test_corr = colwise_pearsonr(dm.adata_r.layers["unspliced"].toarray()[test_idx, :],
                                     dm.adata_r.layers["rec_u"][test_idx, :])
    
        fig, ax = plt.subplots(1, 1, figsize=(5, 5 * 1))
        ax.scatter(genewise_log_count, train_corr, color="blue", s=1, label="train")
        ax.scatter(genewise_log_count, test_corr, color="red", s=1, label="test")
        plt.xlabel("log10 total count")
        plt.ylabel("pearson corr")
        ax.legend()
        plt.title("gene-wise correlation between observed and predicted, unspliced")
        plt.savefig(exp_name+"/train_test_corr_u.png");plt.close("all")
    
    if ms:
        genewise_log_count = np.log10(np.sum(dm.adata_r.layers["spliced_count"].toarray(), axis=0))
        train_idx = dm.idx["train"]
        test_idx = dm.idx["test"]
        train_corr = colwise_pearsonr(dm.adata_r.layers["Ms"][train_idx, :],
                                     dm.adata_r.layers["s_raw"][train_idx, :])
        test_corr = colwise_pearsonr(dm.adata_r.layers["Ms"][test_idx, :],
                                     dm.adata_r.layers["s_raw"][test_idx, :])
    
        fig, ax = plt.subplots(1, 1, figsize=(5, 5 * 1))
        ax.scatter(genewise_log_count, train_corr, color="blue", s=1, label="train")
        ax.scatter(genewise_log_count, test_corr, color="red", s=1, label="test")
        plt.xlabel("log10 total count")
        plt.ylabel("pearson corr")
        ax.legend()
        plt.title("gene-wise correlation between observed and predicted, Ms")
        plt.savefig(exp_name+"/train_test_corr_Ms.png");plt.close("all")

    if mu:
        genewise_log_count = np.log10(np.sum(dm.adata_r.layers["unspliced_count"].toarray(), axis=0))
        train_idx = dm.idx["train"]
        test_idx = dm.idx["test"]
        train_corr = colwise_pearsonr(dm.adata_r.layers["Mu"][train_idx, :],
                                     dm.adata_r.layers["u_raw"][train_idx, :])
        test_corr = colwise_pearsonr(dm.adata_r.layers["Mu"][test_idx, :],
                                     dm.adata_r.layers["u_raw"][test_idx, :])
    
        fig, ax = plt.subplots(1, 1, figsize=(5, 5 * 1))
        ax.scatter(genewise_log_count, train_corr, color="blue", s=1, label="train")
        ax.scatter(genewise_log_count, test_corr, color="red", s=1, label="test")
        plt.xlabel("log10 total count")
        plt.ylabel("pearson corr")
        ax.legend()
        plt.title("gene-wise correlation between observed and predicted, Mu")
        plt.savefig(exp_name+"/train_test_corr_Mu.png");plt.close("all")
        
def plot_genewise_corr_for_fig(dm, exp_name, spliced=False, unspliced=False, ms=False, mu=False):
    if spliced:
        genewise_log_count = np.log10(np.sum(dm.adata_r.layers["spliced"].toarray(), axis=0))
        train_idx = dm.idx["train"]
        test_idx = dm.idx["test"]
        train_corr = colwise_pearsonr(dm.adata_r.layers["spliced"].toarray()[train_idx, :],
                                     dm.adata_r.layers["rec_s"][train_idx, :])
        test_corr = colwise_pearsonr(dm.adata_r.layers["spliced"].toarray()[test_idx, :],
                                     dm.adata_r.layers["rec_s"][test_idx, :])
    
        fig, ax = plt.subplots(1, 1, figsize=(5, 5 * 1), dpi=300)
        ax.scatter(genewise_log_count, train_corr, color="blue", s=1, label="train")
        ax.scatter(genewise_log_count, test_corr, color="red", s=1, label="test")
        plt.xlabel("log10 total count")
        plt.ylabel("pearson correlation")
        ax.legend()
        plt.title("spliced")
        plt.savefig(exp_name+"/train_test_corr_s_for_fig.png", dpi=300);plt.close("all")

    if unspliced:
        genewise_log_count = np.log10(np.sum(dm.adata_r.layers["unspliced"].toarray(), axis=0))
        train_idx = dm.idx["train"]
        test_idx = dm.idx["test"]
        train_corr = colwise_pearsonr(dm.adata_r.layers["unspliced"].toarray()[train_idx, :],
                                     dm.adata_r.layers["rec_u"][train_idx, :])
        test_corr = colwise_pearsonr(dm.adata_r.layers["unspliced"].toarray()[test_idx, :],
                                     dm.adata_r.layers["rec_u"][test_idx, :])
    
        fig, ax = plt.subplots(1, 1, figsize=(5, 5 * 1), dpi=300)
        ax.scatter(genewise_log_count, train_corr, color="blue", s=1, label="train")
        ax.scatter(genewise_log_count, test_corr, color="red", s=1, label="test")
        plt.xlabel("log10 total count")
        plt.ylabel("pearson correlation")
        ax.legend()
        plt.title("unspliced")
        plt.savefig(exp_name+"/train_test_corr_u_for_fig.png", dpi=300);plt.close("all")
    
    if ms:
        genewise_log_count = np.log10(np.sum(dm.adata_r.layers["spliced_count"].toarray(), axis=0))
        train_idx = dm.idx["train"]
        test_idx = dm.idx["test"]
        train_corr = colwise_pearsonr(dm.adata_r.layers["Ms"][train_idx, :],
                                     dm.adata_r.layers["s_raw"][train_idx, :])
        test_corr = colwise_pearsonr(dm.adata_r.layers["Ms"][test_idx, :],
                                     dm.adata_r.layers["s_raw"][test_idx, :])
    
        fig, ax = plt.subplots(1, 1, figsize=(5, 5 * 1), dpi=300)
        ax.scatter(genewise_log_count, train_corr, color="blue", s=1, label="train")
        ax.scatter(genewise_log_count, test_corr, color="red", s=1, label="test")
        plt.xlabel("log10 total count")
        plt.ylabel("pearson correlation")
        ax.legend()
        plt.title("Ms")
        plt.savefig(exp_name+"/train_test_corr_Ms_for_fig.png", dpi=300);plt.close("all")

    if mu:
        genewise_log_count = np.log10(np.sum(dm.adata_r.layers["unspliced_count"].toarray(), axis=0))
        train_idx = dm.idx["train"]
        test_idx = dm.idx["test"]
        train_corr = colwise_pearsonr(dm.adata_r.layers["Mu"][train_idx, :],
                                     dm.adata_r.layers["u_raw"][train_idx, :])
        test_corr = colwise_pearsonr(dm.adata_r.layers["Mu"][test_idx, :],
                                     dm.adata_r.layers["u_raw"][test_idx, :])
    
        fig, ax = plt.subplots(1, 1, figsize=(5, 5 * 1), dpi=300)
        ax.scatter(genewise_log_count, train_corr, color="blue", s=1, label="train")
        ax.scatter(genewise_log_count, test_corr, color="red", s=1, label="test")
        plt.xlabel("log10 total count")
        plt.ylabel("pearson correlation")
        ax.legend()
        plt.title("Mu")
        plt.savefig(exp_name+"/train_test_corr_Mu_for_fig.png", dpi=300);plt.close("all")

def colwise_cos_sim(vel_1, vel_2):
    cos_sim = np.array([1 - scipy.spatial.distance.cosine(vel_1[:,i], vel_2[:,i]) for i in range(vel_1.shape[1])])
    return cos_sim

def plot_genewise_vel_cossim(dm, exp_name, v1_key="dsdt", v2_key="dsdt_obs", filter_idx=None):
    adata_r = dm.adata_r
    if filter_idx is not None:
        filter_idx = filter_idx.numpy()
        adata_r = dm.adata_r[:, filter_idx==1]
    genewise_log_count = np.log10(np.sum(adata_r.layers["spliced_count"].toarray(), axis=0))
    train_idx = dm.idx["train"]
    test_idx = dm.idx["test"]
    train_corr = colwise_cos_sim(adata_r.layers[v1_key][train_idx, :],
                                    adata_r.layers[v2_key][train_idx, :])
    test_corr = colwise_cos_sim(adata_r.layers[v1_key][test_idx, :],
                                    adata_r.layers[v2_key][test_idx, :])
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5 * 1))
    ax.scatter(genewise_log_count, train_corr, color="blue", s=1, label="train")
    ax.scatter(genewise_log_count, test_corr, color="red", s=1, label="test")
    plt.xlabel("log10 total spliced count")
    plt.ylabel("cosine similarity")
    ax.legend()
    plt.title("gene-wise velocity cosine similarity \n between {} and {}, spliced".format(v1_key, v2_key))
    if filter_idx is not None:
        plt.savefig(exp_name+"/train_test_cossim_{}_vs_{}_filtered.png".format(v1_key, v2_key));plt.close("all")
    else:
        plt.savefig(exp_name+"/train_test_cossim_{}_vs_{}.png".format(v1_key, v2_key));plt.close("all")

# gene-wise cos sim




def plot_peakwise_corr(dm, exp_name, a=False, ma=False):
    if a:
        varwise_log_count = np.log10(np.sum(dm.adata_a.X.toarray(), axis=0))
        train_idx = dm.idx["train"]
        test_idx = dm.idx["test"]
        train_corr = colwise_pearsonr(dm.adata_a.X.toarray()[train_idx, :],
                                        dm.adata_a.layers["rec_a"][train_idx, :])
        test_corr = colwise_pearsonr(dm.adata_a.X.toarray()[test_idx, :],
                                        dm.adata_a.layers["rec_a"][test_idx, :])
        
        fig, ax = plt.subplots(1, 1, figsize=(5, 5 * 1))
        ax.scatter(varwise_log_count, train_corr, color="blue", s=0.3, label="train")
        ax.scatter(varwise_log_count, test_corr, color="red", s=0.3, label="test")
        plt.xlabel("log10 total count")
        plt.ylabel("pearson corr")
        ax.legend()
        plt.title("peak-wise correlation between observed and predicted, atac")
        plt.savefig(exp_name+"/train_test_corr_a.png");plt.close("all")

    if ma:
        varwise_log_count = np.log10(np.sum(dm.adata_a.layers["atac_count"].toarray(), axis=0))
        train_idx = dm.idx["train"]
        test_idx = dm.idx["test"]
        train_corr = colwise_pearsonr(dm.adata_a.layers["Ma"][train_idx, :],
                                        dm.adata_a.layers["a_raw"][train_idx, :])
        test_corr = colwise_pearsonr(dm.adata_a.layers["Ma"][test_idx, :],
                                        dm.adata_a.layers["a_raw"][test_idx, :])
        
        fig, ax = plt.subplots(1, 1, figsize=(5, 5 * 1))
        ax.scatter(varwise_log_count, train_corr, color="blue", s=0.3, label="train")
        ax.scatter(varwise_log_count, test_corr, color="red", s=0.3, label="test")
        plt.xlabel("log10 total count")
        plt.ylabel("pearson corr")
        ax.legend()
        plt.title("peak-wise correlation between observed and predicted, Ma")
        plt.savefig(exp_name+"/train_test_corr_Ma.png");plt.close("all")
        
def plot_peakwise_corr_for_fig(dm, exp_name, a=False, ma=False):
    if a:
        varwise_log_count = np.log10(np.sum(dm.adata_a.X.toarray(), axis=0))
        train_idx = dm.idx["train"]
        test_idx = dm.idx["test"]
        train_corr = colwise_pearsonr(dm.adata_a.X.toarray()[train_idx, :],
                                        dm.adata_a.layers["rec_a"][train_idx, :])
        test_corr = colwise_pearsonr(dm.adata_a.X.toarray()[test_idx, :],
                                        dm.adata_a.layers["rec_a"][test_idx, :])
        
        fig, ax = plt.subplots(1, 1, figsize=(5, 5 * 1), dpi=300)
        ax.scatter(varwise_log_count, train_corr, color="blue", s=0.03, label="train")
        ax.scatter(varwise_log_count, test_corr, color="red", s=0.03, label="test")
        plt.xlabel("log10 total count")
        plt.ylabel("pearson correlation")
        ax.legend()
        plt.title("atac")
        plt.savefig(exp_name+"/train_test_corr_a_for_fig.png", dpi=300);plt.close("all")

    if ma:
        varwise_log_count = np.log10(np.sum(dm.adata_a.layers["atac_count"].toarray(), axis=0))
        train_idx = dm.idx["train"]
        test_idx = dm.idx["test"]
        train_corr = colwise_pearsonr(dm.adata_a.layers["Ma"][train_idx, :],
                                        dm.adata_a.layers["a_raw"][train_idx, :])
        test_corr = colwise_pearsonr(dm.adata_a.layers["Ma"][test_idx, :],
                                        dm.adata_a.layers["a_raw"][test_idx, :])
        
        fig, ax = plt.subplots(1, 1, figsize=(5, 5 * 1), dpi=300)
        ax.scatter(varwise_log_count, train_corr, color="blue", s=0.03, label="train")
        ax.scatter(varwise_log_count, test_corr, color="red", s=0.03, label="test")
        plt.xlabel("log10 total count")
        plt.ylabel("pearson correlation")
        ax.legend()
        plt.title("Ma")
        plt.savefig(exp_name+"/train_test_corr_Ma_for_fig.png", dpi=300);plt.close("all")

def plot_protwise_corr(dm, exp_name):
    varwise_log_count = np.log10(np.sum(dm.adata_p.X.toarray(), axis=0))
    train_idx = dm.idx["train"]
    test_idx = dm.idx["test"]
    train_corr = colwise_pearsonr(dm.adata_p.X.toarray()[train_idx, :],
                                    dm.adata_p.layers["rec_a"][train_idx, :])
    test_corr = colwise_pearsonr(dm.adata_p.X.toarray()[test_idx, :],
                                    dm.adata_p.layers["rec_a"][test_idx, :])
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5 * 1))
    ax.scatter(varwise_log_count, train_corr, color="blue", s=1, label="train")
    ax.scatter(varwise_log_count, test_corr, color="red", s=1, label="test")
    plt.xlabel("log10 total count")
    plt.ylabel("pearson corr")
    ax.legend()
    plt.title("protein-wise correlation between observed and predicted, protein")
    plt.savefig(exp_name+"/train_test_corr_p.png");plt.close("all")

def plot_size_factor(dm, exp_name, rna=True, atac=False, protein=False):
    train_idx = dm.idx["train"]
    test_idx = dm.idx["test"]
    if rna:
        obs_count = np.log10(np.sum(dm.adata_r.layers["spliced"].toarray() + dm.adata_r.layers["unspliced"].toarray(), axis=1))
        size_factor = np.log10(dm.adata_r.obsm["lr"])
        fig, ax = plt.subplots(1, 1, figsize=(5, 6 * 1))
        ax.scatter(obs_count[train_idx], size_factor[train_idx], color="blue", s=1, label="train")
        ax.scatter(obs_count[test_idx], size_factor[test_idx], color="red", s=1, label="test")
        plt.xlabel("log10 observed count")
        plt.ylabel("log10 size factor, mean")
        ax.legend()
        plt.title("RNA size factor inference")
        plt.savefig(exp_name+"/size_factor_rna.png");plt.close("all")
    if atac:
        obs_count = np.log10(np.sum(dm.adata_a.X.toarray(), axis=1))
        size_factor = np.log10(dm.adata_a.obsm["la"])
        fig, ax = plt.subplots(1, 1, figsize=(5, 6 * 1))
        ax.scatter(obs_count[train_idx], size_factor[train_idx], color="blue", s=1, label="train")
        ax.scatter(obs_count[test_idx], size_factor[test_idx], color="red", s=1, label="test")
        plt.xlabel("log10 observed count")
        plt.ylabel("log10 size factor, mean")
        ax.legend()
        plt.title("ATAC size factor inference")
        plt.savefig(exp_name+"/size_factor_atac.png");plt.close("all")
    if protein:
        obs_count = np.log10(np.sum(dm.adata_p.X.toarray(), axis=1))
        size_factor = np.log10(dm.adata_p.obsm["la"])
        fig, ax = plt.subplots(1, 1, figsize=(5, 6 * 1))
        ax.scatter(obs_count[train_idx], size_factor[train_idx], color="blue", s=1, label="train")
        ax.scatter(obs_count[test_idx], size_factor[test_idx], color="red", s=1, label="test")
        plt.xlabel("log10 observed count")
        plt.ylabel("log10 size factor, mean")
        ax.legend()
        plt.title("PROTEIN size factor inference")
        plt.savefig(exp_name+"/size_factor_protein.png");plt.close("all")

def plot_size_factor_for_fig(dm, exp_name, rna=False, atac=False, protein=False):
    train_idx = dm.idx["train"]
    test_idx = dm.idx["test"]
    if rna:
        obs_count = np.log10(np.sum(dm.adata_r.layers["spliced"].toarray() + dm.adata_r.layers["unspliced"].toarray(), axis=1))
        size_factor = np.log10(dm.adata_r.obsm["lr"])
        fig, ax = plt.subplots(1, 1, figsize=(5, 5 * 1), dpi=300)
        ax.scatter(obs_count[train_idx], size_factor[train_idx], color="blue", s=1, label="train")
        ax.scatter(obs_count[test_idx], size_factor[test_idx], color="red", s=1, label="test")
        plt.xlabel("log10 observed count")
        plt.ylabel("log10 size factor, mean")
        ax.legend()
        plt.title("RNA size factor")
        plt.savefig(exp_name+"/size_factor_rna_for_fig.png");plt.close("all")
    if atac:
        obs_count = np.log10(np.sum(dm.adata_a.X.toarray(), axis=1))
        size_factor = np.log10(dm.adata_a.obsm["la"])
        fig, ax = plt.subplots(1, 1, figsize=(5, 5 * 1), dpi=300)
        ax.scatter(obs_count[train_idx], size_factor[train_idx], color="blue", s=1, label="train")
        ax.scatter(obs_count[test_idx], size_factor[test_idx], color="red", s=1, label="test")
        plt.xlabel("log10 observed count")
        plt.ylabel("log10 size factor, mean")
        ax.legend()
        plt.title("ATAC size factor")
        plt.savefig(exp_name+"/size_factor_atac_for_fig.png");plt.close("all")
    if protein:
        obs_count = np.log10(np.sum(dm.adata_p.X.toarray(), axis=1))
        size_factor = np.log10(dm.adata_p.obsm["la"])
        fig, ax = plt.subplots(1, 1, figsize=(5, 5 * 1), dpi=300)
        ax.scatter(obs_count[train_idx], size_factor[train_idx], color="blue", s=1, label="train")
        ax.scatter(obs_count[test_idx], size_factor[test_idx], color="red", s=1, label="test")
        plt.xlabel("log10 observed count")
        plt.ylabel("log10 size factor, mean")
        ax.legend()
        plt.title("PROTEIN size factor")
        plt.savefig(exp_name+"/size_factor_protein_for_fig.png");plt.close("all")

"""
def fit_beta_gamma(dm):
    adata_ss = ad.AnnData(X=dm.adata_r.X)
    #adata_ss.layers["spliced"] = dm.adata_r.layers["s_raw"]
    #adata_ss.layers["unspliced"] = dm.adata_r.layers["u_raw"]
    
    adata_ss.layers["spliced"] = dm.adata_r.layers["Ms"]
    adata_ss.layers["unspliced"] = dm.adata_r.layers["Mu"]
    scv.tl.velocity(adata_ss, vkey="ss-model", mode="deterministic", use_raw=True, perc=[5, 95])
    log_gamma_beta = np.log(adata_ss.var["ss-model_gamma"])
    return Parameter(torch.Tensor(log_gamma_beta).to(torch.float32))
"""

def fit_beta_gamma(dm):
    adata_ss = ad.AnnData(X=dm.adata_r.X)
    adata_ss.layers["spliced"] = dm.adata_r.layers["s_raw"] 
    adata_ss.layers["unspliced"] = dm.adata_r.layers["u_raw"]
    scv.tl.velocity(adata_ss, vkey="ss-model", mode="deterministic", use_raw=True, perc=[5, 95])
    log_gamma_beta = np.log(adata_ss.var["ss-model_gamma"])
    if dm.retain_genes_idx is not None:
        for idx in np.where(dm.retain_genes_idx==1)[0]:
            log_gamma_beta[idx] = 1

    return Parameter(torch.Tensor(log_gamma_beta).to(torch.float32))

def fit_beta_gamma_scale(dm, su_scale=False):
    adata_ss = ad.AnnData(X=dm.adata_r.X)
    adata_ss.layers["spliced"] = dm.adata_r.layers["s_raw"] 
    adata_ss.layers["unspliced"] = dm.adata_r.layers["u_raw"]
    scv.tl.velocity(adata_ss, vkey="ss-model", mode="deterministic", use_raw=True, perc=[5, 95])
    gamma = adata_ss.var["ss-model_gamma"]
    if dm.retain_genes_idx is not None:
        for idx in np.where(dm.retain_genes_idx==1)[0]:
            gamma[idx] = 1
    beta = np.ones_like(gamma)

    # scaling by s/u
    if su_scale:
        expr_scale = dm.adata_r.layers["s_raw"].max(axis=0) / dm.adata_r.layers["u_raw"].max(axis=0)
        if dm.retain_genes_idx is not None:
            for idx in np.where(dm.retain_genes_idx==1)[0]:
                expr_scale[idx] = 1
    else:
        expr_scale = 1
    
    log_gamma = np.log(gamma * expr_scale)
    log_beta = np.log(beta * expr_scale)
    
    return Parameter(torch.Tensor(log_beta).to(torch.float32)), Parameter(torch.Tensor(log_gamma).to(torch.float32))



def get_filter_idx_raw(dm, test_threshold=0.1, su_threshold=0.05, su_ratio=20, min_counts_su=20):
    test_idx = dm.idx["test"]
    test_corr_s = colwise_pearsonr(dm.adata_r.layers["Ms"][test_idx, :], dm.adata_r.layers["s_raw"][test_idx, :])
    test_corr_u = colwise_pearsonr(dm.adata_r.layers["Mu"][test_idx, :], dm.adata_r.layers["u_raw"][test_idx, :])
    su_corr = colwise_pearsonr(dm.adata_r.layers["spliced_count"].toarray(), dm.adata_r.layers["unspliced_count"].toarray())
    #filter_idx = (test_corr_s < test_threshold) + (test_corr_u < test_threshold) + (su_corr < su_threshold)

    expr_scale_su = dm.adata_r.layers["spliced_count"].toarray().sum(axis=0) / dm.adata_r.layers["unspliced_count"].toarray().sum(axis=0)
    expr_scale_su[dm.adata_r.layers["unspliced_count"].toarray().sum(axis=0)==0] = 1

    min_counts_s = dm.adata_r.layers["spliced_count"].toarray().sum(axis=0)
    min_counts_u = dm.adata_r.layers["unspliced_count"].toarray().sum(axis=0)

    filter_idx = (expr_scale_su > su_ratio) + (expr_scale_su < (1/su_ratio)) + (min_counts_s < min_counts_su) + (min_counts_u < min_counts_su) + (su_corr < su_threshold)
    print("# genes excluded for estimation", np.sum(filter_idx))
    filter_idx = (filter_idx==0).astype("float32")
    print("# genes for estimation", np.sum(filter_idx))
    return torch.tensor(filter_idx)

def get_filter_idx_au(dm, test_threshold=0.1, au_threshold=0.05):
    test_idx = dm.idx["test"]
    test_corr_a = colwise_pearsonr(dm.adata_a.layers["Ma"][test_idx, :] @ dm.adata_a.varm["peak_gene_association"],
                                    dm.adata_a.layers["a_raw"][test_idx, :] @ dm.adata_a.varm["peak_gene_association"])
    test_corr_u = colwise_pearsonr(dm.adata_r.layers["Mu"][test_idx, :], dm.adata_r.layers["u_raw"][test_idx, :])
    au_corr = colwise_pearsonr(dm.adata_a.layers["atac_count"].toarray() @ dm.adata_a.varm["peak_gene_association"], 
                               dm.adata_r.layers["unspliced_count"].toarray())
    filter_idx = (test_corr_a < test_threshold) + (test_corr_u < test_threshold) + (au_corr < au_threshold) 
    print("# genes excluded for estimation", np.sum(filter_idx))
    filter_idx = (filter_idx==0).astype("float32")
    print("# genes for estimation", np.sum(filter_idx))
    return torch.tensor(filter_idx)

def plot_su_phase_raw(dm, runPath, gene = None, idx = None):
    if gene is None and idx is not None:
        gene = dm.adata_r.var_names[(idx-5):idx]
        print("plotting genes from {} to {}.".format(idx-5, idx))
    elif gene is not None and idx is None:
        gene = gene
        print("plottting designated genes.")
    else:
        raise AssertionError("Either gene or gene idx must be provided.")

    fig, axes = plt.subplots(nrows=1, ncols=len(gene), figsize=(3.0 * len(gene), 3.0))
    for j in range(len(gene)):
        s = dm.adata_r[:,gene[j]].layers["s_raw"].toarray().reshape(-1)
        u = dm.adata_r[:,gene[j]].layers["u_raw"].toarray().reshape(-1)
        dsdt = dm.adata_r[:,gene[j]].layers["dsdt"].toarray().reshape(-1)
        x = np.arange(s.min(), s.max(), (s.max()-s.min())/100)
        y = dm.adata_r[:, gene[j]].var["velocity_gamma"].to_numpy() * x
        if len(gene) == 1:
            axes.scatter(x = s, y = u, s = 2, c = dsdt, vmin = - np.percentile(abs(dsdt), 99), vmax = np.percentile(abs(dsdt), 99), cmap="coolwarm")
            axes.scatter(x = x, y = y, s = 1, c="black")
            axes.set_xlabel("spliced")
            axes.set_ylabel("unspliced")
            axes.set_title(gene[j])
            fig.tight_layout()
            break
        axes[j].scatter(x = s, y = u, s = 2, c = dsdt, vmin = - np.percentile(abs(dsdt), 99), vmax = np.percentile(abs(dsdt), 99), cmap="coolwarm")
        axes[j].scatter(x = x, y = y, s = 1, c="black")
        axes[j].set_xlabel("spliced")
        axes[j].set_ylabel("unspliced")
        axes[j].set_title(gene[j])
        fig.tight_layout()
    if idx is not None:
        plt.savefig(runPath+ "/su_plot_" + str(idx-4) + "_to_" + str(idx)+ ".png")
    else:
        plt.savefig(runPath+ "/su_plot_" + gene[0] + ".png")
    plt.close()


def plot_su_expr_umap(dm, runPath, gene):
    s = dm.adata_r[:,gene].layers["s_raw"].toarray().reshape(-1)
    s_count = dm.adata_r[:,gene].layers["spliced"].toarray().reshape(-1)
    u = dm.adata_r[:,gene].layers["u_raw"].toarray().reshape(-1)
    u_count = dm.adata_r[:,gene].layers["unspliced"].toarray().reshape(-1)
    dsdt = dm.adata_r[:,gene].layers["dsdt"].toarray().reshape(-1)
    x = np.arange(s.min(), s.max(), (s.max()-s.min())/100)
    y = dm.adata_r[:, gene].var["velocity_gamma"].to_numpy() * x

    fig, axes = plt.subplots(nrows=1, ncols=6, figsize=(3.0 * 6, 3.0))
    axes[0].scatter(x = s, y = u, s = 1, c = dsdt, vmin = - np.percentile(abs(dsdt), 99), vmax = np.percentile(abs(dsdt), 99), cmap="coolwarm")
    axes[0].scatter(x = x, y = y, s = 1, c="black")
    axes[0].set_xlabel("spliced")
    axes[0].set_ylabel("unspliced")
    axes[0].set_title(gene)
    axes[1].scatter(x = dm.adata_r.obsm["X_umap"][:,0], y = dm.adata_r.obsm["X_umap"][:,1], s = 1,
                    c = dsdt, vmin = - np.percentile(abs(dsdt), 99), vmax = np.percentile(abs(dsdt), 99),
                    cmap="coolwarm")
    axes[1].set_xlabel("UMAP1")
    axes[1].set_ylabel("UMAP2")
    axes[1].set_title("velocity")
    axes[2].scatter(x = dm.adata_r.obsm["X_umap"][:,0], y = dm.adata_r.obsm["X_umap"][:,1], s = 1,
                    c = s, vmin = min(s), vmax = np.percentile(s, 99),
                    cmap="YlGn")
    axes[2].set_xlabel("UMAP1")
    axes[2].set_ylabel("UMAP2")
    axes[2].set_title("reconstructed spliced")
    axes[3].scatter(x = dm.adata_r.obsm["X_umap"][:,0], y = dm.adata_r.obsm["X_umap"][:,1], s = 1,
                    c = s_count, vmin = min(s_count), vmax = np.percentile(s_count, 100),
                    cmap="YlGn")
    axes[3].set_xlabel("UMAP1")
    axes[3].set_ylabel("UMAP2")
    axes[3].set_title("observed spliced")
    axes[4].scatter(x = dm.adata_r.obsm["X_umap"][:,0], y = dm.adata_r.obsm["X_umap"][:,1], s = 1,
                    c = u, vmin = min(u), vmax = np.percentile(u, 99),
                    cmap="YlGn")
    axes[4].set_xlabel("UMAP1")
    axes[4].set_ylabel("UMAP2")
    axes[4].set_title("reconstructed unspliced")
    axes[5].scatter(x = dm.adata_r.obsm["X_umap"][:,0], y = dm.adata_r.obsm["X_umap"][:,1], s = 1,
                    c = u_count, vmin = min(u_count), vmax = np.percentile(u_count, 100),
                    cmap="YlGn")
    axes[5].set_xlabel("UMAP1")
    axes[5].set_ylabel("UMAP2")
    axes[5].set_title("observed unspliced")
    
    fig.tight_layout()
    plt.savefig(runPath + "/su_expr_umap_" + gene + ".png")
    plt.close()


def plot_su_phase_dsdt_var(dm, runPath, filter_idx, num_to_show=100):
    genes = dm.adata_r.var_names[filter_idx.numpy()==1]
    print("plotting {} / {} genes".format(num_to_show, genes.shape[0]))

    total_iter = num_to_show // 5
    for i in range(total_iter):
        idx = (i+1) * 5
        gene = genes[idx:(idx+5)]
        fig, axes = plt.subplots(nrows=1, ncols=len(gene), figsize=(3.0 * len(gene), 3.0))
        for j in range(len(gene)):
            s = dm.adata_r[:,gene[j]].layers["s_raw"].toarray().reshape(-1)
            u = dm.adata_r[:,gene[j]].layers["u_raw"].toarray().reshape(-1)
            dsdt_var = dm.adata_r[:,gene[j]].layers["s_var"].toarray().reshape(-1)
            x = np.arange(s.min(), s.max(), (s.max()-s.min())/100)
            y = dm.adata_r[:, gene[j]].var["velocity_gamma"].to_numpy() * x
            if len(gene) == 1:
                axes.scatter(x = s, y = u, s = 2, c = dsdt_var, cmap="coolwarm")
                axes.scatter(x = x, y = y, s = 1, c="black")
                axes.set_xlabel("spliced")
                axes.set_ylabel("unspliced")
                axes.set_title(gene[j])
                fig.tight_layout()
                break
            mappable = axes[j].scatter(x = s, y = u, s = 2, c = dsdt_var, cmap="coolwarm")
            axes[j].scatter(x = x, y = y, s = 1, c="black")
            axes[j].set_xlabel("spliced")
            axes[j].set_ylabel("unspliced")
            axes[j].set_title(gene[j])
            cbar = fig.colorbar(mappable, ax=axes[j])
            #cbar.set_label("var_dsdt")
            fig.tight_layout()
        plt.savefig(runPath+ "/su_plot_dsdt_var_" + str(idx-4) + "_to_" + str(idx)+ ".png")
        plt.close()


def plot_fluctuation_umap(dm, runPath):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5.0, 5.0))
    d_var = dm.adata_r.obsm["d_var"].sum(axis=1)
    # Ms Mu
    mappable = axes.scatter(x = dm.adata_r.obsm["X_umap"][:, 0], y = dm.adata_r.obsm["X_umap"][:, 1], s = 1, c = d_var, )
    axes.set_xlabel("UMAP 1")
    axes.set_ylabel("UMAP 2")
    axes.set_title("d fluctuation")
    cbar = fig.colorbar(mappable, ax=axes)
    fig.tight_layout()
    plt.savefig(runPath+ "/d_fluctuation.png")
    plt.close()

def plot_concentration_umap(dm, runPath):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5.0, 5.0))
    concentration = dm.adata_r.obs["concentration"]
    # Ms Mu
    mappable = axes.scatter(x = dm.adata_r.obsm["X_umap"][:, 0], y = dm.adata_r.obsm["X_umap"][:, 1], s = 1, c = concentration, )
    axes.set_xlabel("UMAP 1")
    axes.set_ylabel("UMAP 2")
    axes.set_title("vMF concentration")
    cbar = fig.colorbar(mappable, ax=axes)
    fig.tight_layout()
    plt.savefig(runPath+ "/vMF_concentration.png")
    plt.close()


def plot_su_phase_msmu_vs_recsu(dm, runPath, gene):
    fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(3.0 * 7, 3.0 * 2))
    # Ms Mu
    dm.adata_r.layers["velocity_Ms"] = dm.adata_r.layers["velocity_ss"]
    ms = dm.adata_r[:,gene].layers["Ms"].toarray().reshape(-1)
    mu = dm.adata_r[:,gene].layers["Mu"].toarray().reshape(-1)
    dsdt_msu = dm.adata_r[:,gene].layers["velocity_Ms"].toarray().reshape(-1)
    cluster = dm.adata_r.obs["clusters"].to_numpy()

    sns.scatterplot(x=dm.adata_r.obsm["X_umap"][:, 0], y=dm.adata_r.obsm["X_umap"][:, 1], 
                                    hue=cluster, ax = axes[0,0], legend=False, s = 5,)
    axes[0,0].set_xlabel("UMAP 1")
    axes[0,0].set_ylabel("UMAP 2")
    axes[0,0].set_title("clusters")

    sns.scatterplot(x=ms, y=mu, hue=cluster, ax = axes[0,1], legend=False, s = 5,)
    axes[0,1].set_xlabel("Ms")
    axes[0,1].set_ylabel("Mu")
    axes[0,1].set_title(gene + " moment")

    mappable = axes[0,2].scatter(x = ms, y = mu, s = 2, c = dsdt_msu, cmap="coolwarm", 
                                vmin=-np.max(np.abs(dsdt_msu)), vmax=np.max(np.abs(dsdt_msu)))
    axes[0,2].set_xlabel("Ms")
    axes[0,2].set_ylabel("Mu")
    axes[0,2].set_title(gene + " moment")
    cbar = fig.colorbar(mappable, ax=axes[0,2])

    mappable = axes[0,3].scatter(x = dm.adata_r.obsm["X_umap"][:, 0], y = dm.adata_r.obsm["X_umap"][:, 1], s = 1, c = ms, )
    axes[0,3].set_xlabel("UMAP 1")
    axes[0,3].set_ylabel("UMAP 2")
    axes[0,3].set_title(gene + " Ms")
    cbar = fig.colorbar(mappable, ax=axes[0,3])

    mappable = axes[0,4].scatter(x = dm.adata_r.obsm["X_umap"][:, 0], y = dm.adata_r.obsm["X_umap"][:, 1], s = 1, c = mu, )
    axes[0,4].set_xlabel("UMAP 1")
    axes[0,4].set_ylabel("UMAP 2")
    axes[0,4].set_title(gene + " Mu")
    cbar = fig.colorbar(mappable, ax=axes[0,4])

    mappable = axes[0,5].scatter(x = dm.adata_r.obsm["X_umap"][:, 0], y = dm.adata_r.obsm["X_umap"][:, 1], s = 1, c = dsdt_msu, 
                            cmap="coolwarm", vmin=-np.max(np.abs(dsdt_msu)), vmax=np.max(np.abs(dsdt_msu)))
    axes[0,5].set_xlabel("UMAP 1")
    axes[0,5].set_ylabel("UMAP 2")
    axes[0,5].set_title(gene + " dsdt Ms")
    cbar = fig.colorbar(mappable, ax=axes[0,5])

    # rec_s rec_u
    dm.adata_r.layers["velocity_recs"] = dm.adata_r.layers["dsdt"]
    rec_s = dm.adata_r[:,gene].layers["s_raw"].toarray().reshape(-1)
    rec_u = dm.adata_r[:,gene].layers["u_raw"].toarray().reshape(-1)
    dsdt_rec = dm.adata_r[:,gene].layers["velocity_recs"].toarray().reshape(-1)

    sns.scatterplot(x=dsdt_msu, y=dsdt_rec, hue=cluster, ax = axes[1,0], legend=False, s = 5,)
    axes[1,0].set_xlabel("dsdt Ms")
    axes[1,0].set_ylabel("dsdt recs")
    axes[1,0].set_title(gene + " dsdt")
    axes[1,0].set_xlim(-np.max(np.abs(dsdt_msu)), np.max(np.abs(dsdt_msu)))
    axes[1,0].set_ylim(-np.max(np.abs(dsdt_rec)), np.max(np.abs(dsdt_rec)))

    
    sns.scatterplot(x=rec_s, y=rec_u, hue=cluster, ax = axes[1,1], legend=False, s = 5,)
    axes[1,1].set_xlabel("rec s")
    axes[1,1].set_ylabel("rec u")
    axes[1,1].set_title(gene + " reconstructed")

    mappable = axes[1,2].scatter(x = rec_s, y = rec_u, s = 2, c = dsdt_rec, cmap="coolwarm", 
                                vmin=-np.max(np.abs(dsdt_rec)), vmax=np.max(np.abs(dsdt_rec)))
    axes[1,2].set_xlabel("rec s")
    axes[1,2].set_ylabel("rec u")
    axes[1,2].set_title(gene + " reconstructed")
    cbar = fig.colorbar(mappable, ax=axes[1,2])

    mappable = axes[1,3].scatter(x = dm.adata_r.obsm["X_umap"][:, 0], y = dm.adata_r.obsm["X_umap"][:, 1], s = 1, c = rec_s, )
    axes[1,3].set_xlabel("UMAP 1")
    axes[1,3].set_ylabel("UMAP 2")
    axes[1,3].set_title(gene + " rec s")
    cbar = fig.colorbar(mappable, ax=axes[1,3])

    mappable = axes[1,4].scatter(x = dm.adata_r.obsm["X_umap"][:, 0], y = dm.adata_r.obsm["X_umap"][:, 1], s = 1, c = rec_u, )
    axes[1,4].set_xlabel("UMAP 1")
    axes[1,4].set_ylabel("UMAP 2")
    axes[1,4].set_title(gene + " rec u")
    cbar = fig.colorbar(mappable, ax=axes[1,4])

    mappable = axes[1,5].scatter(x = dm.adata_r.obsm["X_umap"][:, 0], y = dm.adata_r.obsm["X_umap"][:, 1], s = 1, c = dsdt_rec, 
                            cmap="coolwarm", vmin=-np.max(np.abs(dsdt_rec)), vmax=np.max(np.abs(dsdt_rec)))
    axes[1,5].set_xlabel("UMAP 1")
    axes[1,5].set_ylabel("UMAP 2")
    axes[1,5].set_title(gene+" dsdt rec s")
    cbar = fig.colorbar(mappable, ax=axes[1,5])

    fig.tight_layout()
    plt.savefig(runPath+ "/su_plot_dsdt_comparisoon_" + gene + ".png")
    plt.close()

def compute_cossim_genewise_contribution(dm_s, runPath):
    adata_temp = dm_s.adata_r[:, dm_s.adata_r.var["estimated_genes"]] 
    dsdt_unit = adata_temp.layers["dsdt"] / np.linalg.norm(adata_temp.layers["dsdt"], axis=1).reshape(-1,1)
    dsdt_obs_unit = adata_temp.layers["dsdt_obs"] / np.linalg.norm(adata_temp.layers["dsdt_obs"], axis=1).reshape(-1,1)
    contr_score = np.array((dsdt_unit * dsdt_obs_unit).mean(axis=0))
    threshold = np.sqrt(1/ adata_temp.shape[1])

    dir_path = runPath + "/cossim_contribution"
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    # plot histogram
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.0, 3.0))
    ax.hist(contr_score, np.linspace(-1,1,100))
    ax.set_ylim(0,50)
    ax.vlines(x=[-1*threshold, threshold], ymin=0, ymax=100, color="red", alpha=0.2)
    ax.set_xlabel("contribution score")
    ax.set_ylabel("frequency")
    ax.set_title("cossim contribusion score")
    fig.tight_layout()
    plt.savefig(dir_path+ "/cossim_contribution_histogram.png")
    plt.close()

    # genes with overly consistent similarity
    
    genes_pos = adata_temp.var_names[contr_score > np.sqrt(1/ adata_temp.shape[1])]
    for gene in genes_pos:
        plot_su_phase_msmu_vs_recsu(dm_s, dir_path, gene)
    genes_neg = adata_temp.var_names[contr_score < -1 * np.sqrt(1/ adata_temp.shape[1])]
    for gene in genes_neg:
        plot_su_phase_msmu_vs_recsu(dm_s, dir_path, gene) 
    
    genes = {"positive genes" : {"name" : list(genes_pos), 
                                "score" : list(contr_score[contr_score > np.sqrt(1/ adata_temp.shape[1])])}, 
            "negative genes" : {"name" : list(genes_neg), 
                                "score" : list(contr_score[contr_score < -1 * np.sqrt(1/ adata_temp.shape[1])])},
            "uniform_score" : np.sqrt(1/ adata_temp.shape[1]).item()}
    with open(dir_path + "/contr_genes.json", "w") as f:
        json.dump(genes, f, indent=4)
    return contr_score

def plot_genewise_vel_cossim_scale(dm, exp_name, v1_key="dsdt", v2_key="dsdt_obs", filter_idx=None):
    adata_r = dm.adata_r
    if filter_idx is not None:
        filter_idx = filter_idx.numpy()
        adata_r = dm.adata_r[:, filter_idx==1]
    genewise_log_count = np.log10(np.sum(adata_r.layers["spliced_count"].toarray(), axis=0))
    cos_sim = colwise_cos_sim(adata_r.layers[v1_key],
                                    adata_r.layers[v2_key])
    
    expr_scale = dm.adata_r.layers["s_raw"].max(axis=0) / dm.adata_r.layers["u_raw"].max(axis=0)
    expr_scale[dm.adata_r.layers["u_raw"].max(axis=0)==0] = 1
    expr_scale = np.log(expr_scale)
     
    fig, ax = plt.subplots(1, 1, figsize=(5, 5 * 1))
    mappable = ax.scatter(x = genewise_log_count, 
                          y = cos_sim, s = 1, c = expr_scale,
                          cmap="coolwarm", 
                          vmin=-np.max(np.abs(expr_scale)), vmax=np.max(np.abs(expr_scale)))
    cbar = fig.colorbar(mappable, ax=ax)
    cbar.set_label("log(s/u)", rotation=270)
    plt.xlabel("log10 total spliced count")
    plt.ylabel("cosine similarity")
    plt.title("gene-wise velocity cosine similarity \n between {} and {}, spliced".format(v1_key, v2_key))
    fig.tight_layout()
    plt.savefig(exp_name+"/cossim_stratified_by_expr_scale.png");plt.close("all")