import os
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
import scvelo as scv
import cellrank as cr
import scanpy.external as sce
from scipy.io import mmwrite, mmread
import statsmodels.api as sm

np.random.seed(42)

# peak-gene linkage matrix
dir_path = "/home/nomura/Proj/mmvelo/data/10x_multiome_brain_repre_wo_IN/analysis/feature_linkage/peak_gene_linkage.mtx"
peak_gene_linkage = mmread(dir_path).toarray()

# load anndata
dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/anndata"
adata_r = sc.read_loom(dir_path + "/adata_rna.loom", obs_names="obs_names", var_names="var_names")
adata_a = sc.read_loom(dir_path + "/adata_atac.loom", obs_names="obs_names", var_names="var_names")

adata_r.obsm["X_umap"]  = pd.read_csv(dir_path + "/umap_coordinate.tsv", sep="\t", header=None).to_numpy()
adata_r.obs["clusters"] = pd.read_json(dir_path + "/cell_clusters.json", typ="series").astype("category")
adata_r.obs["pseudotime"] = pd.read_csv(dir_path + "/pseudotime.tsv", sep="\t", header=None)[0].to_numpy()

# make dir
dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/neurod2"
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

dir_path = "/home/nomura/Proj/mmvelo/data/10x_multiome_brain_repre_wo_IN/analysis/feature_linkage/peak_promoter_linkage.mtx"
peak2prom_mat = mmread(dir_path).toarray()

dir_path = "/home/nomura/Proj/mmvelo/data/10x_multiome_brain_repre_wo_IN/analysis/feature_linkage/peak_distance.tsv"
distance_df = pd.read_csv(dir_path, sep="\t")

adata_r.layers["dpdt"] = adata_a.layers["dadt"] @ peak2prom_mat
adata_r.layers["p_raw"] = adata_a.layers["a_raw"] @ peak2prom_mat


# restrict to genes with promoters in our dataset
prom_gene = peak2prom_mat.sum(0) > 0

# apply lowess regression, to corporate with variation orthogonal to pseudotime axis
lowess = sm.nonparametric.lowess
frac = 300 / adata_r.n_obs # use 100 neighbor cells for lowess regression
pseudotime = adata_r.obs["pseudotime"]
pseudotime_uni = np.zeros(adata_r.n_obs)
for i, idx in enumerate(np.argsort(pseudotime)):
    pseudotime_uni[idx] = i / adata_r.n_obs

np.argsort(pseudotime)
np.argsort(pseudotime_uni)

p_zero_time = np.where((adata_r.obs["pseudotime"]==0))[0]
p_zero_time = pseudotime_uni[p_zero_time]
adata_r.obs["pseudotime_uni"] = pseudotime_uni #
adata_r_exn = adata_r[adata_r.obs["pseudotime"] >= 0, :]


gene = "Neurod2"
gene_idx = np.where(adata_r_exn.var_names == gene)[0].item()
peak_idx = list(np.where(peak_gene_linkage[:, gene_idx] > 0)[0])

file_path = "/home/nomura/Proj/mmvelo/data/10x_multiome_brain_repre_wo_IN/analysis/feature_linkage/peak_distance.tsv"
dist_tss = pd.read_csv(file_path, sep="\t")

peak_tss_dist = []
for idx in peak_idx:
    peak_name = adata_a.var_names[idx]
    dis_tss_idx = np.where((dist_tss["peak"] == peak_name) & (dist_tss["gene"] == gene))[0].item()
    peak_tss_dist.append(dist_tss["distance"][dis_tss_idx])

len(peak_idx)
len(peak_tss_dist)

# lowess regression for peaks
adata_a_exn = adata_a[adata_r_exn.obs_names, :]
adata_a_exn.obs["pseudotime_uni"] = adata_r_exn.obs["pseudotime_uni"]
pseudotime_uni = adata_a_exn.obs["pseudotime_uni"]
lowess = sm.nonparametric.lowess
frac = 300 / adata_r.n_obs # use 100 neighbor cells for lowess regression

a_s, da_s = [], []

for idx in peak_idx:
    a_s.append(lowess(adata_a_exn[:,idx].layers["a_raw"].toarray().reshape(-1), pseudotime_uni, frac=frac, return_sorted=False))
    da_s.append(lowess(adata_a_exn[:,idx].layers["dadt"].toarray().reshape(-1), pseudotime_uni, frac=frac, return_sorted=False))

a_s[0].shape

p_t_min = adata_r_exn.obs["pseudotime_uni"].min()
p_t_max = adata_r_exn.obs["pseudotime_uni"].max()

dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/neurod2"
file_path = dir_path + "/peaks_lowess"
if not os.path.exists(file_path):
    os.mkdir(file_path)

# tf expression
gene
for i in range(2):
    if i == 0: # spliced
        x_raws = adata_r_exn[:, gene].layers["s_raw"].toarray().reshape(-1)
        dxdt = adata_r_exn[:, gene].layers["dsdt"].toarray().reshape(-1)
        x_raws_s = lowess(adata_r_exn[:, gene].layers["s_raw"].toarray().reshape(-1), pseudotime_uni, frac=frac, return_sorted=False)
        dxdt_s = lowess(adata_r_exn[:, gene].layers["dsdt"].toarray().reshape(-1), pseudotime_uni, frac=frac, return_sorted=False)
    elif i == 1: # unspliced
        x_raws = adata_r_exn[:, gene].layers["u_raw"].toarray().reshape(-1)
        dxdt = adata_r_exn[:, gene].layers["dudt"].toarray().reshape(-1)
        x_raws_s = lowess(adata_r_exn[:, gene].layers["u_raw"].toarray().reshape(-1), pseudotime_uni, frac=frac, return_sorted=False)
        dxdt_s = lowess(adata_r_exn[:, gene].layers["dudt"].toarray().reshape(-1), pseudotime_uni, frac=frac, return_sorted=False)
    
    x_raws = x_raws / np.abs(x_raws_s).max()
    x_raws_s = x_raws_s / np.abs(x_raws_s).max()
    dxdt = dxdt / np.abs(dxdt_s).max()
    dxdt_s = dxdt_s / np.abs(dxdt_s).max()
    
    pseudotime = adata_a_exn.obs["pseudotime_uni"]
    fig, axes = plt.subplots(1, 2, figsize=(5*2, 5))
    axes[0].scatter(x=pseudotime, y=x_raws, s=2, color="black", alpha=0.1)
    xs, ys = zip(*sorted(zip(pseudotime, x_raws_s)))
    if i == 0:
        axes[0].plot(xs, ys, color="black", label="spliced")
    elif i == 1:
        axes[0].plot(xs, ys, color="black", label="unspliced")
    axes[0].legend()
    axes[0].set_xlim(p_t_min, p_t_max)
    axes[0].set_ylim(0, 1.5)
    axes[0].set_xlabel("pseudotime")
    axes[0].set_ylabel("norm rec count")
    axes[0].set_title("{}".format(gene))

    axes[1].scatter(x=pseudotime, y=dxdt, s=2, color="black", alpha=0.1)
    xs, ys = zip(*sorted(zip(pseudotime, dxdt_s)))
    if i == 0:
        axes[1].plot(xs, ys, color="black", label="spliced")
    elif i == 1:
        axes[1].plot(xs, ys, color="black", label="unspliced")
    axes[1].legend()
    axes[1].set_xlim(p_t_min, p_t_max)
    axes[1].set_ylim(-1.5, 1.5)
    axes[1].set_xlabel("pseudotime")
    axes[1].set_ylabel("norm vel")
    axes[1].set_title("{} velocity".format(gene))

    fig.tight_layout()
    if i == 0:
        plt.savefig(file_path + "/time_change_{}_s_smooth.png".format(gene), bbox_inches='tight')
    elif i == 1:
        plt.savefig(file_path + "/time_change_{}_u_smooth.png".format(gene), bbox_inches='tight')
    plt.close()


for i, idx in enumerate(peak_idx):
    peak_name = adata_a_exn.var_names[idx]
    peak_dist = peak_tss_dist[i]
    a_raws = adata_a_exn[:, peak_name].layers["a_raw"].toarray().reshape(-1)
    dadt = adata_a_exn[:, peak_name].layers["dadt"].toarray().reshape(-1)
    a_raws_s = a_s[i]
    dadt_s = da_s[i]
    
    a_raws = a_raws / np.abs(a_raws_s).max()
    a_raws_s = a_raws_s / np.abs(a_raws_s).max()
    
    dadt = dadt / np.abs(dadt_s).max()
    dadt_s = dadt_s / np.abs(dadt_s).max()
    

    pseudotime = adata_a_exn.obs["pseudotime_uni"]

    fig, axes = plt.subplots(1, 2, figsize=(5*2, 5))
    axes[0].scatter(x=pseudotime, y=a_raws, s=2, color="black", alpha=0.1)
    xs, ys = zip(*sorted(zip(pseudotime, a_raws_s)))
    axes[0].plot(xs, ys, color="black", label="distance = {}".format(peak_dist))
    axes[0].legend()
    axes[0].set_xlim(p_t_min, p_t_max)
    axes[0].set_ylim(0, 1.5)
    axes[0].set_xlabel("pseudotime")
    axes[0].set_ylabel("norm rec count")
    axes[0].set_title("{} count".format(peak_name))

    axes[1].scatter(x=pseudotime, y=dadt, s=2, color="black", alpha=0.1)
    xs, ys = zip(*sorted(zip(pseudotime, dadt_s)))
    axes[1].plot(xs, ys, color="black", label="distance = {}".format(peak_dist))
    axes[1].legend()
    axes[1].set_xlim(p_t_min, p_t_max)
    axes[1].set_ylim(-1.5, 1.5)
    axes[1].set_xlabel("pseudotime")
    axes[1].set_ylabel("norm  vel")
    axes[1].set_title("{} velocity".format(peak_name))

    fig.tight_layout()
    plt.savefig(file_path + "/time_change_{}_smooth.png".format(peak_name), bbox_inches='tight')
    plt.close()


neurod2_peak_idx = [0, 14, 2]
fig, axes = plt.subplots(1, 2, figsize=(5*2, 5))
for i, idx in enumerate(neurod2_peak_idx):
    peak_dist = peak_tss_dist[idx]
    a_raws_s = a_s[idx]
    dadt_s = da_s[idx]
    idx = peak_idx[idx]
    peak_name = adata_a_exn.var_names[idx]

    a_raws = adata_a_exn[:, peak_name].layers["a_raw"].toarray().reshape(-1)
    dadt = adata_a_exn[:, peak_name].layers["dadt"].toarray().reshape(-1)
    
    a_raws = a_raws / np.abs(a_raws_s).max()
    a_raws_s = a_raws_s / np.abs(a_raws_s).max()
    dadt = dadt / np.abs(dadt_s).max()
    dadt_s = dadt_s / np.abs(dadt_s).max()
    pseudotime = adata_a_exn.obs["pseudotime_uni"]
    xs, ys = zip(*sorted(zip(pseudotime, a_raws_s)))
    if i == 0:
        axes[0].scatter(x=pseudotime, y=a_raws, s=2, color="green", alpha=0.1)
        axes[0].plot(xs, ys, color="green", label="promoter")
    elif i == 1:
        axes[0].scatter(x=pseudotime, y=a_raws, s=2, color="black", alpha=0.1)
        axes[0].plot(xs, ys, color="black", label="cis enhancer")
    #elif i == 2:
        #axes[0].scatter(x=pseudotime, y=a_raws, s=2, color="black", alpha=0.1)
        #axes[0].plot(xs, ys, color="black", label="distal enhancer")

    xs, ys = zip(*sorted(zip(pseudotime, dadt_s)))
    if i == 0:
        axes[1].scatter(x=pseudotime, y=dadt, s=2, color="green", alpha=0.1)
        axes[1].plot(xs, ys, color="green", label="promoter")
    elif i == 1:
        axes[1].scatter(x=pseudotime, y=dadt, s=2, color="black", alpha=0.1)
        axes[1].plot(xs, ys, color="black", label="cis enhancer")
    #elif i == 2:
        #axes[1].scatter(x=pseudotime, y=dadt, s=2, color="black", alpha=0.1)
        #axes[1].plot(xs, ys, color="black", label="distal enhancer")

for i in range(2):
    if i == 0: # spliced
        x_raws = adata_r_exn[:, gene].layers["s_raw"].toarray().reshape(-1)
        dxdt = adata_r_exn[:, gene].layers["dsdt"].toarray().reshape(-1)
        x_raws_s = lowess(adata_r_exn[:, gene].layers["s_raw"].toarray().reshape(-1), pseudotime_uni, frac=frac, return_sorted=False)
        dxdt_s = lowess(adata_r_exn[:, gene].layers["dsdt"].toarray().reshape(-1), pseudotime_uni, frac=frac, return_sorted=False)
    elif i == 1: # unspliced
        x_raws = adata_r_exn[:, gene].layers["u_raw"].toarray().reshape(-1)
        dxdt = adata_r_exn[:, gene].layers["dudt"].toarray().reshape(-1)
        x_raws_s = lowess(adata_r_exn[:, gene].layers["u_raw"].toarray().reshape(-1), pseudotime_uni, frac=frac, return_sorted=False)
        dxdt_s = lowess(adata_r_exn[:, gene].layers["dudt"].toarray().reshape(-1), pseudotime_uni, frac=frac, return_sorted=False)
    
    x_raws = x_raws / np.abs(x_raws_s).max()
    x_raws_s = x_raws_s / np.abs(x_raws_s).max()
    dxdt = dxdt / np.abs(dxdt_s).max()
    dxdt_s = dxdt_s / np.abs(dxdt_s).max()
    pseudotime = adata_a_exn.obs["pseudotime_uni"]

    xs, ys = zip(*sorted(zip(pseudotime, x_raws_s)))
    if i == 0:
        axes[0].plot(xs, ys, color="blue", label="spliced")
        axes[0].scatter(x=pseudotime, y=x_raws, s=2, color="blue", alpha=0.1)
    elif i == 1:
        axes[0].plot(xs, ys, color="red", label="unspliced")
        axes[0].scatter(x=pseudotime, y=x_raws, s=2, color="red", alpha=0.1)

    xs, ys = zip(*sorted(zip(pseudotime, dxdt_s)))
    if i == 0:
        axes[1].plot(xs, ys, color="blue", label="spliced")
        axes[1].scatter(x=pseudotime, y=dxdt, s=2, color="blue", alpha=0.1)
    elif i == 1:
        axes[1].plot(xs, ys, color="red", label="unspliced")
        axes[1].scatter(x=pseudotime, y=dxdt, s=2, color="red", alpha=0.1)

axes[0].legend()
axes[0].set_xlim(p_t_min, p_t_max)
axes[0].set_ylim(0, 1.5)
axes[0].set_xlabel("pseudotime")
axes[0].set_ylabel("norm rec count")
axes[0].set_title("count")
axes[1].legend()
axes[1].set_xlim(p_t_min, p_t_max)
axes[1].set_ylim(-1.5, 1.5)
axes[1].set_xlabel("pseudotime")
axes[1].set_ylabel("norm  vel")
axes[1].set_title("velocity")
fig.tight_layout()
plt.savefig(file_path + "/time_change_neurod2.png", bbox_inches='tight')
plt.close()


## for fig
neurod2_peak_idx = [0, 14, 2]
fig, axes = plt.subplots(1, 2, figsize=(5*2, 5))
for i, idx in enumerate(neurod2_peak_idx):
    peak_dist = peak_tss_dist[idx]
    a_raws_s = a_s[idx]
    dadt_s = da_s[idx]
    idx = peak_idx[idx]
    peak_name = adata_a_exn.var_names[idx]

    a_raws = adata_a_exn[:, peak_name].layers["a_raw"].toarray().reshape(-1)
    dadt = adata_a_exn[:, peak_name].layers["dadt"].toarray().reshape(-1)
    
    a_raws = a_raws / np.abs(a_raws_s).max()
    a_raws_s = a_raws_s / np.abs(a_raws_s).max()
    dadt = dadt / np.abs(dadt_s).max()
    dadt_s = dadt_s / np.abs(dadt_s).max()
    pseudotime = adata_a_exn.obs["pseudotime_uni"]
    xs, ys = zip(*sorted(zip(pseudotime, a_raws_s)))
    if i == 0:
        axes[0].scatter(x=pseudotime, y=a_raws, s=2, color="green", alpha=0.1)
        axes[0].plot(xs, ys, color="green", label="promoter", linewidth=2)
    elif i == 1:
        axes[0].scatter(x=pseudotime, y=a_raws, s=2, color="black", alpha=0.1)
        axes[0].plot(xs, ys, color="black", label="cis enhancer", linewidth=2)
    #elif i == 2:
        #axes[0].scatter(x=pseudotime, y=a_raws, s=2, color="black", alpha=0.1)
        #axes[0].plot(xs, ys, color="black", label="distal enhancer")

    xs, ys = zip(*sorted(zip(pseudotime, dadt_s)))
    if i == 0:
        axes[1].scatter(x=pseudotime, y=dadt, s=2, color="green", alpha=0.1)
        axes[1].plot(xs, ys, color="green", label="promoter", linewidth=2)
    elif i == 1:
        axes[1].scatter(x=pseudotime, y=dadt, s=2, color="black", alpha=0.1)
        axes[1].plot(xs, ys, color="black", label="cis enhancer", linewidth=2)
    #elif i == 2:
        #axes[1].scatter(x=pseudotime, y=dadt, s=2, color="black", alpha=0.1)
        #axes[1].plot(xs, ys, color="black", label="distal enhancer")

for i in range(2):
    if i == 0: # spliced
        x_raws = adata_r_exn[:, gene].layers["s_raw"].toarray().reshape(-1)
        dxdt = adata_r_exn[:, gene].layers["dsdt"].toarray().reshape(-1)
        x_raws_s = lowess(adata_r_exn[:, gene].layers["s_raw"].toarray().reshape(-1), pseudotime_uni, frac=frac, return_sorted=False)
        dxdt_s = lowess(adata_r_exn[:, gene].layers["dsdt"].toarray().reshape(-1), pseudotime_uni, frac=frac, return_sorted=False)
    elif i == 1: # unspliced
        x_raws = adata_r_exn[:, gene].layers["u_raw"].toarray().reshape(-1)
        dxdt = adata_r_exn[:, gene].layers["dudt"].toarray().reshape(-1)
        x_raws_s = lowess(adata_r_exn[:, gene].layers["u_raw"].toarray().reshape(-1), pseudotime_uni, frac=frac, return_sorted=False)
        dxdt_s = lowess(adata_r_exn[:, gene].layers["dudt"].toarray().reshape(-1), pseudotime_uni, frac=frac, return_sorted=False)
    
    x_raws = x_raws / np.abs(x_raws_s).max()
    x_raws_s = x_raws_s / np.abs(x_raws_s).max()
    dxdt = dxdt / np.abs(dxdt_s).max()
    dxdt_s = dxdt_s / np.abs(dxdt_s).max()
    pseudotime = adata_a_exn.obs["pseudotime_uni"]

    xs, ys = zip(*sorted(zip(pseudotime, x_raws_s)))
    if i == 0:
        axes[0].plot(xs, ys, color="blue", label="spliced", linewidth=2)
        axes[0].scatter(x=pseudotime, y=x_raws, s=2, color="blue", alpha=0.1)
    elif i == 1:
        axes[0].plot(xs, ys, color="red", label="unspliced", linewidth=2)
        axes[0].scatter(x=pseudotime, y=x_raws, s=2, color="red", alpha=0.1)

    xs, ys = zip(*sorted(zip(pseudotime, dxdt_s)))
    if i == 0:
        axes[1].plot(xs, ys, color="blue", label="spliced", linewidth=2)
        axes[1].scatter(x=pseudotime, y=dxdt, s=2, color="blue", alpha=0.1)
    elif i == 1:
        axes[1].plot(xs, ys, color="red", label="unspliced", linewidth=2)
        axes[1].scatter(x=pseudotime, y=dxdt, s=2, color="red", alpha=0.1)

#axes[0].legend()
axes[0].set_xlim(p_t_min, p_t_max)
axes[0].set_ylim(0, 1.5)
#axes[0].set_xlabel("pseudotime")
#axes[0].set_ylabel("norm rec count")
axes[0].set_title("")
[axes[0].spines[side].set_visible(False) for side in ['right','top', "bottom", "left"]]
axes[0].tick_params(color='none')
axes[0].set_xticklabels([])
axes[0].set_yticklabels([])
#axes[1].legend()
axes[1].set_xlim(p_t_min, p_t_max)
axes[1].set_ylim(-1.5, 1.5)
axes[1].set_xlabel("pseudotime")
axes[1].set_ylabel("norm  vel")
axes[1].set_title("")
plt.gca().axis("off")
fig.tight_layout()
plt.savefig(file_path + "/time_change_neurod2_blank.png", bbox_inches='tight', dpi=300)
plt.close()


## for fig (sep)
## count
neurod2_peak_idx = [0, 14, 2]
fig, axes = plt.subplots(1, 1, figsize=(5, 5))
for i, idx in enumerate(neurod2_peak_idx):
    peak_dist = peak_tss_dist[idx]
    a_raws_s = a_s[idx]
    dadt_s = da_s[idx]
    idx = peak_idx[idx]
    peak_name = adata_a_exn.var_names[idx]

    a_raws = adata_a_exn[:, peak_name].layers["a_raw"].toarray().reshape(-1)
    dadt = adata_a_exn[:, peak_name].layers["dadt"].toarray().reshape(-1)
    
    a_raws = a_raws / np.abs(a_raws_s).max()
    a_raws_s = a_raws_s / np.abs(a_raws_s).max()
    dadt = dadt / np.abs(dadt_s).max()
    dadt_s = dadt_s / np.abs(dadt_s).max()
    pseudotime = adata_a_exn.obs["pseudotime_uni"]
    xs, ys = zip(*sorted(zip(pseudotime, a_raws_s)))
    if i == 0:
        axes.scatter(x=pseudotime, y=a_raws, s=2, color="green", alpha=0.1)
        axes.plot(xs, ys, color="green", label="promoter", linewidth=2)
    elif i == 1:
        axes.scatter(x=pseudotime, y=a_raws, s=2, color="black", alpha=0.1)
        axes.plot(xs, ys, color="black", label="cis enhancer", linewidth=2)
    #elif i == 2:
        #axes[0].scatter(x=pseudotime, y=a_raws, s=2, color="black", alpha=0.1)
        #axes[0].plot(xs, ys, color="black", label="distal enhancer")

for i in range(2):
    if i == 0: # spliced
        x_raws = adata_r_exn[:, gene].layers["s_raw"].toarray().reshape(-1)
        dxdt = adata_r_exn[:, gene].layers["dsdt"].toarray().reshape(-1)
        x_raws_s = lowess(adata_r_exn[:, gene].layers["s_raw"].toarray().reshape(-1), pseudotime_uni, frac=frac, return_sorted=False)
        dxdt_s = lowess(adata_r_exn[:, gene].layers["dsdt"].toarray().reshape(-1), pseudotime_uni, frac=frac, return_sorted=False)
    elif i == 1: # unspliced
        x_raws = adata_r_exn[:, gene].layers["u_raw"].toarray().reshape(-1)
        dxdt = adata_r_exn[:, gene].layers["dudt"].toarray().reshape(-1)
        x_raws_s = lowess(adata_r_exn[:, gene].layers["u_raw"].toarray().reshape(-1), pseudotime_uni, frac=frac, return_sorted=False)
        dxdt_s = lowess(adata_r_exn[:, gene].layers["dudt"].toarray().reshape(-1), pseudotime_uni, frac=frac, return_sorted=False)
    
    x_raws = x_raws / np.abs(x_raws_s).max()
    x_raws_s = x_raws_s / np.abs(x_raws_s).max()
    dxdt = dxdt / np.abs(dxdt_s).max()
    dxdt_s = dxdt_s / np.abs(dxdt_s).max()
    pseudotime = adata_a_exn.obs["pseudotime_uni"]

    xs, ys = zip(*sorted(zip(pseudotime, x_raws_s)))
    if i == 0:
        axes.plot(xs, ys, color="blue", label="spliced", linewidth=2)
        axes.scatter(x=pseudotime, y=x_raws, s=2, color="blue", alpha=0.1)
    elif i == 1:
        axes.plot(xs, ys, color="red", label="unspliced", linewidth=2)
        axes.scatter(x=pseudotime, y=x_raws, s=2, color="red", alpha=0.1)

axes.legend()
axes.set_xlim(p_t_min, p_t_max)
axes.set_ylim(0, 1.5)
axes.set_xlabel("pseudotime")
axes.set_ylabel("norm rec count")
axes.set_title("")
#plt.gca().axis("off")
fig.tight_layout()
#plt.savefig(file_path + "/time_change_neurod2_count_blank.png", bbox_inches='tight', dpi=300)
plt.savefig(file_path + "/time_change_neurod2_count.png", bbox_inches='tight', dpi=300)
plt.close()

## for fig (sep)
## velocity
neurod2_peak_idx = [0, 14, 2]
fig, axes = plt.subplots(1, 1, figsize=(5, 5))
for i, idx in enumerate(neurod2_peak_idx):
    peak_dist = peak_tss_dist[idx]
    a_raws_s = a_s[idx]
    dadt_s = da_s[idx]
    idx = peak_idx[idx]
    peak_name = adata_a_exn.var_names[idx]

    a_raws = adata_a_exn[:, peak_name].layers["a_raw"].toarray().reshape(-1)
    dadt = adata_a_exn[:, peak_name].layers["dadt"].toarray().reshape(-1)
    
    a_raws = a_raws / np.abs(a_raws_s).max()
    a_raws_s = a_raws_s / np.abs(a_raws_s).max()
    dadt = dadt / np.abs(dadt_s).max()
    dadt_s = dadt_s / np.abs(dadt_s).max()
    pseudotime = adata_a_exn.obs["pseudotime_uni"]
    xs, ys = zip(*sorted(zip(pseudotime, a_raws_s)))
    #elif i == 2:
        #axes[0].scatter(x=pseudotime, y=a_raws, s=2, color="black", alpha=0.1)
        #axes[0].plot(xs, ys, color="black", label="distal enhancer")

    xs, ys = zip(*sorted(zip(pseudotime, dadt_s)))
    if i == 0:
        axes.scatter(x=pseudotime, y=dadt, s=2, color="green", alpha=0.1)
        axes.plot(xs, ys, color="green", label="promoter", linewidth=2)
    elif i == 1:
        axes.scatter(x=pseudotime, y=dadt, s=2, color="black", alpha=0.1)
        axes.plot(xs, ys, color="black", label="cis enhancer", linewidth=2)
    #elif i == 2:
        #axes[1].scatter(x=pseudotime, y=dadt, s=2, color="black", alpha=0.1)
        #axes[1].plot(xs, ys, color="black", label="distal enhancer")

for i in range(2):
    if i == 0: # spliced
        x_raws = adata_r_exn[:, gene].layers["s_raw"].toarray().reshape(-1)
        dxdt = adata_r_exn[:, gene].layers["dsdt"].toarray().reshape(-1)
        x_raws_s = lowess(adata_r_exn[:, gene].layers["s_raw"].toarray().reshape(-1), pseudotime_uni, frac=frac, return_sorted=False)
        dxdt_s = lowess(adata_r_exn[:, gene].layers["dsdt"].toarray().reshape(-1), pseudotime_uni, frac=frac, return_sorted=False)
    elif i == 1: # unspliced
        x_raws = adata_r_exn[:, gene].layers["u_raw"].toarray().reshape(-1)
        dxdt = adata_r_exn[:, gene].layers["dudt"].toarray().reshape(-1)
        x_raws_s = lowess(adata_r_exn[:, gene].layers["u_raw"].toarray().reshape(-1), pseudotime_uni, frac=frac, return_sorted=False)
        dxdt_s = lowess(adata_r_exn[:, gene].layers["dudt"].toarray().reshape(-1), pseudotime_uni, frac=frac, return_sorted=False)
    
    x_raws = x_raws / np.abs(x_raws_s).max()
    x_raws_s = x_raws_s / np.abs(x_raws_s).max()
    dxdt = dxdt / np.abs(dxdt_s).max()
    dxdt_s = dxdt_s / np.abs(dxdt_s).max()
    pseudotime = adata_a_exn.obs["pseudotime_uni"]


    xs, ys = zip(*sorted(zip(pseudotime, dxdt_s)))
    if i == 0:
        axes.plot(xs, ys, color="blue", label="spliced", linewidth=2)
        axes.scatter(x=pseudotime, y=dxdt, s=2, color="blue", alpha=0.1)
    elif i == 1:
        axes.plot(xs, ys, color="red", label="unspliced", linewidth=2)
        axes.scatter(x=pseudotime, y=dxdt, s=2, color="red", alpha=0.1)

#axes[0].legend()
axes.legend()
axes.set_xlim(p_t_min, p_t_max)
axes.set_ylim(-1.5, 1.5)
axes.set_xlabel("pseudotime")
axes.set_ylabel("norm  vel")
axes.set_title("")
#plt.gca().axis("off")
fig.tight_layout()
#plt.savefig(file_path + "/time_change_neurod2_velocity_blank.png", bbox_inches='tight', dpi=300)
plt.savefig(file_path + "/time_change_neurod2_velocity.png", bbox_inches='tight', dpi=300)
plt.close()