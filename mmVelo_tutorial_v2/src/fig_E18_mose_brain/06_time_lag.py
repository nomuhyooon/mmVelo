import os
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ad
import scanpy as sc
import scvelo as scv
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
dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/time_lag"
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

dir_path = "/home/nomura/Proj/mmvelo/data/10x_multiome_brain_repre_wo_IN/analysis/feature_linkage/peak_promoter_linkage.mtx"
peak2prom_mat = mmread(dir_path).toarray()

dir_path = "/home/nomura/Proj/mmvelo/data/10x_multiome_brain_repre_wo_IN/analysis/feature_linkage/peak_distance.tsv"
distance_df = pd.read_csv(dir_path, sep="\t")

adata_r.layers["dpdt"] = adata_a.layers["dadt"] @ peak2prom_mat
adata_r.layers["p_raw"] = adata_a.layers["a_raw"] @ peak2prom_mat
adata_r.layers["M_p"] = adata_a.layers["Ma"] @ peak2prom_mat
adata_r.layers["p_count"] = adata_a.layers["atac_count"] @ peak2prom_mat

# restrict to NIPC subset...






# restrict to genes with promoters in our dataset
prom_gene = peak2prom_mat.sum(0) > 0

# apply lowess regression, to corporate with variation orthogonal to pseudotime axis
lowess = sm.nonparametric.lowess
frac = 10 / adata_r_ipc.n_obs # use 10 neighbor cells for lowess regression
pseudotime = adata_r_ipc.obs["pseudotime"]

s_s, u_s, p_s = [], [], []
ds_s, du_s, dp_s = [], [], []

for gene in adata_r_ipc.var_names:
    s_s.append(lowess(adata_r_ipc[:,gene].layers["s_raw"].toarray().reshape(-1), pseudotime, frac=frac, return_sorted=False))
    u_s.append(lowess(adata_r_ipc[:,gene].layers["u_raw"].toarray().reshape(-1), pseudotime, frac=frac, return_sorted=False))
    p_s.append(lowess(adata_r_ipc[:,gene].layers["p_raw"].toarray().reshape(-1), pseudotime, frac=frac, return_sorted=False))
    ds_s.append(lowess(adata_r_ipc[:,gene].layers["dsdt"].toarray().reshape(-1), pseudotime, frac=frac, return_sorted=False))
    du_s.append(lowess(adata_r_ipc[:,gene].layers["dudt"].toarray().reshape(-1), pseudotime, frac=frac, return_sorted=False))
    dp_s.append(lowess(adata_r_ipc[:,gene].layers["dpdt"].toarray().reshape(-1), pseudotime, frac=frac, return_sorted=False))

# lowess values are sorted according to pseudotime...

adata_r_ipc.layers["s_s"] = np.array(s_s).T
adata_r_ipc.layers["u_s"] = np.array(u_s).T
adata_r_ipc.layers["p_s"] = np.array(p_s).T
adata_r_ipc.layers["ds_s"] = np.array(ds_s).T
adata_r_ipc.layers["du_s"] = np.array(du_s).T
adata_r_ipc.layers["dp_s"] = np.array(dp_s).T

# assign discrete cluster to IPC cells, with pseudoimte bin
# for tl rank_gene_groups
bin_size = 28 # 279 cells in IPC
adata_r_ipc.obs["pseudoimte_bins"] = None
for i in range(10):
    if i==9:
        cell_in_bin = np.argsort(adata_r_ipc.obs.pseudotime)[(bin_size*i) :]
    else:
        cell_in_bin = np.argsort(adata_r_ipc.obs.pseudotime)[(bin_size*i) : (bin_size * (i+1))]
    adata_r_ipc.obs["pseudoimte_bins"][cell_in_bin] = "p_bin_{}".format(i)
adata_r_ipc.obs["pseudoimte_bins"] = adata_r_ipc.obs["pseudoimte_bins"].astype("category")

sc.tl.rank_genes_groups(adata_r_ipc, groupby="pseudoimte_bins", 
                        groups=["p_bin_0", "p_bin_9"], method="wilcoxon")


dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/time_lag/IPC"
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
sc.pl.rank_genes_groups_violin(adata_r_ipc, groups="p_bin_0", n_genes=10)
fig.tight_layout()
plt.savefig(dir_path + "/rank_genes_groups_bin_0.png", bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
sc.pl.rank_genes_groups_violin(adata_r_ipc, groups="p_bin_9", n_genes=10)
fig.tight_layout()
plt.savefig(dir_path + "/rank_genes_groups_bin_9.png", bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
sc.pl.rank_genes_groups(adata_r_ipc, n_genes=25, sharey=False)
fig.tight_layout()
plt.savefig(dir_path + "/rank_genes_groups.png", bbox_inches='tight')
plt.close()

groups = adata_r_ipc.uns["rank_genes_groups"]["names"].dtype.names
df_pval = pd.DataFrame(
    {group + '_' + key[:1]: adata_r_ipc.uns["rank_genes_groups"][key][group]
    for group in groups for key in ['names', 'pvals']}) # pvals_adj

# analyse genes with non-adjusted pvalue <= 0.05
(df_pval["p_bin_0_p"] <= 0.05).sum() # 352
(df_pval["p_bin_9_p"] <= 0.05).sum() # 275
adata_r_ipc.var_names[((df_pval["p_bin_0_p"] <= 0.05) | (df_pval["p_bin_9_p"] <= 0.05))]

# genes with adjusted pvalue
gene_idx = (prom_gene & ((df_pval["p_bin_0_p"] <= 0.05) | (df_pval["p_bin_9_p"] <= 0.05))) # 175 genes
analysis_gene = adata_r_ipc.var_names[gene_idx]



# check the pattern along pseudotime
dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/time_lag/IPC/gene"
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

for gene in analysis_gene:
    s_raws = adata_r_ipc[:, gene].layers["s_raw"].toarray().reshape(-1)
    u_raws = adata_r_ipc[:, gene].layers["u_raw"].toarray().reshape(-1)
    p_raws = adata_r_ipc[:, gene].layers["p_raw"].toarray().reshape(-1)
    s_s = adata_r_ipc[:, gene].layers["s_s"].toarray().reshape(-1)
    u_s = adata_r_ipc[:, gene].layers["u_s"].toarray().reshape(-1)
    p_s = adata_r_ipc[:, gene].layers["p_s"].toarray().reshape(-1)
    s_s = s_s / np.abs(s_raws).max()
    s_raw = s_raws / np.abs(s_raws).max()
    u_s = u_s / np.abs(u_raws).max()
    u_raw = u_raws / np.abs(u_raws).max()
    p_s = p_s / np.abs(p_raws).max()
    p_raw = p_raws / np.abs(p_raws).max()

    dsdt = adata_r_ipc[:, gene].layers["dsdt"].toarray().reshape(-1)
    dudt = adata_r_ipc[:, gene].layers["dudt"].toarray().reshape(-1)
    dpdt = adata_r_ipc[:, gene].layers["dpdt"].toarray().reshape(-1)
    ds_s = adata_r_ipc[:, gene].layers["ds_s"].toarray().reshape(-1)
    du_s = adata_r_ipc[:, gene].layers["du_s"].toarray().reshape(-1)
    dp_s = adata_r_ipc[:, gene].layers["dp_s"].toarray().reshape(-1)
    ds_s = ds_s / np.abs(s_raws).max()
    dsdt = dsdt / np.abs(s_raws).max()
    du_s = du_s / np.abs(u_raws).max()
    dudt = dudt / np.abs(u_raws).max()
    dp_s = dp_s / np.abs(p_raws).max()
    dpdt = dpdt / np.abs(p_raws).max()
    pseudotime = adata_r_ipc.obs["pseudotime"]
    equal_time_bin = [i/len(pseudotime) for i in range(len(pseudotime))]

    fig, axes = plt.subplots(1, 2, figsize=(5*2, 5))
    xs, ys = zip(*sorted(zip(pseudotime, s_raw)))
    axes[0].scatter(x=equal_time_bin, y=ys, s=5, color="blue", alpha=0.3)
    xs, ys = zip(*sorted(zip(pseudotime, s_s)))
    axes[0].plot(equal_time_bin, ys, color="blue", label="spliced")
    xs, ys = zip(*sorted(zip(pseudotime, u_raw)))
    axes[0].scatter(x=equal_time_bin, y=ys, s=5, color="red", alpha=0.3)
    xs, ys = zip(*sorted(zip(pseudotime, u_s)))
    axes[0].plot(equal_time_bin, ys, color="red", label="unspliced")
    xs, ys = zip(*sorted(zip(pseudotime, p_raw)))
    axes[0].scatter(x=equal_time_bin, y=ys, s=5, color="black", alpha=0.3)
    xs, ys = zip(*sorted(zip(pseudotime, p_s)))
    axes[0].plot(equal_time_bin, ys, color="black", label="promoter")
    axes[0].legend()
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("pseudotime")
    axes[0].set_ylabel("count")
    axes[0].set_title("{} count".format(gene))

    xs, ys = zip(*sorted(zip(pseudotime, dsdt)))
    axes[1].scatter(x=equal_time_bin, y=ys, s=5, color="blue", alpha=0.3)
    xs, ys = zip(*sorted(zip(pseudotime, ds_s)))
    axes[1].plot(equal_time_bin, ys, color="blue", label="dsdt")
    xs, ys = zip(*sorted(zip(pseudotime, dudt)))
    axes[1].scatter(x=equal_time_bin, y=ys, s=5, color="red", alpha=0.3)
    xs, ys = zip(*sorted(zip(pseudotime, du_s)))
    axes[1].plot(equal_time_bin, ys, color="red", label="dudt")
    xs, ys = zip(*sorted(zip(pseudotime, dpdt)))
    axes[1].scatter(x=equal_time_bin, y=ys, s=5, color="black", alpha=0.3)
    xs, ys = zip(*sorted(zip(pseudotime, dp_s)))
    axes[1].plot(equal_time_bin, ys, color="black", label="dpdt")
    axes[1].legend()
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("pseudotime")
    axes[1].set_ylabel("velocity")
    axes[1].set_title("{} velocity".format(gene))
    fig.tight_layout()
    plt.savefig(dir_path + "/time_change_{}_smooth.png".format(gene), bbox_inches='tight')
    plt.close()
    break


##
## lowess 30
##

lowess = sm.nonparametric.lowess
frac = 30 / adata_r_ipc.n_obs # use 10 neighbor cells for lowess regression
pseudotime = adata_r_ipc.obs["pseudotime"]

s_s, u_s, p_s = [], [], []
ds_s, du_s, dp_s = [], [], []

for gene in adata_r_ipc.var_names:
    s_s.append(lowess(adata_r_ipc[:,gene].layers["s_raw"].toarray().reshape(-1), pseudotime, frac=frac, return_sorted=False))
    u_s.append(lowess(adata_r_ipc[:,gene].layers["u_raw"].toarray().reshape(-1), pseudotime, frac=frac, return_sorted=False))
    p_s.append(lowess(adata_r_ipc[:,gene].layers["p_raw"].toarray().reshape(-1), pseudotime, frac=frac, return_sorted=False))
    ds_s.append(lowess(adata_r_ipc[:,gene].layers["dsdt"].toarray().reshape(-1), pseudotime, frac=frac, return_sorted=False))
    du_s.append(lowess(adata_r_ipc[:,gene].layers["dudt"].toarray().reshape(-1), pseudotime, frac=frac, return_sorted=False))
    dp_s.append(lowess(adata_r_ipc[:,gene].layers["dpdt"].toarray().reshape(-1), pseudotime, frac=frac, return_sorted=False))

# lowess values are sorted according to pseudotime...

adata_r_ipc.layers["s_s"] = np.array(s_s).T
adata_r_ipc.layers["u_s"] = np.array(u_s).T
adata_r_ipc.layers["p_s"] = np.array(p_s).T
adata_r_ipc.layers["ds_s"] = np.array(ds_s).T
adata_r_ipc.layers["du_s"] = np.array(du_s).T
adata_r_ipc.layers["dp_s"] = np.array(dp_s).T

# assign discrete cluster to IPC cells, with pseudoimte bin
# for tl rank_gene_groups
bin_size = 28 # 279 cells in IPC
adata_r_ipc.obs["pseudoimte_bins"] = None
for i in range(10):
    if i==9:
        cell_in_bin = np.argsort(adata_r_ipc.obs.pseudotime)[(bin_size*i) :]
    else:
        cell_in_bin = np.argsort(adata_r_ipc.obs.pseudotime)[(bin_size*i) : (bin_size * (i+1))]
    adata_r_ipc.obs["pseudoimte_bins"][cell_in_bin] = "p_bin_{}".format(i)
adata_r_ipc.obs["pseudoimte_bins"] = adata_r_ipc.obs["pseudoimte_bins"].astype("category")

sc.tl.rank_genes_groups(adata_r_ipc, groupby="pseudoimte_bins", 
                        groups=["p_bin_0", "p_bin_9"], method="wilcoxon")


dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/time_lag/IPC"
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
sc.pl.rank_genes_groups_violin(adata_r_ipc, groups="p_bin_0", n_genes=10)
fig.tight_layout()
plt.savefig(dir_path + "/rank_genes_groups_bin_0.png", bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
sc.pl.rank_genes_groups_violin(adata_r_ipc, groups="p_bin_9", n_genes=10)
fig.tight_layout()
plt.savefig(dir_path + "/rank_genes_groups_bin_9.png", bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
sc.pl.rank_genes_groups(adata_r_ipc, n_genes=25, sharey=False)
fig.tight_layout()
plt.savefig(dir_path + "/rank_genes_groups.png", bbox_inches='tight')
plt.close()

groups = adata_r_ipc.uns["rank_genes_groups"]["names"].dtype.names
df_pval = pd.DataFrame(
    {group + '_' + key[:1]: adata_r_ipc.uns["rank_genes_groups"][key][group]
    for group in groups for key in ['names', 'pvals']}) # pvals_adj

# analyse genes with non-adjusted pvalue <= 0.05
(df_pval["p_bin_0_p"] <= 0.05).sum() # 352
(df_pval["p_bin_9_p"] <= 0.05).sum() # 275
adata_r_ipc.var_names[((df_pval["p_bin_0_p"] <= 0.05) | (df_pval["p_bin_9_p"] <= 0.05))]

# genes with adjusted pvalue
gene_idx = (prom_gene & ((df_pval["p_bin_0_p"] <= 0.05) | (df_pval["p_bin_9_p"] <= 0.05))) # 175 genes
analysis_gene = adata_r_ipc.var_names[gene_idx]



# check the pattern along pseudotime
dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/time_lag/IPC/gene"
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

for gene in analysis_gene:
    s_raws = adata_r_ipc[:, gene].layers["s_raw"].toarray().reshape(-1)
    u_raws = adata_r_ipc[:, gene].layers["u_raw"].toarray().reshape(-1)
    p_raws = adata_r_ipc[:, gene].layers["p_raw"].toarray().reshape(-1)
    s_s = adata_r_ipc[:, gene].layers["s_s"].toarray().reshape(-1)
    u_s = adata_r_ipc[:, gene].layers["u_s"].toarray().reshape(-1)
    p_s = adata_r_ipc[:, gene].layers["p_s"].toarray().reshape(-1)
    s_s = s_s / np.abs(s_raws).max()
    s_raw = s_raws / np.abs(s_raws).max()
    u_s = u_s / np.abs(u_raws).max()
    u_raw = u_raws / np.abs(u_raws).max()
    p_s = p_s / np.abs(p_raws).max()
    p_raw = p_raws / np.abs(p_raws).max()

    dsdt = adata_r_ipc[:, gene].layers["dsdt"].toarray().reshape(-1)
    dudt = adata_r_ipc[:, gene].layers["dudt"].toarray().reshape(-1)
    dpdt = adata_r_ipc[:, gene].layers["dpdt"].toarray().reshape(-1)
    ds_s = adata_r_ipc[:, gene].layers["ds_s"].toarray().reshape(-1)
    du_s = adata_r_ipc[:, gene].layers["du_s"].toarray().reshape(-1)
    dp_s = adata_r_ipc[:, gene].layers["dp_s"].toarray().reshape(-1)
    ds_s = ds_s / np.abs(s_raws).max()
    dsdt = dsdt / np.abs(s_raws).max()
    du_s = du_s / np.abs(u_raws).max()
    dudt = dudt / np.abs(u_raws).max()
    dp_s = dp_s / np.abs(p_raws).max()
    dpdt = dpdt / np.abs(p_raws).max()
    pseudotime = adata_r_ipc.obs["pseudotime"]
    equal_time_bin = [i/len(pseudotime) for i in range(len(pseudotime))]

    fig, axes = plt.subplots(1, 2, figsize=(5*2, 5))
    xs, ys = zip(*sorted(zip(pseudotime, s_raw)))
    axes[0].scatter(x=equal_time_bin, y=ys, s=5, color="blue", alpha=0.3)
    xs, ys = zip(*sorted(zip(pseudotime, s_s)))
    axes[0].plot(equal_time_bin, ys, color="blue", label="spliced")
    xs, ys = zip(*sorted(zip(pseudotime, u_raw)))
    axes[0].scatter(x=equal_time_bin, y=ys, s=5, color="red", alpha=0.3)
    xs, ys = zip(*sorted(zip(pseudotime, u_s)))
    axes[0].plot(equal_time_bin, ys, color="red", label="unspliced")
    xs, ys = zip(*sorted(zip(pseudotime, p_raw)))
    axes[0].scatter(x=equal_time_bin, y=ys, s=5, color="black", alpha=0.3)
    xs, ys = zip(*sorted(zip(pseudotime, p_s)))
    axes[0].plot(equal_time_bin, ys, color="black", label="promoter")
    axes[0].legend()
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("pseudotime")
    axes[0].set_ylabel("count")
    axes[0].set_title("{} count".format(gene))

    xs, ys = zip(*sorted(zip(pseudotime, dsdt)))
    axes[1].scatter(x=equal_time_bin, y=ys, s=5, color="blue", alpha=0.3)
    xs, ys = zip(*sorted(zip(pseudotime, ds_s)))
    axes[1].plot(equal_time_bin, ys, color="blue", label="dsdt")
    xs, ys = zip(*sorted(zip(pseudotime, dudt)))
    axes[1].scatter(x=equal_time_bin, y=ys, s=5, color="red", alpha=0.3)
    xs, ys = zip(*sorted(zip(pseudotime, du_s)))
    axes[1].plot(equal_time_bin, ys, color="red", label="dudt")
    xs, ys = zip(*sorted(zip(pseudotime, dpdt)))
    axes[1].scatter(x=equal_time_bin, y=ys, s=5, color="black", alpha=0.3)
    xs, ys = zip(*sorted(zip(pseudotime, dp_s)))
    axes[1].plot(equal_time_bin, ys, color="black", label="dpdt")
    axes[1].legend()
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("pseudotime")
    axes[1].set_ylabel("velocity")
    axes[1].set_title("{} velocity".format(gene))
    fig.tight_layout()
    plt.savefig(dir_path + "/time_change_{}_smooth.png".format(gene), bbox_inches='tight')
    plt.close()
    break
