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


# restrict to genes with promoters in our dataset
prom_gene = peak2prom_mat.sum(0) > 0

# apply lowess regression, to corporate with variation orthogonal to pseudotime axis
lowess = sm.nonparametric.lowess
frac = 200 / adata_r.n_obs # use 100 neighbor cells for lowess regression
pseudotime = adata_r.obs["pseudotime"]
pseudotime_uni = np.zeros(adata_r.n_obs)
for i, idx in enumerate(np.argsort(pseudotime)):
    pseudotime_uni[idx] = i / adata_r.n_obs

np.argsort(pseudotime)
np.argsort(pseudotime_uni)

 

s_s, u_s, p_s = [], [], []
ds_s, du_s, dp_s = [], [], []

for gene in adata_r.var_names:
    s_s.append(lowess(adata_r[:,gene].layers["s_raw"].toarray().reshape(-1), pseudotime_uni, frac=frac, return_sorted=False))
    u_s.append(lowess(adata_r[:,gene].layers["u_raw"].toarray().reshape(-1), pseudotime_uni, frac=frac, return_sorted=False))
    p_s.append(lowess(adata_r[:,gene].layers["p_raw"].toarray().reshape(-1), pseudotime_uni, frac=frac, return_sorted=False))
    ds_s.append(lowess(adata_r[:,gene].layers["dsdt"].toarray().reshape(-1), pseudotime_uni, frac=frac, return_sorted=False))
    du_s.append(lowess(adata_r[:,gene].layers["dudt"].toarray().reshape(-1), pseudotime_uni, frac=frac, return_sorted=False))
    dp_s.append(lowess(adata_r[:,gene].layers["dpdt"].toarray().reshape(-1), pseudotime_uni, frac=frac, return_sorted=False))

# lowess values are sorted according to pseudotime...

adata_r.layers["s_s"] = np.array(s_s).T
adata_r.layers["u_s"] = np.array(u_s).T
adata_r.layers["p_s"] = np.array(p_s).T
adata_r.layers["ds_s"] = np.array(ds_s).T
adata_r.layers["du_s"] = np.array(du_s).T
adata_r.layers["dp_s"] = np.array(dp_s).T

p_zero_time = np.where((adata_r.obs["pseudotime"]==0))[0]
p_zero_time = pseudotime_uni[p_zero_time]

# restrict to cells with pseudoime >= 0, <= 0.8
# this corresponds to ExN lineage

adata_r.obs["pseudotime_uni"] = pseudotime_uni #
adata_r_exn = adata_r[(adata_r.obs["pseudotime"] >= 0) & (adata_r.obs["pseudotime"] <= 0.8), :]

# restrict to genes with max expression in ExN lineage
peak_in_exn_gene = []
for i in range(adata_r.n_vars):
    #s_expr_t = adata_r.obs["pseudotime"][np.argmax(adata_r.layers["s_s"], 0)[i]]
    #u_expr_t = adata_r.obs["pseudotime"][np.argmax(adata_r.layers["u_s"], 0)[i]]
    #p_expr_t = adata_r.obs["pseudotime"][np.argmax(adata_r.layers["p_s"], 0)[i]]
    s_expr_t = pseudotime_uni[np.argmax(adata_r_exn.layers["s_s"], 0)[i]]
    u_expr_t = pseudotime_uni[np.argmax(adata_r_exn.layers["u_s"], 0)[i]]
    p_expr_t = pseudotime_uni[np.argmax(adata_r_exn.layers["p_s"], 0)[i]]
    gene_in = (s_expr_t >= p_zero_time) & (u_expr_t >= p_zero_time) & (p_expr_t >= p_zero_time)
    peak_in_exn_gene.append(gene_in.item())

gene_for_analysis = prom_gene & np.array(peak_in_exn_gene)
print(gene_for_analysis.sum()) # 329



# dpdt max pseudotime
max_cells = np.argmax(adata_r_exn.layers["dp_s"], axis=0)
dpdt_pseudotime = []
for i, idx in enumerate(max_cells):
    #dpdt_pseudotime.append(adata_r_exn.obs["pseudotime"][idx])
    dpdt_pseudotime.append(adata_r_exn.obs["pseudotime_uni"][idx])

# dudt max pseudotime
max_cells = np.argmax(adata_r_exn.layers["du_s"].toarray(), axis=0)
dudt_pseudotime = []
for i, idx in enumerate(max_cells):
    #dudt_pseudotime.append(adata_r_exn.obs["pseudotime"][idx])
    dudt_pseudotime.append(adata_r_exn.obs["pseudotime_uni"][idx])

# dsdt max pseudotime
max_cells = np.argmax(adata_r_exn.layers["ds_s"].toarray(), axis=0)
dsdt_pseudotime = []
for i, idx in enumerate(max_cells):
    #dsdt_pseudotime.append(adata_r_exn.obs["pseudotime"][idx])
    dsdt_pseudotime.append(adata_r_exn.obs["pseudotime_uni"][idx])

time_df = pd.DataFrame(columns=["gene", "dpdt", "dudt", "dsdt"])
for i in range(adata_r_exn.n_vars):
    gene_name = adata_r_exn.var_names[i]
    time_df.loc[i] = [gene_name, dpdt_pseudotime[i], dudt_pseudotime[i], dsdt_pseudotime[i]]

analysis_df = time_df[gene_for_analysis]

up_lag = analysis_df.dudt - analysis_df.dpdt
su_lag = analysis_df.dsdt - analysis_df.dudt
sp_lag = analysis_df.dsdt - analysis_df.dpdt

dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/time_lag"
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.hist(analysis_df.dpdt, bins=50, color="black", alpha=0.3, label="dpdt")
ax.hist(analysis_df.dudt, bins=50, color="red", alpha=0.3, label="dudt")
ax.hist(analysis_df.dsdt, bins=50, color="blue", alpha=0.3, label="dsdt")
#ax.set_xlim(0,1)
ax.set_xlabel("pseudotime")
ax.set_ylabel("freq")
ax.set_title("velocity maximum time distribution")
ax.legend()
fig.tight_layout()
plt.savefig(dir_path + "/argmax_time_distribution.png", bbox_inches='tight')
plt.close()

dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/time_lag"
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.hist(up_lag, bins=30)
ax.set_xlim(-1,1)
ax.set_xlabel("time lag")
ax.set_ylabel("freq")
ax.set_title("dudt - dpdt")
fig.tight_layout()
plt.savefig(dir_path + "/timelag_u_vs_p_smooth.png", bbox_inches='tight')
plt.close()

dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/time_lag"
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.hist(su_lag, bins=30)
ax.set_xlim(-1,1)
ax.set_xlabel("time lag")
ax.set_ylabel("freq")
ax.set_title("dsdt - dudt")
fig.tight_layout()
plt.savefig(dir_path + "/timelag_s_vs_u_smooth.png", bbox_inches='tight')
plt.close()

dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/time_lag"
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.hist(sp_lag, bins=30)
ax.set_xlim(-1,1)
ax.set_xlabel("time lag")
ax.set_ylabel("freq")
ax.set_title("dsdt - dpdt")
fig.tight_layout()
plt.savefig(dir_path + "/timelag_s_vs_p_smooth.png", bbox_inches='tight')
plt.close()

(up_lag >= 0).sum()
(su_lag >= 0).sum()
(sp_lag >= 0).sum()
len(up_lag)


# check the pseudotime
dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/time_lag/gene"
os.mkdir(dir_path)

for gene in analysis_df.gene:
    s_raws = adata_r_exn[:, gene].layers["s_raw"].toarray().reshape(-1)
    u_raws = adata_r_exn[:, gene].layers["u_raw"].toarray().reshape(-1)
    p_raws = adata_r_exn[:, gene].layers["p_raw"].toarray().reshape(-1)
    s_s = adata_r_exn[:, gene].layers["s_s"].toarray().reshape(-1)
    u_s = adata_r_exn[:, gene].layers["u_s"].toarray().reshape(-1)
    p_s = adata_r_exn[:, gene].layers["p_s"].toarray().reshape(-1)
    s_s = s_s / np.abs(s_raws).max()
    s_raw = s_raws / np.abs(s_raws).max()
    u_s = u_s / np.abs(u_raws).max()
    u_raw = u_raws / np.abs(u_raws).max()
    p_s = p_s / np.abs(p_raws).max()
    p_raw = p_raws / np.abs(p_raws).max()

    dsdt = adata_r_exn[:, gene].layers["dsdt"].toarray().reshape(-1)
    dudt = adata_r_exn[:, gene].layers["dudt"].toarray().reshape(-1)
    dpdt = adata_r_exn[:, gene].layers["dpdt"].toarray().reshape(-1)
    ds_s = adata_r_exn[:, gene].layers["ds_s"].toarray().reshape(-1)
    du_s = adata_r_exn[:, gene].layers["du_s"].toarray().reshape(-1)
    dp_s = adata_r_exn[:, gene].layers["dp_s"].toarray().reshape(-1)
    ds_s = ds_s / np.abs(s_raws).max()
    dsdt = dsdt / np.abs(s_raws).max()
    du_s = du_s / np.abs(u_raws).max()
    dudt = dudt / np.abs(u_raws).max()
    dp_s = dp_s / np.abs(p_raws).max()
    dpdt = dpdt / np.abs(p_raws).max()
    #pseudotime = adata_r_exn.obs["pseudotime"]
    pseudotime = adata_r_exn.obs["pseudotime_uni"]

    fig, axes = plt.subplots(1, 2, figsize=(5*2, 5))
    axes[0].scatter(x=pseudotime, y=s_raw, s=2, color="blue", alpha=0.1)
    xs, ys = zip(*sorted(zip(pseudotime, s_s)))
    axes[0].plot(xs, ys, color="blue", label="spliced")
    axes[0].scatter(x=pseudotime, y=u_raw, s=2, color="red", alpha=0.1)
    xs, ys = zip(*sorted(zip(pseudotime, u_s)))
    axes[0].plot(xs, ys, color="red", label="unspliced")
    axes[0].scatter(x=pseudotime, y=p_raw, s=2, color="black", alpha=0.1)
    xs, ys = zip(*sorted(zip(pseudotime, p_s)))
    axes[0].plot(xs, ys, color="black", label="promoter")
    axes[0].legend()
    axes[0].set_xlim(0,1)
    axes[0].set_xlabel("pseudotime")
    axes[0].set_ylabel("count")
    axes[0].set_title("{} count".format(gene))

    axes[1].scatter(x=pseudotime, y=dsdt, s=2, color="blue", alpha=0.1)
    xs, ys = zip(*sorted(zip(pseudotime, ds_s)))
    axes[1].plot(xs, ys, color="blue", label="dsdt")
    axes[1].scatter(x=pseudotime, y=dudt, s=2, color="red", alpha=0.1)
    xs, ys = zip(*sorted(zip(pseudotime, du_s)))
    axes[1].plot(xs, ys, color="red", label="dudt")
    axes[1].scatter(x=pseudotime, y=dpdt, s=2, color="black", alpha=0.1)
    xs, ys = zip(*sorted(zip(pseudotime, dp_s)))
    axes[1].plot(xs, ys, color="black", label="dpdt")
    axes[1].legend()
    axes[1].set_xlim(0,1)
    axes[1].set_xlabel("pseudotime")
    axes[1].set_ylabel("velocity")
    axes[1].set_title("{} velocity".format(gene))
    fig.tight_layout()
    plt.savefig(dir_path + "/time_change_{}_smooth.png".format(gene), bbox_inches='tight')
    plt.close()


file_path = "/home/nomura/Proj/mmvelo/data/10x_multiome_brain_repre_wo_IN/analysis/dorc_gene.tsv"
dorc_genes = pd.read_csv(file_path, header=None)[0].to_numpy()
len(dorc_genes) # 223
len(list(set(analysis_df.gene) & set(dorc_genes))) # 44
dorc_genes = list(set(analysis_df.gene) & set(dorc_genes))

dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/time_lag/dorc_gene"
os.mkdir(dir_path)

for gene in dorc_genes:
    s_raws = adata_r_exn[:, gene].layers["s_raw"].toarray().reshape(-1)
    u_raws = adata_r_exn[:, gene].layers["u_raw"].toarray().reshape(-1)
    p_raws = adata_r_exn[:, gene].layers["p_raw"].toarray().reshape(-1)
    s_s = adata_r_exn[:, gene].layers["s_s"].toarray().reshape(-1)
    u_s = adata_r_exn[:, gene].layers["u_s"].toarray().reshape(-1)
    p_s = adata_r_exn[:, gene].layers["p_s"].toarray().reshape(-1)
    s_s = s_s / np.abs(s_raws).max()
    s_raw = s_raws / np.abs(s_raws).max()
    u_s = u_s / np.abs(u_raws).max()
    u_raw = u_raws / np.abs(u_raws).max()
    p_s = p_s / np.abs(p_raws).max()
    p_raw = p_raws / np.abs(p_raws).max()

    dsdt = adata_r_exn[:, gene].layers["dsdt"].toarray().reshape(-1)
    dudt = adata_r_exn[:, gene].layers["dudt"].toarray().reshape(-1)
    dpdt = adata_r_exn[:, gene].layers["dpdt"].toarray().reshape(-1)
    ds_s = adata_r_exn[:, gene].layers["ds_s"].toarray().reshape(-1)
    du_s = adata_r_exn[:, gene].layers["du_s"].toarray().reshape(-1)
    dp_s = adata_r_exn[:, gene].layers["dp_s"].toarray().reshape(-1)
    ds_s = ds_s / np.abs(s_raws).max()
    dsdt = dsdt / np.abs(s_raws).max()
    du_s = du_s / np.abs(u_raws).max()
    dudt = dudt / np.abs(u_raws).max()
    dp_s = dp_s / np.abs(p_raws).max()
    dpdt = dpdt / np.abs(p_raws).max()
    #pseudotime = adata_r_exn.obs["pseudotime"]
    pseudotime = adata_r_exn.obs["pseudotime_uni"]

    fig, axes = plt.subplots(1, 2, figsize=(5*2, 5))
    axes[0].scatter(x=pseudotime, y=s_raw, s=2, color="blue", alpha=0.1)
    xs, ys = zip(*sorted(zip(pseudotime, s_s)))
    axes[0].plot(xs, ys, color="blue", label="spliced")
    axes[0].scatter(x=pseudotime, y=u_raw, s=2, color="red", alpha=0.1)
    xs, ys = zip(*sorted(zip(pseudotime, u_s)))
    axes[0].plot(xs, ys, color="red", label="unspliced")
    axes[0].scatter(x=pseudotime, y=p_raw, s=2, color="black", alpha=0.1)
    xs, ys = zip(*sorted(zip(pseudotime, p_s)))
    axes[0].plot(xs, ys, color="black", label="promoter")
    axes[0].legend()
    axes[0].set_xlim(0,1)
    axes[0].set_xlabel("pseudotime")
    axes[0].set_ylabel("count")
    axes[0].set_title("{} count".format(gene))

    axes[1].scatter(x=pseudotime, y=dsdt, s=2, color="blue", alpha=0.1)
    xs, ys = zip(*sorted(zip(pseudotime, ds_s)))
    axes[1].plot(xs, ys, color="blue", label="dsdt")
    axes[1].scatter(x=pseudotime, y=dudt, s=2, color="red", alpha=0.1)
    xs, ys = zip(*sorted(zip(pseudotime, du_s)))
    axes[1].plot(xs, ys, color="red", label="dudt")
    axes[1].scatter(x=pseudotime, y=dpdt, s=2, color="black", alpha=0.1)
    xs, ys = zip(*sorted(zip(pseudotime, dp_s)))
    axes[1].plot(xs, ys, color="black", label="dpdt")
    axes[1].legend()
    axes[1].set_xlim(0,1)
    axes[1].set_xlabel("pseudotime")
    axes[1].set_ylabel("velocity")
    axes[1].set_title("{} velocity".format(gene))
    fig.tight_layout()
    plt.savefig(dir_path + "/time_change_{}_smooth.png".format(gene), bbox_inches='tight')
    plt.close()

# peak-gene linkage matrix
peak_gene_linkage.shape # 25071, 3072

dorc_genes_idx = []
for gene in dorc_genes:
    dorc_genes_idx.append(np.where(gene == adata_r_exn.var_names)[0].item())

dorc_peaks_idx = []
for gene_idx in dorc_genes_idx:
    dorc_peaks_idx.append(list(np.where(peak_gene_linkage[:, gene_idx] > 0)[0]))
len(dorc_peaks_idx)

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
adata_a.obs["pseudotime_uni"] = adata_r.obs["pseudotime_uni"]
lowess = sm.nonparametric.lowess
frac = 200 / adata_r.n_obs # use 100 neighbor cells for lowess regression

a_s, da_s = [], []

for idx in peak_idx:
    a_s.append(lowess(adata_a[:,idx].layers["a_raw"].toarray().reshape(-1), pseudotime_uni, frac=frac, return_sorted=False))
    da_s.append(lowess(adata_a[:,idx].layers["dadt"].toarray().reshape(-1), pseudotime_uni, frac=frac, return_sorted=False))

a_s[0].shape

p_t_min = adata_r_exn.obs["pseudotime_uni"].min()
p_t_max = adata_r_exn.obs["pseudotime_uni"].max()

dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/time_lag/dorc_gene_peaks"
os.mkdir(dir_path)

for i, idx in enumerate(peak_idx):
    peak_name = adata_a.var_names[idx]
    peak_dist = peak_tss_dist[i]
    a_raws = adata_a[:, peak_name].layers["a_raw"].toarray().reshape(-1)
    dadt = adata_a[:, peak_name].layers["dadt"].toarray().reshape(-1)
    a_raws_s = a_s[i]
    dadt_s = da_s[i]
    
    a_raws_s = a_raws_s / np.abs(a_raws).max()
    a_raws = a_raws / np.abs(a_raws).max()
    dadt_s = dadt_s / np.abs(dadt).max()
    dadt = dadt / np.abs(dadt).max()

    pseudotime = adata_a.obs["pseudotime_uni"]

    fig, axes = plt.subplots(1, 2, figsize=(5*2, 5))
    axes[0].scatter(x=pseudotime, y=a_raws, s=2, color="black", alpha=0.1)
    xs, ys = zip(*sorted(zip(pseudotime, a_raws_s)))
    axes[0].plot(xs, ys, color="black", label="distance = {}".format(peak_dist))
    axes[0].legend()
    axes[0].set_xlim(p_t_min, p_t_max)
    axes[0].set_xlabel("pseudotime")
    axes[0].set_ylabel("norm rec count")
    axes[0].set_title("{} count".format(peak_name))

    axes[1].scatter(x=pseudotime, y=dadt, s=2, color="black", alpha=0.1)
    xs, ys = zip(*sorted(zip(pseudotime, dadt_s)))
    axes[1].plot(xs, ys, color="black", label="distance = {}".format(peak_dist))
    axes[1].legend()
    axes[1].set_xlim(p_t_min, p_t_max)
    axes[1].set_xlabel("pseudotime")
    axes[1].set_ylabel("norm rec vel")
    axes[1].set_title("{} velocity".format(peak_name))

    fig.tight_layout()
    plt.savefig(dir_path + "/time_change_{}_smooth.png".format(peak_name), bbox_inches='tight')
    plt.close()


# stratify distance >= 10,000, <10,000
fig, axes = plt.subplots(1, 2, figsize=(5*2, 5))
for i, idx in enumerate(peak_idx):
    peak_name = adata_a.var_names[idx]
    peak_dist = peak_tss_dist[i]
    a_raws = adata_a[:, peak_name].layers["a_raw"].toarray().reshape(-1)
    dadt = adata_a[:, peak_name].layers["dadt"].toarray().reshape(-1)
    a_raws_s = a_s[i]
    dadt_s = da_s[i]

    norm_const = np.abs(a_raws_s).max()
    
    a_raws_s = a_raws_s / norm_const
    dadt_s = dadt_s / norm_const

    pseudotime = adata_a.obs["pseudotime_uni"]

    xs, ys = zip(*sorted(zip(pseudotime, a_raws_s)))
    if np.abs(peak_dist) < 10000:
        axes[0].plot(xs, ys, color="blue", label="distance < 10k")
    else:
        axes[0].plot(xs, ys, color="red", label="distance >= 10k")
    axes[0].set_xlim(p_t_min, p_t_max)
    axes[0].set_xlabel("pseudotime")
    axes[0].set_ylabel("norm rec count")
    axes[0].set_title("peak count")

    xs, ys = zip(*sorted(zip(pseudotime, dadt_s)))
    if np.abs(peak_dist) < 10000:
        axes[1].plot(xs, ys, color="blue", label="distance < 10k")
    else:
        axes[1].plot(xs, ys, color="red", label="distance >= 10k")
    
    axes[1].set_xlim(p_t_min, p_t_max)
    axes[1].set_xlabel("pseudotime")
    axes[1].set_ylabel("norm rec vel")
    axes[1].set_title("peak velocity")
#axes[0].legend()
#axes[1].legend()
fig.tight_layout()
plt.savefig(dir_path + "/time_change_stratified_by_dist.png", bbox_inches='tight')
plt.close()

# dist 100000
fig, axes = plt.subplots(1, 2, figsize=(5*2, 5))
for i, idx in enumerate(peak_idx):
    peak_name = adata_a.var_names[idx]
    peak_dist = peak_tss_dist[i]
    a_raws = adata_a[:, peak_name].layers["a_raw"].toarray().reshape(-1)
    dadt = adata_a[:, peak_name].layers["dadt"].toarray().reshape(-1)
    a_raws_s = a_s[i]
    dadt_s = da_s[i]
    
    norm_const = np.abs(a_raws_s).max()
    
    a_raws_s = a_raws_s / norm_const
    dadt_s = dadt_s / norm_const

    pseudotime = adata_a.obs["pseudotime_uni"]

    xs, ys = zip(*sorted(zip(pseudotime, a_raws_s)))
    if np.abs(peak_dist) < 100000:
        axes[0].plot(xs, ys, color="blue", label="distance < 100k")
    else:
        axes[0].plot(xs, ys, color="red", label="distance >= 100k")
    axes[0].set_xlim(p_t_min, p_t_max)
    axes[0].set_xlabel("pseudotime")
    axes[0].set_ylabel("norm rec count")
    axes[0].set_title("peak count")

    xs, ys = zip(*sorted(zip(pseudotime, dadt_s)))
    if np.abs(peak_dist) < 100000:
        axes[1].plot(xs, ys, color="blue", label="distance < 100k")
    else:
        axes[1].plot(xs, ys, color="red", label="distance >= 100k")
    
    axes[1].set_xlim(p_t_min, p_t_max)
    axes[1].set_xlabel("pseudotime")
    axes[1].set_ylabel("norm rec vel")
    axes[1].set_title("peak velocity")
#axes[0].legend()
#axes[1].legend()
fig.tight_layout()
plt.savefig(dir_path + "/time_change_stratified_by_dist_100k.png", bbox_inches='tight')
plt.close()


dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/time_lag/heatmap"
if not os.path.exists(dir_path):
    os.mkdir(dir_path)
adata_r
#var_names = ["Neurod2", "Adamts2", "Dlg2", "Igfbp3"]

adata_r.var["max_pseudotime"] = adata_r.obs["pseudotime"][np.array(adata_r.layers["unspliced"].argmax(axis=0)).reshape(-1)].to_numpy()
dorc_genes = list(adata_r[:, dorc_genes].var.sort_values("max_pseudotime").index)
var_names = dorc_genes

fig, axes = plt.subplots(1, 1, figsize=(5*1, 5))
sc.pl.heatmap(adata_r, var_names=var_names, groupby="pseudotime", cmap="viridis", swap_axes=True,
              layer="s_raw", standard_scale="var")
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_dorc_s.png", bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 1, figsize=(5*1, 5))
sc.pl.heatmap(adata_r, var_names=var_names, groupby="pseudotime", cmap="viridis", swap_axes=True,
              layer="u_raw", standard_scale="var")
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_dorc_u.png", bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 1, figsize=(5*1, 5))
sc.pl.heatmap(adata_r, var_names=var_names, groupby="pseudotime", cmap="viridis", swap_axes=True,
              layer="p_raw", standard_scale="var")
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_dorc_p.png", bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 1, figsize=(5*1, 5))
sc.pl.heatmap(adata_r, var_names=var_names, groupby="pseudotime", cmap="viridis", swap_axes=True,
              layer="M_p", standard_scale="var",)
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_dorc_mp.png", bbox_inches='tight')
plt.close()

adata_r.layers["dsdt_norm"] = adata_r.layers["dsdt"] / adata_r.layers["s_raw"].max(axis=0).toarray()
adata_r.layers["dudt_norm"] = adata_r.layers["dudt"] / adata_r.layers["u_raw"].max(axis=0).toarray()
adata_r.layers["dpdt_norm"] = adata_r.layers["dpdt"] / adata_r.layers["p_raw"].max(axis=0)

fig, axes = plt.subplots(1, 1, figsize=(5*1, 5))
sc.pl.heatmap(adata_r, var_names=var_names, groupby="pseudotime", cmap="coolwarm", swap_axes=True,
              layer="dsdt_norm",vcenter=0, vmin=-0.01, vmax=0.01)
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_dorc_ds.png", bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 1, figsize=(5*1, 5))
sc.pl.heatmap(adata_r, var_names=var_names, groupby="pseudotime", cmap="coolwarm", swap_axes=True,
              layer="dudt_norm",vcenter=0, vmin=-0.01, vmax=0.01)
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_dorc_du.png", bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 1, figsize=(5*1, 5))
sc.pl.heatmap(adata_r, var_names=var_names, groupby="pseudotime", cmap="coolwarm", swap_axes=True,
              layer="dpdt_norm",vcenter=0, vmin=-0.01, vmax=0.01)
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_dorc_dp.png", bbox_inches='tight')
plt.close()


# smoothed value
fig, axes = plt.subplots(1, 1, figsize=(5*1, 5))
sc.pl.heatmap(adata_r, var_names=var_names, groupby="pseudotime", cmap="viridis", swap_axes=True,
              layer="s_s", standard_scale="var", num_categories=100)
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_dorc_smoothed_s.png", bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 1, figsize=(5*1, 5))
sc.pl.heatmap(adata_r, var_names=var_names, groupby="pseudotime", cmap="viridis", swap_axes=True,
              layer="u_s", standard_scale="var", num_categories=100)
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_dorc_smoothed_u.png", bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 1, figsize=(5*1, 5))
sc.pl.heatmap(adata_r, var_names=var_names, groupby="pseudotime", cmap="viridis", swap_axes=True,
              layer="p_s", standard_scale="var", num_categories=100)
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_dorc_smoothed_p.png", bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 1, figsize=(5*1, 5))
sc.pl.heatmap(adata_r, var_names=var_names, groupby="pseudotime", cmap="viridis", swap_axes=True,
              layer="M_p", standard_scale="var", num_categories=100)
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_dorc_mp.png", bbox_inches='tight')
plt.close()

adata_r.layers["dsdt_s_norm"] = adata_r.layers["ds_s"] / adata_r.layers["s_s"].max(axis=0)
adata_r.layers["dudt_s_norm"] = adata_r.layers["du_s"] / adata_r.layers["u_s"].max(axis=0)
adata_r.layers["dpdt_s_norm"] = adata_r.layers["dp_s"] / adata_r.layers["p_s"].max(axis=0)

fig, axes = plt.subplots(1, 1, figsize=(5*1, 5))
sc.pl.heatmap(adata_r, var_names=var_names, groupby="pseudotime", cmap="coolwarm", swap_axes=True,
              layer="dsdt_s_norm", num_categories=100, vcenter=0)
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_dorc_smoothed_ds.png", bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 1, figsize=(5*1, 5))
sc.pl.heatmap(adata_r, var_names=var_names, groupby="pseudotime", cmap="coolwarm", swap_axes=True,
              layer="dudt_s_norm", num_categories=100, vcenter=0)
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_dorc_smoothed_du.png", bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, 1, figsize=(5*1, 5))
sc.pl.heatmap(adata_r, var_names=var_names, groupby="pseudotime", cmap="coolwarm", swap_axes=True,
              layer="dpdt_s_norm", num_categories=100, vcenter=0)
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_dorc_smoothed_dp.png", bbox_inches='tight')
plt.close()




"""
###########
# check the preprocess result
dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/time_lag/gene_preprocess"
os.mkdir(dir_path)
adata_r_exn.layers["Mp"] = adata_r[adata_r.obs["pseudotime"] >= 0, :].layers["M_p"]
adata_r_exn

for gene in analysis_df.gene:
    s_raws = adata_r_exn[:, gene].layers["Ms"].toarray().reshape(-1)
    u_raws = adata_r_exn[:, gene].layers["Mu"].toarray().reshape(-1)
    p_raws = adata_r_exn[:, gene].layers["Mp"].toarray().reshape(-1)

    s_raw = s_raws / np.abs(s_raws).max()
    u_raw = u_raws / np.abs(u_raws).max()
    p_raw = p_raws / np.abs(p_raws).max()
    pseudotime = adata_r_exn.obs["pseudotime"]
    fig, axes = plt.subplots(1, 2, figsize=(5*2, 5))
    axes[0].scatter(x=pseudotime, y=s_raw, s=2, color="blue", alpha=0.2, label="spliced")
    axes[0].scatter(x=pseudotime, y=u_raw, s=2, color="red", alpha=0.2, label="unspliced")
    axes[0].scatter(x=pseudotime, y=p_raw, s=2, color="black", alpha=0.2, label="promoter")
    axes[0].legend()
    axes[0].set_xlim(0,1)
    axes[0].set_xlabel("pseudotime")
    axes[0].set_ylabel("moment")
    axes[0].set_title("{} moment".format(gene))

    s_raws = adata_r_exn[:, gene].layers["s_raw"].toarray().reshape(-1)
    u_raws = adata_r_exn[:, gene].layers["u_raw"].toarray().reshape(-1)
    p_raws = adata_r_exn[:, gene].layers["p_raw"].toarray().reshape(-1)

    s_raw = s_raws / np.abs(s_raws).max()
    u_raw = u_raws / np.abs(u_raws).max()
    p_raw = p_raws / np.abs(p_raws).max()
    pseudotime = adata_r_exn.obs["pseudotime"]

    axes[1].scatter(x=pseudotime, y=s_raw, s=2, color="blue", alpha=0.2, label="spliced")
    axes[1].scatter(x=pseudotime, y=u_raw, s=2, color="red", alpha=0.2, label="unspliced")
    axes[1].scatter(x=pseudotime, y=p_raw, s=2, color="black", alpha=0.2, label="promoter")
    axes[1].legend()
    axes[1].set_xlim(0,1)
    axes[1].set_xlabel("pseudotime")
    axes[1].set_ylabel("reconstructed")
    axes[1].set_title("{} reconstructed".format(gene))
    fig.tight_layout()
    plt.savefig(dir_path + "/time_change_{}_momoent_reconstructed.png".format(gene), bbox_inches='tight')
    plt.close()
    break


# check the preprocess result: open vs close
dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/time_lag/open_close"
os.mkdir(dir_path)
adata_r_exn.layers["p_count"] = adata_r[adata_r.obs["pseudotime"] >= 0, :].layers["p_count"]
adata_r_exn.layers["p_op_cl"] =  np.array(adata_r_exn.layers["p_count"] > 0, np.float32)
adata_r_exn

counter = 0
for gene in analysis_df.gene:
    p_raws = adata_r_exn[:, gene].layers["p_op_cl"].toarray().reshape(-1)
    pseudotime = adata_r_exn.obs["pseudotime"]

    lowess = sm.nonparametric.lowess
    frac = 50 / adata_r_exn.n_obs
    p_raw = lowess(p_raws, pseudotime, frac=frac, return_sorted=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(5*2, 5))
    axes[0].scatter(x=pseudotime, y=p_raws, s=2, color="black", alpha=0.2)
    xs, ys = zip(*sorted(zip(pseudotime, p_raw)))
    axes[0].plot(xs, ys, color="blue")
    axes[0].set_xlim(0,1)
    axes[0].set_xlabel("pseudotime")
    axes[0].set_ylabel("open or close")
    axes[0].set_title("{} binary".format(gene))

    s_raws = adata_r_exn[:, gene].layers["s_raw"].toarray().reshape(-1)
    u_raws = adata_r_exn[:, gene].layers["u_raw"].toarray().reshape(-1)
    p_raws = adata_r_exn[:, gene].layers["p_raw"].toarray().reshape(-1)

    s_raw = s_raws / np.abs(s_raws).max()
    u_raw = u_raws / np.abs(u_raws).max()
    p_raw = p_raws / np.abs(p_raws).max()
    pseudotime = adata_r_exn.obs["pseudotime"]

    axes[1].scatter(x=pseudotime, y=s_raw, s=2, color="blue", alpha=0.2, label="spliced")
    axes[1].scatter(x=pseudotime, y=u_raw, s=2, color="red", alpha=0.2, label="unspliced")
    axes[1].scatter(x=pseudotime, y=p_raw, s=2, color="black", alpha=0.2, label="promoter")
    axes[1].legend()
    axes[1].set_xlim(0,1)
    axes[1].set_xlabel("pseudotime")
    axes[1].set_ylabel("reconstructed")
    axes[1].set_title("{} reconstructed".format(gene))
    fig.tight_layout()
    plt.savefig(dir_path + "/time_change_{}_momoent_reconstructed.png".format(gene), bbox_inches='tight')
    plt.close()
    counter += 1
    if counter == 50:
        break


# Calculate pseudotime confidence score
# The "wide" existence of cells orthogonal to differentiation trajectory worsens time-resolved analysis
# so, choose the pseuditime bin with high confidence
# the confindence is determined by the variance of moments within the pseudotime bin

adata_r_exn.obs["pseudotime"]

# check the distribution of number of cells
num_cells_in_bin = []
for i in range(10):
    time_min = i * 0.1
    time_max = (i+1) * 0.1
    cell_in_bin = (adata_r_exn.obs["pseudotime"] > time_min) & (adata_r_exn.obs["pseudotime"] < time_max)
    num_cells_in_bin.append(cell_in_bin.sum())
print(num_cells_in_bin) # hetero

# bin size 330. 330 = 3299 / 100 cells
np.sort(adata_r_exn.obs["pseudotime"]) # from 0 to 1
ordered_idx = np.argsort(adata_r_exn.obs["pseudotime"])
bin_size = 330
for i in range(10):
    if i == 9:
        cell_in_bin = ordered_idx[(bin_size * i) : ]
    else:
        cell_in_bin = ordered_idx[(bin_size * i) : (bin_size * (i+1))]
    print(adata_r_exn[cell_in_bin[0], :].obs["pseudotime"].item(), adata_r_exn[cell_in_bin[-1], :].obs["pseudotime"].item())
    break
"""
adata_r