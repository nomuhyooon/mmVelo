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


# restrict to genes with promoters in our dataset
prom_gene = peak2prom_mat.sum(0) > 0

# apply lowess regression, to corporate with variation orthogonal to pseudotime axis
lowess = sm.nonparametric.lowess
frac = 100 / adata_r.n_obs # use 100 neighbor cells for lowess regression
pseudotime = adata_r.obs["pseudotime"]

s_s, u_s, p_s = [], [], []
ds_s, du_s, dp_s = [], [], []

for gene in adata_r.var_names:
    s_s.append(lowess(adata_r[:,gene].layers["s_raw"].toarray().reshape(-1), pseudotime, frac=frac, return_sorted=False))
    u_s.append(lowess(adata_r[:,gene].layers["u_raw"].toarray().reshape(-1), pseudotime, frac=frac, return_sorted=False))
    p_s.append(lowess(adata_r[:,gene].layers["p_raw"].toarray().reshape(-1), pseudotime, frac=frac, return_sorted=False))
    ds_s.append(lowess(adata_r[:,gene].layers["dsdt"].toarray().reshape(-1), pseudotime, frac=frac, return_sorted=False))
    du_s.append(lowess(adata_r[:,gene].layers["dudt"].toarray().reshape(-1), pseudotime, frac=frac, return_sorted=False))
    dp_s.append(lowess(adata_r[:,gene].layers["dpdt"].toarray().reshape(-1), pseudotime, frac=frac, return_sorted=False))

# lowess values are sorted according to pseudotime...

adata_r.layers["s_s"] = np.array(s_s).T
adata_r.layers["u_s"] = np.array(u_s).T
adata_r.layers["p_s"] = np.array(p_s).T
adata_r.layers["ds_s"] = np.array(ds_s).T
adata_r.layers["du_s"] = np.array(du_s).T
adata_r.layers["dp_s"] = np.array(dp_s).T


adata_r_exn = adata_r[adata_r.obs["pseudotime"] >= 0, :]

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

# scale to unit variance, to exclude the effect of gene expr scale
adata_r_exn_scale = adata_r_exn.copy()
sc.pp.scale(adata_r_exn_scale, layer="Ms")
sc.pp.scale(adata_r_exn_scale, layer="Mu")
sc.pp.scale(adata_r_exn_scale, layer="M_p")

# violin plot
# for each bin, extract 300 genes with highest variances
df_ss = pd.DataFrame()
df_us = pd.DataFrame()
df_ps = pd.DataFrame()
for i in range(10):
    #if i == 9:
    #    cell_in_bin = ordered_idx[(bin_size * i) : ]
    #else:
    #    cell_in_bin = ordered_idx[(bin_size * i) : (bin_size * (i+1))]
    if i == 9:
        break
    else:
        cell_in_bin = ordered_idx[(len(cell_in_bin)//2 + (bin_size * i)) : (len(cell_in_bin)//2 + (bin_size * (i+1)))]
    
    print(adata_r_exn_scale[cell_in_bin[0], :].obs["pseudotime"].item(), adata_r_exn_scale[cell_in_bin[-1], :].obs["pseudotime"].item())
    adata_sub = adata_r_exn_scale[cell_in_bin, :]
    #df_s = pd.DataFrame(np.var(adata_sub.layers["Ms"], axis=0), columns=["variance"])
    #df_u = pd.DataFrame(np.var(adata_sub.layers["Mu"], axis=0), columns=["variance"])
    #df_p = pd.DataFrame(np.var(adata_sub.layers["M_p"], axis=0), columns=["variance"])
    df_s = pd.DataFrame(np.sort(np.var(adata_sub.layers["Ms"], axis=0))[-300:], columns=["variance"])
    df_u = pd.DataFrame(np.sort(np.var(adata_sub.layers["Mu"], axis=0))[-300:], columns=["variance"])
    df_p = pd.DataFrame(np.sort(np.var(adata_sub.layers["M_p"], axis=0))[-300:], columns=["variance"])
    bin_name = str(adata_r_exn_scale[cell_in_bin[0], :].obs["pseudotime"].item()) + "-" + str(adata_r_exn_scale[cell_in_bin[-1], :].obs["pseudotime"].item())
    df_s["bin"] = bin_name
    df_u["bin"] = bin_name
    df_p["bin"] = bin_name
    df_ss = pd.concat([df_ss, df_s], axis=0)
    df_us = pd.concat([df_us, df_u], axis=0)
    df_ps = pd.concat([df_ps, df_p], axis=0)
df_ss["modality"] = "spliced"
df_us["modality"] = "unspliced"
df_ps["modality"] = "promoter"
df = pd.concat([df_ss, df_us, df_ps], axis=0)

dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/time_lag"
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
sns.violinplot(data=df, x="bin", y="variance", hue="modality")
ax.set_xlabel("pseudoitme bin")
ax.set_ylabel("variance")
ax.legend()
fig.tight_layout()
plt.savefig(dir_path + "/pseudotime_variance_violin_half.png", bbox_inches='tight')
plt.close()


dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/time_lag/bin_distribution"
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

# plot each bin populations on UMAP
for i in range(10):
    if i == 9:
        cell_in_bin = ordered_idx[(bin_size * i) : ]
    else:
        cell_in_bin = ordered_idx[(bin_size * i) : (bin_size * (i+1))]
    print(adata_r_exn_scale[cell_in_bin[0], :].obs["pseudotime"].item(), adata_r_exn_scale[cell_in_bin[-1], :].obs["pseudotime"].item())
    adata_sub = adata_r_exn_scale[cell_in_bin, :]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    ax.scatter(adata_r_exn_scale.obsm["X_umap"][:,0], adata_r_exn_scale.obsm["X_umap"][:,1],
               s=20, color="grey", alpha=0.2)
    cbar = ax.scatter(adata_sub.obsm["X_umap"][:,0], adata_sub.obsm["X_umap"][:,1],
               s=20, c=adata_sub.obs["pseudotime"].to_numpy(), alpha=0.5, vmin=0, vmax=1)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 1")
    ax.set_title("pseudotime bin {}".format(i))
    fig.colorbar(cbar, ax=ax)
    fig.tight_layout()
    plt.savefig(dir_path + "/umap_bin_distribution_{}.png".format(i), bbox_inches='tight')
    plt.close()
    break