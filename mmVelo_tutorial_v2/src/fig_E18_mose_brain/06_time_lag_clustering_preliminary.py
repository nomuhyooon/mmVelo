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

pseudotime = adata_r.obs["pseudotime"]
pseudotime_uni = np.zeros(adata_r.n_obs)
for i, idx in enumerate(np.argsort(pseudotime)):
    pseudotime_uni[idx] = i / adata_r.n_obs
np.argsort(pseudotime)
np.argsort(pseudotime_uni)

p_zero_time = np.where((adata_r.obs["pseudotime"]==0))[0]
p_zero_time = pseudotime_uni[p_zero_time]

# restrict to cells with pseudoime >= 0
# this corresponds to ExN lineage
adata_r.obs["pseudotime_uni"] = pseudotime_uni #
adata_r_exn = adata_r[(adata_r.obs["pseudotime"] >= 0)] # & (adata_r.obs["pseudotime"] <= 0.8), :]

adata_r.obs["pseudotime_uni"]

# clustering
s_raw = adata_r.layers["s_raw"].toarray()
var_name = adata_r.var_names.to_numpy()
corr_mat = np.corrcoef(s_raw, rowvar=False)
dist_mat = 1- corr_mat

from sklearn.cluster import AgglomerativeClustering

clustering = AgglomerativeClustering(n_clusters=10, affinity='precomputed', linkage='average')
clusters = clustering.fit_predict(dist_mat)

# plot heatmap
import seaborn as sns
import scipy
clusters
pseudotime_uni
sorted_s_raw = s_raw[:, np.argsort(clusters)]
sorted_var_name = var_name[np.argsort(clusters)]
sorted_s_raw = sorted_s_raw[np.argsort(pseudotime_uni), :]
sorted_s_raw = scipy.stats.zscore(sorted_s_raw)

dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/"
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
sns.set(font_scale=0.8)
#plt.figure(figsize=(10, 8))
sns.heatmap(sorted_s_raw, cmap='viridis', cbar=True, xticklabels=False, vmin=-3, vmax=3)
plt.xlabel('Cells')
plt.ylabel('Genes')
plt.title('Gene Expression Heatmap')
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_clustering_s.png", bbox_inches='tight')
plt.close()



# binning
pseudotime_uni
num_bins = 30
num_clusters = 3
percentiles = np.percentile(pseudotime_uni, np.linspace(0, 100, num_bins+1))

# compute mean expression in each bin
s_raw = adata_r.layers["s_raw"].toarray()
var_name = adata_r.var_names.to_numpy()
binned_data = []
for i in range(num_bins):
    bin_start = percentiles[i]
    bin_end = percentiles[i+1]
    bin_indices = np.where((pseudotime_uni >= bin_start) & (pseudotime_uni < bin_end))[0]
    bin_mean = np.mean(s_raw[bin_indices, :], axis=0)
    binned_data.append(bin_mean)

binned_s_raw = np.array(binned_data)
corr_mat = np.corrcoef(binned_s_raw, rowvar=False)
dist_mat = 1- corr_mat
clustering = AgglomerativeClustering(n_clusters=num_clusters, affinity='precomputed', linkage='average')
clusters = clustering.fit_predict(dist_mat)

sorted_binned_s_raw = binned_s_raw[:, np.argsort(clusters)]
sorted_var_name = var_name[np.argsort(clusters)]
clusters = clusters[np.argsort(clusters)]
sorted_binned_s_raw = scipy.stats.zscore(sorted_binned_s_raw)
sorted_binned_s_raw = sorted_binned_s_raw.T

def generate_cluster_colors(num_clusters):
    cmap = plt.get_cmap('tab10')
    cluster_colors = [cmap(i) for i in np.linspace(0, 1, num_clusters)]
    return cluster_colors
cluster_colors = generate_cluster_colors(num_clusters)
row_colors = [cluster_colors[i] for i in clusters]

dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/"
sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(10, 8))
sns.clustermap(sorted_binned_s_raw, cmap='viridis', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, row_colors=row_colors)

plt.xlabel('Pseudotime bins')
plt.set_ylabel('Genes')
plt.title('Gene Expression Heatmap')
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_clustering_binned_u.png", bbox_inches='tight')
plt.close()




# binning
pseudotime_uni
num_bins = 30
num_clusters = 4
percentiles = np.percentile(pseudotime_uni, np.linspace(0, 100, num_bins+1))

# compute mean expression in each bin
s_raw = adata_r.layers["dsdt"].toarray()
var_name = adata_r.var_names.to_numpy()
binned_data = []
for i in range(num_bins):
    bin_start = percentiles[i]
    bin_end = percentiles[i+1]
    bin_indices = np.where((pseudotime_uni >= bin_start) & (pseudotime_uni < bin_end))[0]
    bin_mean = np.mean(s_raw[bin_indices, :], axis=0)
    binned_data.append(bin_mean)

binned_s_raw = np.array(binned_data)
cossim_mat = np.dot(binned_s_raw.T, binned_s_raw)
norms = np.linalg.norm(binned_s_raw, axis=0)
cossim_mat = cossim_mat / (norms[:, np.newaxis] * norms)

dist_mat = 1 - cossim_mat
clustering = AgglomerativeClustering(n_clusters=num_clusters, affinity='precomputed', linkage='average')
clusters = clustering.fit_predict(dist_mat)

sorted_binned_s_raw = binned_s_raw[:, np.argsort(clusters)]
sorted_var_name = var_name[np.argsort(clusters)]
clusters = clusters[np.argsort(clusters)]
sorted_binned_s_raw = sorted_binned_s_raw / np.std(sorted_binned_s_raw, axis=0)
sorted_binned_s_raw = sorted_binned_s_raw.T

cluster_colors = generate_cluster_colors(num_clusters)
row_colors = [cluster_colors[i] for i in clusters]

dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/"
sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(10, 8))
sns.clustermap(sorted_binned_s_raw, cmap='coolwarm', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, row_colors=row_colors)

plt.xlabel('Pseudotime bins')
plt.set_ylabel('Genes')
plt.title('Velocity Heatmap')
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_clustering_binned_ds.png", bbox_inches='tight')
plt.close()


