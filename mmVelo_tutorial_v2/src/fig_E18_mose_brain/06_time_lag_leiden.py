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
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import scipy

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
dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/time_lag/clustering"
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



# clustering, velocity
dsdt = adata_r[:, prom_gene].layers["dsdt"].toarray() # 3735, 1109
dudt = adata_r[:, prom_gene].layers["dudt"].toarray()
dpdt = adata_r[:, prom_gene].layers["dpdt"]

s_raw = adata_r[:, prom_gene].layers["s_raw"].toarray() # 3735, 1109
u_raw = adata_r[:, prom_gene].layers["u_raw"].toarray()
p_raw = adata_r[:, prom_gene].layers["p_raw"]

var_name = adata_r[:, prom_gene].var_names.to_numpy()

# binning
num_bins = 20
percentiles = np.percentile(pseudotime_uni, np.linspace(0, 100, num_bins+1))

# compute mean expression in each bin
binned_data_dp, binned_data_du, binned_data_ds = [], [], []
binned_data_p, binned_data_u, binned_data_s = [], [], []
for i in range(num_bins):
    bin_start = percentiles[i]
    bin_end = percentiles[i+1]
    bin_indices = np.where((pseudotime_uni >= bin_start) & (pseudotime_uni < bin_end))[0]
    bin_mean_dp = np.mean(dpdt[bin_indices, :], axis=0)
    bin_mean_du = np.mean(dudt[bin_indices, :], axis=0)
    bin_mean_ds = np.mean(dsdt[bin_indices, :], axis=0)
    bin_mean_p = np.mean(p_raw[bin_indices, :], axis=0)
    bin_mean_u = np.mean(u_raw[bin_indices, :], axis=0)
    bin_mean_s = np.mean(s_raw[bin_indices, :], axis=0)
    binned_data_dp.append(bin_mean_dp)
    binned_data_du.append(bin_mean_du)
    binned_data_ds.append(bin_mean_ds)
    binned_data_p.append(bin_mean_p)
    binned_data_u.append(bin_mean_u)
    binned_data_s.append(bin_mean_s)


binned_data_dp = np.array(binned_data_dp)
binned_data_du = np.array(binned_data_du)
binned_data_ds = np.array(binned_data_ds)
binned_data_dp = binned_data_dp / np.std(binned_data_dp, axis=0)
binned_data_du = binned_data_du / np.std(binned_data_du, axis=0)
binned_data_ds = binned_data_ds / np.std(binned_data_ds, axis=0)
binned_data_p = scipy.stats.zscore(np.array(binned_data_p))
binned_data_u = scipy.stats.zscore(np.array(binned_data_u))
binned_data_s = scipy.stats.zscore(np.array(binned_data_s))

binned_dx = np.concatenate([binned_data_dp, binned_data_du, binned_data_ds])
binned_x_raw = np.concatenate([binned_data_p, binned_data_u, binned_data_s])


# clustering and GO analysis
resolution = 0.8
adata_bin = ad.AnnData(X=binned_dx.T)
adata_bin.layers["binned_x"] = binned_x_raw.T
adata_bin.obs_names = var_name
sc.pp.neighbors(adata_bin, metric="cosine", n_pcs=None, use_rep="X")
sc.tl.leiden(adata_bin, resolution=resolution)
sc.tl.umap(adata_bin)

dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/time_lag/leiden"
fig, ax = plt.subplots(figsize=(5, 5))
sc.pl.umap(adata_bin, color="leiden")
fig.tight_layout()
plt.savefig(dir_path + "/umap_leiden_resolution_{}.png".format(resolution), bbox_inches='tight')
plt.close()

def generate_cluster_colors(num_clusters):
    cmap = plt.get_cmap('tab20')
    cluster_colors = [cmap(i) for i in np.linspace(0, 1, num_clusters)]
    return cluster_colors

clusters = adata_bin.obs["leiden"]
num_clusters = len(adata_bin.obs["leiden"].cat.categories)
cluster_colors = generate_cluster_colors(num_clusters)


sorted_binned_dx = binned_dx[:, np.argsort(clusters)]
sorted_var_name = var_name[np.argsort(clusters)]
sorted_binned_dx = sorted_binned_dx.T
sorted_binned_x_raw = binned_x_raw[:, np.argsort(clusters)]
sorted_binned_x_raw = sorted_binned_x_raw.T
sorted_var_name = var_name[np.argsort(clusters)]
clusters = clusters[np.argsort(clusters)]
row_colors = [cluster_colors[int(i)] for i in clusters]



dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/time_lag/leiden"
sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(10, 8))
sns.clustermap(sorted_binned_dx, cmap='coolwarm', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, row_colors=row_colors)

plt.xlabel('Pseudotime bins')
plt.title('Reconstructed Count Heatmap')
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_clustering_binned_dx_leiden_{}.png".format(resolution), bbox_inches='tight')
#plt.savefig(dir_path + "/heatmap_clustering_binned_dx_{}_complete.png".format(num_clusters), bbox_inches='tight')
#plt.savefig(dir_path + "/heatmap_clustering_binned_dx.png", bbox_inches='tight')
plt.close()

sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(10, 8))
sns.clustermap(sorted_binned_x_raw, cmap='viridis', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, row_colors=row_colors)

plt.xlabel('Pseudotime bins')
plt.title('Velocity Heatmap')
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_clustering_binned_dx_leiden_{}_x.png".format(resolution), bbox_inches='tight')
#plt.savefig(dir_path + "/heatmap_clustering_binned_dx_{}_x_complete.png".format(num_clusters), bbox_inches='tight')
#plt.savefig(dir_path + "/heatmap_clustering_binned_dx_x.png", bbox_inches='tight')
plt.close()


from gprofiler import GProfiler
def perform_go_analysis(gene_list, bg_list):
     gp = GProfiler(return_dataframe=True)
     query = {
         "query": gene_list,
         "background": bg_list
     }
     results = gp.profile(gene_list, organism="mmusculus", sources=["GO"],
                           user_threshold=0.05, background=bg_list)
     return results

dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/time_lag/leiden/go_result_res_{}".format(resolution)
if not os.path.exists(dir_path):
    os.mkdir(dir_path)
background_list = list(adata_bin.obs_names)
for i in adata_bin.obs["leiden"].cat.categories:
    gene_list = list(adata_bin.obs_names[adata_bin.obs["leiden"] == i])
    go_results = perform_go_analysis(gene_list, background_list)
    go_results.to_csv(dir_path+"/go_cluter_bg_prom_{}.tsv".format(i), sep="\t", )

background_list = list(adata_r.var_names)
for i in adata_bin.obs["leiden"].cat.categories:
    gene_list = list(adata_bin.obs_names[adata_bin.obs["leiden"] == i])
    go_results = perform_go_analysis(gene_list, background_list)
    go_results.to_csv(dir_path+"/go_cluter_bg_diff_{}.tsv".format(i), sep="\t", )

# prom vs non prom genes
dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/time_lag/leiden/"
gene_list = list(adata_bin.obs_names)
background_list = list(adata_r.var_names)
go_results = perform_go_analysis(gene_list, background_list)
go_results.to_csv(dir_path+"/go_diff_vs_prom_genes.tsv", sep="\t", )


resolutions = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
for resolution in resolutions:
    resolution

    adata_bin = ad.AnnData(X=binned_dx.T)
    adata_bin.layers["binned_x"] = binned_x_raw.T
    adata_bin.obs_names = var_name
    sc.pp.neighbors(adata_bin, metric="cosine", n_pcs=None, use_rep="X")
    sc.tl.leiden(adata_bin, resolution=resolution)
    sc.tl.umap(adata_bin)

    dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/time_lag/leiden"
    fig, ax = plt.subplots(figsize=(5, 5))
    sc.pl.umap(adata_bin, color="leiden")
    fig.tight_layout()
    plt.savefig(dir_path + "/umap_leiden_resolution_{}.png".format(resolution), bbox_inches='tight')
    plt.close()

    def generate_cluster_colors(num_clusters):
        cmap = plt.get_cmap('tab20')
        cluster_colors = [cmap(i) for i in np.linspace(0, 1, num_clusters)]
        return cluster_colors

    clusters = adata_bin.obs["leiden"]
    num_clusters = len(adata_bin.obs["leiden"].cat.categories)
    cluster_colors = generate_cluster_colors(num_clusters)


    sorted_binned_dx = binned_dx[:, np.argsort(clusters)]
    sorted_var_name = var_name[np.argsort(clusters)]
    sorted_binned_dx = sorted_binned_dx.T
    sorted_binned_x_raw = binned_x_raw[:, np.argsort(clusters)]
    sorted_binned_x_raw = sorted_binned_x_raw.T
    sorted_var_name = var_name[np.argsort(clusters)]
    clusters = clusters[np.argsort(clusters)]
    row_colors = [cluster_colors[int(i)] for i in clusters]



    dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/time_lag/leiden"
    sns.set(font_scale=0.8)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.clustermap(sorted_binned_dx, cmap='coolwarm', xticklabels=False, yticklabels=False,
                row_cluster=False, col_cluster=False, row_colors=row_colors)

    plt.xlabel('Pseudotime bins')
    plt.title('Reconstructed Count Heatmap')
    fig.tight_layout()
    plt.savefig(dir_path + "/heatmap_clustering_binned_dx_leiden_{}.png".format(resolution), bbox_inches='tight')
    #plt.savefig(dir_path + "/heatmap_clustering_binned_dx_{}_complete.png".format(num_clusters), bbox_inches='tight')
    #plt.savefig(dir_path + "/heatmap_clustering_binned_dx.png", bbox_inches='tight')
    plt.close()

    sns.set(font_scale=0.8)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.clustermap(sorted_binned_x_raw, cmap='viridis', xticklabels=False, yticklabels=False,
                row_cluster=False, col_cluster=False, row_colors=row_colors)

    plt.xlabel('Pseudotime bins')
    plt.title('Velocity Heatmap')
    fig.tight_layout()
    plt.savefig(dir_path + "/heatmap_clustering_binned_dx_leiden_{}_x.png".format(resolution), bbox_inches='tight')
    #plt.savefig(dir_path + "/heatmap_clustering_binned_dx_{}_x_complete.png".format(num_clusters), bbox_inches='tight')
    #plt.savefig(dir_path + "/heatmap_clustering_binned_dx_x.png", bbox_inches='tight')
    plt.close()


    from gprofiler import GProfiler
    def perform_go_analysis(gene_list, bg_list):
        gp = GProfiler(return_dataframe=True)
        query = {
            "query": gene_list,
            "background": bg_list
        }
        results = gp.profile(gene_list, organism="mmusculus", sources=["GO"],
                            user_threshold=0.05, background=bg_list)
        return results

    dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/time_lag/leiden/go_result_res_{}".format(resolution)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    background_list = list(adata_bin.obs_names)
    for i in adata_bin.obs["leiden"].cat.categories:
        gene_list = list(adata_bin.obs_names[adata_bin.obs["leiden"] == i])
        go_results = perform_go_analysis(gene_list, background_list)
        go_results.to_csv(dir_path+"/go_cluter_bg_prom_{}.tsv".format(i), sep="\t", )

    background_list = list(adata_r.var_names)
    for i in adata_bin.obs["leiden"].cat.categories:
        gene_list = list(adata_bin.obs_names[adata_bin.obs["leiden"] == i])
        go_results = perform_go_analysis(gene_list, background_list)
        go_results.to_csv(dir_path+"/go_cluter_bg_diff_{}.tsv".format(i), sep="\t", )

    # signaling pathway
    def perform_go_analysis(gene_list, bg_list):
        gp = GProfiler(return_dataframe=True)
        query = {
            "query": gene_list,
            "background": bg_list
        }
        results = gp.profile(gene_list, organism="mmusculus", sources=["KEGG", "REAC", "WP"],
                            user_threshold=0.05, background=bg_list)
        return results

    dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/time_lag/leiden/pathway_result_res_{}".format(resolution)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    background_list = list(adata_bin.obs_names)
    for i in adata_bin.obs["leiden"].cat.categories:
        gene_list = list(adata_bin.obs_names[adata_bin.obs["leiden"] == i])
        go_results = perform_go_analysis(gene_list, background_list)
        go_results.to_csv(dir_path+"/pathway_cluter_bg_prom_{}.tsv".format(i), sep="\t", )

    background_list = list(adata_r.var_names)
    for i in adata_bin.obs["leiden"].cat.categories:
        gene_list = list(adata_bin.obs_names[adata_bin.obs["leiden"] == i])
        go_results = perform_go_analysis(gene_list, background_list)
        go_results.to_csv(dir_path+"/pathway_cluter_bg_diff_{}.tsv".format(i), sep="\t", )

    # KEGG
    def perform_go_analysis(gene_list, bg_list):
        gp = GProfiler(return_dataframe=True)
        query = {
            "query": gene_list,
            "background": bg_list
        }
        results = gp.profile(gene_list, organism="mmusculus", sources=["KEGG"],
                            user_threshold=0.05, background=bg_list)
        return results

    dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/time_lag/leiden/kegg_result_res_{}".format(resolution)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    background_list = list(adata_bin.obs_names)
    for i in adata_bin.obs["leiden"].cat.categories:
        gene_list = list(adata_bin.obs_names[adata_bin.obs["leiden"] == i])
        go_results = perform_go_analysis(gene_list, background_list)
        go_results.to_csv(dir_path+"/kegg_cluter_bg_prom_{}.tsv".format(i), sep="\t", )

    background_list = list(adata_r.var_names)
    for i in adata_bin.obs["leiden"].cat.categories:
        gene_list = list(adata_bin.obs_names[adata_bin.obs["leiden"] == i])
        go_results = perform_go_analysis(gene_list, background_list)
        go_results.to_csv(dir_path+"/kegg_cluter_bg_diff_{}.tsv".format(i), sep="\t", )

    # Reactome
    def perform_go_analysis(gene_list, bg_list):
        gp = GProfiler(return_dataframe=True)
        query = {
            "query": gene_list,
            "background": bg_list
        }
        results = gp.profile(gene_list, organism="mmusculus", sources=["REAC"],
                            user_threshold=0.05, background=bg_list)
        return results

    dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/time_lag/leiden/reac_result_res_{}".format(resolution)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    background_list = list(adata_bin.obs_names)
    for i in adata_bin.obs["leiden"].cat.categories:
        gene_list = list(adata_bin.obs_names[adata_bin.obs["leiden"] == i])
        go_results = perform_go_analysis(gene_list, background_list)
        go_results.to_csv(dir_path+"/reac_cluter_bg_prom_{}.tsv".format(i), sep="\t", )

    background_list = list(adata_r.var_names)
    for i in adata_bin.obs["leiden"].cat.categories:
        gene_list = list(adata_bin.obs_names[adata_bin.obs["leiden"] == i])
        go_results = perform_go_analysis(gene_list, background_list)
        go_results.to_csv(dir_path+"/reac_cluter_bg_diff_{}.tsv".format(i), sep="\t", )


    # WikiPathways
    def perform_go_analysis(gene_list, bg_list):
        gp = GProfiler(return_dataframe=True)
        query = {
            "query": gene_list,
            "background": bg_list
        }
        results = gp.profile(gene_list, organism="mmusculus", sources=["WP"],
                            user_threshold=0.05, background=bg_list)
        return results

    dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/time_lag/leiden/wp_result_res_{}".format(resolution)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    background_list = list(adata_bin.obs_names)
    for i in adata_bin.obs["leiden"].cat.categories:
        gene_list = list(adata_bin.obs_names[adata_bin.obs["leiden"] == i])
        go_results = perform_go_analysis(gene_list, background_list)
        go_results.to_csv(dir_path+"/wp_cluter_bg_prom_{}.tsv".format(i), sep="\t", )

    background_list = list(adata_r.var_names)
    for i in adata_bin.obs["leiden"].cat.categories:
        gene_list = list(adata_bin.obs_names[adata_bin.obs["leiden"] == i])
        go_results = perform_go_analysis(gene_list, background_list)
        go_results.to_csv(dir_path+"/wp_cluter_bg_diff_{}.tsv".format(i), sep="\t", )


    # transfac
    def perform_go_analysis(gene_list, bg_list):
        gp = GProfiler(return_dataframe=True)
        query = {
            "query": gene_list,
            "background": bg_list
        }
        results = gp.profile(gene_list, organism="mmusculus", sources=["TF"],
                            user_threshold=0.05, background=bg_list)
        return results

    dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/time_lag/leiden/tf_result_res_{}".format(resolution)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    background_list = list(adata_bin.obs_names)
    for i in adata_bin.obs["leiden"].cat.categories:
        gene_list = list(adata_bin.obs_names[adata_bin.obs["leiden"] == i])
        go_results = perform_go_analysis(gene_list, background_list)
        go_results.to_csv(dir_path+"/tf_cluter_bg_prom_{}.tsv".format(i), sep="\t", )

    background_list = list(adata_r.var_names)
    for i in adata_bin.obs["leiden"].cat.categories:
        gene_list = list(adata_bin.obs_names[adata_bin.obs["leiden"] == i])
        go_results = perform_go_analysis(gene_list, background_list)
        go_results.to_csv(dir_path+"/tf_cluter_bg_diff_{}.tsv".format(i), sep="\t", )




# use resolution = 0.8 for further analysis
resolution = 0.8
adata_bin = ad.AnnData(X=binned_dx.T)
adata_bin.layers["binned_x"] = binned_x_raw.T
adata_bin.obs_names = var_name
sc.pp.neighbors(adata_bin, metric="cosine", n_pcs=None, use_rep="X")
sc.tl.leiden(adata_bin, resolution=resolution)
sc.tl.umap(adata_bin)

dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/time_lag/leiden"
fig, ax = plt.subplots(figsize=(5, 5))
sc.pl.umap(adata_bin, color="leiden")
fig.tight_layout()
plt.savefig(dir_path + "/umap_leiden_resolution_{}.png".format(resolution), bbox_inches='tight')
plt.close()

def generate_cluster_colors(num_clusters):
    cmap = plt.get_cmap('tab20')
    cluster_colors = [cmap(i) for i in np.linspace(0, 1, num_clusters)]
    return cluster_colors

clusters = adata_bin.obs["leiden"]
num_clusters = len(adata_bin.obs["leiden"].cat.categories)
cluster_colors = generate_cluster_colors(num_clusters)


sorted_binned_dx = binned_dx[:, np.argsort(clusters)]
sorted_var_name = var_name[np.argsort(clusters)]
sorted_binned_dx = sorted_binned_dx.T
sorted_binned_x_raw = binned_x_raw[:, np.argsort(clusters)]
sorted_binned_x_raw = sorted_binned_x_raw.T
sorted_var_name = var_name[np.argsort(clusters)]
clusters = clusters[np.argsort(clusters)]
row_colors = [cluster_colors[int(i)] for i in clusters]



dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/time_lag/leiden"
sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(10, 8))
sns.clustermap(sorted_binned_dx, cmap='coolwarm', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, row_colors=row_colors)

plt.xlabel('Pseudotime bins')
plt.title('Reconstructed Count Heatmap')
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_clustering_binned_dx_leiden_{}.png".format(resolution), bbox_inches='tight')
#plt.savefig(dir_path + "/heatmap_clustering_binned_dx_{}_complete.png".format(num_clusters), bbox_inches='tight')
#plt.savefig(dir_path + "/heatmap_clustering_binned_dx.png", bbox_inches='tight')
plt.close()

sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(10, 8))
sns.clustermap(sorted_binned_x_raw, cmap='viridis', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, row_colors=row_colors)

plt.xlabel('Pseudotime bins')
plt.title('Velocity Heatmap')
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_clustering_binned_dx_leiden_{}_x.png".format(resolution), bbox_inches='tight')
#plt.savefig(dir_path + "/heatmap_clustering_binned_dx_{}_x_complete.png".format(num_clusters), bbox_inches='tight')
#plt.savefig(dir_path + "/heatmap_clustering_binned_dx_x.png", bbox_inches='tight')
plt.close()


from gprofiler import GProfiler
def perform_go_analysis(gene_list, bg_list):
     gp = GProfiler(return_dataframe=True)
     query = {
         "query": gene_list,
         "background": bg_list
     }
     results = gp.profile(gene_list, organism="mmusculus", sources=["GO"],
                           user_threshold=0.05, background=bg_list)
     return results

dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/time_lag/leiden/go_result_res_{}".format(resolution)
if not os.path.exists(dir_path):
    os.mkdir(dir_path)
background_list = list(adata_bin.obs_names)
for i in adata_bin.obs["leiden"].cat.categories:
    gene_list = list(adata_bin.obs_names[adata_bin.obs["leiden"] == i])
    go_results = perform_go_analysis(gene_list, background_list)
    go_results.to_csv(dir_path+"/go_cluter_bg_prom_{}.tsv".format(i), sep="\t", )

background_list = list(adata_r.var_names)
for i in adata_bin.obs["leiden"].cat.categories:
    gene_list = list(adata_bin.obs_names[adata_bin.obs["leiden"] == i])
    go_results = perform_go_analysis(gene_list, background_list)
    go_results.to_csv(dir_path+"/go_cluter_bg_diff_{}.tsv".format(i), sep="\t", )

# prom vs non prom genes
dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/time_lag/leiden/"
gene_list = list(adata_bin.obs_names)
background_list = list(adata_r.var_names)
go_results = perform_go_analysis(gene_list, background_list)
go_results.to_csv(dir_path+"/go_diff_vs_prom_genes.tsv", sep="\t", )

# further analysis
#adata_bin.var_names = adata_bin.var_names.to_numpy().astype(np.int8)
dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/time_lag/leiden/go_result_res_{}".format(resolution)
genes = list(adata_bin.obs_names[adata_bin.obs["leiden"] == "1"])
print(len(genes)) # 189
adata_bin_one = adata_bin[genes, :]

adata_bin_one.X[:, :20]
p_argmax = np.argmax(adata_bin_one.X[:, :20], axis=1)

adata_bin_one.X[:, 20:40]
u_argmax = np.argmax(adata_bin_one.X[:, 20:40], axis=1)

adata_bin_one.X[:, 40:60]
s_argmax = np.argmax(adata_bin_one.X[:, 40:60], axis=1)

(s_argmax - u_argmax < 0).sum() # 45
(u_argmax - p_argmax < 0).sum() # 65

fig, ax = plt.subplots(figsize=(5, 5))
ax.hist(np.array(u_argmax - p_argmax), bins=40, range=(-20,20))
ax.set_xlabel('time lag')
ax.set_ylabel("frequency")
ax.set_title('argmax time lag between u vs. p')
fig.tight_layout()
plt.savefig(dir_path + "/res_1e-8_lag_u_p_hist.png", bbox_inches='tight')
plt.close()

fig, ax = plt.subplots(figsize=(5, 5))
ax.hist(np.array(s_argmax - u_argmax), bins=40, range=(-20,20))
ax.set_xlabel('time lag')
ax.set_ylabel("frequency")
ax.set_title('argmax time lag between s vs. u')
fig.tight_layout()
plt.savefig(dir_path + "/res_1e-8_lag_s_u_hist.png", bbox_inches='tight')
plt.close()

list(adata_bin_one.obs_names[s_argmax - u_argmax < 0])
adata_bin_one.obs_names[s_argmax - u_argmax >= 0]

gene_list = list(adata_bin_one.obs_names[s_argmax - u_argmax < 0])
background_list = list(adata_bin.obs_names)
go_results = perform_go_analysis(gene_list, background_list)
go_results.to_csv(dir_path+"/cluster_1_go_no_time_lag_genes.tsv", sep="\t", )

gene_list = list(adata_bin_one.obs_names[s_argmax - u_argmax >= 0])
background_list = list(adata_bin.obs_names)
go_results = perform_go_analysis(gene_list, background_list)
go_results.to_csv(dir_path+"/cluster_1_go_with_time_lag_genes.tsv", sep="\t", )