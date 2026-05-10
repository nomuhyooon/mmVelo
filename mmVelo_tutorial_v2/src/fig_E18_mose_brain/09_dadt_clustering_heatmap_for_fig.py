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
#dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/dadt_clustering"
dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/dadt_clustering_for_fig"
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

dir_path = "/home/nomura/Proj/mmvelo/data/10x_multiome_brain_repre_wo_IN/analysis/feature_linkage/peak_promoter_linkage.mtx"
peak2prom_mat = mmread(dir_path).toarray()

dir_path = "/home/nomura/Proj/mmvelo/data/10x_multiome_brain_repre_wo_IN/analysis/feature_linkage/peak_distance.tsv"
distance_df = pd.read_csv(dir_path, sep="\t")

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

adata_r_exn.obs["pseudotime_uni"]
adata_r_exn.obs["pseudotime_uni"][np.argsort(adata_r_exn.obs["pseudotime_uni"])]

adata_a_exn = adata_a[adata_r_exn.obs_names]
a_raw = adata_a_exn.layers["a_raw"].toarray()[np.argsort(adata_r_exn.obs["pseudotime_uni"]), :]
dadt = adata_a_exn.layers["dadt"].toarray()[np.argsort(adata_r_exn.obs["pseudotime_uni"]), :]
var_name = adata_a_exn.var_names.to_numpy()

dadt = dadt / np.std(dadt, axis=0)
a_raw = scipy.stats.zscore(a_raw)

# clustering
dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/dadt_clustering"
adata_bin = sc.read_loom(dir_path+"/adata_dadt_cluster.loom")
adata_bin.obs_names = adata_bin.obs["obs_names"]

"""
resolution = 0.8
adata_bin = ad.AnnData(X=dadt.T)
adata_bin.layers["rec_x"] = a_raw.T
adata_bin.obs_names = var_name
sc.pp.neighbors(adata_bin, metric="cosine", n_pcs=None, use_rep="X")
sc.tl.leiden(adata_bin, resolution=resolution)
sc.tl.umap(adata_bin)
"""
resolution = 0.8
dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/dadt_clustering_for_fig"
fig, ax = plt.subplots(figsize=(5, 5))
sc.pl.umap(adata_bin, color="leiden")
fig.tight_layout()
plt.savefig(dir_path + "/umap_leiden_resolution_{}.png".format(resolution), bbox_inches='tight', dpi=300)
plt.close()

dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/dadt_clustering_for_fig"
fig, ax = plt.subplots(figsize=(5, 5))
sc.pl.umap(adata_bin, color="leiden")
plt.gca().axis("off")
plt.title("")
fig.tight_layout()
plt.savefig(dir_path + "/umap_leiden_resolution_{}_blank.png".format(resolution), bbox_inches='tight', dpi=300)
plt.close()

def generate_cluster_colors(num_clusters):
    cmap = plt.get_cmap('tab20')
    cluster_colors = [cmap(i) for i in np.linspace(0, 1, num_clusters)]
    return cluster_colors

clusters = adata_bin.obs["leiden"]
num_clusters = len(adata_bin.obs["leiden"].cat.categories)
cluster_colors = generate_cluster_colors(num_clusters)
clusters = clusters.map({
    #"0":"0", "1":"4", "2":"2", "3":"6", "4":"7", "5":"5", "6":"1", "7":"8", "8":"3",
    "0":0, "1":4, "2":2, "3":6, "4":7, "5":5, "6":1, "7":8, "8":3, 
}).astype("category").to_numpy()


sorted_dadt = dadt[:, np.argsort(clusters)]
sorted_var_name = var_name[np.argsort(clusters)]
sorted_dadt = sorted_dadt.T
sorted_x_raw = a_raw[:, np.argsort(clusters)]
sorted_x_raw = sorted_x_raw.T
clusters = clusters[np.argsort(clusters)]
row_colors = [cluster_colors[int(i)] for i in clusters]


sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(5, 5))
sns.clustermap(sorted_dadt, cmap='coolwarm', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, row_colors=row_colors,
               vmin=-3, vmax=3)
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_clustering_binned_dx_leiden_{}.png".format(resolution), bbox_inches='tight')
#plt.savefig(dir_path + "/heatmap_clustering_binned_dx_{}_complete.png".format(num_clusters), bbox_inches='tight')
#plt.savefig(dir_path + "/heatmap_clustering_binned_dx.png", bbox_inches='tight')
plt.close()

sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(5, 5))
sns.clustermap(sorted_dadt, cmap='coolwarm', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, row_colors=row_colors,
               vmin=-3, vmax=3)
plt.gca().axis("off")
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_clustering_binned_dx_leiden_{}_blank.png".format(resolution), bbox_inches='tight', dpi=300)
#plt.savefig(dir_path + "/heatmap_clustering_binned_dx_{}_complete.png".format(num_clusters), bbox_inches='tight')
#plt.savefig(dir_path + "/heatmap_clustering_binned_dx.png", bbox_inches='tight')
plt.close()


sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(5, 5))
sns.clustermap(sorted_x_raw, cmap='viridis', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, row_colors=row_colors, 
               vmin = -3, vmax = 3)
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_clustering_binned_dx_leiden_{}_x.png".format(resolution), bbox_inches='tight')
#plt.savefig(dir_path + "/heatmap_clustering_binned_dx_{}_x_complete.png".format(num_clusters), bbox_inches='tight')
#plt.savefig(dir_path + "/heatmap_clustering_binned_dx_x.png", bbox_inches='tight')
plt.close()

sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(5, 5))
sns.clustermap(sorted_x_raw, cmap='viridis', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, row_colors=row_colors, 
               vmin = -3, vmax = 3)
plt.gca().axis("off")
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_clustering_binned_dx_leiden_{}_x_blank.png".format(resolution), bbox_inches='tight', dpi=300)
#plt.savefig(dir_path + "/heatmap_clustering_binned_dx_{}_x_complete.png".format(num_clusters), bbox_inches='tight')
#plt.savefig(dir_path + "/heatmap_clustering_binned_dx_x.png", bbox_inches='tight')
plt.close()

# save adata to perform motif enrichment analysis
#dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/dadt_clustering"
#adata_bin.write_loom(dir_path+"/adata_dadt_cluster.loom", write_obsm_varm=True)


# read adata
## change environment to scenicplus
import scanpy as sc
import pycistarget
import pyranges as pr
import pickle
import statsmodels.api as sm

dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/dadt_clustering"
adata_peak = sc.read_loom(dir_path+"/adata_dadt_cluster.loom")
adata_peak.obs_names = adata_peak.obs["obs_names"]
peaks = list(adata_peak.obs_names)

region_sets = dict()
for clst in set(adata_peak.obs["leiden"]):
    clst_peaks = adata_peak.obs_names[adata_peak.obs["leiden"] == clst]
    clst_peaks = pr.from_dict({"Chromosome": [peak.split(':')[0] for peak in clst_peaks],
                      "Start": [int(peak.split(':')[1].split('-')[0]) for peak in clst_peaks],
                      "End": [int(peak.split(':')[1].split('-')[1]) for peak in clst_peaks]})
    key = f'{clst}'
    region_sets[key] = clst_peaks


from pycistarget.motif_enrichment_cistarget import *
cistarget_dict = run_cistarget(ctx_db = '/home/nomura/Proj/mmvelo/pycistarget/mm10_screen_v10_clust.regions_vs_motifs.rankings.feather',
                                                      region_sets = region_sets,
                                                      specie = 'mus_musculus',
                                                      auc_threshold = 0.005,
                                                      nes_threshold = 3.0,
                                                      rank_threshold = 0.05,
                                                      annotation = ['Direct_annot', 'Orthology_annot'],
                                                      annotation_version = 'v10nr_clust',
                                                      path_to_motif_annotations = '/home/nomura/Proj/mmvelo/pycistarget/motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl',
                                                      n_cpu = 4,
                                                      #_temp_dir='/scratch/leuven/313/vsc31305/ray_spill'
                                                      )

dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/dadt_clustering"
with open(dir_path + '/cisTarget_dict.pkl', 'wb') as f:
  pickle.dump(cistarget_dict, f)

dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/dadt_clustering"
infile = open(dir_path + '/cisTarget_dict.pkl', 'rb')
cistarget_dict = pickle.load(infile)
infile.close()

#cistarget_results(cistarget_dict, name='0')
for clst in set(adata_peak.obs["leiden"]):
    out_file = dir_path + f'/cluster_{clst}_motif_enricment.html'
    cistarget_dict[clst].motif_enrichment.to_html(open(out_file, 'w'), escape=False, col_space=80)


"""
clst = "0"
tf_lists = list(cistarget_dict[clst].motif_enrichment["Direct_annot"][:5]) + \
          list(cistarget_dict[clst].motif_enrichment["Orthology_annot"][:5])

tf_list = []
for i in range(len(tf_lists)):
    if isinstance(tf_lists[i], float):
        continue
    tf_lists[i] = tf_lists[i].split(", ")
    tf_list.append(tf_lists[i])
tf_list = sum(tf_list, [])

exists_tf_list = []
for tf in tf_list:
    if tf in adata_r.var_names:
        exists_tf_list.append(tf)


dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/dadt_clustering"
file_path = dir_path + f"/tf_in_cluster_{clst}"
if not os.path.exists(file_path):
    os.mkdir(file_path)

for tf in exists_tf_list:
    argsort = np.argsort(adata_r_exn.obs["pseudotime"])
    p_time = np.linspace(0, 1, adata_r_exn.n_obs)
    tf_s = adata_r_exn[argsort, tf].layers["s_raw"].toarray().reshape(-1)
    tf_s = tf_s / np.std(tf_s)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(p_time, tf_s, s=1)
    ax.set_xlabel('pseudotime')
    ax.set_ylabel('mean expr')
    ax.set_title(f'{tf}  mean expr')
    fig.tight_layout()
    plt.savefig(file_path + f"/{tf}_mean_expr.png", bbox_inches='tight')
    plt.close()
""" 



adata_r_exn
lowess = sm.nonparametric.lowess
frac = 300 / adata_r_exn.n_obs # use 10 neighbor cells for lowess regression
p_time = np.linspace(0, 1, adata_r_exn.n_obs)

dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/dadt_clustering"
for clst in set(adata_peak.obs["leiden"]):
    adata_peak_clst = adata_peak[adata_peak.obs["leiden"] == clst, :]
    peak_clst_dadt = adata_peak_clst.X.toarray().mean(0)

    p_time = np.linspace(0, 1, peak_clst_dadt.shape[0])
    smooth_peak_dadt = lowess(peak_clst_dadt, p_time, frac=frac, return_sorted=False)
    
    norm_peak_clst_dadt = peak_clst_dadt / np.max(np.abs(smooth_peak_dadt))
    norm_smooth_peak_dadt = smooth_peak_dadt / np.max(np.abs(smooth_peak_dadt))
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(p_time, norm_smooth_peak_dadt, color="blue", linewidth=3)
    ax.scatter(p_time, norm_peak_clst_dadt, s=0.1, color="blue")
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('pseudotime')
    ax.set_ylabel('mean velocity')
    ax.set_title(f'Mean velocity in cluster {clst}')
    fig.tight_layout()
    plt.savefig(dir_path + f"/clst_{clst}_mean_vel.png", bbox_inches='tight', dpi=300)
    plt.close()

    tf_motifs = cistarget_dict[clst].motif_enrichment

    tf_lists = list(cistarget_dict[clst].motif_enrichment["Direct_annot"][:5]) + \
            list(cistarget_dict[clst].motif_enrichment["Orthology_annot"][:5])

    tf_list = []
    for i in range(len(tf_lists)):
        if isinstance(tf_lists[i], float):
            continue
        tf_lists[i] = tf_lists[i].split(", ")
        tf_list.append(tf_lists[i])
    tf_list = sum(tf_list, [])

    exists_tf_list = []
    for tf in tf_list:
        if tf in adata_r.var_names:
            exists_tf_list.append(tf)


    dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/dadt_clustering"
    file_path = dir_path + f"/tf_in_cluster_{clst}"
    if not os.path.exists(file_path):
        os.mkdir(file_path)

    for tf in exists_tf_list:
        argsort = np.argsort(adata_r_exn.obs["pseudotime"])
        p_time = np.linspace(0, 1, adata_r_exn.n_obs)
        tf_s = adata_r_exn[argsort, tf].layers["s_raw"].toarray().reshape(-1)
        tf_s = tf_s / np.std(tf_s)
        smooth_tf_s = lowess(tf_s, p_time, frac=frac, return_sorted=False)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(p_time, tf_s, s=0.1, color = "red")
        ax.plot(p_time, smooth_tf_s, color="red", linewidth=3)

        ax.scatter(p_time, peak_clst_dadt, s=0.1, color="blue")
        ax.plot(p_time, smooth_peak_dadt, color="blue", linewidth=3)
        ax.set_xlabel('pseudotime')
        ax.set_ylabel('mean expr/vel')
        ax.set_title(f'{tf}  mean expr vs {clst} mean peak vel')
        fig.tight_layout()
        plt.savefig(file_path + f"/{tf}_mean_expr.png", bbox_inches='tight')
        plt.close()

        fig, ax = plt.subplots(figsize=(5, 5))
        norm_tf_s = tf_s / max(smooth_tf_s)
        norm_smooth_tf_s = smooth_tf_s / max(smooth_tf_s)
        norm_peak_clst_dadt = peak_clst_dadt / np.max(np.abs(smooth_peak_dadt))
        norm_smooth_peak_dadt = smooth_peak_dadt / np.max(np.abs(smooth_peak_dadt))

        ax.scatter(p_time, norm_tf_s, s=0.1, color="red")
        ax.plot(p_time, norm_smooth_tf_s, color="red", linewidth=3)
        ax.scatter(p_time, norm_peak_clst_dadt, s=0.1, color="blue")
        ax.plot(p_time, norm_smooth_peak_dadt, color="blue", linewidth=3)
        ax.set_ylim(-1.5, 1.5)

        ax.set_xlabel('pseudotime')
        ax.set_ylabel('norm mean expr/vel')
        ax.set_title(f'{tf}  mean expr vs {clst} mean peak vel')
        fig.tight_layout()
        plt.savefig(file_path + f"/{tf}_mean_expr_norm.png", bbox_inches='tight')
        plt.close()

        # for fig
        fig, ax = plt.subplots(figsize=(5, 5))

        ax.scatter(p_time, norm_tf_s, s=0.1, color="red")
        ax.plot(p_time, norm_smooth_tf_s, color="red", linewidth=3)
        ax.scatter(p_time, norm_peak_clst_dadt, s=0.1, color="blue")
        ax.plot(p_time, norm_smooth_peak_dadt, color="blue", linewidth=3)
        ax.set_ylim(-1.5, 1.5)

        ax.set_xlabel('pseudotime')
        ax.set_ylabel('norm mean expr/vel')
        plt.gca().axis("off")
        fig.tight_layout()
        plt.savefig(file_path + f"/{tf}_mean_expr_norm_blank.png", bbox_inches='tight', dpi=300)
        plt.close()
