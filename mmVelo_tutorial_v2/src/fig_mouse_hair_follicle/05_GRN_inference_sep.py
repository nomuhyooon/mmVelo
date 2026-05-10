import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
from scipy.io import mmwrite, mmread
import seaborn as sns
import scipy
from arboreto.algo import grnboost2
import pickle
import collections
import pycistarget
import pyranges as pr
import pickle
from pycistarget.motif_enrichment_cistarget import *
from pycistarget.motif_enrichment_dem import *
from scipy.stats import ranksums
from statsmodels.stats.multitest import fdrcorrection

np.random.seed(42)


# load anndata
dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/anndata"
adata_r = sc.read_loom(dir_path + "/adata_rna.loom", obs_names="obs_names", var_names="var_names")
adata_a = sc.read_loom(dir_path + "/adata_atac.loom", obs_names="obs_names", var_names="var_names")
pseudotime = pd.read_csv(dir_path+"/pseudotime.tsv", sep="\t", header=None)[0].to_numpy()
adata_r.obs["pseudotime"] = pseudotime
refined_clusters = pd.read_csv(dir_path+"/refined_clusters.tsv", sep="\t", header=None)[0].to_numpy()
adata_r.obs["ref_clusters"] = refined_clusters

# make dir
dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/GRN_inference_sep"
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

# cluster-wise pseudotime
# maybe this should be done with TAC-subclustering i.e. IRS-TAC, HS-CC TAC
#set(adata_r.obs["clusters"])
#cell_clusters = ['TAC', 'Inner Root Sheath', 'Hair Shaft-Cuticle/Cortex', 'Medulla']
set(adata_r.obs["ref_clusters"])
cell_clusters = ['TAC', 'IRS-TAC', 'Inner Root Sheath', 'HS-TAC', 'Hair Shaft-Cuticle/Cortex', 'Medulla']
sorted_cell_barcode = []
for cell_cluster in cell_clusters:
    cell_cluster
    adata_r_cluster = adata_r[adata_r.obs["ref_clusters"] == cell_cluster, :]
    sorted_cells = list(adata_r_cluster.obs_names[np.argsort(adata_r_cluster.obs["pseudotime"])])
    sorted_cell_barcode.append(sorted_cells)
sorted_cell_barcode = sum(sorted_cell_barcode,[])
adata_r = adata_r[sorted_cell_barcode, :]
adata_a = adata_a[sorted_cell_barcode, :]


# load peak-clustered data
dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/dadt_clustering"
adata_peak = sc.read_loom(dir_path+"/adata_dadt_cluster.loom")
adata_peak.obs_names = adata_peak.obs["obs_names"]
adata_peak.var_names = adata_a.obs_names
peaks = list(adata_peak.obs_names)

dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/dadt_clustering/scenicplus_sep"
num_clusters = len(set(adata_peak.obs["leiden"]))
# foreground
for i in range(num_clusters):
    cluster = i    
    print(f"Inferring GRN in cluster {i} ...")
    dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/dadt_clustering/scenicplus_sep"
    with open(dir_path + "/exist_tf_motifcluster_{}.txt".format(cluster), "r") as output:
        exists_tf_list = output.read()
    exists_tf_list = exists_tf_list.replace("[", "").replace("]", "").replace("'", "").replace(",", "").split()
    exists_tf_list = list(set(exists_tf_list))
    
    adata_peak_sub = adata_peak[adata_peak.obs["leiden"] == str(cluster), :]
    ex_matrix = adata_peak_sub.X.toarray().T
    ex_matrix = pd.DataFrame(ex_matrix,
                            columns = adata_peak_sub.obs_names.to_list(),
                            index = adata_peak_sub.var_names.to_list())

    tf_matrix = np.log1p(
        np.exp(adata_r[:, exists_tf_list].layers["s_raw"].toarray()) - 1 \
        + np.exp(adata_r[:, exists_tf_list].layers["u_raw"].toarray()) - 1
    )
    
    tf_matrix = (tf_matrix - np.mean(tf_matrix, axis=0)) / np.std(tf_matrix, axis=0)
    tf_matrix = pd.DataFrame(tf_matrix, 
                            columns = exists_tf_list,
                            index = adata_r.obs_names.to_list())

    ex_matrix = ex_matrix.join(tf_matrix)
    tf_names = exists_tf_list
    
    print(f"# peaks : ", adata_peak_sub.n_obs)
    print(f"# TFs : ", len(exists_tf_list))

    network = grnboost2(expression_data = ex_matrix,
                        tf_names = tf_names)
    
    target_peak_idx = []
    for j, target in enumerate(network["target"]):
        if not target in tf_names:
            target_peak_idx.append(j)
    network = network.iloc[target_peak_idx]
    network = network.sort_values(by=["target", "importance"], ascending=False)

    print("saving results ...")
    dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/GRN_inference_sep"
    network.to_csv(dir_path + f'/grnboost2_network_clst_{cluster}.tsv', sep='\t', header=False, index=False)

# background
num_permutations = 5
for k in range(num_permutations):
    for i in range(num_clusters):
        cluster = i    
        print(f"Inferring background round {k} in cluster {i} ...")
        dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/dadt_clustering/scenicplus_sep"
        with open(dir_path + "/exist_tf_motifcluster_{}.txt".format(cluster), "r") as output:
            exists_tf_list = output.read()
        exists_tf_list = exists_tf_list.replace("[", "").replace("]", "").replace("'", "").replace(",", "").split()
        exists_tf_list = list(set(exists_tf_list))
        
        adata_peak_sub = adata_peak[adata_peak.obs["leiden"] == str(cluster), :]
        ex_matrix = adata_peak_sub.X.toarray().T
        ex_matrix = pd.DataFrame(ex_matrix,
                                columns = adata_peak_sub.obs_names.to_list(),
                                index = adata_peak_sub.var_names.to_list())

        tf_matrix = np.log1p(
            np.exp(adata_r[:, exists_tf_list].layers["s_raw"].toarray()) - 1 \
            + np.exp(adata_r[:, exists_tf_list].layers["u_raw"].toarray()) - 1
        )
        
        tf_matrix = (tf_matrix - np.mean(tf_matrix, axis=0)) / np.std(tf_matrix, axis=0)
        
        permutation_index = np.random.permutation(adata_r.n_obs)
        tf_matrix_permuted = tf_matrix[permutation_index, :]
        tf_matrix_permuted = pd.DataFrame(tf_matrix_permuted, 
                                columns = exists_tf_list,
                                index = adata_r.obs_names.to_list())
        
        ex_matrix = ex_matrix.join(tf_matrix_permuted)
        tf_names = exists_tf_list
        
        print(f"# peaks : ", adata_peak_sub.n_obs)
        print(f"# TFs : ", len(exists_tf_list))

        network = grnboost2(expression_data = ex_matrix,
                            tf_names = tf_names)
        
        target_peak_idx = []
        for j, target in enumerate(network["target"]):
            if not target in tf_names:
                target_peak_idx.append(j)
        network = network.iloc[target_peak_idx]
        network = network.sort_values(by=["target", "importance"], ascending=False)

        print("saving results ...")
        dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/GRN_inference_sep"
        network.to_csv(dir_path + f'/grnboost2_network_background_{k}_clst_{cluster}.tsv', sep='\t', header=False, index=False)

dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/GRN_inference_sep"

# check background
num_permutations = 5
bg_importance_score = []
for k in range(num_permutations):
    print(f"loading permutation {k} ...")
    bg_network_df = []
    for i in range(num_clusters):
        cluster = i
        print(f"loading cluster {i} ...")
        network = pd.read_csv(dir_path + f'/grnboost2_network_background_{k}_clst_{cluster}.tsv', sep="\t",
                    header=None)
        bg_network_df.append(network)
    bg_network_df = pd.concat(bg_network_df)
    bg_network_df.columns = ["TF", "target", "importance"]
    bg_importance_score.append(bg_network_df["importance"].to_list())

def plot_hist_bg(hist_list, save_dir, save_name= "/tf_bg_importance_score.png"):
    num_permutations = len(hist_list)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5.0, 5.0), dpi=300)
    for k in range(num_permutations):
        ax.hist(hist_list[k], bins = 50, alpha=0.2)
    ax.set_xlabel("feature importance score")
    ax.set_ylabel("frequency")
    ax.set_title("background importance score")
    fig.tight_layout()
    plt.savefig(save_dir + save_name,
                bbox_inches='tight', dpi=300)
    plt.close("all")

plot_hist_bg(bg_importance_score, dir_path)

# check foreground
network_df = []
for i in range(num_clusters):
    cluster = i
    print(f"loading cluster {i} ...")
    network = pd.read_csv(dir_path + f'/grnboost2_network_clst_{cluster}.tsv', sep="\t",
                header=None)
    network["cluster"] = i
    network_df.append(network)
network_df = pd.concat(network_df)
network_df.columns = ["TF", "target", "importance", "cluster"]

adata_a.layers["dadt"] = adata_a.layers["dadt"].toarray()
adata_r.layers["s_raw"] = adata_r.layers["s_raw"].toarray()
adata_r.layers["u_raw"] = adata_r.layers["u_raw"].toarray()

def plot_bar(tf_list, save_dir, save_name= "/tf_regulated_peaks_count.png",
             label_size = 5):
    counter_res = collections.Counter(tf_list)
    elements, counts = zip(*counter_res.items())
    sorted_indices = sorted(range(len(counts)), key=lambda k: counts[k], reverse=True)
    sorted_elements, sorted_counts = [elements[i] for i in sorted_indices], [counts[i] for i in sorted_indices]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10.0, 10.0), dpi=300)    
    ax.bar(sorted_elements, sorted_counts)
    ax.set_ylabel("# regulated peaks")
    ax.tick_params(axis="x", labelrotation=45, labelsize=label_size)
    fig.tight_layout()
    plt.savefig(save_dir + save_name,
                bbox_inches='tight', dpi=300)
    plt.close("all")
    
def plot_hist_fg(hist_list, save_dir, save_name= "/tf_fg_importance_score.png"):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5.0, 5.0), dpi=300)    
    ax.hist(hist_list, bins = 50)
    ax.set_xlabel("feature importance score")
    ax.set_ylabel("frequency")
    fig.tight_layout()
    plt.savefig(save_dir + save_name,
                bbox_inches='tight', dpi=300)
    plt.close("all")
    
plot_hist_fg(network_df["importance"].to_list(), dir_path)

def plot_hist_fg_vs_bg(bg_hist_list, fg_hist_lidt, save_dir, save_name= "/tf_fg_vs_bg_importance_score.png"):    
    num_permutations = len(bg_hist_list)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5.0, 5.0), dpi=300)
    ax.hist(bg_hist_list, bins = 50, alpha=0.6)
    ax.hist(fg_hist_lidt, bins = 50, alpha=0.6)
    ax.set_xlabel("feature importance score")
    ax.set_ylabel("frequency")
    fig.tight_layout()
    plt.savefig(save_dir + save_name,
                bbox_inches='tight', dpi=300)
    plt.close("all")

bg_importance_score = sum(bg_importance_score, [])
fg_importance_score = network_df["importance"].to_list()
plot_hist_fg_vs_bg(bg_importance_score, fg_importance_score, dir_path)


# calculate p-value
num_bgs = len(bg_importance_score)
bg_importance_score = np.array(bg_importance_score)
pvalue_list = list()
counter = 0
for score in network_df["importance"]:
    pvalue = np.sum(bg_importance_score >= score)
    if pvalue == 0:
        pvalue = 1 / num_bgs
    else:
        pvalue = pvalue / num_bgs
    pvalue_list.append(pvalue)
    counter += 1
    if counter % 1000 == 0:
        print(counter)
pvalue_list
fdr_alpha = 1e-3
fdr_list = fdrcorrection(pvalue_list, alpha=fdr_alpha, method = "indep", is_sorted=False)[1]

network_df["pvalue"] = pvalue_list
network_df["qvalue"] = fdr_list
network_df.to_csv(dir_path + '/network_score_p_q_value.tsv', sep='\t', header=True, index=False)
filtered_network_df = network_df[network_df["qvalue"] <= fdr_alpha]
print(len(network_df), len(filtered_network_df)) # 903336, 101644

# calc correlation
tf_peak_corr_list = list()
for index, rows in filtered_network_df.iterrows():
    tf, peak = rows["TF"], rows["target"]
    dadt = adata_a[:, peak].layers["dadt"].reshape(-1).toarray()
    tf_expr = adata_r[:, tf].layers["spliced_count"].toarray().reshape(-1) + adata_r[:, tf].layers["unspliced_count"].toarray().reshape(-1)
    tf_expr = np.log1p(
            np.exp(adata_r[:, tf].layers["s_raw"].reshape(-1)) - 1 \
            + np.exp(adata_r[:, tf].layers["u_raw"].reshape(-1)) - 1
        )
    
    tf_dadt_corr = np.corrcoef(tf_expr, dadt)[0,1]
    tf_peak_corr_list.append(tf_dadt_corr)
filtered_network_df["correlation"] = tf_peak_corr_list
filtered_network_df.to_csv(dir_path + '/network_filtered_fdr_1e-4_bh.tsv', sep='\t', header=True, index=False)

save_dir = dir_path+ "/num_regulated_peaks"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

plot_bar(filtered_network_df["TF"].to_list(),
        save_dir=save_dir, save_name= "/num_tf_regulated_peaks.png", label_size=4)
plot_hist_fg(filtered_network_df["importance"].to_list(),
        save_dir=save_dir, save_name= "/fdr_filtered_feature_importance_score.png")
print(filtered_network_df["importance"].min(), filtered_network_df["importance"].max()) # 7.700609293586442 90.18368868175855

# positively regulated
plot_bar(filtered_network_df[filtered_network_df["correlation"]>0]["TF"].to_list(),
            save_dir=save_dir, save_name= "/num_tf_regulated_peaks_positive.png", label_size=4)
# nagatively regulated
plot_bar(filtered_network_df[filtered_network_df["correlation"]<0]["TF"].to_list(),
            save_dir=save_dir, save_name= "/num_tf_regulated_peaks_negative.png", label_size=4)


# compare motif enrichment
## locally - intra cluster
save_dir = dir_path+ "/motif_enrichment_analysis_intra_cluster"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


from pycistarget.motif_enrichment_dem import *
from pycistarget.utils import load_motif_annotations

region_sets = dict()
pr_peaks = pr.from_dict({"Chromosome": [peak.split(':')[0] for peak in list(adata_peak.obs_names)],
                        "Start": [int(peak.split(':')[1].split('-')[0]) for peak in list(adata_peak.obs_names)],
                        "End": [int(peak.split(':')[1].split('-')[1]) for peak in list(adata_peak.obs_names)]})
region_sets["peaks"] = pr_peaks

dem_db = DEMDatabase('/home/nomura/Proj/mmvelo/scenicplus/mm10_screen_v10_clust.regions_vs_motifs.scores.feather',
            region_sets = region_sets,
            fraction_overlap = 0.4
            )
dem_db.db_scores

# TF to motif annotation
motif_annot_df = load_motif_annotations(specie = 'mus_musculus',
                       version = 'v10nr_clust',
                       fname = '/home/nomura/Proj/mmvelo/pycistarget/motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl',
                       )

all_tfs = list(set(filtered_network_df.TF))
motif_annot_df = motif_annot_df.fillna("").drop(columns = ["Motif_similarity_annot", "Motif_similarity_and_Orthology_annot"])

motif_tf_dict = dict()
for motif in motif_annot_df.index:
    tf_list = list(motif_annot_df.loc[motif])
    tf_direct, tf_ortho = tf_list[0], tf_list[1]    
    
    if tf_direct == "":
        tf_direct = []
    elif ", " in tf_direct:
        tf_direct = tf_direct.split(", ")
    else:
        tf_direct = tf_direct.split()
        
    if tf_ortho == "":
        tf_ortho = []
    elif ", " in tf_ortho:
        tf_ortho = tf_ortho.split(", ")
    else:
        tf_ortho = tf_ortho.split()    
    
    tf_motif = tf_direct + tf_ortho    
    motif_tf_dict[motif] = tf_motif

df_tf_motif = pd.DataFrame(columns=["motif", "TF"])

for key in motif_tf_dict.keys():
    tfs = motif_tf_dict[key]
    if len(tfs) == 0:
        continue
    motif = [key for i in range(len(tfs))]
    df = pd.DataFrame(list(zip(motif, tfs)), columns=["motif", "TF"])
    df_tf_motif = pd.concat([df_tf_motif, df])

df_tf_motif = df_tf_motif[df_tf_motif.motif.isin(dem_db.db_scores.index)]
dem_db.db_scores

target_dict = dict()
nontarget_dict = dict()
for cluster in set(filtered_network_df.cluster):
    cluster_peaks = set(adata_peak[adata_peak.obs.leiden == str(cluster), :].obs_names)
    f_network_cluster = filtered_network_df[filtered_network_df.cluster == cluster]
    target_dict[f"cluster_{cluster}"], nontarget_dict[f"cluster_{cluster}"] = dict(), dict()
    for tf in set(f_network_cluster.TF):
        target_peaks = set(f_network_cluster[f_network_cluster.TF == tf]["target"])
        nontarget_peaks = cluster_peaks.difference(target_peaks)
        n_target_peaks, n_nontarget_peaks = len(target_peaks), len(nontarget_peaks)
        print(tf, n_target_peaks, n_nontarget_peaks)    
        target_dict[f"cluster_{cluster}"][tf], nontarget_dict[f"cluster_{cluster}"][tf] = dict(), dict()
        target_query_peaks = dem_db.regions_to_db["peaks"][dem_db.regions_to_db["peaks"]["Target"].isin(target_peaks)]["Query"]
        nontarget_query_peaks = dem_db.regions_to_db["peaks"][dem_db.regions_to_db["peaks"]["Target"].isin(nontarget_peaks)]["Query"]            
        tf_motif = list(set(df_tf_motif[df_tf_motif.TF == tf].motif))
        for motif in tf_motif:
            db_score_motif = dem_db.db_scores.loc[motif,:]
            target_dict[f"cluster_{cluster}"][tf][motif] = db_score_motif[target_query_peaks].to_list()
            nontarget_dict[f"cluster_{cluster}"][tf][motif] = db_score_motif[nontarget_query_peaks].to_list()

target_crm_mean, nontarget_crm_mean = list(), list()
for cluster in target_dict.keys():
    for tf in target_dict[cluster].keys():
        for motif in target_dict[cluster][tf].keys():
            target_crm_mean.append(np.mean(target_dict[cluster][tf][motif]))
            nontarget_crm_mean.append(np.mean(nontarget_dict[cluster][tf][motif]))
    
def plot_scatter(target, nontarget, save_dir, save_name= "/motif_enrichment_comparison.png",s=1,
                 text = True):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5.0, 5.0), dpi=300)
    ax.scatter(nontarget, target, c="blue", s=s)
    
    min_x, max_x = min(min(nontarget), min(target)), max(max(nontarget), max(target))
    x_l = np.linspace(min_x, max_x, 100)
    ax.plot(x_l, x_l, color="black")
    
    pvalue = ranksums(target, nontarget, alternative="greater")[1] # H1: y > x
    pos_x = (max_x - min_x) * 0.5
    if text:
        ax.text(pos_x+1, pos_x, f"target > background\np-value: {round(pvalue, 4)}")
    
    ax.set_xlabel("background CRM")
    ax.set_ylabel("target CRM")
    ax.set_title("target vs non-target peak motif CRM")
    fig.tight_layout()
    plt.savefig(save_dir + save_name,
                bbox_inches='tight', dpi=300)
    plt.close("all") 

plot_scatter(target_crm_mean, nontarget_crm_mean, dir_path,
             save_name="/motif_enrichment_comparison_all_motifs.png", text=True)
plot_scatter(target_crm_mean, nontarget_crm_mean, dir_path,
             save_name="/motif_enrichment_comparison_all_motifs_wo_text.png", text=False)

target_crm_mean_max, nontarget_crm_mean_max = list(), list()
for cluster in target_dict.keys():
    for tf in target_dict[cluster].keys():
        target_tf_motif_mean_list, nontarget_tf_motif_mean_list = list(), list()
        for motif in target_dict[cluster][tf].keys():
            target_tf_motif_mean_list.append(np.mean(target_dict[cluster][tf][motif]))
            nontarget_tf_motif_mean_list.append(np.mean(nontarget_dict[cluster][tf][motif]))
        target_crm_mean_max.append(max(target_tf_motif_mean_list))
        nontarget_crm_mean_max.append(max(nontarget_tf_motif_mean_list))

plot_scatter(target_crm_mean_max, nontarget_crm_mean_max, dir_path, s=5,
             save_name="/motif_enrichment_comparison_max_motifs.png", text=True)
plot_scatter(target_crm_mean_max, nontarget_crm_mean_max, dir_path, s=5,
             save_name="/motif_enrichment_comparison_max_motifs_wo_text.png", text=False)

target_dict[cluster][tf]
motif_annot_df



# calculate genomic distance
import random

chromosome = list()
start_list, end_list = list(), list()
for item in filtered_network_df.target.str.split(":"):
    chromosome.append(item[0])
    start, end = item[1].split("-")
    start_list.append(start)
    end_list.append(end)
filtered_network_df["chromosome"] = chromosome
filtered_network_df["start"] = start_list
filtered_network_df["end"] = end_list

allpeak_df = pd.DataFrame(data=list(adata_a.var_names), columns=["peak"])
chromosome = list()
start_list, end_list = list(), list()
for item in allpeak_df.peak.str.split(":"):
    chromosome.append(item[0])
    start, end = item[1].split("-")
    start_list.append(start)
    end_list.append(end)
allpeak_df["chromosome"] = chromosome
allpeak_df["start"] = start_list
allpeak_df["end"] = end_list

def find_closest_value(target, values):
    closest_value = min(values, key=lambda x: abs(x - target))
    return closest_value



tf_target_dist_dict = dict()
nontarget_num_iter = 10
for tf in set(filtered_network_df.TF):
    tf_target_dist_dict[tf] = dict()
    target_dist_list, nontarget_dist_list = list(), list()
    tf_target_df = filtered_network_df[filtered_network_df.TF == tf]
    print(tf, len(tf_target_df))
    chromosome, num_target = collections.Counter(list(tf_target_df.chromosome)).most_common()[0]
    target_peaks = list(tf_target_df[tf_target_df["chromosome"] == chromosome].target)
    start_list = list(tf_target_df[tf_target_df["chromosome"] == chromosome].start.astype("int"))
    for peak in target_peaks:
        start = int(peak.split(":")[1].split("-")[0])
        other_starts = [value for value in start_list if value != start]
        nearest_start = find_closest_value(start, other_starts)
        distance = abs(start - nearest_start)
        target_dist_list.append(distance)
    tf_target_dist_dict[tf]["target"] = target_dist_list
    
    for i in range(nontarget_num_iter):
        nontarget_peaks = list(allpeak_df[allpeak_df["chromosome"] == chromosome].peak)
        nontarget_peaks = random.sample(nontarget_peaks, k=num_target)
        nontarget_start_list = [int(peak.split(":")[1].split("-")[0]) for peak in nontarget_peaks]
        for peak in nontarget_peaks:
            start = int(peak.split(":")[1].split("-")[0])
            other_starts = [value for value in nontarget_start_list if value != start]
            nearest_start = find_closest_value(start, other_starts)
            distance = abs(start - nearest_start)
            nontarget_dist_list.append(distance)
    tf_target_dist_dict[tf]["nontarget"] = nontarget_dist_list

target_dist_mean_list = list()
nontarget_dist_mean_list = list()
for tf in tf_target_dist_dict.keys():
    target_dist_mean = np.mean(np.log(tf_target_dist_dict[tf]["target"]))
    nontarget_dist_mean = np.mean(np.log(tf_target_dist_dict[tf]["nontarget"]))
    target_dist_mean_list.append(target_dist_mean)
    nontarget_dist_mean_list.append(nontarget_dist_mean)
    
def plot_scatter(target, nontarget, save_dir, save_name= "/target_peak_dist_comparison.png"
                 ,s=1, text=True):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5.0, 5.0), dpi=300)
    ax.scatter(nontarget, target, c="blue", s=s)
    
    min_x, max_x = min(min(nontarget), min(target)), max(max(nontarget), max(target))
    x_l = np.linspace(min_x, max_x, 100)
    ax.plot(x_l, x_l, color="black")
    
    pvalue = ranksums(target, nontarget, alternative="less")[1] # H1: y < x
    pos_x = min_x + (max_x- min_x) * 0.5
    
    if text:
        ax.text(pos_x+1, pos_x, f"target > background\np-value: {round(pvalue, 4)}")
    
    ax.set_xlabel("background log distance")
    ax.set_ylabel("target log distance")
    ax.set_title("target vs non-target peak distance")
    fig.tight_layout()
    plt.savefig(save_dir + save_name,
                bbox_inches='tight', dpi=300)
    plt.close("all") 

plot_scatter(target_dist_mean_list, nontarget_dist_mean_list, dir_path,
             save_name="/target_peak_dist_comparison.png", s=5)
plot_scatter(target_dist_mean_list, nontarget_dist_mean_list, dir_path,
             save_name="/target_peak_dist_comparison_wo_text.png", s=5, text=False)

tf_target_dist_dict[tf]["target"]

## compare relative frequency
bins = [0, 1000, 10000, 100000, 1000000, float("inf")]

df_rfreq = pd.DataFrame(columns=[f"{bins[i]} - {bins[i+1]}" for i in range(len(bins)) if i < len(bins) - 1])
for tf in tf_target_dist_dict.keys():
    hist, bin_edges = np.histogram(tf_target_dist_dict[tf]["target"], bins)
    total_num = len(tf_target_dist_dict[tf]["target"])
    relative_freq_target = (hist+1) / (total_num+5) # add 1 to avoid zero_division
    
    hist, bin_edges = np.histogram(tf_target_dist_dict[tf]["nontarget"], bins)
    total_num = len(tf_target_dist_dict[tf]["nontarget"])
    relative_freq_nontarget = (hist+1) / (total_num+5) # add 1 to avoid zero_division
    
    rfreq_target_vs_nontarget = relative_freq_target / relative_freq_nontarget
    rfreq_target_vs_nontarget = pd.DataFrame(rfreq_target_vs_nontarget.reshape(1, len(rfreq_target_vs_nontarget)),
                                             columns=[f"{bins[i]} - {bins[i+1]}" for i in range(len(bins)) if i < len(bins) - 1],
                                             index=[tf])
    df_rfreq = pd.concat([df_rfreq, rfreq_target_vs_nontarget])
df_rfreq = np.log(df_rfreq)



import seaborn as sns
def plot_box(df_rfreq, save_dir, save_name= "/target_peak_dist_boxplot.png",):
    df_rfreq_melt = pd.melt(df_rfreq)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.0, 5.0), dpi=300)
    sns.boxplot(x="variable", y="value", data=df_rfreq_melt)
    sns.stripplot(x="variable", y="value", data=df_rfreq_melt, color="black", s=3)
    
    ax.set_xlabel("distance")
    ax.set_ylabel("log relative frequency")
    ax.set_title("target vs non-target relative frequency")
    fig.tight_layout()
    plt.xticks(rotation=15)
    plt.savefig(save_dir + save_name,
                bbox_inches='tight', dpi=300)
    plt.close("all")
plot_box(df_rfreq, dir_path, save_name= "/target_peak_dist_boxplot.png",) 