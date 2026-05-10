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
from scipy.stats import ranksums

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
dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/GRN_inference"
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

dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/dadt_clustering"
infile = open(dir_path + '/cisTarget_dict.pkl', 'rb')
cistarget_dict = pickle.load(infile)
infile.close()

num_clusters = len(set(adata_peak.obs["leiden"]))
for i in range(num_clusters):
    cluster = i
    print(f"Inferring GRN in cluster {i} ...")
    tf_direct = cistarget_dict[str(cluster)].motif_enrichment["Direct_annot"].dropna().to_list()
    tf_ortho = cistarget_dict[str(cluster)].motif_enrichment["Orthology_annot"].dropna().to_list()
    tf_motif = tf_direct + tf_ortho

    exists_tf_list = []
    for tf in tf_motif:
        if tf in adata_r.var_names:
            exists_tf_list.append(tf)
    exists_tf_list = list(set(exists_tf_list))
    
    adata_peak_sub = adata_peak[adata_peak.obs["leiden"] == str(cluster), :]
    ex_matrix = adata_peak_sub.X.toarray().T
    ex_matrix = pd.DataFrame(ex_matrix,
                            columns = adata_peak_sub.obs_names.to_list(),
                            index = adata_peak_sub.var_names.to_list())

    tf_matrix = adata_r[:, exists_tf_list].layers["s_raw"].toarray()
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
    dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/GRN_inference"
    network.to_csv(dir_path + f'/grnboost2_network_clst_{cluster}.tsv', sep='\t', header=False, index=False)


adata_a.layers["dadt"] = adata_a.layers["dadt"].toarray()
adata_r.layers["s_raw"] = adata_r.layers["s_raw"].toarray()


def plot_bar(tf_list, save_dir, save_name= "/tf_regulated_peaks_count.png"):
    counter_res = collections.Counter(tf_list)
    elements, counts = zip(*counter_res.items())
    sorted_indices = sorted(range(len(counts)), key=lambda k: counts[k], reverse=True)
    sorted_elements, sorted_counts = [elements[i] for i in sorted_indices], [counts[i] for i in sorted_indices]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10.0, 10.0), dpi=300)    
    ax.bar(sorted_elements, sorted_counts)
    ax.set_ylabel("# regulated peaks")
    ax.tick_params(axis="x", labelrotation=45, labelsize=6)
    fig.tight_layout()
    plt.savefig(save_dir + save_name,
                bbox_inches='tight', dpi=300)
    plt.close("all")
    
def plot_hist(hist_list, save_dir, save_name= "/tf_importance_score.png"):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5.0, 5.0), dpi=300)    
    ax.hist(hist_list, bins = 50)
    ax.set_xlabel("feature importance score")
    ax.set_ylabel("frequency")
    fig.tight_layout()
    plt.savefig(save_dir + save_name,
                bbox_inches='tight', dpi=300)
    plt.close("all")


all_tfs, all_scores = [], []
all_tfs_pos, all_tfs_neg = [], []
save_dir = dir_path+ "/num_regulated_peaks"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
for i in range(num_clusters):
    cluster = i
    print(f"working cluster {i} ...")
    adata_peak_sub = adata_peak[adata_peak.obs["leiden"] == str(cluster), :]
    network = pd.read_csv(dir_path + f'/grnboost2_network_clst_{cluster}.tsv', sep="\t",
                header=None, names=["TF", "target", "importance"])

    counter = 0
    for peak in adata_peak_sub.obs_names:
        # take the maximum score
        peak_regulator = network[network["target"] == peak].iloc[0]
        peak_regulator = peak_regulator.to_frame().T
        peak_regulator_tf = peak_regulator["TF"].to_list()
        
        # calc corr between TF mRNA(s) expr and da/dt
        tf_dadt_corr = np.corrcoef(adata_a[:, peak].layers["dadt"].reshape(-1), 
                                y = adata_r[:, peak_regulator_tf].layers["s_raw"].reshape(-1))
        tf_dadt_corr = tf_dadt_corr[0, 1]
        peak_regulator["correlation"] = tf_dadt_corr
        
        if counter == 0:
            peak_regulators = peak_regulator.copy()
            counter += 1
        else:
            peak_regulators = pd.concat([peak_regulators, peak_regulator])
        counter += 1
    peak_regulators.to_csv(dir_path + f'/peak_regulator_clst_{cluster}.tsv', sep='\t', header=True, index=False)
    plot_bar(peak_regulators["TF"].to_list(),
             save_dir=save_dir, save_name= f"/tf_regulated_peaks_count_clst_{cluster}.png")
    plot_hist(peak_regulators["importance"].to_list(),
              save_dir=save_dir, save_name= f"/tf_importance_score_clst_{cluster}.png")
    # positively regulated
    plot_bar(peak_regulators[peak_regulators["correlation"]>0]["TF"].to_list(),
             save_dir=save_dir, save_name= f"/tf_regulated_peaks_positive_count_clst_{cluster}.png")
    # nagatively regulated
    plot_bar(peak_regulators[peak_regulators["correlation"]<0]["TF"].to_list(),
             save_dir=save_dir, save_name= f"/tf_regulated_peaks_negative_count_clst_{cluster}.png")
    
    all_tfs.append(peak_regulators["TF"].to_list())
    all_scores.append(peak_regulators["importance"].to_list())
    all_tfs_pos.append(peak_regulators[peak_regulators["correlation"] > 0]["TF"].to_list())
    all_tfs_neg.append(peak_regulators[peak_regulators["correlation"] < 0]["TF"].to_list())

peak_regulators["correlation"]
all_tfs, all_scores = sum(all_tfs, []), sum(all_scores, [])
all_tfs_pos, all_tfs_neg = sum(all_tfs_pos, []), sum(all_tfs_neg, [])
plot_bar(all_tfs, save_dir=dir_path,)
plot_bar(all_tfs_pos, save_dir=dir_path, save_name="/tf_regulated_peaks_count_positive.png")
plot_bar(all_tfs_neg, save_dir=dir_path, save_name="/tf_regulated_peaks_count_negative.png")
plot_hist(all_scores, save_dir=dir_path,)

peak_regulators[peak_regulators["TF"] == "Lhx2"].target
peak_regulators[peak_regulators["importance"] < 10]


# compare motif enrichment
save_dir = dir_path+ "/motif_enrichment_analysis_inter_cluster"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# cluster-wise NES comparison
nes_threshold = -1.0
df = pd.DataFrame(columns=["peak_cluster", "TF", "target_NES", "nontarget_NES"])
for i in range(num_clusters):
    cluster = i
    print(f"working cluster {i} ...")
    adata_peak_sub = adata_peak[adata_peak.obs["leiden"] == str(cluster), :]
    network = pd.read_csv(dir_path + f'/peak_regulator_clst_{cluster}.tsv', sep="\t",
                header=0)
    
    tf_all_in_clst = list(set(network["TF"]))
    for tf in tf_all_in_clst:
        target_peaks = network[network["TF"]==tf]["target"].to_list()
        nontarget_peaks = network[network["TF"]!=tf]["target"].to_list()
        target_peaks = pr.from_dict({"Chromosome": [peak.split(':')[0] for peak in target_peaks],
                      "Start": [int(peak.split(':')[1].split('-')[0]) for peak in target_peaks],
                      "End": [int(peak.split(':')[1].split('-')[1]) for peak in target_peaks]})
        nontarget_peaks = pr.from_dict({"Chromosome": [peak.split(':')[0] for peak in nontarget_peaks],
                      "Start": [int(peak.split(':')[1].split('-')[0]) for peak in nontarget_peaks],
                      "End": [int(peak.split(':')[1].split('-')[1]) for peak in nontarget_peaks]})
        target_key = tf
        nontarget_key = "background_clst_" + tf
        region_sets = dict()
        region_sets[target_key] = target_peaks
        region_sets[nontarget_key] = nontarget_peaks
        cistarget_dict = run_cistarget(ctx_db = '/home/nomura/Proj/mmvelo/pycistarget/mm10_screen_v10_clust.regions_vs_motifs.rankings.feather',
                                                      region_sets = region_sets,
                                                      specie = 'mus_musculus',
                                                      auc_threshold = 0.005,
                                                      nes_threshold = nes_threshold,
                                                      rank_threshold = 0.05,
                                                      annotation = ['Direct_annot', 'Orthology_annot'],
                                                      annotation_version = 'v10nr_clust',
                                                      path_to_motif_annotations = '/home/nomura/Proj/mmvelo/pycistarget/motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl',
                                                      n_cpu = 8,
                                                      #_temp_dir='/scratch/leuven/313/vsc31305/ray_spill'
                                                      )
        with open(save_dir + f'/cisTarget_dict_clst_{cluster}_{target_key}.pkl', 'wb') as f:
            pickle.dump(cistarget_dict, f)
        
        t_direct_nes_max = cistarget_dict[target_key].motif_enrichment[cistarget_dict[target_key].motif_enrichment['Direct_annot'].str.contains(tf, na=False)]["NES"]
        t_orthology_nes_max = cistarget_dict[target_key].motif_enrichment[cistarget_dict[target_key].motif_enrichment['Orthology_annot'].str.contains(tf, na=False)]["NES"]
        if np.isnan(t_direct_nes_max.max()):
            t_direct_nes_max = pd.Series(nes_threshold)
        if np.isnan(t_orthology_nes_max.max()):
            t_orthology_nes_max = pd.Series(nes_threshold)
        target_nes = max([t_direct_nes_max.max(), t_orthology_nes_max.max()])
        
        nt_direct_nes_max = cistarget_dict[nontarget_key].motif_enrichment[cistarget_dict[nontarget_key].motif_enrichment['Direct_annot'].str.contains(tf, na=False)]["NES"]
        if np.isnan(nt_direct_nes_max.max()):
            nt_direct_nes_max = pd.Series(nes_threshold)
        nt_orthology_nes_max = cistarget_dict[nontarget_key].motif_enrichment[cistarget_dict[nontarget_key].motif_enrichment['Orthology_annot'].str.contains(tf, na=False)]["NES"]
        if np.isnan(nt_orthology_nes_max.max()):
            nt_orthology_nes_max = pd.Series(nes_threshold)
        nontarget_nes = max([nt_direct_nes_max.max(), nt_orthology_nes_max.max()])
        df_tf = pd.DataFrame({"peak_cluster" : cluster, 
                              "TF" : target_key,
                              "target_NES" : target_nes,
                              "nontarget_NES" : nontarget_nes},
                             index = [str(cluster) + "_" + target_key])
        df = pd.concat([df, df_tf])

save_dir
df.to_csv(save_dir + f'/motif_enrichment_comparison_inter_cluster.tsv', sep='\t', header=True, index=False)

def plot_scatter(x, y, save_dir, save_name= "/motif_enrichment_comparison.png",
                 color=None):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.0, 5.0), dpi=300)
    if color is not None:
        cbar = ax.scatter(x, y, c=color)
        fig.colorbar(cbar, ax=ax, label="mean importance score")
    else:
        ax.scatter(x, y, c="blue")
    
    min_x, max_x = min(min(x), min(y)), max(max(x), max(y))
    x_l = np.linspace(min_x, max_x, 100)
    ax.plot(x_l, x_l, color="black")
    
    pvalue = ranksums(y, x, alternative="greater")[1] # H1: y > x
    pos_x = (max_x - min_x) * 0.5
    ax.text(pos_x+1, pos_x, f"target > background\np-value: {round(pvalue, 4)}")
    
    ax.set_xlabel("background NES")
    ax.set_ylabel("target NES")
    ax.set_title("target vs non-target peak motif NES")
    fig.tight_layout()
    plt.savefig(save_dir + save_name,
                bbox_inches='tight', dpi=300)
    plt.close("all") 

save_dir = dir_path+ "/motif_enrichment_analysis_inter_cluster"
df = pd.read_csv(save_dir + f'/motif_enrichment_comparison_inter_cluster.tsv', sep='\t',)

mean_importance_scores = []
for clst in set(df["peak_cluster"]):
    df_clst = df[df["peak_cluster"] == clst]
    peak_regulator_clst = pd.read_csv(dir_path+f"/peak_regulator_clst_{clst}.tsv", sep="\t")
    for tf in df_clst["TF"]:
        mean_importance = peak_regulator_clst[peak_regulator_clst["TF"] == tf]["importance"].mean()
        mean_importance_scores.append(mean_importance)
df["mean_importance"] = mean_importance_scores

plot_scatter(df["nontarget_NES"].to_numpy(), df["target_NES"].to_numpy(),
             save_dir, color=df["mean_importance"].to_numpy())


# compare motif enrichment globally
save_dir = dir_path+ "/motif_enrichment_analysis_intra_cluster"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# cluster-wise NES comparison
nes_threshold = -1.0
df = pd.DataFrame(columns=["peak_cluster", "TF", "target_NES", "nontarget_NES"])

network_df = []
for i in range(num_clusters):
    cluster = i
    print(f"loading cluster {i} ...")
    network = pd.read_csv(dir_path + f'/peak_regulator_clst_{cluster}.tsv', sep="\t",
                header=0)
    network_df.append(network)
    
network_df = pd.concat(network_df)
all_tfs = list(set(network_df["TF"]))

for tf in all_tfs:
    target_peaks = network_df[network_df["TF"]==tf]["target"].to_list()
    nontarget_peaks = network_df[network_df["TF"]!=tf]["target"].to_list()
    target_peaks = pr.from_dict({"Chromosome": [peak.split(':')[0] for peak in target_peaks],
                    "Start": [int(peak.split(':')[1].split('-')[0]) for peak in target_peaks],
                    "End": [int(peak.split(':')[1].split('-')[1]) for peak in target_peaks]})
    nontarget_peaks = pr.from_dict({"Chromosome": [peak.split(':')[0] for peak in nontarget_peaks],
                    "Start": [int(peak.split(':')[1].split('-')[0]) for peak in nontarget_peaks],
                    "End": [int(peak.split(':')[1].split('-')[1]) for peak in nontarget_peaks]})
    target_key = tf
    nontarget_key = "background_clst_" + tf
    region_sets = dict()
    region_sets[target_key] = target_peaks
    region_sets[nontarget_key] = nontarget_peaks
    cistarget_dict = run_cistarget(ctx_db = '/home/nomura/Proj/mmvelo/pycistarget/mm10_screen_v10_clust.regions_vs_motifs.rankings.feather',
                                                    region_sets = region_sets,
                                                    specie = 'mus_musculus',
                                                    auc_threshold = 0.005,
                                                    nes_threshold = nes_threshold,
                                                    rank_threshold = 0.05,
                                                    annotation = ['Direct_annot', 'Orthology_annot'],
                                                    annotation_version = 'v10nr_clust',
                                                    path_to_motif_annotations = '/home/nomura/Proj/mmvelo/pycistarget/motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl',
                                                    n_cpu = 8,
                                                    #_temp_dir='/scratch/leuven/313/vsc31305/ray_spill'
                                                    )
    with open(save_dir + f'/cisTarget_dict_{target_key}.pkl', 'wb') as f:
        pickle.dump(cistarget_dict, f)
    
    t_direct_nes_max = cistarget_dict[target_key].motif_enrichment[cistarget_dict[target_key].motif_enrichment['Direct_annot'].str.contains(tf, na=False)]["NES"]
    t_orthology_nes_max = cistarget_dict[target_key].motif_enrichment[cistarget_dict[target_key].motif_enrichment['Orthology_annot'].str.contains(tf, na=False)]["NES"]
    if np.isnan(t_direct_nes_max.max()):
        t_direct_nes_max = pd.Series(nes_threshold)
    if np.isnan(t_orthology_nes_max.max()):
        t_orthology_nes_max = pd.Series(nes_threshold)
    target_nes = max([t_direct_nes_max.max(), t_orthology_nes_max.max()])
    
    nt_direct_nes_max = cistarget_dict[nontarget_key].motif_enrichment[cistarget_dict[nontarget_key].motif_enrichment['Direct_annot'].str.contains(tf, na=False)]["NES"]
    if np.isnan(nt_direct_nes_max.max()):
        nt_direct_nes_max = pd.Series(nes_threshold)
    nt_orthology_nes_max = cistarget_dict[nontarget_key].motif_enrichment[cistarget_dict[nontarget_key].motif_enrichment['Orthology_annot'].str.contains(tf, na=False)]["NES"]
    if np.isnan(nt_orthology_nes_max.max()):
        nt_orthology_nes_max = pd.Series(nes_threshold)
    nontarget_nes = max([nt_direct_nes_max.max(), nt_orthology_nes_max.max()])
    df_tf = pd.DataFrame({"peak_cluster" : cluster, 
                            "TF" : target_key,
                            "target_NES" : target_nes,
                            "nontarget_NES" : nontarget_nes},
                            index = [str(cluster) + "_" + target_key])
    df = pd.concat([df, df_tf])

save_dir
df.to_csv(save_dir + f'/motif_enrichment_comparison_intra_cluster.tsv', sep='\t', header=True, index=False)

save_dir = dir_path+ "/motif_enrichment_analysis_intra_cluster"
df = pd.read_csv(save_dir + f'/motif_enrichment_comparison_intra_cluster.tsv', sep='\t',)

mean_importance_scores = []
for tf in df["TF"]:
    mean = network_df[network_df["TF"] == tf]["importance"].mean()
    mean_importance_scores.append(mean)
df["mean_importance"] = mean_importance_scores

plot_scatter(df["nontarget_NES"].to_numpy(), df["target_NES"].to_numpy(),
             save_dir, color=df["mean_importance"].to_numpy())

