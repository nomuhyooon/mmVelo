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

with open("/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/dadt_clustering/scenicplus" + "/filtered_cistromes.pkl", "rb") as tf:
    cistromes = pickle.load(tf)

target_df = list()
for key in cistromes.keys():
    tf_name = key.split("_")[0]
    if len(key.split("_")) == 2:
        motif_annot = "Direct"
    elif len(key.split("_")) == 3:
        motif_annot = "Orthology"
    else:
        motif_annot = None
        print(tf_name, "Ambiguous Annotation")
    tf_exist = tf_name in adata_r.var_names
    
    targets = cistromes[key].df
    
    targets["peak"] = targets["Chromosome"].astype("str") + ":"  \
                    + targets["Start"].astype("str") + "-" \
                    + targets["End"].astype("str")
    targets["cluster"] = adata_peak[targets["peak"], :].obs["leiden"].astype("int").to_numpy()
    targets["TF"] = tf_name
    targets["Annotation"] = motif_annot
    targets["TF_exist"] = tf_exist
    target_df.append(targets)
target_df = pd.concat(target_df) # 123184 x 7
target_df = target_df[target_df["TF_exist"] == True] # 123184 x 7

len(set(target_df["peak"])) # only 11395 peaks have putative regulator TFs

num_reg_tfs = list()
for peak in set(target_df["peak"]):
    target_df_peak = target_df[target_df["peak"] == peak]
    num_reg_tfs.append(len(set(target_df_peak["TF"])))

dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/GRN_inference_DEM"

# histogram of the number of regulator TFs for each peak
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5.0, 5.0), dpi=300)
ax.hist(num_reg_tfs, bins = max(num_reg_tfs), alpha=0.6)
ax.set_xlabel("# regulator TFs")
ax.set_ylabel("frequency")
ax.set_title("# regulator TFs for each peak")
fig.tight_layout()
plt.savefig(dir_path + "/num_tfs_for_each_peak.png",
            bbox_inches='tight', dpi=300)
plt.close("all")





for tf in set(target_df["TF"]):
    target_df_tf = target_df[target_df["TF"] == tf]
    set(target_df_tf.peak)
    break



dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/dadt_clustering"
infile = open(dir_path + '/cisTarget_dict.pkl', 'rb')
cistarget_dict = pickle.load(infile)
infile.close()

num_clusters = len(set(adata_peak.obs["leiden"]))

# foreground
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

    tf_matrix = adata_r[:, exists_tf_list].layers["spliced_count"].toarray() + adata_r[:, exists_tf_list].layers["unspliced_count"].toarray()
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

# background
num_permutations = 5
for k in range(num_permutations):
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

        tf_matrix = adata_r[:, exists_tf_list].layers["spliced_count"].toarray() + adata_r[:, exists_tf_list].layers["unspliced_count"].toarray()
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
        dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/GRN_inference"
        network.to_csv(dir_path + f'/grnboost2_network_background_{k}_clst_{cluster}.tsv', sep='\t', header=False, index=False)

dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/GRN_inference"

# check background
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
fdr_alpha = 1e-1
fdr_list = fdrcorrection(pvalue_list, alpha=fdr_alpha, method = "indep", is_sorted=False)[1]

network_df["pvalue"] = pvalue_list
network_df["qvalue"] = fdr_list
network_df.to_csv(dir_path + '/network_score_p_q_value.tsv', sep='\t', header=True, index=False)
filtered_network_df = network_df[network_df["qvalue"] <= fdr_alpha]

# calc correlation
tf_peak_corr_list = list()
for index, rows in filtered_network_df.iterrows():
    tf, peak = rows["TF"], rows["target"]
    dadt = adata_a[:, peak].layers["dadt"].reshape(-1).toarray()
    tf_expr = adata_r[:, tf].layers["spliced_count"].toarray().reshape(-1) + adata_r[:, tf].layers["unspliced_count"].toarray().reshape(-1)
    tf_dadt_corr = np.corrcoef(tf_expr, dadt)[0,1]
    tf_peak_corr_list.append(tf_dadt_corr)
filtered_network_df["correlation"] = tf_peak_corr_list
filtered_network_df.to_csv(dir_path + '/network_filtered_fdr_1e-1_bh.tsv', sep='\t', header=True, index=False)

all_tfs, all_scores = [], []
all_tfs_pos, all_tfs_neg = [], []
save_dir = dir_path+ "/num_regulated_peaks"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

plot_bar(filtered_network_df["TF"].to_list(),
        save_dir=save_dir, save_name= "/num_tf_regulated_peaks.png")
plot_hist_fg(filtered_network_df["importance"].to_list(),
        save_dir=save_dir, save_name= "/fdr_filtered_feature_importance_score.png")
print(filtered_network_df["importance"].min(), filtered_network_df["importance"].max()) # 7.700609293586442 90.18368868175855

# positively regulated
plot_bar(filtered_network_df[filtered_network_df["correlation"]>0]["TF"].to_list(),
            save_dir=save_dir, save_name= "/num_tf_regulated_peaks_positive.png")
# nagatively regulated
plot_bar(filtered_network_df[filtered_network_df["correlation"]<0]["TF"].to_list(),
            save_dir=save_dir, save_name= "/num_tf_regulated_peaks_negative.png")


# compare motif enrichment
## locally - intra cluster
save_dir = dir_path+ "/motif_enrichment_analysis_intra_cluster"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    
# motif annotation
motif_annot = pd.read_table("/home/nomura/Proj/mmvelo/pycistarget/motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl",
              header=0)
motif_annot.columns
motif_annot["#motif_id"]
motif_annot["motif_name"]
motif_annot["motif_description"]
motif_annot["gene_name"]
motif_annot["orthologous_gene_name"]
motif_annot["orthologous_identity"]
motif_annot["orthologous_species"]
motif_annot["description"]

db.db_scores.index
len(set(motif_annot["#motif_id"]) & set(db.db_scores.index))

## cluster-wise NES comparison
nes_threshold = -1.0
df = pd.DataFrame(columns=["peak_cluster", "TF", "target_NES", "nontarget_NES"])
all_targeted_peaks = list(set(filtered_network_df["target"])) # 16138
for i in range(num_clusters):
    cluster = i
    print(f"working cluster {i} ...")
    adata_peak_sub = adata_peak[adata_peak.obs["leiden"] == str(cluster), :]
    filtered_network_sub = filtered_network_df[filtered_network_df["cluster"] == cluster]
    tf_in_clst_i = list(set(filtered_network_sub["TF"]))
    
    for tf in tf_in_clst_i:
        region_sets = dict()
        target_peaks = list(set(filtered_network_sub[filtered_network_sub["TF"]==tf]["target"]))
        nontarget_peaks = list(set(adata_peak_sub.obs_names).difference(target_peaks))
        target_peaks = pr.from_dict({"Chromosome": [peak.split(':')[0] for peak in target_peaks],
                      "Start": [int(peak.split(':')[1].split('-')[0]) for peak in target_peaks],
                      "End": [int(peak.split(':')[1].split('-')[1]) for peak in target_peaks]})
        nontarget_peaks = pr.from_dict({"Chromosome": [peak.split(':')[0] for peak in nontarget_peaks],
                      "Start": [int(peak.split(':')[1].split('-')[0]) for peak in nontarget_peaks],
                      "End": [int(peak.split(':')[1].split('-')[1]) for peak in nontarget_peaks]})
        target_key = tf
        nontarget_key = "background_clst_" + tf
        region_sets[target_key] = target_peaks
        region_sets[nontarget_key] = nontarget_peaks
        
        db = DEMDatabase(fname="/home/nomura/Proj/mmvelo/scenicplus/mm10_screen_v10_clust.regions_vs_motifs.scores.feather",
                    region_sets=region_sets)
        db.db_scores
        db.regions_to_db
        break
    
    
    
        
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
        print("saving cisTarget results...")
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
df.to_csv(save_dir + f'/motif_enrichment_comparison_intra_cluster.tsv', sep='\t', header=True, index=False)


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

save_dir = dir_path+ "/motif_enrichment_analysis_intra_cluster"
df = pd.read_csv(save_dir + f'/motif_enrichment_comparison_intra_cluster.tsv', sep='\t',)

filtered_network_df

mean_importance_scores = []
for clst in set(df["peak_cluster"]):
    df_clst = df[df["peak_cluster"] == clst]
    peak_regulator_clst = filtered_network_df[filtered_network_df["cluster"] == clst]
    for tf in df_clst["TF"]:
        mean_importance = peak_regulator_clst[peak_regulator_clst["TF"] == tf]["importance"].mean()
        mean_importance_scores.append(mean_importance)
df["mean_importance"] = mean_importance_scores

num_reg_peaks = []
for tf in df["TF"]:
    print(tf)
    num_reg_peak = sum(filtered_network_df["TF"] == tf)
    num_reg_peaks.append(num_reg_peak)
df["num_reg_peaks"] = num_reg_peaks

plot_scatter(df["nontarget_NES"].to_numpy(), df["target_NES"].to_numpy(),
             save_dir, color=df["mean_importance"].to_numpy())


# compare motif enrichment globally
## inter-cluster
save_dir = dir_path+ "/motif_enrichment_analysis_inter_cluster"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# cluster-wise NES comparison
nes_threshold = -1.0
df = pd.DataFrame(columns=["peak_cluster", "TF", "target_NES", "nontarget_NES"])

filtered_network_df    
all_tfs = list(set(filtered_network_df["TF"]))

for tf in all_tfs:
    target_peaks = list(set(filtered_network_df[filtered_network_df["TF"]==tf]["target"]))
    nontarget_peaks = list(set(adata_peak.obs_names).difference(target_peaks))
    
    target_peaks = pr.from_dict({"Chromosome": [peak.split(':')[0] for peak in target_peaks],
                    "Start": [int(peak.split(':')[1].split('-')[0]) for peak in target_peaks],
                    "End": [int(peak.split(':')[1].split('-')[1]) for peak in target_peaks]})
    nontarget_peaks = pr.from_dict({"Chromosome": [peak.split(':')[0] for peak in nontarget_peaks],
                    "Start": [int(peak.split(':')[1].split('-')[0]) for peak in nontarget_peaks],
                    "End": [int(peak.split(':')[1].split('-')[1]) for peak in nontarget_peaks]})
    target_key = tf
    nontarget_key = "background_" + tf
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
df.to_csv(save_dir + f'/motif_enrichment_comparison_inter_cluster.tsv', sep='\t', header=True, index=False)

save_dir = dir_path+ "/motif_enrichment_analysis_inter_cluster"
df = pd.read_csv(save_dir + f'/motif_enrichment_comparison_inter_cluster.tsv', sep='\t',)

filtered_network_df
mean_importance_scores = []
for tf in df["TF"]:
    mean = filtered_network_df[filtered_network_df["TF"] == tf]["importance"].mean()
    mean_importance_scores.append(mean)
df["mean_importance"] = mean_importance_scores

plot_scatter(df["nontarget_NES"].to_numpy(), df["target_NES"].to_numpy(),
             save_dir, color=df["mean_importance"].to_numpy())

