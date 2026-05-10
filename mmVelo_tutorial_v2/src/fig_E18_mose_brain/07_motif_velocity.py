import os
import json
import numpy as np
import pandas as pd
from scipy.io import mmread
import scanpy as sc
import scvelo as scv
import matplotlib.pyplot as plt
import seaborn as sns


# load anndata
dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/anndata"
adata_r = sc.read_loom(dir_path + "/adata_rna.loom", obs_names="obs_names", var_names="var_names")
adata_a = sc.read_loom(dir_path + "/adata_atac.loom", obs_names="obs_names", var_names="var_names")

adata_r.obsm["X_umap"]  = pd.read_csv(dir_path + "/umap_coordinate.tsv", sep="\t", header=None).to_numpy()
adata_r.obs["clusters"] = pd.read_json(dir_path + "/cell_clusters.json", typ="series").astype("category")
adata_r.obs["pseudotime"] = pd.read_csv(dir_path + "/pseudotime.tsv", sep="\t", header=None)[0].to_numpy()

# motif velocity
dir_path = "/home/nomura/Proj/mmvelo/data/10x_multiome_brain_repre_wo_IN/motif_score/"
motif_name = pd.read_csv(dir_path + "motif_names.tsv", sep="\t", header=None)[0]
motif_score_mat = mmread(dir_path + "motif_score.mtx").tocsr()
motif_bool_mat = mmread(dir_path + "motif_bool.mtx").tocsr()

# make dir
dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/motif_velocity"
if not os.path.exists(dir_path):
    os.mkdir(dir_path)




def plot_motif_umap(adata_r, adata_a, motif_idx, save_dir):
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(5.0 * 4, 5.0 * 1))

    var_name = motif_name[motif_idx]
    motif_score = (adata_a.layers["a_raw"] @ motif_score_mat[:, motif_idx]).toarray().reshape(-1)
    #motif_score = np.log10(motif_score)
    mappable = axes[0].scatter(x = adata_r.obsm["X_umap"][:, 0], y = adata_r.obsm["X_umap"][:, 1], 
                               s = 1, c = motif_score, )
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")
    axes[0].set_title(var_name + " motif score")
    cbar = fig.colorbar(mappable, ax=axes[0])

    d_motif_score = (adata_a.layers["dadt"] @ motif_score_mat[:, motif_idx]).toarray().reshape(-1)
    mappable = axes[1].scatter(x = adata_r.obsm["X_umap"][:, 0], y = adata_r.obsm["X_umap"][:, 1], 
                               s = 1, c = d_motif_score, 
                               cmap="coolwarm", vmin=-np.max(np.abs(d_motif_score)), vmax=np.max(np.abs(d_motif_score)))
    axes[1].set_xlabel("UMAP 1")
    axes[1].set_ylabel("UMAP 2")
    axes[1].set_title(var_name + " motif score velocity")
    cbar = fig.colorbar(mappable, ax=axes[1])


    motif_num = (adata_a.layers["a_raw"] @ motif_bool_mat[:, motif_idx]).toarray().reshape(-1)
    mappable = axes[2].scatter(x = adata_r.obsm["X_umap"][:, 0], y = adata_r.obsm["X_umap"][:, 1], 
                               s = 1, c = motif_num, )
    axes[2].set_xlabel("UMAP 1")
    axes[2].set_ylabel("UMAP 2")
    axes[2].set_title(var_name + " motif count")
    cbar = fig.colorbar(mappable, ax=axes[2])

    d_motif_score = (adata_a.layers["dadt"] @ motif_bool_mat[:, motif_idx]).toarray().reshape(-1)
    mappable = axes[3].scatter(x = adata_r.obsm["X_umap"][:, 0], y = adata_r.obsm["X_umap"][:, 1], 
                               s = 1, c = d_motif_score, 
                               cmap="coolwarm", vmin=-np.max(np.abs(d_motif_score)), vmax=np.max(np.abs(d_motif_score)))
    axes[3].set_xlabel("UMAP 1")
    axes[3].set_ylabel("UMAP 2")
    axes[3].set_title(var_name + " motif count velocity")
    cbar = fig.colorbar(mappable, ax=axes[3])

    fig.tight_layout()
    plt.savefig(save_dir+ "/motif_" + var_name + ".png")
    plt.close()

save_dir = dir_path
gene_name = "Neurod2"
print(np.where(motif_name == gene_name)) # 731, 781
plot_motif_umap(adata_r, adata_a, 731, save_dir)
plot_motif_umap(adata_r, adata_a, 781, save_dir)

gene_name = "Rbpj"
print(np.where(motif_name == "Rbpjl")) # 497
plot_motif_umap(adata_r, adata_a, 497, save_dir)

gene_name = "Ascl1"
print(np.where(motif_name == "ASCL1")) # 504, 559
plot_motif_umap(adata_r, adata_a, 504, save_dir)

gene_name = "Neurog2"
print(np.where(motif_name == "NEUROG2")) # 126, 569
plot_motif_umap(adata_r, adata_a, 126, save_dir)

gene_name = "Mef2c"
print(np.where(motif_name == "MEF2C")) # 52
plot_motif_umap(adata_r, adata_a, 52, save_dir)

expected_a_mat = adata_a.layers["a_raw"].sum(1).reshape(-1,1) @ adata_a.layers["a_raw"].sum(0).reshape(1,-1) / adata_a.layers["a_raw"].sum()
expected_da_mat = adata_a.layers["dadt"].sum(1).reshape(-1,1) @ adata_a.layers["dadt"].sum(0).reshape(1,-1) / adata_a.layers["dadt"].sum()

def plot_motif_raw_dev_umap(adata_r, adata_a, motif_idx, save_dir):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5.0 * 2, 5.0 * 1))

    var_name = motif_name[motif_idx]
    motif_num = (adata_a.layers["a_raw"] @ motif_bool_mat[:, motif_idx]).toarray().reshape(-1)
    motif_mean = np.array(expected_a_mat @ motif_bool_mat[:, motif_idx]).reshape(-1)
    raw_dev_count = motif_num - motif_mean
    mappable = axes[0].scatter(x = adata_r.obsm["X_umap"][:, 0], y = adata_r.obsm["X_umap"][:, 1], 
                               s = 1, c = raw_dev_count,
                               cmap="coolwarm", vmin=-np.max(np.abs(raw_dev_count)), vmax=np.max(np.abs(raw_dev_count)))
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")
    axes[0].set_title(var_name + " motif count raw deviation")
    cbar = fig.colorbar(mappable, ax=axes[0])

    d_motif_num = (adata_a.layers["dadt"] @ motif_bool_mat[:, motif_idx]).toarray().reshape(-1)
    d_motif_mean = np.array(expected_da_mat @ motif_bool_mat[:, motif_idx]).reshape(-1)
    raw_dev_d_count = d_motif_num - d_motif_mean
    mappable = axes[1].scatter(x = adata_r.obsm["X_umap"][:, 0], y = adata_r.obsm["X_umap"][:, 1], 
                               s = 1, c = raw_dev_d_count, 
                               cmap="coolwarm", vmin=-np.max(np.abs(raw_dev_d_count)), vmax=np.max(np.abs(raw_dev_d_count)))
    axes[1].set_xlabel("UMAP 1")
    axes[1].set_ylabel("UMAP 2")
    axes[1].set_title(var_name + " motif count velocity raw deviation")
    cbar = fig.colorbar(mappable, ax=axes[1])

    fig.tight_layout()
    plt.savefig(save_dir+ "/motif_dev_" + var_name + ".png")
    plt.close()

save_dir = dir_path
gene_name = "Neurod2"
print(np.where(motif_name == gene_name))
plot_motif_raw_dev_umap(adata_r, adata_a, 731, save_dir)
plot_motif_raw_dev_umap(adata_r, adata_a, 781, save_dir)


gene_name = "Rbpj"
print(np.where(motif_name == "Rbpjl"))
plot_motif_raw_dev_umap(adata_r, adata_a, 497, save_dir)

gene_name = "Ascl1"
print(np.where(motif_name == "ASCL1"))
plot_motif_raw_dev_umap(adata_r, adata_a, 504, save_dir)

gene_name = "Neurog2"
print(np.where(motif_name == "NEUROG2"))
plot_motif_raw_dev_umap(adata_r, adata_a, 126, save_dir)

gene_name = "Mef2c"
print(np.where(motif_name == "MEF2C"))
plot_motif_raw_dev_umap(adata_r, adata_a, 52, save_dir)


# get peak sequence
import pysam
import tqdm
import re
file_path = "/home/nomura/Proj/mmvelo/data/10x_multiome_brain_repre_wo_IN/refdata_mm10/fasta/genome.fa"
fasta = pysam.Fastafile(file_path)

adata_a.var['peak_sequence'] = [None] * adata_a.n_vars
for i in tqdm.tqdm(range(adata_a.n_vars)):
    delimiter = "-"
    chrom_start_end = re.split(delimiter, adata_a.var_names[i].replace(":", "-"))
    chrom, start, end = chrom_start_end[0], int(chrom_start_end[1]), int(chrom_start_end[2])
    adata_a.var['peak_sequence'][i] = fasta.fetch(chrom, start, end).upper()
adata_a.var["peak_sequence"]

# compute GC bias
GC_bias = np.zeros(adata_a.n_vars)
for i in tqdm.tqdm(range(adata_a.n_vars)):
    sequence = adata_a.var["peak_sequence"][i]
    freq_a = sequence.count("A")
    freq_t = sequence.count("T")
    freq_g = sequence.count("G")
    freq_c = sequence.count("C")
    GC_bias[i] = (freq_g + freq_c) / (freq_a + freq_t + freq_g + freq_c)
adata_a.var["GC_bias"] = GC_bias

# compute log10 accessibility sum for each peak
adata_a.var["log10_raw_acceccibility"] = np.array(np.log10(adata_a.layers["a_raw"].sum(0))).reshape(-1)

# get kNN to get background peak sets
import scipy
from sklearn.neighbors import NearestNeighbors
import numpy as np
from pynndescent import NNDescent

def get_bg_peak_sets(adata_a, num_iter=30):
    num_iter = num_iter + 1
    GC_acc_peaks = np.array([adata_a.var["log10_raw_acceccibility"], 
                             adata_a.var["GC_bias"]])
    # Mahalanobis transformation
    GC_acc_peaks = GC_acc_peaks - GC_acc_peaks.mean(1).reshape(-1, 1)
    chol_cov = np.linalg.cholesky((np.cov(GC_acc_peaks)))
    norm_GC_acc_peaks = scipy.linalg.solve_triangular(a=chol_cov, b= GC_acc_peaks, lower=True).T # solve ax = b for x
    idx = NNDescent(norm_GC_acc_peaks, metric="euclidean", n_neighbors=num_iter, n_jobs=8)
    knn_idx, _ = idx.query(norm_GC_acc_peaks, num_iter)
    adata_a.varm["bg_peak_idx"] = knn_idx
    return knn_idx

def compute_B(adata_a, num_iter=50):
    print("computing background peaks...")
    knn_idx = get_bg_peak_sets(adata_a, num_iter)
    B = np.zeros([adata_a.n_vars, adata_a.n_vars])
    for i in tqdm.tqdm(range(adata_a.n_vars)):
        bg_idx = knn_idx[i, 1:]
        B[bg_idx, i] = 1 / num_iter
    return B

def compute_E(adata_a):
    E_a = adata_a.layers["a_raw"].sum(1).reshape(-1,1) @ adata_a.layers["a_raw"].sum(0).reshape(1,-1) / adata_a.layers["a_raw"].sum()
    E_dadt = adata_a.layers["dadt"].sum(1).reshape(-1,1) @ adata_a.layers["dadt"].sum(0).reshape(1,-1) / adata_a.layers["dadt"].sum()
    return E_a, E_dadt

def compute_Y(adata_a):
    E_a, E_dadt = compute_E(adata_a)
    
    obs_a = adata_a.layers["a_raw"] @ motif_bool_mat
    exp_a = E_a @ motif_bool_mat
    Y_a = (obs_a - exp_a) / exp_a

    obs_dadt = adata_a.layers["dadt"] @ motif_bool_mat
    exp_dadt = E_dadt @ motif_bool_mat
    #Y_dadt = (obs_dadt - exp_dadt) #/ exp_dadt
    Y_dadt = obs_dadt # dadt is already centered at zero

    return Y_a, Y_dadt

def compute_Y_prime(adata_a, num_iter=50):
    E_a, E_dadt = compute_E(adata_a)
    B = compute_B(adata_a, num_iter=num_iter)

    B_prod_M = B @ motif_bool_mat

    obs_a = adata_a.layers["a_raw"] @ B_prod_M
    exp_a = E_a @ B_prod_M
    Y_a_prime = (obs_a - exp_a) / (E_a @ motif_bool_mat)

    obs_dadt = adata_a.layers["dadt"] @ B_prod_M
    exp_dadt = E_dadt @ B_prod_M
    # Y_dadt_prime = (obs_dadt - exp_dadt) #/ exp_dadt
    Y_dadt_prime = obs_dadt # dadt is already centered at zero

    return Y_a_prime, Y_dadt_prime

def compute_dev_zscore(adata_a, num_iter=30):
    Y_a, Y_dadt = compute_Y(adata_a)
    Y_a_prime, Y_dadt_prime = compute_Y_prime(adata_a, num_iter=num_iter)

    Z_a = (Y_a - Y_a_prime.mean(0)) / np.std(Y_a_prime, 0)
    Z_dadt = (Y_dadt - Y_dadt_prime.mean(0)) # / np.std(Y_dadt_prime, 0)

    return Z_a, Z_dadt

# get deviation z-score
Z_a, Z_dadt = compute_dev_zscore(adata_a, num_iter=30)



def plot_motif_z_dev_umap(adata_r, adata_a, motif_idx, Z_a, Z_dadt, save_dir):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5.0 * 2, 5.0 * 1))
    var_name = motif_name[motif_idx]
    
    z_a = np.array(Z_a[:, motif_idx]).reshape(-1)
    mappable = axes[0].scatter(x = adata_r.obsm["X_umap"][:, 0], y = adata_r.obsm["X_umap"][:, 1], 
                               s = 1, c = z_a, 
                               cmap="coolwarm", vmin=-np.max(np.abs(z_a)), vmax=np.max(np.abs(z_a)))
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")
    axes[0].set_title(var_name + " motif count z-score")
    cbar = fig.colorbar(mappable, ax=axes[0])

    z_dadt = np.array(Z_dadt[:, motif_idx]).reshape(-1)
    mappable = axes[1].scatter(x = adata_r.obsm["X_umap"][:, 0], y = adata_r.obsm["X_umap"][:, 1], 
                               s = 1, c = z_dadt, 
                               cmap="coolwarm", vmin=-np.max(np.abs(z_dadt)), vmax=np.max(np.abs(z_dadt)))
    axes[1].set_xlabel("UMAP 1")
    axes[1].set_ylabel("UMAP 2")
    axes[1].set_title(var_name + " motif count velocity z-score")
    cbar = fig.colorbar(mappable, ax=axes[1])

    fig.tight_layout()
    plt.savefig(save_dir+ "/motif_z_dev_" + var_name + ".png")
    plt.close()



save_dir = dir_path
gene_name = "Neurod2"
print(np.where(motif_name == gene_name))
plot_motif_z_dev_umap(adata_r, adata_a, 731, Z_a, Z_dadt, save_dir)
plot_motif_z_dev_umap(adata_r, adata_a, 781, Z_a, Z_dadt, save_dir)


gene_name = "Rbpj"
print(np.where(motif_name == "Rbpjl"))
plot_motif_z_dev_umap(adata_r, adata_a, 497, Z_a, Z_dadt, save_dir)


gene_name = "Ascl1"
print(np.where(motif_name == "ASCL1"))
plot_motif_z_dev_umap(adata_r, adata_a, 504, Z_a, Z_dadt, save_dir)

gene_name = "Neurog2"
print(np.where(motif_name == "NEUROG2"))
plot_motif_z_dev_umap(adata_r, adata_a, 126, Z_a, Z_dadt, save_dir)

gene_name = "Mef2c"
print(np.where(motif_name == "MEF2C"))
plot_motif_z_dev_umap(adata_r, adata_a, 52, Z_a, Z_dadt, save_dir)

gene_name = "Dlg2"
print(np.where(motif_name == "DLG22"))
plot_motif_z_dev_umap(adata_r, adata_a, 52, Z_a, Z_dadt, save_dir)



#####
def plot_motif_raw_z_d_dev_umap(dm_s, motif_idx, Y_dadt, Z_dadt, save_dir):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5.0 * 2, 5.0 * 1))
    var_name = motif_name[motif_idx]
    
    z_a = Y_dadt[:, motif_idx]
    mappable = axes[0].scatter(x = dm_s.adata_r.obsm["X_umap"][:, 0], y = dm_s.adata_r.obsm["X_umap"][:, 1], 
                               s = 1, c = z_a, 
                               cmap="coolwarm", vmin=-np.max(np.abs(z_a)), vmax=np.max(np.abs(z_a)))
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")
    axes[0].set_title(var_name + " motif velocity raw")
    cbar = fig.colorbar(mappable, ax=axes[0])

    z_dadt = Z_dadt[:, motif_idx]
    mappable = axes[1].scatter(x = dm_s.adata_r.obsm["X_umap"][:, 0], y = dm_s.adata_r.obsm["X_umap"][:, 1], 
                               s = 1, c = z_dadt, 
                               cmap="coolwarm", vmin=-np.max(np.abs(z_dadt)), vmax=np.max(np.abs(z_dadt)))
    axes[1].set_xlabel("UMAP 1")
    axes[1].set_ylabel("UMAP 2")
    axes[1].set_title(var_name + " motif count velocity normalized")
    cbar = fig.colorbar(mappable, ax=axes[1])

    fig.tight_layout()
    plt.savefig(save_dir+ "/motif_dz_dev_" + var_name + ".png")
    plt.close()



save_dir = runPath + "/downstream_analysis"
gene_name = "Neurod2"
print(np.where(motif_name == gene_name))
plot_motif_raw_z_d_dev_umap(dm_s, 731, Y_dadt, Z_dadt, save_dir)
plot_motif_raw_z_d_dev_umap(dm_s, 781, Y_dadt, Z_dadt, save_dir)
plot_su_phase_msmu_vs_recsu(dm_s, save_dir, gene_name)

gene_name = "Pax6"
print(np.where(motif_name == gene_name))
plot_su_phase_msmu_vs_recsu(dm_s, save_dir, gene_name)

gene_name = "Rbpj"
print(np.where(motif_name == "Rbpjl"))
plot_motif_raw_z_d_dev_umap(dm_s, 497, Y_dadt, Z_dadt, save_dir)
plot_su_phase_msmu_vs_recsu(dm_s, save_dir, gene_name)

gene_name = "Ascl1"
print(np.where(motif_name == "ASCL1"))
plot_motif_raw_z_d_dev_umap(dm_s, 504, Y_dadt, Z_dadt, save_dir)
plot_su_phase_msmu_vs_recsu(dm_s, save_dir, gene_name)

gene_name = "Neurog2"
print(np.where(motif_name == "NEUROG2"))
plot_motif_raw_z_d_dev_umap(dm_s, 126, Y_dadt, Z_dadt, save_dir)
plot_su_phase_msmu_vs_recsu(dm_s, save_dir, gene_name)

gene_name = "Mef2c"
print(np.where(motif_name == "MEF2C"))
plot_motif_raw_z_d_dev_umap(dm_s, 52, Y_dadt, Z_dadt, save_dir)
plot_su_phase_msmu_vs_recsu(dm_s, save_dir, gene_name)


def plot_motif_raw_z_dev_umap(dm_s, motif_idx, Y_a, Z_a, save_dir):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(5.0 * 2, 5.0 * 1))
    var_name = motif_name[motif_idx]
    
    z_a = Y_a[:, motif_idx]
    mappable = axes[0].scatter(x = dm_s.adata_r.obsm["X_umap"][:, 0], y = dm_s.adata_r.obsm["X_umap"][:, 1], 
                               s = 1, c = z_a, 
                               cmap="coolwarm", vmin=-np.max(np.abs(z_a)), vmax=np.max(np.abs(z_a)))
    axes[0].set_xlabel("UMAP 1")
    axes[0].set_ylabel("UMAP 2")
    axes[0].set_title(var_name + " motif deviation raw")
    cbar = fig.colorbar(mappable, ax=axes[0])

    z_dadt = Z_a[:, motif_idx]
    mappable = axes[1].scatter(x = dm_s.adata_r.obsm["X_umap"][:, 0], y = dm_s.adata_r.obsm["X_umap"][:, 1], 
                               s = 1, c = z_dadt, 
                               cmap="coolwarm", vmin=-np.max(np.abs(z_dadt)), vmax=np.max(np.abs(z_dadt)))
    axes[1].set_xlabel("UMAP 1")
    axes[1].set_ylabel("UMAP 2")
    axes[1].set_title(var_name + " motif deviation normalized")
    cbar = fig.colorbar(mappable, ax=axes[1])

    fig.tight_layout()
    plt.savefig(save_dir+ "/motif_z_raw_dev_" + var_name + ".png")
    plt.close()



save_dir = runPath + "/downstream_analysis"
gene_name = "Neurod2"
print(np.where(motif_name == gene_name))
plot_motif_raw_z_dev_umap(dm_s, 731, Y_a, Z_a, save_dir)
plot_motif_raw_z_dev_umap(dm_s, 781, Y_a, Z_a, save_dir)
plot_su_phase_msmu_vs_recsu(dm_s, save_dir, gene_name)

gene_name = "Pax6"
print(np.where(motif_name == gene_name))
plot_su_phase_msmu_vs_recsu(dm_s, save_dir, gene_name)

gene_name = "Rbpj"
print(np.where(motif_name == "Rbpjl"))
plot_motif_raw_z_dev_umap(dm_s, 497, Y_a, Z_a, save_dir)
plot_su_phase_msmu_vs_recsu(dm_s, save_dir, gene_name)

gene_name = "Ascl1"
print(np.where(motif_name == "ASCL1"))
plot_motif_raw_z_dev_umap(dm_s, 504, Y_a, Z_a, save_dir)
plot_su_phase_msmu_vs_recsu(dm_s, save_dir, gene_name)

gene_name = "Neurog2"
print(np.where(motif_name == "NEUROG2"))
plot_motif_raw_z_dev_umap(dm_s, 126, Y_a, Z_a, save_dir)
plot_su_phase_msmu_vs_recsu(dm_s, save_dir, gene_name)

gene_name = "Mef2c"
print(np.where(motif_name == "MEF2C"))
plot_motif_raw_z_dev_umap(dm_s, 52, Y_a, Z_a, save_dir)
plot_su_phase_msmu_vs_recsu(dm_s, save_dir, gene_name)




