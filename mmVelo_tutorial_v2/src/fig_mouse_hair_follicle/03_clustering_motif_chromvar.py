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

np.random.seed(42)


# load anndata
dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/anndata"
adata_r = sc.read_loom(dir_path + "/adata_rna.loom", obs_names="obs_names", var_names="var_names")
adata_a = sc.read_loom(dir_path + "/adata_atac.loom", obs_names="obs_names", var_names="var_names")
pseudotime = pd.read_csv(dir_path+"/pseudotime.tsv", sep="\t", header=None)[0].to_numpy()
adata_r.obs["pseudotime"] = pseudotime

# load motif 
dir_path = "/home/nomura/Proj/mmvelo/data/share_seq_hf_/motif_score"
motif_bool_mat = mmread(dir_path + "/motif_bool.mtx") # 25000 1205
motif_ids = pd.read_csv(dir_path+"/motif_ids.tsv", sep="\t", header=None)[0].to_numpy()
motif_names = pd.read_csv(dir_path+"/motif_names.tsv", sep="\t", header=None)[0].to_numpy()
adata_a.var_names

# make dir
dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/motif_clustering_chromVAR"
if not os.path.exists(dir_path):
    os.mkdir(dir_path)



######
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
# note a_raw is already log-transformed accessibility
adata_a.var["log10_raw_acceccibility"] = adata_a.layers["a_raw"].toarray().sum(0)

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
    E_a = adata_a.layers["a_raw"].toarray().sum(1).reshape(-1,1) @ adata_a.layers["a_raw"].toarray().sum(0).reshape(1,-1) / adata_a.layers["a_raw"].toarray().sum()
    E_dadt = adata_a.layers["dadt"].toarray().sum(1).reshape(-1,1) @ adata_a.layers["dadt"].toarray().sum(0).reshape(1,-1) / adata_a.layers["dadt"].toarray().sum()
    return E_a, E_dadt

def compute_Y(adata_a):
    E_a, E_dadt = compute_E(adata_a)
    
    obs_a = adata_a.layers["a_raw"].toarray() @ motif_bool_mat
    exp_a = E_a @ motif_bool_mat
    Y_a = (obs_a - exp_a) / exp_a

    obs_dadt = adata_a.layers["dadt"] @ motif_bool_mat
    exp_dadt = E_dadt @ motif_bool_mat
    #Y_dadt = obs_dadt - exp_dadt
    Y_dadt = (obs_dadt - exp_dadt) / (E_a @ motif_bool_mat)
    #Y_dadt = obs_dadt # dadt is already centered at zero

    return Y_a, Y_dadt

def compute_Y_prime(adata_a, num_iter=50):
    E_a, E_dadt = compute_E(adata_a)
    B = compute_B(adata_a, num_iter=num_iter)

    B_prod_M = B @ motif_bool_mat

    obs_a = adata_a.layers["a_raw"].toarray() @ B_prod_M
    exp_a = E_a @ B_prod_M
    Y_a_prime = (obs_a - exp_a) / (E_a @ motif_bool_mat)

    obs_dadt = adata_a.layers["dadt"].toarray() @ B_prod_M
    exp_dadt = E_dadt @ B_prod_M
    #Y_dadt_prime = (obs_dadt - exp_dadt)
    Y_dadt_prime = (obs_dadt - exp_dadt) / (E_a @ motif_bool_mat)
    #Y_dadt_prime = obs_dadt # dadt is already centered at zero

    return Y_a_prime, Y_dadt_prime

def compute_dev_zscore(adata_a, num_iter=30):
    Y_a, Y_dadt = compute_Y(adata_a)
    Y_a_prime, Y_dadt_prime = compute_Y_prime(adata_a, num_iter=num_iter)

    Z_a = (Y_a - Y_a_prime.mean(0)) / np.std(Y_a_prime, 0)
    Z_dadt = (Y_dadt - Y_dadt_prime.mean(0)) / np.std(Y_dadt_prime, 0)

    return Z_a, Z_dadt

# get deviation z-score
Z_a, Z_dadt = compute_dev_zscore(adata_a, num_iter=50)


#motif_match_mat = adata_a.layers["a_raw"].toarray() @ motif_bool_mat.toarray()
#motif_match_mat = (motif_match_mat - np.mean(motif_match_mat, axis=0)) / np.std(motif_match_mat, axis=0)

adata_motif = ad.AnnData(X=Z_a)
adata_motif.obs["clusters"] = adata_r.obs["clusters"].to_numpy()
adata_motif.obs["pseudotime"] = adata_r.obs["pseudotime"].to_numpy()
adata_motif.obs_names = adata_r.obs_names
adata_motif.var_names = motif_names
sc.pp.neighbors(adata_motif, n_neighbors=15, metric="correlation", n_pcs=None, use_rep="X")
sc.tl.umap(adata_motif)

dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/motif_clustering_chromVAR"
fig, ax = plt.subplots(figsize=(7, 5))
sc.pl.umap(adata_motif, color=["pseudotime", "clusters", ])
fig.tight_layout()
plt.savefig(dir_path + "/umap_motif_cell.png", bbox_inches='tight', dpi=300)
plt.close()


# motif velocity

adata_d_motif = ad.AnnData(X=Z_dadt)
adata_d_motif.obs["clusters"] = adata_r.obs["clusters"].to_numpy()
adata_d_motif.obs["pseudotime"] = adata_r.obs["pseudotime"].to_numpy()
adata_d_motif.obs_names = adata_r.obs_names
adata_d_motif.var_names = motif_names
sc.pp.neighbors(adata_d_motif, n_neighbors=15, metric="cosine", n_pcs=None, use_rep="X")
sc.tl.umap(adata_d_motif)

dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/motif_clustering_chromVAR"
fig, ax = plt.subplots(figsize=(7, 5))
sc.pl.umap(adata_d_motif, color=["pseudotime", "clusters", ])
fig.tight_layout()
plt.savefig(dir_path + "/umap_d_motif_cell.png", bbox_inches='tight', dpi=300)
plt.close()

# show differential motifs
save_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/motif_clustering_chromVAR/dev_diff_motifs"
if not os.path.exists(save_path):
    os.mkdir(save_path)
    
diff_motifs = np.argsort(-1 * (adata_motif.X > 3).sum(0))[:100]
for i, idx in enumerate(diff_motifs):
    motif_name = adata_motif.var_names[idx]
    X = adata_motif[:, idx].X.reshape(-1)
    dX = adata_d_motif[:, idx].X.reshape(-1)
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(3.0 * 2, 3.0), dpi=100)
    
    cbar0 = axes[0].scatter(x = adata_r.obsm["X_umap"][:,0], y = adata_r.obsm["X_umap"][:,1], s = 1,
                    #c = X, vmin = -1, vmax = 1, cmap="viridis")
                    c = X, cmap="viridis")
    axes[0].set_xlabel("UMAP1")
    axes[0].set_ylabel("UMAP2")
    axes[0].set_title("{} motif activity".format(motif_name))
    fig.colorbar(cbar0, ax=axes[0])

    cbar1 = axes[1].scatter(x = adata_r.obsm["X_umap"][:,0], y = adata_r.obsm["X_umap"][:,1], s = 1,
                    c = dX, 
                    vmin = -np.max(np.abs(dX)), vmax = np.max(np.abs(dX)) ,cmap="coolwarm")
    axes[1].set_xlabel("UMAP1")
    axes[1].set_ylabel("UMAP2")
    axes[1].set_title("{} motif velocity".format(motif_name))
    fig.colorbar(cbar1, ax=axes[1])
    fig.tight_layout()
    plt.savefig(save_path + "/{}_motif_activity_velocity.png".format(motif_name),
                bbox_inches='tight', dpi=100)
    plt.close("all")
    
# plot heatmap ??







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
    E_a = adata_a.layers["a_raw"].toarray().sum(1).reshape(-1,1) @ adata_a.layers["a_raw"].toarray().sum(0).reshape(1,-1) / adata_a.layers["a_raw"].toarray().sum()
    E_dadt = adata_a.layers["dadt"].toarray().sum(1).reshape(-1,1) @ adata_a.layers["dadt"].toarray().sum(0).reshape(1,-1) / adata_a.layers["dadt"].toarray().sum()
    return E_a, E_dadt

def compute_Y(adata_a):
    E_a, E_dadt = compute_E(adata_a)
    
    obs_a = adata_a.layers["a_raw"].toarray() @ motif_bool_mat
    exp_a = E_a @ motif_bool_mat
    Y_a = (obs_a - exp_a) / exp_a

    
    Y_dadt_1 = (adata_a.layers["dadt"].toarray() @ motif_bool_mat - E_dadt @ motif_bool_mat) * (E_a @ motif_bool_mat)
    Y_dadt_2 = (adata_a.layers["a_raw"].toarray() @ motif_bool_mat - E_a @ motif_bool_mat) * (E_dadt @ motif_bool_mat)
    Y_dadt_3 = (E_a @ motif_bool_mat) * (E_a @ motif_bool_mat)
    
    #obs_dadt = adata_a.layers["dadt"] @ motif_bool_mat
    #exp_dadt = E_dadt @ motif_bool_mat
    #Y_dadt = obs_dadt - exp_dadt
    #Y_dadt = (obs_dadt - exp_dadt) / (E_a @ motif_bool_mat)
    #Y_dadt = obs_dadt # dadt is already centered at zero
    Y_dadt = (Y_dadt_1 - Y_dadt_2) / Y_dadt_3

    return Y_a, Y_dadt

def compute_Y_prime(adata_a, num_iter=50):
    E_a, E_dadt = compute_E(adata_a)
    B = compute_B(adata_a, num_iter=num_iter)

    B_prod_M = B @ motif_bool_mat

    obs_a = adata_a.layers["a_raw"].toarray() @ B_prod_M
    exp_a = E_a @ B_prod_M
    Y_a_prime = (obs_a - exp_a) / (E_a @ motif_bool_mat)

    
    Yp_dadt_1 = (adata_a.layers["dadt"].toarray() @ B_prod_M - E_dadt @ B_prod_M) * (E_a @ B_prod_M)
    Yp_dadt_2 = (adata_a.layers["a_raw"].toarray() @ B_prod_M - E_a @ B_prod_M) * (E_dadt @ B_prod_M)
    Yp_dadt_3 = (E_a @ motif_bool_mat) * (E_a @ motif_bool_mat)
    
    #obs_dadt = adata_a.layers["dadt"].toarray() @ B_prod_M
    #exp_dadt = E_dadt @ B_prod_M
    #Y_dadt_prime = (obs_dadt - exp_dadt)
    #Y_dadt_prime = (obs_dadt - exp_dadt) / (E_a @ motif_bool_mat)
    #Y_dadt_prime = obs_dadt # dadt is already centered at zero
    Y_dadt_prime = (Yp_dadt_1 - Yp_dadt_2) / Yp_dadt_3

    return Y_a_prime, Y_dadt_prime

def compute_dev_zscore(adata_a, num_iter=30):
    Y_a, Y_dadt = compute_Y(adata_a)
    Y_a_prime, Y_dadt_prime = compute_Y_prime(adata_a, num_iter=num_iter)

    Z_a = (Y_a - Y_a_prime.mean(0)) / np.std(Y_a_prime, 0)
    Z_dadt = (Y_dadt - Y_dadt_prime.mean(0)) / np.std(Y_dadt_prime, 0)

    return Z_a, Z_dadt

# get deviation z-score
Z_a, Z_dadt = compute_dev_zscore(adata_a, num_iter=50)


#motif_match_mat = adata_a.layers["a_raw"].toarray() @ motif_bool_mat.toarray()
#motif_match_mat = (motif_match_mat - np.mean(motif_match_mat, axis=0)) / np.std(motif_match_mat, axis=0)

adata_motif = ad.AnnData(X=Z_a)
adata_motif.obs["clusters"] = adata_r.obs["clusters"].to_numpy()
adata_motif.obs["pseudotime"] = adata_r.obs["pseudotime"].to_numpy()
adata_motif.obs_names = adata_r.obs_names
adata_motif.var_names = motif_names
sc.pp.neighbors(adata_motif, n_neighbors=15, metric="correlation", n_pcs=None, use_rep="X")
sc.tl.umap(adata_motif)

dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/motif_clustering_chromVAR"
fig, ax = plt.subplots(figsize=(7, 5))
sc.pl.umap(adata_motif, color=["pseudotime", "clusters", ])
fig.tight_layout()
plt.savefig(dir_path + "/umap_motif_cell.png", bbox_inches='tight', dpi=300)
plt.close()


# motif velocity

adata_d_motif = ad.AnnData(X=Z_dadt)
adata_d_motif.obs["clusters"] = adata_r.obs["clusters"].to_numpy()
adata_d_motif.obs["pseudotime"] = adata_r.obs["pseudotime"].to_numpy()
adata_d_motif.obs_names = adata_r.obs_names
adata_d_motif.var_names = motif_names
sc.pp.neighbors(adata_d_motif, n_neighbors=15, metric="cosine", n_pcs=None, use_rep="X")
sc.tl.umap(adata_d_motif)

dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/motif_clustering_chromVAR"
fig, ax = plt.subplots(figsize=(7, 5))
sc.pl.umap(adata_d_motif, color=["pseudotime", "clusters", ])
fig.tight_layout()
plt.savefig(dir_path + "/umap_d_motif_cell.png", bbox_inches='tight', dpi=300)
plt.close()

# show differential motifs
save_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/motif_clustering_chromVAR/dev_diff_motifs"
if not os.path.exists(save_path):
    os.mkdir(save_path)
    
diff_motifs = np.argsort(-1 * (adata_motif.X > 3).sum(0))[:100]
for i, idx in enumerate(diff_motifs):
    motif_name = adata_motif.var_names[idx]
    X = adata_motif[:, idx].X.reshape(-1)
    dX = adata_d_motif[:, idx].X.reshape(-1)
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(3.0 * 2, 3.0), dpi=100)
    
    cbar0 = axes[0].scatter(x = adata_r.obsm["X_umap"][:,0], y = adata_r.obsm["X_umap"][:,1], s = 1,
                    #c = X, vmin = -1, vmax = 1, cmap="viridis")
                    c = X, cmap="viridis")
    axes[0].set_xlabel("UMAP1")
    axes[0].set_ylabel("UMAP2")
    axes[0].set_title("{} motif activity".format(motif_name))
    fig.colorbar(cbar0, ax=axes[0])

    cbar1 = axes[1].scatter(x = adata_r.obsm["X_umap"][:,0], y = adata_r.obsm["X_umap"][:,1], s = 1,
                    c = dX, 
                    vmin = -np.max(np.abs(dX)), vmax = np.max(np.abs(dX)) ,cmap="coolwarm")
    axes[1].set_xlabel("UMAP1")
    axes[1].set_ylabel("UMAP2")
    axes[1].set_title("{} motif velocity".format(motif_name))
    fig.colorbar(cbar1, ax=axes[1])
    fig.tight_layout()
    plt.savefig(save_path + "/{}_motif_activity_velocity.png".format(motif_name),
                bbox_inches='tight', dpi=100)
    plt.close("all")