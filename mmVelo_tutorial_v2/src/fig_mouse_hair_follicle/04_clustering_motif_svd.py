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
dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/GRN_inference"
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

dadt = adata_a.layers["dadt"].toarray()
U, S, Vh = np.linalg.svd(dadt, full_matrices=False)
print([(lambda x: x.shape)(mat) for mat in [U, S, Vh]])

# squared covariance fraction; SCF
scf = S**2 / (S**2).sum()
scf_max = 100
x = np.array([i for i in range(scf_max)])
y = scf[:scf_max]

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, y)
fig.tight_layout()
plt.savefig(dir_path + "/SVD_squared_covariance_fraction.png", bbox_inches="tight")
plt.close()

# use top 20 components

x = adata_r.obsm["X_umap"][:, 0]
y = adata_r.obsm["X_umap"][:, 1]

fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(7,14))
for i in range(10):
    n, m = 0, 0
    if i >=5:
        n, m = -5, 1
        
    c = U[:, i]
        
    #axes[i+n, m].scatter(x, y, c=c, s=1)
    cbar = axes[i+n, m].scatter(x, y, c=c, s=1, 
                         vmin=-np.abs(c).max(), vmax=np.abs(c).max(), cmap="coolwarm")
    axes[i+n, m].set_title(f"SVD dim {i}, {scf[i]}")
    axes[i+n, m].set_xlabel("UMAP 1")
    axes[i+n, m].set_ylabel("UMAP 2")
    axes[i+n, m].set_xticks([]), axes[i+n, m].set_yticks([])
    
fig.tight_layout()
plt.savefig(dir_path + "/SVD_on_UMAP.png", bbox_inches="tight")
plt.close()


for i in range(10):
    svd_dim = i
    svd_dim_i_ranked_idx = np.argsort(-np.abs(Vh[svd_dim, :]))
    svd_dim_i_cumsum = np.zeros_like(Vh[svd_dim, :])
    for j, idx in enumerate(svd_dim_i_ranked_idx):
        if j == 0:
            svd_dim_i_cumsum[j] += np.abs(Vh[svd_dim, idx])
        else:
            svd_dim_i_cumsum[j] += np.abs(Vh[svd_dim, idx]) + svd_dim_i_cumsum[j-1]
    ranked_order = np.arange(25000)
    print(svd_dim_i_cumsum.max())

    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(ranked_order, svd_dim_i_cumsum)
    ax.set_xlabel("ranked peaks")
    ax.set_ylabel("cumulative weight")
    ax.set_title(f"SVD weight cumsum module {i}")
    fig.tight_layout()
    plt.savefig(dir_path + f"/SVD_cumsum_module_{svd_dim}.png", bbox_inches="tight")
    plt.close()