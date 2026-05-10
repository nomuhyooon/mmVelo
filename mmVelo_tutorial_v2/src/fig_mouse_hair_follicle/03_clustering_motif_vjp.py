import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
from scipy.io import mmwrite, mmread
import seaborn as sns
import scipy
import torch
from torch.func import vmap
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from argparse import ArgumentParser

sys.path.append("/home/nomura/Proj/mmvelo/src")
from mmvelo_multi.dataset import MultiomeBrainDataModule_Pre, ShareSeqDataModule_Pre, DynDataModule_Smooth, ShareSeqFilteredDataModule_Pre, SHARESeqHFDataModule_Pre
from mmvelo_multi.utils import fit_beta_gamma, plot_umap, plot_genewise_corr, plot_peakwise_corr, plot_size_factor, plot_vec_embed, fit_beta_gamma, get_filter_idx_raw, plot_su_expr_umap, plot_su_phase_raw, plot_genewise_vel_cossim, plot_su_phase_dsdt_var, plot_fluctuation_umap, fit_beta_gamma_scale, compute_cossim_genewise_contribution, plot_vec_embed_tanh, plot_genewise_vel_cossim_scale, plot_filtered_vec_embed
from mmvelo_multi.models import DREG_PRE, DREG_DYN, EarlyStoppingWithWarmup

np.random.seed(42)


# load anndata
dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/anndata"
adata_r = sc.read_loom(dir_path + "/adata_rna.loom", obs_names="obs_names", var_names="var_names")
adata_a = sc.read_loom(dir_path + "/adata_atac.loom", obs_names="obs_names", var_names="var_names")
pseudotime = pd.read_csv(dir_path+"/pseudotime.tsv", sep="\t", header=None)[0].to_numpy()
adata_r.obs["pseudotime"] = pseudotime
refined_clusters = pd.read_csv(dir_path+"/refined_clusters.tsv", sep="\t", header=None)[0].to_numpy()
adata_r.obs["ref_clusters"] = refined_clusters

def plot_umap(adata, dir_name, n_neighbors=30, min_dist=0.2, cluster_name="clusters", 
              fig_name="umap.png", legend_loc="right margin", color_map=None):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
    sc.pl.umap(adata, return_fig=True, color=cluster_name, legend_loc=legend_loc, color_map=color_map)
    plt.savefig(dir_name+"/"+fig_name, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
save_dir = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/TAC_subclustering"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
plot_umap(adata_r, save_dir, cluster_name="ref_clusters", fig_name="umap_refined_clusters.png")

# load motif 
dir_path = "/home/nomura/Proj/mmvelo/data/share_seq_hf_/motif_score"
motif_bool_mat = mmread(dir_path + "/motif_bool.mtx") # 25000 1205
motif_ids = pd.read_csv(dir_path+"/motif_ids.tsv", sep="\t", header=None)[0].to_numpy()
motif_names = pd.read_csv(dir_path+"/motif_names.tsv", sep="\t", header=None)[0].to_numpy()
adata_a.var_names

# make dir
dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/motif_clustering_vjp_refined_clusters_for_fig"
if not os.path.exists(dir_path):
    os.mkdir(dir_path)
    
# load trained models
parser = ArgumentParser(description="mmVelo hyperparameters")
parser.add_argument('--experiment', type=str, default='SHARE-seq_hf')
#parser.add_argument('--experiment', type=str, default='multiome_brain_rep_wo_IN')
#parser.add_argument('--experiment', type=str, default='multiome_brain_rep')
#parser.add_argument('--experiment_names', type=str, default='dsdt_param_reg_filtered')
parser.add_argument('--n_genes', type=int, default=3000)
parser.add_argument('--n_peaks', type=int, default=20000)
parser.add_argument('--min_counts_genes', type=int, default=10)
parser.add_argument('--min_counts_peaks', type=int, default=10)
parser.add_argument('--r_h1dim', type=int, default=128)
parser.add_argument('--r_h2dim', type=int, default= 64)
parser.add_argument('--a_h1dim', type=int, default=128)
parser.add_argument('--a_h2dim', type=int, default=64)
parser.add_argument('--zdim', type=int, default=10)
parser.add_argument('--d_h_dim', type=int, default=64)
parser.add_argument('--z_learnable', type=bool, default=True)
parser.add_argument('--d_learnable', type=bool, default=True)
parser.add_argument('--d_coeff', type=float, default=1e-2)
parser.add_argument('--num_epochs', type=int, default=1000) #
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_moment', type=float, default=1e-2)
parser.add_argument('--lr_dyn', type=float, default=1e-4)
parser.add_argument('--su_corr', type=float, default=-1)
parser.add_argument('--su_ratio', type=int, default=50)
parser.add_argument('--min_counts_su', type=int, default=20)
parser.add_argument('--llik_scaling', type=bool, default=True)
parser.add_argument('--patience', type=int, default=30)
parser.add_argument('--n_neighbors', type=int, default=50)
parser.add_argument('--warmup', type=int, default=30)
parser.add_argument('--warmup_dyn', type=int, default=10)
parser.add_argument('--seed', type=int, default=43)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--comment', type=str, default="Multiome E18 brain. dyn inference by only RNA, using DetWarmup, ZINB dist. momoent. vMF loss, using tanh")
args = parser.parse_args()

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
np.random.seed(args.seed)

runPath = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis"
model = DREG_DYN.load_from_checkpoint(runPath+"/checkpoint_pre_sec.ckpt",
                    rna_dim=adata_r.n_vars, atac_dim=adata_a.n_vars, r_h1_dim=args.r_h1dim, r_h2_dim=args.r_h2dim, 
                    a_h1_dim=args.a_h1dim, a_h2_dim=args.a_h2dim, z_dim=args.zdim, d_h_dim=args.d_h_dim,
                    l_prior_r=None, l_prior_a=None, lr=args.lr_dyn,
                    z_learnable=args.z_learnable, d_coeff=args.d_coeff,
                    filter_idx=None, strict=False
                    )
norm_mat_a = torch.tensor(np.loadtxt("/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/anndata/norm_mat_a.txt"))
latent = torch.tensor(adata_r.obsm["latent"])
d_latent = torch.tensor(adata_r.obsm["dynamics"])
M = torch.tensor(motif_bool_mat.toarray())

def calc_motif_score(z):
    A = model.vaes[1].dec_a(z)[0] * norm_mat_a
    E = (A.sum(1).reshape(-1, 1) @ A.sum(0).reshape(1, -1)) / A.sum()
    motif_score = (A @ M - E @ M) / (E @ M)
    return motif_score
    
motif_score = calc_motif_score(latent)
dec_jvp = lambda vz, vd: torch.func.jvp(calc_motif_score, (vz,), (vd,))
#d_motif_score = torch.func.vmap(dec_jvp, in_dims=(0,0))(latent, d_latent)
res = dec_jvp(latent, d_latent)
motif_score, d_motif_score = res[0].detach().numpy(), res[1].detach().numpy()

dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/motif_clustering_vjp_refined_clusters_for_fig"
#dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/motif_clustering_vjp_refined_clusters"
#dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/motif_clustering_vjp"

adata_motif = ad.AnnData(X=motif_score)
#adata_motif.obs["clusters"] = adata_r.obs["clusters"].to_numpy()
adata_motif.obs["clusters"] = adata_r.obs["ref_clusters"].to_numpy()
adata_motif.obs["pseudotime"] = adata_r.obs["pseudotime"].to_numpy()
adata_motif.obs_names = adata_r.obs_names
adata_motif.var_names = motif_names
sc.pp.neighbors(adata_motif, n_neighbors=10, metric="correlation", n_pcs=None, use_rep="X")
sc.tl.umap(adata_motif)
fig, ax = plt.subplots(figsize=(7, 5))
sc.pl.umap(adata_motif, color=["pseudotime", "clusters", ])
fig.tight_layout()
plt.savefig(dir_path + "/umap_motif_cell.png", bbox_inches='tight', dpi=300)
plt.close()


adata_d_motif = ad.AnnData(X=d_motif_score)
#adata_d_motif.obs["clusters"] = adata_r.obs["clusters"].to_numpy()
adata_d_motif.obs["clusters"] = adata_r.obs["ref_clusters"].to_numpy()
adata_d_motif.obs["pseudotime"] = adata_r.obs["pseudotime"].to_numpy()
adata_d_motif.obs_names = adata_r.obs_names
adata_d_motif.var_names = motif_names
sc.pp.neighbors(adata_d_motif, n_neighbors=10, metric="cosine", n_pcs=None, use_rep="X")
sc.tl.umap(adata_d_motif)
fig, ax = plt.subplots(figsize=(7, 5))
sc.pl.umap(adata_d_motif, color=["pseudotime", "clusters", ])
fig.tight_layout()
plt.savefig(dir_path + "/umap_d_motif_cell.png", bbox_inches='tight', dpi=300)
plt.close()

# limit to expressed TFs
genes = list(adata_r.var_names)
genes_upper = [gene.upper() for gene in genes]
expressed_motif_ids = []
expressed_motifs = []
expressed_tfs = []
motif_names, motif_ids
for motif_name, motif_id in zip(motif_names, motif_ids):
    if "::" in motif_name:
        motif_components = motif_name.split("::")
        for component in motif_components:
            if (component in genes) or (component.upper() in genes_upper):
                expressed_motif_ids.append(motif_id)
                expressed_motifs.append(motif_name)
                expressed_tfs.append(component.lower().capitalize())
    else:
        if (motif_name in genes) or (motif_name.upper() in genes_upper):
            expressed_motif_ids.append(motif_id)
            expressed_motifs.append(motif_name)
            expressed_tfs.append(motif_name.lower().capitalize())
expressed_motif_ids = list(set(expressed_motif_ids))
expressed_motifs = list(set(expressed_motifs))
expressed_tfs = list(set(expressed_tfs))
print(len(expressed_motif_ids), len(expressed_motifs), len(expressed_tfs)) # 296, 194, 138

len(expressed_motifs) # 194
len(set([motif.upper() for motif in expressed_motifs])) # 166
len(expressed_tfs) # 138


# umap by expressed TF motifs    
adata_motif = ad.AnnData(X=motif_score)
adata_motif.obs["clusters"] = adata_r.obs["clusters"].to_numpy()
adata_motif.obs["clusters"] = adata_r.obs["ref_clusters"].to_numpy()
adata_motif.obs["pseudotime"] = adata_r.obs["pseudotime"].to_numpy()
adata_motif.obs_names = adata_r.obs_names
adata_motif.var_names = motif_ids
adata_motif = adata_motif[:, expressed_motif_ids]
sc.pp.neighbors(adata_motif, n_neighbors=10, metric="correlation", n_pcs=None, use_rep="X")
sc.tl.umap(adata_motif)
fig, ax = plt.subplots(figsize=(7, 5))
sc.pl.umap(adata_motif, color=["pseudotime", "clusters", ])
fig.tight_layout()
plt.savefig(dir_path + "/umap_motif_cell.png", bbox_inches='tight', dpi=300)
plt.close()

adata_d_motif = ad.AnnData(X=d_motif_score)
#adata_d_motif.obs["clusters"] = adata_r.obs["clusters"].to_numpy()
adata_d_motif.obs["clusters"] = adata_r.obs["ref_clusters"].to_numpy()
adata_d_motif.obs["pseudotime"] = adata_r.obs["pseudotime"].to_numpy()
adata_d_motif.obs_names = adata_r.obs_names
adata_d_motif.var_names = motif_ids
adata_d_motif = adata_d_motif[:, expressed_motif_ids]
sc.pp.neighbors(adata_d_motif, n_neighbors=10, metric="cosine", n_pcs=None, use_rep="X")
sc.tl.umap(adata_d_motif)
fig, ax = plt.subplots(figsize=(7, 5))
sc.pl.umap(adata_d_motif, color=["pseudotime", "clusters", ])
fig.tight_layout()
plt.savefig(dir_path + "/umap_d_motif_cell.png", bbox_inches='tight', dpi=300)
plt.close()

# for motif clustering, motif x cluster
adata_motif_pxc = ad.AnnData(X=d_motif_score.T)
adata_motif_pxc.layers["motif_score"] = motif_score.T
adata_motif_pxc.obs_names = motif_ids
adata_motif_pxc.obs["motif_name"] = motif_names
adata_motif_pxc.var_names = adata_r.obs_names
adata_motif_pxc = adata_motif_pxc[expressed_motif_ids, :]

sc.pp.neighbors(adata_motif_pxc, n_neighbors=10, metric="cosine", n_pcs=None, use_rep="X")
sc.tl.umap(adata_motif_pxc)
sc.tl.leiden(adata_motif_pxc)
fig, ax = plt.subplots(figsize=(7, 5))
sc.pl.umap(adata_motif_pxc, color=["leiden", ])
fig.tight_layout()
plt.savefig(dir_path + "/umap_motif_motif_velocity.png", bbox_inches='tight', dpi=300)
plt.close()

save_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/motif_clustering_vjp_refined_clusters_for_fig/dev_diff_motifs"
#save_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/motif_clustering_vjp_refined_clusters/dev_diff_motifs"
if not os.path.exists(save_path):
    os.mkdir(save_path)

adata_r.layers["s_raw"].toarray()
    

for idx in range(adata_motif_pxc.n_obs):
    motif_id = adata_motif_pxc.obs_names[idx]
    motif_name = adata_motif_pxc.obs["motif_name"][idx]
    X = adata_motif_pxc[idx, :].layers["motif_score"].reshape(-1)
    dX = adata_motif_pxc[idx, :].X.reshape(-1)
    
    if motif_name in genes_upper:
        gene_idx = genes_upper.index(motif_name)
        gene_name = adata_r.var_names[gene_idx]
        gene_expr = adata_r[:, gene_name].layers["s_raw"].toarray().reshape(-1)
    else:
        continue
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(4.0 * 3, 3.0), dpi=300)
    
    cbar0 = axes[0].scatter(x = adata_r.obsm["X_umap"][:,0], y = adata_r.obsm["X_umap"][:,1], s = 1,
                    #c = X, vmin = -1, vmax = 1, cmap="viridis")
                    c = X, cmap="viridis")
    axes[0].set_xlabel("UMAP1")
    axes[0].set_ylabel("UMAP2")
    axes[0].set_title("{} motif activity".format(motif_name))
    axes[0].axis("off")
    fig.colorbar(cbar0, ax=axes[0])

    cbar1 = axes[1].scatter(x = adata_r.obsm["X_umap"][:,0], y = adata_r.obsm["X_umap"][:,1], s = 1,
                    c = dX, 
                    vmin = -np.max(np.abs(dX)), vmax = np.max(np.abs(dX)) ,cmap="coolwarm")
                    #cmap="coolwarm")
    axes[1].set_xlabel("UMAP1")
    axes[1].set_ylabel("UMAP2")
    axes[1].set_title("{} motif velocity".format(motif_name))
    axes[1].axis("off")
    fig.colorbar(cbar1, ax=axes[1])
    
    cbar2 = axes[2].scatter(x = adata_r.obsm["X_umap"][:,0], y = adata_r.obsm["X_umap"][:,1], s = 1,
                    c = gene_expr, 
                    vmin = np.min(gene_expr), vmax = np.max(gene_expr) ,cmap="viridis")
                    #cmap="coolwarm")
    axes[2].set_xlabel("UMAP1")
    axes[2].set_ylabel("UMAP2")
    axes[2].set_title("{} spliced expression".format(gene_name))
    axes[2].axis("off")
    fig.colorbar(cbar2, ax=axes[2])
    
    fig.tight_layout()
    plt.savefig(save_path + "/{}_motif_activity_velocity.png".format(motif_id + "_" +motif_name),
                bbox_inches='tight', dpi=300)
    plt.close("all")


from sklearn import mixture, model_selection
X = adata_motif_pxc.X.copy()
# std normalize
X = X / np.std(X, axis=1).reshape(-1,1)

# model selection
"""
# baysian GMM: not used
elbo_list = []
num_comp_list = []
for num in range(50):
    num_components = num + 1
    print(f"number of components: {num_components}")
    bgmm = mixture.BayesianGaussianMixture(
    n_components=num_components, covariance_type="full", verbose=1, 
    random_state=42)
    bgmm.fit(X)
    clusters = bgmm.predict(X)
    print("# clusters:", len(set(clusters)))
    print("ELBO: ", bgmm.lower_bound_)
    elbo_list.append(bgmm.lower_bound_)
    num_comp_list.append(num_components)


# choose the best num_components
num_components = 15
bgmm = mixture.BayesianGaussianMixture(
    n_components=num_components, covariance_type="full", verbose=1, 
    random_state=42, weight_concentration_prior=0.01)
bgmm.fit(X)
clusters = bgmm.predict(X)
print("# clusters:", len(set(clusters)))
print("ELBO: ", bgmm.lower_bound_)

adata_motif_pxc.obs["bgmm_clusters"] = [str(cluster) for cluster in clusters]
dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/motif_clustering_vjp"
fig, ax = plt.subplots(figsize=(7, 5))
sc.pl.umap(adata_motif_pxc, color=["bgmm_clusters", ])
fig.tight_layout()
plt.savefig(dir_path + "/umap_vi_gmm_clusters_n_comp_{}.png".format(num_components), bbox_inches='tight', dpi=300)
plt.close()
"""

# try EM GMM and model selection
"""
def gmm_bic_score(estimator, X):
    return -estimator.bic(X)

param_gird = {
    "n_components" : range(1, 31),
    "covariance_type" : ["spherical", "tied", "diag", "full"], 
}
grid_search = model_selection.GridSearchCV(
    mixture.GaussianMixture(verbose=1), param_grid=param_gird, scoring=gmm_bic_score
)
grid_search.fit(X)

df = pd.DataFrame(grid_search.cv_results_)[
    ["param_n_components", "param_covariance_type", "mean_test_score"]
]
df["mean_test_score"] = -df["mean_test_score"]
df = df.rename(
    columns={
        "param_n_components": "Number of components",
        "param_covariance_type": "Type of covariance",
        "mean_test_score": "BIC score",
    }
)
df["log BIC score"] = np.log(df["BIC score"])
df.sort_values(by="log BIC score").head()

fig, ax = plt.subplots(figsize=(15, 5))
sns.catplot(
    data=df,
    kind="bar",
    x="Number of components",
    y="log BIC score",
    hue="Type of covariance",
)
fig.tight_layout()
plt.savefig(dir_path + "/em_gmm_grid_search.png", bbox_inches='tight')
plt.close()
"""

# optimal num components: 7
num_components = 7
bgmm = mixture.GaussianMixture(
    n_components=num_components, covariance_type="spherical", verbose=1, 
    random_state=42)
bgmm.fit(X)
clusters = bgmm.predict(X)
print("# clusters:", len(set(clusters)))
print("ELBO: ", bgmm.lower_bound_)

adata_motif_pxc.obs["bgmm_clusters"] = [str(cluster) for cluster in clusters]
dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/motif_clustering_vjp_refined_clusters"
fig, ax = plt.subplots(figsize=(7, 5))
sc.pl.umap(adata_motif_pxc, color=["bgmm_clusters", ])
fig.tight_layout()
plt.savefig(dir_path + "/umap_em_gmm_clusters_n_comp_{}.png".format(num_components), bbox_inches='tight', dpi=300)
plt.close()



# plot heatmap
## cluster-wise pseudotime
## maybe this should be done with TAC-subclustering i.e. IRS-TAC, HS-CC TAC
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
adata_motif_pxc_sorted = adata_motif_pxc[:, sorted_cell_barcode]

adata_r[adata_r.obs["ref_clusters"] == "Hair Shaft-Cuticle/Cortex", :].obs["pseudotime"].max()
adata_r[adata_r.obs["ref_clusters"] == "Medulla", :].obs["pseudotime"].max()
adata_r[adata_r.obs["ref_clusters"] == "Inner Root Sheath", :].obs["pseudotime"].max()

var_name = adata_motif_pxc_sorted.obs_names.to_numpy()

motif_raw = adata_motif_pxc_sorted.layers["motif_score"].toarray()
motif_raw = scipy.stats.zscore(motif_raw, axis=1)
motif_dt = adata_motif_pxc_sorted.X.toarray()
motif_dt = motif_dt / np.std(motif_dt, axis=1).reshape(-1, 1)

# heatmap
adata_bin = ad.AnnData(X=motif_dt)
adata_bin.layers["rec_x"] = motif_raw
adata_bin.obs_names = var_name
adata_bin.obs["clusters"] = adata_motif_pxc.obs["bgmm_clusters"]

def generate_cluster_colors(num_clusters):
    cmap = plt.get_cmap('tab20')
    cluster_colors = [cmap(i) for i in np.linspace(0, 1, num_clusters)]
    return cluster_colors

clusters = adata_bin.obs["clusters"]
num_clusters = len(adata_bin.obs["clusters"].cat.categories)
cluster_colors = generate_cluster_colors(num_clusters)

sorted_dadt = motif_dt[np.argsort(clusters), :]
sorted_var_name = var_name[np.argsort(clusters)]
sorted_x_raw = motif_raw[np.argsort(clusters), :]
clusters = clusters[np.argsort(clusters)]
row_colors = [cluster_colors[int(i)] for i in clusters]

# set columns colors
col_annots = [list(adata_r.obs["ref_clusters"]), list(adata_r.obs["pseudotime"])]
col_annots = list(zip(*col_annots))
col_annots = pd.MultiIndex.from_tuples(col_annots, names=["ref_clusters", "pseudotime"])

sc.pl.umap(adata_r, color="ref_clusters")
plt.close("all")
clusters_labels = col_annots.get_level_values("ref_clusters")
print(adata_r.obs["ref_clusters"].cat.categories[[5, 2, 3, 0, 1, 4]])
clusters_pal = [matplotlib.colors.to_rgb(hex) for hex in adata_r.uns["ref_clusters_colors"][[5, 2, 3, 0, 1, 4]]]
clusters_lut = dict(zip(map(str, clusters_labels.unique()), clusters_pal))
clusters_colors = pd.Series(clusters_labels, index=col_annots).map(clusters_lut) 

pdt_labels = col_annots.get_level_values("pseudotime")
pdt_pal = sns.color_palette("viridis", pdt_labels.unique().size)
pdt_lut = dict(zip(pdt_labels.unique().sort_values(), pdt_pal))
pdt_colors = pd.Series(pdt_labels, index=col_annots).map(pdt_lut)
clusters_pdt_colors = pd.concat([clusters_colors, pdt_colors], axis=1)


dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/motif_clustering_vjp_refined_clusters"
sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
sns.clustermap(pd.DataFrame(sorted_dadt, columns=col_annots),
               cmap='coolwarm', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, 
               row_colors=row_colors, col_colors=clusters_pdt_colors,
               vmin=-3, vmax=3)
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_clustering_binned_dx_em_gmm_n_comp_{}.png".format(num_components), bbox_inches='tight')
plt.close("all")

sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
sns.clustermap(pd.DataFrame(sorted_x_raw, columns=col_annots),
               cmap='viridis', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, 
               row_colors=row_colors, col_colors=clusters_pdt_colors,
               vmin=-4, vmax=4)
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_clustering_binned_dx_em_gmm_n_comp_{}_x.png".format(num_components), bbox_inches='tight')
plt.close()


# heatmap, leiden
resolution = 0.4
adata_bin = ad.AnnData(X=motif_dt)
adata_bin.layers["rec_x"] = motif_raw
adata_bin.obs_names = var_name
sc.pp.neighbors(adata_bin, n_neighbors=15, metric="cosine", n_pcs=None, use_rep="X")
sc.tl.leiden(adata_bin, resolution=resolution)
sc.tl.umap(adata_bin)

def generate_cluster_colors(num_clusters):
    cmap = plt.get_cmap('tab20')
    cluster_colors = [cmap(i) for i in np.linspace(0, 1, num_clusters)]
    return cluster_colors

clusters = adata_bin.obs["leiden"]
num_clusters = len(adata_bin.obs["leiden"].cat.categories)
cluster_colors = generate_cluster_colors(num_clusters)

sorted_dadt = motif_dt[np.argsort(clusters), :]
sorted_var_name = var_name[np.argsort(clusters)]
sorted_x_raw = motif_raw[np.argsort(clusters), :]
clusters = clusters[np.argsort(clusters)]
row_colors = [cluster_colors[int(i)] for i in clusters]

# set columns colors
col_annots = [list(adata_r.obs["ref_clusters"]), list(adata_r.obs["pseudotime"])]
col_annots = list(zip(*col_annots))
col_annots = pd.MultiIndex.from_tuples(col_annots, names=["ref_clusters", "pseudotime"])

sc.pl.umap(adata_r, color="ref_clusters")
plt.close("all")
clusters_labels = col_annots.get_level_values("ref_clusters")
print(adata_r.obs["ref_clusters"].cat.categories[[5, 2, 3, 0, 1, 4]])
clusters_pal = [matplotlib.colors.to_rgb(hex) for hex in adata_r.uns["ref_clusters_colors"][[5, 2, 3, 0, 1, 4]]]
clusters_lut = dict(zip(map(str, clusters_labels.unique()), clusters_pal))
clusters_colors = pd.Series(clusters_labels, index=col_annots).map(clusters_lut) 

pdt_labels = col_annots.get_level_values("pseudotime")
pdt_pal = sns.color_palette("viridis", pdt_labels.unique().size)
pdt_lut = dict(zip(pdt_labels.unique().sort_values(), pdt_pal))
pdt_colors = pd.Series(pdt_labels, index=col_annots).map(pdt_lut)
clusters_pdt_colors = pd.concat([clusters_colors, pdt_colors], axis=1)


dir_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/motif_clustering_vjp_refined_clusters"
sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
sns.clustermap(pd.DataFrame(sorted_dadt, columns=col_annots),
               cmap='coolwarm', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, 
               row_colors=row_colors, col_colors=clusters_pdt_colors,
               vmin=-3, vmax=3)
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_clustering_binned_dx_leiden_{}.png".format(resolution), bbox_inches='tight')
plt.close("all")

sns.set(font_scale=0.8)
fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
sns.clustermap(pd.DataFrame(sorted_x_raw, columns=col_annots),
               cmap='viridis', xticklabels=False, yticklabels=False,
               row_cluster=False, col_cluster=False, 
               row_colors=row_colors, col_colors=clusters_pdt_colors,
               vmin=-4, vmax=4)
fig.tight_layout()
plt.savefig(dir_path + "/heatmap_clustering_binned_dx_leiden_{}_x.png".format(resolution), bbox_inches='tight')
plt.close()

# leiden 0.4
df_motif = pd.DataFrame(adata_bin.obs["leiden"])
df_motif["motif_name"] = adata_motif_pxc.obs["motif_name"]
df_motif = df_motif.sort_values("leiden")

save_path = dir_path + "/leiden_motif_cluster"
if not os.path.exists(save_path):
    os.mkdir(save_path)
for clst in df_motif["leiden"].cat.categories:
    df_motif_clst = df_motif[df_motif["leiden"]==clst]
    df_motif_clst["motif_name"].to_csv(save_path+f"/motif_leiden_{clst}.tsv",
                                       sep="\t", index=False, header=False)
    

