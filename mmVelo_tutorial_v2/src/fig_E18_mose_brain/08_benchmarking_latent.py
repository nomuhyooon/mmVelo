import os
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
import scvelo as scv
import multivelo as mv
from scvelo.preprocessing.moments import get_moments

# load anndata
dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/anndata"
adata_r = sc.read_loom(dir_path + "/adata_rna.loom", obs_names="obs_names", var_names="var_names")
adata_a = sc.read_loom(dir_path + "/adata_atac.loom", obs_names="obs_names", var_names="var_names")

adata_r.obsm["X_umap"]  = pd.read_csv(dir_path + "/umap_coordinate.tsv", sep="\t", header=None).to_numpy()
adata_r.obs["clusters"] = pd.read_json(dir_path + "/cell_clusters.json", typ="series").astype("category")
adata_r.obs["pseudotime"] = pd.read_csv(dir_path + "/pseudotime.tsv", sep="\t", header=None)[0].to_numpy()

# make dir
dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/benchmarking"
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

# plotting function
def plot_velocity_map(adata, dir_name, vkey="velocity", color="clusters", method_name="mmVelo"):
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    scv.pl.velocity_embedding_grid(adata, basis="umap", vkey=vkey, color="clusters", title=method_name)
    plt.savefig(dir_name+"/"+ method_name + "_grid.png", bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    scv.pl.velocity_embedding_stream(adata, basis="umap", vkey=vkey, color="clusters",
                                     title=method_name, legend_loc="right margin")
    plt.savefig(dir_name+"/"+ method_name + "_stream.png", bbox_inches='tight')
    plt.close(fig)

# velocyto
n_neighbors = 100 # this may be changed
adata_velocyto = ad.AnnData(X=adata_r.layers["spliced_count"].copy())
adata_velocyto.obs_names, adata_velocyto.var_names = adata_r.obs_names, adata_r.var_names
adata_velocyto.layers["spliced"] = adata_r.layers["spliced_count"].copy()
adata_velocyto.layers["unspliced"] = adata_r.layers["unspliced_count"].copy()
adata_velocyto.obsm["latent"] = adata_r.obsm["latent"]
adata_velocyto.obs["clusters"] = adata_r.obs["clusters"]
adata_velocyto.obsm["X_umap"] = adata_r.obsm["X_umap"]
scv.pp.normalize_per_cell(adata_velocyto, counts_per_cell_after=1e4)
scv.pp.log1p(adata_velocyto)
sc.pp.neighbors(adata_velocyto, n_neighbors=n_neighbors, use_rep="latent")
scv.pp.moments(adata_velocyto, n_neighbors=n_neighbors, use_rep="latent")

scv.tl.velocity(adata_velocyto, mode="deterministic", )
scv.tl.velocity_graph(adata_velocyto, xkey="Ms", n_jobs=16)
scv.tl.velocity_embedding(adata_velocyto, basis="umap", )
plot_velocity_map(adata_velocyto, dir_path, vkey="velocity", color="clusters", method_name="velocyto")

# scVelo
adata_scvelo = ad.AnnData(X=adata_r.layers["spliced_count"].copy())
adata_scvelo.obs_names, adata_scvelo.var_names = adata_r.obs_names, adata_r.var_names
adata_scvelo.layers["spliced"] = adata_r.layers["spliced_count"].copy()
adata_scvelo.layers["unspliced"] = adata_r.layers["unspliced_count"].copy()
adata_scvelo.obsm["latent"] = adata_r.obsm["latent"]
adata_scvelo.obs["clusters"] = adata_r.obs["clusters"]
adata_scvelo.obsm["X_umap"] = adata_r.obsm["X_umap"]
scv.pp.normalize_per_cell(adata_scvelo, counts_per_cell_after=1e4)
scv.pp.log1p(adata_scvelo)
sc.pp.neighbors(adata_scvelo, n_neighbors=n_neighbors, use_rep="latent")
scv.pp.moments(adata_scvelo, n_neighbors=n_neighbors, use_rep="latent")

# 2355 genes
scv.tl.recover_dynamics(adata_scvelo, n_jobs=16)
scv.tl.velocity(adata_scvelo, mode="dynamical", n_jobs=16)
scv.tl.velocity_graph(adata_scvelo, xkey="Ms", n_jobs=16)
scv.tl.velocity_embedding(adata_scvelo, basis="umap", )
plot_velocity_map(adata_scvelo, dir_path, vkey="velocity", color="clusters", method_name="scvelo")

# MultiVelo
## RNA
## Note: MultiVelo demo uses only 1000 highly variable genes. 
## Here we use 3,000 over genes and result may vary
n_neighbors = 100
adata_mv_rna = ad.AnnData(X=adata_r.layers["spliced_count"].copy())
adata_mv_rna.obs_names, adata_mv_rna.var_names = adata_r.obs_names, adata_r.var_names
adata_mv_rna.layers["spliced"] = adata_r.layers["spliced_count"].copy()
adata_mv_rna.layers["unspliced"] = adata_r.layers["unspliced_count"].copy()
adata_mv_rna.obsm["latent"] = adata_r.obsm["latent"]
adata_mv_rna.obs["clusters"] = adata_r.obs["clusters"]
adata_mv_rna.obsm["X_umap"] = adata_r.obsm["X_umap"]
scv.pp.normalize_per_cell(adata_mv_rna, counts_per_cell_after=1e4)
scv.pp.log1p(adata_mv_rna)
scv.pp.filter_genes_dispersion(adata_mv_rna, n_top_genes=1000) # to avoid zero division
sc.pp.neighbors(adata_mv_rna, n_neighbors=n_neighbors, use_rep="latent")
scv.pp.moments(adata_mv_rna, n_neighbors=n_neighbors, use_rep="latent")

dir_path = "/home/nomura/Proj/mmvelo/data/10x_multiome_brain_repre_wo_IN/"
adata_mv_atac = sc.read_10x_mtx(dir_path + '/out/filtered_feature_bc_matrix/', var_names='gene_symbols', cache=True, gex_only=False)
adata_mv_atac = adata_mv_atac[:,adata_mv_atac.var['feature_types'] == "Peaks"]
adata_mv_atac = mv.aggregate_peaks_10x(adata_mv_atac,
                                    dir_path + '/out/e18_mouse_brain_fresh_5k_atac_peak_annotation.tsv',
                                    dir_path + '/analysis/feature_linkage/feature_linkage.bedpe',
                                    verbose=True)
sc.pp.filter_cells(adata_mv_atac, min_counts=2000)
sc.pp.filter_cells(adata_mv_atac, max_counts=60000)
mv.tfidf_norm(adata_mv_atac)

shared_cells = pd.Index(np.intersect1d(adata_mv_rna.obs_names, adata_mv_atac.obs_names))
shared_genes = pd.Index(np.intersect1d(adata_mv_rna.var_names, adata_mv_atac.var_names))
len(shared_cells), len(shared_genes) # 3420, 2918  # when num_genes==1000, 2918 -> 928

adata_mv_rna = adata_mv_rna[shared_cells, shared_genes]
adata_mv_atac = adata_mv_atac[shared_cells, shared_genes]
adata_mv_atac.uns["neighbors"] = adata_mv_rna.uns["neighbors"]
adata_mv_atac.obsp["distances"] = adata_mv_rna.obsp["distances"]
adata_mv_atac.obsp["connectivities"] = adata_mv_rna.obsp["connectivities"]
#adata_mv_atac.layers["Mc"] = get_moments(adata_mv_atac)
mv.knn_smooth_chrom(adata_mv_atac, conn=adata_mv_rna.obsp["connectivities"])

dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/benchmarking"
fig, ax = plt.subplots(1, 1, figsize=(3, 3))
scv.pl.umap(adata_mv_rna, color="clusters")
plt.savefig(dir_path + "/multivelo_umap_clusters.png", bbox_inches='tight')
plt.close(fig)

adata_result = mv.recover_dynamics_chrom(adata_mv_rna,
                                         adata_mv_atac,
                                         gene_list = adata_scvelo.var_names[adata_scvelo.var["velocity_genes"]], # 2027 genes
                                         max_iter=5,
                                         init_mode="invert",
                                         verbose=True,
                                         parallel=True,
                                         save_plot=False,
                                         rna_only=False,
                                         fit=True,
                                         n_anchors=500,
                                         extra_color_key='clusters'
                                        )
# WARNING: You’re trying to run this on 928 dimensions of `.X`, if you really want this, set `use_rep='X'`.
#         Falling back to preprocessing with `sc.pp.pca` and default params.
# 707 genes will be fitted

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
mv.velocity_embedding_stream(adata_result, basis='umap', )
plt.savefig(dir_path + "/mv_stream.png", bbox_inches='tight')
plt.close(fig)


adata_result = mv.recover_dynamics_chrom(adata_mv_rna,
                                         adata_mv_atac,
                                         gene_list = adata_scvelo.var_names[adata_scvelo.var["velocity_genes"]], # 2027 genes
                                         max_iter=5,
                                         init_mode="invert",
                                         verbose=True,
                                         parallel=True,
                                         save_plot=False,
                                         rna_only=False,
                                         fit=True,
                                         n_anchors=500,
                                         extra_color_key='clusters',
                                         use_rep = "latent"
                                        )
# WARNING: You’re trying to run this on 928 dimensions of `.X`, if you really want this, set `use_rep='X'`.
#         Falling back to preprocessing with `sc.pp.pca` and default params.
# 707 genes will be fitted

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
mv.velocity_embedding_stream(adata_result, basis='umap', )
plt.savefig(dir_path + "/mv_stream.png", bbox_inches='tight')
plt.close(fig)



# Deep Layer, mmVelo vs velocyto
adata_r
adata_velocyto

dsdt_mm = adata_r[adata_r.obs["clusters"] == "Deeper layer", :].layers["dsdt"].toarray()
dsdt_ss = adata_velocyto[adata_r.obs["clusters"] == "Deeper layer", :].layers["velocity"]
dsdt_mm = dsdt_mm / np.linalg.norm(dsdt_mm, ord=2, axis=1)[:, np.newaxis]
dsdt_ss = dsdt_ss / np.linalg.norm(dsdt_ss, ord=2, axis=1)[:, np.newaxis]
gene_score = np.array((dsdt_ss * dsdt_mm).sum(0))

np.sum(gene_score >= 0) # 2243
np.sum(gene_score < 0) # 829

print(np.sort(gene_score)[0], np.sort(gene_score)[-1])
fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.hist(gene_score, bins=100)
ax.set_ylim(0, 50)
plt.savefig(dir_path + "/gene_score_mm_vs_ss.png", bbox_inches='tight')
plt.close(fig)

# bottom 5 genes seems to have bad effects on velocity estimation
import seaborn as sns
for i, idx in enumerate(np.argsort(gene_score)[:5]):
    fig, axes = plt.subplots(nrows=1, ncols=7, figsize=(3.0 * 7, 3.0 * 1))
    gene_name = adata_r.var_names[idx]
    print(gene_name, gene_score[idx])
    cell_idx = adata_r.obs["clusters"] == "Deeper layer"
    dsdt_mm = adata_r[:, gene_name].layers["dsdt"].toarray().reshape(-1)
    dsdt_ss = adata_velocyto[:, gene_name].layers["velocity"].toarray().reshape(-1)
    s_raw = adata_r[:, gene_name].layers["s_raw"].toarray().reshape(-1)
    u_raw = adata_r[:, gene_name].layers["u_raw"].toarray().reshape(-1)
    cluster = adata_r.obs["clusters"].to_numpy()
    
    sns.scatterplot(x=adata_r[:, gene_name].layers["s_raw"].toarray().reshape(-1),
                    y=adata_r[:, gene_name].layers["u_raw"].toarray().reshape(-1),
                    hue = cluster,
                    ax = axes[0], legend=False, s = 5,)
    axes[0].set_xlabel("rec s")
    axes[0].set_ylabel("rec u")
    axes[0].set_title(gene_name + " reconstructed")

    mappable = axes[1].scatter(x=adata_r[:, gene_name].layers["s_raw"].toarray().reshape(-1),
                                y=adata_r[:, gene_name].layers["u_raw"].toarray().reshape(-1),
                                c = adata_r[:, gene_name].layers["dsdt"].toarray().reshape(-1), 
                                s = 2,  cmap="coolwarm", 
                                vmin=-np.max(np.abs(dsdt_mm)), vmax=np.max(np.abs(dsdt_mm)))
    axes[1].set_xlabel("rec s")
    axes[1].set_ylabel("rec u")
    axes[1].set_title(gene_name + " mmVelo")
    cbar = fig.colorbar(mappable, ax=axes[1])

    mappable = axes[2].scatter(x=s_raw, y=u_raw, s = 2, c = dsdt_ss, cmap="coolwarm", 
                                vmin=-np.max(np.abs(dsdt_ss)), vmax=np.max(np.abs(dsdt_ss)))
    axes[2].set_xlabel("rec s")
    axes[2].set_ylabel("rec u")
    axes[2].set_title(gene_name + " velocyto")
    cbar = fig.colorbar(mappable, ax=axes[2])

    mappable = axes[3].scatter(x = adata_r.obsm["X_umap"][:, 0], y = adata_r.obsm["X_umap"][:, 1], s = 1, 
                               c = adata_r[:, gene_name].layers["s_raw"].toarray().reshape(-1), )
    axes[3].set_xlabel("UMAP 1")
    axes[3].set_ylabel("UMAP 2")
    axes[3].set_title(gene_name + " rec s")
    cbar = fig.colorbar(mappable, ax=axes[3])

    mappable = axes[4].scatter(x = adata_r.obsm["X_umap"][:, 0], y = adata_r.obsm["X_umap"][:, 1], s = 1,
                               c = adata_r[:, gene_name].layers["u_raw"].toarray().reshape(-1), )
    axes[4].set_xlabel("UMAP 1")
    axes[4].set_ylabel("UMAP 2")
    axes[4].set_title(gene_name + " rec u")
    cbar = fig.colorbar(mappable, ax=axes[4])

    
    axes[5].scatter(x = adata_r.obsm["X_umap"][:, 0], y = adata_r.obsm["X_umap"][:, 1], s = 1, color="grey", alpha=0.2)
    mappable = axes[5].scatter(x = adata_r.obsm["X_umap"][:, 0], y = adata_r.obsm["X_umap"][:, 1], s = 1, c = dsdt_mm, 
                               cmap="coolwarm", vmin=-np.max(np.abs(dsdt_mm)), vmax=np.max(np.abs(dsdt_mm)))
    axes[5].set_xlabel("UMAP 1")
    axes[5].set_ylabel("UMAP 2")
    axes[5].set_title(gene_name+" dsdt mmvelo")
    cbar = fig.colorbar(mappable, ax=axes[5])

    axes[6].scatter(x = adata_r.obsm["X_umap"][:, 0], y = adata_r.obsm["X_umap"][:, 1], s = 1, color="grey", alpha=0.2)
    mappable = axes[6].scatter(x = adata_r.obsm["X_umap"][:, 0], y = adata_r.obsm["X_umap"][:, 1], s = 1, c = dsdt_ss, 
                               cmap="coolwarm", vmin=-np.max(np.abs(dsdt_ss)), vmax=np.max(np.abs(dsdt_ss)))
    axes[6].set_xlabel("UMAP 1")
    axes[6].set_ylabel("UMAP 2")
    axes[6].set_title(gene_name+" dsdt mmvelo")
    cbar = fig.colorbar(mappable, ax=axes[6])

    fig.tight_layout()
    plt.savefig(dir_path+ "/su_plot_dsdt_comparisoon_{}_".format(i) + gene_name + ".png")
    plt.close()

for i, idx in enumerate(np.argsort(gene_score)[-5:]):
    fig, axes = plt.subplots(nrows=1, ncols=7, figsize=(3.0 * 7, 3.0 * 1))
    gene_name = adata_r.var_names[idx]
    print(gene_name, gene_score[idx])
    cell_idx = adata_r.obs["clusters"] == "Deeper layer"
    dsdt_mm = adata_r[:, gene_name].layers["dsdt"].toarray().reshape(-1)
    dsdt_ss = adata_velocyto[:, gene_name].layers["velocity"].toarray().reshape(-1)
    s_raw = adata_r[:, gene_name].layers["s_raw"].toarray().reshape(-1)
    u_raw = adata_r[:, gene_name].layers["u_raw"].toarray().reshape(-1)
    cluster = adata_r.obs["clusters"].to_numpy()
    
    sns.scatterplot(x=adata_r[:, gene_name].layers["s_raw"].toarray().reshape(-1),
                    y=adata_r[:, gene_name].layers["u_raw"].toarray().reshape(-1),
                    hue = cluster,
                    ax = axes[0], legend=False, s = 5,)
    axes[0].set_xlabel("rec s")
    axes[0].set_ylabel("rec u")
    axes[0].set_title(gene_name + " reconstructed")

    mappable = axes[1].scatter(x=adata_r[:, gene_name].layers["s_raw"].toarray().reshape(-1),
                                y=adata_r[:, gene_name].layers["u_raw"].toarray().reshape(-1),
                                c = adata_r[:, gene_name].layers["dsdt"].toarray().reshape(-1), 
                                s = 2,  cmap="coolwarm", 
                                vmin=-np.max(np.abs(dsdt_mm)), vmax=np.max(np.abs(dsdt_mm)))
    axes[1].set_xlabel("rec s")
    axes[1].set_ylabel("rec u")
    axes[1].set_title(gene_name + " mmVelo")
    cbar = fig.colorbar(mappable, ax=axes[1])

    mappable = axes[2].scatter(x=s_raw, y=u_raw, s = 2, c = dsdt_ss, cmap="coolwarm", 
                                vmin=-np.max(np.abs(dsdt_ss)), vmax=np.max(np.abs(dsdt_ss)))
    axes[2].set_xlabel("rec s")
    axes[2].set_ylabel("rec u")
    axes[2].set_title(gene_name + " velocyto")
    cbar = fig.colorbar(mappable, ax=axes[2])

    mappable = axes[3].scatter(x = adata_r.obsm["X_umap"][:, 0], y = adata_r.obsm["X_umap"][:, 1], s = 1, 
                               c = adata_r[:, gene_name].layers["s_raw"].toarray().reshape(-1), )
    axes[3].set_xlabel("UMAP 1")
    axes[3].set_ylabel("UMAP 2")
    axes[3].set_title(gene_name + " rec s")
    cbar = fig.colorbar(mappable, ax=axes[3])

    mappable = axes[4].scatter(x = adata_r.obsm["X_umap"][:, 0], y = adata_r.obsm["X_umap"][:, 1], s = 1,
                               c = adata_r[:, gene_name].layers["u_raw"].toarray().reshape(-1), )
    axes[4].set_xlabel("UMAP 1")
    axes[4].set_ylabel("UMAP 2")
    axes[4].set_title(gene_name + " rec u")
    cbar = fig.colorbar(mappable, ax=axes[4])

    
    axes[5].scatter(x = adata_r.obsm["X_umap"][:, 0], y = adata_r.obsm["X_umap"][:, 1], s = 1, color="grey", alpha=0.2)
    mappable = axes[5].scatter(x = adata_r.obsm["X_umap"][:, 0], y = adata_r.obsm["X_umap"][:, 1], s = 1, c = dsdt_mm, 
                               cmap="coolwarm", vmin=-np.max(np.abs(dsdt_mm)), vmax=np.max(np.abs(dsdt_mm)))
    axes[5].set_xlabel("UMAP 1")
    axes[5].set_ylabel("UMAP 2")
    axes[5].set_title(gene_name+" dsdt mmvelo")
    cbar = fig.colorbar(mappable, ax=axes[5])

    axes[6].scatter(x = adata_r.obsm["X_umap"][:, 0], y = adata_r.obsm["X_umap"][:, 1], s = 1, color="grey", alpha=0.2)
    mappable = axes[6].scatter(x = adata_r.obsm["X_umap"][:, 0], y = adata_r.obsm["X_umap"][:, 1], s = 1, c = dsdt_ss, 
                               cmap="coolwarm", vmin=-np.max(np.abs(dsdt_ss)), vmax=np.max(np.abs(dsdt_ss)))
    axes[6].set_xlabel("UMAP 1")
    axes[6].set_ylabel("UMAP 2")
    axes[6].set_title(gene_name+" dsdt mmvelo")
    cbar = fig.colorbar(mappable, ax=axes[6])

    fig.tight_layout()
    plt.savefig(dir_path+ "/su_plot_dsdt_comparisoon_-{}_".format(i) + gene_name + ".png")
    plt.close()

for i, idx in enumerate(np.argsort(gene_score)[:5]):
    gene_name = adata_r.var_names[idx]
    print(gene_name, gene_score[idx])

for i, idx in enumerate(np.argsort(gene_score)[-5:]):
    gene_name = adata_r.var_names[idx]
    print(gene_name, gene_score[idx])



# Deep Layer, mmVelo vs scVelo
adata_r
adata_scvelo.var["velocity_genes"]
gene_names = adata_scvelo.var_names[adata_scvelo.var["velocity_genes"]]

dsdt_mm = adata_r[:, adata_scvelo.var["velocity_genes"]][adata_r.obs["clusters"] == "Deeper layer", :].layers["dsdt"].toarray()
dsdt_sc = adata_scvelo[:, adata_scvelo.var["velocity_genes"]][adata_r.obs["clusters"] == "Deeper layer", :].layers["velocity"]
dsdt_mm = dsdt_mm / np.linalg.norm(dsdt_mm, ord=2, axis=1)[:, np.newaxis]
dsdt_sc = dsdt_sc / np.linalg.norm(dsdt_sc, ord=2, axis=1)[:, np.newaxis]
gene_score = np.array((dsdt_sc * dsdt_mm).sum(0))

np.sum(gene_score >= 0) # 1200
np.sum(gene_score < 0) # 827

print(np.sort(gene_score)[0], np.sort(gene_score)[-1])
fig, ax = plt.subplots(1, 1, figsize=(3, 3))
ax.hist(gene_score, bins=100)
ax.set_ylim(0, 50)
plt.savefig(dir_path + "/gene_score_mm_vs_scvelo.png", bbox_inches='tight')
plt.close(fig)


# bottom 3 genes seems to have bad effects on velocity estimation
import seaborn as sns
for i, idx in enumerate(np.argsort(gene_score)[:3]):
    fig, axes = plt.subplots(nrows=1, ncols=7, figsize=(3.0 * 7, 3.0 * 1))
    gene_name = gene_names[idx]
    print(gene_name, gene_score[idx])
    cell_idx = adata_r.obs["clusters"] == "Deeper layer"
    dsdt_mm = adata_r[:, gene_name].layers["dsdt"].toarray().reshape(-1)
    dsdt_sc = adata_scvelo[:, gene_name].layers["velocity"].toarray().reshape(-1)
    s_raw = adata_r[:, gene_name].layers["s_raw"].toarray().reshape(-1)
    u_raw = adata_r[:, gene_name].layers["u_raw"].toarray().reshape(-1)
    cluster = adata_r.obs["clusters"].to_numpy()
    
    sns.scatterplot(x=adata_r[:, gene_name].layers["s_raw"].toarray().reshape(-1),
                    y=adata_r[:, gene_name].layers["u_raw"].toarray().reshape(-1),
                    hue = cluster,
                    ax = axes[0], legend=False, s = 5,)
    axes[0].set_xlabel("rec s")
    axes[0].set_ylabel("rec u")
    axes[0].set_title(gene_name + " reconstructed")

    mappable = axes[1].scatter(x=adata_r[:, gene_name].layers["s_raw"].toarray().reshape(-1),
                                y=adata_r[:, gene_name].layers["u_raw"].toarray().reshape(-1),
                                c = adata_r[:, gene_name].layers["dsdt"].toarray().reshape(-1), 
                                s = 2,  cmap="coolwarm", 
                                vmin=-np.max(np.abs(dsdt_mm)), vmax=np.max(np.abs(dsdt_mm)))
    axes[1].set_xlabel("rec s")
    axes[1].set_ylabel("rec u")
    axes[1].set_title(gene_name + " mmVelo")
    cbar = fig.colorbar(mappable, ax=axes[1])

    mappable = axes[2].scatter(x=s_raw, y=u_raw, s = 2, c = dsdt_sc, cmap="coolwarm", 
                                vmin=-np.max(np.abs(dsdt_sc)), vmax=np.max(np.abs(dsdt_sc)))
    axes[2].set_xlabel("rec s")
    axes[2].set_ylabel("rec u")
    axes[2].set_title(gene_name + " scvelo")
    cbar = fig.colorbar(mappable, ax=axes[2])

    mappable = axes[3].scatter(x = adata_r.obsm["X_umap"][:, 0], y = adata_r.obsm["X_umap"][:, 1], s = 1, 
                               c = adata_r[:, gene_name].layers["s_raw"].toarray().reshape(-1), )
    axes[3].set_xlabel("UMAP 1")
    axes[3].set_ylabel("UMAP 2")
    axes[3].set_title(gene_name + " rec s")
    cbar = fig.colorbar(mappable, ax=axes[3])

    mappable = axes[4].scatter(x = adata_r.obsm["X_umap"][:, 0], y = adata_r.obsm["X_umap"][:, 1], s = 1,
                               c = adata_r[:, gene_name].layers["u_raw"].toarray().reshape(-1), )
    axes[4].set_xlabel("UMAP 1")
    axes[4].set_ylabel("UMAP 2")
    axes[4].set_title(gene_name + " rec u")
    cbar = fig.colorbar(mappable, ax=axes[4])

    
    axes[5].scatter(x = adata_r.obsm["X_umap"][:, 0], y = adata_r.obsm["X_umap"][:, 1], s = 1, color="grey", alpha=0.2)
    mappable = axes[5].scatter(x = adata_r.obsm["X_umap"][:, 0], y = adata_r.obsm["X_umap"][:, 1], s = 1, c = dsdt_mm, 
                               cmap="coolwarm", vmin=-np.max(np.abs(dsdt_mm)), vmax=np.max(np.abs(dsdt_mm)))
    axes[5].set_xlabel("UMAP 1")
    axes[5].set_ylabel("UMAP 2")
    axes[5].set_title(gene_name+" dsdt mmvelo")
    cbar = fig.colorbar(mappable, ax=axes[5])

    axes[6].scatter(x = adata_r.obsm["X_umap"][:, 0], y = adata_r.obsm["X_umap"][:, 1], s = 1, color="grey", alpha=0.2)
    mappable = axes[6].scatter(x = adata_r.obsm["X_umap"][:, 0], y = adata_r.obsm["X_umap"][:, 1], s = 1, c = dsdt_sc, 
                               cmap="coolwarm", vmin=-np.max(np.abs(dsdt_sc)), vmax=np.max(np.abs(dsdt_sc)))
    axes[6].set_xlabel("UMAP 1")
    axes[6].set_ylabel("UMAP 2")
    axes[6].set_title(gene_name+" dsdt scVelo")
    cbar = fig.colorbar(mappable, ax=axes[6])

    fig.tight_layout()
    plt.savefig(dir_path+ "/su_plot_dsdt_comparisoon_scvelo_{}_".format(i) + gene_name + ".png")
    plt.close()