import os
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
import scvelo as scv
import scanpy.external as sce
from scipy.io import mmwrite, mmread

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
dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/expr_umap"
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

def plot_umap(adata, dir_name, fig_name, 
              cluster_name="clusters", legend_loc="right margin", color_map=None):
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    scv.pl.scatter(adata, color=cluster_name, legend_loc=legend_loc, color_map=color_map)
    plt.savefig(dir_name+"/"+fig_name, bbox_inches='tight')
    plt.close(fig)

fig_name = "umap_pseudotime"
plot_umap(adata_r, dir_path, fig_name, cluster_name="pseudotime")

fig_name = "umap_cluster"
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
scv.pl.scatter(adata_r, color="clusters", legend_loc="right_margin")
plt.savefig(dir_path+"/"+fig_name, bbox_inches='tight')
plt.close(fig)

var_name = "Gria2"
fig_name = "umap_{}".format(var_name)
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
scv.pl.scatter(adata_r, color=var_name, layer="s_raw", legend_loc="right_margin")
plt.savefig(dir_path+"/"+fig_name, bbox_inches='tight')
plt.close(fig)

fig_name = "umap_{}_velocity".format(var_name)
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
scv.pl.scatter(adata_r, color=var_name, layer="dsdt", legend_loc="right_margin", color_map="coolwarm")
plt.savefig(dir_path+"/"+fig_name, bbox_inches='tight')
plt.close(fig)

# this could take a while
adata_r.layers["atac_raw"] = adata_a.layers["a_raw"] @ peak_gene_linkage
adata_r.layers["datac_dt"] = adata_a.layers["dadt"] @ peak_gene_linkage

def plot_expr_umap(adata, dir_path, gene, spliced=True, unspliced=True, atac=True):
    if spliced:
        s = adata[:,gene].layers["s_raw"].toarray().reshape(-1)
        dsdt = adata[:,gene].layers["dsdt"].toarray().reshape(-1)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(3.0 * 2, 3.0), dpi=150)
        cbar0 = axes[0].scatter(x = adata.obsm["X_umap"][:,0], y = adata.obsm["X_umap"][:,1], s = 1,
                        c = s, vmin = min(s), vmax = np.percentile(s, 99), cmap="YlGn")
        axes[0].set_xlabel("UMAP1")
        axes[0].set_ylabel("UMAP2")
        axes[0].set_title("{} reconstructed spliced".format(gene))
        fig.colorbar(cbar0, ax=axes[0])
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        

        cbar1 = axes[1].scatter(x = adata.obsm["X_umap"][:,0], y = adata.obsm["X_umap"][:,1], s = 1,
                        c = dsdt, vmin = - np.percentile(abs(dsdt), 99), vmax = np.percentile(abs(dsdt), 99),
                        cmap="coolwarm")
        axes[1].set_xlabel("UMAP1")
        axes[1].set_ylabel("UMAP2")
        axes[1].set_title("{} inferred dsdt".format(gene))
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        fig.colorbar(cbar1, ax=axes[1])
        fig.tight_layout()
        plt.savefig(dir_path + "/s_expr_vel_{}.png".format(gene), bbox_inches='tight')
        plt.close()

    if unspliced:
        u = adata[:,gene].layers["u_raw"].toarray().reshape(-1)
        dudt = adata[:,gene].layers["dudt"].toarray().reshape(-1)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(3.0 * 2, 3.0), dpi=150)
        cbar0 = axes[0].scatter(x = adata.obsm["X_umap"][:,0], y = adata.obsm["X_umap"][:,1], s = 1,
                        c = u, vmin = min(u), vmax = np.percentile(u, 99), cmap="YlGn")
        axes[0].set_xlabel("UMAP1")
        axes[0].set_ylabel("UMAP2")
        axes[0].set_title("{} reconstructed unspliced".format(gene))
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        fig.colorbar(cbar0, ax=axes[0])

        cbar1 = axes[1].scatter(x = adata.obsm["X_umap"][:,0], y = adata.obsm["X_umap"][:,1], s = 1,
                        c = dudt, vmin = - np.percentile(abs(dudt), 99), vmax = np.percentile(abs(dudt), 99),
                        cmap="coolwarm")
        axes[1].set_xlabel("UMAP1")
        axes[1].set_ylabel("UMAP2")
        axes[1].set_title("{} inferred dudt".format(gene))
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        fig.colorbar(cbar1, ax=axes[1])
        fig.tight_layout()
        plt.savefig(dir_path + "/u_expr_vel_{}.png".format(gene), bbox_inches='tight')
        plt.close()

    if atac:
        idx = np.where(adata.var_names== gene)[0].item()
        num_peaks = peak_gene_linkage[:, idx].sum()
        a = adata[:,gene].layers["atac_raw"].toarray().reshape(-1)
        dadt = adata[:,gene].layers["datac_dt"].toarray().reshape(-1)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(3.0 * 2, 3.0), dpi=150)
        cbar0 = axes[0].scatter(x = adata.obsm["X_umap"][:,0], y = adata.obsm["X_umap"][:,1], s = 1,
                        c = a, vmin = min(a), vmax = np.percentile(a, 99), cmap="YlGn")
        axes[0].set_xlabel("UMAP1")
        axes[0].set_ylabel("UMAP2")
        axes[0].set_title("{} reconstructed atac, {} peaks".format(gene, num_peaks))
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        fig.colorbar(cbar0, ax=axes[0])

        cbar1 = axes[1].scatter(x = adata.obsm["X_umap"][:,0], y = adata.obsm["X_umap"][:,1], s = 1,
                        c = dadt, vmin = - np.percentile(abs(dadt), 99), vmax = np.percentile(abs(dadt), 99),
                        cmap="coolwarm")
        axes[1].set_xlabel("UMAP1")
        axes[1].set_ylabel("UMAP2")
        axes[1].set_title("{} inferred dadt".format(gene))
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        fig.colorbar(cbar1, ax=axes[1])
        fig.tight_layout()
        plt.savefig(dir_path + "/a_expr_vel_{}.png".format(gene), bbox_inches='tight')
        plt.close()

def plot_corr_peak(adata_r, adata_a, dir_path, gene):
    gene_idx = np.where(adata_r.var_names== gene)[0].item()
    num_peaks = peak_gene_linkage[:, gene_idx].sum()
    if num_peaks == 0:
        return None
    else:
        corr_peak_idx = np.where(peak_gene_linkage[:, gene_idx] > 0)[0]
        for idx in corr_peak_idx:
            a = adata_a[:,idx].layers["a_raw"].toarray().reshape(-1)
            dadt = adata_a[:,idx].layers["dadt"].toarray().reshape(-1)
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(3.0 * 2, 3.0), dpi=150)
            cbar0 = axes[0].scatter(x = adata_r.obsm["X_umap"][:,0], y = adata_r.obsm["X_umap"][:,1], s = 1,
                            c = a, vmin = min(a), vmax = np.percentile(a, 99), cmap="YlGn")
            axes[0].set_xlabel("UMAP1")
            axes[0].set_ylabel("UMAP2")
            axes[0].set_title("{}, {}".format(gene, adata_a.var_names[idx]))
            axes[0].set_xticks([])
            axes[0].set_yticks([])
            fig.colorbar(cbar0, ax=axes[0])

            cbar1 = axes[1].scatter(x = adata_r.obsm["X_umap"][:,0], y = adata_r.obsm["X_umap"][:,1], s = 1,
                            c = dadt, vmin = - np.percentile(abs(dadt), 99), vmax = np.percentile(abs(dadt), 99),
                            cmap="coolwarm")
            axes[1].set_xlabel("UMAP1")
            axes[1].set_ylabel("UMAP2")
            axes[1].set_title("{} inferred dadt".format(adata_a.var_names[idx]))
            axes[1].set_xticks([])
            axes[1].set_yticks([])
            fig.colorbar(cbar1, ax=axes[1])
            fig.tight_layout()
            plt.savefig(dir_path + "/{}_corr_peak_{}.png".format(gene, adata_a.var_names[idx]), bbox_inches='tight')
            plt.close()





plot_expr_umap(adata_r, dir_path, "Gria2")
plot_expr_umap(adata_r, dir_path, "Mef2c")
plot_expr_umap(adata_r, dir_path, "Robo2")
plot_expr_umap(adata_r, dir_path, "Grin2b")
plot_expr_umap(adata_r, dir_path, "Nfix")
plot_expr_umap(adata_r, dir_path, "Epha5")
plot_expr_umap(adata_r, dir_path, "Pax6")
plot_expr_umap(adata_r, dir_path, "Gad1")
plot_expr_umap(adata_r, dir_path, "Rbpj")
plot_expr_umap(adata_r, dir_path, "Neurod2")
plot_corr_peak(adata_r, adata_a, dir_path, "Neurod2")

# total variation spliced
fig, axes = plt.subplots(1, 2, figsize=(5*2, 5))
s = np.array(np.abs(adata_r.layers["s_raw"]).sum(1)).reshape(-1)
dsdt = np.array(np.abs(adata_r.layers["dsdt"]).sum(1)).reshape(-1)
cbar0 = axes[0].scatter(x = adata_r.obsm["X_umap"][:,0], y = adata_r.obsm["X_umap"][:,1], s = 1,
                c = s, vmin = min(s), vmax = np.percentile(abs(s), 99),
                cmap="coolwarm")
axes[0].set_xlabel("UMAP1")
axes[0].set_ylabel("UMAP2")
axes[0].set_title("spliced total count")
axes[0].set_xticks([])
axes[0].set_yticks([])
fig.colorbar(cbar0, ax=axes[0])
cbar1 = axes[1].scatter(x = adata_r.obsm["X_umap"][:,0], y = adata_r.obsm["X_umap"][:,1], s = 1,
                c = dsdt, vmin = min(abs(dsdt)), vmax = np.percentile(abs(dsdt), 99),
                cmap="coolwarm")
axes[1].set_xlabel("UMAP1")
axes[1].set_ylabel("UMAP2")
axes[1].set_title("spliced velocity total variation")
axes[1].set_xticks([])
axes[1].set_yticks([])
fig.colorbar(cbar1, ax=axes[1])
fig.tight_layout()
plt.savefig(dir_path + "/total_variation_s.png", bbox_inches='tight')
plt.close()


# total variation unspliced
fig, axes = plt.subplots(1, 2, figsize=(5*2, 5))
s = np.array(np.abs(adata_r.layers["u_raw"]).sum(1)).reshape(-1)
dsdt = np.array(np.abs(adata_r.layers["dudt"]).sum(1)).reshape(-1)
cbar0 = axes[0].scatter(x = adata_r.obsm["X_umap"][:,0], y = adata_r.obsm["X_umap"][:,1], s = 1,
                c = s, vmin = min(s), vmax = np.percentile(abs(s), 99),
                cmap="coolwarm")
axes[0].set_xlabel("UMAP1")
axes[0].set_ylabel("UMAP2")
axes[0].set_title("unspliced total count")
axes[0].set_xticks([])
axes[0].set_yticks([])
fig.colorbar(cbar0, ax=axes[0])
cbar1 = axes[1].scatter(x = adata_r.obsm["X_umap"][:,0], y = adata_r.obsm["X_umap"][:,1], s = 1,
                c = dsdt, vmin = min(abs(dsdt)), vmax = np.percentile(abs(dsdt), 99),
                cmap="coolwarm")
axes[1].set_xlabel("UMAP1")
axes[1].set_ylabel("UMAP2")
axes[1].set_title("unspliced velocity total variation")
axes[1].set_xticks([])
axes[1].set_yticks([])
fig.colorbar(cbar1, ax=axes[1])
fig.tight_layout()
plt.savefig(dir_path + "/total_variation_u.png", bbox_inches='tight')
plt.close()

# total variation atac
fig, axes = plt.subplots(1, 2, figsize=(5*2, 5))
s = np.array(np.abs(adata_a.layers["a_raw"]).sum(1)).reshape(-1)
dsdt = np.array(np.abs(adata_a.layers["dadt"]).sum(1)).reshape(-1)
cbar0 = axes[0].scatter(x = adata_r.obsm["X_umap"][:,0], y = adata_r.obsm["X_umap"][:,1], s = 1,
                c = s, vmin = min(s), vmax = np.percentile(abs(s), 99),
                cmap="coolwarm")
axes[0].set_xlabel("UMAP1")
axes[0].set_ylabel("UMAP2")
axes[0].set_title("atac total count")
axes[0].set_xticks([])
axes[0].set_yticks([])
fig.colorbar(cbar0, ax=axes[0])
cbar1 = axes[1].scatter(x = adata_r.obsm["X_umap"][:,0], y = adata_r.obsm["X_umap"][:,1], s = 1,
                c = dsdt, vmin = min(abs(dsdt)), vmax = np.percentile(abs(dsdt), 99),
                cmap="coolwarm")
axes[1].set_xlabel("UMAP1")
axes[1].set_ylabel("UMAP2")
axes[1].set_title("atac velocity total variation")
axes[1].set_xticks([])
axes[1].set_yticks([])
fig.colorbar(cbar1, ax=axes[1])
fig.tight_layout()
plt.savefig(dir_path + "/total_variation_a.png", bbox_inches='tight')
plt.close()