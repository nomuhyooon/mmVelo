import os
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import matplotlib
import anndata as ad
import scanpy as sc
import scvelo as scv
import scanpy.external as sce
from scipy.io import mmwrite, mmread
import scipy

np.random.seed(42)

# peak-gene linkage matrix
dir_path = "/home/nomura/Proj/mmvelo/data/10x_multiome_brain_repre_wo_IN/analysis/feature_linkage/peak_gene_linkage.mtx"
peak_gene_linkage = mmread(dir_path).toarray()

# peak-promoter linkage matrix
dir_path = "/home/nomura/Proj/mmvelo/data/10x_multiome_brain_repre_wo_IN/analysis/feature_linkage/peak_promoter_linkage.mtx"
peak2prom_mat = mmread(dir_path).toarray()

(peak2prom_mat.sum(0) > 1)


# load anndata
dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/anndata"
adata_r = sc.read_loom(dir_path + "/adata_rna.loom", obs_names="obs_names", var_names="var_names")
adata_a = sc.read_loom(dir_path + "/adata_atac.loom", obs_names="obs_names", var_names="var_names")

adata_r.obsm["X_umap"]  = pd.read_csv(dir_path + "/umap_coordinate.tsv", sep="\t", header=None).to_numpy()
adata_r.obs["clusters"] = pd.read_json(dir_path + "/cell_clusters.json", typ="series").astype("category")
adata_r.obs["pseudotime"] = pd.read_csv(dir_path + "/pseudotime.tsv", sep="\t", header=None)[0].to_numpy()

adata_r.layers["dpdt"] = adata_a.layers["dadt"] @ peak2prom_mat
adata_r.layers["p_raw"] = adata_a.layers["a_raw"] @ peak2prom_mat
adata_r.layers["p_count"] = adata_a.layers["atac_count"] @ peak2prom_mat
prom_gene = adata_r.var_names[peak2prom_mat.sum(0) > 0]
prom_dup_gene = adata_r.var_names[peak2prom_mat.sum(0) > 1]

# make dir
#dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/expr_umap"
dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/expr_umap_for_fig"
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

plt.rcParams['font.family'] = 'sans-serif'

def plot_umap(adata, dir_name, fig_name, 
              cluster_name="clusters", legend_loc="right margin", color_map=None):
    
    fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300)
    scv.pl.scatter(adata, color=cluster_name, legend_loc=legend_loc, color_map=color_map)
    plt.savefig(dir_name+"/"+fig_name, bbox_inches='tight', dpi=300)
    plt.close(fig)

fig_name = "umap_pseudotime"
plot_umap(adata_r, dir_path, fig_name, cluster_name="pseudotime")

fig_name = "umap_cluster"
fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300)
scv.pl.scatter(adata_r, color="clusters", legend_loc="right_margin")
plt.savefig(dir_path+"/"+fig_name, bbox_inches='tight', dpi=300)
plt.close(fig)


# Neurod2 promoter and cis-enhancer
# promoter: "chr11:98329299-98330151"
# cis-enhancer: "chr11:98320243-98320844"

peak = "chr11:98329299-98330151"
a_raw = adata_a[:, peak].layers["a_raw"].toarray().reshape(-1)
dadt = adata_a[:, peak].layers["dadt"].toarray().reshape(-1)

fig_name = f"Neurod2_promoter_{peak}_rec_count"
fig, ax = plt.subplots(1, 1, figsize=(4, 3))
cbar = ax.scatter(x=adata_r.obsm["X_umap"][:, 0], y=adata_r.obsm["X_umap"][:, 1], s=1,
                  c=a_raw, vmin=min(a_raw), vmax=max(a_raw), cmap="viridis")
fig.colorbar(cbar, ax=ax)
ax.axis("off")
fig.tight_layout()
plt.savefig(dir_path+"/" + fig_name + ".png", bbox_inches='tight', dpi=300)
plt.close(fig)

fig_name = f"Neurod2_promoter_{peak}_velocity"
fig, ax = plt.subplots(1, 1, figsize=(4, 3))
cbar = ax.scatter(x=adata_r.obsm["X_umap"][:, 0], y=adata_r.obsm["X_umap"][:, 1], s=1,
                  c=dadt, vmin=-max(abs(dadt)), vmax=max(abs(dadt)), cmap="coolwarm")
fig.colorbar(cbar, ax=ax)
ax.axis("off")
fig.tight_layout()
plt.savefig(dir_path+"/" + fig_name + ".png", bbox_inches='tight', dpi=300)
plt.close(fig)


peak = "chr11:98320243-98320844"
a_raw = adata_a[:, peak].layers["a_raw"].toarray().reshape(-1)
dadt = adata_a[:, peak].layers["dadt"].toarray().reshape(-1)

fig_name = f"Neurod2_cis_enhancer_{peak}_rec_count"
fig, ax = plt.subplots(1, 1, figsize=(4, 3))
cbar = ax.scatter(x=adata_r.obsm["X_umap"][:, 0], y=adata_r.obsm["X_umap"][:, 1], s=1,
                  c=a_raw, vmin=min(a_raw), vmax=max(a_raw), cmap="viridis")
fig.colorbar(cbar, ax=ax)
ax.axis("off")
fig.tight_layout()
plt.savefig(dir_path+"/" + fig_name + ".png", bbox_inches='tight', dpi=300)
plt.close(fig)

fig_name = f"Neurod2_cis_enhancer_{peak}_velocity"
fig, ax = plt.subplots(1, 1, figsize=(4, 3))
cbar = ax.scatter(x=adata_r.obsm["X_umap"][:, 0], y=adata_r.obsm["X_umap"][:, 1], s=1,
                  c=dadt, vmin=-max(abs(dadt)), vmax=max(abs(dadt)), cmap="coolwarm")
fig.colorbar(cbar, ax=ax)
ax.axis("off")
fig.tight_layout()
plt.savefig(dir_path+"/" + fig_name + ".png", bbox_inches='tight', dpi=300)
plt.close(fig)


def plot_x_expr_umap(adata, dir_name, gene_name,
                     spliced=True, unspliced=False, promoter=False, 
                ):
    if spliced:
        fig_name = "{}_s_expr".format(gene_name)
    elif unspliced:
        fig_name = "{}_u_expr".format(gene_name)
    elif promoter:
        fig_name = "{}_a_acc".format(gene_name)
    x_umap = adata.obsm["X_umap"]
    if spliced:
        gene_expr = adata[:, gene_name].layers["s_raw"].toarray().reshape(-1)
    elif unspliced:
        gene_expr = adata[:, gene_name].layers["u_raw"].toarray().reshape(-1)
    elif promoter:
        gene_expr = adata[:, gene_name].layers["p_raw"].toarray().reshape(-1)
    gene_expr = scipy.stats.zscore(gene_expr)

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    cbar = ax.scatter(x=x_umap[:, 0], y=x_umap[:, 1], s=1, c=gene_expr)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    if spliced:
        ax.set_title("{} spliced imputed".format(gene_name))
    elif unspliced:
        ax.set_title("{} unspliced imputed".format(gene_name))
    elif promoter:
        ax.set_title("{} promoter imputed".format(gene_name))
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(cbar, ax=ax)
    plt.savefig(dir_name+"/" + fig_name + ".png", bbox_inches='tight', dpi=300)
    plt.close(fig)

    # blank ver
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.scatter(x=x_umap[:, 0], y=x_umap[:, 1], s=1, c=gene_expr)
    ax.axis("off")
    plt.savefig(dir_name+"/" + fig_name + "_blank.png", bbox_inches='tight', dpi=300)
    plt.close(fig)

def plot_x_vel_umap(adata, dir_name, gene_name,
                    spliced=True, unspliced=False, promoter=False,
                ):
    if spliced:
        fig_name = "{}_s_vel".format(gene_name)
    elif unspliced:
        fig_name = "{}_u_vel".format(gene_name)
    elif promoter:
        fig_name = "{}_p_vel".format(gene_name)

    x_umap = adata.obsm["X_umap"]
    if spliced:
        gene_vel = adata[:, gene_name].layers["dsdt"].toarray().reshape(-1)
    elif unspliced:
        gene_vel = adata[:, gene_name].layers["dudt"].toarray().reshape(-1)
    elif promoter:
        gene_vel = adata[:, gene_name].layers["dpdt"].toarray().reshape(-1)
    gene_vel = gene_vel / np.std(gene_vel)

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    cbar = ax.scatter(x=x_umap[:, 0], y=x_umap[:, 1], s=1, c=gene_vel,
                      cmap="coolwarm", norm=matplotlib.colors.CenteredNorm())
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    if spliced:
        ax.set_title("{} dsdt".format(gene_name))
    elif unspliced:
        ax.set_title("{} dudt".format(gene_name))
    elif promoter:
        ax.set_title("{} dpdt".format(gene_name))
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(cbar, ax=ax)
    plt.savefig(dir_name+"/" + fig_name + ".png", bbox_inches='tight', dpi=300)
    plt.close(fig)

    # blank ver
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    cbar = ax.scatter(x=x_umap[:, 0], y=x_umap[:, 1], s=1, c=gene_vel,
                      cmap="coolwarm", norm=matplotlib.colors.CenteredNorm())
    ax.axis("off")
    plt.savefig(dir_name+"/" + fig_name + "_blank.png", bbox_inches='tight', dpi=300)
    plt.close(fig)

gene_name = "Eomes"
print(gene_name in prom_gene) # False
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False, promoter=False)
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True, promoter=False)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False, promoter=False)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True, promoter=False)

#gene_name = "Tle4"
#plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False)
#plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True)
#plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False)
#plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True)

gene_name = "Satb2"
print(gene_name in prom_gene) # True
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False, promoter=False)
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True, promoter=False)
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=False, promoter=True)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False, promoter=False)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True, promoter=False)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=False, promoter=True)

#gene_name = "Gria2"
#print(gene_name in prom_gene) # False
#plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False)
#plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True)
#plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False)
#plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True)


gene_name = "Neurod2"
print(gene_name in prom_gene) # True
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False, promoter=False)
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True, promoter=False)
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=False, promoter=True)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False, promoter=False)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True, promoter=False)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=False, promoter=True)

gene_name = "Ncor2"
print(gene_name in prom_gene) # True
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False, promoter=False)
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True, promoter=False)
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=False, promoter=True)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False, promoter=False)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True, promoter=False)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=False, promoter=True)

gene_name = "Zic4"
print(gene_name in prom_gene) # True
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False, promoter=False)
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True, promoter=False)
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=False, promoter=True)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False, promoter=False)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True, promoter=False)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=False, promoter=True)

gene_name = "Myt1"
print(gene_name in prom_gene) # True
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False, promoter=False)
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True, promoter=False)
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=False, promoter=True)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False, promoter=False)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True, promoter=False)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=False, promoter=True)

gene_name = "Dlx1"
print(gene_name in prom_gene) # True
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False, promoter=False)
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True, promoter=False)
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=False, promoter=True)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False, promoter=False)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True, promoter=False)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=False, promoter=True)

gene_name = "Hes1"
print(gene_name in prom_gene) # True
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False, promoter=False)
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True, promoter=False)
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=False, promoter=True)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False, promoter=False)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True, promoter=False)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=False, promoter=True)

gene_name = "Myo6"
print(gene_name in prom_gene) # True
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False, promoter=False)
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True, promoter=False)
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=False, promoter=True)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False, promoter=False)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True, promoter=False)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=False, promoter=True)

gene_name = "Notch1"
print(gene_name in prom_gene) # True
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False, promoter=False)
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True, promoter=False)
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=False, promoter=True)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False, promoter=False)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True, promoter=False)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=False, promoter=True)

gene_name = "Zic1"
print(gene_name in prom_gene) # True
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False, promoter=False)
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True, promoter=False)
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=False, promoter=True)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False, promoter=False)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True, promoter=False)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=False, promoter=True)

gene_name = "Aldh1l1"
print(gene_name in prom_gene) # True
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False, promoter=False)
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True, promoter=False)
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=False, promoter=True)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False, promoter=False)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True, promoter=False)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=False, promoter=True)

print(prom_gene[0:10])
print(prom_gene[10:20])

gene_name = "Dlx2"
print(gene_name in prom_gene) # True
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False, promoter=False)
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True, promoter=False)
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=False, promoter=True)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False, promoter=False)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True, promoter=False)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=False, promoter=True)

gene_name = "Klf15"
print(gene_name in prom_gene) # True
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False, promoter=False)
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True, promoter=False)
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=False, promoter=True)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False, promoter=False)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True, promoter=False)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=False, promoter=True)

gene_name = "Nfix"
print(gene_name in prom_gene) # True
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False, promoter=False)
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True, promoter=False)
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=False, promoter=True)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False, promoter=False)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True, promoter=False)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=False, promoter=True)

gene_name = "Notch2"
print(gene_name in prom_gene) # True
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False, promoter=False)
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True, promoter=False)
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=False, promoter=True)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False, promoter=False)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True, promoter=False)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=False, promoter=True)

gene_name = "Nr2f2"
print(gene_name in prom_gene) # True
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False, promoter=False)
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True, promoter=False)
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=False, promoter=True)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False, promoter=False)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True, promoter=False)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=False, promoter=True)

gene_name = "Meis2"
print(gene_name in prom_gene) # True
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False, promoter=False)
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True, promoter=False)
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=False, promoter=True)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False, promoter=False)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True, promoter=False)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=False, promoter=True)

gene_name = "Igfbpl1"
print(gene_name in prom_gene) # True
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False, promoter=False)
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True, promoter=False)
plot_x_expr_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=False, promoter=True)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=True, unspliced=False, promoter=False)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=True, promoter=False)
plot_x_vel_umap(adata_r, dir_path, gene_name, spliced=False, unspliced=False, promoter=True)