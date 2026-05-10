import os
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
import scvelo as scv
import scanpy.external as sce

np.random.seed(42)

# load anndata
dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/anndata"
adata_r = sc.read_loom(dir_path + "/adata_rna.loom", obs_names="obs_names", var_names="var_names")

adata_r.obsm["X_umap"]  = pd.read_csv(dir_path + "/umap_coordinate.tsv", sep="\t", header=None).to_numpy()
adata_r.obs["clusters"] = pd.read_json(dir_path + "/cell_clusters.json", typ="series").astype("category")


# plot umap first
def embed_z(adata, n_neighbors=30, min_dist=0.2, densmap=False):
    z_mat = adata.obsm["latent"]
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, densmap=densmap)
    z_embed = reducer.fit_transform(z_mat)
    adata.obsm["X_umap"] = z_embed
    return z_embed

def plot_umap(adata, dir_name, embedding=False, n_neighbors=30, min_dist=0.2, cluster_name="clusters", 
              fig_name=None, legend_loc="right margin", color_map=None):
    if embedding:
        adata.obsm["X_umap"] = embed_z(adata, n_neighbors=n_neighbors, min_dist=min_dist)
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    sc.pl.umap(adata, return_fig=True, color=cluster_name, legend_loc=legend_loc, color_map=color_map)
    if fig_name is None:
        plt.savefig(dir_name+"/umap_" + cluster_name + ".png", bbox_inches='tight')
    else:
        plt.savefig(dir_name+"/"+fig_name, bbox_inches='tight')
    plt.close(fig)

save_dir = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/pseudotime"
os.mkdir(save_dir)
plot_umap(adata_r, save_dir, embedding=False, cluster_name="clusters", fig_name="umap_clusters.png")


# check diffusion component
sc.pp.neighbors(adata_r, n_neighbors=15, use_rep="latent")
sc.tl.diffmap(adata_r, random_state=1, n_comps=10)

def plot_diff_map(adata, dir_name, basis="diffmap", components=[1,2], color=None, size=None, fig_name=None):
    num_plots = len(color)
    fig, ax = plt.subplots(1, num_plots, figsize=(3, 3 * num_plots))
    sc.pl.scatter(adata, basis=basis, components=components,
                  color=color, size=size)
    plt.savefig(dir_name+"/"+fig_name, bbox_inches='tight')
    plt.close(fig)

plot_diff_map(adata_r, dir_path, components=[0,0], color="clusters",
              fig_name = "diff_comp_0_0.png")
plot_diff_map(adata_r, dir_path, components=[1,2], color="clusters",
              fig_name = "diff_comp_1_2.png")
plot_diff_map(adata_r, dir_path, components=[3,4], color="clusters",
              fig_name = "diff_comp_3_4.png")
plot_diff_map(adata_r, dir_path, components=[5,6], color="clusters",
              fig_name = "diff_comp_5_6.png")
plot_diff_map(adata_r, dir_path, components=[7,8], color="clusters",
              fig_name = "diff_comp_7_8.png")
plot_diff_map(adata_r, dir_path, components=[9,10], color="clusters",
              fig_name = "diff_comp_9_10.png")
# -> difficult to determine start cell by diffusion component

# calculate cell cycle score and determine starting cell which has highest G2/M signature in the diverging population
gcc_result = scv.tl.score_genes_cell_cycle(adata_r, copy=True)
plot_umap(gcc_result, dir_path, embedding=False, cluster_name="G2M_score", fig_name="umap_G2M_score.png")

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
scv.pl.scatter(gcc_result, color='G2M_score - S_score', size=80, frameon=True, title="Cell cycle score")
plt.savefig(dir_path+"/umap_cellcycle_score.png", bbox_inches='tight')
plt.close(fig)

# agrmax cellcycle score をする
#start_cells = 
start_cell = adata_r.obs_names[gcc_result.obs["G2M_score"].argmax()]

# pseudotime inference with palantir
sce.tl.palantir(adata_r, n_components=10, knn=15,
                use_adjacency_matrix=True, distances_key="distances")

# check palantir diffusion component
plot_diff_map(adata_r, dir_path, basis="palantir_diff_comp",
              components=[1,2], color="clusters", fig_name = "diff_comp_palantir_1_2.png")
plot_diff_map(adata_r, dir_path, basis="palantir_diff_comp",
              components=[3,4], color="clusters", fig_name = "diff_comp_palantir_3_4.png")
plot_diff_map(adata_r, dir_path, basis="palantir_diff_comp",
              components=[5,6], color="clusters", fig_name = "diff_comp_palantir_5_6.png")
plot_diff_map(adata_r, dir_path, basis="palantir_diff_comp",
              components=[7,8], color="clusters", fig_name = "diff_comp_palantir_7_8.png")
plot_diff_map(adata_r, dir_path, basis="palantir_diff_comp",
              components=[9,10], color="clusters", fig_name = "diff_comp_palantir_9_10.png")
# palantir diff comp 1 semms to capture the direction of differentiation
end_cells = [adata_r.obs_names[adata_r.obsm["X_palantir_diff_comp"][:, 1].argmin()],
             adata_r.obs_names[adata_r.obsm["X_palantir_diff_comp"][:, 1].argmax()]]



pr_res = sce.tl.palantir_results(adata_r,
                                 early_cell=start_cell,
                                 terminal_states=end_cells,
                                 ms_data = "X_umap", 
                                 num_waypoints=500)

adata_r.obs["pseudotime"] = pr_res.pseudotime
plot_umap(adata_r, dir_path, embedding=False, cluster_name="pseudotime", fig_name="umap_pseudotime.png", color_map="viridis")
# IN cells must be removed...

start_end_cells = [start_cell, end_cells[0], end_cells[1]]
fig, ax = plt.subplots(1, 1, figsize=(3, 3))
plt.scatter(adata_r.obsm["X_umap"][:, 0], adata_r.obsm["X_umap"][:, 1], s=1)
for cell in start_end_cells:
    idx = np.where(adata_r.obs_names == cell)[0][0]
    plt.scatter(adata_r.obsm["X_umap"][idx, 0], adata_r.obsm["X_umap"][idx, 1], s=10, color="red")    
plt.savefig(dir_path+"/umap_start_end.png", bbox_inches='tight')
plt.close(fig)


# remove IN cells 

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
plt.scatter(adata_r.obsm["X_umap"][:, 0], adata_r.obsm["X_umap"][:, 1], s=0.1)
plt.savefig(dir_path+"/umap_coord_macro.png", bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
plt.scatter(adata_r.obsm["X_umap"][:, 0], adata_r.obsm["X_umap"][:, 1], s=0.1,)
plt.xlim(left=5, right=10)
plt.ylim(bottom=5, top=7)
plt.savefig(dir_path+"/umap_coord_micro.png", bbox_inches='tight')
plt.close(fig)

# 6 < x < 8, 5.5 < y < 6.5
remove_cells = (adata_r.obsm["X_umap"][:, 0] > 6) & (adata_r.obsm["X_umap"][:, 0] < 8) \
                & (adata_r.obsm["X_umap"][:, 1] > 5.5) & (adata_r.obsm["X_umap"][:, 1] < 6.5)
adata_r.obs_names[~remove_cells]

adata_r_wo_IN = adata_r[~remove_cells, :]

pr_res = sce.tl.palantir_results(adata_r_wo_IN,
                                 early_cell=start_cell,
                                 terminal_states=end_cells,
                                 ms_data = "X_umap", 
                                 num_waypoints=500)

adata_r_wo_IN.obs["pseudotime"] = pr_res.pseudotime
plot_umap(adata_r_wo_IN, dir_path, embedding=False, cluster_name="pseudotime", fig_name="umap_pseudotime_wo_IN.png", color_map="viridis")

# plot diff comp in woIN data
## Note components' indices start from 1 not zero!!
plot_diff_map(adata_r_wo_IN, dir_path, basis="palantir_diff_comp",
              components=[1,2], color="clusters", fig_name = "diff_comp_palantir_woIN_1_2.png")
plot_diff_map(adata_r_wo_IN, dir_path, basis="palantir_diff_comp",
              components=[3,4], color="clusters", fig_name = "diff_comp_palantir_woIN_3_4.png")
plot_diff_map(adata_r_wo_IN, dir_path, basis="palantir_diff_comp",
              components=[5,6], color="clusters", fig_name = "diff_comp_palantir_woIN_5_6.png")
plot_diff_map(adata_r_wo_IN, dir_path, basis="palantir_diff_comp",
              components=[7,8], color="clusters", fig_name = "diff_comp_palantir_woIN_7_8.png")
plot_diff_map(adata_r_wo_IN, dir_path, basis="palantir_diff_comp",
              components=[9,10], color="clusters", fig_name = "diff_comp_palantir_woIN_9_10.png")


for i in range(10):
    adata_r_wo_IN.obs["diff_comp"] = adata_r_wo_IN.obsm["X_palantir_diff_comp"][:, i]
    plot_umap(adata_r_wo_IN, dir_path, embedding=False, cluster_name="diff_comp", fig_name="umap_diff_comp_{}_wo_IN.png".format(i), color_map="viridis")

# with IN
adata_r.obs["diff_comp"] = adata_r.obsm["X_palantir_diff_comp"][:, 0]
plot_umap(adata_r, dir_path, embedding=False, cluster_name="diff_comp", fig_name="umap_diff_comp_0.png", color_map="viridis")

adata_r.obs["diff_comp"] = adata_r.obsm["X_palantir_diff_comp"][:, 1]
plot_umap(adata_r, dir_path, embedding=False, cluster_name="diff_comp", fig_name="umap_diff_comp_1.png", color_map="viridis")

# Decided to use diff comp 1 as a variation axis...
adata_r.obs["diff_comp"] = adata_r.obsm["X_palantir_diff_comp"][:, 1]
start_cell = adata_r.obs_names[gcc_result.obs["G2M_score"].argmax()]
adata_r[start_cell, :].obs["diff_comp"].item()
adata_r.obs["pseudotime"] = (adata_r.obs["diff_comp"] - adata_r[start_cell, :].obs["diff_comp"].item()) * -1
adata_r.obs["pseudotime"] = adata_r.obs["pseudotime"] / adata_r.obs["pseudotime"].max()
plot_umap(adata_r, dir_path, embedding=False, cluster_name="pseudotime", fig_name="umap_diff_comp_1_pseudotime.png", color_map="viridis")
plot_umap(adata_r, dir_path, embedding=False, cluster_name="pseudotime", fig_name="umap_pseudotime_for_analysis.png", color_map="viridis")

dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/anndata"
file_name = "/pseudotime.tsv"
pd.DataFrame(adata_r.obs["pseudotime"]).to_csv(dir_path+file_name, sep="\t", header=False, index=False)

# check
pd.read_csv(dir_path+file_name, sep="\t", header=None)[0]
adata_r.obs["pseudotime"]



"""
#####
# use PCA
adata_copy = adata_r.copy()
sc.tl.pca(adata_copy, n_comps=100)

sce.tl.palantir(adata_copy, n_components=10, knn=30,)

# check palantir diffusion component
plot_diff_map(adata_copy, dir_path, basis="palantir_diff_comp",
              components=[0,1], color="clusters", fig_name = "diff_comp_palantir_pca_0_1.png")
plot_diff_map(adata_copy, dir_path, basis="palantir_diff_comp",
              components=[2,3], color="clusters", fig_name = "diff_comp_palantir_pca_2_3.png")
plot_diff_map(adata_copy, dir_path, basis="palantir_diff_comp",
              components=[4,5], color="clusters", fig_name = "diff_comp_palantir_pca_4_5.png")
# palantir diff comp 2 semms to capture the direction of differentiation
end_cells = [adata_copy.obs_names[adata_copy.obsm["X_palantir_diff_comp"][:, 2].argmin()],
             adata_copy.obs_names[adata_copy.obsm["X_palantir_diff_comp"][:, 2].argmax()]]


pr_res = sce.tl.palantir_results(adata_copy,
                                 early_cell=start_cell,
                                 terminal_states=end_cells,
                                 ms_data = "X_palantir_multiscale", 
                                 num_waypoints=500)

adata_copy.obs["pseudotime"] = pr_res.pseudotime
plot_umap(adata_copy, dir_path, embedding=False, cluster_name="pseudotime", fig_name="umap_pca_pseudotime.png")
# this doesnt seem to work
plot_diff_map(adata_copy, dir_path, basis="palantir_diff_comp",
              components=[0,1], color="pseudotime", fig_name = "diff_comp_palantir_pca_0_1_pseudotime.png")
plot_diff_map(adata_copy, dir_path, basis="palantir_diff_comp",
              components=[2,3], color="pseudotime", fig_name = "diff_comp_palantir_pca_2_3_pseudotime.png")
plot_diff_map(adata_copy, dir_path, basis="palantir_diff_comp",
              components=[4,5], color="pseudotime", fig_name = "diff_comp_palantir_pca_4_5_pseudotime.png")
"""

