import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
from scipy.io import mmwrite, mmread
import seaborn as sns
import scipy

np.random.seed(42)

dir_path = "/home/nomura/Proj/mmvelo/experiments/Greenleaf_multiome_Cond_merged_all_missing/2024-01-19T11:22:03_barch_128/downstream_analysis/result/anndata"
adata_r = sc.read_loom(dir_path + "/adata_rna.loom", obs_names="obs_names", var_names="var_names")
adata_a = sc.read_loom(dir_path + "/adata_atac.loom", obs_names="obs_names", var_names="var_names")

# make dir
dir_path = "/home/nomura/Proj/mmvelo/experiments/Greenleaf_multiome_Cond_merged_all_missing/2024-01-19T11:22:03_barch_128/downstream_analysis/result/annotation"
if not os.path.exists(dir_path):
    os.mkdir(dir_path)
    
set(adata_r.obs["clusters"])
clusters_dict = {'GluN7' : "GluN",
                'GluN4' : "GluN",
                'GluN1' : "GluN",
                'Early RG' : "RG", 
                'GluN5' : "GluN",
                'GluN8' : "GluN", 
                'OPC/Oligo' : 'mGPC/OPC',
                'mGPC' : 'mGPC/OPC',
                'GluN6' : "GluN",
                'RG' : "RG",
                'Cyc. Prog.' : 'Cyc. Prog.',
                'GluN3' : "GluN",
                'Late RG' : "RG",
                #'nIPC/GluN1' : 'nIPC/GluN',
                'nIPC/GluN1' : 'GluN', 
                'tRG' : "RG",
                'GluN2' : "GluN",
                'SP' : 'SP', 
                'mGPC/OPC' : 'mGPC/OPC',
                'nIPC' : 'nIPC/GluN'}
adata_r.obs["ref_clusters"] = pd.Categorical(adata_r.obs["clusters"]).map(clusters_dict).astype("category")

def plot_umap(adata, dir_name, n_neighbors=30, min_dist=0.2, cluster_name="clusters", 
              fig_name="umap.png", legend_loc="right margin", color_map=None):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    sc.pl.umap(adata, return_fig=True, color=cluster_name, legend_loc=legend_loc, color_map=color_map)
    plt.savefig(dir_name+"/"+fig_name, bbox_inches='tight')
    plt.close(fig)
    
save_dir =dir_path
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
plot_umap(adata_r, save_dir, cluster_name="clusters", fig_name="umap_clusters.png")
plot_umap(adata_r, save_dir, cluster_name="ref_clusters", fig_name="umap_clusters_refined.png")

save_path =dir_path + "/cluster_annotation_refined.txt"
adata_r.obs["ref_clusters"].to_csv(save_path, sep="\t")
#adata_r.obs["ref_clusters_2"] = pd.read_csv(save_path, sep='\t', index_col=0)

# exclude outliers
plot_umap(adata_r[adata_r.obsm["X_umap"][:, 0] <= 10, :], save_dir, cluster_name="clusters", fig_name="umap_clusters_remove_outliers.png")
cells_excluded = adata_r[adata_r.obsm["X_umap"][:, 0] > 10, :].obs_names
cells_included = adata_r[adata_r.obsm["X_umap"][:, 0] <= 10, :].obs_names

dir_path = "/home/nomura/Proj/mmvelo/experiments/Greenleaf_multiome_Cond_merged_all_missing/2024-01-19T11:22:03_barch_128/downstream_analysis/result/anndata"
pd.DataFrame(cells_excluded).to_csv(dir_path + "/cells_excluded.txt", sep="\t", columns=["obs_names"], header=False, index=False)
pd.DataFrame(cells_included).to_csv(dir_path + "/cells_included.txt", sep="\t", columns=["obs_names"], header=False, index=False)
#pd.read_csv(dir_path + "/cells_included.txt", sep="\t", header=None)[0]

adata_r_ex = adata_r[cells_included, :]
adata_a_ex = adata_a[cells_included, :]

# confine to ExN lineage
exn_list = list()
for cell_cluster in adata_r_ex.obs["ref_clusters"]:
    exn_list.append(cell_cluster in ["nIPC/GluN", 'GluN'])
plot_umap(adata_r_ex[exn_list, :], save_dir, cluster_name="ref_clusters", fig_name="umap_ExN.png")

adata_r_ex = adata_r_ex[exn_list, :]
adata_a_ex = adata_a_ex[exn_list, :]


# compute diffusionmap
sc.pp.neighbors(adata_r_ex, n_neighbors=15, use_rep="latent")
sc.tl.diffmap(adata_r_ex, random_state=1, n_comps=10)

def plot_diff_map(adata, dir_name, basis="diffmap", components=[1,2], color=None, size=None, fig_name=None):
    num_plots = len(color)
    fig, ax = plt.subplots(1, num_plots, figsize=(3, 3 * num_plots))
    sc.pl.scatter(adata, basis=basis, components=components,
                  color=color, size=size)
    plt.savefig(dir_name+"/"+fig_name, bbox_inches='tight')
    plt.close(fig)

plot_diff_map(adata_r_ex, save_dir, components=[0,1], color=["clusters", "ref_clusters"],
              fig_name = "diff_comp_0_1.png")
plot_diff_map(adata_r_ex, save_dir, components=[2,3], color=["clusters", "ref_clusters"],
              fig_name = "diff_comp_2_3.png")
plot_diff_map(adata_r_ex, save_dir, components=[4,5], color=["clusters", "ref_clusters"],
              fig_name = "diff_comp_4_5.png")
plot_diff_map(adata_r_ex, save_dir, components=[6,7], color=["clusters", "ref_clusters"],
              fig_name = "diff_comp_6_7.png")
plot_diff_map(adata_r_ex, save_dir, components=[8,9], color=["clusters", "ref_clusters"],
              fig_name = "diff_comp_8_9.png")

root_ixs = adata_r_ex.obsm["X_diffmap"][:, 4].argmin()
adata_r_ex.uns["iroot"] = root_ixs
sc.tl.dpt(adata_r_ex)
plot_umap(adata_r_ex, save_dir, cluster_name="dpt_pseudotime", fig_name="umap_dpt_pseudotime.png")

adata_r_ex.obs["dpt_pseudotime"].to_csv(save_dir + "/dpt_pseudotime.tsv", sep="\t", header=None) 
#pd.read_csv(save_dir + "/dpt_pseudotime.tsv", sep="\t", header=None)

# analyse inter-patient variation in velocity
dpt_pseudotime = adata_r_ex.obs["dpt_pseudotime"].to_numpy()
num_q = 10
labels = [f"Q_{i}" for i in range(num_q)]
pd.qcut(adata_r_ex.obs["dpt_pseudotime"], num_q, )
adata_r_ex.obs["dpt_bins"] = pd.qcut(adata_r_ex.obs["dpt_pseudotime"], num_q, labels=labels, )


inter_patient_variation_dict = dict()
for bin in labels:
    adata_r_ex_bin = adata_r_ex[adata_r_ex.obs["dpt_bins"] == bin, :]
    adata_a_ex_bin = adata_a_ex[adata_r_ex_bin.obs_names, :]
    patient_cell_mat = pd.get_dummies(adata_r_ex_bin.obs["Sample.ID"]).to_numpy().T
    num_cells = np.sum(patient_cell_mat, axis=1).reshape(-1, 1)
    
    mean_dsdt = (patient_cell_mat @ adata_r_ex_bin.layers["dsdt"].toarray()) / num_cells
    std_dsdt = np.std(mean_dsdt, axis=0)
    
    mean_dadt = (patient_cell_mat @ adata_a_ex_bin.layers["dadt"].toarray()) / num_cells
    std_dadt = np.std(mean_dadt, axis=0)
    
    inter_patient_variation_dict[bin] = {
        "mean_dsdt" : mean_dsdt,
        "std_dsdt" : std_dsdt,
        "mean_dadt" : mean_dadt,
        "std_dadt" : std_dadt
    }

dsdt_std_list, dadt_std_list = list(), list()
for bin in labels:
    bin_dict = inter_patient_variation_dict[bin]
    dsdt_bin_std = bin_dict["std_dsdt"].sum() / bin_dict["std_dsdt"].shape
    dadt_bin_std = bin_dict["std_dadt"].sum() / bin_dict["std_dadt"].shape
    dsdt_std_list.append(dsdt_bin_std)
    dadt_std_list.append(dadt_bin_std)

save_dir = '/home/nomura/Proj/mmvelo/experiments/Greenleaf_multiome_Cond_merged_all_missing/2024-01-19T11:22:03_barch_128/downstream_analysis/result/inter_patient_variation'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

plot_umap(adata_r_ex, save_dir, cluster_name="dpt_bins", fig_name="umap_dpt_bins.png")
    
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(dsdt_std_list)
plt.savefig(save_dir + "/dsdt_std.png", bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(dadt_std_list)
plt.savefig(save_dir + "/dadt_std.png", bbox_inches='tight')
plt.close(fig)



# violin plot
import seaborn as sns
inter_patient_variation_dict["Q_0"]["std_dsdt"]
inter_patient_variation_dict["Q_0"]["std_dadt"]

df_std_ds_melt = pd.melt(pd.concat([pd.DataFrame(inter_patient_variation_dict[bin]["std_dsdt"], columns=[bin]) for bin in labels],
          axis=1))
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.0, 5.0), dpi=300)
sns.boxplot(x="variable", y="value", data=df_std_ds_melt, sym="",)
sns.stripplot(x="variable", y="value", data=df_std_ds_melt, color="black", s=1)
ax.set_xlabel("pseudotime bin")
ax.set_ylabel("std")
ax.set_ylim(top=0.025, bottom=0)
ax.set_title("dsdt std")
fig.tight_layout()
plt.xticks(rotation=15)
plt.savefig(save_dir + "/violin_dsdt_std.png", bbox_inches='tight', dpi=300)
plt.close("all")


df_std_da_melt = pd.melt(pd.concat([pd.DataFrame(inter_patient_variation_dict[bin]["std_dadt"], columns=[bin]) for bin in labels],
          axis=1))
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.0, 5.0), dpi=300)
sns.boxplot(x="variable", y="value", data=df_std_da_melt, sym="",)
sns.stripplot(x="variable", y="value", data=df_std_da_melt, color="black", s=0.5)
ax.set_xlabel("pseudotime bin")
ax.set_ylabel("std")
ax.set_ylim(top=0.0008, bottom=0)
ax.set_title("dadt std")
fig.tight_layout()
plt.xticks(rotation=15)
plt.savefig(save_dir + "/violin_dadt_std.png", bbox_inches='tight', dpi=300)
plt.close("all")