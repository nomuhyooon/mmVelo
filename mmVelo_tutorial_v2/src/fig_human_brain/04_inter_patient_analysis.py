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

# load cell annotation
dir_path = "/home/nomura/Proj/mmvelo/experiments/Greenleaf_multiome_Cond_merged_all_missing/2024-01-19T11:22:03_barch_128/downstream_analysis/result/annotation"
save_path =dir_path + "/cluster_annotation_refined.txt"
adata_r.obs["ref_clusters"] = pd.read_csv(save_path, sep='\t', index_col=0)

# exclude outliers
#plot_umap(adata_r[adata_r.obsm["X_umap"][:, 0] <= 10, :], save_dir, cluster_name="clusters", fig_name="umap_clusters_remove_outliers.png")
#cells_excluded = adata_r[adata_r.obsm["X_umap"][:, 0] > 10, :].obs_names
#cells_included = adata_r[adata_r.obsm["X_umap"][:, 0] <= 10, :].obs_names

dir_path = "/home/nomura/Proj/mmvelo/experiments/Greenleaf_multiome_Cond_merged_all_missing/2024-01-19T11:22:03_barch_128/downstream_analysis/result/anndata"
cells_included = pd.read_csv(dir_path + "/cells_included.txt", sep="\t", header=None)[0]

adata_r_ex = adata_r[cells_included, :]
adata_a_ex = adata_a[cells_included, :]

# confine to ExN lineage
exn_list = list()
for cell_cluster in adata_r_ex.obs["ref_clusters"]:
    exn_list.append(cell_cluster in ["nIPC/GluN", 'GluN'])
adata_r_ex = adata_r_ex[exn_list, :]
adata_a_ex = adata_a_ex[exn_list, :]


# load diffusionmap
save_dir = '/home/nomura/Proj/mmvelo/experiments/Greenleaf_multiome_Cond_merged_all_missing/2024-01-19T11:22:03_barch_128/downstream_analysis/result/annotation'
adata_r_ex.obs["dpt_pseudotime"] = pd.read_csv(save_dir + "/dpt_pseudotime.tsv", sep="\t", header=None, index_col=0)

# analyse inter-patient variation in velocity
dpt_pseudotime = adata_r_ex.obs["dpt_pseudotime"].to_numpy()
num_q = 10
labels = [f"Q_{i}" for i in range(num_q)]
pd.qcut(adata_r_ex.obs["dpt_pseudotime"], num_q, )
adata_r_ex.obs["dpt_bins"] = pd.qcut(adata_r_ex.obs["dpt_pseudotime"], num_q, labels=labels, )

# var-wise normalization to account for expr scale
adata_r_ex.layers["dsdt"] = adata_r_ex.layers["dsdt"].toarray() / np.std(adata_r_ex.layers["dsdt"].toarray(), axis=0)
adata_r_ex.layers["dudt"] = adata_r_ex.layers["dudt"].toarray() / np.std(adata_r_ex.layers["dudt"].toarray(), axis=0)
adata_a_ex.layers["dadt"] = adata_a_ex.layers["dadt"].toarray() / np.std(adata_a_ex.layers["dadt"].toarray(), axis=0)

inter_patient_variation_dict = dict()
for bin in labels:
    adata_r_ex_bin = adata_r_ex[adata_r_ex.obs["dpt_bins"] == bin, :]
    adata_a_ex_bin = adata_a_ex[adata_r_ex_bin.obs_names, :]
    patient_cell_mat = pd.get_dummies(adata_r_ex_bin.obs["Sample.ID"]).to_numpy().T
    num_cells = np.sum(patient_cell_mat, axis=1).reshape(-1, 1)
    
    dsdt = adata_r_ex_bin.layers["dsdt"]
    #dsdt = dsdt / np.linalg.norm(dsdt, axis=1).reshape(-1,1) # to account for cell-wise variation
    dsdt = dsdt / np.std(dsdt, axis=0) # to account for intra-bin cell state diversity
    dadt = adata_a_ex_bin.layers["dadt"]
    #dadt = dadt / np.linalg.norm(dadt, axis=1).reshape(-1,1) # to account for cell-wise variation
    dadt = dadt / np.std(dadt, axis=0) # to account for intra-bin cell state diversity
    
    mean_dsdt = (patient_cell_mat @ dsdt) / num_cells
    std_dsdt = np.std(mean_dsdt, axis=0)
    
    mean_dadt = (patient_cell_mat @ dadt) / num_cells
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

#plot_umap(adata_r_ex, save_dir, cluster_name="dpt_bins", fig_name="umap_dpt_bins.png")
    
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(dsdt_std_list)
ax.set_xlabel("pseudotime bin")
ax.set_ylabel("std")
ax.set_title("dsdt std mean")
#plt.savefig(save_dir + "/dsdt_std.png", bbox_inches='tight')
plt.savefig(save_dir + "/dsdt_std_normalized_cell_bin.png", bbox_inches='tight')
#plt.savefig(save_dir + "/dsdt_std_normalized_cell_bin_cellvar.png", bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.plot(dadt_std_list)
ax.set_xlabel("pseudotime bin")
ax.set_ylabel("std")
ax.set_title("dadt std mean")
#plt.savefig(save_dir + "/dadt_std.png", bbox_inches='tight')
plt.savefig(save_dir + "/dadt_std_normalized_cell_bin.png", bbox_inches='tight')
#plt.savefig(save_dir + "/dadt_std_normalized_cell_bin_cellvar.png", bbox_inches='tight')
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
#ax.set_ylim(top=0.025, bottom=0)
ax.set_title("dsdt std")
fig.tight_layout()
plt.xticks(rotation=15)
#plt.savefig(save_dir + "/violin_dsdt_std.png", bbox_inches='tight', dpi=300)
plt.savefig(save_dir + "/violin_dsdt_std_normalized_cell_bin.png", bbox_inches='tight', dpi=300)
#plt.savefig(save_dir + "/violin_dsdt_std_normalized_cell_bin_cellvar.png", bbox_inches='tight', dpi=300)
plt.close("all")


df_std_da_melt = pd.melt(pd.concat([pd.DataFrame(inter_patient_variation_dict[bin]["std_dadt"], columns=[bin]) for bin in labels],
          axis=1))
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6.0, 5.0), dpi=300)
sns.boxplot(x="variable", y="value", data=df_std_da_melt, sym="",)
sns.stripplot(x="variable", y="value", data=df_std_da_melt, color="black", s=0.5)
ax.set_xlabel("pseudotime bin")
ax.set_ylabel("std")
#ax.set_ylim(top=0.0008, bottom=0)
ax.set_title("dadt std")
fig.tight_layout()
plt.xticks(rotation=15)
#plt.savefig(save_dir + "/violin_dadt_std.png", bbox_inches='tight', dpi=300)
plt.savefig(save_dir + "/violin_dadt_std_normalized_cell_bin.png", bbox_inches='tight', dpi=300)
#plt.savefig(save_dir + "/violin_dadt_std_normalized_cell_bin_cellvar.png", bbox_inches='tight', dpi=300)
plt.close("all")


# check genes & peaks with top contribution
# genes
bin_dict = inter_patient_variation_dict["Q_5"]
dsdt_q5_std = np.array(bin_dict["std_dsdt"])
dsdt_q5_std_top_genes = adata_r.var_names[dsdt_q5_std >= np.percentile(dsdt_q5_std, 90)] # 143 genes
pd.DataFrame(dsdt_q5_std_top_genes.to_list()).to_csv(save_dir + "/dsdt_q5_top_std_genes_90_percentile.tsv",
                                                     sep="\t", index=False, header=False)
pd.DataFrame(adata_r.var_names[np.argsort(-dsdt_q5_std)].to_list()).to_csv(save_dir + "/dsdt_q5_top_std_genes_ranked_list.tsv",
                                                     sep="\t", index=False, header=False)

bin_dict = inter_patient_variation_dict["Q_4"]
dsdt_q4_std = np.array(bin_dict["std_dsdt"])
dsdt_q4_std_top_genes = adata_r.var_names[dsdt_q4_std >= np.percentile(dsdt_q4_std, 90)] # 143 genes
pd.DataFrame(dsdt_q4_std_top_genes.to_list()).to_csv(save_dir + "/dsdt_q4_top_std_genes_90_percentile.tsv",
                                                     sep="\t", index=False, header=False)
pd.DataFrame(adata_r.var_names[np.argsort(-dsdt_q4_std)].to_list()).to_csv(save_dir + "/dsdt_q4_top_std_genes_ranked_list.tsv",
                                                     sep="\t", index=False, header=False)

# peaks
bin_dict = inter_patient_variation_dict["Q_4"]
dadt_q5_std = np.array(bin_dict["std_dadt"])
dadt_q5_std_top_peaks = adata_a.var_names[dadt_q5_std >= np.percentile(dadt_q5_std, 90)].to_list()
pd.DataFrame(dadt_q5_std_top_peaks).to_csv(save_dir + "/dadt_top_std_genes_90_percentile.tsv",
                                                     sep="\t", index=False, header=False)


with open(save_dir + "/dadt_top_std_genes_90_percentile.bed", "w") as f:
    for peak in dadt_q5_std_top_peaks:
        chrom, positions = peak.split(':')
        start, end = positions.split('-')
        f.write(f"{chrom}\t{start}\t{end}\n")
        
bin_dict = inter_patient_variation_dict["Q_4"]
dadt_q5_std = np.array(bin_dict["std_dadt"])
dadt_q5_std_top_peaks = adata_a.var_names[dadt_q5_std >= np.percentile(dadt_q5_std, 95)].to_list()
pd.DataFrame(dadt_q5_std_top_peaks).to_csv(save_dir + "/dadt_top_std_genes_95_percentile.tsv",
                                                     sep="\t", index=False, header=False)

with open(save_dir + "/dadt_top_std_genes_95_percentile.bed", "w") as f:
    for peak in dadt_q5_std_top_peaks:
        chrom, positions = peak.split(':')
        start, end = positions.split('-')
        f.write(f"{chrom}\t{start}\t{end}\n")

bg_peaks = adata_a.var_names.to_list()
with open(save_dir + "/dadt_top_std_genes_background.bed", "w") as f:
    for peak in bg_peaks:
        chrom, positions = peak.split(':')
        start, end = positions.split('-')
        f.write(f"{chrom}\t{start}\t{end}\n")


# check the variable peaks associated gene in Q_4 with genes in Q_5
dadt_genes = pd.read_csv("/home/nomura/Proj/mmvelo/experiments/Greenleaf_multiome_Cond_merged_all_missing/2024-01-19T11:22:03_barch_128/downstream_analysis/result/inter_patient_variation/GREAT_result/great_res.tsv",
            sep="\t", ).index.to_list()
dsdt_genes = pd.read_csv(save_dir + "/dsdt_q5_top_std_genes_ranked_list.tsv", sep="\t", header=None)[0].to_list()

rank_list = []
for gene in dadt_genes:
    if gene in dsdt_genes:
        rank_list.append((gene, dsdt_genes.index(gene)))
    else:
        rank_list.append((gene, np.nan))

