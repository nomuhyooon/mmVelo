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
from scipy.sparse import csr_matrix

np.random.seed(42)

# load anndata
dir_path = "/home/nomura/Proj/mmvelo/data/10x_multiome_brain_repre_wo_IN"
adata_r = sc.read_loom(dir_path + "/adata_rna.loom", obs_names="obs_names", var_names="var_names")
adata_a = sc.read_loom(dir_path + "/adata_atac.loom", obs_names="obs_names", var_names="var_names")

dir_path = "/home/nomura/Proj/mmvelo/data/10x_multiome_brain_repre_wo_IN/analysis/feature_linkage/peak_gene_associations_500kb_cut_p005_sc001.tsv"
peak_gene_linkage = pd.read_csv(dir_path, sep="\t")
peak_gene_linkage.peak = peak_gene_linkage.peak.str.replace("-", ":", 1)

peak_gene_linkage["peak"][0]
adata_a.var_names[0]

common_peaks = list(set(peak_gene_linkage.peak) & set(adata_a.var_names))
len(common_peaks)
adata_a.n_vars

peak_gene_mat = np.zeros((adata_a.n_vars, adata_r.n_vars), dtype=int)
peak_gene_mat.shape # n_peaks x n_genes

for i, peak in enumerate(common_peaks):
    print(i / len(common_peaks))
    peak_idx = np.where(adata_a.var_names == peak)[0].item()
    gene_rows = peak_gene_linkage[peak_gene_linkage.peak == peak].index.to_numpy()
    for gene_row in gene_rows:
        gene = peak_gene_linkage.loc[gene_row].gene
        if gene in adata_r.var_names:
            gene_idx = np.where(adata_r.var_names == gene)[0].item()
            peak_gene_mat[peak_idx, gene_idx] += 1



dir_path = "/home/nomura/Proj/mmvelo/data/10x_multiome_brain_repre_wo_IN/analysis/feature_linkage/peak_gene_linkage.mtx"
mmwrite(dir_path, csr_matrix(peak_gene_mat))

# check
(mmread(dir_path).toarray() == peak_gene_mat).sum() == (25071 * 3072)