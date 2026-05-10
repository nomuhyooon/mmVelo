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

dir_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/dadt_clustering"
adata_peak = sc.read_loom(dir_path+"/adata_dadt_cluster.loom")
adata_peak.obs_names = adata_peak.obs["obs_names"]
peaks = list(adata_peak.obs_names)

save_path = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis/downstream_analysis/result/dadt_clustering/GREAT"
if not os.path.exists(save_path):
    os.mkdir(save_path)

for cluster in range(int(max(adata_peak.obs["leiden"])) + 1):
    cluster = str(cluster)
    cluster_peaks = adata_peak.obs_names[adata_peak.obs["leiden"] == cluster].to_list()
    print(cluster, len(cluster_peaks))
    pd.DataFrame(cluster_peaks).to_csv(save_path + f"/dadt_cluster_{cluster}_peaks.tsv",
                                       sep="\t", header=None, index=None)
    
    with open(save_path + f"/dadt_cluster_{cluster}_peaks.bed", "w") as f:
        for peak in cluster_peaks:
            chrom, positions = peak.split(':')
            start, end = positions.split('-')
            f.write(f"{chrom}\t{start}\t{end}\n")


pd.DataFrame(peaks).to_csv(save_path + f"/dadt_cluster_background_peaks.tsv",
                                       sep="\t", header=None, index=None)
with open(save_path + f"/dadt_cluster_background_peaks.bed", "w") as f:
    for peak in peaks:
        chrom, positions = peak.split(':')
        start, end = positions.split('-')
        f.write(f"{chrom}\t{start}\t{end}\n")