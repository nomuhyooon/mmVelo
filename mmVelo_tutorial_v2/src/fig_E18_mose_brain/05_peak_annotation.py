import numpy as np
import pybedtools
import scanpy as sc
import pandas as pd
import re
from scipy.io import mmwrite, mmread
from scipy.sparse import csr_matrix

file_path = "/home/nomura/Proj/mmvelo/data/10x_multiome_brain_repre_wo_IN/adata_rna.loom"
adata_r = sc.read_loom(file_path, obs_names="obs_names", var_names="var_names")

file_path = "/home/nomura/Proj/mmvelo/data/10x_multiome_brain_repre_wo_IN/adata_atac.loom"
adata_a = sc.read_loom(file_path, obs_names="obs_names", var_names="var_names")
peak_names = adata_a.var_names

columns = ["chrom", "start", "end"]
peak_df = pd.DataFrame(columns=columns)
for peak in peak_names:
    chrom, pos = peak.split(":")
    start, end = pos.split("-")
    peak_df = peak_df.append(pd.Series([chrom, int(start), int(end)], index=columns),
                             ignore_index=True)
peak_df

# read peak annotation file provided by 10x
file_path = "/home/nomura/Proj/mmvelo/data/10x_multiome_brain_repre_wo_IN/analysis/feature_linkage/peak_annotation.tsv"
peak_anno = pd.read_csv(file_path, sep="\t")
(peak_anno.peak_type == "promoter").sum() # 17899

# restrict to promoter peak
peak_anno_prom = peak_anno[peak_anno.peak_type == "promoter"]
peak_anno_prom.index = np.arange(peak_anno_prom.shape[0])
peak_anno_prom

# create matrix which maps chromatin states to promoter states
peak2prom_mat = np.zeros((adata_a.n_vars, adata_r.n_vars))
for index, row in peak_anno_prom.iterrows():
    if row["gene"] in adata_r.var_names:
        peak_name = row["chrom"] + ":" + str(row["start"]) + "-" + str(row["end"])
        if peak_name in adata_a.var_names:    
            peak_idx = np.where(adata_a.var_names == peak_name)[0].item()
            gene_idx = np.where(adata_r.var_names == row["gene"])[0].item()
            peak2prom_mat[peak_idx, gene_idx] += 1

# check how many gene promoters are in data
peak2prom_mat.sum() # 1227
(peak2prom_mat.sum(0) > 0).sum() # 1109 genes

# save
dir_path = "/home/nomura/Proj/mmvelo/data/10x_multiome_brain_repre_wo_IN/analysis/feature_linkage/peak_promoter_linkage.mtx"
mmwrite(dir_path, csr_matrix(peak2prom_mat))

# check wheteher the data is correctly saved
(mmread(dir_path).toarray() == peak2prom_mat).sum() == (25071 * 3072)


#####
# next make gene x TSS dataframe
# read gtf annotation file
file_path = "/home/nomura/Proj/mmvelo/data/10x_multiome_brain_repre_wo_IN/analysis/feature_linkage/gencode.vM10.annotation.gtf"
gtf_df = pd.read_csv(file_path, sep='\t', comment='#', header=None)
gtf_df

gene_df = gtf_df[gtf_df[2] == "gene"]
gene_df = gene_df[[0,3,4,6,8]]
gene_df.columns = ["chr", "start", "end", "strand", "annotation"]
gene_df["tss"] = gene_df.apply(lambda row: row["start"] if row["strand"] == "+" else row["end"], axis=1)
gene_df["gene"] = gene_df[gene_df.columns[-2]].str.extract(r'gene_name\s+"([^"]+)"')
gene_df = gene_df.drop(columns=["annotation"])
gene_df

# some genes have multiple TSSs, so remove overlap
# use the most upstream TSS as unified TSS
# when strand is +, minimum. when strand is -, maximum.
gene_df_demulti = pd.DataFrame(columns= gene_df.columns)
for gene in list(set(gene_df.gene)):
    gene_i_df = gene_df[gene_df.gene == gene]
    if gene_i_df.shape[0] > 1:
        if list(set(gene_i_df.strand))[0] == "+":
            tss = gene_i_df.tss.min()
            gene_i_df = gene_i_df[gene_i_df.tss == tss]
        elif list(set(gene_i_df.strand))[0] == "-":
            tss = gene_i_df.tss.max()
            gene_i_df = gene_i_df[gene_i_df.tss == tss]
    gene_df_demulti = gene_df_demulti.append(gene_i_df)

c = collections.Counter(list(set(gene_df_demulti.gene)))
c.most_common()[-1]
gene_df_demulti.shape # 48338 6
len(set(gene_df_demulti.gene)) # 48321

# save tsv file
file_path = "/home/nomura/Proj/mmvelo/data/10x_multiome_brain_repre_wo_IN/analysis/feature_linkage/gene_TSS.tsv"
gene_df_demulti.to_csv(file_path, sep="\t", index=None)

gene_df = gene_df_demulti
# next annotate peak with distance.
# read peak-gene linkage file
dir_path = "/home/nomura/Proj/mmvelo/data/10x_multiome_brain_repre_wo_IN/analysis/feature_linkage/peak_gene_associations_500kb_cut_p005_sc001.tsv"
peak_gene_linkage = pd.read_csv(dir_path, sep="\t")
peak_gene_linkage.peak = peak_gene_linkage.peak.str.replace("-", ":", 1)

# restrict to peak contained in our data
peak_mat = []
for peak in peak_gene_linkage.peak:
    if peak in adata_a.var_names:
        peak_mat.append(True)
    else:
        peak_mat.append(False)
peak_gene_linkage = peak_gene_linkage[peak_mat]

#tss_list, strand_list, distance_list = [], [], []
distance_df = pd.DataFrame(columns= ["peak", "gene", "TSS", "distance", "strand"])
for index, row in peak_gene_linkage.iterrows():
    peak_name = row["peak"]
    gene_name = row["gene"]
    if not any(gene_df.gene == gene_name):
        continue
    tss = gene_df[gene_df.gene==gene_name]
    tss, strand = tss["tss"].item(), tss["strand"].item()
    distance = (row["start"]+row["end"])/2 - tss
    if strand == "-":
        distance *= -1
    #tss_list.append(tss)
    #strand_list.append(strand)
    #distance_list.append(distance)
    distance_df.loc[index] = [peak_name, gene_name, tss, distance, strand]

file_path = "/home/nomura/Proj/mmvelo/data/10x_multiome_brain_repre_wo_IN/analysis/feature_linkage/peak_distance.tsv"
distance_df.to_csv(file_path, sep="\t", index=None)








"""
#file_path = "/home/nomura/Proj/mmvelo/data/10x_multiome_brain_repre_wo_IN/refdata_mm10/genes/genes.gtf"
#anno_bed = pybedtools.BedTool(file_path)
file_path = "/home/nomura/Proj/mmvelo/data/10x_multiome_brain_repre_wo_IN/analysis/feature_linkage/gencode.vM10.annotation.gtf"
anno_bed = pybedtools.BedTool(file_path)

file_path = "/home/nomura/Proj/mmvelo/data/10x_multiome_brain_repre_wo_IN/adata_atac.loom"
adata_a = sc.read_loom(file_path, obs_names="obs_names", var_names="var_names")
peak_names = adata_a.var_names

columns = ["chrom", "start", "end"]
peak_df = pd.DataFrame(columns=columns)
for peak in peak_names:
    chrom, pos = peak.split(":")
    start, end = pos.split("-")
    peak_df = peak_df.append(pd.Series([chrom, int(start), int(end)], index=columns),
                             ignore_index=True)
peak_bed = pybedtools.BedTool.from_dataframe(peak_df)

overlap_features = peak_bed.intersect(anno_bed, wa=True, wb=True)
overlap_df = overlap_features.to_dataframe()
overlap_df.columns



gene = []
for index, row in overlap_df.iterrows():
    peak_name = row["chrom"] + ":" + str(row["start"]) + "-" + str(row["end"])
    block_starts = row["blockStarts"]
    match = re.search(r'gene_name\s+"([^"]+)"', block_starts)
    if match:
        gene_name = match.group(1)
    else:
        gene_name = ""
    gene.append(gene_name)
overlap_df["gene"] = gene
"""
