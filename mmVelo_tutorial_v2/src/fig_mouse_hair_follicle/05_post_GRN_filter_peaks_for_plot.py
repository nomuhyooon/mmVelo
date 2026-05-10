import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import anndata as ad
import scanpy as sc
from scipy.io import mmwrite, mmread
import seaborn as sns
import scipy
from arboreto.algo import grnboost2
import pickle
import collections
import pycistarget
import pyranges as pr
import pickle
from pycistarget.motif_enrichment_cistarget import *
from scipy.stats import ranksums
from statsmodels.stats.multitest import fdrcorrection
import collections



# Load the TSV file into a DataFrame
file_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/GRN_inference_sep/network_filtered_fdr_1e-4_bh.tsv"
data = pd.read_csv(file_path, sep='\t')
data[['chr', 'start', 'end']] = data.target.str.extract(r'([^:]+):(\d+)-(\d+)')


collections.Counter(data[data.correlation > 0].TF)

# Create a PyRanges object from the data
gr = pr.PyRanges(data.rename(columns={'chr': 'Chromosome', 'start': 'Start', 'end': 'End'}))

# Define the target region as a PyRanges object
target_range = pr.PyRanges(pd.DataFrame({
    'Chromosome': ['chr11'],
    'Start': [94616117],
    'End': [95623212]
}))

# Perform the overlap operation
overlap_result = gr.intersect(target_range)

# Convert the result back to a DataFrame and save to a TSV file
overlap_result_df = overlap_result.df.rename(columns={'Chromosome': 'chr', 'Start': 'start', 'End': 'end'})
output_path_genomic_ranges = '/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/GRN_inference_sep/filtered_target_overlap_genomic_ranges.tsv'
overlap_result_df.to_csv(output_path_genomic_ranges, sep='\t', index=False)

len(set(data.TF)) # 135

len(overlap_result_df) # 353
len(set(overlap_result_df.TF)) # 97
len(set(overlap_result_df.target)) # 53

len(overlap_result_df[overlap_result_df.correlation > 0]) # 178
len(set(overlap_result_df[overlap_result_df.correlation > 0].TF)) # 63
len(set(overlap_result_df[overlap_result_df.correlation > 0].target)) # 46

len(set(overlap_result_df[overlap_result_df.correlation < 0].TF)) # 63
len(set(overlap_result_df[overlap_result_df.correlation < 0].target)) # 46

overlap_result_df_pos = overlap_result_df[overlap_result_df.correlation > 0]
overlap_result_df_pos.loc[overlap_result_df_pos.groupby('target')["qvalue"].idxmin()]
len(set(overlap_result_df_pos.loc[overlap_result_df_pos.groupby('target')["qvalue"].idxmin()].TF)) # 30 
len(set(overlap_result_df_pos.loc[overlap_result_df_pos.groupby('target')["qvalue"].idxmin()].target)) # 46
collections.Counter(overlap_result_df_pos.loc[overlap_result_df_pos.groupby('target')["qvalue"].idxmin()].TF)

overlap_result_df_neg = overlap_result_df[overlap_result_df.correlation < 0]
overlap_result_df_neg.loc[overlap_result_df_neg.groupby('target')["qvalue"].idxmin()]
len(set(overlap_result_df_neg.loc[overlap_result_df_neg.groupby('target')["qvalue"].idxmin()].TF)) # 35
len(set(overlap_result_df_neg.loc[overlap_result_df_neg.groupby('target')["qvalue"].idxmin()].target)) # 50