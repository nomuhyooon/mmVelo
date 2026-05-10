import os
import pandas as pd
import collections
import pyranges as pr


# Load the TSV file into a DataFrame
file_path = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/GRN_inference_sep/network_filtered_fdr_1e-4_bh.tsv"
data = pd.read_csv(file_path, sep='\t')
data[['chr', 'start', 'end']] = data.target.str.extract(r'([^:]+):(\d+)-(\d+)')

collections.Counter(data[data.correlation > 0].TF)

# Create a PyRanges object from the data
gr = pr.PyRanges(data.rename(columns={'chr': 'Chromosome', 'start': 'Start', 'end': 'End'}))

# Define the target region as a PyRanges object: Lef1
target_range = pr.PyRanges(pd.DataFrame({
    'Chromosome': ['chr3'],
    'Start': [130800000],
    'End': [131350000]
}))

# Perform the overlap operation
overlap_result = gr.intersect(target_range)

# Convert the result back to a DataFrame and save to a TSV file
overlap_result_df = overlap_result.df.rename(columns={'Chromosome': 'chr', 'Start': 'start', 'End': 'end'})
output_dir = '/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/GRN_inference_sep/downstream_Lef1'
os.mkdir(output_dir)
output_path_genomic_ranges = output_dir + 'filtered_target_overlap_genomic_ranges.tsv'
overlap_result_df.to_csv(output_path_genomic_ranges, sep='\t', index=False)

len(set(data.TF)) # 135

len(overlap_result_df) # 278
len(set(overlap_result_df.TF)) # 78
len(set(overlap_result_df.target)) # 44

len(overlap_result_df[overlap_result_df.correlation > 0]) # 152
len(set(overlap_result_df[overlap_result_df.correlation > 0].TF)) # 53
len(set(overlap_result_df[overlap_result_df.correlation > 0].target)) # 39

len(set(overlap_result_df[overlap_result_df.correlation < 0].TF)) # 40
len(set(overlap_result_df[overlap_result_df.correlation < 0].target)) # 35

overlap_result_df_pos = overlap_result_df[overlap_result_df.correlation > 0]
overlap_result_df_pos.loc[overlap_result_df_pos.groupby('target')["qvalue"].idxmin()]
len(set(overlap_result_df_pos.loc[overlap_result_df_pos.groupby('target')["qvalue"].idxmin()].TF)) # 30 
len(set(overlap_result_df_pos.loc[overlap_result_df_pos.groupby('target')["qvalue"].idxmin()].target)) # 46
c = collections.Counter(overlap_result_df_pos.loc[overlap_result_df_pos.groupby('target')["qvalue"].idxmin()].TF)
top_tfs = c.most_common()[:9] # top 9 TFs regulate >=2 peaks
for tf in top_tfs:
    tf = tf[0]
    tf_target = list(overlap_result_df_pos.loc[overlap_result_df_pos.groupby('target')["qvalue"].idxmin()][overlap_result_df_pos.loc[overlap_result_df_pos.groupby('target')["qvalue"].idxmin()].TF == tf].target)
    print(tf, tf_target)
    
    # create bed file for tf
    bed_entries = []
    for peak in tf_target:
        chromosome, start, end = peak.split(":")[0], peak.split(":")[1].split("-")[0], peak.split(":")[1].split("-")[1]
        bed_entries.append([chromosome, start, end])
    bed_df = pd.DataFrame(bed_entries, columns=['chr', 'start', 'end'])
    
    output_bed_file = output_dir + "/" + tf + "_bed.bed"
    bed_df[["chr", "start", "end"]].to_csv(output_bed_file, sep='\t', header=False, index=False)



overlap_result_df_neg = overlap_result_df[overlap_result_df.correlation < 0]
overlap_result_df_neg.loc[overlap_result_df_neg.groupby('target')["qvalue"].idxmin()]
len(set(overlap_result_df_neg.loc[overlap_result_df_neg.groupby('target')["qvalue"].idxmin()].TF)) # 35
len(set(overlap_result_df_neg.loc[overlap_result_df_neg.groupby('target')["qvalue"].idxmin()].target)) # 50
collections.Counter(overlap_result_df_neg.loc[overlap_result_df_neg.groupby('target')["qvalue"].idxmin()].TF)