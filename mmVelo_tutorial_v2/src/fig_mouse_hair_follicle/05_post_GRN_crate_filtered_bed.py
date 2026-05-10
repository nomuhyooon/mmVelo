import os
import sys
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

sys.path.append("/home/nomura/Proj/mmvelo/src")
from mmvelo_multi.dataset import MultiomeBrainDataModule_Pre, ShareSeqDataModule_Pre, DynDataModule_Smooth, ShareSeqFilteredDataModule_Pre, SHARESeqHFDataModule_Pre
from mmvelo_multi.utils import fit_beta_gamma, plot_umap, plot_genewise_corr, plot_peakwise_corr, plot_size_factor, plot_vec_embed, fit_beta_gamma, get_filter_idx_raw, plot_su_expr_umap, plot_su_phase_raw, plot_genewise_vel_cossim, plot_su_phase_dsdt_var, plot_fluctuation_umap, fit_beta_gamma_scale, compute_cossim_genewise_contribution, plot_vec_embed_tanh, plot_genewise_vel_cossim_scale, plot_filtered_vec_embed
from mmvelo_multi.models import DREG_PRE, DREG_DYN, EarlyStoppingWithWarmup


parser = ArgumentParser(description="mmVelo hyperparameters")
parser.add_argument('--experiment', type=str, default='SHARE-seq_hf')
#parser.add_argument('--experiment', type=str, default='multiome_brain_rep_wo_IN')
#parser.add_argument('--experiment', type=str, default='multiome_brain_rep')
#parser.add_argument('--experiment_names', type=str, default='dsdt_param_reg_filtered')
parser.add_argument('--n_genes', type=int, default=3000)
parser.add_argument('--n_peaks', type=int, default=20000)
parser.add_argument('--min_counts_genes', type=int, default=10)
parser.add_argument('--min_counts_peaks', type=int, default=10)
parser.add_argument('--r_h1dim', type=int, default=128)
parser.add_argument('--r_h2dim', type=int, default= 64)
parser.add_argument('--a_h1dim', type=int, default=128)
parser.add_argument('--a_h2dim', type=int, default=64)
parser.add_argument('--zdim', type=int, default=10)
parser.add_argument('--d_h_dim', type=int, default=64)
parser.add_argument('--z_learnable', type=bool, default=True)
parser.add_argument('--d_learnable', type=bool, default=True)
parser.add_argument('--d_coeff', type=float, default=1e-2)
parser.add_argument('--num_epochs', type=int, default=1000) #
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_moment', type=float, default=1e-2)
parser.add_argument('--lr_dyn', type=float, default=1e-4)
parser.add_argument('--su_corr', type=float, default=-1)
parser.add_argument('--su_ratio', type=int, default=50)
parser.add_argument('--min_counts_su', type=int, default=20)
parser.add_argument('--llik_scaling', type=bool, default=True)
parser.add_argument('--patience', type=int, default=30)
parser.add_argument('--n_neighbors', type=int, default=50)
parser.add_argument('--warmup', type=int, default=30)
parser.add_argument('--warmup_dyn', type=int, default=10)
parser.add_argument('--seed', type=int, default=43)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--comment', type=str, default="Multiome E18 brain. dyn inference by only RNA, using DetWarmup, ZINB dist. momoent. vMF loss, using tanh")
args = parser.parse_args()


# random seed
# https://pytorch.org/docs/stable/notes/randomness.html
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# set up run path
os.chdir('/home/nomura/Proj/mmvelo') #
runPath = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis"
tb_logger = pl_loggers.TensorBoardLogger(save_dir=runPath+"/")
print(runPath)

print("loading DataModule...")
dm = SHARESeqHFDataModule_Pre(batch_size=args.batch_size, n_top_genes=args.n_genes, n_top_peaks=args.n_peaks, min_counts_genes=args.min_counts_genes, min_counts_peaks=args.min_counts_peaks)
count_mat = pd.DataFrame(dm.adata_a.X.toarray())

peaks = pd.DataFrame(dm.adata_a.var_names.str.extract(r'([^:]+):(\d+)-(\d+)'))
peaks.columns = ["chr", "start", "end"]

cell_bcs = list(dm.adata_a.obs_names)

bed_entries = list()
counter = 0

for peak_idx, (chr_name, start, end) in peaks.iterrows():
    for cell_id, count in count_mat.iloc[:, peak_idx].items():
        if count > 0:
            for _ in range(int(count)):
                bed_entries.append([chr_name, start, end, cell_id])
        counter += 1
        print(counter)
        
bed_df = pd.DataFrame(bed_entries, columns=['chr', 'start', 'end', 'cell_id'])
output_bed_file = "/home/nomura/Proj/mmvelo/experiments/SHARE-seq_hf/2023-08-03T13:31:54_nb_k50_for_analysis/downstream_analysis/result/GRN_inference_sep/filtered_bed.bed"
bed_df[["chr", "start", "end"]].to_csv(output_bed_file, sep='\t', header=False, index=False)
