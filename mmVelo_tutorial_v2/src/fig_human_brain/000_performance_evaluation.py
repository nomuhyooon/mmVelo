import os
import sys
import json
from typing import Optional
from argparse import ArgumentParser
import numpy as np
import pickle
from pathlib import Path
from tempfile import mkdtemp
import datetime
import scanpy as sc
import scvelo as scv
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

sys.path.append("/home/nomura/Proj/mmvelo/src")
#from mmvelo_multi_cond.dataset_all import MultiomeHumanBrainDataModule_MissingModalityPrediction, DynDataModule_Smooth_MissingModalityPrediction
from mmvelo_multi_cond.dataset_all_modality import MultiomeHumanBrainDataModule_MissingModalityPrediction, DynDataModule_Smooth_MissingModalityPrediction
from mmvelo_multi_cond.utils import fit_beta_gamma, plot_umap, plot_genewise_corr, plot_peakwise_corr, plot_size_factor, plot_vec_embed, fit_beta_gamma, get_filter_idx_raw, plot_su_expr_umap, plot_su_phase_raw, plot_genewise_vel_cossim, plot_su_phase_dsdt_var, plot_fluctuation_umap, fit_beta_gamma_scale, compute_cossim_genewise_contribution, plot_vec_embed_tanh, plot_genewise_vel_cossim_scale, plot_filtered_vec_embed
#from mmvelo_multi_cond.models_missingmodality_all_adv import DREG_PRE, DREG_DYN, EarlyStoppingWithWarmup
from mmvelo_multi_cond.models_missingmodality_all_adv_modadv import DREG_PRE, DREG_DYN, EarlyStoppingWithWarmup
from mmvelo_multi.utils import plot_genewise_corr_for_fig, plot_peakwise_corr_for_fig, plot_size_factor_for_fig


parser = ArgumentParser(description="mmVelo hyperparameters")
#parser.add_argument('--experiment', type=str, default='Greenleaf_multiome_Cond')
parser.add_argument('--experiment', type=str, default='Greenleaf_multiome_Cond_merged_all_missing')
parser.add_argument('--experiment_names', type=str, default='missing_modality_prediction')
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
parser.add_argument('--pre_num_epochs', type=int, default=500)
parser.add_argument('--num_epochs', type=int, default=500) #
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_moment', type=float, default=1e-2)
parser.add_argument('--lr_dyn', type=float, default=1e-4)
parser.add_argument('--su_corr', type=float, default=-1)
parser.add_argument('--su_ratio', type=int, default=50)
parser.add_argument('--min_counts_su', type=int, default=20)
parser.add_argument('--llik_scaling', type=bool, default=True)
parser.add_argument('--patience', type=int, default=30)
parser.add_argument('--n_neighbors', type=int, default=50)
parser.add_argument('--warmup', type=int, default=30) #
parser.add_argument('--warmup_post_pre', type=int, default=1) #
parser.add_argument('--warmup_dyn', type=int, default=0)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=128) #
parser.add_argument('--comment', type=str, default="Trevino et al., 2021 Cell multiome data. dyn inference by only RNA, using DetWarmup, ZINB dist. momoent. cossim loss, using tanh")
args = parser.parse_args()

# random seed
# https://pytorch.org/docs/stable/notes/randomness.html
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# set up run path
os.chdir('/home/nomura/Proj/mmvelo') #
runPath = "/home/nomura/Proj/mmvelo/experiments/Greenleaf_multiome_Cond_merged_all_missing/2024-01-19T11:22:03_barch_128_for_analysis"
tb_logger = pl_loggers.TensorBoardLogger(save_dir=runPath+"/")
print(runPath)

print("loading DataModule...")
dm = MultiomeHumanBrainDataModule_MissingModalityPrediction(batch_size=args.batch_size,
                                       min_counts_genes=args.min_counts_genes, min_counts_peaks=args.min_counts_peaks,
                                       batch_sub=False, filter_outliers=False,
                                       pretrain_multi=True,
                                       )

print("loading Models...")
model = DREG_PRE(dm.rna_dim, dm.atac_dim, args.r_h1dim, args.r_h2dim, args.a_h1dim, args.a_h2dim,
                 args.zdim, args.d_h_dim, dm.num_cat, dm.l_prior_r, dm.l_prior_a, args.lr, 
                 z_learnable=args.z_learnable, d_coeff=args.d_coeff,
                 warmup=args.warmup, llik_scaling=args.llik_scaling,
                 pretrain_multi_end=False
                 )
model.set_norm_mat(dm)
model.set_retain_gene_idx(dm)
trainer = pl.Trainer(gpus=1, max_epochs=args.pre_num_epochs, logger = tb_logger, 
                    callbacks=[EarlyStoppingWithWarmup(monitor="val_elbo_loss", mode="min", patience=args.patience, warmup=args.warmup, verbose=True),
                            ModelCheckpoint(dirpath=runPath, filename="checkpoint_pre_multi", monitor="val_elbo_loss", save_top_k=1)])

dm.pretrain_multi_end()
model.pretrain_multi_end=True

trainer = pl.Trainer(gpus=1, max_epochs=args.num_epochs, logger = tb_logger, 
                    callbacks=[EarlyStoppingWithWarmup(monitor="val_elbo_loss", mode="min", patience=args.patience, warmup=args.warmup_post_pre, verbose=True),
                            ModelCheckpoint(dirpath=runPath, filename="checkpoint_pre", monitor="val_elbo_loss", save_top_k=1)])

print("pretrain ended.")
result = trainer.predict(model=model, dataloaders=dm.all_dataloader(), ckpt_path=runPath+"/checkpoint_pre.ckpt")

zs = []
ps_zs = []
pu_zs = []
pa_zs = []
lr = []
la = []
s_raw = []
u_raw = []

for i in range(len(result)):
    zs += [result[i][0].cpu()] 
    ps_zs += [result[i][1].cpu()] # note this is sizefactor-multiplied rec_s, so not suitable for computing velocity embedding 
    pu_zs += [result[i][2].cpu()]
    pa_zs += [result[i][3].cpu()]
    lr += [result[i][4].cpu()]
    la += [result[i][5].cpu()]
    s_raw += [result[i][6].cpu()]
    u_raw += [result[i][7].cpu()]
del result

zs = torch.cat(zs).numpy()
ps_zs = torch.cat(ps_zs).numpy()
pu_zs = torch.cat(pu_zs).numpy()
pa_zs = torch.cat(pa_zs).numpy()
lr = torch.cat(lr).numpy()
la = torch.cat(la).numpy()
s_raw = torch.cat(s_raw).numpy()
u_raw = torch.cat(u_raw).numpy()

dm.adata_r.obsm["latent"] = zs
dm.adata_r.layers["rec_s"] = ps_zs
dm.adata_r.layers["rec_u"] = pu_zs
dm.adata_r.obsm["lr"] = lr
dm.adata_r.layers["s_raw"] = s_raw
dm.adata_r.layers["u_raw"] = u_raw

dm.adata_a.obsm["latent"] = zs
dm.adata_a.layers["rec_a"] = pa_zs
dm.adata_a.obsm["la"] = la

plot_umap(dm, runPath, embedding=True, cluster_name="clusters")

plot_genewise_corr(dm, runPath, spliced=True, unspliced=True)
plot_peakwise_corr(dm, runPath, a=True)
plot_size_factor(dm, runPath, rna=True, atac=True)

plot_genewise_corr_for_fig(dm, runPath, spliced=True)
plot_genewise_corr_for_fig(dm, runPath, unspliced=True)
plot_peakwise_corr_for_fig(dm, runPath, a=True)
plot_size_factor_for_fig(dm, runPath, rna=True)
plot_size_factor_for_fig(dm, runPath, atac=True)



dm_s = DynDataModule_Smooth_MissingModalityPrediction(dm, n_neighbors=args.n_neighbors,
                                                      modality_wise_smoothing=False) #
model = DREG_PRE.load_from_checkpoint(runPath+"/checkpoint_pre.ckpt",
                    rna_dim=dm.rna_dim, atac_dim=dm.atac_dim, r_h1_dim=args.r_h1dim, r_h2_dim=args.r_h2dim, 
                    a_h1_dim=args.a_h1dim, a_h2_dim=args.a_h2dim, z_dim=args.zdim, d_h_dim=args.d_h_dim, cat_dim=dm_s.num_cat,
                    l_prior_r=dm.l_prior_r, l_prior_a=dm.l_prior_a, lr=args.lr_moment,
                    z_learnable=args.z_learnable, d_coeff=args.d_coeff,
                    warmup=args.warmup, llik_scaling=args.llik_scaling, 
                    pretrain_first_end=True, strict=False
                    )
model.set_norm_mat(dm_s)
model.set_retain_gene_idx(dm_s)

trainer = pl.Trainer(gpus=1, max_epochs=args.num_epochs, logger = tb_logger, 
                    callbacks=[EarlyStoppingWithWarmup(monitor="val_elbo_loss", mode="min", patience=args.patience, warmup=args.warmup, verbose=True),
                            ModelCheckpoint(dirpath=runPath, filename="checkpoint_pre_sec", monitor="val_elbo_loss", save_top_k=1)])

print("pretrain ended.")
result = trainer.predict(model=model, dataloaders=dm_s.all_dataloader(), ckpt_path=runPath+"/checkpoint_pre_sec.ckpt")

zs = []
s_raw = []
u_raw = []
a_raw = []

for i in range(len(result)):
    zs += [result[i][0].cpu()] 
    s_raw += [result[i][6].cpu()]
    u_raw += [result[i][7].cpu()]
    a_raw += [result[i][8].cpu()]
del result

zs = torch.cat(zs).numpy()
s_raw = torch.cat(s_raw).numpy()
u_raw = torch.cat(u_raw).numpy()
a_raw = torch.cat(a_raw).numpy()

dm_s.adata_r.layers["s_raw"] = s_raw
dm_s.adata_r.layers["u_raw"] = u_raw
dm_s.adata_a.layers["a_raw"] = a_raw

plot_genewise_corr(dm_s, runPath, ms=True, mu=True)
plot_peakwise_corr(dm_s, runPath, ma=True)

# implement with missing modality...
import scipy
def colwise_pearsonr(r_c, r_ld):
    val = np.array([scipy.stats.pearsonr(r_c[:, i], r_ld[:, i])[0] for i in range(r_c.shape[1])])
    return(val)

# set train/test/val split in adata
dm_s.adata_r.obs["train_test_val"] = np.array(dm_s.adata_r.obs["modality"].copy())
dm_s.adata_a.obs["train_test_val"] = np.array(dm_s.adata_r.obs["modality"].copy())
for key in dm_s.idx.keys():
    dm_s.adata_r.obs["train_test_val"][dm_s.idx[key]] = key
    dm_s.adata_a.obs["train_test_val"][dm_s.idx[key]] = key
    
def plot_genewise_corr_for_fig_for_missing(dm, exp_name, ms=False, mu=False):
    rna_nonzero_cells = (dm.adata_r.obs["modality"] != "atac")
    adata_r_nonzero_cells = dm.adata_r[rna_nonzero_cells, :]
    if ms:
        genewise_log_count = np.log10(np.sum(adata_r_nonzero_cells.layers["spliced_count"].toarray(), axis=0))
        train_idx = (adata_r_nonzero_cells.obs["train_test_val"] == "train")
        test_idx = (adata_r_nonzero_cells.obs["train_test_val"] == "test")
        train_corr = colwise_pearsonr(adata_r_nonzero_cells.layers["Ms"][train_idx, :],
                                     adata_r_nonzero_cells.layers["s_raw"][train_idx, :])
        test_corr = colwise_pearsonr(adata_r_nonzero_cells.layers["Ms"][test_idx, :],
                                     adata_r_nonzero_cells.layers["s_raw"][test_idx, :])
    
        fig, ax = plt.subplots(1, 1, figsize=(5, 5 * 1), dpi=300)
        ax.scatter(genewise_log_count, train_corr, color="blue", s=1, label="train")
        ax.scatter(genewise_log_count, test_corr, color="red", s=1, label="test")
        plt.xlabel("log10 total count")
        plt.ylabel("pearson correlation")
        ax.legend()
        plt.title("Ms")
        plt.savefig(exp_name+"/train_test_corr_Ms_for_fig.png", dpi=300);plt.close("all")

    if mu:
        genewise_log_count = np.log10(np.sum(adata_r_nonzero_cells.layers["unspliced_count"].toarray(), axis=0))
        train_idx = (adata_r_nonzero_cells.obs["train_test_val"] == "train")
        test_idx = (adata_r_nonzero_cells.obs["train_test_val"] == "test")
        train_corr = colwise_pearsonr(adata_r_nonzero_cells.layers["Mu"][train_idx, :],
                                     adata_r_nonzero_cells.layers["u_raw"][train_idx, :])
        test_corr = colwise_pearsonr(adata_r_nonzero_cells.layers["Mu"][test_idx, :],
                                     adata_r_nonzero_cells.layers["u_raw"][test_idx, :])
    
        fig, ax = plt.subplots(1, 1, figsize=(5, 5 * 1), dpi=300)
        ax.scatter(genewise_log_count, train_corr, color="blue", s=1, label="train")
        ax.scatter(genewise_log_count, test_corr, color="red", s=1, label="test")
        plt.xlabel("log10 total count")
        plt.ylabel("pearson correlation")
        ax.legend()
        plt.title("Mu")
        plt.savefig(exp_name+"/train_test_corr_Mu_for_fig.png", dpi=300);plt.close("all")

def plot_peakwise_corr_for_fig_for_missing(dm, exp_name, ma=False):
    atac_nonzero_cells = (dm.adata_r.obs["modality"] != "rna")
    adata_a_nonzero_cells = dm.adata_a[atac_nonzero_cells, :]
    if ma:
        varwise_log_count = np.log10(np.sum(adata_a_nonzero_cells.layers["atac_count"].toarray(), axis=0))
        train_idx = (adata_a_nonzero_cells.obs["train_test_val"] == "train")
        test_idx = (adata_a_nonzero_cells.obs["train_test_val"] == "test")
        train_corr = colwise_pearsonr(adata_a_nonzero_cells.layers["Ma"][train_idx, :],
                                        adata_a_nonzero_cells.layers["a_raw"][train_idx, :])
        test_corr = colwise_pearsonr(adata_a_nonzero_cells.layers["Ma"][test_idx, :],
                                        adata_a_nonzero_cells.layers["a_raw"][test_idx, :])
        
        fig, ax = plt.subplots(1, 1, figsize=(5, 5 * 1), dpi=300)
        ax.scatter(varwise_log_count, train_corr, color="blue", s=0.03, label="train")
        ax.scatter(varwise_log_count, test_corr, color="red", s=0.03, label="test")
        plt.xlabel("log10 total count")
        plt.ylabel("pearson correlation")
        ax.legend()
        plt.title("Ma")
        plt.savefig(exp_name+"/train_test_corr_Ma_for_fig.png", dpi=300);plt.close("all")

# rna
plot_genewise_corr_for_fig_for_missing(dm_s, runPath, ms=True)
plot_genewise_corr_for_fig_for_missing(dm_s, runPath, mu=True)
plot_peakwise_corr_for_fig_for_missing(dm_s, runPath, ma=True)

"""
plot_vec_embed(dm_s, runPath, ss_model=True, color="clusters")

# check reconstructed count ss model
adata_ss = dm_s.adata_r.copy()
del adata_ss.layers["Ms"], adata_ss.layers["Mu"]
adata_ss.layers["Ms"] = dm_s.adata_r.layers["s_raw"]
adata_ss.layers["Mu"] = dm_s.adata_r.layers["u_raw"]
scv.tl.velocity(adata_ss, vkey="velocity_ss", mode="deterministic", )
scv.tl.velocity_graph(adata_ss, vkey="velocity_ss", xkey="Ms", n_jobs=16)
scv.tl.velocity_embedding(adata_ss, basis="umap", vkey="velocity_ss")
scv.pl.velocity_embedding_grid(adata_ss, vkey="velocity_ss", color="clusters", save=runPath+"/dsdt_rec_ss_grid_" + "clusters" +".png", title="dsdt_rec_ss", dpi=300)
scv.pl.velocity_embedding_stream(adata_ss, vkey="velocity_ss", color="clusters", save=runPath+"/dsdt_rec_ss_streamline_" + "clusters" + ".png", title="dsdt_rec_ss")
del adata_ss


print("pre-training second evaluation ended.")

filter_idx = get_filter_idx_raw(dm_s, test_threshold=-1, su_threshold=args.su_corr,
                                su_ratio=args.su_ratio, min_counts_su=args.min_counts_su)
dm_s.adata_r.var["estimated_genes"] = (filter_idx.numpy() == 1)
device = "cuda" if torch.cuda.is_available() else "cpu"



print("loading Models...")
model = DREG_DYN.load_from_checkpoint(runPath+"/checkpoint_pre_sec.ckpt",
                    rna_dim=dm.rna_dim, atac_dim=dm.atac_dim, r_h1_dim=args.r_h1dim, r_h2_dim=args.r_h2dim, 
                    a_h1_dim=args.a_h1dim, a_h2_dim=args.a_h2dim, z_dim=args.zdim, d_h_dim=args.d_h_dim, cat_dim=dm.num_cat,
                    l_prior_r=dm.l_prior_r, l_prior_a=dm.l_prior_a, lr=args.lr_dyn,
                    z_learnable=args.z_learnable, d_coeff=args.d_coeff,
                    filter_idx=filter_idx.to(device), strict=False
                    )
model.set_norm_mat(dm_s)
model.log_beta, model.log_gamma = fit_beta_gamma_scale(dm_s, su_scale=False) # fit kinetics parameter with steady-state model
model.set_beta_gamma_ss_ratio()
model.set_filter_idx(filter_idx)
model.set_grad_for_training()
model.set_retain_gene_idx(dm_s)


trainer = pl.Trainer(gpus=1, max_epochs=args.num_epochs, logger = tb_logger,
                    #track_grad_norm=2, detect_anomaly=True, gradient_clip_algorithm="value", gradient_clip_val=1.,
                    callbacks=[EarlyStoppingWithWarmup(monitor="val_elbo_loss_d", mode="min", patience=args.patience, warmup=args.warmup_dyn, verbose=True),
                            ModelCheckpoint(dirpath=runPath, filename="checkpoint", monitor="val_elbo_loss_d", save_top_k=1)])

model.compute_dadt = True
result = trainer.predict(model=model, dataloaders=dm_s.all_dataloader(), ckpt_path=runPath+"/checkpoint.ckpt")

zs = []
ds = []
dsdt = []
dsdt_obs = []
ds_var = []
dadt = []
dudt = []

for i in range(len(result)):
    zs += [result[i][0].cpu()]
    ds += [result[i][1].cpu()]
    dsdt += [result[i][2].cpu()]
    dsdt_obs += [result[i][3].cpu()]
    ds_var += [result[i][4].cpu()]
    dadt += [result[i][5].cpu()]
    dudt += [result[i][6].cpu()]

del result
zs = torch.cat(zs).numpy()
ds = torch.cat(ds).numpy()
dsdt = torch.cat(dsdt).numpy()
dsdt_obs = torch.cat(dsdt_obs).numpy()
ds_var = torch.cat(ds_var).numpy()
dadt = torch.cat(dadt).numpy()
dudt = torch.cat(dudt).numpy()
dm_s.adata_r.obsm["latent"] = zs
dm_s.adata_r.obsm["dynamics"] = ds
dm_s.adata_r.layers["dsdt"] = dsdt
dm_s.adata_r.layers["dsdt_obs"] = dsdt_obs
dm_s.adata_r.obsm["d_var"] = ds_var ** 2
dm_s.adata_a.layers["dadt"] = dadt
dm_s.adata_r.layers["dudt"] = dudt

dm_s.adata_r
dm_s.adata_a



save_dir = runPath + "/downstream_analysis/result/anndata"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
    
dm_s.adata_r.write_loom(save_dir + "/adata_rna.loom", write_obsm_varm=True)
dm_s.adata_a.write_loom(save_dir + "/adata_atac.loom", write_obsm_varm=True)

np.savetxt(save_dir+"/norm_mat_a.txt", model.norm_mat_a.numpy())
np.savetxt(save_dir+"/norm_mat_s.txt", model.norm_mat_s.numpy())
np.savetxt(save_dir+"/norm_mat_u.txt", model.norm_mat_u.numpy())

# check whether the data was correctly saved
#adata_r = sc.read_loom(save_dir + "/adata_rna.loom", obs_names="obs_names", var_names="var_names")
#adata_a = sc.read_loom(save_dir + "/adata_atac.loom", obs_names="obs_names", var_names="var_names")
"""