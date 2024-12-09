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

sys.path.append("./src")
from dataset import MultiomeBrainDataModule_Pre, DynDataModule_Smooth
from utils import fit_beta_gamma, plot_umap, plot_genewise_corr, plot_peakwise_corr, plot_size_factor, plot_vec_embed, fit_beta_gamma, get_filter_idx_raw, plot_su_expr_umap, plot_su_phase_raw, plot_genewise_vel_cossim, plot_su_phase_dsdt_var, plot_fluctuation_umap, fit_beta_gamma_scale, compute_cossim_genewise_contribution, plot_vec_embed_tanh, plot_genewise_vel_cossim_scale, plot_filtered_vec_embed
from models import DREG_PRE, DREG_DYN, EarlyStoppingWithWarmup

parser = ArgumentParser(description="mmVelo hyperparameters")
parser.add_argument('--experiment', type=str, default='multiome_mouse_brain')
parser.add_argument('--custom_data', type=bool, default=False)
parser.add_argument('--runpath', type=str, default=None)
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
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--comment', type=str, default="Multiome mouse E18 brain.")
args = parser.parse_args()

# random seed
# https://pytorch.org/docs/stable/notes/randomness.html
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# set up run path
#os.chdir('/home/nomura/Proj/mmvelo') #
runId = datetime.datetime.now().isoformat()
experiment_dir = Path('./experiments/' + args.experiment)
experiment_dir.mkdir(parents=True, exist_ok=True)
if args.runpath is None:
    runPath = mkdtemp(prefix=runId, dir=str(experiment_dir))
else:
    runPath = str(experiment_dir) + "/" + args.runpath
    os.mkdir(runPath)
tb_logger = pl_loggers.TensorBoardLogger(save_dir=runPath+"/")
print(f"Output will be saved in {runPath}")

with open(runPath + "/params.json", mode="w") as f:
    json.dump(args.__dict__, f, indent=4)

print("loading DataModule...")
if args.custom_data is False:
    dm = MultiomeBrainDataModule_Pre(batch_size=args.batch_size, n_top_genes=args.n_genes, n_top_peaks=args.n_peaks, min_counts_genes=args.min_counts_genes, min_counts_peaks=args.min_counts_peaks,
                                 wo_in=True)
else:
    dm = MultiomeBrainDataModule_Pre(batch_size=args.batch_size, n_top_genes=args.n_genes, n_top_peaks=args.n_peaks, min_counts_genes=args.min_counts_genes, min_counts_peaks=args.min_counts_peaks,
                                 wo_in=True)

print("loading Models...")
model = DREG_PRE(dm.rna_dim, dm.atac_dim, args.r_h1dim, args.r_h2dim, args.a_h1dim, args.a_h2dim,
                 args.zdim, args.d_h_dim, dm.l_prior_r, dm.l_prior_a, args.lr, 
                 z_learnable=args.z_learnable, d_coeff=args.d_coeff,
                 warmup=args.warmup ,llik_scaling=args.llik_scaling
                 )
model.set_norm_mat(dm)
model.set_retain_gene_idx(dm)
trainer = pl.Trainer(gpus=1, max_epochs=args.num_epochs, logger = tb_logger, 
                    callbacks=[EarlyStoppingWithWarmup(monitor="val_elbo_loss", mode="min", patience=args.patience, warmup=args.warmup, verbose=True),
                            ModelCheckpoint(dirpath=runPath, filename="checkpoint_pre", monitor="val_elbo_loss", save_top_k=1)])

print("start training for cell state inference.")
trainer.fit(model=model, datamodule=dm)

print("Ended training for cell state inference.")
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

dm.adata_r.obsm["latent"] = torch.cat(zs).numpy()
dm.adata_r.layers["rec_s"] = torch.cat(ps_zs).numpy()
dm.adata_r.layers["rec_u"] = torch.cat(pu_zs).numpy()

dm.adata_r.obsm["lr"] = torch.cat(lr).numpy()
dm.adata_r.layers["s_raw"] = torch.cat(s_raw).numpy()
dm.adata_r.layers["u_raw"] = torch.cat(u_raw).numpy()

dm.adata_a.obsm["latent"] = torch.cat(zs).numpy()
dm.adata_a.obsm["la"] = la = torch.cat(la).numpy()
dm.adata_a.layers["rec_a"] = torch.cat(pa_zs).numpy()


sc.pp.neighbors(dm.adata_r, n_neighbors=30, use_rep="latent")
if args.custom_data:
    sc.tl.leiden(dm.adata_r, key_added="clusters")
plot_umap(dm, runPath, embedding=True, cluster_name="clusters")

save_dir = runPath + "/pretrain_evaluation"
os.mkdir(save_dir)

plot_genewise_corr(dm, save_dir, spliced=True, unspliced=True)
plot_peakwise_corr(dm, save_dir, a=True)
plot_size_factor(dm, save_dir, rna=True, atac=True)
print("Ended evaluation for cell state inference.")

print("Start training for smoothed profile reconstruction") 
dm_s = DynDataModule_Smooth(dm, n_neighbors=args.n_neighbors)
model = DREG_PRE.load_from_checkpoint(runPath+"/checkpoint_pre.ckpt",
                    rna_dim=dm.rna_dim, atac_dim=dm.atac_dim, r_h1_dim=args.r_h1dim, r_h2_dim=args.r_h2dim, 
                    a_h1_dim=args.a_h1dim, a_h2_dim=args.a_h2dim, z_dim=args.zdim, d_h_dim=args.d_h_dim,
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
trainer.fit(model=model, datamodule=dm_s)

print("Ended training for smoothed profile reconstruction.")
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
dm_s.adata_r.layers["s_raw"] = torch.cat(s_raw).numpy()
dm_s.adata_r.layers["u_raw"] = torch.cat(u_raw).numpy()
dm_s.adata_a.layers["a_raw"] = torch.cat(a_raw).numpy()


plot_genewise_corr(dm_s, save_dir, ms=True, mu=True)
plot_peakwise_corr(dm_s, save_dir, ma=True)
print("Ended evaluation for smoothed profile reconstruction.")

filter_idx = get_filter_idx_raw(dm_s, test_threshold=-1, su_threshold=args.su_corr,
                                su_ratio=args.su_ratio, min_counts_su=args.min_counts_su)
dm_s.adata_r.var["estimated_genes"] = (filter_idx.numpy() == 1)
device = "cuda" if torch.cuda.is_available() else "cpu"

print("loading Models...")
model = DREG_DYN.load_from_checkpoint(runPath+"/checkpoint_pre_sec.ckpt",
                    rna_dim=dm.rna_dim, atac_dim=dm.atac_dim, r_h1_dim=args.r_h1dim, r_h2_dim=args.r_h2dim, 
                    a_h1_dim=args.a_h1dim, a_h2_dim=args.a_h2dim, z_dim=args.zdim, d_h_dim=args.d_h_dim,
                    l_prior_r=dm.l_prior_r, l_prior_a=dm.l_prior_a, lr=args.lr_dyn,
                    z_learnable=args.z_learnable, d_coeff=args.d_coeff,
                    filter_idx=filter_idx.to(device), strict=False
                    )
model.set_norm_mat(dm_s)
model.log_gamma_beta = fit_beta_gamma(dm_s) # fit kinetics parameter with steady-state model
model.log_beta, model.log_gamma = fit_beta_gamma_scale(dm_s, su_scale=False) # fit kinetics parameter with steady-state model
model.set_beta_gamma_ss_ratio()
model.set_filter_idx(filter_idx)
model.set_grad_for_training()
model.set_retain_gene_idx(dm_s)


trainer = pl.Trainer(gpus=1, max_epochs=args.num_epochs, logger = tb_logger,
                    #track_grad_norm=2, detect_anomaly=True, gradient_clip_algorithm="value", gradient_clip_val=1.,
                    callbacks=[EarlyStoppingWithWarmup(monitor="val_elbo_loss_d", mode="min", patience=args.patience, warmup=args.warmup_dyn, verbose=True),
                            ModelCheckpoint(dirpath=runPath, filename="checkpoint", monitor="val_elbo_loss_d", save_top_k=1)])


print("Start training for inferring cell state-dependent dynamics.")
trainer.fit(model=model, datamodule=dm_s)

print("Ended training for inferring cell state-dependent dynamics..")
print("loading trained models...")

model.compute_dadt = True
result = trainer.predict(model=model, dataloaders=dm_s.all_dataloader(), ckpt_path=runPath+"/checkpoint.ckpt")

zs = []
ds = []
dsdt = []
dsdt_obs = []
ds_var = []
dadt = []

for i in range(len(result)):
    zs += [result[i][0].cpu()]
    ds += [result[i][1].cpu()]
    dsdt += [result[i][2].cpu()]
    dsdt_obs += [result[i][3].cpu()]
    ds_var += [result[i][4].cpu()]
    dadt += [result[i][5].cpu()]

del result
dm_s.adata_r.obsm["latent"] = torch.cat(zs).numpy()
dm_s.adata_r.obsm["dynamics"] = torch.cat(ds).numpy()
dm_s.adata_r.layers["dsdt"] = torch.cat(dsdt).numpy()
dm_s.adata_r.layers["dsdt_obs"] = torch.cat(dsdt_obs).numpy()
dm_s.adata_r.obsm["d_var"] = torch.cat(ds_var).numpy()
dm_s.adata_a.layers["dadt"] = torch.cat(dadt).numpy()

with open(runPath + "/learned_params.json", mode="w") as f:
    json.dump(model.get_params(), f, indent=4)

save_dir = runPath + "/est_velocity"
os.mkdir(save_dir)

plot_vec_embed_tanh(dm_s, save_dir, latent=True, spliced=True, dadt=True, color="clusters")

# save results
dm_s.adata_a.write_h5ad(runPath + "/adata_est_atac.h5ad")
dm_s.adata_r.write_h5ad(runPath + "/adata_est_rna.h5ad")