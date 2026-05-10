import os
import sys
from argparse import ArgumentParser
import numpy as np
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
#parser.add_argument('--experiment', type=str, default='SHARE-seq_hf')
parser.add_argument('--experiment', type=str, default='multiome_brain_rep_wo_IN')
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
parser.add_argument('--n_neighbors', type=int, default=100)
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
runPath = "/home/nomura/Proj/mmvelo/experiments/multiome_brain_rep_wo_IN/2023-05-07T15:19:02_s43_k100_for_analysis"
tb_logger = pl_loggers.TensorBoardLogger(save_dir=runPath+"/")
print(runPath)

print("loading DataModule...")
dm = MultiomeBrainDataModule_Pre(batch_size=args.batch_size, n_top_genes=args.n_genes, n_top_peaks=args.n_peaks, min_counts_genes=args.min_counts_genes, min_counts_peaks=args.min_counts_peaks)
#dm = ShareSeqDataModule_Pre(batch_size=args.batch_size, min_counts_genes=args.min_counts_genes, min_counts_peaks=args.min_counts_peaks, n_top_genes=args.n_genes)
#dm = ShareSeqFilteredDataModule_Pre(batch_size=args.batch_size, 
#                                    n_top_genes=args.n_genes, n_top_peaks=args.n_peaks,
#                                    min_shared_counts=args.min_counts_genes)
# dm = SHARESeqHFDataModule_Pre(batch_size=args.batch_size, n_top_genes=args.n_genes, n_top_peaks=args.n_peaks, min_counts_genes=args.min_counts_genes, min_counts_peaks=args.min_counts_peaks)

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

sc.pp.neighbors(dm.adata_r, n_neighbors=15, use_rep="latent")

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

print("loading trained models...")
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
dm_s.adata_r.layers["dudt"] = dudt
dm_s.adata_r.layers["dsdt_obs"] = dsdt_obs
dm_s.adata_r.obsm["d_var"] = ds_var ** 2
dm_s.adata_a.layers["dadt"] = dadt


save_dir = runPath + "/downstream_analysis/result/anndata"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

dm_s.adata_r.write_loom(save_dir + "/adata_rna.loom", write_obsm_varm=True)
dm_s.adata_a.write_loom(save_dir + "/adata_atac.loom", write_obsm_varm=True)

# check whether the data was correctly saved
#adata_r = sc.read_loom(save_dir + "/adata_rna.loom", obs_names="obs_names", var_names="var_names")
#adata_a = sc.read_loom(save_dir + "/adata_atac.loom", obs_names="obs_names", var_names="var_names")