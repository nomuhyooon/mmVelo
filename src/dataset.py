import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import torch
import pytorch_lightning as pl
from scipy.sparse import csr_matrix
from scvelo.preprocessing.moments import get_moments

def split_data(adata, val_ratio=0.1, test_ratio=0.1):
    total_num = adata.shape[0]
    val_num = int(total_num*val_ratio)
    test_num  = int(total_num*test_ratio)
    idx = np.random.permutation(np.arange(total_num))
    val_idx, test_idx, train_idx = idx[:val_num], idx[val_num:(val_num + test_num)], idx[(val_num + test_num):]
    return dict(val = val_idx, test = test_idx, train = train_idx)

def get_l_prior_r(adata_r):
    #total_count = np.sum(adata.layers["spliced"], axis=1)
    total_count = np.sum(adata_r.layers["spliced"] + adata_r.layers["unspliced"], axis=1)
    mean = np.mean(np.log(total_count))
    std = np.std(np.log(total_count))
    print("RNA library size log-mean, log-std; ", mean, std)
    return torch.distributions.log_normal.LogNormal(mean, std)

def get_norm_mat_r(adata_r, moment=False):
    s, u = adata_r.layers["spliced"], adata_r.layers["unspliced"]
    if moment:
        s_count, u_count = adata_r.layers["spliced_count"], adata_r.layers["unspliced_count"]
    norm_mat_s = np.sum(s, axis=0) / np.sum(s>0, axis=0).reshape(-1)
    norm_mat_s = np.asarray(norm_mat_s).reshape(-1)
    norm_mat_u = np.sum(u, axis=0) / np.sum(u>0, axis=0).reshape(-1)
    norm_mat_u = np.asarray(norm_mat_u).reshape(-1)
    return norm_mat_s, norm_mat_u

def get_norm_mat_a(adata_a, moment=False):
    a = adata_a.X
    if moment:
        a = adata_a.layers["Ma"]
    norm_mat_a = np.sum(a, axis=0) / np.sum(a>0, axis=0).reshape(-1)
    norm_mat_a = np.asarray(norm_mat_a).reshape(-1)
    return norm_mat_a

def get_l_prior_a(adata_a):
    #total_count = np.sum(adata.layers["spliced"], axis=1)
    total_count = np.sum(adata_a.X, axis=1)
    mean = np.mean(np.log(total_count))
    std = np.std(np.log(total_count))
    print("ATAC library size log-mean, log-std; ", mean, std)
    return torch.distributions.log_normal.LogNormal(mean, std)

class VAEDataSet(torch.utils.data.Dataset):
    def __init__(self, adata_r, adata_a, moment=False):
        self.moment = moment
        if self.moment:
            self.s = adata_r.layers["spliced_count"].toarray()
            self.u = adata_r.layers["unspliced_count"].toarray()
            self.a = adata_a.layers["atac_count"].toarray()
            self.ms = adata_r.layers["Ms"]
            self.mu = adata_r.layers["Mu"]
            self.ma = adata_a.layers["Ma"]
        else:
            self.s = adata_r.layers["spliced"].toarray()
            self.u = adata_r.layers["unspliced"].toarray()
            self.a = adata_a.X.toarray()
        self.shape = self.s.shape
        self.obs_names = adata_r.obs_names
        self.gene_names = adata_r.var_names
        self.peak_names = adata_a.var_names

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        s = torch.tensor(self.s[idx,:])
        u = torch.tensor(self.u[idx,:])
        a = torch.tensor(self.a[idx,:])
        if self.moment:
            ms = torch.tensor(self.ms[idx,:])
            mu = torch.tensor(self.mu[idx,:])
            ma = torch.tensor(self.ma[idx,:])
            return s, u, a, ms, mu, ma
        return s, u, a
        

class DynDataModule_Smooth(pl.LightningDataModule):
    def __init__(self, dm_pre, batch_size : int =128, num_workers=2, n_neighbors=30):
        super().__init__()
        self.save_hyperparameters()
        self.adata_r = dm_pre.adata_r.copy()
        self.adata_a = dm_pre.adata_a.copy()
        del (self.adata_r.layers["rec_s"], self.adata_r.layers["rec_u"], 
            self.adata_r.layers["s_raw"], self.adata_r.layers["u_raw"],
            self.adata_r.obsm["lr"], self.adata_a.obsm["la"])
        self.rna_dim, self.atac_dim = self.adata_r.shape[1], self.adata_a.shape[1]
        print("RNA dim : {}".format(self.rna_dim), "ATAC dim : {}".format(self.atac_dim))
        self.idx = dm_pre.idx
        self.n_neighbors = n_neighbors
        self.normalize_counts() ##
        self.norm_mat_r = get_norm_mat_r(self.adata_r)
        self.norm_mat_a = get_norm_mat_a(self.adata_a)
        self.calc_neighbors(n_neighbors = self.n_neighbors)
        self.calc_moments()

        self.retain_genes_idx = dm_pre.retain_genes_idx
        for idx in np.where(self.retain_genes_idx==1)[0]:
            self.norm_mat_r[1][idx] = 0

    def normalize_counts(self):
        self.adata_r.layers["spliced_count"] = self.adata_r.layers["spliced"].copy()
        self.adata_r.layers["unspliced_count"] = self.adata_r.layers["unspliced"].copy()
        self.adata_a.layers["atac_count"] = self.adata_a.X.copy()
        scv.pp.normalize_per_cell(self.adata_r, counts_per_cell_after=1e4)
        scv.pp.log1p(self.adata_r)
        sc.pp.normalize_total(self.adata_a, target_sum=1e4)
        sc.pp.log1p(self.adata_a)

    def calc_neighbors(self, n_neighbors=30):
        sc.pp.neighbors(self.adata_r, n_neighbors=n_neighbors, use_rep="latent")
        self.adata_a.uns["neighbors"] = self.adata_r.uns["neighbors"]
        self.adata_a.obsp["distances"] = self.adata_r.obsp["distances"]
        self.adata_a.obsp["connectivities"] = self.adata_r.obsp["connectivities"]
        self.n_neighbors = n_neighbors

    def calc_moments(self):
        scv.pp.moments(self.adata_r, n_neighbors=30, use_rep="latent")
        self.adata_a.layers["Ma"] = get_moments(self.adata_a)


    def train_dataloader(self):
        train_set = VAEDataSet(self.adata_r[self.idx["train"], :], self.adata_a[self.idx["train"], :], moment=True)
        return torch.utils.data.DataLoader(train_set, batch_size = self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        val_set = VAEDataSet(self.adata_r[self.idx["val"], :], self.adata_a[self.idx["val"], :], moment=True)
        return torch.utils.data.DataLoader(val_set, batch_size = self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False, pin_memory=True)

    def test_dataloader(self):
        test_set = VAEDataSet(self.adata_r[self.idx["test"], :], self.adata_a[self.idx["test"], :], moment=True)
        return torch.utils.data.DataLoader(test_set, batch_size = self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False, pin_memory=True)

    def all_dataloader(self):
        all_set = VAEDataSet(self.adata_r, self.adata_a, moment=True)
        return torch.utils.data.DataLoader(all_set, batch_size = self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False, pin_memory=True)


class MultiomeBrainDataModule_Pre(pl.LightningDataModule):
    def __init__(self, batch_size : int =128, num_workers=2, 
                 min_counts_genes=10, min_counts_peaks=10,
                 n_top_genes=3000, n_top_peaks=20000, sub = False):
        super().__init__()
        self.save_hyperparameters()
        print("Loading data...")
        self.adata_r = sc.read_loom("./data/adata_rna.loom")
        self.adata_a = sc.read_loom("./data/adata_atac.loom")
        self.adata_r.obs_names = self.adata_r.obs.obs_names
        self.adata_r.var_names = self.adata_r.var.var_names
        self.adata_a.obs_names = self.adata_a.obs.obs_names
        self.adata_a.var_names = self.adata_a.var.var_names
        self.rna_dim, self.atac_dim = self.adata_r.shape[1], self.adata_a.shape[1]
        print("RNA dim : {}".format(self.rna_dim), "ATAC dim : {}".format(self.atac_dim))
        print("# cells :{}".format(self.adata_r.shape[0]))
        self.idx = split_data(self.adata_r)
        self.l_prior_r, self.l_prior_a  = get_l_prior_r(self.adata_r), get_l_prior_a(self.adata_a)
        self.norm_mat_r = get_norm_mat_r(self.adata_r)
        self.norm_mat_a = get_norm_mat_a(self.adata_a)
        self.retain_genes_idx = None

        self.adata_r.obs["clusters"] = pd.read_json("./data/cell_clusters.json", typ="series").astype("category")


    def train_dataloader(self):
        train_set = VAEDataSet(self.adata_r[self.idx["train"], :], self.adata_a[self.idx["train"], :])
        return torch.utils.data.DataLoader(train_set, batch_size = self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        val_set = VAEDataSet(self.adata_r[self.idx["val"], :], self.adata_a[self.idx["val"], :])
        return torch.utils.data.DataLoader(val_set, batch_size = self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False, pin_memory=True)

    def test_dataloader(self):
        test_set = VAEDataSet(self.adata_r[self.idx["test"], :], self.adata_a[self.idx["test"], :])
        return torch.utils.data.DataLoader(test_set, batch_size = self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False, pin_memory=True)

    def all_dataloader(self):
        all_set = VAEDataSet(self.adata_r, self.adata_a)
        return torch.utils.data.DataLoader(all_set, batch_size = self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False, pin_memory=True)
    
class CustomDataModule_Pre(pl.LightningDataModule):
    def __init__(self, rna_dir, atac_dir,
                 batch_size : int =128, num_workers=2, 
                 min_counts_genes=10, min_counts_peaks=10,
                 n_top_genes=3000, n_top_peaks=20000, sub = False):
        super().__init__()
        self.save_hyperparameters()
        print("Loading data...")
        self.adata_r = sc.read_loom("./data/adata_rna.loom")
        self.adata_a = sc.read_loom("./data/adata_atac.loom")
        self.adata_r.obs_names = self.adata_r.obs.obs_names
        self.adata_r.var_names = self.adata_r.var.var_names
        self.adata_a.obs_names = self.adata_a.obs.obs_names
        self.adata_a.var_names = self.adata_a.var.var_names
        self.rna_dim, self.atac_dim = self.adata_r.shape[1], self.adata_a.shape[1]
        print("RNA dim : {}".format(self.rna_dim), "ATAC dim : {}".format(self.atac_dim))
        print("# cells :{}".format(self.adata_r.shape[0]))
        self.idx = split_data(self.adata_r)
        self.l_prior_r, self.l_prior_a  = get_l_prior_r(self.adata_r), get_l_prior_a(self.adata_a)
        self.norm_mat_r = get_norm_mat_r(self.adata_r)
        self.norm_mat_a = get_norm_mat_a(self.adata_a)
        self.retain_genes_idx = None


    def train_dataloader(self):
        train_set = VAEDataSet(self.adata_r[self.idx["train"], :], self.adata_a[self.idx["train"], :])
        return torch.utils.data.DataLoader(train_set, batch_size = self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        val_set = VAEDataSet(self.adata_r[self.idx["val"], :], self.adata_a[self.idx["val"], :])
        return torch.utils.data.DataLoader(val_set, batch_size = self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False, pin_memory=True)

    def test_dataloader(self):
        test_set = VAEDataSet(self.adata_r[self.idx["test"], :], self.adata_a[self.idx["test"], :])
        return torch.utils.data.DataLoader(test_set, batch_size = self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False, pin_memory=True)

    def all_dataloader(self):
        all_set = VAEDataSet(self.adata_r, self.adata_a)
        return torch.utils.data.DataLoader(all_set, batch_size = self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False, pin_memory=True)