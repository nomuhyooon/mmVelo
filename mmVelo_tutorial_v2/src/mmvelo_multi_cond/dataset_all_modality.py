import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
import torch
import pytorch_lightning as pl
from scipy.sparse import csr_matrix
from scvelo.preprocessing.moments import get_moments
import copy

def load_greenleaf_missingmodal_data():
    adata_rna = sc.read_h5ad("/mmVelo_tutorial_/data/human_brain/joint_rna_adata.h5ad")
    adata_atac = sc.read_h5ad("/mmVelo_tutorial_/data/human_brain/joint_atac_adata.h5ad")
    return adata_rna, adata_atac

def one_hot(adata, onehot_names="clusters"):
    print("setting one hot...")
    one_hot = pd.get_dummies(adata.obs[onehot_names]).to_numpy(dtype="float32")
    adata.obsm["one_hot"] = one_hot

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
    print("log-mean, log-std; ",mean, std)
    return torch.distributions.log_normal.LogNormal(mean, std)

def get_l_priors_r_cond(adata_r):
    l_priors_list = []
    for i in range(adata_r.obsm["one_hot"].shape[1]):
        adata_cond_i = adata_r[adata_r.obsm["one_hot"][:, i] == 1, :]
        total_count = np.sum((adata_cond_i.layers["spliced"] + adata_cond_i.layers["unspliced"]).toarray(), axis=1)
        mean = np.mean(np.log(total_count))
        std = np.std(np.log(total_count))
        print("RNA log-mean, log-std in batch {}; ".format(i), mean, std)
        l_priors_list.append([mean, std])
    return l_priors_list

def get_norm_mat_r(adata_r, moment=False):
    s, u = adata_r.layers["spliced"], adata_r.layers["unspliced"]
    if moment:
        s, u = adata_r.layers["Ms"], adata_r.layers["Mu"]
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
    print("log-mean, log-std; ",mean, std)
    return torch.distributions.log_normal.LogNormal(mean, std)

def get_l_priors_a_cond(adata_a):
    l_priors_list = []
    for i in range(adata_a.obsm["one_hot"].shape[1]):
        adata_cond_i = adata_a[adata_a.obsm["one_hot"][:, i] == 1, :]
        total_count = np.sum(adata_cond_i.X.toarray(), axis=1)
        mean = np.mean(np.log(total_count))
        std = np.std(np.log(total_count))
        print("ATAC log-mean, log-std in batch {}; ".format(i), mean, std)
        l_priors_list.append([mean, std])
    return l_priors_list


def one_hot_modality(adata):
    print("setting one hot for modality...")
    modality_array = np.zeros([adata.n_obs, 2], dtype="float32")
    modality_array[np.where(adata.obs["modality"] == "multiome")[0], :] = [1., 1.] # RNA, ATAC
    modality_array[np.where(adata.obs["modality"] == "rna")[0], :] = [1., 0.] # RNA, 0
    modality_array[np.where(adata.obs["modality"] == "atac")[0], :] = [0., 1.] # 0, ATAC
    adata.obsm["one_hot_modality"] = modality_array
    
    adata.obs["modality_id"] = adata.obs["modality"]


def one_hot_modality_batch(adata):
    print("setting one hot for modality and batch...")
    modalities = ["multiome", "rna", "atac"]
    sample_ids = ['hft_ctx_w21_dc1r3_r1', 'hft_ctx_w21_dc2r2_r1', 'hft_ctx_w21_dc2r2_r2']
    
    adata.obs["modality_Sample.ID"] = np.empty([adata.n_obs], "<U30")
    for m in modalities:
        for sample_id in sample_ids:
            adata_m = adata[adata.obs["modality_id"] == m, :]
            adata_m_s = adata_m[adata_m.obs["Sample.ID"] == sample_id, :]
            adata.obs["modality_Sample.ID"][adata_m_s.obs_names] = m + "_" + sample_id
        
    modality_sample_onehot = pd.get_dummies(adata.obs["modality_Sample.ID"]).to_numpy()
    adata.obsm["one_hot"] = modality_sample_onehot.astype(np.float32)
    
def one_hot_for_modality_adv_loss(adata, onehot_names="modality"):
    print("setting one hot...")
    one_hot = pd.get_dummies(adata.obs[onehot_names]).to_numpy(dtype="float32")
    adata.obsm["one_hot_mod_adv"] = one_hot
    

class CVAEDataSet_MissingModality(torch.utils.data.Dataset):
    def __init__(self, adata_r, adata_a, moment=False, only_multi=False):
        self.moment = moment
        self.only_multi = only_multi
        if only_multi:
            adata_r = adata_r[adata_r.obs["modality"] == "multiome", :]
            adata_a = adata_a[adata_a.obs["modality"] == "multiome", :]
            
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
        self.one_hot = adata_r.obsm["one_hot"]
        self.one_hot_modality = adata_r.obsm["one_hot_modality"]
        self.one_hot_mod_adv = adata_r.obsm["one_hot_mod_adv"]
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
        one_hot = torch.tensor(self.one_hot[idx])
        one_hot_modality = torch.tensor(self.one_hot_modality[idx])
        one_hot_mod_adv = torch.tensor(self.one_hot_mod_adv[idx])
        if self.moment:
            ms = torch.tensor(self.ms[idx,:])
            mu = torch.tensor(self.mu[idx,:])
            ma = torch.tensor(self.ma[idx,:])
            return s, u, a, one_hot, one_hot_modality, one_hot_mod_adv, ms, mu, ma
        return s, u, a, one_hot, one_hot_modality, one_hot_mod_adv



class MultiomeHumanBrainDataModule_MissingModalityPrediction(pl.LightningDataModule):
    def __init__(self, batch_size : int =128, num_workers=2, 
                 min_counts_genes=20, min_counts_peaks=10, 
                 n_top_genes=3000, n_top_peaks=20000,
                 batch_sub=True, filter_outliers=True,
                 use_onehot_modality_sample=True,
                 pretrain_multi = True):
        super().__init__()
        self.save_hyperparameters()
        self.adata_r, self.adata_a = load_greenleaf_missingmodal_data()

        scv.pp.filter_genes(self.adata_r, min_counts=min_counts_genes, min_counts_u=min_counts_genes)
        #scv.pp.filter_genes_dispersion(self.adata_r, n_top_genes=n_top_genes)
        sc.pp.filter_genes(self.adata_a, min_counts=min_counts_peaks)
        
        self.rna_dim, self.atac_dim = self.adata_r.shape[1], self.adata_a.shape[1]
        
        # separate train/test/validation
        self.idx = split_data(self.adata_r)
        
        # set rna / atac annotation if necessary
         
        # set batch / modality annotation
        one_hot(self.adata_r, onehot_names = "Condtioning_ID")
        one_hot_modality(self.adata_r)
        one_hot_for_modality_adv_loss(self.adata_r)
                    
        self.adata_a.obsm["one_hot"] = self.adata_r.obsm["one_hot"]
        self.adata_a.obsm["one_hot_modality"] = self.adata_r.obsm["one_hot_modality"]
        self.adata_a.obsm["one_hot_mod_adv"] = self.adata_r.obsm["one_hot_mod_adv"]
        
        self.l_prior_r, self.l_prior_a  = get_l_priors_r_cond(self.adata_r), get_l_priors_a_cond(self.adata_a)
        print("setting the mean and var in non-observed modality to 5., and 5. to avoid ValueError")
        atac_batch = [3, 5, 7, 8]
        for batch in atac_batch:
            self.l_prior_r[batch] = [5., 5.]
        rna_batch = [4, 6]
        for batch in rna_batch:
            self.l_prior_a[batch] = [5., 5.]
        
        
        self.norm_mat_r = get_norm_mat_r(self.adata_r) # maybe norm_mat sholud be batch-dependent...
        self.norm_mat_a = get_norm_mat_a(self.adata_a)
        self.num_cat = self.adata_r.obsm["one_hot"].shape[1]
        self.retain_genes_idx = None
        print("DataModule setting Done.")
        print("# genes: {}".format(self.adata_r.shape[1]))
        print("# peaks: {}".format(self.adata_a.shape[1]))
        print("# cells: {}".format(self.adata_r.shape[0]))
        print("# conds: {}".format(self.num_cat))
        
        if pretrain_multi:
            self.only_multi = True
        else:
            self.only_multi = False
            
    def pretrain_multi_end(self):
        self.only_multi = False

    def train_dataloader(self):
        train_set = CVAEDataSet_MissingModality(self.adata_r[self.idx["train"], :], self.adata_a[self.idx["train"], :],
                                                only_multi=self.only_multi)
        return torch.utils.data.DataLoader(train_set, batch_size = self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        val_set = CVAEDataSet_MissingModality(self.adata_r[self.idx["val"], :], self.adata_a[self.idx["val"], :],
                                              only_multi=self.only_multi)
        return torch.utils.data.DataLoader(val_set, batch_size = self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False, pin_memory=True)

    def test_dataloader(self):
        test_set = CVAEDataSet_MissingModality(self.adata_r[self.idx["test"], :], self.adata_a[self.idx["test"], :],
                                               only_multi=self.only_multi)
        return torch.utils.data.DataLoader(test_set, batch_size = self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False, pin_memory=True)

    def all_dataloader(self):
        all_set = CVAEDataSet_MissingModality(self.adata_r, self.adata_a, only_multi=self.only_multi)
        return torch.utils.data.DataLoader(all_set, batch_size = self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False, pin_memory=True)


class DynDataModule_Smooth_MissingModalityPrediction(pl.LightningDataModule):
    def __init__(self, dm_pre, batch_size : int =128, num_workers=2, n_neighbors=50, 
                 modality_wise_smoothing = True):
        super().__init__()
        # dm_pre contains AnnData / LogNormal objects that cannot be pickled;
        # excluding it prevents a ValueError during checkpoint serialisation.
        self.save_hyperparameters(ignore=["dm_pre"])
        self.adata_r = dm_pre.adata_r.copy()
        self.adata_a = dm_pre.adata_a.copy()
        del (self.adata_r.layers["rec_s"], self.adata_r.layers["rec_u"], 
            self.adata_r.layers["s_raw"], self.adata_r.layers["u_raw"],
            self.adata_r.obsm["lr"])
        self.rna_dim, self.atac_dim = self.adata_r.shape[1], self.adata_a.shape[1]
        print("RNA dim : {}".format(self.rna_dim), "ATAC dim : {}".format(self.atac_dim))
        self.idx = dm_pre.idx
        
        self.n_neighbors = n_neighbors        
        self.normalize_counts() ##
        self.norm_mat_r = get_norm_mat_r(self.adata_r)
        self.norm_mat_a = get_norm_mat_a(self.adata_a)
        self.calc_neighbors(n_neighbors = self.n_neighbors)
        
        if modality_wise_smoothing:
            # separate adata based on measured modality
            self.sc_rna_idx = np.where((dm_pre.adata_r.obs["modality"] == "rna").to_numpy())[0]
            self.sn_rna_idx = np.where((dm_pre.adata_r.obs["modality"] == "multiome").to_numpy())[0]
            self.sc_atac_idx = np.where((dm_pre.adata_r.obs["modality"] == "atac").to_numpy())[0]
            self.sn_atac_idx = np.where((dm_pre.adata_r.obs["modality"] == "multiome").to_numpy())[0]
            
            adata_sc_r_moment = self.adata_r[self.sc_rna_idx, :].copy()
            adata_sn_r_moment = self.adata_r[self.sn_rna_idx, :].copy()
            adata_sc_a_moment = self.adata_a[self.sc_atac_idx, :].copy()
            adata_sn_a_moment = self.adata_a[self.sn_atac_idx, :].copy()
            
            sc.pp.neighbors(adata_sc_r_moment, n_neighbors=n_neighbors, use_rep="latent")
            sc.pp.neighbors(adata_sn_r_moment, n_neighbors=n_neighbors, use_rep="latent")
            sc.pp.neighbors(adata_sc_a_moment, n_neighbors=n_neighbors, use_rep="latent")
            sc.pp.neighbors(adata_sn_a_moment, n_neighbors=n_neighbors, use_rep="latent")
            self.calc_moments(adata_sc_r_moment, adata_sc_a_moment), self.calc_moments(adata_sn_r_moment, adata_sn_a_moment)
            
            self.adata_r.layers["Ms"], self.adata_r.layers["Mu"] = np.zeros_like(self.adata_r.X.toarray()), np.zeros_like(self.adata_r.X.toarray())
            self.adata_a.layers["Ma"] = np.zeros_like(self.adata_a.X.toarray())
            
            self.adata_r.layers["Ms"][self.sc_rna_idx, :] = adata_sc_r_moment.layers["Ms"].copy()
            self.adata_r.layers["Mu"][self.sc_rna_idx, :] = adata_sc_r_moment.layers["Mu"].copy()
            self.adata_r.layers["Ms"][self.sn_rna_idx, :] = adata_sn_r_moment.layers["Ms"].copy()
            self.adata_r.layers["Mu"][self.sn_rna_idx, :] = adata_sn_r_moment.layers["Mu"].copy()
            self.adata_a.layers["Ma"][self.sc_atac_idx, :] = adata_sc_a_moment.layers["Ma"].copy()
            self.adata_a.layers["Ma"][self.sn_atac_idx, :] = adata_sn_a_moment.layers["Ma"].copy()
            del adata_sc_r_moment, adata_sn_r_moment, adata_sc_a_moment, adata_sn_a_moment
            
        else:
            # separate adata based on measured modality
            self.rna_idx = np.concatenate([np.where((dm_pre.adata_r.obs["modality"] == "multiome").to_numpy())[0],
                                        np.where((dm_pre.adata_r.obs["modality"] == "rna").to_numpy())[0]])
            self.atac_idx = np.concatenate([np.where((dm_pre.adata_r.obs["modality"] == "multiome").to_numpy())[0], 
                                            np.where((dm_pre.adata_r.obs["modality"] == "atac").to_numpy())[0]])
            adata_r_moment = self.adata_r[self.rna_idx, :].copy()
            adata_a_moment = self.adata_a[self.atac_idx, :].copy()
            sc.pp.neighbors(adata_r_moment, n_neighbors=n_neighbors, use_rep="latent")
            sc.pp.neighbors(adata_a_moment, n_neighbors=n_neighbors, use_rep="latent")
            self.calc_moments(adata_r_moment, adata_a_moment)
            
            self.adata_r.layers["Ms"], self.adata_r.layers["Mu"] = np.zeros_like(self.adata_r.X.toarray()), np.zeros_like(self.adata_r.X.toarray())
            self.adata_a.layers["Ma"] = np.zeros_like(self.adata_a.X.toarray())
            
            self.adata_r.layers["Ms"][self.rna_idx, :] = adata_r_moment.layers["Ms"].copy()
            self.adata_r.layers["Mu"][self.rna_idx, :] = adata_r_moment.layers["Mu"].copy()
            self.adata_a.layers["Ma"][self.atac_idx, :] = adata_a_moment.layers["Ma"].copy()
        
        #[adata_r_moment.obs_names, :].layers["Ms"] = ms_copy
        #self.adata_r[adata_r_moment.obs_names, :].layers["Mu"] = mu_copy
        #self.adata_a[adata_a_moment.obs_names, :].layers["Ma"] = ma_copy
        
        
        self.adata_r.obsm["one_hot"], self.adata_r.obsm["one_hot_modality"]  = dm_pre.adata_r.obsm["one_hot"], dm_pre.adata_r.obsm["one_hot_modality"]
        self.adata_a.obsm["one_hot"], self.adata_a.obsm["one_hot_modality"] = dm_pre.adata_a.obsm["one_hot"], dm_pre.adata_a.obsm["one_hot_modality"]
        self.adata_r.obsm["one_hot_mod_adv"] =  dm_pre.adata_r.obsm["one_hot_mod_adv"]
        self.adata_a.obsm["one_hot_mod_adv"] =  dm_pre.adata_a.obsm["one_hot_mod_adv"]
        self.num_cat = self.adata_r.obsm["one_hot"].shape[1]
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
        

    def calc_moments(self, adata_r_moment, adata_a_moment):
        scv.pp.moments(adata_r_moment, n_neighbors=30, use_rep="latent")
        adata_a_moment.layers["Ma"] = get_moments(adata_a_moment)


    def train_dataloader(self):
        train_set = CVAEDataSet_MissingModality(self.adata_r[self.idx["train"], :], self.adata_a[self.idx["train"], :], moment=True)
        return torch.utils.data.DataLoader(train_set, batch_size = self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        val_set = CVAEDataSet_MissingModality(self.adata_r[self.idx["val"], :], self.adata_a[self.idx["val"], :], moment=True)
        return torch.utils.data.DataLoader(val_set, batch_size = self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False, pin_memory=True)

    def test_dataloader(self):
        test_set = CVAEDataSet_MissingModality(self.adata_r[self.idx["test"], :], self.adata_a[self.idx["test"], :], moment=True)
        return torch.utils.data.DataLoader(test_set, batch_size = self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False, pin_memory=True)

    def all_dataloader(self):
        all_set = CVAEDataSet_MissingModality(self.adata_r, self.adata_a, moment=True)
        return torch.utils.data.DataLoader(all_set, batch_size = self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=False, pin_memory=True)