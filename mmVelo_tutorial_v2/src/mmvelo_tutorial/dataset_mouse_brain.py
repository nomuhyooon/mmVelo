"""
dataset_mouse_brain.py
----------------------
DataModule and helper utilities for the mouse brain 10x Multiome tutorial.

The original mmvelo_multi DataModule (MultiomeBrainDataModule_Pre) reads from
hardcoded absolute paths.  This module provides a self-contained alternative
that reads from relative paths inside the tutorial repository.
"""

import os
import numpy as np
import scipy.sparse
import torch
import pytorch_lightning as pl
import scanpy as sc
import scvelo as scv


# ── Helpers ───────────────────────────────────────────────────────────────────

def split_data(adata, val_ratio=0.1, test_ratio=0.1, seed=43):
    """Randomly split cells into train / val / test sets."""
    rng = np.random.RandomState(seed)
    n = adata.shape[0]
    idx = rng.permutation(n)
    n_val  = int(n * val_ratio)
    n_test = int(n * test_ratio)
    return dict(
        val=idx[:n_val],
        test=idx[n_val:n_val + n_test],
        train=idx[n_val + n_test:],
    )


def get_l_prior_r(adata_r):
    s = adata_r.layers["spliced"]
    u = adata_r.layers["unspliced"]
    total = np.asarray(s.sum(axis=1) + u.sum(axis=1)).reshape(-1)
    mean, std = float(np.mean(np.log(total))), float(np.std(np.log(total)))
    return torch.distributions.LogNormal(mean, std)


def get_l_prior_a(adata_a):
    total = np.asarray(adata_a.X.sum(axis=1)).reshape(-1)
    mean, std = float(np.mean(np.log(total))), float(np.std(np.log(total)))
    return torch.distributions.LogNormal(mean, std)


def get_norm_mat_r(adata_r):
    s = adata_r.layers["spliced"]
    u = adata_r.layers["unspliced"]
    norm_s = np.asarray(s.sum(0) / (s > 0).sum(0)).reshape(-1)
    norm_u = np.asarray(u.sum(0) / (u > 0).sum(0)).reshape(-1)
    return norm_s, norm_u


def get_norm_mat_a(adata_a):
    a = adata_a.X
    return np.asarray(a.sum(0) / (a > 0).sum(0)).reshape(-1)


# ── Dataset ───────────────────────────────────────────────────────────────────

class VAEDataSet(torch.utils.data.Dataset):
    def __init__(self, adata_r, adata_a):
        def _dense(x):
            return x.toarray().astype(np.float32) if scipy.sparse.issparse(x) else np.asarray(x, np.float32)
        self.s = _dense(adata_r.layers["spliced"])
        self.u = _dense(adata_r.layers["unspliced"])
        self.a = _dense(adata_a.X)

    def __len__(self):
        return self.s.shape[0]

    def __getitem__(self, idx):
        return (
            torch.tensor(self.s[idx]),
            torch.tensor(self.u[idx]),
            torch.tensor(self.a[idx]),
        )


# ── DataModule ────────────────────────────────────────────────────────────────

class TutorialBrainDataModule(pl.LightningDataModule):
    """
    DataModule for the mouse brain 10x Multiome tutorial dataset.

    Parameters
    ----------
    data_dir : str
        Directory containing ``adata_rna.loom`` and ``adata_atac.loom``.
    batch_size : int
    num_workers : int
    seed : int
    """

    def __init__(self, data_dir="data/mouse_brain",
                 batch_size=128, num_workers=2, seed=43):
        super().__init__()
        self.data_dir    = data_dir
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.seed        = seed

        print("Loading RNA data...")
        self.adata_r = sc.read_loom(
            os.path.join(data_dir, "adata_rna.loom"),
            obs_names="obs_names", var_names="var_names",
        )
        print("Loading ATAC data...")
        self.adata_a = sc.read_loom(
            os.path.join(data_dir, "adata_atac.loom"),
            obs_names="obs_names", var_names="var_names",
        )

        self.rna_dim  = self.adata_r.shape[1]
        self.atac_dim = self.adata_a.shape[1]
        print(f"RNA dim  : {self.rna_dim}")
        print(f"ATAC dim : {self.atac_dim}")
        print(f"# cells  : {self.adata_r.shape[0]}")

        self.idx            = split_data(self.adata_r, seed=seed)
        self.l_prior_r      = get_l_prior_r(self.adata_r)
        self.l_prior_a      = get_l_prior_a(self.adata_a)
        self.norm_mat_r     = get_norm_mat_r(self.adata_r)
        self.norm_mat_a     = get_norm_mat_a(self.adata_a)
        self.retain_genes_idx = None

        print(f"train: {len(self.idx['train'])}  val: {len(self.idx['val'])}  "
              f"test: {len(self.idx['test'])}")

    def _make_loader(self, adata_r, adata_a, shuffle, drop_last=False):
        ds = VAEDataSet(adata_r, adata_a)
        return torch.utils.data.DataLoader(
            ds, batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle, drop_last=drop_last, pin_memory=True,
        )

    def train_dataloader(self):
        return self._make_loader(
            self.adata_r[self.idx["train"]],
            self.adata_a[self.idx["train"]],
            shuffle=True, drop_last=True,
        )

    def val_dataloader(self):
        return self._make_loader(
            self.adata_r[self.idx["val"]],
            self.adata_a[self.idx["val"]],
            shuffle=False,
        )

    def test_dataloader(self):
        return self._make_loader(
            self.adata_r[self.idx["test"]],
            self.adata_a[self.idx["test"]],
            shuffle=False,
        )

    def all_dataloader(self):
        return self._make_loader(self.adata_r, self.adata_a, shuffle=False)
