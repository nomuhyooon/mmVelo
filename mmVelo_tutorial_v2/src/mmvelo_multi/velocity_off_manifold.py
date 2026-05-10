"""
velocity_off_manifold.py

Quantifying the off-manifold instability of latent dynamics in mmVelo.

Estimating local tangent spaces from the kNN neighbors of each cell using batched SVD,
and decomposing posterior d samples into on-manifold and off-manifold components
to evaluate the variance of each component.

Defined metrics (m ∈ {RNA, ATAC}):
  U_dyn^m          : dynamics fluctuation uncertainty (all components)
                     Var_s[ cos(v^(s), v_bar) ]
  U_dyn_par^m      : on-manifold dynamics fluctuation
                     Var_s[ cos(v_par^(s), v_par_bar) ]
  U_dyn_perp^m     : off-manifold instability
                     Var_s[ cos(v_perp^(s), v_perp_bar) ]
  R_n              : off-manifold energy ratio
                     mean_s[ ||d_perp^(s)||^2 / ||d^(s)||^2 ]
"""

import os
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as dist
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader


# ──────────────────────────────────────────────────────────────
# utlity functions
# ──────────────────────────────────────────────────────────────

def _cosine_sim_variance(samples: torch.Tensor) -> torch.Tensor:
    """
    samples : (S, N, dim)
    returns : (N,)  各 cellの velocity 方向分散 (Bessel 補正あり)
    """
    v_bar = samples.mean(0)                                        # (N, dim)
    cos   = (F.normalize(samples, dim=-1, eps=1e-8)
             * F.normalize(v_bar.unsqueeze(0), dim=-1, eps=1e-8)
             ).sum(-1)                                             # (S, N)
    return cos.var(dim=0, unbiased=True)                           # (N,)


def _get_z_hat(model, s, u, a) -> torch.Tensor:
    """posterior mean z_hat = (mu_r + mu_a) / 2"""
    qz_mu_r, qz_logvar_r = model.vaes[0].enc_z(s, u)
    qz_mu_a, qz_logvar_a = model.vaes[1].enc_z(a)
    return (qz_mu_r + qz_mu_a) / 2  # (N, z_dim)


def _local_tangent_projector(
    z_hat_batch: torch.Tensor,
    z_neigh_batch: torch.Tensor,
    manifold_dim: int,
) -> torch.Tensor:
    """
    compute the projection matrix onto the local tangent space 

    Parameters
    ----------
    z_hat_batch   : (n_cells, z_dim)    各細胞の posterior mean
    z_neigh_batch : (n_cells, k, z_dim) 近傍細胞の posterior mean
    manifold_dim  : r  局所接線空間の次元数 (top-r PCs)

    Returns
    -------
    P_n : (n_cells, z_dim, z_dim)  射影行列 P = U_n U_n^T
    """
    # 近傍を中心化 (N, k, z_dim)
    centered = z_neigh_batch - z_hat_batch.unsqueeze(1)

    # batched SVD
    # Vh: (n_cells, min(k,z_dim), z_dim)  各行が右特異ベクトル
    _, _, Vh = torch.linalg.svd(centered, full_matrices=False)

    r = min(manifold_dim, Vh.shape[1])
    U_n = Vh[:, :r, :].transpose(-1, -2)         # (n_cells, z_dim, r)
    P_n = U_n @ U_n.transpose(-1, -2)             # (n_cells, z_dim, z_dim)
    return P_n


# ──────────────────────────────────────────────────────────────
# フェーズ 1
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def _collect_z_hat(model, dataloader: DataLoader, device: str) -> np.ndarray:
    """compute the posterior mean z_hat for all cells """
    model.eval()
    z_hat_list = []
    for batch in dataloader:
        s, u, a = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        z_hat_list.append(_get_z_hat(model, s, u, a).cpu().numpy())
    return np.concatenate(z_hat_list, axis=0)  # (N_total, z_dim)


# ──────────────────────────────────────────────────────────────
# フェーズ 2: on/off-manifold decomposition
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def _batch_off_manifold(
    model,
    batch,
    z_neigh_batch: torch.Tensor,
    n_samples: int,
    manifold_dim: int,
    device: str,
) -> dict:
    """
    compute the on/off-manifold uncertainty decomposition for 1 batch.

    Returns
    -------
    dict:
      z_hat              (n_cells, z_dim)
      vel_mean_rna       (n_cells, rna_dim)   RNA velocity 平均 (全成分)
      vel_mean_atac      (n_cells, atac_dim)  ATAC velocity 平均 (全成分)
      u_dyn_rna          (n_cells,)  dynamics fluctuation uncertainty
      u_dyn_atac         (n_cells,)
      u_dyn_par_rna      (n_cells,)  on-manifold dynamics fluctuation
      u_dyn_par_atac     (n_cells,)
      u_dyn_perp_rna     (n_cells,)  off-manifold instability
      u_dyn_perp_atac    (n_cells,)
      off_manifold_ratio (n_cells,)  off-manifold エネルギー比 R_n
    """
    s, u, a = batch[0].to(device), batch[1].to(device), batch[2].to(device)
    n_cells = s.shape[0]
    z_hat   = _get_z_hat(model, s, u, a)           # (n_cells, z_dim)
    z_dim   = z_hat.shape[1]

    # 局所接線空間の射影行列  P_n, (I - P_n)
    P_n = _local_tangent_projector(z_hat, z_neigh_batch, manifold_dim)  # (n_cells, z_dim, z_dim)
    I_n = torch.eye(z_dim, device=device).unsqueeze(0).expand(n_cells, -1, -1)

    # z_hat での base decode を一度だけ計算
    s_base = model.vaes[0].dec_su(z_hat)[0][0] * model.norm_mat_s  # (n_cells, rna_dim)
    a_base = model.vaes[1].dec_ald(z_hat)[0]   * model.norm_mat_a  # (n_cells, atac_dim)

    # q(d | z_hat)
    qd = model.vaes[0].enc_dyn(z_hat)

    # list
    vel_rna_full_list  = []
    vel_atac_full_list = []
    vel_rna_par_list   = []
    vel_atac_par_list  = []
    vel_rna_perp_list  = []
    vel_atac_perp_list = []
    off_ratio_list     = []

    for _ in range(n_samples):
        d_s = qd.rsample()  # (n_cells, z_dim)

        # on-manifold / off-manifold 成分
        d_col  = d_s.unsqueeze(-1)                        # (n_cells, z_dim, 1)
        d_par  = (P_n        @ d_col).squeeze(-1)         # (n_cells, z_dim)
        d_perp = ((I_n - P_n) @ d_col).squeeze(-1)       # (n_cells, z_dim)

        # off-manifold energy ratio
        norm_sq_full = (d_s    ** 2).sum(-1).clamp(min=1e-8)  # (n_cells,)
        norm_sq_perp = (d_perp ** 2).sum(-1)
        off_ratio_list.append(norm_sq_perp / norm_sq_full)

        # velocity = f(z_hat + rho*d) * norm - f(z_hat) * norm
        def _vel(d):
            z_dt = z_hat + model.d_coeff * d
            s_dt = model.vaes[0].dec_su(z_dt)[0][0] * model.norm_mat_s
            a_dt = model.vaes[1].dec_ald(z_dt)[0]   * model.norm_mat_a
            return s_dt - s_base, a_dt - a_base

        v_rna,      v_atac      = _vel(d_s)
        v_rna_par,  v_atac_par  = _vel(d_par)
        v_rna_perp, v_atac_perp = _vel(d_perp)

        vel_rna_full_list.append(v_rna)
        vel_atac_full_list.append(v_atac)
        vel_rna_par_list.append(v_rna_par)
        vel_atac_par_list.append(v_atac_par)
        vel_rna_perp_list.append(v_rna_perp)
        vel_atac_perp_list.append(v_atac_perp)

    def _var(lst):
        return _cosine_sim_variance(torch.stack(lst, dim=0)).cpu().numpy()

    rna_full_stack  = torch.stack(vel_rna_full_list,  dim=0)
    atac_full_stack = torch.stack(vel_atac_full_list, dim=0)

    return {
        "z_hat":              z_hat.cpu().numpy(),
        "vel_mean_rna":       rna_full_stack.mean(0).cpu().numpy(),
        "vel_mean_atac":      atac_full_stack.mean(0).cpu().numpy(),
        "u_dyn_rna":          _cosine_sim_variance(rna_full_stack).cpu().numpy(),
        "u_dyn_atac":         _cosine_sim_variance(atac_full_stack).cpu().numpy(),
        "u_dyn_par_rna":      _var(vel_rna_par_list),
        "u_dyn_par_atac":     _var(vel_atac_par_list),
        "u_dyn_perp_rna":     _var(vel_rna_perp_list),
        "u_dyn_perp_atac":    _var(vel_atac_perp_list),
        "off_manifold_ratio": torch.stack(off_ratio_list, dim=0).mean(0).cpu().numpy(),
    }


# ──────────────────────────────────────────────────────────────
# main function
# ──────────────────────────────────────────────────────────────

def compute_off_manifold_uncertainty(
    model,
    dataloader: DataLoader,
    n_samples: int = 200,
    n_neighbors: int = 30,
    manifold_dim: int = 5,
    device: str = "cuda",
) -> dict:
    """
    compute the on/off-manifold uncertainty decomposition for all cells.

    Phase 1: Collect z_hat for all cells and build a kNN graph.
    Phase 2: For each batch, estimate the local tangent space using batched SVD,
              decompose d samples into on/off-manifold components, and evaluate velocity variance.

    Parameters
    ----------
    model        : trained model (DREG_DYN)
    dataloader   : shuffling disabled DataLoader for all cells
    n_samples    : number of samples to draw from d (S)
    n_neighbors  : number of kNN neighbors for local tangent space estimation (k)
    manifold_dim : dimension of the local tangent space (r, top-r PCs)
    device       : torch device string

    Returns
    -------
    dict:
      z_hat              (N, z_dim)
      vel_mean_rna       (N, rna_dim)
      vel_mean_atac      (N, atac_dim)
      u_dyn_rna          (N,)  dynamics fluctuation uncertainty (全成分)
      u_dyn_atac         (N,)
      u_dyn_par_rna      (N,)  on-manifold dynamics fluctuation
      u_dyn_par_atac     (N,)
      u_dyn_perp_rna     (N,)  off-manifold instability
      u_dyn_perp_atac    (N,)
      off_manifold_ratio (N,)  off-manifold energy ratio R_n
    """
    model.to(device)
    model.eval()

    # ── フェーズ 1: z_hat & kNN ─────────────────────────
    print("[Phase 1] Collecting z_hat for all cells...")
    all_z_hat = _collect_z_hat(model, dataloader, device)  # (N, z_dim)
    N_total   = len(all_z_hat)

    k = min(n_neighbors, N_total - 1)
    print(f"[Phase 1] Building {k}-NN graph (N={N_total}, z_dim={all_z_hat.shape[1]})...")
    nn_model = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
    nn_model.fit(all_z_hat)
    nn_idx = nn_model.kneighbors(all_z_hat, return_distance=False)  # (N, k)

    z_hat_tensor  = torch.tensor(all_z_hat, dtype=torch.float32, device=device)
    nn_idx_tensor = torch.tensor(nn_idx,    dtype=torch.long,    device=device)
    z_neighbors   = z_hat_tensor[nn_idx_tensor]  # (N, k, z_dim)

    # ── フェーズ 2: バッチごとに on/off-manifold 分解 ─────────
    print(f"[Phase 2] Computing on/off-manifold decomposition "
          f"(n_samples={n_samples}, manifold_dim={manifold_dim})...")
    accum       = defaultdict(list)
    cell_offset = 0

    for batch in dataloader:
        n_cells       = batch[0].shape[0]
        batch_slice   = slice(cell_offset, cell_offset + n_cells)
        z_neigh_batch = z_neighbors[batch_slice]  # (n_cells, k, z_dim)

        out = _batch_off_manifold(
            model, batch, z_neigh_batch,
            n_samples    = n_samples,
            manifold_dim = manifold_dim,
            device       = device,
        )
        for key, val in out.items():
            accum[key].append(val)
        cell_offset += n_cells

    return {key: np.concatenate(vals, axis=0) for key, vals in accum.items()}


# ──────────────────────────────────────────────────────────────
# save
# ──────────────────────────────────────────────────────────────

def save_off_manifold_uncertainty(
    dm,
    results: dict,
    save_dir: str,
    filename_rna:  str = "adata_rna_offmanifold.loom",
    filename_atac: str = "adata_atac_offmanifold.loom",
) -> None:
    """
    save the off-manifold decomposition results to AnnData and save as .loom files.

    RNA AnnData:
      layers["vel_mean"]        : mean RNA velocity       (N, rna_dim)
      obsm["z_hat"]             : posterior mean z        (N, z_dim)
      obs["u_dyn"]              : dynamics fluctuation uncertainty
      obs["u_dyn_par"]          : on-manifold dynamics fluctuation
      obs["u_dyn_perp"]         : off-manifold instability
      obs["off_manifold_ratio"] : off-manifold energy ratio R_n

    ATAC AnnData:
      layers["vel_mean"]        : ATAC velocity 平均      (N, atac_dim)
      obs["u_dyn"]              : dynamics fluctuation uncertainty
      obs["u_dyn_par"]          : on-manifold dynamics fluctuation
      obs["u_dyn_perp"]         : off-manifold instability
    """
    os.makedirs(save_dir, exist_ok=True)

    dm.adata_r.layers["vel_mean"]        = results["vel_mean_rna"]
    dm.adata_r.obsm["z_hat"]             = results["z_hat"]
    dm.adata_r.obs["u_dyn"]              = results["u_dyn_rna"]
    dm.adata_r.obs["u_dyn_par"]          = results["u_dyn_par_rna"]
    dm.adata_r.obs["u_dyn_perp"]         = results["u_dyn_perp_rna"]
    dm.adata_r.obs["off_manifold_ratio"] = results["off_manifold_ratio"]

    dm.adata_a.layers["vel_mean"]        = results["vel_mean_atac"]
    dm.adata_a.obs["u_dyn"]              = results["u_dyn_atac"]
    dm.adata_a.obs["u_dyn_par"]          = results["u_dyn_par_atac"]
    dm.adata_a.obs["u_dyn_perp"]         = results["u_dyn_perp_atac"]

    rna_path  = os.path.join(save_dir, filename_rna)
    atac_path = os.path.join(save_dir, filename_atac)
    dm.adata_r.write_loom(rna_path,  write_obsm_varm=True)
    dm.adata_a.write_loom(atac_path, write_obsm_varm=True)

    print(f"Saved RNA  AnnData → {rna_path}")
    print(f"Saved ATAC AnnData → {atac_path}")
