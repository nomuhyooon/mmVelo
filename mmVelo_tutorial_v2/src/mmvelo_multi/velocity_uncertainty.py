"""
velocity_uncertainty.py

Quantifying the uncertainty of RNA velocity / chromatin velocity in mmVelo by decomposing it into three layers.

3-layer decomposition:
  1. State-estimation uncertainty     : sample z from q(z|a,s,u) (d fixed at posterior mean)
                                        → reliability of local cell-state estimation
  2. Dynamics fluctuation uncertainty : sample d from q(d|z_hat) (z fixed at posterior mean)
                                        → locality of local dynamics (biological variability near bifurcation points)
  3. Transition ambiguity             : sample both z and d, and evaluate the directional variance of kNN transition weights
                                        → ambiguity in downstream state transitions
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
# utility functions
# ──────────────────────────────────────────────────────────────

def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity for each element.
    a, b : (..., dim)
    returns : (...,)
    """
    a_n = F.normalize(a, dim=-1, eps=1e-8)
    b_n = F.normalize(b, dim=-1, eps=1e-8)
    return (a_n * b_n).sum(-1)


def _cosine_sim_variance(samples: torch.Tensor) -> torch.Tensor:
    """
    Compute the variance of cosine similarities for each element.
    Quantifying the variance of velocity directions for each cell.

    Steps:
      1. Compute the mean velocity v_bar = samples.mean(0)     (N, dim)
      2. Compute cos(v^(s), v_bar) for all samples      (S, N)
      3. Return the variance (Bessel correction)           (N,)
    """
    v_bar = samples.mean(0)                              # (N, dim)
    cos   = _cosine_sim(samples, v_bar.unsqueeze(0))     # (S, N)
    return cos.var(dim=0, unbiased=True)                  # (N,)


def _get_z_moe_dist(model, s, u, a):
    """
    Return the distribution and posterior mean of q(z_n) from the RNA and ATAC encoders.

    Returns
    -------
    qz_r  : dist.Normal  (RNA posterior)
    qz_a  : dist.Normal  (ATAC posterior)
    z_hat : Tensor (N, z_dim)  posterior mean z_hat = (mu_r + mu_a) / 2
    """
    qz_mu_r, qz_logvar_r = model.vaes[0].enc_z(s, u)
    qz_r = dist.Normal(qz_mu_r, F.softplus(qz_logvar_r))

    qz_mu_a, qz_logvar_a = model.vaes[1].enc_z(a)
    qz_a = dist.Normal(qz_mu_a, F.softplus(qz_logvar_a))

    z_hat = (qz_r.mean + qz_a.mean) / 2  # (N, z_dim)
    return qz_r, qz_a, z_hat


def _velocity(model, z: torch.Tensor, d: torch.Tensor):
    """
    Compute RNA/ATAC velocity from latent state z and dynamics d.

    v_RNA  = f^s(z + rho*d) * norm_s - f^s(z) * norm_s
    v_ATAC = f^a(z + rho*d) * norm_a - f^a(z) * norm_a

    Parameters
    ----------
    z : (N, z_dim)
    d : (N, z_dim)

    Returns
    -------
    v_rna  : (N, rna_dim)
    v_atac : (N, atac_dim)
    """
    z_dt = z + model.d_coeff * d

    s_base = model.vaes[0].dec_su(z)[0][0]    * model.norm_mat_s
    s_dt   = model.vaes[0].dec_su(z_dt)[0][0] * model.norm_mat_s
    a_base = model.vaes[1].dec_ald(z)[0]       * model.norm_mat_a
    a_dt   = model.vaes[1].dec_ald(z_dt)[0]    * model.norm_mat_a

    return s_dt - s_base, a_dt - a_base  # (N, rna_dim), (N, atac_dim)


# ──────────────────────────────────────────────────────────────
# 1. State-estimation / Dynamics fluctuation uncertainty
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def _batch_state_dynamics_uncertainty(
    model,
    batch,
    n_samples: int,
    device: str,
) -> dict:
    """
    Compute state-estimation uncertainty and dynamics fluctuation uncertainty for 1 batch.

    State-estimation uncertainty
      z^(s) ~ q(z|a,s,u),  d = d_hat (fixed)
      U_state^m = Var_s[ cos(v^(s), v_bar^z) ]

    Dynamics fluctuation uncertainty
      z = z_hat (fixed),  d^(s) ~ q(d|z_hat)
      U_dyn^m = Var_s[ cos(v^(s), v_bar^d) ]

    Returns
    -------
    dict:
      z_hat          (N, z_dim)
      d_hat          (N, z_dim)
      vel_mean_rna   (N, rna_dim)   d-sampling 下での平均 RNA velocity
      vel_mean_atac  (N, atac_dim)  d-sampling 下での平均 ATAC velocity
      u_state_rna    (N,)
      u_state_atac   (N,)
      u_dyn_rna      (N,)
      u_dyn_atac     (N,)
    """
    s, u, a = batch[0].to(device), batch[1].to(device), batch[2].to(device)
    qz_r, qz_a, z_hat = _get_z_moe_dist(model, s, u, a)

    # d の posterior mean d_hat = E[q(d|z_hat)]
    qd_hat = model.vaes[0].enc_dyn(z_hat)
    d_hat  = qd_hat.mean  # (N, z_dim)

    # ── State-estimation uncertainty ─────────────────────────
    # z^(s) ~ q(z|a,s,u), d = d_hat 固定
    z_vel_rna_list  = []
    z_vel_atac_list = []
    for _ in range(n_samples):
        z_s = (qz_r.rsample() + qz_a.rsample()) / 2  # (N, z_dim)
        v_rna, v_atac = _velocity(model, z_s, d_hat)
        z_vel_rna_list.append(v_rna)
        z_vel_atac_list.append(v_atac)

    rna_z_stack  = torch.stack(z_vel_rna_list,  dim=0)  # (S, N, rna_dim)
    atac_z_stack = torch.stack(z_vel_atac_list, dim=0)  # (S, N, atac_dim)
    u_state_rna  = _cosine_sim_variance(rna_z_stack).cpu().numpy()   # (N,)
    u_state_atac = _cosine_sim_variance(atac_z_stack).cpu().numpy()  # (N,)

    # ── Dynamics fluctuation uncertainty ─────────────────────
    # z = z_hat 固定, d^(s) ~ q(d|z_hat)
    d_vel_rna_list  = []
    d_vel_atac_list = []
    qd_fixed = model.vaes[0].enc_dyn(z_hat)
    for _ in range(n_samples):
        d_s = qd_fixed.rsample()  # (N, z_dim)
        v_rna, v_atac = _velocity(model, z_hat, d_s)
        d_vel_rna_list.append(v_rna)
        d_vel_atac_list.append(v_atac)

    rna_d_stack  = torch.stack(d_vel_rna_list,  dim=0)
    atac_d_stack = torch.stack(d_vel_atac_list, dim=0)
    u_dyn_rna  = _cosine_sim_variance(rna_d_stack).cpu().numpy()   # (N,)
    u_dyn_atac = _cosine_sim_variance(atac_d_stack).cpu().numpy()  # (N,)

    return {
        "z_hat":         z_hat.cpu().numpy(),
        "d_hat":         d_hat.cpu().numpy(),
        "vel_mean_rna":  rna_d_stack.mean(0).cpu().numpy(),
        "vel_mean_atac": atac_d_stack.mean(0).cpu().numpy(),
        "u_state_rna":   u_state_rna,
        "u_state_atac":  u_state_atac,
        "u_dyn_rna":     u_dyn_rna,
        "u_dyn_atac":    u_dyn_atac,
    }


# ──────────────────────────────────────────────────────────────
# 2. Transition ambiguity
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def _compute_transition_ambiguity(
    model,
    dataloader: DataLoader,
    all_z_hat: np.ndarray,
    n_samples: int,
    n_neighbors: int,
    tau: float,
    device: str,
) -> dict:
    """
    Compute transition ambiguity using the kNN graph of all cells.

    z^(s) ~ q(z|a,s,u),  d^(s) ~ q(d|z^(s))
    z_future^(s) = z^(s) + rho * d^(s)

    T_{n->j}^(s) ∝ exp(-||z_future^(s) - z_j||^2 / tau)
    z_tilde^(s) = sum_j T_{n->j}^(s) * z_j

    U_trans^z   = Var_s[ cos(z_tilde^(s) - z_hat, z_tilde_bar - z_hat) ]
    U_trans^RNA = Var_s[ cos(x_tilde_RNA^(s) - x_bar_RNA, x_tilde_bar_RNA - x_bar_RNA) ]
    U_trans^ATAC= Var_s[ cos(x_tilde_ATAC^(s)- x_bar_ATAC,x_tilde_bar_ATAC- x_bar_ATAC) ]

    Parameters
    ----------
    all_z_hat : (N_total, z_dim)  フェーズ1で収集した全細胞の posterior mean

    Returns
    -------
    dict:
      u_trans_z    (N_total,)
      u_trans_rna  (N_total,)
      u_trans_atac (N_total,)
    """
    N_total, z_dim = all_z_hat.shape

    # ── construct kNN graph ──────────────────────────────────────
    k = min(n_neighbors, N_total - 1)
    nn_model = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
    nn_model.fit(all_z_hat)
    nn_idx = nn_model.kneighbors(all_z_hat, return_distance=False)  # (N_total, k)

    z_hat_tensor  = torch.tensor(all_z_hat, dtype=torch.float32, device=device)
    nn_idx_tensor = torch.tensor(nn_idx, dtype=torch.long, device=device)   # (N, k)
    z_neighbors   = z_hat_tensor[nn_idx_tensor]                              # (N, k, z_dim)

    # ── Compute transition ambiguity for each batch ──────────────
    u_trans_z_list    = []
    u_trans_rna_list  = []
    u_trans_atac_list = []

    cell_offset = 0
    for batch in dataloader:
        s, u, a = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        n_cells = s.shape[0]

        qz_r, qz_a, z_hat_batch = _get_z_moe_dist(model, s, u, a)

        # このバッチに対応する近傍テンソル
        batch_slice   = slice(cell_offset, cell_offset + n_cells)
        z_neigh_batch = z_neighbors[batch_slice]  # (n_cells, k, z_dim)

        # S 回サンプリング
        z_tilde_samples     = []
        x_tilde_rna_samples = []
        x_tilde_atac_samples= []

        for _ in range(n_samples):
            # z^(s) ~ q(z|a,s,u), d^(s) ~ q(d|z^(s))
            z_s     = (qz_r.rsample() + qz_a.rsample()) / 2   # (n_cells, z_dim)
            d_s     = model.vaes[0].enc_dyn(z_s).rsample()    # (n_cells, z_dim)

            z_future = z_s + model.d_coeff * d_s              # (n_cells, z_dim)

            # 近傍への遷移重み: T_{n->j} ∝ exp(-||z_future - z_j||^2 / tau)
            diff    = z_future.unsqueeze(1) - z_neigh_batch   # (n_cells, k, z_dim)
            sq_dist = (diff ** 2).sum(-1)                      # (n_cells, k)
            w       = torch.softmax(-sq_dist / tau, dim=-1)   # (n_cells, k)

            # 期待 future latent state
            z_tilde = (w.unsqueeze(-1) * z_neigh_batch).sum(1)  # (n_cells, z_dim)
            z_tilde_samples.append(z_tilde)

            # modality への投影
            x_rna  = model.vaes[0].dec_su(z_tilde)[0][0] * model.norm_mat_s   # (n_cells, rna_dim)
            x_atac = model.vaes[1].dec_ald(z_tilde)[0]   * model.norm_mat_a   # (n_cells, atac_dim)
            x_tilde_rna_samples.append(x_rna)
            x_tilde_atac_samples.append(x_atac)

        # (S, n_cells, dim) に集約
        z_tilde_stack    = torch.stack(z_tilde_samples,      dim=0)  # (S, n_cells, z_dim)
        rna_tilde_stack  = torch.stack(x_tilde_rna_samples,  dim=0)
        atac_tilde_stack = torch.stack(x_tilde_atac_samples, dim=0)

        # 基準点: z_hat, x_bar = f(z_hat) での現在状態
        z_hat_b    = z_hat_batch.unsqueeze(0)                                       # (1, n_cells, z_dim)
        x_bar_rna  = model.vaes[0].dec_su(z_hat_batch)[0][0] * model.norm_mat_s   # (n_cells, rna_dim)
        x_bar_atac = model.vaes[1].dec_ald(z_hat_batch)[0]   * model.norm_mat_a   # (n_cells, atac_dim)

        # 変位ベクトル (現在状態からの差分)
        z_disp    = z_tilde_stack    - z_hat_b                  # (S, n_cells, z_dim)
        rna_disp  = rna_tilde_stack  - x_bar_rna.unsqueeze(0)   # (S, n_cells, rna_dim)
        atac_disp = atac_tilde_stack - x_bar_atac.unsqueeze(0)  # (S, n_cells, atac_dim)

        u_trans_z_list.append(   _cosine_sim_variance(z_disp).cpu().numpy()    )
        u_trans_rna_list.append( _cosine_sim_variance(rna_disp).cpu().numpy()  )
        u_trans_atac_list.append(_cosine_sim_variance(atac_disp).cpu().numpy() )

        cell_offset += n_cells

    return {
        "u_trans_z":    np.concatenate(u_trans_z_list,    axis=0),
        "u_trans_rna":  np.concatenate(u_trans_rna_list,  axis=0),
        "u_trans_atac": np.concatenate(u_trans_atac_list, axis=0),
    }


# ──────────────────────────────────────────────────────────────
# 3. main function to compute and save velocity uncertainty
# ──────────────────────────────────────────────────────────────

def compute_velocity_uncertainty(
    model,
    dataloader: DataLoader,
    n_samples: int = 100,
    n_neighbors: int = 30,
    tau: Optional[float] = None,
    device: str = "cuda",
) -> dict:
    """
    Scan the entire DataLoader and perform 3-layer uncertainty decomposition.

    Phase 1: Compute state-estimation uncertainty and dynamics fluctuation uncertainty for each batch, and collect z_hat.
    Phase 2: Construct a kNN graph with all cells' z_hat and compute transition ambiguity.

    Parameters
    ----------
    model        : trained DREG_DYN instance
    dataloader   : DataLoader
    n_samples    : number of sampling iterations (common across all layers)
    n_neighbors  : number of neighbors in the kNN graph (for transition ambiguity)
    tau          : bandwidth for transition ambiguity (in squared distance units).
                   If None, the median of neighbor distances is used.
    device       : torch device string

    Returns
    -------
    dict:
      z_hat          (N, z_dim)    posterior mean latent state
      d_hat          (N, z_dim)    posterior mean dynamics
      vel_mean_rna   (N, rna_dim)  RNA velocity 平均 (d-sampling)
      vel_mean_atac  (N, atac_dim) ATAC velocity 平均 (d-sampling)
      u_state_rna    (N,)  state-estimation uncertainty (RNA)
      u_state_atac   (N,)  state-estimation uncertainty (ATAC)
      u_dyn_rna      (N,)  dynamics fluctuation uncertainty (RNA)
      u_dyn_atac     (N,)  dynamics fluctuation uncertainty (ATAC)
      u_trans_z      (N,)  latent-space transition ambiguity
      u_trans_rna    (N,)  RNA-projected transition ambiguity
      u_trans_atac   (N,)  ATAC-projected transition ambiguity
    """
    model.to(device)
    model.eval()

    # ── フェーズ 1: state-estimation / dynamics fluctuation uncertainty ──
    print("[Phase 1] Computing state-estimation and dynamics fluctuation uncertainty...")
    accum = defaultdict(list)
    for batch in dataloader:
        out = _batch_state_dynamics_uncertainty(model, batch, n_samples, device)
        for k, v in out.items():
            accum[k].append(v)

    results = {k: np.concatenate(v, axis=0) for k, v in accum.items()}

    # ── bandwidth の自動決定 ─────────────────────────────────
    all_z_hat = results["z_hat"]  # (N, z_dim)
    if tau is None:
        k_tmp = min(10, len(all_z_hat) - 1)
        nn_tmp = NearestNeighbors(n_neighbors=k_tmp, metric="euclidean")
        nn_tmp.fit(all_z_hat)
        dists_tmp, _ = nn_tmp.kneighbors(all_z_hat)
        tau = float(np.median(dists_tmp ** 2))
        tau = max(tau, 1e-6)
        print(f"[Phase 2] Auto bandwidth tau = {tau:.6f}")

    # ── フェーズ 2: transition ambiguity ─────────────────────
    print("[Phase 2] Computing transition ambiguity...")
    trans_results = _compute_transition_ambiguity(
        model, dataloader, all_z_hat,
        n_samples=n_samples,
        n_neighbors=n_neighbors,
        tau=tau,
        device=device,
    )
    results.update(trans_results)

    return results


# ──────────────────────────────────────────────────────────────
# 4. saving results to AnnData and plotting (example downstream analysis)
# ──────────────────────────────────────────────────────────────

def save_velocity_uncertainty(
    dm,
    results: dict,
    save_dir: str,
    filename_rna: str = "adata_rna_uncertainty.loom",
    filename_atac: str = "adata_atac_uncertainty.loom",
) -> None:
    """
    save 3-layer decomposition uncertainty results to AnnData and save as .loom files.

    Storage locations (RNA AnnData):
      layers["vel_mean"]        : RNA velocity average           (N, rna_dim)
      obsm["z_hat"]             : posterior mean latent state (N, z_dim)
      obsm["d_hat"]             : posterior mean dynamics     (N, z_dim)
      obs["u_state"]            : state-estimation uncertainty
      obs["u_dyn"]              : dynamics fluctuation uncertainty
      obs["u_trans"]            : RNA-projected transition ambiguity
      obs["u_trans_z"]          : latent-space transition ambiguity

    Storage locations (ATAC AnnData):
      layers["vel_mean"]        : ATAC velocity average          (N, atac_dim)
      obs["u_state"]            : state-estimation uncertainty
      obs["u_dyn"]              : dynamics fluctuation uncertainty
      obs["u_trans"]            : ATAC-projected transition ambiguity

    Parameters
    ----------
    dm           : DataModule (adata_r / adata_a を持つもの)
    results      : compute_velocity_uncertainty() の戻り値
    save_dir     : 保存先ディレクトリ
    filename_rna : RNA AnnData の loom ファイル名
    filename_atac: ATAC AnnData の loom ファイル名
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── RNA AnnData ───────────────────────────────────────────
    dm.adata_r.layers["vel_mean"]   = results["vel_mean_rna"]
    dm.adata_r.obsm["z_hat"]        = results["z_hat"]
    dm.adata_r.obsm["d_hat"]        = results["d_hat"]
    dm.adata_r.obs["u_state"]       = results["u_state_rna"]
    dm.adata_r.obs["u_dyn"]         = results["u_dyn_rna"]
    dm.adata_r.obs["u_trans"]       = results["u_trans_rna"]
    dm.adata_r.obs["u_trans_z"]     = results["u_trans_z"]

    # ── ATAC AnnData ──────────────────────────────────────────
    dm.adata_a.layers["vel_mean"]   = results["vel_mean_atac"]
    dm.adata_a.obs["u_state"]       = results["u_state_atac"]
    dm.adata_a.obs["u_dyn"]         = results["u_dyn_atac"]
    dm.adata_a.obs["u_trans"]       = results["u_trans_atac"]

    # ── .loom に保存 ──────────────────────────────────────────
    rna_path  = os.path.join(save_dir, filename_rna)
    atac_path = os.path.join(save_dir, filename_atac)

    dm.adata_r.write_loom(rna_path,  write_obsm_varm=True)
    dm.adata_a.write_loom(atac_path, write_obsm_varm=True)

    print(f"Saved RNA  AnnData → {rna_path}")
    print(f"Saved ATAC AnnData → {atac_path}")


# ──────────────────────────────────────────────────────────────
# 5. example main function to run the entire pipeline
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Pretrained model and data module are assumed to be loaded
    # model_dyn : DREG_DYN (学習済み)
    # dm_s      : DynDataModule_Smooth など

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dm_s.setup()

    results = compute_velocity_uncertainty(
        model_dyn,
        dm_s.all_dataloader(), 
        n_samples=200,
        n_neighbors=30,
        tau=None,                #bandwidth
        device=device,
    )

    save_dir = runPath + "/downstream_analysis/result/anndata"
    save_velocity_uncertainty(dm_s, results, save_dir)
