"""
velocity_off_manifold_mod.py

We quantify on- and off-manifold uncertainty of latent dynamics
in mmVelo directly within each modality-specific space.

Unlike the latent-space decomposition implemented in velocity_off_manifold.py, 
this approach constructs local tangent spaces 
based on denoised modality-specific representations (RNA: PCA; ATAC: TruncatedSVD/LSI). 
Modality-specific velocity samples are then decomposed into on- and off-manifold components 
within these spaces.

Processing flow:
  Phase 1 : Collect z_hat and denoised x_RNA, x_ATAC for all cells
  Phase 2 : Compute low-dimensional embeddings for each modality using PCA / TruncatedSVD,
            and construct kNN graphs for each modality
  Phase 3 : Sample d for each batch and compute velocity,
            then project into the local tangent spaces of each modality to evaluate on/off-manifold variance

Defined metrics (m ∈ {RNA, ATAC}):
  U_dyn^m          : dynamics fluctuation uncertainty (all components)
  U_dyn_par^m      : on-manifold dynamics fluctuation
  U_dyn_perp^m     : off-manifold instability
  R_n^m            : off-manifold energy ratio, defined as
                     mean_s[ ||v_perp^m,(s)||^2 / ||v^m,(s)||^2 ]
"""

import os
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as dist
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA, TruncatedSVD
from torch.utils.data import DataLoader


# ──────────────────────────────────────────────────────────────
# utlities
# ──────────────────────────────────────────────────────────────

def _cosine_sim_variance(samples: torch.Tensor) -> torch.Tensor:
    """
    samples : (S, N, dim)
    returns : (N,)  各セルの cosine 類似度の分散 (Bessel 補正あり)
    """
    v_bar = samples.mean(0)                                         # (N, dim)
    cos   = (F.normalize(samples, dim=-1, eps=1e-8)
             * F.normalize(v_bar.unsqueeze(0), dim=-1, eps=1e-8)
             ).sum(-1)                                              # (S, N)
    return cos.var(dim=0, unbiased=True)                            # (N,)


def _get_z_hat(model, s, u, a) -> torch.Tensor:
    """posterior mean z_hat = (mu_r + mu_a) / 2"""
    qz_mu_r, _ = model.vaes[0].enc_z(s, u)
    qz_mu_a, _ = model.vaes[1].enc_z(a)
    return (qz_mu_r + qz_mu_a) / 2  # (N, z_dim)


def _local_tangent_projector_batched(
    embed_batch: torch.Tensor,
    neigh_embed_batch: torch.Tensor,
    manifold_dim: int,
) -> torch.Tensor:
    """
    Compute the projection matrix onto the local tangent space 
    in the low-dimensional embedding of each modality.

    Parameters
    ----------
    embed_batch       : (n_cells, d_emb)    各細胞の埋め込み
    neigh_embed_batch : (n_cells, k, d_emb) 近傍細胞の埋め込み
    manifold_dim      : r  top-r PCs の数

    Returns
    -------
    P_n : (n_cells, d_emb, d_emb)  射影行列 P = U_n U_n^T
    """
    # 近傍を中心化
    centered = neigh_embed_batch - embed_batch.unsqueeze(1)    # (n_cells, k, d_emb)

    # batched SVD: Vh[:, :r, :] が上位 r 右特異ベクトル (各行 = 主方向)
    _, _, Vh = torch.linalg.svd(centered, full_matrices=False)  # Vh: (n_cells, min(k,d_emb), d_emb)

    r   = min(manifold_dim, Vh.shape[1])
    U_n = Vh[:, :r, :].transpose(-1, -2)     # (n_cells, d_emb, r)
    P_n = U_n @ U_n.transpose(-1, -2)         # (n_cells, d_emb, d_emb)
    return P_n


# ──────────────────────────────────────────────────────────────
# Phase 1: Compute z_hat and denoised modality representations
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def _collect_representations(
    model, dataloader: DataLoader, device: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute z_hat, x_RNA = f^s(z_hat)*C_s, x_ATAC = f^a(z_hat)*C_a

    Returns
    -------
    all_z_hat   : (N, z_dim)
    all_x_rna   : (N, rna_dim)
    all_x_atac  : (N, atac_dim)
    """
    model.eval()
    z_hat_list  = []
    x_rna_list  = []
    x_atac_list = []

    for batch in dataloader:
        s, u, a = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        z_hat   = _get_z_hat(model, s, u, a)
        x_rna   = model.vaes[0].dec_su(z_hat)[0][0] * model.norm_mat_s
        x_atac  = model.vaes[1].dec_ald(z_hat)[0]   * model.norm_mat_a
        z_hat_list.append(z_hat.cpu().numpy())
        x_rna_list.append(x_rna.cpu().numpy())
        x_atac_list.append(x_atac.cpu().numpy())

    return (
        np.concatenate(z_hat_list,  axis=0),
        np.concatenate(x_rna_list,  axis=0),
        np.concatenate(x_atac_list, axis=0),
    )


# ──────────────────────────────────────────────────────────────
# Phase 2: compute low dimensional embeddings and construct a kNN graph
# ──────────────────────────────────────────────────────────────

def _build_modality_embedding_and_knn(
    x_rna: np.ndarray,
    x_atac: np.ndarray,
    n_pca_rna: int,
    n_lsi_atac: int,
    n_neighbors: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For RNA and ATAC:
      - compute low-dimensional embeddings (PCA / TruncatedSVD)
      - obtain loading matrix (to project velocity onto the low-dimensional space)
      - construct a kNNgraph

    Returns
    -------
    embed_rna   : (N, n_pca_rna)   RNA 低次元埋め込み
    W_rna       : (rna_dim, n_pca_rna) PCA loading 行列
    nn_idx_rna  : (N, k)           RNA kNN インデックス
    embed_atac  : (N, n_lsi_atac)  ATAC 低次元埋め込み
    W_atac      : (atac_dim, n_lsi_atac) TruncatedSVD loading 行列
    nn_idx_atac : (N, k)           ATAC kNN インデックス
    """
    N = x_rna.shape[0]
    k = min(n_neighbors, N - 1)

    # ── RNA: PCA ──────────────────────────────────────────────
    n_comp_rna  = min(n_pca_rna, x_rna.shape[1], N - 1)
    pca_rna     = PCA(n_components=n_comp_rna, whiten=False)
    embed_rna   = pca_rna.fit_transform(x_rna).astype(np.float32)  # (N, n_pca_rna)
    W_rna       = pca_rna.components_.T.astype(np.float32)          # (rna_dim, n_pca_rna)

    nn_rna = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
    nn_rna.fit(embed_rna)
    nn_idx_rna = nn_rna.kneighbors(embed_rna, return_distance=False)  # (N, k)

    # ── ATAC: TruncatedSVD (LSI) ──────────────────────────────
    n_comp_atac = min(n_lsi_atac, x_atac.shape[1], N - 1)
    svd_atac    = TruncatedSVD(n_components=n_comp_atac, algorithm="randomized")
    embed_atac  = svd_atac.fit_transform(x_atac).astype(np.float32)   # (N, n_lsi_atac)
    W_atac      = svd_atac.components_.T.astype(np.float32)            # (atac_dim, n_lsi_atac)

    nn_atac = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="auto")
    nn_atac.fit(embed_atac)
    nn_idx_atac = nn_atac.kneighbors(embed_atac, return_distance=False)  # (N, k)

    return embed_rna, W_rna, nn_idx_rna, embed_atac, W_atac, nn_idx_atac


# ──────────────────────────────────────────────────────────────
# Phase 3: compute on/off-manifold decomposition
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def _batch_off_manifold_mod(
    model,
    batch,
    # RNA modality
    embed_rna_batch: torch.Tensor,       # (n_cells, n_pca_rna)
    neigh_embed_rna_batch: torch.Tensor, # (n_cells, k, n_pca_rna)
    W_rna: torch.Tensor,                 # (rna_dim, n_pca_rna)
    manifold_dim_rna: int,
    # ATAC modality
    embed_atac_batch: torch.Tensor,      # (n_cells, n_lsi_atac)
    neigh_embed_atac_batch: torch.Tensor,# (n_cells, k, n_lsi_atac)
    W_atac: torch.Tensor,                # (atac_dim, n_lsi_atac)
    manifold_dim_atac: int,
    # sampling
    n_samples: int,
    device: str,
) -> dict:
    """
    Compute the modality-specific on/off-manifold uncertainty decomposition for one batch.

    The velocity is projected onto the low-dimensional embeddings of each modality
    before decomposing it into on- and off-manifold components.

    Returns
    -------
    dict:
      z_hat              (n_cells, z_dim)
      vel_mean_rna       (n_cells, rna_dim)   RNA velocity 平均
      vel_mean_atac      (n_cells, atac_dim)  ATAC velocity 平均
      u_dyn_rna          (n_cells,)
      u_dyn_atac         (n_cells,)
      u_dyn_par_rna      (n_cells,)
      u_dyn_par_atac     (n_cells,)
      u_dyn_perp_rna     (n_cells,)
      u_dyn_perp_atac    (n_cells,)
      off_ratio_rna      (n_cells,)
      off_ratio_atac     (n_cells,)
    """
    s, u, a = batch[0].to(device), batch[1].to(device), batch[2].to(device)
    z_hat   = _get_z_hat(model, s, u, a)   # (n_cells, z_dim)

    # ── modality-specific 局所接線空間の射影行列 ──────────────
    # RNA: (n_cells, n_pca_rna, n_pca_rna)
    P_rna  = _local_tangent_projector_batched(embed_rna_batch,  neigh_embed_rna_batch,  manifold_dim_rna)
    I_rna  = torch.eye(P_rna.shape[-1],  device=device).unsqueeze(0).expand(z_hat.shape[0], -1, -1)

    # ATAC: (n_cells, n_lsi_atac, n_lsi_atac)
    P_atac = _local_tangent_projector_batched(embed_atac_batch, neigh_embed_atac_batch, manifold_dim_atac)
    I_atac = torch.eye(P_atac.shape[-1], device=device).unsqueeze(0).expand(z_hat.shape[0], -1, -1)

    # base decode
    s_base = model.vaes[0].dec_su(z_hat)[0][0] * model.norm_mat_s  # (n_cells, rna_dim)
    a_base = model.vaes[1].dec_ald(z_hat)[0]   * model.norm_mat_a  # (n_cells, atac_dim)

    # q(d | z_hat)
    qd = model.vaes[0].enc_dyn(z_hat)

    # (high-dim velocity — for mean and total U_dyn)
    vel_rna_full_list  = []
    vel_atac_full_list = []
    # low-dim projected velocity (for on/off decomposition)
    vel_rna_pca_list   = []
    vel_atac_lsi_list  = []

    off_ratio_rna_list  = []
    off_ratio_atac_list = []

    for _ in range(n_samples):
        d_s = qd.rsample()  # (n_cells, z_dim)

        # velocity in high-dim space
        z_dt  = z_hat + model.d_coeff * d_s
        v_rna  = model.vaes[0].dec_su(z_dt)[0][0] * model.norm_mat_s - s_base   # (n_cells, rna_dim)
        v_atac = model.vaes[1].dec_ald(z_dt)[0]   * model.norm_mat_a - a_base   # (n_cells, atac_dim)

        vel_rna_full_list.append(v_rna)
        vel_atac_full_list.append(v_atac)

        # velocity を低次元に射影
        v_rna_pca  = v_rna  @ W_rna   # (n_cells, n_pca_rna)
        v_atac_lsi = v_atac @ W_atac  # (n_cells, n_lsi_atac)

        vel_rna_pca_list.append(v_rna_pca)
        vel_atac_lsi_list.append(v_atac_lsi)

        # off-manifold energy ratio (低次元空間で計算)
        v_rna_perp_pca  = ((I_rna  - P_rna)  @ v_rna_pca.unsqueeze(-1)).squeeze(-1)
        v_atac_perp_lsi = ((I_atac - P_atac) @ v_atac_lsi.unsqueeze(-1)).squeeze(-1)

        norm_rna_sq   = (v_rna_pca  ** 2).sum(-1).clamp(min=1e-8)
        norm_atac_sq  = (v_atac_lsi ** 2).sum(-1).clamp(min=1e-8)
        off_ratio_rna_list.append( (v_rna_perp_pca  ** 2).sum(-1) / norm_rna_sq  )
        off_ratio_atac_list.append((v_atac_perp_lsi ** 2).sum(-1) / norm_atac_sq )

    # (S, n_cells, dim) に集約
    rna_full_stack  = torch.stack(vel_rna_full_list,  dim=0)  # (S, n_cells, rna_dim)
    atac_full_stack = torch.stack(vel_atac_full_list, dim=0)
    rna_pca_stack   = torch.stack(vel_rna_pca_list,   dim=0)  # (S, n_cells, n_pca_rna)
    atac_lsi_stack  = torch.stack(vel_atac_lsi_list,  dim=0)

    # on-manifold / off-manifold 成分 (低次元空間)
    def _proj(stack, P, I):
        # stack: (S, n_cells, d_emb), P: (n_cells, d_emb, d_emb)
        v_col = stack.unsqueeze(-1)                    # (S, n_cells, d_emb, 1)
        par   = (P.unsqueeze(0)       @ v_col).squeeze(-1)   # (S, n_cells, d_emb)
        perp  = ((I - P).unsqueeze(0) @ v_col).squeeze(-1)
        return par, perp

    rna_par_stack,  rna_perp_stack  = _proj(rna_pca_stack,  P_rna,  I_rna)
    atac_par_stack, atac_perp_stack = _proj(atac_lsi_stack, P_atac, I_atac)

    def _var(stack):
        return _cosine_sim_variance(stack).cpu().numpy()

    return {
        "z_hat":             z_hat.cpu().numpy(),
        "vel_mean_rna":      rna_full_stack.mean(0).cpu().numpy(),
        "vel_mean_atac":     atac_full_stack.mean(0).cpu().numpy(),
        # total (high-dim velocity)
        "u_dyn_rna":         _var(rna_full_stack),
        "u_dyn_atac":        _var(atac_full_stack),
        # on-manifold (low-dim projected)
        "u_dyn_par_rna":     _var(rna_par_stack),
        "u_dyn_par_atac":    _var(atac_par_stack),
        # off-manifold (low-dim projected)
        "u_dyn_perp_rna":    _var(rna_perp_stack),
        "u_dyn_perp_atac":   _var(atac_perp_stack),
        # off-manifold エネルギー比
        "off_ratio_rna":     torch.stack(off_ratio_rna_list,  dim=0).mean(0).cpu().numpy(),
        "off_ratio_atac":    torch.stack(off_ratio_atac_list, dim=0).mean(0).cpu().numpy(),
    }


# ──────────────────────────────────────────────────────────────
# main function
# ──────────────────────────────────────────────────────────────

def compute_off_manifold_uncertainty_mod(
    model,
    dataloader: DataLoader,
    n_samples: int = 200,
    n_neighbors: int = 50,
    n_pca_rna: int = 50,
    n_lsi_atac: int = 50,
    manifold_dim_rna: int = 10,
    manifold_dim_atac: int = 10,
    device: str = "cuda",
) -> dict:
    """
    Compute modality-specific on/off-manifold uncertainty decomposition for all cells.

    Phase 1: Compute z_hat and denoised modality representations (x_RNA, x_ATAC).
    Phase 2: Compute low-dimension embeddings for each modality (RNA: PCA, ATAC: TruncatedSVD)
             and construct kNN graphs for each modality.
    Phase 3: Sample d for each batch and compute velocity, then estimate local tangent spaces in the low-dimensional modality space and decompose the uncertainty.

    Parameters
    ----------
    model           : trained model (DREG_DYN)
    dataloader      : DataLoader
    n_samples       : number of sampling iterations S
    n_neighbors     : number of kNN neighbors k
    n_pca_rna       : dimension of RNA low-dimensional embedding
    n_lsi_atac      : dimension of ATAC low-dimensional embedding
    manifold_dim_rna : dimension of RNA local tangent space r
    manifold_dim_atac: dimension of ATAC local tangent space r
    device          : Device to run the computation on (e.g., "cuda" or "cpu")

    Returns
    -------
    dict:
      z_hat              (N, z_dim)
      vel_mean_rna       (N, rna_dim)
      vel_mean_atac      (N, atac_dim)
      u_dyn_rna          (N,)  total dynamics fluctuation uncertainty (RNA)
      u_dyn_atac         (N,)  total dynamics fluctuation uncertainty (ATAC)
      u_dyn_par_rna      (N,)  on-manifold dynamics fluctuation (RNA)
      u_dyn_par_atac     (N,)  on-manifold dynamics fluctuation (ATAC)
      u_dyn_perp_rna     (N,)  off-manifold instability (RNA)
      u_dyn_perp_atac    (N,)  off-manifold instability (ATAC)
      off_ratio_rna      (N,)  off-manifold energy ratio R_n^RNA
      off_ratio_atac     (N,)  off-manifold energy ratio R_n^ATAC
    """
    model.to(device)
    model.eval()

    # ── Phase 1 ─────────────────────────────
    print("[Phase 1] Collecting z_hat, x_RNA, x_ATAC for all cells...")
    all_z_hat, all_x_rna, all_x_atac = _collect_representations(model, dataloader, device)
    N = len(all_z_hat)
    print(f"          N={N}, rna_dim={all_x_rna.shape[1]}, atac_dim={all_x_atac.shape[1]}")

    # ── Phase 2────────────────────────
    print(f"[Phase 2] Building modality-specific embeddings and kNN graphs "
          f"(RNA: PCA-{n_pca_rna}, ATAC: SVD-{n_lsi_atac}, k={n_neighbors})...")
    (embed_rna, W_rna,
     nn_idx_rna,
     embed_atac, W_atac,
     nn_idx_atac) = _build_modality_embedding_and_knn(
        all_x_rna, all_x_atac,
        n_pca_rna=n_pca_rna,
        n_lsi_atac=n_lsi_atac,
        n_neighbors=n_neighbors,
    )
    print(f"          embed_rna={embed_rna.shape}, embed_atac={embed_atac.shape}")

    # テンソル化
    embed_rna_t   = torch.tensor(embed_rna,   dtype=torch.float32, device=device)  # (N, n_pca_rna)
    embed_atac_t  = torch.tensor(embed_atac,  dtype=torch.float32, device=device)  # (N, n_lsi_atac)
    W_rna_t       = torch.tensor(W_rna,       dtype=torch.float32, device=device)  # (rna_dim, n_pca_rna)
    W_atac_t      = torch.tensor(W_atac,      dtype=torch.float32, device=device)  # (atac_dim, n_lsi_atac)

    nn_rna_t  = torch.tensor(nn_idx_rna,  dtype=torch.long, device=device)  # (N, k)
    nn_atac_t = torch.tensor(nn_idx_atac, dtype=torch.long, device=device)  # (N, k)

    # 近傍埋め込み配列
    neigh_embed_rna  = embed_rna_t[nn_rna_t]   # (N, k, n_pca_rna)
    neigh_embed_atac = embed_atac_t[nn_atac_t]  # (N, k, n_lsi_atac)

    # ── Phase 3───────────
    print(f"[Phase 3] Computing modality-specific on/off-manifold decomposition "
          f"(n_samples={n_samples}, manifold_dim_rna={manifold_dim_rna}, "
          f"manifold_dim_atac={manifold_dim_atac})...")
    accum       = defaultdict(list)
    cell_offset = 0

    for batch in dataloader:
        n_cells     = batch[0].shape[0]
        sl          = slice(cell_offset, cell_offset + n_cells)

        out = _batch_off_manifold_mod(
            model, batch,
            embed_rna_batch        = embed_rna_t[sl],
            neigh_embed_rna_batch  = neigh_embed_rna[sl],
            W_rna                  = W_rna_t,
            manifold_dim_rna       = manifold_dim_rna,
            embed_atac_batch       = embed_atac_t[sl],
            neigh_embed_atac_batch = neigh_embed_atac[sl],
            W_atac                 = W_atac_t,
            manifold_dim_atac      = manifold_dim_atac,
            n_samples              = n_samples,
            device                 = device,
        )
        for key, val in out.items():
            accum[key].append(val)
        cell_offset += n_cells

    return {key: np.concatenate(vals, axis=0) for key, vals in accum.items()}


# ──────────────────────────────────────────────────────────────
# save
# ──────────────────────────────────────────────────────────────

def save_off_manifold_uncertainty_mod(
    dm,
    results: dict,
    save_dir: str,
    filename_rna:  str = "adata_rna_offmanifold_mod.loom",
    filename_atac: str = "adata_atac_offmanifold_mod.loom",
) -> None:
    """
    save the results of modality-specific off-manifold decomposition 
    to AnnData and save as .loom files.

    RNA AnnData:
      layers["vel_mean"]       : RNA velocity 平均      (N, rna_dim)
      obsm["z_hat"]            : posterior mean z       (N, z_dim)
      obs["u_dyn"]             : total dynamics fluctuation uncertainty
      obs["u_dyn_par"]         : on-manifold dynamics fluctuation
      obs["u_dyn_perp"]        : off-manifold instability
      obs["off_ratio"]         : off-manifold energy ratio R_n^RNA

    ATAC AnnData:
      layers["vel_mean"]       : ATAC velocity 平均     (N, atac_dim)
      obs["u_dyn"]             : total dynamics fluctuation uncertainty
      obs["u_dyn_par"]         : on-manifold dynamics fluctuation
      obs["u_dyn_perp"]        : off-manifold instability
      obs["off_ratio"]         : off-manifold energy ratio R_n^ATAC
    """
    os.makedirs(save_dir, exist_ok=True)

    dm.adata_r.layers["vel_mean"]  = results["vel_mean_rna"]
    dm.adata_r.obsm["z_hat"]       = results["z_hat"]
    dm.adata_r.obs["u_dyn"]        = results["u_dyn_rna"]
    dm.adata_r.obs["u_dyn_par"]    = results["u_dyn_par_rna"]
    dm.adata_r.obs["u_dyn_perp"]   = results["u_dyn_perp_rna"]
    dm.adata_r.obs["off_ratio"]    = results["off_ratio_rna"]

    dm.adata_a.layers["vel_mean"]  = results["vel_mean_atac"]
    dm.adata_a.obs["u_dyn"]        = results["u_dyn_atac"]
    dm.adata_a.obs["u_dyn_par"]    = results["u_dyn_par_atac"]
    dm.adata_a.obs["u_dyn_perp"]   = results["u_dyn_perp_atac"]
    dm.adata_a.obs["off_ratio"]    = results["off_ratio_atac"]

    rna_path  = os.path.join(save_dir, filename_rna)
    atac_path = os.path.join(save_dir, filename_atac)
    dm.adata_r.write_loom(rna_path,  write_obsm_varm=True)
    dm.adata_a.write_loom(atac_path, write_obsm_varm=True)

    print(f"Saved RNA  AnnData → {rna_path}")
    print(f"Saved ATAC AnnData → {atac_path}")
