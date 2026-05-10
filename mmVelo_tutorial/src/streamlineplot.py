""" 
copied from scvelo, modified scv.tl.velocity_graph
to incorporate tanh() transformation for delta and d
before calculating cosine similarity.
"""

import os
import numpy as np
from scipy.sparse import coo_matrix, issparse
from scvelo import logging as logg
from scvelo import settings
from scvelo.core import get_n_jobs, l2_norm, parallelize
from scvelo.preprocessing.moments import get_moments
from scvelo.preprocessing.neighbors import (
    get_n_neighs,
    get_neighs,
    neighbors,
    pca,
    verify_neighbors,
)



import warnings
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
import matplotlib.pyplot as pl
from scvelo.core import l2_norm, prod_sum, sum
warnings.simplefilter("ignore")


# TODO: Add docstrings
def round(k, dec=2, as_str=None):
    """TODO."""
    if isinstance(k, (list, tuple, np.record, np.ndarray)):
        return [round(ki, dec) for ki in k]
    if "e" in f"{k}":
        k_str = f"{k}".split("e")
        result = f"{np.round(float(k_str[0]), dec)}1e{k_str[1]}"
        return f"{result}" if as_str else float(result)
    result = np.round(float(k), dec)
    return f"{result}" if as_str else result


# TODO: Add docstrings
def mean(x, axis=0):
    """TODO."""
    return x.mean(axis).A1 if issparse(x) else x.mean(axis)


# TODO: Add docstrings
def make_dense(X):
    """TODO."""
    XA = X.A if issparse(X) and X.ndim == 2 else X.A1 if issparse(X) else X
    if XA.ndim == 2:
        XA = XA[0] if XA.shape[0] == 1 else XA[:, 0] if XA.shape[1] == 1 else XA
    return np.array(XA)


# TODO: Add docstrings
def R_squared(residual, total):
    """TODO."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r2 = np.ones(residual.shape[1]) - prod_sum(
            residual, residual, axis=0
        ) / prod_sum(total, total, axis=0)
    r2[np.isnan(r2)] = 0
    return r2


# TODO: Add docstrings
def cosine_correlation(dX, Vi):
    """TODO."""
    dx = dX - dX.mean(-1)[:, None]
    Vi_norm = l2_norm(Vi, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if Vi_norm == 0:
            result = np.zeros(dx.shape[0])
        else:
            result = (
                np.einsum("ij, j", dx, Vi) / (l2_norm(dx, axis=1) * Vi_norm)[None, :]
            )
    return result


# TODO: Add docstrings
def normalize(X):
    """TODO."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if issparse(X):
            return X.multiply(csr_matrix(1.0 / np.abs(X).sum(1)))
        else:
            return X / X.sum(1)


# TODO: Add docstrings
def scale(X, min=0, max=1):
    """TODO."""
    idx = np.isfinite(X)
    if any(idx):
        X = X - X[idx].min() + min
        xmax = X[idx].max()
        X = X / xmax * max if xmax != 0 else X * max
    return X


# TODO: Add docstrings
def get_indices(dist, n_neighbors=None, mode_neighbors="distances"):
    """TODO."""
    from scvelo.preprocessing.neighbors import compute_connectivities_umap

    D = dist.copy()
    D.data += 1e-6

    n_counts = sum(D > 0, axis=1)
    n_neighbors = (
        n_counts.min() if n_neighbors is None else min(n_counts.min(), n_neighbors)
    )
    rows = np.where(n_counts > n_neighbors)[0]
    cumsum_neighs = np.insert(n_counts.cumsum(), 0, 0)
    dat = D.data

    for row in rows:
        n0, n1 = cumsum_neighs[row], cumsum_neighs[row + 1]
        rm_idx = n0 + dat[n0:n1].argsort()[n_neighbors:]
        dat[rm_idx] = 0
    D.eliminate_zeros()

    D.data -= 1e-6
    if mode_neighbors == "distances":
        indices = D.indices.reshape((-1, n_neighbors))
    elif mode_neighbors == "connectivities":
        knn_indices = D.indices.reshape((-1, n_neighbors))
        knn_distances = D.data.reshape((-1, n_neighbors))
        _, conn = compute_connectivities_umap(
            knn_indices, knn_distances, D.shape[0], n_neighbors
        )
        indices = get_indices_from_csr(conn)
    return indices, D


# TODO: Add docstrings
def get_indices_from_csr(conn):
    """TODO."""
    # extracts indices from connectivity matrix, pads with nans
    ixs = np.ones((conn.shape[0], np.max((conn > 0).sum(1)))) * np.nan
    for i in range(ixs.shape[0]):
        cell_indices = conn[i, :].indices
        ixs[i, : len(cell_indices)] = cell_indices
    return ixs


# TODO: Add docstrings
def get_iterative_indices(
    indices,
    index,
    n_recurse_neighbors=2,
    max_neighs=None,
):
    """TODO."""

    def iterate_indices(indices, index, n_recurse_neighbors):
        if n_recurse_neighbors > 1:
            index = iterate_indices(indices, index, n_recurse_neighbors - 1)
        ix = np.append(index, indices[index])  # direct and indirect neighbors
        if np.isnan(ix).any():
            ix = ix[~np.isnan(ix)]
        return ix.astype(int)

    indices = np.unique(iterate_indices(indices, index, n_recurse_neighbors))
    if max_neighs is not None and len(indices) > max_neighs:
        indices = np.random.choice(indices, max_neighs, replace=False)
    return indices


# TODO: Add docstrings
def geometric_matrix_sum(C, n_power=2):  # computes C + C^2 + C^3 + ...
    """TODO."""
    C_n = (
        geometric_matrix_sum(C, n_power - 1) if n_power > 2 else C if n_power > 1 else 0
    )
    return C + C.dot(C_n)


# TODO: Add docstrings
def groups_to_bool(adata, groups, groupby=None):
    """TODO."""
    groups = [groups] if isinstance(groups, str) else groups
    if isinstance(groups, (list, tuple, np.ndarray, np.record)):
        groupby = (
            groupby
            if groupby in adata.obs.keys()
            else "clusters"
            if "clusters" in adata.obs.keys()
            else "louvain"
            if "louvain" in adata.obs.keys()
            else None
        )
        if groupby is not None:
            groups = np.array([key in groups for key in adata.obs[groupby]])
        else:
            raise ValueError("groupby attribute not valid.")
    return groups


# TODO: Add docstrings
def most_common_in_list(lst):
    """TODO."""
    lst = [item for item in lst if item is not np.nan and item != "nan"]
    lst = list(lst)
    return max(set(lst), key=lst.count)


# TODO: Add docstrings
def randomized_velocity(adata, vkey="velocity", add_key="velocity_random"):
    """TODO."""
    V_rnd = adata.layers[vkey].copy()
    for i in range(V_rnd.shape[1]):
        np.random.shuffle(V_rnd[:, i])
        V_rnd[:, i] = V_rnd[:, i] * np.random.choice(
            np.array([+1, -1]), size=V_rnd.shape[0]
        )
    adata.layers[add_key] = V_rnd

    from .velocity_embedding import velocity_embedding
    from .velocity_graph import velocity_graph

    velocity_graph(adata, vkey=add_key)
    velocity_embedding(adata, vkey=add_key, autoscale=False)


# TODO: Add docstrings
def extract_int_from_str(array):
    """TODO."""

    def str_to_int(item):
        num = "".join(filter(str.isdigit, item))
        num = int(num) if len(num) > 0 else -1
        return num

    if isinstance(array, str):
        nums = str_to_int(array)
    elif len(array) > 1 and isinstance(array[0], str):
        nums = []
        for item in array:
            nums.append(str_to_int(item))
    else:
        nums = array
    nums = pd.Categorical(nums) if array.dtype == "category" else np.array(nums)
    return nums


# TODO: Finish docstrings
def strings_to_categoricals(adata):
    """Transform string annotations to categoricals."""
    from pandas import Categorical
    from pandas.api.types import is_bool_dtype, is_integer_dtype, is_string_dtype

    def is_valid_dtype(values):
        return (
            is_string_dtype(values) or is_integer_dtype(values) or is_bool_dtype(values)
        )

    df = adata.obs
    df_keys = [key for key in df.columns if is_valid_dtype(df[key])]
    for key in df_keys:
        c = df[key]
        c = Categorical(c)
        if 1 < len(c.categories) < min(len(c), 100):
            df[key] = c

    df = adata.var
    df_keys = [key for key in df.columns if is_string_dtype(df[key])]
    for key in df_keys:
        c = df[key].astype("U")
        c = Categorical(c)
        if 1 < len(c.categories) < min(len(c), 100):
            df[key] = c


# TODO: Add docstrings
def merge_groups(adata, key, map_groups, key_added=None, map_colors=None):
    """TODO."""
    strings_to_categoricals(adata)
    if len(map_groups) != len(adata.obs[key].cat.categories):
        map_coarse = {}
        for c in adata.obs[key].cat.categories:
            for group in map_groups:
                if any(cluster == c for cluster in map_groups[group]):
                    map_coarse[c] = group
            if c not in map_coarse:
                map_coarse[c] = c
        map_groups = map_coarse

    if key_added is None:
        key_added = f"{key}_coarse"

    from pandas.api.types import CategoricalDtype

    adata.obs[key_added] = adata.obs[key].map(map_groups).astype(CategoricalDtype())
    old_categories = adata.obs[key].cat.categories
    new_categories = adata.obs[key_added].cat.categories

    # map_colors is passed
    if map_colors is not None:
        old_colors = None
        if f"{key}_colors" in adata.uns:
            old_colors = adata.uns[f"{key}_colors"]
        new_colors = []
        for group in adata.obs[key_added].cat.categories:
            if group in map_colors:
                new_colors.append(map_colors[group])
            elif group in old_categories and old_colors is not None:
                new_colors.append(old_colors[old_categories.get_loc(group)])
            else:
                raise ValueError(f"You didn't specify a color for {group}.")
        adata.uns[f"{key_added}_colors"] = new_colors

    # map_colors is not passed
    elif f"{key}_colors" in adata.uns:
        old_colors = adata.uns[f"{key}_colors"]
        inverse_map_groups = {g: [] for g in new_categories}
        for old_group in old_categories:
            inverse_map_groups[map_groups[old_group]].append(old_group)
        new_colors = []
        for group in new_categories:
            # take the largest of the old groups
            old_group = (
                adata.obs[key][adata.obs[key].isin(inverse_map_groups[group])]
                .value_counts()
                .index[0]
            )
            new_colors.append(old_colors[old_categories.get_loc(old_group)])
        adata.uns[f"{key_added}_colors"] = new_colors


# TODO: Add docstrings
def cutoff_small_velocities(
    adata, vkey="velocity", key_added="velocity_cut", frac_of_max=0.5, use_raw=False
):
    """TODO."""
    x = adata.layers["spliced"] if use_raw else adata.layers["Ms"]
    y = adata.layers["unspliced"] if use_raw else adata.layers["Mu"]

    x_max = x.max(0).A[0] if issparse(x) else x.max(0)
    y_max = y.max(0).A[0] if issparse(y) else y.max(0)

    xy_norm = x / np.clip(x_max, 1e-3, None) + y / np.clip(y_max, 1e-3, None)
    W = xy_norm >= np.percentile(xy_norm, 98, axis=0) * frac_of_max

    adata.layers[key_added] = csr_matrix(W).multiply(adata.layers[vkey]).tocsr()

    from .velocity_embedding import velocity_embedding
    from .velocity_graph import velocity_graph

    velocity_graph(adata, vkey=key_added, approx=True)
    velocity_embedding(adata, vkey=key_added)


# TODO: Add docstrings
def make_unique_list(key, allow_array=False):
    """TODO."""
    from pandas import Index, unique

    if isinstance(key, Index):
        key = key.tolist()
    is_list = (
        isinstance(key, (list, tuple, np.record))
        if allow_array
        else isinstance(key, (list, tuple, np.ndarray, np.record))
    )
    is_list_of_str = is_list and all(isinstance(item, str) for item in key)
    return (
        unique(key) if is_list_of_str else key if is_list and len(key) < 20 else [key]
    )


# TODO: Finish docstrings
def test_bimodality(x, bins=30, kde=True, plot=False):
    """Test for bimodal distribution."""
    from scipy.stats import gaussian_kde, norm

    lb, ub = np.min(x), np.percentile(x, 99.9)
    grid = np.linspace(lb, ub if ub <= lb else np.max(x), bins)
    kde_grid = (
        gaussian_kde(x)(grid) if kde else np.histogram(x, bins=grid, density=True)[0]
    )

    idx = int(bins / 2) - 2
    end = idx + 4
    idx += np.argmin(kde_grid[idx:end])

    peak_0 = kde_grid[:idx].argmax()
    peak_1 = kde_grid[idx:].argmax()
    kde_peak = kde_grid[idx:][
        peak_1
    ]  # min(kde_grid[:idx][peak_0], kde_grid[idx:][peak_1])
    kde_mid = kde_grid[idx:].mean()  # kde_grid[idx]

    t_stat = (kde_peak - kde_mid) / np.clip(np.std(kde_grid) / np.sqrt(bins), 1, None)
    p_val = norm.sf(t_stat)

    grid_0 = grid[:idx]
    grid_1 = grid[idx:]
    means = [
        (grid_0[peak_0] + grid_0[min(peak_0 + 1, len(grid_0) - 1)]) / 2,
        (grid_1[peak_1] + grid_1[min(peak_1 + 1, len(grid_1) - 1)]) / 2,
    ]

    if plot:
        color = "grey"
        if kde:
            pl.plot(grid, kde_grid, color=color)
            pl.fill_between(grid, 0, kde_grid, alpha=0.4, color=color)
        else:
            pl.hist(x, bins=grid, alpha=0.4, density=True, color=color)
        pl.axvline(means[0], color=color)
        pl.axvline(means[1], color=color)
        pl.axhline(kde_mid, alpha=0.2, linestyle="--", color=color)
        pl.show()

    return t_stat, p_val, means  # ~ t_test (reject unimodality if t_stat > 3)


# TODO: Add docstrings
def random_subsample(adata, fraction=0.1, return_subset=False, copy=False):
    """TODO."""
    adata_sub = adata.copy() if copy else adata
    p, size = fraction, adata.n_obs
    subset = np.random.choice([True, False], size=size, p=[p, 1 - p])
    adata_sub._inplace_subset_obs(subset)
    return adata_sub if copy else subset if return_subset else None


# TODO: Add docstrings
def get_duplicates(array):
    """TODO."""
    from collections import Counter

    return np.array([item for (item, count) in Counter(array).items() if count > 1])


# TODO: Add docstrings
def corrcoef(x, y, mode="pearsons"):
    """TODO."""
    from scipy.stats import pearsonr, spearmanr

    corr, _ = spearmanr(x, y) if mode == "spearmans" else pearsonr(x, y)
    return corr


def vcorrcoef(X, y, mode="pearsons", axis=-1):
    """Pearsons/Spearmans correlation coefficients.
    Use Pearsons / Spearmans to test for linear / monotonic relationship.
    Arguments
    ----------
    X: `np.ndarray`
        Data vector or matrix
    y: `np.ndarray`
        Data vector or matrix
    mode: 'pearsons' or 'spearmans' (default: `'pearsons'`)
        Which correlation metric to use.
    """
    if issparse(X):
        X = np.array(X.A)
    if issparse(y):
        y = np.array(y.A)
    if axis == 0:
        if X.ndim > 1:
            X = np.array(X.T)
        if y.ndim > 1:
            y = np.array(y.T)
    if X.shape[axis] != y.shape[axis]:
        X = X.T
    if mode in {"spearmans", "spearman"}:
        from scipy.stats.stats import rankdata

        X = np.apply_along_axis(rankdata, axis=-1, arr=X)
        y = np.apply_along_axis(rankdata, axis=-1, arr=y)
    Xm = np.array(X - (np.nanmean(X, -1)[:, None] if X.ndim > 1 else np.nanmean(X, -1)))
    ym = np.array(y - (np.nanmean(y, -1)[:, None] if y.ndim > 1 else np.nanmean(y, -1)))
    corr = np.nansum(Xm * ym, -1) / np.sqrt(
        np.nansum(Xm**2, -1) * np.nansum(ym**2, -1)
    )
    return corr


# TODO: Add docstrings
def isin(x, y):
    """TODO."""
    return np.array(pd.DataFrame(x).isin(y)).flatten()


# TODO: Add docstrings
def indices_to_bool(indices, n):
    """TODO."""
    return isin(np.arange(n), indices)


# TODO: Add docstrings
def convolve(adata, x):
    """TODO."""
    from scvelo.preprocessing.neighbors import get_connectivities

    conn = get_connectivities(adata)
    if isinstance(x, str) and x in adata.layers.keys():
        x = adata.layers[x]
    if x.ndim == 1:
        return conn.dot(x)
    idx_valid = ~np.isnan(x.sum(0))
    Y = np.ones(x.shape) * np.nan
    Y[:, idx_valid] = conn.dot(x[:, idx_valid])
    return Y


# TODO: Finish docstrings
def get_extrapolated_state(adata, vkey="velocity", dt=1, use_raw=None, dropna=True):
    """Get extrapolated cell state."""
    S = adata.layers["spliced" if use_raw else "Ms"]
    if dropna:
        St = S + dt * adata.layers[vkey]
        St = St[:, np.isfinite(np.sum(St, 0))]
    else:
        St = S + dt * np.nan_to_num(adata.layers[vkey])
    return St


# TODO: Add docstrings
# TODO: Generalize to use arbitrary modality i.e., not only layers
def get_plasticity_score(adata, modality="Ms"):
    """TODO."""
    idx_top_genes = np.argsort(adata.var["gene_count_corr"].values)[::-1][:200]
    Ms = np.array(adata.layers[modality][:, idx_top_genes])
    return scale(np.mean(Ms / np.max(Ms, axis=0), axis=1))



##################
import warnings

import numpy as np

from scvelo import logging as logg
from scvelo import settings
from scvelo.core import LinearRegression
from scvelo.preprocessing.moments import moments, second_order_moments
# TODO: Add docstrings
def get_weight(x, y=None, perc=95):
    """TODO."""
    xy_norm = np.array(x.A if issparse(x) else x)
    if y is not None:
        if issparse(y):
            y = y.A
        xy_norm = xy_norm / np.clip(np.max(xy_norm, axis=0), 1e-3, None)
        xy_norm += y / np.clip(np.max(y, axis=0), 1e-3, None)
    if isinstance(perc, numbers.Number):
        weights = xy_norm >= np.percentile(xy_norm, perc, axis=0)
    else:
        lb, ub = np.percentile(xy_norm, perc, axis=0)
        weights = (xy_norm <= lb) | (xy_norm >= ub)
    return weights


# TODO: Add docstrings
def leastsq_NxN(x, y, fit_offset=False, perc=None, constraint_positive_offset=True):
    """Solves least squares X*b=Y for b."""
    warnings.warn(
        "`leastsq_NxN` is deprecated since scVelo v0.2.4 and will be removed in a "
        "future version. Please use `LinearRegression` from `scvelo/core/` instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if perc is not None:
        if not fit_offset and isinstance(perc, (list, tuple)):
            perc = perc[1]
        weights = csr_matrix(get_weight(x, y, perc=perc)).astype(bool)
        x, y = weights.multiply(x).tocsr(), weights.multiply(y).tocsr()
    else:
        weights = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xx_ = prod_sum(x, x, axis=0)
        xy_ = prod_sum(x, y, axis=0)

        if fit_offset:
            n_obs = x.shape[0] if weights is None else sum(weights, axis=0)
            x_ = sum(x, axis=0) / n_obs
            y_ = sum(y, axis=0) / n_obs
            gamma = (xy_ / n_obs - x_ * y_) / (xx_ / n_obs - x_**2)
            offset = y_ - gamma * x_

            # fix negative offsets:
            if constraint_positive_offset:
                idx = offset < 0
                if gamma.ndim > 0:
                    gamma[idx] = xy_[idx] / xx_[idx]
                else:
                    gamma = xy_ / xx_
                offset = np.clip(offset, 0, None)
        else:
            gamma = xy_ / xx_
            offset = np.zeros(x.shape[1]) if x.ndim > 1 else 0
    nans_offset, nans_gamma = np.isnan(offset), np.isnan(gamma)
    if np.any([nans_offset, nans_gamma]):
        offset[np.isnan(offset)], gamma[np.isnan(gamma)] = 0, 0
    return offset, gamma


leastsq = leastsq_NxN


# TODO: Add docstrings
def optimize_NxN(x, y, fit_offset=False, perc=None):
    """Just to compare with closed-form solution."""
    if perc is not None:
        if not fit_offset and isinstance(perc, (list, tuple)):
            perc = perc[1]
        weights = get_weight(x, y, perc).astype(bool)
        if issparse(weights):
            weights = weights.A
    else:
        weights = None

    x, y = x.astype(np.float64), y.astype(np.float64)

    n_vars = x.shape[1]
    offset, gamma = np.zeros(n_vars), np.zeros(n_vars)

    for i in range(n_vars):
        xi = x[:, i] if weights is None else x[:, i][weights[:, i]]
        yi = y[:, i] if weights is None else y[:, i][weights[:, i]]

        def _loss_fun(m, x, y):
            if m.size > 1:
                return np.sum((-y + x * m[1] + m[0]) ** 2)
            else:
                return np.sum((-y + x * m) ** 2)

        if fit_offset:
            offset[i], gamma[i] = minimize(
                _loss_fun,
                args=(xi, yi),
                method="L-BFGS-B",
                x0=np.array([0, 0.1]),
                bounds=[(0, None), (None, None)],
            ).x
        else:
            gamma[i] = minimize(
                _loss_fun, args=(xi, yi), x0=np.array([0.1]), method="L-BFGS-B"
            ).x
    offset[np.isnan(offset)], gamma[np.isnan(gamma)] = 0, 0
    return offset, gamma


# TODO: Add docstrings
def leastsq_generalized(
    x,
    y,
    x2,
    y2,
    res_std=None,
    res2_std=None,
    fit_offset=False,
    fit_offset2=False,
    perc=None,
):
    """Solution to the 2-dim generalized least squares: gamma = inv(X'QX)X'QY"""
    if perc is not None:
        if not fit_offset and isinstance(perc, (list, tuple)):
            perc = perc[1]
        weights = csr_matrix(
            get_weight(x, y, perc=perc) | get_weight(x, perc=perc)
        ).astype(bool)
        x, y = weights.multiply(x).tocsr(), weights.multiply(y).tocsr()
        # x2, y2 = weights.multiply(x2).tocsr(), weights.multiply(y2).tocsr()

    n_obs, n_var = x.shape
    offset, offset_ss = (
        np.zeros(n_var, dtype="float32"),
        np.zeros(n_var, dtype="float32"),
    )
    gamma = np.ones(n_var, dtype="float32")

    if (res_std is None) or (res2_std is None):
        res_std, res2_std = np.ones(n_var), np.ones(n_var)
    ones, zeros = np.ones(n_obs), np.zeros(n_obs)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x, y = (
            np.vstack((make_dense(x) / res_std, x2 / res2_std)),
            np.vstack((make_dense(y) / res_std, y2 / res2_std)),
        )

    if fit_offset and fit_offset2:
        for i in range(n_var):
            A = np.c_[
                np.vstack(
                    (np.c_[ones / res_std[i], zeros], np.c_[zeros, ones / res2_std[i]])
                ),
                x[:, i],
            ]
            offset[i], offset_ss[i], gamma[i] = np.linalg.pinv(A.T.dot(A)).dot(
                A.T.dot(y[:, i])
            )
    elif fit_offset:
        for i in range(n_var):
            A = np.c_[np.hstack((ones / res_std[i], zeros)), x[:, i]]
            offset[i], gamma[i] = np.linalg.pinv(A.T.dot(A)).dot(A.T.dot(y[:, i]))
    elif fit_offset2:
        for i in range(n_var):
            A = np.c_[np.hstack((zeros, ones / res2_std[i])), x[:, i]]
            offset_ss[i], gamma[i] = np.linalg.pinv(A.T.dot(A)).dot(A.T.dot(y[:, i]))
    else:
        for i in range(n_var):
            A = np.c_[x[:, i]]
            gamma[i] = np.linalg.pinv(A.T.dot(A)).dot(A.T.dot(y[:, i]))

    offset[np.isnan(offset)] = 0
    offset_ss[np.isnan(offset_ss)] = 0
    gamma[np.isnan(gamma)] = 0

    return offset, offset_ss, gamma


# TODO: Add docstrings
def maximum_likelihood(Ms, Mu, Mus, Mss, fit_offset=False, fit_offset2=False):
    """Maximizing the log likelihood using weights according to empirical Bayes."""
    n_obs, n_var = Ms.shape
    offset = np.zeros(n_var, dtype="float32")
    offset_ss = np.zeros(n_var, dtype="float32")
    gamma = np.ones(n_var, dtype="float32")

    def sse(A, data, b):
        sigma = (A.dot(data) - b).std(1)
        return np.log(sigma).sum()

    def _loss_fun(m, offset_type: int):
        if m.size == 3:
            return sse(
                np.array([[1, -m[2], 0, 0], [1, m[2], 2, -2 * m[2]]]),
                data,
                b=np.array(m[0], m[1]),
            )
        elif offset_type == 1:
            return sse(
                np.array([[1, -m[1], 0, 0], [1, m[1], 2, -2 * m[1]]]),
                data,
                b=np.array(m[0], 0),
            )
        elif offset_type == 2:
            sse(
                np.array([[1, -m[1], 0, 0], [1, m[1], 2, -2 * m[1]]]),
                data,
                b=np.array(0, m[0]),
            )
        else:
            sse(np.array([[1, -m, 0, 0], [1, m, 2, -2 * m]]), data, b=0)

    if fit_offset and fit_offset2:
        for i in range(n_var):
            data = np.vstack((Mu[:, i], Ms[:, i], Mus[:, i], Mss[:, i]))
            offset[i], offset_ss[i], gamma[i] = minimize(
                _loss_fun,
                args=(-1),
                x0=(1e-4, 1e-4, 1),
                method="L-BFGS-B",
            ).x
    elif fit_offset:
        for i in range(n_var):
            data = np.vstack((Mu[:, i], Ms[:, i], Mus[:, i], Mss[:, i]))
            offset[i], gamma[i] = minimize(
                _loss_fun,
                args=(1),
                x0=(1e-4, 1),
                method="L-BFGS-B",
            ).x
    elif fit_offset2:
        for i in range(n_var):
            data = np.vstack((Mu[:, i], Ms[:, i], Mus[:, i], Mss[:, i]))
            offset_ss[i], gamma[i] = minimize(
                _loss_fun,
                args=(2),
                x0=(1e-4, 1),
                method="L-BFGS-B",
            ).x
    else:
        for i in range(n_var):
            data = np.vstack((Mu[:, i], Ms[:, i], Mus[:, i], Mss[:, i]))
            gamma[i] = minimize(
                _loss_fun,
                args=(-1),
                x0=gamma[i],
                method="L-BFGS-B",
            ).x
    return offset, offset_ss, gamma

warnings.simplefilter(action="ignore", category=FutureWarning)


# TODO: Add docstrings
class Velocity:
    """TODO."""

    def __init__(
        self,
        adata=None,
        Ms=None,
        Mu=None,
        groups_for_fit=None,
        groupby=None,
        residual=None,
        constrain_ratio=None,
        min_r2=0.01,
        min_ratio=0.01,
        use_highly_variable=True,
        r2_adjusted=True,
        use_raw=False,
    ):
        self._adata = adata
        self._Ms, self._Mu = Ms, Mu
        if Ms is None:
            self._Ms = adata.layers["spliced"] if use_raw else adata.layers["Ms"]
        if Mu is None:
            self._Mu = adata.layers["unspliced"] if use_raw else adata.layers["Mu"]
        self._Ms, self._Mu = make_dense(self._Ms), make_dense(self._Mu)

        n_obs, n_vars = self._Ms.shape
        self._residual, self._residual2 = residual, None
        self._offset = np.zeros(n_vars, dtype=np.float32)
        self._offset2 = np.zeros(n_vars, dtype=np.float32)
        self._gamma = np.zeros(n_vars, dtype=np.float32)
        self._qreg_ratio = np.zeros(n_vars, dtype=np.float32)
        self._r2 = np.zeros(n_vars, dtype=np.float32)
        self._beta = np.ones(n_vars, dtype=np.float32)
        self._velocity_genes = np.ones(n_vars, dtype=bool)
        self._groups_for_fit = groups_to_bool(adata, groups_for_fit, groupby)
        self._constrain_ratio = constrain_ratio
        self._r2_adjusted = r2_adjusted
        self._min_r2 = min_r2
        self._min_ratio = min_ratio
        self._highly_variable = None
        if use_highly_variable is not None and adata is not None:
            if "highly_variable" in adata.var.keys():
                self._highly_variable = adata.var["highly_variable"]

    # TODO: Add docstrings
    def compute_deterministic(self, fit_offset=False, perc=None):
        """TODO."""
        subset = self._groups_for_fit
        Ms = self._Ms if subset is None else self._Ms[subset]
        Mu = self._Mu if subset is None else self._Mu[subset]

        lr = LinearRegression(fit_intercept=fit_offset, percentile=perc)
        lr.fit(Ms, Mu)
        self._offset = lr.intercept_
        self._gamma = lr.coef_

        if self._constrain_ratio is not None:
            if np.size(self._constrain_ratio) < 2:
                self._constrain_ratio = [None, self._constrain_ratio]
            cr = self._constrain_ratio
            self._gamma = np.clip(self._gamma, cr[0], cr[1])

        self._residual = self._Mu - self._gamma * self._Ms
        if fit_offset:
            self._residual -= self._offset
        _residual = self._residual

        # velocity genes
        if self._r2_adjusted:
            lr = LinearRegression(fit_intercept=fit_offset)
            lr.fit(Ms, Mu)
            _offset = lr.intercept_
            _gamma = lr.coef_

            _residual = self._Mu - _gamma * self._Ms
            if fit_offset:
                _residual -= _offset

        self._qreg_ratio = np.array(self._gamma)  # quantile regression ratio

        self._r2 = R_squared(_residual, total=self._Mu - self._Mu.mean(0))
        self._velocity_genes = (
            (self._r2 > self._min_r2)
            & (self._gamma > self._min_ratio)
            & (np.max(self._Ms > 0, 0) > 0)
            & (np.max(self._Mu > 0, 0) > 0)
        )

        if self._highly_variable is not None:
            self._velocity_genes &= self._highly_variable

        if np.sum(self._velocity_genes) < 2:
            min_r2 = np.percentile(self._r2, 80)
            self._velocity_genes = self._r2 > min_r2
            min_r2 = np.round(min_r2, 4)
            logg.warn(
                f"You seem to have very low signal in splicing dynamics.\n"
                f"The correlation threshold has been reduced to {min_r2}.\n"
                f"Please be cautious when interpreting results."
            )

    # TODO: Add docstrings
    def compute_stochastic(
        self, fit_offset=False, fit_offset2=False, mode=None, perc=None
    ):
        """TODO."""
        if self._residual is None:
            self.compute_deterministic(fit_offset=fit_offset, perc=perc)

        idx = np.ones(self._velocity_genes.shape, dtype=bool)
        if np.any(self._velocity_genes):
            idx = self._velocity_genes
        is_subset = len(set(idx)) > 1

        _adata = self._adata[:, idx] if is_subset else self._adata
        _Ms = self._Ms[:, idx] if is_subset else self._Ms
        _Mu = self._Mu[:, idx] if is_subset else self._Mu
        _residual = self._residual[:, idx] if is_subset else self._residual

        _Mss, _Mus = second_order_moments(_adata)

        var_ss = 2 * _Mss - _Ms
        cov_us = 2 * _Mus + _Mu

        lr = LinearRegression(fit_intercept=fit_offset2)
        lr.fit(var_ss, cov_us)
        _offset2 = lr.intercept_
        _gamma2 = lr.coef_

        # initialize covariance matrix
        res_std = _residual.std(0)
        res2_std = (cov_us - _gamma2 * var_ss - _offset2).std(0)

        # solve multiple regression
        self._offset[idx], self._offset2[idx], self._gamma[idx] = (
            maximum_likelihood(_Ms, _Mu, _Mus, _Mss, fit_offset, fit_offset2)
            if mode == "bayes"
            else leastsq_generalized(
                _Ms,
                _Mu,
                var_ss,
                cov_us,
                res_std,
                res2_std,
                fit_offset,
                fit_offset2,
                perc,
            )
        )

        self._residual = self._Mu - self._gamma * self._Ms
        if fit_offset:
            self._residual -= self._offset

        _residual2 = (cov_us - 2 * _Ms * _Mu) - self._gamma[idx] * (
            var_ss - 2 * _Ms**2
        )
        if fit_offset:
            _residual2 += 2 * self._offset[idx] * _Ms
        if fit_offset2:
            _residual2 -= self._offset2[idx]
        if is_subset:
            self._residual2 = np.zeros(self._Ms.shape, dtype=np.float32)
            self._residual2[:, idx] = _residual2
        else:
            self._residual2 = _residual2

    # TODO: Add docstrings
    def get_pars(self):
        """TODO."""
        return (
            self._offset,
            self._offset2,
            self._beta,
            self._gamma,
            self._qreg_ratio,
            self._r2,
            self._velocity_genes,
        )

    # TODO: Add docstrings
    def get_pars_names(self):
        """TODO."""
        return [
            "_offset",
            "_offset2",
            "_beta",
            "_gamma",
            "_qreg_ratio",
            "_r2",
            "_genes",
        ]


# TODO: Add docstrings
def write_residuals(adata, vkey, residual=None, cell_subset=None):
    """TODO."""
    if residual is not None:
        if cell_subset is None:
            adata.layers[vkey] = residual
        else:
            if vkey not in adata.layers.keys():
                adata.layers[vkey] = np.zeros(adata.shape, dtype=np.float32)
            adata.layers[vkey][cell_subset] = residual


# TODO: Add docstrings
def write_pars(adata, vkey, pars, pars_names, add_key=None):
    """TODO."""
    for i, key in enumerate(pars_names):
        key = f"{vkey}{key}_{add_key}" if add_key is not None else f"{vkey}{key}"
        if len(set(pars[i])) > 1:
            adata.var[key] = pars[i]
        elif key in adata.var.keys():
            del adata.var[key]


def velocity(
    data,
    vkey="velocity",
    mode="stochastic",
    fit_offset=False,
    fit_offset2=False,
    filter_genes=False,
    groups=None,
    groupby=None,
    groups_for_fit=None,
    constrain_ratio=None,
    use_raw=False,
    use_latent_time=None,
    perc=None,
    min_r2=1e-2,
    min_likelihood=1e-3,
    r2_adjusted=None,
    use_highly_variable=True,
    diff_kinetics=None,
    copy=False,
    **kwargs,
):
    """Estimates velocities in a gene-specific manner.
    The steady-state model :cite:p:`LaManno18` determines velocities by quantifying how
    observations deviate from a presumed steady-state equilibrium ratio of unspliced to
    spliced mRNA levels. This steady-state ratio is obtained by performing a linear
    regression restricting the input data to the extreme quantiles. By including
    second-order moments, the stochastic model :cite:p:`Bergen20` exploits not only the balance
    of unspliced to spliced mRNA levels but also their covariation. By contrast, the
    likelihood-based dynamical model :cite:p:`Bergen20` solves the full splicing kinetics and
    generalizes RNA velocity estimation to transient systems. It is also capable of
    capturing non-observed steady states.
    .. image:: https://user-images.githubusercontent.com/31883718/69636491-ff057100-1056-11ea-90b7-d04098112ce1.png
    Arguments
    ---------
    data: :class:`~anndata.AnnData`
        Annotated data matrix.
    vkey: `str` (default: `'velocity'`)
        Name under which to refer to the computed velocities
        for `velocity_graph` and `velocity_embedding`.
    mode: `'deterministic'`, `'stochastic'` or `'dynamical'` (default: `'stochastic'`)
        Whether to run the estimation using the steady-state/deterministic,
        stochastic or dynamical model of transcriptional dynamics.
        The dynamical model requires to run `tl.recover_dynamics` first.
    fit_offset: `bool` (default: `False`)
        Whether to fit with offset for first order moment dynamics.
    fit_offset2: `bool`, (default: `False`)
        Whether to fit with offset for second order moment dynamics.
    filter_genes: `bool` (default: `True`)
        Whether to remove genes that are not used for further velocity analysis.
    groups: `str`, `list` (default: `None`)
        Subset of groups, e.g. [‘g1’, ‘g2’, ‘g3’],
        to which velocity analysis shall be restricted.
    groupby: `str`, `list` or `np.ndarray` (default: `None`)
        Key of observations grouping to consider.
    groups_for_fit: `str`, `list` or `np.ndarray` (default: `None`)
        Subset of groups, e.g. [‘g1’, ‘g2’, ‘g3’],
        to which steady-state fitting shall be restricted.
    constrain_ratio: `float` or tuple of type `float` or None: (default: `None`)
        Bounds for the steady-state ratio.
    use_raw: `bool` (default: `False`)
        Whether to use raw data for estimation.
    use_latent_time: `bool`or `None` (default: `None`)
        Whether to use latent time as a regularization for velocity estimation.
    perc: `float` (default: `[5, 95]`)
        Percentile, e.g. 98, for extreme quantile fit.
    min_r2: `float` (default: 0.01)
        Minimum threshold for coefficient of determination
    min_likelihood: `float` (default: `None`)
        Minimal likelihood for velocity genes to fit the model on.
    r2_adjusted: `bool` (default: `None`)
        Whether to compute coefficient of determination
        on full data fit (adjusted) or extreme quantile fit (None)
    use_highly_variable: `bool` (default: True)
        Whether to use highly variable genes only, stored in .var['highly_variable'].
    copy: `bool` (default: `False`)
        Return a copy instead of writing to `adata`.
    Returns
    -------
    velocity: `.layers`
        velocity vectors for each individual cell
    velocity_genes, velocity_beta, velocity_gamma, velocity_r2: `.var`
        parameters
    """
    if perc is None:
        perc = [5, 95]
    adata = data.copy() if copy else data
    if not use_raw and "Ms" not in adata.layers.keys():
        moments(adata)

    logg.info("computing velocities", r=True)

    strings_to_categoricals(adata)

    if mode is None or (mode == "dynamical" and "fit_alpha" not in adata.var.keys()):
        mode = "stochastic"
        logg.warn(
            "Falling back to stochastic model. "
            "For the dynamical model run tl.recover_dynamics first."
        )

    if mode in {"dynamical", "dynamical_residuals"}:
        from .dynamical_model_utils import get_divergence, get_reads, get_vars

        gene_subset = ~np.isnan(adata.var["fit_alpha"].values)
        vdata = adata[:, gene_subset]
        alpha, beta, gamma, scaling, t_ = get_vars(vdata)

        connect = not adata.uns["recover_dynamics"]["use_raw"]
        kwargs_ = {
            "kernel_width": None,
            "normalized": True,
            "var_scale": True,
            "reg_par": None,
            "min_confidence": 1e-2,
            "constraint_time_increments": False,
            "fit_steady_states": True,
            "fit_basal_transcription": None,
            "use_connectivities": connect,
            "time_connectivities": connect,
            "use_latent_time": use_latent_time,
        }
        kwargs_.update(adata.uns["recover_dynamics"])
        kwargs_.update(**kwargs)

        if "residuals" in mode:
            u, s = get_reads(vdata, use_raw=adata.uns["recover_dynamics"]["use_raw"])
            if kwargs_["fit_basal_transcription"]:
                u, s = u - adata.var["fit_u0"], s - adata.var["fit_s0"]
            o = vdata.layers["fit_t"] < t_
            vt = u * beta - s * gamma  # ds/dt
            wt = (alpha * o - beta * u) * scaling  # du/dt
        else:
            vt, wt = get_divergence(vdata, mode="velocity", **kwargs_)

        vgenes = adata.var.fit_likelihood > min_likelihood
        if min_r2 is not None:
            if "fit_r2" not in adata.var.keys():
                velo = Velocity(
                    adata,
                    groups_for_fit=groups_for_fit,
                    groupby=groupby,
                    constrain_ratio=constrain_ratio,
                    min_r2=min_r2,
                    use_highly_variable=use_highly_variable,
                    use_raw=use_raw,
                )
                velo.compute_deterministic(fit_offset=fit_offset, perc=perc)
                adata.var["fit_r2"] = velo._r2
            vgenes &= adata.var.fit_r2 > min_r2

        lb, ub = np.nanpercentile(adata.var.fit_scaling, [10, 90])
        vgenes = (
            vgenes
            & (adata.var.fit_scaling > np.min([lb, 0.03]))
            & (adata.var.fit_scaling < np.max([ub, 3]))
        )

        adata.var[f"{vkey}_genes"] = vgenes

        adata.layers[vkey] = np.ones(adata.shape) * np.nan
        adata.layers[vkey][:, gene_subset] = vt

        adata.layers[f"{vkey}_u"] = np.ones(adata.shape) * np.nan
        adata.layers[f"{vkey}_u"][:, gene_subset] = wt

        if filter_genes and len(set(vgenes)) > 1:
            adata._inplace_subset_var(vgenes)

    elif mode in {"steady_state", "deterministic", "stochastic"}:
        categories = (
            adata.obs[groupby].cat.categories
            if groupby is not None and groups is None and groups_for_fit is None
            else [None]
        )

        for cat in categories:
            groups = cat if cat is not None else groups

            cell_subset = groups_to_bool(adata, groups, groupby)
            _adata = adata if groups is None else adata[cell_subset]
            velo = Velocity(
                _adata,
                groups_for_fit=groups_for_fit,
                groupby=groupby,
                constrain_ratio=constrain_ratio,
                min_r2=min_r2,
                r2_adjusted=r2_adjusted,
                use_highly_variable=use_highly_variable,
                use_raw=use_raw,
            )
            velo.compute_deterministic(fit_offset=fit_offset, perc=perc)

            if mode == "stochastic":
                if filter_genes and len(set(velo._velocity_genes)) > 1:
                    adata._inplace_subset_var(velo._velocity_genes)
                    residual = velo._residual[:, velo._velocity_genes]
                    _adata = adata if groups is None else adata[cell_subset]
                    velo = Velocity(
                        _adata,
                        residual=residual,
                        groups_for_fit=groups_for_fit,
                        groupby=groupby,
                        constrain_ratio=constrain_ratio,
                        use_highly_variable=use_highly_variable,
                    )
                velo.compute_stochastic(fit_offset, fit_offset2, mode, perc=perc)

            write_residuals(adata, vkey, velo._residual, cell_subset)
            write_residuals(adata, f"variance_{vkey}", velo._residual2, cell_subset)
            write_pars(adata, vkey, velo.get_pars(), velo.get_pars_names(), add_key=cat)

            if filter_genes and len(set(velo._velocity_genes)) > 1:
                adata._inplace_subset_var(velo._velocity_genes)

    else:
        raise ValueError(
            "Mode can only be one of these: deterministic, stochastic or dynamical."
        )

    if f"{vkey}_genes" in adata.var.keys() and np.sum(adata.var[f"{vkey}_genes"]) < 10:
        logg.warn(
            "Too few genes are selected as velocity genes. "
            "Consider setting a lower threshold for min_r2 or min_likelihood."
        )

    if diff_kinetics:
        if not isinstance(diff_kinetics, str):
            diff_kinetics = "fit_diff_kinetics"
        if diff_kinetics in adata.var.keys():
            if diff_kinetics in adata.uns["recover_dynamics"]:
                groupby = adata.uns["recover_dynamics"]["fit_diff_kinetics"]
            else:
                groupby = "clusters"
            clusters = adata.obs[groupby]
            for i, v in enumerate(np.array(adata.var[diff_kinetics].values, dtype=str)):
                if len(v) > 0 and v != "nan":
                    idx = 1 - clusters.isin([a.strip() for a in v.split(",")])
                    adata.layers[vkey][:, i] *= idx
                    if mode == "dynamical":
                        adata.layers[f"{vkey}_u"][:, i] *= idx

    adata.uns[f"{vkey}_params"] = {"mode": mode, "fit_offset": fit_offset, "perc": perc}

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added \n"
        f"    '{vkey}', velocity vectors for each individual cell (adata.layers)"
    )

    return adata if copy else None


def velocity_genes(
    data,
    vkey="velocity",
    min_r2=0.01,
    min_ratio=0.01,
    use_highly_variable=True,
    copy=False,
):
    """Estimates velocities in a gene-specific manner.
    Arguments
    ---------
    data: :class:`~anndata.AnnData`
        Annotated data matrix.
    vkey: `str` (default: `'velocity'`)
        Name under which to refer to the computed velocities.
    min_r2: `float` (default: 0.01)
        Minimum threshold for coefficient of determination
    min_ratio: `float` (default: 0.01)
        Minimum threshold for quantile regression un/spliced ratio.
    use_highly_variable: `bool` (default: True)
        Whether to use highly variable genes only, stored in .var['highly_variable'].
    copy: `bool` (default: `False`)
        Return a copy instead of writing to `adata`.
    Returns
    -------
    Updates `adata` attributes
    velocity_genes: `.var`
        genes to be used for further velocity analysis (velocity graph and embedding)
    """
    adata = data.copy() if copy else data
    if f"{vkey}_genes" not in adata.var.keys():
        velocity(adata, vkey)
    vgenes = np.ones(adata.n_vars, dtype=bool)

    if "Ms" in adata.layers.keys() and "Mu" in adata.layers.keys():
        vgenes &= np.max(adata.layers["Ms"] > 0, 0) > 0
        vgenes &= np.max(adata.layers["Mu"] > 0, 0) > 0

    if min_r2 is not None and f"{vkey}_r2" in adata.var.keys():
        vgenes &= adata.var[f"{vkey}_r2"] > min_r2

    if min_ratio is not None and f"{vkey}_qreg_ratio" in adata.var.keys():
        vgenes &= adata.var[f"{vkey}_qreg_ratio"] > min_ratio

    if use_highly_variable and "highly_variable" in adata.var.keys():
        vgenes &= adata.var["highly_variable"].values

    if np.sum(vgenes) < 2:
        logg.warn(
            "You seem to have very low signal in splicing dynamics.\n"
            "Consider reducing the thresholds and be cautious with interpretations.\n"
        )

    adata.var[f"{vkey}_genes"] = vgenes

    logg.info("Number of obtained velocity_genes:", np.sum(adata.var[f"{vkey}_genes"]))

    return adata if copy else None
















# TODO: Add docstrings
def vals_to_csr(vals, rows, cols, shape, split_negative=False):
    """TODO."""
    graph = coo_matrix((vals, (rows, cols)), shape=shape)

    if split_negative:
        graph_neg = graph.copy()

        graph.data = np.clip(graph.data, 0, 1)
        graph_neg.data = np.clip(graph_neg.data, -1, 0)

        graph.eliminate_zeros()
        graph_neg.eliminate_zeros()

        return graph.tocsr(), graph_neg.tocsr()

    else:
        return graph.tocsr()


# TODO: Add docstrings
class VelocityGraph:
    """TODO."""

    def __init__(
        self,
        adata,
        vkey="velocity",
        xkey="Ms",
        tkey=None,
        basis=None,
        n_neighbors=None,
        sqrt_transform=None,
        n_recurse_neighbors=None,
        random_neighbors_at_max=None,
        gene_subset=None,
        approx=None,
        report=False,
        compute_uncertainties=None,
        mode_neighbors="distances",
    ):

        subset = np.ones(adata.n_vars, bool)
        if gene_subset is not None:
            var_names_subset = adata.var_names.isin(gene_subset)
            subset &= var_names_subset if len(var_names_subset) > 0 else gene_subset
        elif f"{vkey}_genes" in adata.var.keys():
            subset &= np.array(adata.var[f"{vkey}_genes"].values, dtype=bool)

        xkey = xkey if xkey in adata.layers.keys() else "spliced"

        X = np.array(
            adata.layers[xkey].A[:, subset]
            if issparse(adata.layers[xkey])
            else adata.layers[xkey][:, subset]
        )
        V = np.array(
            adata.layers[vkey].A[:, subset]
            if issparse(adata.layers[vkey])
            else adata.layers[vkey][:, subset]
        )

        nans = np.isnan(np.sum(V, axis=0))
        if np.any(nans):
            X = X[:, ~nans]
            V = V[:, ~nans]

        if approx is True and X.shape[1] > 100:
            X_pca, PCs, _, _ = pca(X, n_comps=30, svd_solver="arpack", return_info=True)
            self.X = np.array(X_pca, dtype=np.float32)
            self.V = (V - V.mean(0)).dot(PCs.T)
            self.V[V.sum(1) == 0] = 0
        else:
            self.X = np.array(X, dtype=np.float32)
            self.V = np.array(V, dtype=np.float32)
        self.V_raw = np.array(self.V)

        self.sqrt_transform = sqrt_transform
        uns_key = f"{vkey}_params"
        if self.sqrt_transform is None:
            if uns_key in adata.uns.keys() and "mode" in adata.uns[uns_key]:
                self.sqrt_transform = adata.uns[uns_key]["mode"] == "stochastic"
        if self.sqrt_transform:
            self.V = np.sqrt(np.abs(self.V)) * np.sign(self.V)
        self.V -= np.nanmean(self.V, axis=1)[:, None]

        self.n_recurse_neighbors = n_recurse_neighbors
        if self.n_recurse_neighbors is None:
            if n_neighbors is not None or mode_neighbors == "connectivities":
                self.n_recurse_neighbors = 1
            else:
                self.n_recurse_neighbors = 2

        if "neighbors" not in adata.uns.keys():
            neighbors(adata)
        if np.min((get_neighs(adata, "distances") > 0).sum(1).A1) == 0:
            raise ValueError(
                "Your neighbor graph seems to be corrupted. "
                "Consider recomputing via pp.neighbors."
            )
        if n_neighbors is None or n_neighbors <= get_n_neighs(adata):
            self.indices = get_indices(
                dist=get_neighs(adata, "distances"),
                n_neighbors=n_neighbors,
                mode_neighbors=mode_neighbors,
            )[0]
        else:
            if basis is None:
                basis_keys = ["X_pca", "X_tsne", "X_umap"]
                basis = [key for key in basis_keys if key in adata.obsm.keys()][-1]
            elif f"X_{basis}" in adata.obsm.keys():
                basis = f"X_{basis}"

            if isinstance(approx, str) and approx in adata.obsm.keys():
                from sklearn.neighbors import NearestNeighbors

                neighs = NearestNeighbors(n_neighbors=n_neighbors + 1)
                neighs.fit(adata.obsm[approx])
                self.indices = neighs.kneighbors_graph(
                    mode="connectivity"
                ).indices.reshape((-1, n_neighbors + 1))
            else:
                from scvelo import Neighbors

                neighs = Neighbors(adata)
                neighs.compute_neighbors(
                    n_neighbors=n_neighbors, use_rep=basis, n_pcs=10
                )
                self.indices = get_indices(
                    dist=neighs.distances, mode_neighbors=mode_neighbors
                )[0]

        self.max_neighs = random_neighbors_at_max

        gkey, gkey_ = f"{vkey}_graph", f"{vkey}_graph_neg"
        self.graph = adata.uns[gkey] if gkey in adata.uns.keys() else []
        self.graph_neg = adata.uns[gkey_] if gkey_ in adata.uns.keys() else []

        if tkey in adata.obs.keys():
            self.t0 = adata.obs[tkey].copy()
            init = min(self.t0) if isinstance(min(self.t0), int) else 0
            self.t0.cat.categories = np.arange(init, len(self.t0.cat.categories))
            self.t1 = self.t0.copy()
            self.t1.cat.categories = self.t0.cat.categories + 1
        else:
            self.t0 = None

        self.compute_uncertainties = compute_uncertainties
        self.uncertainties = None
        self.self_prob = None
        self.report = report
        self.adata = adata

    # TODO: Add docstrings
    def compute_cosines(self, n_jobs=None, backend="loky"):
        """TODO."""
        n_jobs = get_n_jobs(n_jobs=n_jobs)

        n_obs = self.X.shape[0]

        # TODO: Use batches and vectorize calculation of dX in self._calculate_cosines
        res = parallelize(
            self._compute_cosines,
            range(self.X.shape[0]),
            n_jobs=n_jobs,
            unit="cells",
            backend=backend,
        )()
        uncertainties, vals, rows, cols = map(_flatten, zip(*res))

        vals = np.hstack(vals)
        vals[np.isnan(vals)] = 0

        self.graph, self.graph_neg = vals_to_csr(
            vals, rows, cols, shape=(n_obs, n_obs), split_negative=True
        )
        if self.compute_uncertainties:
            uncertainties = np.hstack(uncertainties)
            uncertainties[np.isnan(uncertainties)] = 0
            self.uncertainties = vals_to_csr(
                uncertainties, rows, cols, shape=(n_obs, n_obs), split_negative=False
            )
            self.uncertainties.eliminate_zeros()

        confidence = self.graph.max(1).A.flatten()
        self.self_prob = np.clip(np.percentile(confidence, 98) - confidence, 0, 1)

    def _compute_cosines(self, obs_idx, queue):
        vals, rows, cols, uncertainties = [], [], [], []
        if self.compute_uncertainties:
            moments = get_moments(self.adata, np.sign(self.V_raw), second_order=True)

        for obs_id in obs_idx:
            if self.V[obs_id].max() != 0 or self.V[obs_id].min() != 0:
                neighs_idx = get_iterative_indices(
                    self.indices, obs_id, self.n_recurse_neighbors, self.max_neighs
                )

                if self.t0 is not None:
                    t0, t1 = self.t0[obs_id], self.t1[obs_id]
                    if t0 >= 0 and t1 > 0:
                        t1_idx = np.where(self.t0 == t1)[0]
                        if len(t1_idx) > len(neighs_idx):
                            t1_idx = np.random.choice(
                                t1_idx, len(neighs_idx), replace=False
                            )
                        if len(t1_idx) > 0:
                            neighs_idx = np.unique(np.concatenate([neighs_idx, t1_idx]))

                dX = self.X[neighs_idx] - self.X[obs_id, None]  # 60% of runtime
                if self.sqrt_transform:
                    dX = np.sqrt(np.abs(dX)) * np.sign(dX)

                val = cosine_correlation(np.tanh(dX), np.tanh(self.V[obs_id]))  # 40% of runtime

                if self.compute_uncertainties:
                    dX /= l2_norm(dX)[:, None]
                    uncertainties.extend(
                        np.nansum(dX**2 * moments[obs_id][None, :], 1)
                    )

                vals.extend(val)
                rows.extend(np.ones(len(neighs_idx)) * obs_id)
                cols.extend(neighs_idx)

            if queue is not None:
                queue.put(1)

        if queue is not None:
            queue.put(None)

        return uncertainties, vals, rows, cols


def _flatten(iterable):
    return [i for it in iterable for i in it]


def velocity_graph(
    data,
    vkey="velocity",
    xkey="Ms",
    tkey=None,
    basis=None,
    n_neighbors=None,
    n_recurse_neighbors=None,
    random_neighbors_at_max=None,
    sqrt_transform=None,
    variance_stabilization=None,
    gene_subset=None,
    compute_uncertainties=None,
    approx=None,
    mode_neighbors="distances",
    copy=False,
    n_jobs=None,
    backend="loky",
):
    r"""Computes velocity graph based on cosine similarities.
    The cosine similarities are computed between velocities and potential cell state
    transitions, i.e. it measures how well a corresponding change in gene expression
    :math:`\delta_{ij} = x_j - x_i` matches the predicted change according to the
    velocity vector :math:`\nu_i`,
    .. math::
        \pi_{ij} = \cos\angle(\delta_{ij}, \nu_i)
        = \frac{\delta_{ij}^T \nu_i}{\left\lVert\delta_{ij}\right\rVert
        \left\lVert \nu_i \right\rVert}.
    Arguments
    ---------
    data: :class:`~anndata.AnnData`
        Annotated data matrix.
    vkey: `str` (default: `'velocity'`)
        Name of velocity estimates to be used.
    xkey: `str` (default: `'Ms'`)
        Layer key to extract count data from.
    tkey: `str` (default: `None`)
        Observation key to extract time data from.
    basis: `str` (default: `None`)
        Basis / Embedding to use.
    n_neighbors: `int` or `None` (default: None)
        Use fixed number of neighbors or do recursive neighbor search (if `None`).
    n_recurse_neighbors: `int` (default: `None`)
        Number of recursions for neighbors search. Defaults to
        2 if mode_neighbors is 'distances', and 1 if mode_neighbors is 'connectivities'.
    random_neighbors_at_max: `int` or `None` (default: `None`)
        If number of iterative neighbors for an individual cell is higher than this
        threshold, a random selection of such are chosen as reference neighbors.
    sqrt_transform: `bool` (default: `False`)
        Whether to variance-transform the cell states changes
        and velocities before computing cosine similarities.
    gene_subset: `list` of `str`, subset of adata.var_names or `None`(default: `None`)
        Subset of genes to compute velocity graph on exclusively.
    compute_uncertainties: `bool` (default: `None`)
        Whether to compute uncertainties along with cosine correlation.
    approx: `bool` or `None` (default: `None`)
        If True, first 30 pc's are used instead of the full count matrix
    mode_neighbors: 'str' (default: `'distances'`)
        Determines the type of KNN graph used. Options are 'distances' or
        'connectivities'. The latter yields a symmetric graph.
    copy: `bool` (default: `False`)
        Return a copy instead of writing to adata.
    n_jobs: `int` or `None` (default: `None`)
        Number of parallel jobs.
    backend: `str` (default: "loky")
        Backend used for multiprocessing. See :class:`joblib.Parallel` for valid
        options.
    Returns
    -------
    velocity_graph: `.uns`
        sparse matrix with correlations of cell state transitions with velocities
    """
    adata = data.copy() if copy else data
    verify_neighbors(adata)
    if vkey not in adata.layers.keys():
        velocity(adata, vkey=vkey)
    if sqrt_transform is None:
        sqrt_transform = variance_stabilization

    vgraph = VelocityGraph(
        adata,
        vkey=vkey,
        xkey=xkey,
        tkey=tkey,
        basis=basis,
        n_neighbors=n_neighbors,
        approx=approx,
        n_recurse_neighbors=n_recurse_neighbors,
        random_neighbors_at_max=random_neighbors_at_max,
        sqrt_transform=sqrt_transform,
        gene_subset=gene_subset,
        compute_uncertainties=compute_uncertainties,
        report=True,
        mode_neighbors=mode_neighbors,
    )

    if isinstance(basis, str):
        logg.warn(
            f"The velocity graph is computed on {basis} embedding coordinates.\n"
            f"        Consider computing the graph in an unbiased manner \n"
            f"        on full expression space by not specifying basis.\n"
        )

    n_jobs = get_n_jobs(n_jobs=n_jobs)
    logg.info(
        f"computing velocity graph (using {n_jobs}/{os.cpu_count()} cores)", r=True
    )
    vgraph.compute_cosines(n_jobs=n_jobs, backend=backend)

    adata.uns[f"{vkey}_graph"] = vgraph.graph
    adata.uns[f"{vkey}_graph_neg"] = vgraph.graph_neg

    if vgraph.uncertainties is not None:
        adata.uns[f"{vkey}_graph_uncertainties"] = vgraph.uncertainties

    adata.obs[f"{vkey}_self_transition"] = vgraph.self_prob

    if f"{vkey}_params" in adata.uns.keys():
        if "embeddings" in adata.uns[f"{vkey}_params"]:
            del adata.uns[f"{vkey}_params"]["embeddings"]
    else:
        adata.uns[f"{vkey}_params"] = {}
    adata.uns[f"{vkey}_params"]["mode_neighbors"] = mode_neighbors
    adata.uns[f"{vkey}_params"]["n_recurse_neighbors"] = vgraph.n_recurse_neighbors

    logg.info("    finished", time=True, end=" " if settings.verbosity > 2 else "\n")
    logg.hint(
        "added \n"
        f"    '{vkey}_graph', sparse matrix with cosine correlations (adata.uns)"
    )

    return adata if copy else None