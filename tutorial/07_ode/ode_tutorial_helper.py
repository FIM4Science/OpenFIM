"""
Helper functions for the FIM-ODE tutorial notebook.

These are self-contained versions of utilities from scripts/ode/helpers.py
and scripts/ode/finetune.py, adapted so the tutorial has no dependency on
the repository's scripts/ directory.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_odebench_system(
    eq_id: int,
    json_path: str | Path,
    noise_sigma: float = 0.03,
    subsample_fraction: float = 0.5,
    const_idx: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load one system from a local strogatz_extended.json, add noise, subsample.

    Parameters
    ----------
    eq_id             : ODEBench system id.
    json_path         : Path to strogatz_extended.json.
    noise_sigma       : Std-dev of multiplicative Gaussian noise.
    subsample_fraction: Fraction of time points to keep.
    const_idx         : Index into the ``consts`` / ``solutions`` list.
    rng               : NumPy random generator (created if None).

    Returns
    -------
    times  : (N_ic, T_sub, 1)
    trajs  : (N_ic, T_sub, D)
    meta   : dict with keys eq_id, eq, dim, consts, n_pts_original, t_end
    """
    if rng is None:
        rng = np.random.default_rng()

    with open(json_path) as f:
        db = json.load(f)

    entry = next((item for item in db if item["id"] == eq_id), None)
    if entry is None:
        raise ValueError(f"ODEBench system id={eq_id} not found in {json_path}.")

    solutions_for_const = entry["solutions"][const_idx]

    ic_times, ic_trajs = [], []
    t_full_last = None
    for sol in solutions_for_const:
        if not sol.get("success", True):
            continue
        t_full = np.asarray(sol["t"])
        y_full = np.asarray(sol["y"]).T       # (T, D)
        t_full_last = t_full

        T     = len(t_full)
        n_sub = max(2, int(round(T * subsample_fraction)))
        idx   = np.linspace(0, T - 1, n_sub, dtype=int)

        t_sub   = t_full[idx]
        y_sub   = y_full[idx]
        y_noisy = y_sub * (1.0 + noise_sigma * rng.standard_normal(y_sub.shape))

        ic_times.append(t_sub[:, np.newaxis])
        ic_trajs.append(y_noisy)

    times = np.stack(ic_times, axis=0)   # (N_ic, T_sub, 1)
    trajs = np.stack(ic_trajs, axis=0)   # (N_ic, T_sub, D)

    meta = {
        "eq_id":          eq_id,
        "eq":             entry["eq"],
        "dim":            entry["dim"],
        "consts":         entry["consts"][const_idx] if entry["consts"] else [],
        "n_pts_original": len(t_full_last),
        "t_end":          float(t_full_last[-1]),
    }
    return times, trajs, meta


# ---------------------------------------------------------------------------
# FIM-ODE vector-field wrapper
# ---------------------------------------------------------------------------

def make_fim_vf_fn(
    model,
    trajs: np.ndarray,
    times: np.ndarray,
    device: str = "cpu",
) -> Callable[[np.ndarray], np.ndarray]:
    """Encode context trajectories once; return a numpy vector-field callable.

    Parameters
    ----------
    model  : FIMODE instance (already on ``device``).
    trajs  : (N_ic, T, D) context trajectories.
    times  : (N_ic, T, 1) observation times.
    device : torch device string.

    Returns
    -------
    vf : Callable ``(N, D) -> (N, D)``
    """
    dev = torch.device(device)
    D   = trajs.shape[-1]

    traj_t  = torch.tensor(trajs, dtype=torch.float32, device=dev).unsqueeze(0)
    times_t = torch.tensor(times, dtype=torch.float32, device=dev).unsqueeze(0)
    mask_t  = torch.ones(*traj_t.shape[:-1], 1, dtype=torch.bool, device=dev)

    with torch.no_grad():
        wrapped_D, feature_mask, concept = model.trajectory_encoding(traj_t, times_t, mask_t)

    def vf(x: np.ndarray) -> np.ndarray:
        x_t = torch.tensor(x[np.newaxis], dtype=torch.float32, device=dev)
        with torch.no_grad():
            loc      = model.pad_if_necessary(x_t)
            loc_norm = model.spatial_norm.normalization_map(loc, concept._states_norm_stats)
            out      = model.function_decoding(loc_norm, feature_mask, wrapped_D, concept)
            pred     = model.get_prediction_for_eval(out)[0, :, :D]
        return pred.cpu().numpy()

    return vf


# ---------------------------------------------------------------------------
# Fixed-point finding
# ---------------------------------------------------------------------------

def find_fixed_points(
    f: Callable[[np.ndarray], np.ndarray],
    D: int,
    n_starts: int = 400,
    x_range: Tuple[float, float] = (-6.0, 6.0),
    tol_rel: float = 0.05,
    dedup_dist: float = 0.1,
    n_grid: int = 40,
    top_k: Optional[int] = 4,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[np.ndarray, float]]:
    """Multi-start L-BFGS-B search for fixed points of ``f``.

    Evaluates ``||f||`` on a coarse grid, seeds refinement from the ``n_starts``
    best cells, deduplicates results, and returns up to ``top_k`` candidates
    sorted by residual.

    Returns
    -------
    List of (point, residual) tuples, sorted by residual ascending.
    """
    from scipy.optimize import minimize

    lo, hi = x_range

    def residual_sq(x1d):
        r = f(x1d[np.newaxis])[0]
        return float(np.dot(r, r))

    axes  = [np.linspace(lo, hi, n_grid) for _ in range(D)]
    grid  = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, D)
    norms = np.linalg.norm(f(grid), axis=-1)
    tol   = tol_rel * float(np.median(norms))

    seeds = grid[np.argsort(norms)[:n_starts]]

    fps: List[np.ndarray] = []
    for x0 in seeds:
        try:
            res = minimize(residual_sq, x0, method='L-BFGS-B',
                           options={'ftol': 1e-30, 'gtol': 1e-15, 'maxiter': 500})
            sol = res.x
        except Exception:
            continue
        if np.sqrt(residual_sq(sol)) > tol:
            continue
        if all(np.linalg.norm(sol - fp) > dedup_dist for fp in fps):
            fps.append(sol.copy())

    fps_with_res = sorted(
        [(fp, float(np.linalg.norm(f(fp[np.newaxis])[0]))) for fp in fps],
        key=lambda t: t[1],
    )
    if top_k is not None:
        fps_with_res = fps_with_res[:top_k]
    return fps_with_res


# ---------------------------------------------------------------------------
# Jacobian and stability
# ---------------------------------------------------------------------------

def numerical_jacobian(
    f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Central-difference Jacobian of ``f`` at ``x``.

    Returns J of shape (D, D) where J[i, j] = ∂f_i/∂x_j.
    """
    D = len(x)
    J = np.zeros((D, D))
    for j in range(D):
        xp, xm = x.copy(), x.copy()
        xp[j] += eps
        xm[j] -= eps
        J[:, j] = (f(xp[np.newaxis])[0] - f(xm[np.newaxis])[0]) / (2 * eps)
    return J


def stability_analysis(J: np.ndarray) -> dict:
    """Classify a fixed point from its Jacobian eigenvalues.

    Returns a dict with keys ``eigenvalues``, ``max_real``, ``label``.
    """
    evals  = np.linalg.eigvals(J)
    max_re = float(np.max(evals.real))
    min_re = float(np.min(evals.real))
    has_im = np.any(np.abs(evals.imag) > 1e-8)

    if max_re < -1e-8:
        label = "stable spiral" if has_im else "stable node"
    elif min_re > 1e-8:
        label = "unstable spiral" if has_im else "unstable node"
    elif max_re > 1e-8 and min_re < -1e-8:
        label = "saddle"
    else:
        label = "centre / marginal"

    return {"eigenvalues": evals, "max_real": max_re, "label": label}


# ---------------------------------------------------------------------------
# MoCap helpers
# ---------------------------------------------------------------------------

def integrate_from_context(
    model,
    ctx_traj: np.ndarray,    # (n_paths, T_ctx, D)
    ctx_times: np.ndarray,   # (T_ctx,)
    y0: np.ndarray,          # (D,)
    t_eval: np.ndarray,      # (L,)
    device: str = "cpu",
) -> np.ndarray:             # (L, D)
    """Encode context trajectories once, then integrate from y0 with RK45.

    Returns predicted trajectory of shape (L, D).
    Raises RuntimeError if the IVP solver fails.
    """
    from scipy.integrate import solve_ivp

    if ctx_traj.ndim == 2:
        ctx_traj = ctx_traj[np.newaxis]
    n_paths, T, D = ctx_traj.shape

    traj_t = torch.tensor(ctx_traj, dtype=torch.float32, device=device).unsqueeze(0)
    time_t = (torch.tensor(ctx_times, dtype=torch.float32, device=device)
              .unsqueeze(0).unsqueeze(0).unsqueeze(-1)
              .expand(1, n_paths, T, 1).contiguous())
    mask_t = torch.ones(1, n_paths, T, 1, dtype=torch.bool, device=device)

    @torch.no_grad()
    def fim_fn(t: float, y: np.ndarray) -> np.ndarray:
        loc = torch.tensor(y, dtype=torch.float32, device=device).view(1, 1, D)
        out = model.model_forward(traj_t, time_t, loc, mask_t)
        return model.get_prediction_for_eval(out).squeeze().cpu().numpy()[:D]

    sol = solve_ivp(
        fim_fn,
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        y0=np.asarray(y0, dtype=float),
        t_eval=t_eval,
        method="RK45", rtol=1e-4, atol=1e-6,
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    return sol.y.T   # (L, D)


def mocap_pca_to_50d(
    traj_norm: np.ndarray,
    pca_components: np.ndarray,
    pca_data_mean: np.ndarray,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    n_dims: int = 3,
) -> np.ndarray:
    """Back-project from normalized PCA space to 50D joint-angle space.

    Parameters
    ----------
    traj_norm      : (T, n_dims) or (N, T, n_dims) — normalized PCA coords
    pca_components : (5, 50) — PCA eigenvectors
    pca_data_mean  : (50,)   — PCA data mean (pca.mean_)
    norm_mean      : (1,1,5) — normalization mean
    norm_std       : (1,1,5) — normalization std
    n_dims         : number of modelled PCA dims (rest padded with 0)

    Returns (T, 50) or (N, T, 50).
    """
    single = traj_norm.ndim == 2
    if single:
        traj_norm = traj_norm[np.newaxis]
    N, T, _ = traj_norm.shape

    # Pad to 5D
    pad       = np.zeros((N, T, 5 - n_dims), dtype=traj_norm.dtype)
    traj_5d   = np.concatenate([traj_norm, pad], axis=-1)   # (N, T, 5)

    # Unnormalize
    traj_pca  = traj_5d * norm_std + norm_mean               # (N, T, 5)

    # PCA inverse: (N, T, 5) @ (5, 50) + (50,)
    traj_50d  = traj_pca @ pca_components + pca_data_mean    # (N, T, 50)

    return traj_50d[0] if single else traj_50d


def plot_mocap_pca(
    trn_ys: np.ndarray,
    tst_ys: np.ndarray,
    preds: List[np.ndarray],
    title: str,
    n_test: int = 2,
    figsize: Tuple[float, float] = (14, 5),
) -> "plt.Figure":
    """1×3 PCA projection figure for one MoCap variant.

    Parameters
    ----------
    trn_ys : (n_trn, T_trn, ≥3) training trajectories (first 3 dims used)
    tst_ys : (n_tst, T_tst, ≥3) ground-truth test trajectories
    preds  : list of (T_tst, ≥3) FIM-ODE predicted trajectories
    title  : figure suptitle
    n_test : number of test trajectories to draw
    """
    import matplotlib.pyplot as plt

    PROJ_PAIRS   = [(0, 1), (0, 2), (1, 2)]
    PROJ_XLABELS = ["PC 1", "PC 1", "PC 2"]
    PROJ_YLABELS = ["PC 2", "PC 3", "PC 3"]
    PROJ_TITLES  = ["PC 1 vs PC 2", "PC 1 vs PC 3", "PC 2 vs PC 3"]
    COLOR_PRED   = "#2ca02c"

    n_test = min(n_test, tst_ys.shape[0])
    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for col, (di, dj) in enumerate(PROJ_PAIRS):
        ax = axes[col]

        # Training context (faint blue)
        for tr in trn_ys:
            ax.plot(tr[:, di], tr[:, dj], color="#aabbcc", lw=0.6, alpha=0.5)

        # Ground-truth test trajectories (black ×)
        for k in range(n_test):
            ax.scatter(tst_ys[k, :, di], tst_ys[k, :, dj],
                       s=15, marker="x", color="black", linewidths=1.2,
                       alpha=0.7, zorder=3)

        # FIM-ODE predictions (green)
        for k, pr in enumerate(preds[:n_test]):
            ax.plot(pr[:, di], pr[:, dj],
                    color=COLOR_PRED, lw=2.0, alpha=0.9, zorder=4,
                    solid_capstyle="round")
            ax.scatter(pr[0, di], pr[0, dj],
                       s=120, marker="s", facecolor=COLOR_PRED,
                       edgecolors="black", linewidths=1.5, zorder=5)

        ax.set_xlabel(PROJ_XLABELS[col], fontsize=10)
        ax.set_ylabel(PROJ_YLABELS[col], fontsize=10)
        ax.set_title(PROJ_TITLES[col], fontsize=11)
        ax.set_aspect("equal", adjustable="datalim")

    return fig
