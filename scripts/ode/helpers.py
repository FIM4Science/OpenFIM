"""
scripts/ode/helpers.py
======================
Shared library for FIM-ODE experiments.

Sections
--------
1. ODE dynamics
2. Model inference helpers
3. Context trajectory generation (unified sampler + ensemble builder)
4. Reference dataset preprocessing
5. Evaluation (MSE statistics)
6. Reporting (console, Markdown, LaTeX)
7. Plotting
8. Autoregressive context expansion
9. ODEBench utilities (fixed-point & stability analysis)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

# ─────────────────────────────────────────────────────────────────────────────
# Compatibility patches (applied once at import / each reload)
# ─────────────────────────────────────────────────────────────────────────────

# NumPy 2.0 removed np.infty; ODEFormer still uses it.
if not hasattr(np, "infty"):
    np.infty = np.inf

# PyTorch >= 2.6 defaulted weights_only=True; ODEFormer needs False.
# Reset to the true original each time this module is (re)loaded so that
# stale recursive closures from a previous run are never reused.
import torch.serialization as _ts
torch.load = _ts.load
def _apply_torch_load_patch():
    _orig = torch.load          # local — immune to global overwrites
    def _patched(*a, **kw):
        kw.setdefault('weights_only', False)
        return _orig(*a, **kw)
    _patched._weights_only_patched = True
    torch.load = _patched
_apply_torch_load_patch()

# ─────────────────────────────────────────────────────────────────────────────
# Type aliases
# ─────────────────────────────────────────────────────────────────────────────
Rng = np.random.Generator
Array = np.ndarray


# =============================================================================
# 1. ODE dynamics
# =============================================================================

def vdp_dynamics(t: float, x: Sequence[float], mu: float = 0.5) -> List[float]:
    """
    Van der Pol oscillator.

        dx₁/dt =  x₂
        dx₂/dt = -x₁ + μ · x₂ · (1 − x₁²)

    Parameters
    ----------
    t   : current time (unused; autonomous system)
    x   : state [x₁, x₂]
    mu  : damping coefficient (default 0.5)
    """
    x1, x2 = x
    return [x2, -x1 + mu * x2 * (1 - x1**2)]


def fhn_dynamics(t: float, x: Sequence[float]) -> List[float]:
    """
    FitzHugh-Nagumo model.

        dx₁/dt = 3·(x₁ − x₁³/3 + x₂)
        dx₂/dt = (0.2 − 3·x₁ − 0.2·x₂) / 3

    Parameters
    ----------
    t : current time (unused; autonomous system)
    x : state [x₁, x₂]
    """
    x1, x2 = x
    return [3.0 * (x1 - x1**3 / 3.0 + x2),
            (0.2 - 3.0 * x1 - 0.2 * x2) / 3.0]


# =============================================================================
# 2. Model inference helpers
# =============================================================================

def make_fim_ode_fn(
    model,
    trajectories: torch.Tensor,   # (1, n_paths, T, D)
    times: torch.Tensor,           # (1, n_paths, T, 1)
    mask: torch.Tensor,            # (1, n_paths, T, 1)  bool
) -> Callable[[float, Array], Array]:
    """
    Pre-encode context trajectories once and return a scalar ODE RHS compatible
    with scipy.integrate.solve_ivp.

    The encoding (trajectory_encoding) is computed a single time; subsequent
    calls to the returned function only run the lightweight function_decoding
    step. This amortises the encoding cost when integrating over many steps.

    Parameters
    ----------
    model        : FimOdeonUnified instance (eval mode)
    trajectories : context observations tensor
    times        : corresponding time stamps tensor
    mask         : boolean mask (True = observed, False = padding)

    Returns
    -------
    fim_fn(t, y) -> np.ndarray (D,)
        RHS evaluated at state y; t is ignored (model is time-homogeneous).
    """
    with torch.no_grad():
        wrapped_D, feature_mask, concept = model.trajectory_encoding(
            trajectories, times, mask
        )

    D_in = trajectories.shape[-1]   # original state dimension before any padding

    def fim_fn(t: float, y: Array) -> Array:
        with torch.no_grad():
            loc = torch.tensor(y, dtype=torch.float32, device=trajectories.device
                               ).reshape(1, 1, -1)                        # [1, 1, D_in]
            loc = model.pad_if_necessary(loc)                             # [1, 1, dim_max]
            loc_norm = model.spatial_norm.normalization_map(
                loc, concept._states_norm_stats
            )
            out   = model.function_decoding(loc_norm, feature_mask, wrapped_D, concept)
            drift = model.get_prediction_for_eval(out)[0, 0, :D_in].cpu().numpy()
        return drift

    return fim_fn


def predict_and_integrate_ode(
    model,
    context_trajs: Array,    # (n_paths, T, D) — noisy context observations
    context_times: Array,    # (T,) — shared time stamps for all context paths
    y0: Array,               # (D,) — initial condition for numerical integration
    t_eval: Array,           # (L,) — output evaluation times
    device: str = "cpu",
) -> object:                 # scipy OdeResult; sol.y.T is (L, D)
    """
    Encode context trajectories with FIM-ODE, then integrate from y0.

    The context encoding is computed once via make_fim_ode_fn; scipy evaluates
    the model at each integration step through the cached representation.

    Notes
    -----
    - context_times and t_eval may be different grids; context_times is used
      only for the in-context encoding, while t_eval drives the integration.
    - Integration always starts at t_eval[0] with initial condition y0.

    Returns
    -------
    sol : scipy OdeResult
        sol.y.T has shape (L, D).
    """
    n_paths, T, D = context_trajs.shape

    traj_t = torch.tensor(context_trajs, dtype=torch.float32, device=device
                          ).unsqueeze(0)                              # [1, n_paths, T, D]
    time_t = (torch.tensor(context_times, dtype=torch.float32, device=device)
              .unsqueeze(0).unsqueeze(-1)
              .expand(1, n_paths, T, 1).contiguous())                 # [1, n_paths, T, 1]
    mask_t = torch.ones(1, n_paths, T, 1, dtype=torch.bool, device=device)

    fim_fn = make_fim_ode_fn(model, traj_t, time_t, mask_t)

    return solve_ivp(
        fim_fn,
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        y0=np.asarray(y0, dtype=float),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-4,
        atol=1e-6,
    )


def predict_vector_field_2d(
    model,
    context_trajs: Array,              # (n_paths, T, D) or (n_segs, max_len, D)
    context_times: Array,              # (T,) or (n_segs, max_len)
    x1_range: Tuple[float, float],
    x2_range: Tuple[float, float],
    n_grid: int = 20,
    device: str = "cpu",
    mask: Optional[Array] = None,      # (n_segs, max_len) bool — None → all valid
) -> Tuple[Array, Array, Array, Array]:
    """
    Evaluate the inferred vector field on a uniform 2-D grid.

    Handles both the standard case (contiguous context, no padding) and the
    FHN case (padded sub-trajectories with an explicit boolean mask).

    Parameters
    ----------
    context_trajs : (n_paths, T, D) for VDP; (n_segs, max_len, D) for FHN
    context_times : (T,) for VDP; (n_segs, max_len) for FHN
    x1_range, x2_range : axis ranges for the evaluation grid
    n_grid        : number of grid points per axis
    device        : torch device string
    mask          : (n_segs, max_len) bool validity mask for padded context
                    (output of split_fhn_trajectories).  Pass None (default)
                    for contiguous VDP context — an all-ones mask is used.

    Returns
    -------
    X1, X2 : (n_grid, n_grid) meshgrid arrays
    U, V   : (n_grid, n_grid) predicted drift components
    """
    x1 = np.linspace(*x1_range, n_grid)
    x2 = np.linspace(*x2_range, n_grid)
    X1, X2 = np.meshgrid(x1, x2)
    grid   = np.stack([X1.ravel(), X2.ravel()], axis=-1)   # (n_grid², 2)

    traj_t = torch.tensor(context_trajs, dtype=torch.float32, device=device).unsqueeze(0)
    locs_t = torch.tensor(grid,          dtype=torch.float32, device=device).unsqueeze(0)

    if mask is not None:
        # Padded context (FHN): times are (n_segs, max_len), mask is explicit
        time_t = torch.tensor(context_times, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
        mask_t = torch.tensor(mask,          dtype=torch.bool,    device=device).unsqueeze(0).unsqueeze(-1)
    else:
        # Contiguous context (VDP): times are (T,), all positions valid
        n_paths, T, _ = context_trajs.shape
        time_t = (torch.tensor(context_times, dtype=torch.float32, device=device)
                  .unsqueeze(0).unsqueeze(-1)
                  .expand(1, n_paths, T, 1).contiguous())
        mask_t = torch.ones(1, n_paths, T, 1, dtype=torch.bool, device=device)

    with torch.no_grad():
        out   = model.model_forward(traj_t, time_t, locs_t, mask_t)
        drift = model.get_prediction_for_eval(out)[0].cpu().numpy()   # (n_grid², 2)

    return X1, X2, drift[:, 0].reshape(n_grid, n_grid), drift[:, 1].reshape(n_grid, n_grid)


# =============================================================================
# 3. Context trajectory generation
# =============================================================================

def sample_context_trajectories(
    dynamics: Callable,
    y0: Array,                                    # (D,) reference IC (perturbed mode centre)
    t_eval: Array,                                # (T,) observation times
    noise_var: float,                             # observation noise variance σ²
    k: int,                                       # number of trajectories to generate
    rng: Rng,
    *,
    ic_mode: str,                                 # "perturbed" | "random"
    ic_noise_var: float = 0.1,                    # IC perturbation variance (ic_mode="perturbed")
    x0_bounds: Optional[Tuple[float, float]] = None,  # (lo, hi) (ic_mode="random")
) -> Tuple[Array, Array]:
    """
    Generate k context trajectories with freshly sampled ICs and noise.

    ICs and observation noise are always resampled on every call (the only
    supported mode — the most realistic evaluation scenario).

    IC modes
    --------
    "perturbed" : ICs ~ N(y0, ic_noise_var · I)
    "random"    : ICs ~ U(x0_bounds[0], x0_bounds[1]) per dimension

    Parameters
    ----------
    dynamics     : callable (t, x) -> dx/dt
    y0           : (D,) reference IC (centre of perturbations in perturbed mode)
    t_eval       : (T,) observation times
    noise_var    : observation noise variance
    k            : number of trajectories to generate
    rng          : numpy Generator (updated in-place)
    ic_mode      : IC sampling strategy (see above)
    ic_noise_var : perturbation variance for ic_mode="perturbed"
    x0_bounds    : (lo, hi) per-dimension bounds for ic_mode="random"

    Returns
    -------
    noisy : (k, T, D) float32 — noisy trajectories
    clean : (k, T, D) float32 — noise-free trajectories
    """
    T, D      = len(t_eval), len(y0)
    t_span    = (float(t_eval[0]), float(t_eval[-1]))
    noise_std = float(np.sqrt(noise_var))

    noisy = np.empty((k, T, D), dtype=np.float32)
    clean = np.empty((k, T, D), dtype=np.float32)

    for j in range(k):
        if ic_mode == "perturbed":
            ic = np.asarray(y0, dtype=float) + rng.normal(0.0, np.sqrt(ic_noise_var), D)
        elif ic_mode == "random":
            lo, hi = x0_bounds
            ic = rng.uniform(lo, hi, D)
        else:
            raise ValueError(f"Unknown ic_mode: {ic_mode!r}")
        sol      = solve_ivp(dynamics, t_span, list(ic), t_eval=t_eval, rtol=1e-8, atol=1e-8)
        clean[j] = sol.y.T.astype(np.float32)
        noisy[j] = clean[j] + rng.normal(0.0, noise_std, (T, D)).astype(np.float32)

    return noisy, clean


def generate_ensemble(
    dynamics: Callable,
    ref_traj: Array,                              # (T, D) reference trajectory (from dataset)
    ref_times: Array,                             # (T,) observation times
    y0: Array,                                    # (D,) nominal IC of the reference trajectory
    noise_var: float,                             # observation noise variance for generated trajs
    x0_bounds: Tuple[float, float],               # (lo, hi) per-dim bounds for random IC mode
    *,
    n_repeats: int = 100,
    k_extra_list: Sequence[int] = (2, 8, 49),
    ic_noise_var: float = 0.1,
    seed: int = 0,
) -> Dict[str, Array]:
    """
    Build context ensembles around a fixed reference trajectory.

    For each combination of k_extra ∈ k_extra_list and
    ic_mode ∈ {"perturbed", "random"}, generates n_repeats context sets where:
      - trajectory index 0 is always the provided reference dataset trajectory
      - trajectories 1..k_extra are freshly sampled (ICs + noise) each repeat

    IC modes
    --------
    "perturbed" : additional ICs ~ N(y0, ic_noise_var · I)
    "random"    : additional ICs ~ U(x0_bounds[0], x0_bounds[1]) per dimension

    Parameters
    ----------
    dynamics     : callable (t, x) -> dx/dt
    ref_traj     : (T, D) reference trajectory (already noisy, from the dataset)
    ref_times    : (T,) observation times shared by all trajectories
    y0           : (D,) nominal IC (reference point for perturbed mode)
    noise_var    : observation noise variance for the generated trajectories
    x0_bounds    : (lo, hi) per-dimension uniform bounds for random IC sampling
    n_repeats    : number of independent context sets per condition
    k_extra_list : additional trajectory counts to try; context totals are k_extra + 1
    ic_noise_var : IC perturbation variance for perturbed mode
    seed         : RNG seed

    Returns
    -------
    dict with keys "perturbed_k{N}" and "random_k{N}" for N = 1 + k_extra,
    each array of shape (n_repeats, N, T, D) float32, where
    result[key][:, 0] == ref_traj for all keys.
    Also includes "ref_traj" (T, D) and "ref_times" (T,).

    Note for FHN: apply split_fhn_trajectories to each result[key][r, 1:]
    (the additional trajectories) and concatenate with the already-split
    reference context from the dataset.
    """
    rng = np.random.default_rng(seed)
    y0  = np.asarray(y0, dtype=float)
    ref = np.asarray(ref_traj, dtype=np.float32)   # (T, D)
    T, D = ref.shape

    out: Dict[str, Any] = {"ref_traj": ref, "ref_times": np.asarray(ref_times)}

    for k_extra in k_extra_list:
        k_total = 1 + k_extra
        for ic_mode in ("perturbed", "random"):
            key  = f"{ic_mode}_k{k_total}"
            data = np.empty((n_repeats, k_total, T, D), dtype=np.float32)
            for r in range(n_repeats):
                extra, _ = sample_context_trajectories(
                    dynamics, y0, ref_times, noise_var, k_extra, rng,
                    ic_mode=ic_mode, ic_noise_var=ic_noise_var, x0_bounds=x0_bounds,
                )
                data[r, 0]  = ref    # reference stays fixed across repeats
                data[r, 1:] = extra  # (k_extra, T, D) freshly sampled
            out[key] = data          # (n_repeats, k_total, T, D)

    return out


def split_fhn_trajectories(
    trajs: Array,    # (k, T, D) — k context trajectories on a shared grid
    times: Array,    # (T,) — shared time stamps
    min_len: int = 2,
) -> Tuple[Optional[Array], Optional[Array], Optional[Array]]:
    """
    Split FHN context trajectories at missing-region boundaries.

    The FHN missing region is the phase-space quadrant x₁ > 0 & x₂ < 0.
    Feeding a trajectory that straddles this region to the model would create
    artificial transition features across the gap. Instead, each contiguous
    observed block is treated as a separate sub-trajectory (path).

    Example: a trajectory that crosses the missing region twice produces three
    sub-trajectories:
        x[0..i], x[j..k], x[l..T]

    All sub-trajectories are zero-padded to the same length and returned with a
    boolean mask so they can be stacked as parallel context paths for the model.

    Parameters
    ----------
    trajs   : (k, T, D) context trajectories
    times   : (T,) shared time grid
    min_len : discard segments shorter than this (avoids degenerate segments)

    Returns
    -------
    ctx_trajs : (n_segs, max_len, D) float32 — zero-padded sub-trajectories
    ctx_times : (n_segs, max_len)    float32 — zero-padded time stamps
    ctx_mask  : (n_segs, max_len)    bool    — True at valid positions

    Returns (None, None, None) if no valid segment is found.
    """
    missing = (trajs[..., 0] > 0) & (trajs[..., 1] < 0)   # (k, T) True = missing region
    seg_trajs: List[Array] = []
    seg_times: List[Array] = []

    for j in range(len(trajs)):
        obs_j = ~missing[j]    # (T,) True = observed
        i0    = None
        for i, obs in enumerate(obs_j):
            if obs and i0 is None:
                i0 = i
            elif not obs and i0 is not None:
                if i - i0 >= min_len:
                    seg_trajs.append(trajs[j, i0:i])
                    seg_times.append(times[i0:i])
                i0 = None
        # Flush the final open segment
        if i0 is not None and len(times) - i0 >= min_len:
            seg_trajs.append(trajs[j, i0:])
            seg_times.append(times[i0:])

    if not seg_trajs:
        return None, None, None

    D       = trajs.shape[-1]
    max_len = max(len(t) for t in seg_times)
    n_segs  = len(seg_trajs)
    ctx_trajs = np.zeros((n_segs, max_len, D), dtype=np.float32)
    ctx_times = np.zeros((n_segs, max_len),    dtype=np.float32)
    ctx_mask  = np.zeros((n_segs, max_len),    dtype=bool)

    for j, (tr, ts) in enumerate(zip(seg_trajs, seg_times)):
        l = len(ts)
        ctx_trajs[j, :l] = tr
        ctx_times[j, :l] = ts
        ctx_mask[j,  :l] = True

    return ctx_trajs, ctx_times, ctx_mask


# =============================================================================
# 4. Reference dataset preprocessing (Hegde et al.)
# =============================================================================

def preprocess_ref_vdp_uniform(ref: Dict) -> Tuple[Array, Array]:
    """
    Fix VDP-uniform reference data: the raw test set covers [0, 14] (100 pts).
    Keep only the last 50 points ([7, 14]) to match the forecast target window.

    Parameters
    ----------
    ref : npz dict with keys 'test_ts', 'test_ys'

    Returns
    -------
    test_ts : (50,)    — corrected test time stamps
    test_ys : (1,50,2) — corrected test observations
    """
    return ref["test_ts"][-50:], ref["test_ys"][:, -50:]


def preprocess_ref_vdp_nonuniform(ref: Dict, seed: int = 0) -> Tuple[Array, Array]:
    """
    Fix VDP-nonuniform reference data: the raw test set has 100 points on [7, 14].
    Randomly subsample 50 points (fixed seed for reproducibility).

    Parameters
    ----------
    ref  : npz dict with keys 'test_ts', 'test_ys'
    seed : random seed for subsampling

    Returns
    -------
    test_ts : (50,)    — subsampled test time stamps
    test_ys : (1,50,2) — subsampled test observations
    """
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(ref["test_ts"]), 50, replace=False))
    return ref["test_ts"][idx], ref["test_ys"][:, idx]


def preprocess_ref_fhn(
    ref: Dict,
    min_len: int = 2,
) -> Tuple[Array, Array, Array, Array, Array, Array]:
    """
    Prepare FHN reference data: split observed trajectory at missing-region
    boundaries, build padded context arrays, and identify target time steps.

    The FHN dataset provides a single trajectory; the missing region is marked
    by interpolation_mask == True. We extract contiguous observed segments and
    pad them so they can be stacked as parallel context paths.

    Parameters
    ----------
    ref     : npz dict with keys 'full_ts', 'full_ys', 'interpolation_mask', 'x0'
    min_len : minimum segment length to keep

    Returns
    -------
    ctx_trajs  : (n_segs, max_len, D)  — padded context sub-trajectories
    ctx_times  : (n_segs, max_len)     — padded time stamps
    ctx_mask   : (n_segs, max_len)     — True at valid positions
    miss_mask  : (T,) bool             — True at missing time steps (target)
    target_ts  : (T_miss,)             — time stamps of the missing region
    target_ys  : (T_miss, D)           — ground-truth states in missing region
    """
    full_ts  = ref["full_ts"]                                   # (T,)
    full_ys  = ref["full_ys"][0]                                # (T, D)
    obs_mask = ~ref["interpolation_mask"].astype(bool)          # True = observed

    # ── Identify contiguous observed segments ─────────────────────────────
    segs: List[Tuple[int, int]] = []
    i0 = None
    for i, obs in enumerate(obs_mask):
        if obs and i0 is None:
            i0 = i
        elif not obs and i0 is not None:
            segs.append((i0, i))
            i0 = None
    if i0 is not None:
        segs.append((i0, len(obs_mask)))

    # ── Pad segments to uniform length ────────────────────────────────────
    max_len = max(e - s for s, e in segs)
    D       = full_ys.shape[-1]
    n_segs  = len(segs)
    ctx_trajs = np.zeros((n_segs, max_len, D),  dtype=np.float32)
    ctx_times = np.zeros((n_segs, max_len),      dtype=np.float32)
    ctx_mask  = np.zeros((n_segs, max_len),      dtype=bool)
    for j, (s, e) in enumerate(segs):
        l = e - s
        ctx_trajs[j, :l] = full_ys[s:e]
        ctx_times[j, :l] = full_ts[s:e]
        ctx_mask[j,  :l] = True

    miss_mask = ref["interpolation_mask"].astype(bool)          # (T,)
    target_ts = full_ts[miss_mask]                              # (T_miss,)
    target_ys = full_ys[miss_mask]                              # (T_miss, D)

    return ctx_trajs, ctx_times, ctx_mask, miss_mask, target_ts, target_ys


# =============================================================================
# 5. Evaluation (MSE statistics and experiment runners)
# =============================================================================

def mse_stats(errs: Sequence[float]) -> Dict[str, float]:
    """
    Compute summary statistics over a collection of per-run MSE values.

    Returns a dict with keys: mean, std, median, q05, q95.
    """
    a = np.asarray(errs, dtype=float)
    return dict(
        mean=float(np.mean(a)),
        std=float(np.std(a)),
        median=float(np.median(a)),
        q05=float(np.quantile(a, 0.05)),
        q95=float(np.quantile(a, 0.95)),
    )


def eval_vdp(
    dynamics: Callable,
    train_times: Array,      # (T_train,) — context observation times
    full_times: Array,       # (T_full,) — integration grid [t0, t_end]
    y0: Array,               # (D,) — nominal IC at full_times[0]
    noise_var: float,
    target: Array,           # (T_test, D) — ground-truth test trajectory
    label: str,
    *,
    model,
    device: str,
    context_length: int,     # index splitting train from test in full_times
    ctx_sizes: Sequence[int],
    n_runs: int = 100,
    ic_mode: str = "perturbed",   # "nominal" | "perturbed" | "random"
    ic_noise_var: float = 0.1,
    x0_bounds: Optional[Tuple[float, float]] = None,
    randomize_times: bool = False,
    resample_ics: bool = False,
    seed: int = 0,
) -> Dict[int, Dict[str, float]]:
    """
    Evaluate VDP forecasting MSE over n_runs context realisations.

    Integrates from y0 over full_times for each realisation and computes MSE
    on full_times[context_length:] (the extrapolation window).

    IC / noise sampling protocol
    ----------------------------
    ``resample_ics=False`` (default) — fixed IC protocol:
        ICs are drawn once per k; n_runs independent noise matrices are applied
        to the same clean trajectories.  This isolates the effect of observation
        noise while keeping the underlying dynamics fixed.

    ``resample_ics=True`` — joint protocol:
        ICs, integrations, and noise are all resampled independently for each
        of the n_runs realisations.  This is the more realistic protocol: the
        model must generalise across diverse context sets.

    When ``randomize_times=True`` the observation grid is resampled from
    U(t₀, tₑ) for every run (VDP-nonuniform variant).  This implies per-run
    sampling regardless of ``resample_ics``.

    Parameters
    ----------
    dynamics        : ODE right-hand side callable
    train_times     : context observation times (may be resampled per run)
    full_times      : full integration grid (train + test concatenated)
    y0              : nominal initial condition
    noise_var       : observation noise variance
    target          : ground-truth trajectory on the test interval
    label           : display label for progress messages
    model           : FimOdeonUnified instance
    device          : torch device string
    context_length  : split index between training and test time steps
    ctx_sizes       : list of k values to evaluate
    n_runs          : number of independent realisations
    ic_mode         : IC sampling strategy (see sample_context_trajectories)
    ic_noise_var    : IC perturbation variance for ic_mode="perturbed"
    x0_bounds       : (lo, hi) for ic_mode="random"
    randomize_times : resample the observation grid each run
    resample_ics    : if True use joint protocol (re-draw ICs each run)
    seed            : base random seed; actual seed per k = seed * 1000 + k

    Returns
    -------
    dict mapping k -> mse_stats dict
    """
    results: Dict[int, Dict] = {}
    for k in ctx_sizes:
        print(f"  [{label}]  n_ctx={k:<2}  ic={ic_mode}  ×{n_runs} ...", end=" ", flush=True)
        t0  = time.time()
        rng = np.random.default_rng(seed * 1000 + k)
        errs: List[float] = []

        if randomize_times:
            # Times differ each run → must sample one realisation at a time
            for _ in range(n_runs):
                ctx, _, used_times = sample_context_trajectories(
                    dynamics, y0, train_times, noise_var, k, rng,
                    ic_mode=ic_mode, ic_noise_var=ic_noise_var, x0_bounds=x0_bounds,
                    randomize_times=True, n_realizations=1,
                )
                sol = predict_and_integrate_ode(model, ctx[0], used_times, y0, full_times, device)
                errs.append(float(np.mean((sol.y.T[context_length:] - target) ** 2)))
        else:
            # Batch: generate all n_runs realisations in one call
            ctx, _, used_times = sample_context_trajectories(
                dynamics, y0, train_times, noise_var, k, rng,
                ic_mode=ic_mode, ic_noise_var=ic_noise_var, x0_bounds=x0_bounds,
                randomize_times=False, n_realizations=n_runs, resample_ics=resample_ics,
            )   # ctx: (n_runs, k, T, D)
            for i in range(n_runs):
                sol = predict_and_integrate_ode(model, ctx[i], used_times, y0, full_times, device)
                errs.append(float(np.mean((sol.y.T[context_length:] - target) ** 2)))

        results[k] = mse_stats(errs)
        print(f"done  ({time.time()-t0:.1f}s)")
    return results


def eval_fhn(
    dynamics: Callable,
    y0: Array,               # (D,) — nominal IC at t=t_fhn[0]
    noise_var: float,
    target: Array,           # (T_miss, D) — ground-truth at missing steps
    miss_mask: Array,        # (T,) bool — True = missing region
    t_fhn: Array,            # (T,) — full FHN time grid
    label: str,
    *,
    model,
    device: str,
    ctx_sizes: Sequence[int],
    n_runs: int = 100,
    ic_mode: str = "perturbed",
    ic_noise_var: float = 0.1,
    resample_ics: bool = False,
    seed: int = 0,
) -> Dict[int, Dict[str, float]]:
    """
    Evaluate FHN imputation MSE over n_runs context realisations.

    Context trajectories are split into contiguous observed sub-trajectories at
    missing-region boundaries (via split_fhn_trajectories) before encoding.
    The model is then integrated from y0 over the full grid; MSE is measured
    at the missing time steps.

    Sampling protocol follows the same ``resample_ics`` logic as ``eval_vdp``:
    ``False`` → fixed ICs, varying noise; ``True`` → joint IC + noise resampling.
    Realisations where all context segments fall entirely inside the missing
    quadrant (split returns None) are skipped and reported separately.

    Parameters
    ----------
    dynamics     : ODE right-hand side callable (typically fhn_dynamics)
    y0           : nominal IC at t=t_fhn[0]
    noise_var    : observation noise variance
    target       : ground-truth states at missing time steps
    miss_mask    : (T,) boolean mask — True = missing region (target)
    t_fhn        : full FHN evaluation grid
    label        : display label
    model        : FimOdeonUnified instance
    device       : torch device string
    ctx_sizes    : list of k values to evaluate
    n_runs       : number of independent realisations
    ic_mode      : IC sampling strategy (see sample_context_trajectories)
    ic_noise_var : IC perturbation variance
    resample_ics : if True use joint protocol (re-draw ICs each run)
    seed         : base random seed (actual seed per k = seed * 1000 + k)

    Returns
    -------
    dict mapping k -> mse_stats dict
    """
    results: Dict[int, Dict] = {}
    for k in ctx_sizes:
        print(f"  [{label}]  n_ctx={k:<2}  ic={ic_mode}  ×{n_runs} ...", end=" ", flush=True)
        t0      = time.time()
        rng     = np.random.default_rng(seed * 1000 + k)
        errs: List[float] = []
        skipped = 0

        ctx, _, _ = sample_context_trajectories(
            dynamics, y0, t_fhn, noise_var, k, rng,
            ic_mode=ic_mode, ic_noise_var=ic_noise_var,
            n_realizations=n_runs, resample_ics=resample_ics,
        )   # ctx: (n_runs, k, T, D)

        for i in range(n_runs):
            ctx_trajs, ctx_times, ctx_mask_arr = split_fhn_trajectories(ctx[i], t_fhn)
            if ctx_trajs is None:
                skipped += 1
                continue

            traj_t = torch.tensor(ctx_trajs,    dtype=torch.float32, device=device).unsqueeze(0)
            time_t = torch.tensor(ctx_times,    dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
            mask_t = torch.tensor(ctx_mask_arr, dtype=torch.bool,    device=device).unsqueeze(0).unsqueeze(-1)
            fim_fn = make_fim_ode_fn(model, traj_t, time_t, mask_t)
            sol    = solve_ivp(
                fim_fn,
                t_span=(float(t_fhn[0]), float(t_fhn[-1])),
                y0=np.asarray(y0, dtype=float),
                t_eval=t_fhn,
                method="RK45", rtol=1e-4, atol=1e-6,
            )
            errs.append(float(np.mean((sol.y.T[miss_mask] - target) ** 2)))

        results[k] = mse_stats(errs)
        suffix = f"  [{skipped} skipped]" if skipped else ""
        print(f"done  ({time.time()-t0:.1f}s){suffix}")
    return results


def eval_ref_vdp(
    model,
    device: str,
    train_ts: Array,    # (T_train,)
    train_ys: Array,    # (1, T_train, D)
    x0: Array,          # (D,)
    test_ts: Array,     # (T_test,)
    test_ys: Array,     # (1, T_test, D)
    label: str,
) -> Tuple[Array, Array, float]:
    """
    Single-trajectory evaluation on a reference VDP dataset.

    Integrates from x0 on the combined train+test time grid and recovers
    predictions at the test time stamps via index matching.

    Parameters
    ----------
    model    : FimOdeonUnified instance
    device   : torch device string
    train_ts : (T_train,) — context time stamps
    train_ys : (1, T_train, D) — context observations (single trajectory)
    x0       : (D,) — initial condition at train_ts[0]
    test_ts  : (T_test,) — evaluation time stamps for MSE
    test_ys  : (1, T_test, D) — ground-truth test observations
    label    : display label

    Returns
    -------
    pred     : (T_full, D) — predicted trajectory on the full combined grid
    t_full   : (T_full,) — combined time grid (sorted)
    mse      : scalar MSE on the test window
    """
    ctx    = train_ys[0][np.newaxis]                                   # (1, T_train, D)
    t_full = np.sort(np.concatenate([train_ts, test_ts]))              # (T_full,)
    sol    = predict_and_integrate_ode(model, ctx, train_ts, x0, t_full, device)
    pred   = sol.y.T                                                    # (T_full, D)

    # Match predictions to the exact test time stamps
    idx_test  = np.searchsorted(t_full, test_ts)
    pred_test = pred[idx_test]                                         # (T_test, D)
    mse = float(np.mean((pred_test - test_ys[0]) ** 2))
    print(f"  {label:<25s}  MSE = {mse:.5f}")
    return pred, t_full, mse


def eval_ref_fhn(
    model,
    device: str,
    ctx_trajs: Array,   # (n_segs, max_len, D) — padded context
    ctx_times: Array,   # (n_segs, max_len)
    ctx_mask: Array,    # (n_segs, max_len) bool
    full_ts: Array,     # (T,)
    x0: Array,          # (D,)
    miss_mask: Array,   # (T,) bool — True = missing region
    target_ys: Array,   # (T_miss, D)
    label: str = "FHN (missing quadrant)",
) -> Tuple[Array, float]:
    """
    Single-trajectory evaluation on the FHN reference dataset.

    Encodes the padded observed segments, integrates from x0, and computes MSE
    at the missing-region time steps.

    Returns
    -------
    pred : (T, D) — predicted trajectory on the full grid
    mse  : scalar MSE on the missing region
    """
    traj_t = torch.tensor(ctx_trajs, dtype=torch.float32, device=device).unsqueeze(0)
    time_t = torch.tensor(ctx_times, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
    mask_t = torch.tensor(ctx_mask,  dtype=torch.bool,    device=device).unsqueeze(0).unsqueeze(-1)

    fim_fn = make_fim_ode_fn(model, traj_t, time_t, mask_t)
    sol    = solve_ivp(
        fim_fn,
        t_span=(float(full_ts[0]), float(full_ts[-1])),
        y0=np.asarray(x0, dtype=float),
        t_eval=full_ts,
        method="RK45", rtol=1e-4, atol=1e-6,
    )
    pred = sol.y.T                                                     # (T, D)
    mse  = float(np.mean((pred[miss_mask] - target_ys) ** 2))
    print(f"  {label:<25s}  MSE = {mse:.5f}")
    return pred, mse


# =============================================================================
# 5b. ODEFormer — loading and reference-dataset evaluation
# =============================================================================

def load_odeformer(
    device: str = "cpu",
    beam_size: int = 50,
    beam_temperature: float = 0.1,
):
    """
    Load the pretrained ODEFormer model (sdascoli/odeformer).

    ODEFormer performs symbolic regression on ODE trajectories: given one or
    more noisy time-series it searches for a symbolic expression that fits the
    underlying ODE and then integrates it for forecasting / interpolation.

    Parameters
    ----------
    device          : torch device string ("cpu", "cuda", "mps")
    beam_size       : beam-search width (higher → more candidates, slower)
    beam_temperature: sampling temperature for beam search

    Returns
    -------
    SymbolicTransformerRegressor instance with pretrained weights loaded.

    Notes
    -----
    Requires the `odeformer` package (pip install odeformer).
    After loading, call model.fit(times=[ts], trajectories=[ys]) then
    model.predict(times=[eval_ts], y0=[ic]) for single-trajectory evaluation.
    """
    from odeformer.model import SymbolicTransformerRegressor
    odeformer = SymbolicTransformerRegressor(from_pretrained=True)
    odeformer.set_model_args({"beam_size": beam_size, "beam_temperature": beam_temperature})
    odeformer.model.to(device)
    odeformer.model.eval()
    print(f"ODEFormer loaded  (beam_size={beam_size}, beam_temperature={beam_temperature})")
    return odeformer


def eval_ref_vdp_odeformer(
    odeformer,
    train_ts: Array,     # (T_train,) — context observation times
    train_ys: Array,     # (1, T_train, D) — context observations
    x0: Array,           # (D,) — true IC at t=train_ts[0] (i.e. t=0)
    test_ts: Array,      # (T_test,) — evaluation times for MSE
    test_ys: Array,      # (1, T_test, D) — ground-truth test observations
    label: str,
    verbose: bool = True,
) -> Tuple[Optional[Array], Array, float]:
    """
    Evaluate ODEFormer on a reference VDP dataset.

    Workflow:
      1. Fit ODEFormer on the training observations (symbolic regression).
      2. Integrate the inferred ODE from x0 over the combined train+test grid.
      3. Match predictions to the test time stamps and compute MSE.

    Integration starts at x0 (the true IC at t=0) so that the forecast window
    [7, 14] is reached by propagating the dynamics forward — the same protocol
    used by FIM-ODE for a fair comparison.

    Parameters
    ----------
    odeformer : pretrained SymbolicTransformerRegressor
    train_ts  : (T_train,) context time stamps
    train_ys  : (1, T_train, D) noisy training observations
    x0        : (D,) initial condition at t=train_ts[0]
    test_ts   : (T_test,) evaluation time stamps
    test_ys   : (1, T_test, D) ground-truth test values
    label     : display label

    Returns
    -------
    pred   : (T_full, D) predicted trajectory on the combined grid, or None on failure
    t_full : (T_full,) combined time grid (sorted)
    mse    : MSE on the test window (NaN on failure)
    """
    # ODEFormer's float encoder requires float64 — cast everything up front
    _train_ts = np.asarray(train_ts, dtype=np.float64)
    _train_ys = np.asarray(train_ys[0], dtype=np.float64)   # (T_train, D)
    _test_ts  = np.asarray(test_ts,  dtype=np.float64)

    # Fit on training observations (single trajectory)
    odeformer.fit(times=[_train_ts], trajectories=[_train_ys])

    # Build combined grid and integrate from the true IC at t=0
    t_full = np.sort(np.concatenate([_train_ts, _test_ts]))
    x0_arr = np.asarray(x0, dtype=np.float64).ravel()

    try:
        pred = odeformer.predict(times=[t_full], y0=[x0_arr])
        if pred is None:
            raise ValueError("predict returned None (beam search found no valid candidate)")
        pred = np.asarray(pred, dtype=float)
        if not np.all(np.isfinite(pred)):
            raise ValueError("prediction contains NaN or Inf")

        idx_test  = np.searchsorted(t_full, test_ts)
        pred_test = pred[idx_test]
        mse = float(np.mean((pred_test - test_ys[0]) ** 2))
        if verbose:
            print(f"  {label:<25s}  MSE = {mse:.5f}")
        return pred, t_full, mse

    except Exception as exc:
        if verbose:
            print(f"  {label:<25s}  FAILED — {exc}")
        return None, t_full, float("nan")


def eval_ref_fhn_odeformer(
    odeformer,
    full_ts: Array,       # (T,) — full FHN time grid
    full_ys: Array,       # (T, D) — full trajectory (single sample, shape (T, D))
    obs_mask: Array,      # (T,) bool — True = observed (NOT missing)
    x0: Array,            # (D,) — true IC at t=full_ts[0]
    miss_mask: Array,     # (T,) bool — True = missing region (target)
    target_ys: Array,     # (T_miss, D) — ground-truth states in missing region
    label: str = "FHN (missing quadrant)",
    verbose: bool = True,
) -> Tuple[Optional[Array], float]:
    """
    Evaluate ODEFormer on the reference FHN interpolation task.

    Workflow:
      1. Fit ODEFormer on the observed time steps only (those outside the
         missing quadrant x₁>0 & x₂<0).
      2. Integrate the inferred ODE from x0 over the full time grid.
      3. Compute MSE at the missing-region time steps.

    ODEFormer receives a single (possibly non-contiguous) trajectory of observed
    points; it does not need explicit segment splitting because it operates on
    (time, state) pairs independently via its input embedding.

    Parameters
    ----------
    odeformer : pretrained SymbolicTransformerRegressor
    full_ts   : (T,) full time grid
    full_ys   : (T, D) full trajectory (first sample from the npz)
    obs_mask  : (T,) True = observed step
    x0        : (D,) IC at t=full_ts[0]
    miss_mask : (T,) True = missing step (evaluation target)
    target_ys : (T_miss, D) ground-truth at missing steps
    label     : display label

    Returns
    -------
    pred : (T, D) predicted trajectory on the full grid, or None on failure
    mse  : MSE on the missing region (NaN on failure)
    """
    # ODEFormer's float encoder requires float64 — cast everything up front
    _full_ts = np.asarray(full_ts, dtype=np.float64)
    _full_ys = np.asarray(full_ys, dtype=np.float64)

    # Fit only on observed time steps
    obs_ts = _full_ts[obs_mask]
    obs_ys = _full_ys[obs_mask]           # (T_obs, D)
    odeformer.fit(times=[obs_ts], trajectories=[obs_ys])

    x0_arr = np.asarray(x0, dtype=np.float64).ravel()

    try:
        pred = odeformer.predict(times=[_full_ts], y0=[x0_arr])
        if pred is None:
            raise ValueError("predict returned None")
        pred = np.asarray(pred, dtype=float)
        if not np.all(np.isfinite(pred)):
            raise ValueError("prediction contains NaN or Inf")

        mse = float(np.mean((pred[miss_mask] - target_ys) ** 2))
        if verbose:
            print(f"  {label:<25s}  MSE = {mse:.5f}")
        return pred, mse

    except Exception as exc:
        if verbose:
            print(f"  {label:<25s}  FAILED — {exc}")
        return None, float("nan")


# =============================================================================
# 6. Reporting — console, Markdown, LaTeX
# =============================================================================

_W = 82   # default table width

def print_stats_table(
    stats: Dict,
    title: str,
    ctx_sizes: Sequence[int],
    sections: Optional[List[Tuple[str, List]]] = None,
) -> None:
    """
    Print a formatted console table of MSE statistics.

    Two modes depending on ``sections``:

    **Grouped mode** (``sections`` provided) — stats maps ``(group_key, k)``
    to mse_stats dicts. Rows are printed under named section headings,
    useful for comparing multiple IC strategies in one table.

    **Flat mode** (``sections=None``) — stats maps ``k`` directly to
    mse_stats dicts. Rows are printed in ``ctx_sizes`` order with no
    Case label column, suitable for single-strategy results.

    Parameters
    ----------
    stats     : dict mapping (group_key, k) -> mse_stats (grouped) or
                k -> mse_stats (flat)
    title     : table title
    ctx_sizes : ordered list of k values
    sections  : list of (section_label, list_of_keys) for grouped mode;
                omit or pass None for flat mode
    """
    print(f"\n{'═'*_W}")
    print(f" {title}")
    print(f"{'═'*_W}")

    if sections is not None:
        print(f"  {'Case':<22} {'n_ctx':>5}  {'mean':>8} {'std':>8} {'median':>8} {'q05':>8} {'q95':>8}")
        print(f"  {'-'*(_W-2)}")
        for sec_name, keys in sections:
            first = True
            for key in keys:
                s   = stats[key]
                lbl = sec_name if first else ""
                print(f"  {lbl:<22} {key[1]:>5}  "
                      f"{s['mean']:>8.4f} {s['std']:>8.4f} {s['median']:>8.4f} "
                      f"{s['q05']:>8.4f} {s['q95']:>8.4f}")
                first = False
    else:
        print(f"  {'n_ctx':>5}  {'mean':>8} {'std':>8} {'median':>8} {'q05':>8} {'q95':>8}")
        print(f"  {'-'*(_W-2)}")
        for k in ctx_sizes:
            s = stats[k]
            print(f"  {k:>5}  {s['mean']:>8.4f} {s['std']:>8.4f} {s['median']:>8.4f} "
                  f"{s['q05']:>8.4f} {s['q95']:>8.4f}")


def stats_to_markdown(
    stats: Dict,
    title: str,
    sections: List[Tuple[str, List]],
    fmt: str = ".4f",
) -> str:
    """
    Render an MSE stats dict as a Markdown table string.

    Parameters
    ----------
    stats    : dict mapping (group_key, k) -> mse_stats dict
    title    : H3 heading for this block
    sections : same structure as used in print_stats_table
    fmt      : Python format spec for numeric columns (e.g. ".4f")

    Returns
    -------
    Markdown string (heading + table)
    """
    header = (
        f"### {title}\n\n"
        "| Case | n\\_ctx | mean | std | median | q05 | q95 |\n"
        "|------|---------:|-----:|----:|-------:|----:|----:|"
    )
    rows = []
    for sec_name, keys in sections:
        first = True
        for key in keys:
            s   = stats[key]
            lbl = sec_name if first else ""
            vals = " | ".join(format(s[c], fmt) for c in ("mean", "std", "median", "q05", "q95"))
            rows.append(f"| {lbl} | {key[1]} | {vals} |")
            first = False
    return header + "\n" + "\n".join(rows)


def stats_to_latex(
    stats: Dict,
    caption: str,
    sections: List[Tuple[str, List]],
    label: str = "tab:mse",
    fmt: str = ".4f",
) -> str:
    """
    Render an MSE stats dict as a LaTeX table string (booktabs style).

    Parameters
    ----------
    stats    : dict mapping (group_key, k) -> mse_stats dict
    caption  : LaTeX \\caption text
    sections : same structure as print_stats_table
    label    : LaTeX \\label key
    fmt      : Python format spec for numeric columns

    Returns
    -------
    LaTeX string (table environment)
    """
    col_fmt = "llrrrrr"
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
        r"Case & $n_{\mathrm{ctx}}$ & mean & std & median & q05 & q95 \\",
        r"\midrule",
    ]
    for sec_name, keys in sections:
        first = True
        for key in keys:
            s   = stats[key]
            lbl = sec_name.replace("₀", "$_0$").replace("₁", "$_1$") if first else ""
            vals = " & ".join(f"{s[c]:{fmt}}" for c in ("mean", "std", "median", "q05", "q95"))
            lines.append(rf"{lbl} & {key[1]} & {vals} \\")
            first = False
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def save_tables(
    results: Dict[str, Tuple[Dict, str, List]],
    output_path: Path,
    fmt: str = "md",
) -> None:
    """
    Save multiple MSE tables to a single file.

    Parameters
    ----------
    results     : dict mapping table_id -> (stats, title, sections)
    output_path : destination file path (.md or .tex)
    fmt         : "md" for Markdown, "tex" for LaTeX
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as fh:
        if fmt == "md":
            fh.write("# FIM-ODE — MSE Statistics\n\n")
            for _, (stats, title, sections) in results.items():
                fh.write(stats_to_markdown(stats, title, sections))
                fh.write("\n\n")
        elif fmt == "tex":
            for table_id, (stats, caption, sections) in results.items():
                fh.write(stats_to_latex(stats, caption, sections, label=f"tab:{table_id}"))
                fh.write("\n\n")
        else:
            raise ValueError(f"Unknown fmt: {fmt!r}; choose 'md' or 'tex'")

    print(f"Saved: {output_path}")


# =============================================================================
# 7. Plotting
# =============================================================================

_VF_CMAP = "plasma"


def true_vf_grid(
    dynamics: Callable,
    x1_range: Tuple[float, float],
    x2_range: Tuple[float, float],
    n_grid: int = 22,
) -> Tuple[Array, Array, Array, Array]:
    """Evaluate the analytical RHS on an (n_grid × n_grid) meshgrid."""
    x1 = np.linspace(*x1_range, n_grid)
    x2 = np.linspace(*x2_range, n_grid)
    X1, X2 = np.meshgrid(x1, x2)
    U, V = np.zeros_like(X1), np.zeros_like(X1)
    for i in range(n_grid):
        for j in range(n_grid):
            f = dynamics(0, [X1[i, j], X2[i, j]])
            U[i, j], V[i, j] = f[0], f[1]
    return X1, X2, U, V


def _draw_vf_panel(ax, X1, X2, U, V, vmin: float = 0, vmax: float = None):
    """
    Draw speed-coloured streamlines on ax.

    Returns the LineCollection so the caller can attach a colorbar.
    """
    speed = np.sqrt(U**2 + V**2)
    if vmax is None:
        vmax = float(speed.max())
    strm = ax.streamplot(
        X1, X2, U, V,
        color=speed, cmap=_VF_CMAP,
        norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
        linewidth=0.9, density=0.9, arrowsize=0.8, zorder=0,
    )
    return strm.lines


def _draw_trajectories(
    ax,
    ctx_list: Optional[List[Array]] = None,   # list of (T, D) arrays
    sol_xy_full: Optional[Array] = None,       # (L, D)
    gt_xy: Optional[Array] = None,             # (L, D)
    miss_xy: Optional[Array] = None,           # (L, D)
) -> None:
    """Overlay context (orange), FIM-ODE prediction (red), and ground truth (black)."""
    if ctx_list is not None:
        for j, tr in enumerate(ctx_list):
            ax.plot(tr[:, 0], tr[:, 1],
                    color="C1", alpha=0.5, lw=0.8,
                    marker="o", ms=2.5, markerfacecolor="C1", zorder=1,
                    label="context" if j == 0 else "_nolegend_")
    if sol_xy_full is not None:
        ax.plot(sol_xy_full[:, 0], sol_xy_full[:, 1],
                color="C3", lw=1.5, label="FIM-ODE", zorder=3)
    if gt_xy is not None:
        ax.plot(gt_xy[:, 0], gt_xy[:, 1], "k--", lw=1.5, label="GT (target)", zorder=2)
    if miss_xy is not None:
        ax.plot(miss_xy[:, 0], miss_xy[:, 1], "rx", ms=5,
                label="GT (missing region)", zorder=4)


def _style_ax(
    ax,
    title: str,
    xlabel: str = "$x_1$",
    ylabel: str = "$x_2$",
    show_legend: bool = False,
) -> None:
    """Apply common axis styling."""
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    if show_legend:
        ax.legend(fontsize=7, loc="best")


def compute_case(
    label: str,
    ctx_np: Array,           # (k, T, D) or (n_segs, max_len, D) when ctx_mask given
    ctx_times: Array,        # (T,) or (n_segs, max_len) when ctx_mask given
    y0: Array,               # (D,)
    t_eval: Array,           # full integration grid
    x1_range: Tuple[float, float],
    x2_range: Tuple[float, float],
    model,
    device: str,
    n_grid: int = 22,
    ctx_mask: Optional[Array] = None,  # (n_segs, max_len) bool — None for VDP (all valid)
) -> Dict:
    """
    Compute FIM-ODE integration + vector field for one (system, k) case.

    Handles both the standard case (contiguous context, no padding) and the
    FHN case (padded sub-trajectories with an explicit boolean mask).  Pass
    the output of ``split_fhn_trajectories`` as ``ctx_np``, ``ctx_times``, and
    ``ctx_mask`` for FHN; leave ``ctx_mask=None`` for VDP.

    Returns
    -------
    dict with keys: label, ctx_list, sol_xy, fim_grid
    """
    t0 = time.time()

    if ctx_mask is not None:
        # FHN path: padded segments with an explicit validity mask
        traj_t = torch.tensor(ctx_np,   dtype=torch.float32, device=device).unsqueeze(0)
        time_t = torch.tensor(ctx_times, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(-1)
        mask_t = torch.tensor(ctx_mask,  dtype=torch.bool,    device=device).unsqueeze(0).unsqueeze(-1)
        fim_fn = make_fim_ode_fn(model, traj_t, time_t, mask_t)
        sol    = solve_ivp(
            fim_fn,
            t_span=(float(t_eval[0]), float(t_eval[-1])),
            y0=np.asarray(y0, dtype=float),
            t_eval=t_eval,
            method="RK45", rtol=1e-4, atol=1e-6,
        )
        ctx_list = [ctx_np[j, ctx_mask[j]] for j in range(len(ctx_np))]
    else:
        # VDP path: contiguous context, all positions valid
        sol      = predict_and_integrate_ode(model, ctx_np, ctx_times, y0, t_eval, device)
        ctx_list = [ctx_np[j] for j in range(len(ctx_np))]

    fim_grid = predict_vector_field_2d(
        model, ctx_np, ctx_times, x1_range, x2_range, n_grid, device, mask=ctx_mask
    )
    print(f"  {label:45s}  ({time.time()-t0:.1f}s)")
    return dict(label=label, ctx_list=ctx_list, sol_xy=sol.y.T, fim_grid=fim_grid)


def plot_ensemble_phase(ens_vdp_u, ens_vdp_n, ens_fhn) -> None:
    """
    Plot phase-space portraits of the clean trajectories for all three systems.

    Shows the nominal trajectory (black), perturbed-IC trajectories (blue),
    and random-IC trajectories (green) without observation noise.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Clean trajectories (no observation noise)", fontsize=12)

    def _plot(ax, ens, title):
        clean = ens["clean_trajs"]        # (1+2n_ctx, T, D)
        n     = (len(clean) - 1) // 2
        for tr in clean[1 : n + 1]:
            ax.plot(tr[:, 0], tr[:, 1], color="C0", alpha=0.2, lw=0.7)
        for tr in clean[n + 1 :]:
            ax.plot(tr[:, 0], tr[:, 1], color="C2", alpha=0.2, lw=0.7)
        ax.plot(clean[0, :, 0], clean[0, :, 1], color="k", lw=2)
        handles = [
            Line2D([0],[0], color="k",  lw=2,              label="nominal $x_0$"),
            Line2D([0],[0], color="C0", lw=1.5, alpha=0.8, label=f"perturbed $x_0$ ({n})"),
            Line2D([0],[0], color="C2", lw=1.5, alpha=0.8, label=f"random $x_0$ ({n})"),
        ]
        ax.legend(handles=handles, fontsize=7)
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)

    _plot(axes[0], ens_vdp_u, "VDP-uniform")
    _plot(axes[1], ens_vdp_n, "VDP-nonuniform")
    _plot(axes[2], ens_fhn,   "FHN")
    fig.tight_layout()
    plt.show()


def plot_phase_figure(
    system_name: str,
    cases: List[Dict],       # list of compute_case outputs (k=1, k=K pert, [k=K rand])
    gt_xy: Array,            # (T, D) — ground truth trajectory
    true_grid: Tuple,        # output of true_vf_grid
    vmax: float,
    subtitles: List[str],    # one label per case panel
    n_show: int = 0,
    miss_xy: Optional[Array] = None,  # if given, highlights missing region (FHN)
) -> plt.Figure:
    """
    Phase-space figure: one GT-VF panel followed by one panel per case.

    Used for both VDP (typically 3 case panels: k=1, k=K perturbed, k=K random)
    and FHN (typically 2 case panels: k=1, k=K perturbed).  The ``miss_xy``
    argument, when provided, is forwarded to ``_draw_trajectories`` to mark the
    missing-quadrant region (FHN only).

    Parameters
    ----------
    system_name : figure suptitle prefix (e.g. "VDP-uniform", "FHN")
    cases       : list of dicts returned by ``compute_case``
    gt_xy       : ground-truth trajectory for trajectory overlay
    true_grid   : (X1, X2, U, V) from ``true_vf_grid``
    vmax        : colour scale upper bound for stream-plot speed
    subtitles   : one title string per case panel (len must equal len(cases))
    n_show      : noise-realization index shown in the suptitle
    miss_xy     : (T_miss, D) missing-region states to highlight (FHN only)
    """
    n_panels = 1 + len(cases)
    width    = 5.5 * n_panels
    fig, axs = plt.subplots(1, n_panels, figsize=(width, 5.5))
    fig.suptitle(f"{system_name} — phase space  (noise realization n={n_show})", fontsize=12)

    # Panel 0: ground-truth vector field
    lc = _draw_vf_panel(axs[0], *true_grid, vmax=vmax)
    _style_ax(axs[0], "Ground-truth VF")
    plt.colorbar(lc, ax=axs[0], label=r"$\|f(x)\|$", shrink=0.85)

    for col, (case, subtitle) in enumerate(zip(cases, subtitles), start=1):
        ax = axs[col]
        lc = _draw_vf_panel(ax, *case["fim_grid"], vmax=vmax)
        _draw_trajectories(ax,
                           ctx_list=case["ctx_list"],
                           sol_xy_full=case["sol_xy"],
                           gt_xy=gt_xy,
                           miss_xy=miss_xy)
        _style_ax(ax, subtitle, show_legend=(col == n_panels - 1))
        plt.colorbar(lc, ax=ax, label=r"$\|f(x)\|$", shrink=0.85)

    fig.tight_layout()
    return fig


def plot_ref_evaluation(
    ref_vdp_u, ref_vdp_n, ref_fhn,
    pred_vdp_u: Array, t_full_u: Array, mse_vdp_u: float,
    pred_vdp_n: Array, t_full_n: Array, mse_vdp_n: float,
    pred_fhn:   Array, mse_fhn: float,
    ref_vdp_u_test_ts, ref_vdp_u_test_ys,
    ref_vdp_n_test_ts, ref_vdp_n_test_ys,
    fhn_ref_miss_mask, fhn_ref_target_ts, fhn_ref_target_ys,
) -> plt.Figure:
    """
    2×3 time-series figure for the Hegde et al. reference dataset evaluation.
    """
    full_ts = ref_fhn["full_ts"]
    full_ys = ref_fhn["full_ys"][0]
    obs_mask = ~fhn_ref_miss_mask

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(
        "FIM-ODE — single-trajectory evaluation on Hegde et al. reference datasets",
        fontsize=12,
    )

    for dim, row, ylabel in [(0, 0, "$x_1$"), (1, 1, "$x_2$")]:
        # VDP-uniform
        ax = axes[row, 0]
        ax.axvspan(ref_vdp_u["train_ts"][0], ref_vdp_u["train_ts"][-1],
                   alpha=0.07, color="gray", label="training window")
        ax.plot(ref_vdp_u["train_ts"], ref_vdp_u["train_ys"][0, :, dim], "C1o", ms=3, label="context (train)")
        ax.plot(ref_vdp_u_test_ts, ref_vdp_u_test_ys[0, :, dim], "k--", lw=1.5, label="GT (test)")
        ax.plot(t_full_u, pred_vdp_u[:, dim], "C3-", lw=1.3, label="FIM-ODE")
        ax.set_xlabel("t"); ax.set_ylabel(ylabel)
        ax.set_title(f"VDP-uniform — {ylabel}(t)  [MSE = {mse_vdp_u:.4f}]")
        ax.spines[["top", "right"]].set_visible(False)
        if dim == 0: ax.legend(fontsize=7)

        # VDP-nonuniform
        ax = axes[row, 1]
        ax.axvspan(ref_vdp_n["train_ts"][0], ref_vdp_n["train_ts"][-1],
                   alpha=0.07, color="gray", label="training window")
        ax.plot(ref_vdp_n["train_ts"], ref_vdp_n["train_ys"][0, :, dim], "C1o", ms=3, label="context (train)")
        ax.plot(ref_vdp_n_test_ts, ref_vdp_n_test_ys[0, :, dim], "k--", lw=1.5, label="GT (test)")
        ax.plot(t_full_n, pred_vdp_n[:, dim], "C3-", lw=1.3, label="FIM-ODE")
        ax.set_xlabel("t"); ax.set_ylabel(ylabel)
        ax.set_title(f"VDP-nonuniform — {ylabel}(t)  [MSE = {mse_vdp_n:.4f}]")
        ax.spines[["top", "right"]].set_visible(False)
        if dim == 0: ax.legend(fontsize=7)

        # FHN
        ax = axes[row, 2]
        ax.plot(full_ts[obs_mask], full_ys[obs_mask, dim], "C1o", ms=3, label="context (observed)")
        ax.plot(fhn_ref_target_ts, fhn_ref_target_ys[:, dim], "k--", lw=1.5, label="GT (missing)")
        ax.plot(full_ts, pred_fhn[:, dim], "C3-", lw=1.3, label="FIM-ODE")
        ax.axvspan(fhn_ref_target_ts[0], fhn_ref_target_ts[-1], alpha=0.07, color="red", label="missing region")
        ax.set_xlabel("t"); ax.set_ylabel(ylabel)
        ax.set_title(f"FHN — {ylabel}(t)  [MSE = {mse_fhn:.4f}]")
        ax.spines[["top", "right"]].set_visible(False)
        if dim == 0: ax.legend(fontsize=7)

    fig.tight_layout()
    return fig


# =============================================================================
# 8. Autoregressive context expansion
# =============================================================================

def autoregressive_predict(
    model,
    init_ctx_traj: Array,    # (1, T_train, D) — initial noisy context
    init_ctx_times: Array,   # (T_train,) — context observation times
    y0: Array,               # (D,) — IC at t_full[0]
    t_full: Array,           # (T_full,) — complete time grid [t_start, t_end]
    context_length: int,     # index splitting training from test in t_full
    n_steps: int,            # number of test steps between context updates
    device: str,
) -> Tuple[Array, Array, List[Tuple[float, Array]]]:
    """
    Autoregressive ODE integration with progressive context expansion.

    Integrates from y0 in chunks of n_steps time steps. After each chunk the
    predicted endpoint is appended to the context and the model is re-encoded,
    so later segments benefit from self-generated observations in the test domain.

    This allows the model to progressively extend its context window beyond the
    training interval. Smaller n_steps → more context updates, lower error
    accumulation, higher computational cost.

    Parameters
    ----------
    model          : FimOdeonUnified instance
    init_ctx_traj  : (1, T_train, D) — noisy training-domain context
    init_ctx_times : (T_train,) — observation times
    y0             : (D,) — initial condition at t_full[0]
    t_full         : (T_full,) — full evaluation grid (training + test)
    context_length : index separating training [0:context_length] from test
    n_steps        : test steps per integration chunk
    device         : torch device string

    Returns
    -------
    pred_times  : (T_full,) — time stamps (mirrors t_full)
    pred_states : (T_full, D) — predicted states
    checkpoints : list of (t, x̂) pairs appended to context
    """
    ctx_traj  = init_ctx_traj.copy()
    ctx_times = init_ctx_times.copy()

    pred_times_list:  List[Array] = []
    pred_states_list: List[Array] = []
    checkpoints: List[Tuple[float, Array]] = []

    # ── Segment 0: integrate from y0 through training + first n_steps test steps ──
    seg_end_idx = context_length + n_steps
    t_seg       = t_full[:seg_end_idx]
    sol         = predict_and_integrate_ode(model, ctx_traj, ctx_times, y0, t_seg, device)
    seg_states  = sol.y.T

    pred_times_list.append(t_seg)
    pred_states_list.append(seg_states)

    current_t = float(t_seg[-1])
    current_y = seg_states[-1].copy()
    checkpoints.append((current_t, current_y))
    ctx_traj  = np.concatenate([ctx_traj, current_y[np.newaxis, np.newaxis]], axis=1)
    ctx_times = np.append(ctx_times, current_t)

    # ── Subsequent segments ───────────────────────────────────────────────
    test_ptr = n_steps
    while test_ptr < len(t_full) - context_length:
        next_ptr  = min(test_ptr + n_steps, len(t_full) - context_length)
        start_idx = context_length + test_ptr - 1
        end_idx   = context_length + next_ptr
        t_seg     = t_full[start_idx:end_idx]

        sol        = predict_and_integrate_ode(model, ctx_traj, ctx_times, current_y, t_seg, device)
        seg_states = sol.y.T

        # Skip first point (= last point of previous segment)
        pred_times_list.append(t_seg[1:])
        pred_states_list.append(seg_states[1:])

        current_t = float(t_seg[-1])
        current_y = seg_states[-1].copy()
        test_ptr  = next_ptr

        if test_ptr < len(t_full) - context_length:
            checkpoints.append((current_t, current_y))
            ctx_traj  = np.concatenate([ctx_traj, current_y[np.newaxis, np.newaxis]], axis=1)
            ctx_times = np.append(ctx_times, current_t)

    pred_times  = np.concatenate(pred_times_list)
    pred_states = np.concatenate(pred_states_list)

    return pred_times, pred_states, checkpoints


# =============================================================================
# 9. ODEBench utilities (fixed-point & stability analysis)
# =============================================================================

_ODEBENCH_JSON = Path(__file__).parent / "odebench" / "data" / "strogatz_extended.json"


def load_odebench_system(
    eq_id: int,
    noise_sigma: float = 0.05,
    subsample_fraction: float = 0.5,
    const_idx: int = 0,
    rng: Optional[np.random.Generator] = None,
    data_path: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load one system from ODEBench, add multiplicative noise and subsample.

    Parameters
    ----------
    eq_id             : ODEBench system id (integer).
    noise_sigma       : Std-dev of multiplicative Gaussian noise:
                        ``y_noisy = y * (1 + sigma * N(0,1))``.
    subsample_fraction: Fraction of time points to keep (uniform subsampling).
    const_idx         : Index into the ``consts`` / ``solutions`` list.
    rng               : NumPy random generator (created if None).
    data_path         : Override path to ``strogatz_extended.json``.

    Returns
    -------
    times  : (N_ic, T_sub, 1) — observation times, shape matches FIM convention.
    trajs  : (N_ic, T_sub, D) — noisy, subsampled trajectories.
    meta   : dict with keys ``eq_id``, ``eq``, ``dim``, ``consts``,
             ``n_pts_original``, ``t_end``.
    """
    if rng is None:
        rng = np.random.default_rng()

    path = data_path if data_path is not None else _ODEBENCH_JSON
    with open(path) as f:
        db = json.load(f)

    # Locate the entry by id
    entry = next((item for item in db if item["id"] == eq_id), None)
    if entry is None:
        raise ValueError(f"ODEBench system id={eq_id} not found in {path}.")

    solutions_for_const = entry["solutions"][const_idx]  # list of per-IC dicts

    ic_times_list: List[np.ndarray] = []
    ic_trajs_list: List[np.ndarray] = []

    for sol in solutions_for_const:
        if not sol.get("success", True):
            continue
        t_full = np.asarray(sol["t"])            # (T,)
        y_full = np.asarray(sol["y"]).T          # (T, D)

        T = len(t_full)
        n_sub = max(2, int(round(T * subsample_fraction)))
        idx = np.linspace(0, T - 1, n_sub, dtype=int)

        t_sub = t_full[idx]                      # (T_sub,)
        y_sub = y_full[idx]                      # (T_sub, D)

        noise = rng.standard_normal(y_sub.shape)
        y_noisy = y_sub * (1.0 + noise_sigma * noise)

        ic_times_list.append(t_sub[:, np.newaxis])   # (T_sub, 1)
        ic_trajs_list.append(y_noisy)                 # (T_sub, D)

    times = np.stack(ic_times_list, axis=0)           # (N_ic, T_sub, 1)
    trajs = np.stack(ic_trajs_list, axis=0)           # (N_ic, T_sub, D)

    meta = {
        "eq_id":          eq_id,
        "eq":             entry["eq"],
        "dim":            entry["dim"],
        "consts":         entry["consts"][const_idx] if entry["consts"] else {},
        "n_pts_original": T,
        "t_end":          float(t_full[-1]),
    }
    return times, trajs, meta


def make_fim_vf_fn(
    model,
    trajs: np.ndarray,
    times: np.ndarray,
    device: str = "cpu",
):
    """Return a vector-field callable for a fitted FIM-ODE model.

    Context trajectories are encoded once; subsequent calls only run the
    lightweight function-decoding path.

    Parameters
    ----------
    model  : FimOdeonUnified instance.
    trajs  : (N_ic, T, D) numpy array — context trajectories.
    times  : (N_ic, T, 1) numpy array — observation times.
    device : torch device string.

    Returns
    -------
    vf : numpy callable ``(N, D) -> (N, D)``.
    """
    dev = torch.device(device)
    D = trajs.shape[-1]

    # Model expects (B, N_traj, T_obs, D) — add outer batch dim of 1.
    traj_t  = torch.tensor(trajs, dtype=torch.float32, device=dev).unsqueeze(0)
    times_t = torch.tensor(times, dtype=torch.float32, device=dev).unsqueeze(0)
    mask_t  = torch.ones(*traj_t.shape[:-1], 1, dtype=torch.bool, device=dev)

    with torch.no_grad():
        wrapped_D, feature_mask, concept = model.trajectory_encoding(traj_t, times_t, mask_t)

    def vf(x: np.ndarray) -> np.ndarray:
        """Numpy callable (N, D) -> (N, D)."""
        x_t = torch.tensor(x[np.newaxis], dtype=torch.float32, device=dev)
        with torch.no_grad():
            loc      = model.pad_if_necessary(x_t)
            loc_norm = model.spatial_norm.normalization_map(loc, concept._states_norm_stats)
            out      = model.function_decoding(loc_norm, feature_mask, wrapped_D, concept)
            pred     = model.get_prediction_for_eval(out)[0, :, :D]  # (N, D)
        return pred.cpu().numpy()

    return vf


def build_odeformer_vf_fn(
    eq_str: str,
    D: int,
) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Parse an ODEFormer prediction string into a numpy vector-field callable.

    ODEFormer equations use ``x_0, x_1, ...`` as variable names and ``|``
    as a separator between dimensions.

    Parameters
    ----------
    eq_str : Equation string returned by ODEFormer, e.g.
             ``"x_1 | -x_0 - 0.43*(x_0**2 - 1)*x_1"``.
    D      : Number of state dimensions.

    Returns
    -------
    Callable ``f(x: (N, D)) -> (N, D)``, or ``None`` if parsing fails.
    """
    import sympy as sp

    symbols = [sp.Symbol(f"x_{i}") for i in range(D)]
    parts   = eq_str.split("|")
    if len(parts) != D:
        return None

    try:
        exprs = [sp.sympify(p.strip()) for p in parts]
    except Exception:
        return None

    # Build one lambdified function per dimension
    funcs = []
    for expr in exprs:
        f_lam = sp.lambdify(symbols, expr, modules="numpy")
        funcs.append(f_lam)

    def vf(x: np.ndarray) -> np.ndarray:
        """Evaluate ODEFormer vector field at locations x (N, D) → (N, D)."""
        cols = [x[:, i] for i in range(D)]
        outs = []
        for f in funcs:
            v = f(*cols)
            if np.isscalar(v):
                v = np.full(x.shape[0], float(v))
            outs.append(v)
        return np.stack(outs, axis=-1)

    return vf


def find_fixed_points(
    f: Callable[[np.ndarray], np.ndarray],
    D: int,
    n_starts: int = 50,
    x_range: Tuple[float, float] = (-5.0, 5.0),
    tol_rel: float = 0.05,
    dedup_dist: float = 0.1,
    n_grid: int = 40,
    top_k: Optional[int] = 1,
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[np.ndarray, float]]:
    """Find fixed points of ``f`` via grid-seeded multi-start minimisation of ``‖f(x)‖²``.

    Uses ``scipy.optimize.minimize`` (L-BFGS-B) rather than ``fsolve``, which
    is more robust for approximate vector fields such as neural networks where
    the VF may never reach exactly zero at the true fixed point.

    Strategy
    --------
    1. Evaluate ``‖f‖`` on a coarse ``n_grid^D`` grid.  The median value sets
       the scale; candidates are accepted when ``‖f(x*)‖ < tol_rel * median``.
       This makes the threshold relative to the VF magnitude in the given
       phase-space region, so the same ``tol_rel`` works across systems with
       very different scales.
    2. Seed L-BFGS-B from the ``n_starts`` grid cells with smallest ``‖f‖``.
    3. Accept the refined result if it passes the relative tolerance check.
    4. De-duplicate: discard candidates within ``dedup_dist`` of an already
       accepted point.

    Parameters
    ----------
    f          : Callable ``(N, D) -> (N, D)`` — the vector field.
    D          : State dimension.
    n_starts   : Number of best grid cells used as L-BFGS-B seeds.
    x_range    : Uniform sampling box ``(low, high)``.
    tol_rel    : Accept a candidate when ``‖f(x*)‖ < tol_rel * median‖f‖``.
                 ``0.05`` (5 % of median) works well for neural-network VFs.
    dedup_dist : Minimum Euclidean distance between distinct fixed points.
    n_grid     : Side length of the coarse evaluation grid (``n_grid^D`` pts).
    top_k      : If set, return only the ``top_k`` candidates with lowest
                 residual. ``top_k=1`` returns just the single best fixed
                 point. ``None`` returns all accepted candidates.
    rng        : NumPy random generator (unused but kept for API compatibility).

    Returns
    -------
    List of ``(point, residual)`` tuples, sorted by residual ascending.
    ``point`` has shape ``(D,)``; ``residual`` is ``‖f(point)‖`` (L2 norm).
    """
    from scipy.optimize import minimize

    lo, hi = x_range

    def residual_sq(x1d: np.ndarray) -> float:
        r = f(x1d[np.newaxis])[0]
        return float(np.dot(r, r))

    # ── 1. Coarse grid evaluation ─────────────────────────────────────────
    axes  = [np.linspace(lo, hi, n_grid) for _ in range(D)]
    grid  = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, D)
    norms = np.linalg.norm(f(grid), axis=-1)       # (n_grid^D,)
    tol   = tol_rel * float(np.median(norms))      # absolute threshold, scale-adaptive

    top_idx = np.argsort(norms)[:n_starts]
    seeds   = grid[top_idx]

    # ── 2. Local refinement from each seed ───────────────────────────────
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
        is_new = all(np.linalg.norm(sol - fp) > dedup_dist for fp in fps)
        if is_new:
            fps.append(sol.copy())

    # Sort by residual and optionally truncate
    fps_with_res = sorted(
        [(fp, float(np.linalg.norm(f(fp[np.newaxis])[0]))) for fp in fps],
        key=lambda t: t[1],
    )
    if top_k is not None:
        fps_with_res = fps_with_res[:top_k]
    return fps_with_res


def numerical_jacobian(
    f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Compute the Jacobian of ``f`` at point ``x`` via central differences.

    Parameters
    ----------
    f   : Callable ``(N, D) -> (N, D)`` — the vector field.
    x   : ``(D,)`` point at which to evaluate.
    eps : Finite-difference step size.

    Returns
    -------
    J   : ``(D, D)`` Jacobian matrix (``J[i, j] = ∂f_i/∂x_j``).
    """
    D = len(x)
    J = np.zeros((D, D))
    for j in range(D):
        xp, xm = x.copy(), x.copy()
        xp[j] += eps
        xm[j] -= eps
        fp = f(xp[np.newaxis])[0]
        fm = f(xm[np.newaxis])[0]
        J[:, j] = (fp - fm) / (2 * eps)
    return J


def symbolic_jacobian(
    eq_str: str,
    D: int,
    x: np.ndarray,
) -> np.ndarray:
    """Evaluate the Jacobian of an ODEFormer symbolic equation at point ``x``.

    More accurate than finite differences because it uses exact symbolic
    differentiation via SymPy.

    Parameters
    ----------
    eq_str : ODEFormer equation string (``|``-separated, ``x_i`` variables,
             ``^`` exponentiation).  Constants must already be substituted.
    D      : State dimension.
    x      : ``(D,)`` point at which to evaluate.

    Returns
    -------
    J : ``(D, D)`` Jacobian matrix.
    """
    import sympy as sp

    eq_str = eq_str.replace('^', '**')
    parts  = [p.strip() for p in eq_str.split('|')]
    xs     = [sp.Symbol(f'x_{i}') for i in range(D)]
    exprs  = [sp.sympify(p) for p in parts]
    J_sym  = sp.Matrix([[sp.diff(e, xi) for xi in xs] for e in exprs])
    subs   = {xs[i]: float(x[i]) for i in range(D)}
    return np.array(J_sym.subs(subs).evalf().tolist(), dtype=float)


def stability_analysis(J: np.ndarray) -> dict:
    """Characterise a fixed point from its Jacobian.

    Parameters
    ----------
    J : ``(D, D)`` Jacobian evaluated at the fixed point.

    Returns
    -------
    dict with keys:
    * ``eigenvalues``  : complex array, shape ``(D,)``
    * ``max_real``     : maximum real part of eigenvalues
    * ``label``        : human-readable stability string
    """
    evals = np.linalg.eigvals(J)
    max_re = float(np.max(evals.real))
    min_re = float(np.min(evals.real))

    has_imag = np.any(np.abs(evals.imag) > 1e-8)

    if max_re < -1e-8:
        label = "stable spiral" if has_imag else "stable node"
    elif min_re > 1e-8:
        label = "unstable spiral" if has_imag else "unstable node"
    elif max_re > 1e-8 and min_re < -1e-8:
        label = "saddle"
    else:
        label = "centre / marginal"

    return {"eigenvalues": evals, "max_real": max_re, "label": label}


# =============================================================================
# 10. MoCap evaluation helpers
# =============================================================================

def mocap_pca_to_50d(
    traj_norm: Array,
    pca,
    pca_normalize,
    n_pca_dims: int = 3,
) -> Array:
    """
    Convert a predicted trajectory in normalized PCA space to 50D joint-angle space.

    FIM-ODE can only model the first ``n_pca_dims`` PCA components (≤ 3).
    The remaining 5 - n_pca_dims components are set to zero before back-projection.

    Parameters
    ----------
    traj_norm  : (T, n_pca_dims) or (n, T, n_pca_dims) — predicted trajectory
                 in the normalized PCA subspace (output of FIM-ODE integration).
    pca        : sklearn PCA fitted on 50D data; ``pca.components_.shape == (5, 50)``
    pca_normalize : Normalize with ``mean/std`` of shape ``(1, 1, 5)``
    n_pca_dims : number of PCA components that were actually modelled (≤ 5)

    Returns
    -------
    Array of shape (T, 50) or (n, T, 50) — raw joint-angle trajectories.
    """
    single = traj_norm.ndim == 2
    if single:
        traj_norm = traj_norm[np.newaxis]          # (1, T, n_pca_dims)
    n, T, _ = traj_norm.shape

    # Pad to full 5D (zeros for the un-modelled components)
    if n_pca_dims < 5:
        pad = np.zeros((n, T, 5 - n_pca_dims), dtype=traj_norm.dtype)
        traj_norm_5d = np.concatenate([traj_norm, pad], axis=-1)   # (n, T, 5)
    else:
        traj_norm_5d = traj_norm

    # Unnormalize: (n, T, 5) → PCA coordinate space
    traj_pca = pca_normalize.inverse(traj_norm_5d)                 # (n, T, 5)

    # Inverse PCA: (n*T, 5) → (n*T, 50) → (n, T, 50)
    traj_50d = pca.inverse_transform(
        traj_pca.reshape(-1, 5)
    ).reshape(n, T, 50)

    return traj_50d[0] if single else traj_50d


def eval_mocap_zero_shot(
    model,
    dataset,
    device: str = "cpu",
    n_pca_dims: int = 3,
) -> Tuple[Array, float]:
    """
    Zero-shot MoCap evaluation.

    Encodes all training paths as shared context, then integrates each test path
    from its own initial condition.  MSE is computed in the original 50D
    joint-angle space after back-projecting through the PCA.

    Un-modelled PCA components (dims ``n_pca_dims`` … 4) are set to zero when
    projecting back to 50D, so any variance they explain is counted as error.

    Parameters
    ----------
    model      : FimOdeonUnified (should be in eval mode)
    dataset    : MocapDataset loaded from pkl; must have
                 ``.trn.ys``, ``.trn.ts``, ``.tst.ys``, ``.tst.ts``,
                 ``.pca``, ``.pca_normalize``
    device     : torch device string
    n_pca_dims : PCA components to model (≤ 3 for the base FIM-ODE checkpoint)

    Returns
    -------
    preds_50d : (n_tst, T_tst, 50)  predicted joint-angle trajectories
    mse_50d   : float — mean MSE over all n_tst test paths in 50D space
    """
    trn_ys_full = np.asarray(dataset.trn.ys, dtype=np.float32)   # (n_trn, T_trn, 5)
    trn_ts      = np.asarray(dataset.trn.ts, dtype=np.float32)
    if trn_ts.ndim == 2:
        trn_ts = trn_ts[0]

    tst_ys_full = np.asarray(dataset.tst.ys, dtype=np.float32)   # (n_tst, T_tst, 5)
    tst_ts      = np.asarray(dataset.tst.ts, dtype=np.float32)
    if tst_ts.ndim == 2:
        tst_ts = tst_ts[0]

    pca           = dataset.pca
    pca_normalize = dataset.pca_normalize

    # Context uses only the first n_pca_dims components
    trn_ys = trn_ys_full[:, :, :n_pca_dims]   # (n_trn, T_trn, n_pca_dims)

    n_tst     = len(tst_ys_full)
    preds_50d = []
    mses      = []

    model.eval()
    for i in range(n_tst):
        y0  = tst_ys_full[i, 0, :n_pca_dims].astype(float)   # (n_pca_dims,)
        sol = predict_and_integrate_ode(model, trn_ys, trn_ts, y0, tst_ts, device)

        if not sol.success:
            print(f"  [warn] test path {i}: integration failed — {sol.message}")
            preds_50d.append(np.full((len(tst_ts), 50), np.nan))
            continue

        pred_3d  = sol.y.T                                       # (T_tst, n_pca_dims)
        pred_50d = mocap_pca_to_50d(pred_3d, pca, pca_normalize, n_pca_dims)   # (T_tst, 50)
        true_50d = mocap_pca_to_50d(
            tst_ys_full[i], pca, pca_normalize, n_pca_dims=5    # use all 5 for GT
        )                                                        # (T_tst, 50)

        preds_50d.append(pred_50d)
        mses.append(float(np.mean((pred_50d - true_50d) ** 2)))

    return np.stack(preds_50d, axis=0), float(np.mean(mses))


# =============================================================================
# CLI entry point
# =============================================================================

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FIM-ODE experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",  type=Path, required=True,
                   help="Path to data/ode/hedge_gp_odes_data/")
    p.add_argument("--output-dir", type=Path, default=Path("results/fim-ode"),
                   help="Directory to save tables and figures")
    p.add_argument("--fmt", choices=["md", "tex"], default="md",
                   help="Output format for tables")
    p.add_argument("--device", default="cpu",
                   help="PyTorch device string")
    p.add_argument("--n-runs", type=int, default=100,
                   help="Number of independent realisations for joint experiments")
    return p


if __name__ == "__main__":
    from fim.models.fim_ode import FIMODE as FimOdeonUnified
    from fim.models.fim_ode_trainer import FIMODEConfig as TrainingWrapperConfiguration
    import json
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    args = _build_arg_parser().parse_args()

    # ── Load model ────────────────────────────────────────────────────────
    REPO_ID   = "FIM4Science/fim-ode"
    SUBFOLDER = "base_model/checkpoints/best-model"
    config_path  = hf_hub_download(REPO_ID, f"{SUBFOLDER}/config.json")
    weights_path = hf_hub_download(REPO_ID, f"{SUBFOLDER}/model.safetensors")
    with open(config_path) as f:
        config_dict = json.load(f)
    config = TrainingWrapperConfiguration()
    config.model_config = config_dict["model_config"]
    config.train_config = config_dict["train_config"]
    model = FimOdeonUnified(config)
    state_dict = load_file(weights_path, device=args.device)
    state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(args.device).eval()
    print(f"Model loaded  ({sum(p.numel() for p in model.parameters()):,} parameters)")

    # ── Load reference datasets ───────────────────────────────────────────
    data_dir = Path(args.data_dir)
    ref_vdp_u = np.load(data_dir / "vdp_uniform.npz")
    ref_vdp_n = np.load(data_dir / "vdp_nonuniform.npz")
    ref_fhn   = np.load(data_dir / "fhn_interpolation.npz")

    # ── Preprocess ────────────────────────────────────────────────────────
    ref_vdp_u_test_ts, ref_vdp_u_test_ys = preprocess_ref_vdp_uniform(ref_vdp_u)
    ref_vdp_n_test_ts, ref_vdp_n_test_ys = preprocess_ref_vdp_nonuniform(ref_vdp_n)
    ctx_trajs_fhn, ctx_times_fhn, ctx_mask_fhn, miss_mask_fhn, tgt_ts_fhn, tgt_ys_fhn = \
        preprocess_ref_fhn(ref_fhn)

    # ── Evaluate ──────────────────────────────────────────────────────────
    print("\n=== Reference dataset evaluation ===\n")
    pred_u, t_full_u, mse_u = eval_ref_vdp(
        model, args.device,
        ref_vdp_u["train_ts"], ref_vdp_u["train_ys"], ref_vdp_u["x0"],
        ref_vdp_u_test_ts, ref_vdp_u_test_ys, "VDP-uniform",
    )
    pred_n, t_full_n, mse_n = eval_ref_vdp(
        model, args.device,
        ref_vdp_n["train_ts"], ref_vdp_n["train_ys"], ref_vdp_n["x0"],
        ref_vdp_n_test_ts, ref_vdp_n_test_ys, "VDP-nonuniform",
    )
    pred_f, mse_f = eval_ref_fhn(
        model, args.device,
        ctx_trajs_fhn, ctx_times_fhn, ctx_mask_fhn,
        ref_fhn["full_ts"], ref_fhn["x0"],
        miss_mask_fhn, tgt_ys_fhn,
    )

    print(f"\nVDP-uniform MSE    : {mse_u:.5f}")
    print(f"VDP-nonuniform MSE : {mse_n:.5f}")
    print(f"FHN MSE            : {mse_f:.5f}")
