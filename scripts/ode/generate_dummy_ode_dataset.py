"""
Generate a simple dummy training dataset for FIMODE from three ODE systems:
Van der Pol (VDP, 2D), FitzHugh-Nagumo (FHN, 2D), and Lorenz (3D).

Each sample corresponds to a *distinct ODE instance* — the system parameters
(e.g. mu for VDP, sigma/rho/beta for Lorenz) are drawn randomly per sample,
so the dataset covers a family of related systems rather than a single fixed one.
Each sample then contains T trajectories from different initial conditions of
that same parameterised system, observed at N randomly chosen time points.
Observations are stored clean (no noise); noise and temporal dropout are applied
on-the-fly by DataCorruptionModel during training.

Output
------
data/ode/<subdir>/{vdp,fhn,lorenz}.pt

Each file is a dict with these keys (S plays the role of batch dim B at
training time):

    obs_values             float32  [S, T, N, D]  noisy trajectory observations
    obs_times              float32  [S, T, N, 1]  observation times
    obs_mask               bool     [S, T, N, 1]  all True (no structural missingness)
    locations              float32  [S, L, D]     random query locations
    drift_at_locations     float32  [S, L, D]     ground-truth drift at query locations
    drift_at_observations  float32  [S, T, N, D]  ground-truth drift at the clean
                                                   trajectory states (sampled at
                                                   observation times, before noise)
    dimension_mask         bool     [S, 1, D]     all True

Note: "drift_at_observations" stores f(x_clean), NOT f(x_noisy).  The name
follows the convention used throughout the rest of the FIM codebase.  Internally
we call the corresponding variables drift_traj / drift_at_trajectory_list to
make this distinction clear.

Usage
-----
    python scripts/ode/generate_dummy_ode_dataset.py          # default settings
    python scripts/ode/generate_dummy_ode_dataset.py --help   # show options
"""
from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import torch
from scipy.integrate import solve_ivp


# ─────────────────────────────────────────────────────────────
# Vector field definitions  (all accept extra params via args)
# ─────────────────────────────────────────────────────────────

def _vdp(t: float, y: np.ndarray, mu: float) -> np.ndarray:
    """Van der Pol.  dx1/dt = x2,  dx2/dt = mu*(1-x1²)*x2 - x1."""
    x1, x2 = y
    return np.array([x2, mu * (1 - x1**2) * x2 - x1])


def _fhn(t: float, y: np.ndarray, a: float, b: float, tau: float) -> np.ndarray:
    """FitzHugh-Nagumo.  dx1/dt = x1 - x1³/3 + x2,  dx2/dt = (a - x1 - b*x2)/tau."""
    x1, x2 = y
    return np.array([x1 - x1**3 / 3 + x2, (a - x1 - b * x2) / tau])


def _lorenz(t: float, y: np.ndarray, sigma: float, rho: float, beta: float) -> np.ndarray:
    """Lorenz-63.  dx/dt = σ(y-x),  dy/dt = x(ρ-z)-y,  dz/dt = xy - βz."""
    x, yy, z = y
    return np.array([sigma * (yy - x), x * (rho - z) - yy, x * yy - beta * z])


def _sample_vdp_params(rng: np.random.Generator) -> dict:
    """Sample Van der Pol parameters.  mu controls limit-cycle stiffness."""
    return {"mu": float(rng.uniform(0.1, 3.0))}


def _sample_fhn_params(rng: np.random.Generator) -> dict:
    """Sample FitzHugh-Nagumo parameters.
    a  — offset in recovery equation  (affects fixed point)
    b  — recovery damping             (0 < b < 1 for oscillations)
    tau — timescale separation        (larger = slower recovery)
    """
    return {
        "a":   float(rng.uniform(0.5,  1.5)),
        "b":   float(rng.uniform(0.1,  0.8)),
        "tau": float(rng.uniform(8.0, 20.0)),
    }


def _sample_lorenz_params(rng: np.random.Generator) -> dict:
    """Sample Lorenz-63 parameters around the classic chaotic regime."""
    return {
        "sigma": float(rng.uniform(8.0,  12.0)),
        "rho":   float(rng.uniform(24.0, 32.0)),
        "beta":  float(rng.uniform(2.0,   3.0)),
    }


SYSTEMS: Dict[str, dict] = {
    "vdp": {
        "fn":            _vdp,
        "sample_params": _sample_vdp_params,
        "dim":           2,
        "ic_scale":      2.0,
        "t_max":         10.0,
    },
    "fhn": {
        "fn":            _fhn,
        "sample_params": _sample_fhn_params,
        "dim":           2,
        "ic_scale":      1.5,
        "t_max":         20.0,
    },
    "lorenz": {
        "fn":            _lorenz,
        "sample_params": _sample_lorenz_params,
        "dim":           3,
        "ic_scale":      5.0,
        "t_max":         5.0,
    },
}


# ─────────────────────────────────────────────────────────────
# Core generation
# ─────────────────────────────────────────────────────────────

def _solve_and_sample(
    fn: Callable,
    ic: np.ndarray,
    t_max: float,
    n_obs: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate the ODE and subsample N random observation times.

    Parameters
    ----------
    fn    : ODE right-hand side, already bound to its parameters via partial()
    ic    : initial condition  (D,)
    t_max : integration horizon
    n_obs : number of observation time points to subsample

    Returns
    -------
    times  : (N,)   sorted observation times in [0, t_max]
    states : (N, D) clean state values at those times
    """
    t_dense = np.linspace(0, t_max, max(1000, n_obs * 20))
    sol = solve_ivp(fn, [0, t_max], ic, t_eval=t_dense, method="RK45",
                    rtol=1e-7, atol=1e-9, dense_output=False)
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")

    idx = np.sort(rng.choice(len(t_dense), size=n_obs, replace=False))
    return t_dense[idx], sol.y[:, idx].T   # (N,), (N, D)


def generate_dataset(
    system: str,
    num_samples: int,
    num_trajectories: int,
    num_obs: int,
    num_locations: int,
    seed: int,
) -> dict:
    """Generate one dataset dict for the given system family.

    Observations are stored as clean ODE states (no noise).  Noise and temporal
    dropout are applied on-the-fly by DataCorruptionModel during training.

    Parameters
    ----------
    system           : one of "vdp", "fhn", "lorenz"
    num_samples      : S — number of independent ODE instances (each with its
                       own randomly drawn parameters)
    num_trajectories : T — trajectories per instance (different ICs, same params)
    num_obs          : N — observation time points per trajectory (random subset)
    num_locations    : L — random query locations per instance
    seed             : RNG seed for reproducibility
    """
    cfg      = SYSTEMS[system]
    D        = cfg["dim"]
    t_max    = cfg["t_max"]
    ic_scale = cfg["ic_scale"]

    rng = np.random.default_rng(seed)

    obs_values_list            = []
    obs_times_list             = []
    drift_at_trajectory_list   = []   # drift at clean trajectory states
    locations_list             = []
    drift_at_locations_list    = []

    for _ in range(num_samples):
        # Each sample is a distinct ODE: draw parameters independently
        params = cfg["sample_params"](rng)
        fn     = partial(cfg["fn"], **params)

        # Sample T initial conditions for this instance
        ics = rng.normal(0, ic_scale, size=(num_trajectories, D))

        traj_obs    = []   # (T, N, D)  noisy observations
        traj_times  = []   # (T, N)     observation times
        drift_traj  = []   # (T, N, D)  drift at clean trajectory states

        for ic in ics:
            times, states = _solve_and_sample(fn, ic, t_max, num_obs, rng)

            # Store clean states — corruption (noise + dropout) is applied on-the-fly
            # by DataCorruptionModel during training.
            obs = states

            # Ground-truth drift evaluated at the clean states
            drift = np.stack([fn(times[i], states[i]) for i in range(num_obs)])

            traj_obs.append(obs)
            traj_times.append(times)
            drift_traj.append(drift)

        obs_arr    = np.stack(traj_obs,   axis=0)   # (T, N, D)
        times_arr  = np.stack(traj_times, axis=0)   # (T, N)
        drift_arr  = np.stack(drift_traj, axis=0)   # (T, N, D)

        # Query locations: uniformly sampled from the bounding box of observed states
        flat_states = obs_arr.reshape(-1, D)
        lo, hi = flat_states.min(axis=0), flat_states.max(axis=0)
        locs   = rng.uniform(lo, hi, size=(num_locations, D))
        drift_locs = np.stack([fn(0.0, locs[i]) for i in range(num_locations)])

        obs_values_list.append(obs_arr)
        obs_times_list.append(times_arr)
        drift_at_trajectory_list.append(drift_arr)
        locations_list.append(locs)
        drift_at_locations_list.append(drift_locs)

    def _t(a, dtype=torch.float32):
        return torch.tensor(np.stack(a, axis=0), dtype=dtype)

    obs_values = _t(obs_values_list)                         # [S, T, N, D]
    obs_times  = _t(obs_times_list).unsqueeze(-1)            # [S, T, N, 1]
    drift_traj = _t(drift_at_trajectory_list)                # [S, T, N, D]
    locations  = _t(locations_list)                          # [S, L, D]
    drift_locs = _t(drift_at_locations_list)                 # [S, L, D]

    S, T, N, _ = obs_values.shape
    obs_mask       = torch.ones(S, T, N, 1, dtype=torch.bool)
    dimension_mask = torch.ones(S, 1, D,   dtype=torch.bool)

    return {
        "obs_values":            obs_values,
        "obs_times":             obs_times,
        "obs_mask":              obs_mask,
        "locations":             locations,
        "drift_at_locations":    drift_locs,
        "drift_at_observations": drift_traj,   # key follows FIM codebase convention
        "dimension_mask":        dimension_mask,
    }


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate dummy FIMODE training data.")
    parser.add_argument("--systems",       nargs="+", default=["vdp", "fhn", "lorenz"],
                        choices=list(SYSTEMS), help="Systems to generate")
    parser.add_argument("--num_samples",   type=int, default=200,
                        help="Independent ODE instances per system — each gets its "
                             "own randomly drawn parameters (S)")
    parser.add_argument("--num_traj",      type=int, default=5,
                        help="Trajectories per instance, same params different ICs (T)")
    parser.add_argument("--num_obs",       type=int, default=32,
                        help="Observation time points per trajectory (N)")
    parser.add_argument("--num_locations", type=int, default=16,
                        help="Random query locations per instance (L)")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--out_dir",       type=str,
                        default=str(Path(__file__).resolve().parent.parent.parent / "data" / "ode"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for system in args.systems:
        cfg = SYSTEMS[system]
        print(
            f"Generating {system} ({cfg['dim']}D, randomised params) — "
            f"{args.num_samples} samples × {args.num_traj} traj × {args.num_obs} obs …",
            end=" ", flush=True,
        )
        data = generate_dataset(
            system=system,
            num_samples=args.num_samples,
            num_trajectories=args.num_traj,
            num_obs=args.num_obs,
            num_locations=args.num_locations,
            seed=args.seed,
        )
        out_path = out_dir / f"{system}.pt"
        torch.save(data, out_path)
        print(f"saved → {out_path}")
        for k, v in data.items():
            print(f"    {k:<28} {tuple(v.shape)}")


if __name__ == "__main__":
    main()
