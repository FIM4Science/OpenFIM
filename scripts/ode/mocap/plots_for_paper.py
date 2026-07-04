"""
scripts/ode/mocap/plots_for_paper.py
=====================================
Paper-quality MoCap trajectory comparison plots: PCA 2D projections.

Three figures (one per subject: 09-short, 35-short, 39-short), each 1×3
with PCA projections PC0vPC1, PC0vPC2, PC1vPC2.  Style matches
``scripts/ode/odebench/plots_for_paper.py::plot_comparison_1x3``:

  - Gray streamplot vector field (model-inferred, conditioned on context)
  - Ground truth data shown as black "x" crosses
  - Model-predicted trajectories in green with start marker

Usage
-----
    python scripts/ode/mocap/plots_for_paper.py [--mocap-dir data/mocap] [--ckpt-dir <dir>]

Or import ``plot_mocap_subject`` / ``plot_all_mocap_subjects`` from a notebook.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from fim.models.ode import load_fim_ode_hf, load_fim_ode_local


matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# ── project root on sys.path ──────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent  # scripts/ode/mocap/
_ROOT = _HERE.parent.parent.parent  # repo root

for _p in [str(_ROOT / "src"), str(_ROOT / "scripts" / "ode")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


_ft_path = _ROOT / "scripts" / "ode" / "finetune.py"
_ft_spec = importlib.util.spec_from_file_location("fim_ode_ft", _ft_path)
_ft = importlib.util.module_from_spec(_ft_spec)
_ft_spec.loader.exec_module(_ft)

integrate_from_context = _ft.integrate_from_context
prepare_context_tensors = _ft.prepare_context_tensors
load_mocap = _ft.load_mocap

# ── style constants (matches plot_comparison_1x3) ────────────────────────────
FONT_SIZE = 45
COLOR_PRED = "#2ca02c"  # green  — model prediction
GRID_RESOLUTION = 25

_PROJ_PAIRS = [(0, 1), (0, 2), (1, 2)]
_PROJ_TITLES = ["PC 0 vs PC 1", "PC 0 vs PC 2", "PC 1 vs PC 2"]
_PROJ_XLABELS = ["PC 0", "PC 0", "PC 1"]
_PROJ_YLABELS = ["PC 1", "PC 2", "PC 2"]


# =============================================================================
# Internal helpers
# =============================================================================


def _axis_limits(tst_ys, preds, n_test, di, dj):
    all_i = np.concatenate([tst_ys[:n_test, :, di].ravel()] + [p[:, di] for p in preds])
    all_j = np.concatenate([tst_ys[:n_test, :, dj].ravel()] + [p[:, dj] for p in preds])
    pad_i = (all_i.max() - all_i.min()) * 0.15 + 1e-6
    pad_j = (all_j.max() - all_j.min()) * 0.15 + 1e-6
    return (all_i.min() - pad_i, all_i.max() + pad_i), (all_j.min() - pad_j, all_j.max() + pad_j)


def _vector_field_2d(model, traj_t, time_t, mask_t, di, dj, xlim, ylim, trn_ys, device, resolution=GRID_RESOLUTION):
    """Model vector field on a 2D grid; the unused 3rd dim is fixed at its mean."""
    dk = 3 - di - dj
    dk_val = float(np.mean(trn_ys[:, :, dk]))

    xi_g = np.linspace(xlim[0], xlim[1], resolution)
    xj_g = np.linspace(ylim[0], ylim[1], resolution)
    Xi, Xj = np.meshgrid(xi_g, xj_g)

    pts = np.zeros((Xi.size, 3), dtype=np.float32)
    pts[:, di] = Xi.ravel()
    pts[:, dj] = Xj.ravel()
    pts[:, dk] = dk_val

    loc_t = torch.tensor(pts, device=device).unsqueeze(0)  # (1, N, 3)
    with torch.no_grad():
        out = model.model_forward(traj_t, time_t, loc_t, mask_t)
        vf = model.get_prediction_for_eval(out).squeeze(0).cpu().numpy()  # (N, 3)

    U = vf[:, di].reshape(Xi.shape)
    V = vf[:, dj].reshape(Xj.shape)
    return xi_g, xj_g, U, V


# =============================================================================
# Public API
# =============================================================================


def plot_mocap_subject(
    data: dict,
    model,
    device: str,
    subject_label: str,
    n_test: int = 2,
    figsize: Tuple[float, float] = (30, 10),
    grid_resolution: int = GRID_RESOLUTION,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """1×3 figure showing PCA 2D projections for one MoCap subject.

    Parameters
    ----------
    data          : dict returned by ``load_mocap``
    model         : FIMODE (eval mode, on ``device``)
    device        : "cpu" or "cuda"
    subject_label : figure suptitle, e.g. "Subject 09 – short"
    n_test        : number of test trajectories to draw
    figsize       : figure size — default matches plot_comparison_1x3
    """
    trn_ys = data["trn_ys"]  # (n_trn, T_trn, 3)
    trn_ts = data["trn_ts"]
    tst_ys = data["tst_ys"]  # (n_tst, T_tst, 3)
    tst_ts = data["tst_ts"]
    n_test = min(n_test, tst_ys.shape[0])

    # Integrate predictions
    preds = []
    for i in range(n_test):
        y0 = tst_ys[i, 0].astype(float)
        pred = integrate_from_context(model, trn_ys, trn_ts, y0, tst_ts, device)
        preds.append(pred)

    # Context tensors (reused for every vector field evaluation)
    traj_t, time_t, mask_t = prepare_context_tensors(trn_ys, trn_ts, device)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    # Extra top margin so suptitle and subplot titles don't overlap;
    # extra left/wspace so y-axis labels don't overlap with adjacent subplots.
    plt.subplots_adjust(left=0.10, right=0.97, top=0.78, bottom=0.12, wspace=0.20)

    for col, (di, dj) in enumerate(_PROJ_PAIRS):
        ax = axes[col]

        xlim, ylim = _axis_limits(tst_ys, preds, n_test, di, dj)

        # ── Gray vector field ───────────────────────────────────────────────
        xi_g, xj_g, U, V = _vector_field_2d(
            model,
            traj_t,
            time_t,
            mask_t,
            di,
            dj,
            xlim,
            ylim,
            trn_ys,
            device,
            grid_resolution,
        )
        ax.streamplot(xi_g, xj_g, U, V, color="#bbbbbb", density=0.9, linewidth=3.0, arrowsize=3.0)

        # ── GT: black "x" crosses ───────────────────────────────────────────
        for k in range(n_test):
            ax.scatter(
                tst_ys[k, :, di],
                tst_ys[k, :, dj],
                s=80,
                marker="x",
                color="black",
                linewidths=2,
                alpha=0.8,
                zorder=1002,
            )

        # ── Predicted trajectories: green lines + start marker ──────────────
        for k in range(n_test):
            pr = preds[k]
            ax.plot(pr[:, di], pr[:, dj], color=COLOR_PRED, linewidth=8, alpha=1.0, zorder=1000, solid_capstyle="round")
            ax.scatter(pr[:, di], pr[:, dj], s=15, color="black", alpha=0.5, zorder=1002, edgecolors="lightgrey", linewidths=1)
            ax.scatter(pr[0, di], pr[0, dj], s=1000, marker="s", facecolor=COLOR_PRED, edgecolors="black", linewidths=3, zorder=1003)
            ax.text(pr[0, di], pr[0, dj], str(k), fontsize=28, ha="center", va="center", color="white", fontweight="bold", zorder=1004)

        ax.set_xlabel(_PROJ_XLABELS[col], fontsize=0.75 * FONT_SIZE)
        ax.set_ylabel(_PROJ_YLABELS[col], fontsize=0.75 * FONT_SIZE, labelpad=15)
        ax.set_title(_PROJ_TITLES[col], fontsize=FONT_SIZE, fontweight="bold", pad=20)
        ax.tick_params(axis="both", which="major", labelsize=0.75 * FONT_SIZE, width=2, length=6)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)

    fig.suptitle(subject_label, fontsize=FONT_SIZE, fontweight="bold", y=0.96)
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  saved → {save_path}")
    return fig


def plot_all_mocap_subjects(
    mocap_dir: Path,
    model,
    device: str,
    n_test: int = 2,
    save_dir: Optional[Path] = None,
) -> list:
    """Return a list of figures for subjects 09, 35, 39 (short variant)."""
    subjects = [("09", "short"), ("35", "short"), ("39", "short")]
    figs = []
    for subj, var in subjects:
        print(f"Plotting subject {subj}-{var} …", flush=True)
        data = load_mocap(Path(mocap_dir), subject=subj, variant=var)
        label = f"Subject {subj} – {var}"
        fig = plot_mocap_subject(data, model, device, label, n_test=n_test)
        figs.append(fig)
        if save_dir is not None:
            out = Path(save_dir) / f"mocap_{subj}_{var}_pca.pdf"
            fig.savefig(out, bbox_inches="tight")
            print(f"  saved → {out}")
    return figs


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MoCap PCA projection plots for paper")
    parser.add_argument("--mocap-dir", default=str(_ROOT / "data" / "mocap"))
    parser.add_argument("--ckpt-dir", default=None, help="Local checkpoint dir (uses HuggingFace pretrained if omitted)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--n-test", type=int, default=2)
    args = parser.parse_args()

    if args.ckpt_dir is not None:
        _model = load_fim_ode_local(Path(args.ckpt_dir), device=args.device)
    else:
        _model = load_fim_ode_hf(device=args.device)
    _model.eval()

    _figs = plot_all_mocap_subjects(
        mocap_dir=Path(args.mocap_dir),
        model=_model,
        device=args.device,
        n_test=args.n_test,
        save_dir=Path(args.save_dir) if args.save_dir else None,
    )
    plt.show()
