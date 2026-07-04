"""Plotting utilities for mocap tasks: base vs finetuned model comparison.

Supports 3D trajectory plots and 2D components-vs-time plots, with task selection.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Literal

import matplotlib
import numpy as np
from data_gen_mocap import MocapDataset
from matplotlib import pyplot as plt
from utils.data_models import trajectory, trajectory_list_from_h5_files

# Pickles were saved with class path "data_gen_mocap.MocapDataset"; register alias for unpickling.
from utils.helpers import (
    load_odeon_model_from_checkpoint,
    predict_and_integrate_ode,
)


sys.modules["data_gen_mocap"] = sys.modules["experiments.data_gen_mocap"]
# Plot styling (global vars)
REF_SCALE = 2.0
FONT_SIZE_TITLE = REF_SCALE * 14
FONT_SIZE_LABEL = REF_SCALE * 12
FONT_SIZE_LEGEND = REF_SCALE * 10
TICK_LABELSIZE = REF_SCALE * 10
AXES_TITLEPAD = REF_SCALE * 10
LINE_WIDTH = REF_SCALE * 2.0

COLOR_TRUE = "green"
COLOR_BASE = "orange"
COLOR_FINETUNED = "red"
COLOR_CONTEXT = "blue"

title = {
    "mocap09short": "MoCap: Subject 09 (short context trajectories)",
    "mocap09long": "MoCap: Subject 09 (long context trajectories)",
    "mocap35short": "MoCap: Subject 35 (short context trajectories)",
    "mocap35long": "MoCap: Subject 35 (long context trajectories)",
    "mocap39short": "MoCap: Subject 39 (short context trajectories)",
    "mocap39long": "MoCap: Subject 39 (long context trajectories)",
}

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams.update(
    {
        "axes.titlesize": FONT_SIZE_TITLE,
        "axes.labelsize": FONT_SIZE_LABEL,
        "xtick.labelsize": TICK_LABELSIZE,
        "ytick.labelsize": TICK_LABELSIZE,
        "legend.fontsize": FONT_SIZE_LEGEND,
        "axes.titlepad": AXES_TITLEPAD,
    }
)

# Default task name -> data path (train/5d)
MOCAP_TASK_DATA_PATHS = {
    "mocap09short": Path("experiments/mocap/mocap09short/data/train/5d"),
    "mocap09long": Path("experiments/mocap/mocap09long/data/train/5d"),
    "mocap35short": Path("experiments/mocap/mocap35short/data/train/5d"),
    "mocap35long": Path("experiments/mocap/mocap35long/data/train/5d"),
    "mocap39short": Path("experiments/mocap/mocap39short/data/train/5d"),
    "mocap39long": Path("experiments/mocap/mocap39long/data/train/5d"),
}


def _split_5d_to_left_right(traj_list_5d: list[trajectory]) -> tuple[list[trajectory], list[trajectory]]:
    """Split 5D trajectories into left (comps 0,1,2) and right (comps 1,3,4)."""
    traj_list_left: list[trajectory] = []
    traj_list_right: list[trajectory] = []
    for traj_5d in traj_list_5d:
        ts = traj_5d.ts
        xs_5d = traj_5d.xs
        xs_left = xs_5d[:, [0, 1, 2]]
        xs_right = xs_5d[:, [1, 3, 4]]
        traj_list_left.append(trajectory(xs=xs_left, ts=ts))
        traj_list_right.append(trajectory(xs=xs_right, ts=ts))
    return (traj_list_left, traj_list_right)


def load_mocap_task_data(
    task_name: str,
    data_path: Path | None = None,
) -> tuple[list[trajectory], list[trajectory], np.ndarray]:
    """Load context (left 3D), test trajectories (left 3D), and prediction time grid.

    Args:
        task_name: Key in MOCAP_TASK_DATA_PATHS, or used only if data_path is set.
        data_path: Override path to task data dir (must contain obs_values.h5, obs_times.h5, mocap_dataset.pkl).

    Returns:
        ctx_trajs_left: Context trajectories (3D).
        test_trajs_left: Test trajectories used as ground truth and for ICs (3D).
        pred_ts: Time grid for integration (same length as test trajectories).
    """
    if data_path is None:
        data_path = MOCAP_TASK_DATA_PATHS.get(task_name)
        if data_path is None:
            raise ValueError(f"Unknown task {task_name!r}. Choose from {list(MOCAP_TASK_DATA_PATHS.keys())} or pass data_path.")
    data_path = Path(data_path)

    ctx_trajs_5d: list[trajectory] = trajectory_list_from_h5_files(
        path_to_xs=data_path / "obs_values.h5",
        path_to_ts=data_path / "obs_times.h5",
    )
    ctx_trajs_left, _ = _split_5d_to_left_right(ctx_trajs_5d)

    with open(data_path / "mocap_dataset.pkl", "rb") as f:
        dataset: MocapDataset = pickle.load(f)

    test_trajs_5d = [trajectory(dataset.tst.ys[i, :, :], dataset.tst.ts) for i in range(dataset.tst.ys.shape[0])]
    test_trajs_left, _ = _split_5d_to_left_right(test_trajs_5d)
    pred_ts = test_trajs_5d[0].ts

    return ctx_trajs_left, test_trajs_left, pred_ts


def plot_mocap_comparison(
    task_name: str,
    base_checkpoint_dir: Path | str,
    finetuned_checkpoint_dir: Path | str,
    plot_mode: Literal["3d", "2d"] = "3d",
    test_idx: int = 0,
    data_path: Path | None = None,
    save_path: Path | str | None = None,
    show_context: bool = False,
) -> matplotlib.figure.Figure:
    """Plot base vs finetuned model predictions for a mocap task.

    Args:
        task_name: Task identifier (e.g. 'mocap09short', 'mocap35long').
        base_checkpoint_dir: Path to base model checkpoints.
        finetuned_checkpoint_dir: Path to finetuned model checkpoints.
        plot_mode: '3d' for 3D trajectory, '2d' for three panels (comp0, comp1, comp2 vs time).
        test_idx: Index of test trajectory to plot (0-based).
        data_path: Override data directory for the task.
        save_path: If set, save figure to this path.
        show_context: Whether to draw context trajectories (3D) or context points (2D).

    Returns:
        The matplotlib Figure.
    """
    base_checkpoint_dir = Path(base_checkpoint_dir)
    finetuned_checkpoint_dir = Path(finetuned_checkpoint_dir)

    ctx_trajs_left, test_trajs_left, pred_ts = load_mocap_task_data(task_name, data_path=data_path)
    if test_idx >= len(test_trajs_left):
        raise ValueError(f"test_idx {test_idx} out of range (max {len(test_trajs_left) - 1}).")

    true_traj = test_trajs_left[test_idx]
    y0 = true_traj.xs[0, :]

    base_model = load_odeon_model_from_checkpoint(base_checkpoint_dir, epoch=None)
    finetuned_model = load_odeon_model_from_checkpoint(finetuned_checkpoint_dir, epoch=None)

    pred_base = predict_and_integrate_ode(base_model, ctx_trajs_left, y0, pred_ts)
    pred_finetuned = predict_and_integrate_ode(finetuned_model, ctx_trajs_left, y0, pred_ts)

    if plot_mode == "3d":
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        if show_context:
            for k, traj in enumerate(ctx_trajs_left):
                x, y, z = traj.xs[:, 0], traj.xs[:, 1], traj.xs[:, 2]
                label = "Context" if k == 0 else None  # label once for legend
                ax.plot(x, y, z, color=COLOR_CONTEXT, lw=1, alpha=0.6, zorder=1, label=label)
                ax.scatter(x, y, z, color=COLOR_CONTEXT, s=2, alpha=0.6, zorder=1)

        # True (ground truth)
        x, y, z = true_traj.xs[:, 0], true_traj.xs[:, 1], true_traj.xs[:, 2]
        ax.plot(x, y, z, color=COLOR_TRUE, lw=LINE_WIDTH, alpha=0.9, label="Ground truth", zorder=2)
        ax.scatter(x[0:1], y[0:1], z[0:1], color="white", s=40, marker="o", edgecolors="black", zorder=5)

        # Base prediction
        xb, yb, zb = pred_base.xs[:, 0], pred_base.xs[:, 1], pred_base.xs[:, 2]
        ax.plot(xb, yb, zb, color=COLOR_BASE, lw=LINE_WIDTH, alpha=0.8, label="Base", zorder=3)
        # Finetuned prediction (drawn last with highest zorder so it appears in foreground)
        xf, yf, zf = pred_finetuned.xs[:, 0], pred_finetuned.xs[:, 1], pred_finetuned.xs[:, 2]
        ax.plot(xf, yf, zf, color=COLOR_FINETUNED, lw=LINE_WIDTH, alpha=0.8, label="Finetuned", zorder=4)

        ax.set_xlabel("Comp 1")
        ax.set_ylabel("Comp 2")
        ax.set_zlabel("Comp 3")
        ax.legend(loc="lower left")
        # ax.set_title(f"{title[task_name]}")
        # fig.suptitle(f"{title[task_name]}", fontsize=FONT_SIZE_TITLE, fontweight="bold", y=1.02)
        fig.tight_layout()

    else:  # "2d" — three components vs time
        fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
        ts = true_traj.ts

        for comp in range(3):
            ax = axs[comp]
            ax.plot(ts, true_traj.xs[:, comp], "-", color=COLOR_TRUE, lw=LINE_WIDTH, alpha=0.9, label="Ground truth")
            ax.plot(ts, pred_base.xs[:, comp], "-", color=COLOR_BASE, lw=LINE_WIDTH, alpha=0.8, label="Base model")
            ax.plot(ts, pred_finetuned.xs[:, comp], "-", color=COLOR_FINETUNED, lw=LINE_WIDTH, alpha=0.8, label="Finetuned model")
            """
            if show_context and ctx_trajs_left:
                for traj in ctx_trajs_left:
                    ax.scatter(traj.ts, traj.xs[:, comp], color=COLOR_CONTEXT, s=8, alpha=0.5, zorder=3)
            """
            ax.set_ylabel(f"Comp {comp}")
            ax.legend(loc="lower left", fontsize=FONT_SIZE_LEGEND)
            ax.grid(True, alpha=0.3)

        axs[2].set_xlabel("Time")
        # fig.suptitle(f"{title[task_name]}", fontsize=FONT_SIZE_TITLE, fontweight="bold", y=1.02)
        fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    # Example: select task and paths, then plot 3D and 2D
    TASK = "mocap35short"
    BASE_CKPT = Path("models/base_model/checkpoints")
    FINETUNED_CKPT = Path("models/mocap35short/mocap35short_01-29-0614/checkpoints")

    fig_3d = plot_mocap_comparison(
        task_name=TASK,
        base_checkpoint_dir=BASE_CKPT,
        finetuned_checkpoint_dir=FINETUNED_CKPT,
        plot_mode="3d",
        test_idx=0,
        save_path=Path("experiments/mocap") / f"{TASK}_comparison_3d.pdf",
    )
    plt.close(fig_3d)

    fig_2d = plot_mocap_comparison(
        task_name=TASK,
        base_checkpoint_dir=BASE_CKPT,
        finetuned_checkpoint_dir=FINETUNED_CKPT,
        plot_mode="2d",
        test_idx=0,
        save_path=Path("experiments/mocap") / f"{TASK}_comparison_2d.pdf",
    )
    plt.close(fig_2d)
