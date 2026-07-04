from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from ODEs import VDP_ode
from scipy.integrate import solve_ivp
from utils.data_models import trajectory
from utils.h5 import get_ndarray_from_h5
from utils.helpers import (
    load_odeon_model_from_checkpoint,
    predict_and_integrate_ode,
    predict_vector_field,
)


matplotlib.rcParams["pdf.fonttype"] = 42


# Plot styling (global vars)
REF_SCALE = 2.0
FONT_SIZE_TITLE = REF_SCALE * 14
FONT_SIZE_LABEL = REF_SCALE * 12
FONT_SIZE_LEGEND = REF_SCALE * 10
TICK_LABELSIZE = REF_SCALE * 10
AXES_TITLEPAD = REF_SCALE * 10

LINE_WIDTH = REF_SCALE * 2.0
STREAMPLOT_LINEWIDTH = REF_SCALE * 1.5

CROSS_SIZE = REF_SCALE * 40
CROSS_LINEWIDTHS = REF_SCALE * 2

COLOR_CROSSES = "black"
COLOR_TRUE = "green"
COLOR_BASE = "red"
COLOR_FINETUNED = "red"

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

if __name__ == "__main__":
    base_model = load_odeon_model_from_checkpoint(Path("models/base_model/checkpoints"), epoch=None)
    finetuned_model = load_odeon_model_from_checkpoint(Path("models/vdp1/vdp1_01-29-0707/checkpoints"), epoch=None)

    fig, ax = plt.subplots(1, 3, figsize=(30, 10))
    data_path = Path("experiments/vdp1/data_gpode")

    # Load noisy observations
    context_xs = get_ndarray_from_h5(data_path / "obs_values.h5")  # (1, 1, T, 2)
    context_ts = get_ndarray_from_h5(data_path / "obs_times.h5")  # (1, 1, T, 1)
    context_traj = trajectory(context_xs, context_ts)
    obs_points = context_traj.xs.squeeze()  # (T, 2)

    # Determine plot range from observations
    x_min, x_max = obs_points[:, 0].min() - 0.5, obs_points[:, 0].max() + 0.5
    y_min, y_max = obs_points[:, 1].min() - 0.5, obs_points[:, 1].max() + 0.5

    # Create grid for vector field
    grid_resolution = 30
    x = np.linspace(x_min, x_max, grid_resolution)
    y = np.linspace(y_min, y_max, grid_resolution)
    X, Y = np.meshgrid(x, y)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)  # (N, 2)

    y0 = np.array([-1.5, 2.5])
    t_eval = np.linspace(0, 14, 1000)

    # --- Ground truth (ax[0]) ---
    vf_true = np.array([VDP_ode(0, point) for point in grid_points])
    U_true = vf_true[:, 0].reshape(X.shape)
    V_true = vf_true[:, 1].reshape(Y.shape)
    ax[0].streamplot(X, Y, U_true, V_true, color="#dddddd", density=0.6, linewidth=STREAMPLOT_LINEWIDTH, arrowsize=1.5)
    sol_true = solve_ivp(VDP_ode, (0.0, 14.0), y0, t_eval=t_eval, method="RK45")
    ax[0].plot(sol_true.y.T[:, 0], sol_true.y.T[:, 1], "-", color=COLOR_TRUE, label="True integral curve", linewidth=LINE_WIDTH, alpha=0.8)
    ax[0].scatter(
        obs_points[:, 0],
        obs_points[:, 1],
        color=COLOR_CROSSES,
        s=CROSS_SIZE,
        marker="x",
        label="Observations",
        zorder=5,
        edgecolors="black",
        linewidths=CROSS_LINEWIDTHS,
    )
    ax[0].set_title("Ground truth", fontsize=FONT_SIZE_TITLE, fontweight="bold")
    ax[0].set_aspect("equal", adjustable="box")

    # --- Base model (ax[1]) ---
    vf_base = predict_vector_field(base_model, [context_traj], grid_points)
    U_base = vf_base[:, 0].reshape(X.shape)
    V_base = vf_base[:, 1].reshape(Y.shape)
    ax[1].streamplot(X, Y, U_base, V_base, color="#dddddd", density=0.6, linewidth=STREAMPLOT_LINEWIDTH, arrowsize=1.5)
    pred_traj_base = predict_and_integrate_ode(base_model, [context_traj], y0, t_eval)
    ax[1].plot(
        pred_traj_base.xs[:, 0],
        pred_traj_base.xs[:, 1],
        "-",
        color=COLOR_BASE,
        label="Predicted integral curve",
        linewidth=LINE_WIDTH,
        alpha=0.8,
    )
    ax[1].scatter(
        obs_points[:, 0],
        obs_points[:, 1],
        color=COLOR_CROSSES,
        s=CROSS_SIZE,
        marker="x",
        label="Observations",
        zorder=5,
        edgecolors="black",
        linewidths=CROSS_LINEWIDTHS,
    )
    ax[1].set_title("Base model", fontsize=FONT_SIZE_TITLE, fontweight="bold")
    ax[1].set_aspect("equal", adjustable="box")
    ax[1].plot(sol_true.y.T[:, 0], sol_true.y.T[:, 1], "-", color=COLOR_TRUE, label="True integral curve", linewidth=LINE_WIDTH, alpha=0.8)

    # --- Finetuned model (ax[2]) ---
    vf_ft = predict_vector_field(finetuned_model, [context_traj], grid_points)
    U_ft = vf_ft[:, 0].reshape(X.shape)
    V_ft = vf_ft[:, 1].reshape(Y.shape)
    ax[2].streamplot(X, Y, U_ft, V_ft, color="#dddddd", density=0.6, linewidth=STREAMPLOT_LINEWIDTH, arrowsize=1.5)
    pred_traj_ft = predict_and_integrate_ode(finetuned_model, [context_traj], y0, t_eval)
    ax[2].plot(
        pred_traj_ft.xs[:, 0],
        pred_traj_ft.xs[:, 1],
        "-",
        color=COLOR_FINETUNED,
        label="Predicted integral curve",
        linewidth=LINE_WIDTH,
        alpha=0.8,
    )
    ax[2].scatter(
        obs_points[:, 0],
        obs_points[:, 1],
        color=COLOR_CROSSES,
        s=CROSS_SIZE,
        marker="x",
        label="Observations",
        zorder=5,
        edgecolors="black",
        linewidths=CROSS_LINEWIDTHS,
    )
    ax[2].set_title("Finetuned model", fontsize=FONT_SIZE_TITLE, fontweight="bold")
    ax[2].set_aspect("equal", adjustable="box")
    ax[2].plot(sol_true.y.T[:, 0], sol_true.y.T[:, 1], "-", color=COLOR_TRUE, label="True integral curve", linewidth=LINE_WIDTH, alpha=0.8)

    for a in ax:
        a.set_xlabel("$x_1$")
        a.set_ylabel("$x_2$")
        a.grid(False)

    fig.tight_layout()
    plt.savefig("experiments/vdp1/vdp1_comparison.pdf", bbox_inches="tight")
    plt.close()
