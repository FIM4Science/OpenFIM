import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

from utils.data_models import trajectory, trajectory_list_from_h5_files

import torch
from scipy.integrate import solve_ivp

from utils.eval_models import PredictionModel, OdeonEval

from pathlib import Path

from utils.helpers import predict_vector_field, load_odeon_model_from_checkpoint


def plot_3d_paths(*, pred_list: list[trajectory] = [], ctx_list: list[trajectory] = [], obs_list: list[trajectory] = [],
                  labels=("Prediction", "Context", "Observed"),
                  view=(25, -65), ax=None):
    """
    pred_list, obs_list: lists of arrays, each of shape (T, 3)
    
    Args:
        pred_list: List of predicted trajectories
        ctx_list: List of context trajectories
        obs_list: List of observed trajectories
        labels: Tuple of labels for legend
        view: Tuple of (elevation, azimuth) for 3D view
        ax: Optional matplotlib Axes3D object. If provided, plots will be drawn on it.
            If None, a new figure will be created and displayed.
    
    Returns:
        The axes object used for plotting
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        show_plot = True
    else:
        fig = None
        show_plot = False

    # Predicted trajectories
    for traj in pred_list:
        x, y, z = traj.xs[:, 0], traj.xs[:, 1], traj.xs[:, 2]
        ax.plot(x, y, z, color="red", lw=1, alpha=0.7)
        ax.scatter(x, y, z, color="red", s=2, alpha=0.5)

    # Start and end points of predicted trajectories
    for traj in pred_list:
        # Start point (first point)
        start_x, start_y, start_z = traj.xs[0, 0], traj.xs[0, 1], traj.xs[0, 2]
        ax.scatter([start_x], [start_y], [start_z], color="white", s=20, marker="o", edgecolors="black", linewidths=1, label="Start" if traj == pred_list[0] else "")
        # End point (last point)
        end_x, end_y, end_z = traj.xs[-1, 0], traj.xs[-1, 1], traj.xs[-1, 2]
        ax.scatter([end_x], [end_y], [end_z], color="black", s=20, marker="s", edgecolors="black", linewidths=1, label="End" if traj == pred_list[0] else "")

    # Context trajectories
    for traj in ctx_list:
        x, y, z = traj.xs[:, 0], traj.xs[:, 1], traj.xs[:, 2]
        ax.plot(x, y, z, color="blue", lw=1, alpha=0.6)
        ax.scatter(x, y, z, color="blue", s=2, marker="+", alpha=0.6)

    # Observed trajectories
    for traj in obs_list:
        x, y, z = traj.xs[:, 0], traj.xs[:, 1], traj.xs[:, 2]
        ax.plot(x, y, z, color="green", lw=1, alpha=0.6)
        ax.scatter(x, y, z, color="green", s=2, marker="+", alpha=0.6)

    ax.set_xlabel("Comp 1")
    ax.set_ylabel("Comp 2")
    ax.set_zlabel("Comp 3")

    # Dummy artists for clean legend
    ax.scatter([], [], [], color="red", s=20, label=labels[0])
    ax.scatter([], [], [], color="blue", s=40, marker="+", label=labels[1])
    ax.scatter([], [], [], color="green", s=40, marker="+", label=labels[2])
    ax.legend(loc="upper right")

    if view is not None:
        elev, azim = view
        ax.view_init(elev=elev, azim=azim)

    if show_plot:
        plt.tight_layout()
        #plt.show()
        plt.savefig("experiments/plot_3d_paths.png")
    
    return ax

def plot_2d_paths(*, pred_list: list[trajectory] = [], ctx_list: list[trajectory] = [], obs_list: list[trajectory] = [],
                  labels=("Prediction", "Context", "Observed")):
    """
    pred_list, obs_list: lists of arrays, each of shape (T, 2)
    """
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    
    # Predicted trajectories
    for traj in pred_list:
        x, y = traj.xs[:, 0], traj.xs[:, 1]
        ax.plot(x, y, color="red", lw=1, alpha=0.7)
        ax.scatter(x, y, color="red", s=2, alpha=0.5)
    
    # Start and end points of predicted trajectories
    for traj in pred_list:
        # Start point (first point)
        start_x, start_y = traj.xs[0, 0], traj.xs[0, 1]
        ax.scatter([start_x], [start_y], color="white", s=20, marker="o", edgecolors="black", linewidths=1, label="Start" if traj == pred_list[0] else "")
        # End point (last point)
        end_x, end_y = traj.xs[-1, 0], traj.xs[-1, 1]
        ax.scatter([end_x], [end_y], color="black", s=20, marker="s", edgecolors="black", linewidths=1, label="End" if traj == pred_list[0] else "")
    
    # Context trajectories
    for traj in ctx_list:
        x, y = traj.xs[:, 0], traj.xs[:, 1]
        ax.plot(x, y, color="blue", lw=1, alpha=0.6)
        ax.scatter(x, y, color="blue", s=2, marker="+", alpha=0.6)
    
    # Observed trajectories
    for traj in obs_list:
        x, y = traj.xs[:, 0], traj.xs[:, 1]
        ax.plot(x, y, color="green", lw=1, alpha=0.6)
        ax.scatter(x, y, color="green", s=2, marker="+", alpha=0.6)
    
    ax.set_xlabel("Comp 1")
    ax.set_ylabel("Comp 2")
    
    # Dummy artists for clean legend
    ax.scatter([], [], color="red", s=20, label=labels[0])
    ax.scatter([], [], color="blue", s=40, marker="+", label=labels[1])
    ax.scatter([], [], color="green", s=40, marker="+", label=labels[2])
    ax.legend(loc="upper right")
    
    plt.tight_layout()
    plt.show()


def plot_3d_vector_field(*, model: PredictionModel, context_trajs: list[trajectory] or None = None):
    assert context_trajs

    # grid
    x, y, z = np.meshgrid(
        np.linspace(-1, 1, 5),
        np.linspace(-1, 1, 5),
        np.linspace(-1, 1, 5)
    )           # The type of this meshgrid is tuple of ndarrays!

    #print(x.shape)     # 8,8,8
    #print(y.shape)     # 8,8,8
    #print(z.shape)     # 8,8,8
    
    print(
        type(
        np.meshgrid(
            np.linspace(-1, 1, 8),
            np.linspace(-1, 1, 8),
            np.linspace(-1, 1, 8)
        )
        )
    )


    # vector field (u, v, w)
    u = np.zeros_like(x)
    v = np.zeros_like(y)
    w = np.zeros_like(z)

    grid_points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)  # [N, 3]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Get vector field predictions at grid points using the helper function
    vf = predict_vector_field(model, context_trajs, grid_points)  # [N, 3]
    
    # Reshape for plotting
    u = vf[:, 0].reshape(x.shape)
    v = vf[:, 1].reshape(y.shape)
    w = vf[:, 2].reshape(z.shape)

    # Compute magnitude for coloring
    magnitude = np.sqrt(u**2 + v**2 + w**2)


    # magnitude is (nx, ny, nz)
    m = magnitude.ravel()

    cmap = cm.viridis  # or whatever
    norm = Normalize(vmin=m.min(), vmax=m.max())

    quiver = ax.quiver(
        x, y, z, u, v, w,
        length=0.15,
        normalize=True,
        arrow_length_ratio=0.3,
        cmap=cmap,
        norm=norm,
    )

    # Attach per-arrow scalars so the quiver uses the colormap
    quiver.set_array(m)

    # Colorbar consistent with the quiver
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label("Vector Field Magnitude")



    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()


if __name__ == "__main__":
    model = load_odeon_model_from_checkpoint(checkpoints_dir=Path("models/longtrainedmocap09long/checkpoints"))

    data_path = Path("experiments/mocap/") / "mocap39long" / "data/test/3d+2d/3d"
    # lets just do mocap subject 09 long for now.
    context_trajs: list[trajectory] = trajectory_list_from_h5_files(path_to_xs=data_path/"obs_values.h5", path_to_ts=data_path/"obs_times.h5")

    plot_3d_vector_field(model=model, context_trajs=context_trajs)
