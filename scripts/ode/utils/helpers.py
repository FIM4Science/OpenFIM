"""Simple script to load Odeon model checkpoint and perform inference.

This script demonstrates how to:
1. Load a trained Odeon model from a checkpoint
2. Provide context trajectory and time grid
3. Predict the vector field at arbitrary locations
4. Integrate the learned ODE to get trajectory predictions
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import solve_ivp

from utils.eval_models import PredictionModel, OdeonEval
from utils.data_models import trajectory

from typing import Literal


def load_odeon_model_from_checkpoint(checkpoints_dir: Path | str, epoch: int = None):
    """Load an Odeon model from checkpoint directory.
    
    Args:
        checkpoint_dir: Path to the checkpoints directory (e.g., 
            "models/base_model/checkpoints")
        epoch: Specific epoch to load, or None for best-model
    
    Returns:
        OdeonEval: Loaded model ready for inference
    """
    if type(checkpoints_dir) == str:
        checkpoints_dir = Path(checkpoints_dir)
    model = OdeonEval(checkpoints_dir, epoch=epoch)
    return model


def predict_vector_field(
    model: PredictionModel, 
    context_trajs: list[trajectory], 
    locations: np.ndarray
) -> np.ndarray:
    """Predict vector field at given locations.
    
    Args:
        model: Loaded OdeonEval model
        context_trajs: List of context trajectory objects
        locations: Locations at which to evaluate vector field, shape [N, D]
    
    Returns:
        np.ndarray: Predicted vector field at locations, shape [N, D]
    """
    # Convert single trajectory to list if needed (backward compatibility)
    if isinstance(context_trajs, trajectory):
        context_trajs = [context_trajs]
    
    # Stack all context trajectories along the trajectory dimension (P)
    # Work with numpy arrays, convert to torch only when needed for model
    traj_list = []
    times_list = []
    
    for traj_obj in context_trajs:
        # Get numpy arrays and add batch/trajectory dimensions
        traj_np = traj_obj.xs[np.newaxis, :, :]  # [1, T, D]
        times_np = traj_obj.ts[np.newaxis, :, np.newaxis]  # [1, T, 1]
        traj_list.append(traj_np)
        times_list.append(times_np)
    
    # Stack along trajectory dimension using numpy, then convert to torch
    traj_np = np.stack(traj_list, axis=1)  # [1, len(context_trajs), T, D]
    times_np = np.stack(times_list, axis=1)  # [1, len(context_trajs), T, 1]
    
    # Convert to torch tensors for model operations and fit model
    traj = torch.from_numpy(traj_np).float().to(model.device)
    times = torch.from_numpy(times_np).float().to(model.device)
    model.fit(traj, times)
    
    # Prepare locations for model (add batch dimension if needed)
    if locations.ndim == 2:
        locations_batch = locations[np.newaxis, :, :]  # [1, N, D]
    else:
        locations_batch = locations
    
    # Convert to torch tensor and evaluate
    locations_tensor = torch.from_numpy(locations_batch).float().to(model.device)
    predictions = model.system(locations_tensor)  # [1, N, D]
    
    return predictions.squeeze(0).cpu().numpy()  # [N, D]


def predict_and_integrate_ode(
    model: PredictionModel, 
    context_trajs: list[trajectory], 
    y0: np.ndarray, 
    t_eval: np.ndarray
) -> trajectory:
    """Integrate the learned ODE from initial condition.
    
    Args:
        model: Loaded OdeonEval model
        context_trajs: List of context trajectory objects
        y0: Initial condition at time t=0, shape [D]
        t_eval: Time points at which to evaluate solution, shape [L]
    
    Returns:
        trajectory: Predicted trajectory object
    """
    # Convert single trajectory to list if needed (backward compatibility)
    if isinstance(context_trajs, trajectory):
        context_trajs = [context_trajs]
    
    # Ensure y0 is numpy array
    y0_np = np.asarray(y0)
    
    # Stack all context trajectories along the trajectory dimension (P)
    # Work with numpy arrays, convert to torch only when needed for model
    traj_list = []
    times_list = []
    
    for traj_obj in context_trajs:
        # Get numpy arrays and add batch/trajectory dimensions
        traj_np = traj_obj.xs[np.newaxis, :, :]  # [1, T, D]
        times_np = traj_obj.ts[np.newaxis, :, np.newaxis]  # [1, T, 1]
        traj_list.append(traj_np)
        times_list.append(times_np)
    
    # Stack along trajectory dimension using numpy, then convert to torch
    traj_np = np.stack(traj_list, axis=1)  # [1, len(context_trajs), T, D]
    times_np = np.stack(times_list, axis=1)  # [1, len(context_trajs), T, 1]
    
    # Convert to torch tensors for model operations and fit model
    traj = torch.from_numpy(traj_np).float().to(model.device)
    times = torch.from_numpy(times_np).float().to(model.device)
    model.fit(traj, times)
    
    # Define ODE function for scipy
    def ode(t, y):
        # y is numpy array from scipy, convert to torch tensor for model
        y_batch = y[np.newaxis, np.newaxis, :]  # [1, 1, D] - single location
        y_tensor = torch.from_numpy(y_batch).float().to(model.device)
        out = model.system(y_tensor)  # [1, 1, D]
        return out.squeeze(0).squeeze(0).cpu().numpy()  # [D]
    
    # Integrate from t=0 to t_eval[-1], with y0 at t=0
    # This allows t_eval to start at any time >= 0
    time_span = (0.0, t_eval[-1])
    sol = solve_ivp(ode, time_span, y0_np, t_eval=t_eval, method='RK45')
    
    # Return as trajectory object
    return trajectory(xs=sol.y.T, ts=t_eval)  # sol.y.T is [L, D]


def plot_vector_field_and_trajectories(
    model: PredictionModel,
    context_trajs: list[trajectory] = None,
    context_trajs_sparse: list[trajectory] = None,
    predicted_trajs: list[trajectory] = None,
    true_trajs: list[trajectory] = None,
    x_range: tuple = None,
    y_range: tuple = None,
    grid_resolution: int = 20,
    save_path: Path = None,
    show_plot: bool = True,
    ax: matplotlib.axes.Axes = None,
    plot_type: Literal["1D", "2D", "2D_streamplot"] = "2D",
    show_legend: bool = True,
    draw_start_and_end_points: bool = True,
):
    """Plot predicted vector field and context trajectories.
    
    Args:
        model: Loaded OdeonEval model
        context_trajs: List of context trajectory objects (must be 2D for plotting)
        predicted_trajs: Optional list of predicted trajectory objects to overlay
        x_range: (x_min, x_max) for plotting grid. If None, inferred from trajectories
        y_range: (y_min, y_max) for plotting grid. If None, inferred from trajectories
        grid_resolution: Number of grid points per dimension
        save_path: Optional path to save the figure
        show_plot: Whether to display the plot
        plot_type: "1D" = time vs coordinates (no vector field); "2D" = phase plane + vector field; "2D_streamplot" = phase plane + vector field with streamplot
        ax: Optional matplotlib Axes object. If provided, plots will be drawn on it.
            If None, a new figure will be created and displayed.
        show_legend: Whether to show the legend
        draw_start_and_end_points: Whether to draw start and end points of trajectories
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
        show_plot = False
    else:
        fig = None
        show_plot = False

    if plot_type == "1D":
        # Time on x-axis, x1 and x2 coordinates on y-axis; no vector field.
        def _plot_traj_time_series(ax, traj_list, colors, name_prefix: str) -> None:
            for i, t in enumerate(traj_list):
                ts = np.ravel(t.ts)
                c = colors[i]
                ax.plot(ts, t.xs[:, 0], color=c, linewidth=2, linestyle="-",
                        label=f"{name_prefix} {i+1} $x_1$", alpha=0.9)
                ax.plot(ts, t.xs[:, 1], color=c, linewidth=2, linestyle="--",
                        label=f"{name_prefix} {i+1} $x_2$", alpha=0.9)

        if context_trajs is not None:
            colors = plt.cm.Blues(np.linspace(0.6, 0.9, len(context_trajs)))
            _plot_traj_time_series(ax, context_trajs, colors, "Context")
        if context_trajs_sparse is not None:
            for i, t in enumerate(context_trajs_sparse):
                ts = np.ravel(t.ts)
                label = "Observations" if i == 0 else None
                ax.scatter(ts, t.xs[:, 0], c="black", s=30, marker="o",
                           label=label, alpha=0.9, zorder=4)
                ax.scatter(ts, t.xs[:, 1], c="black", s=30, marker="o",
                           label=None, alpha=0.9, zorder=4)
        if predicted_trajs is not None:
            colors = plt.cm.Reds(np.linspace(0.6, 1.0, len(predicted_trajs)))
            _plot_traj_time_series(ax, predicted_trajs, colors, "Predicted")
        if true_trajs is not None:
            colors = plt.cm.Greens(np.linspace(0.6, 1.0, len(true_trajs)))
            _plot_traj_time_series(ax, true_trajs, colors, "True")

        ax.set_xlabel("Time", fontsize=14)
        ax.set_ylabel("Coordinate", fontsize=14)
        ax.set_title("Trajectories (time series)", fontsize=16, fontweight="bold")
        if show_legend:
            ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

    else:
        # plot_type == "2D" or "2D_streamplot": phase plane with vector field
        all_traj = []
        if true_trajs is not None:
            for t in true_trajs:
                all_traj.append(t.xs)
        if context_trajs is not None:
            for ctx_traj_obj in context_trajs:
                all_traj.append(ctx_traj_obj.xs)
        if context_trajs_sparse is not None:
            for ctx_sparse_traj_obj in context_trajs_sparse:
                all_traj.append(ctx_sparse_traj_obj.xs)
        if predicted_trajs is not None:
            for pred_traj_obj in predicted_trajs:
                all_traj.append(pred_traj_obj.xs)

        all_points = np.vstack(all_traj)
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()

        x_pad = (x_max - x_min) * 0.1
        y_pad = (y_max - y_min) * 0.1
        x_min -= x_pad
        x_max += x_pad
        y_min -= y_pad
        y_max += y_pad

        if x_range is not None:
            x_min, x_max = x_range
        if y_range is not None:
            y_min, y_max = y_range

        x = np.linspace(x_min, x_max, grid_resolution)
        y = np.linspace(y_min, y_max, grid_resolution)
        X, Y = np.meshgrid(x, y)
        grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)

        all_context_trajs = []
        if context_trajs is not None:
            all_context_trajs += context_trajs
        if context_trajs_sparse is not None:
            all_context_trajs += context_trajs_sparse
        vf = predict_vector_field(model, all_context_trajs, grid_points)

        U = vf[:, 0].reshape(X.shape)
        V = vf[:, 1].reshape(Y.shape)
        magnitude = np.sqrt(U**2 + V**2)

        if plot_type == "2D":
            quiver = ax.quiver(X, Y, U, V, magnitude, cmap="viridis", scale=None,
                            angles="xy", scale_units="xy", width=0.003, alpha=0.3)
            plt.colorbar(quiver, ax=ax, label="Vector Field Magnitude")
        elif plot_type == "2D_streamplot":
            ax.streamplot(X, Y, U, V, color="#dddddd", density=0.6, linewidth=1.5, arrowsize=1.5)
            #plt.colorbar(quiver, ax=ax, label="Vector Field Magnitude")

        if context_trajs is not None:
            colors_context = plt.cm.Blues(np.linspace(0.6, 0.9, len(context_trajs)))
            for i, ctx_traj_obj in enumerate(context_trajs):
                ctx_np = ctx_traj_obj.xs
                if i == 0:
                    ax.scatter(ctx_np[0, 0], ctx_np[0, 1],
                              c="green", s=60, marker="o", label="Start", zorder=5)
                    ax.scatter(ctx_np[-1, 0], ctx_np[-1, 1],
                              c="red", s=50, marker="s", label="End", zorder=5)
                else:
                    ax.scatter(ctx_np[0, 0], ctx_np[0, 1],
                              c="green", s=60, marker="o", zorder=5)
                    ax.scatter(ctx_np[-1, 0], ctx_np[-1, 1],
                              c="red", s=50, marker="s", zorder=5)
                label = f"Context trajectory {i+1}"
                ax.plot(ctx_np[:, 0], ctx_np[:, 1],
                        color=colors_context[i], linewidth=2, label=label, alpha=1.0)

        if context_trajs_sparse is not None:
            colors_sparse = plt.cm.Greys(np.linspace(0.8, 1, len(context_trajs_sparse)))
            for i, ctx_sparse_traj_obj in enumerate(context_trajs_sparse):
                ctx_sparse_np = ctx_sparse_traj_obj.xs
                label = "Observations" if i == 0 else None
                ax.scatter(ctx_sparse_np[:, 0], ctx_sparse_np[:, 1],
                           color=colors_sparse[i], s=30, marker="x",
                           label=label, alpha=0.9, zorder=4, edgecolors="black", linewidths=3)

        if predicted_trajs is not None:
            colors_pred = plt.cm.Reds(np.linspace(0.6, 1.0, len(predicted_trajs)))
            for i, pred_traj_obj in enumerate(predicted_trajs):
                pred_np = pred_traj_obj.xs
                label = f"Predicted trajectory {i+1}"
                ax.plot(pred_np[:, 0], pred_np[:, 1],
                        color=colors_pred[i], linestyle="-", linewidth=2, label=label, alpha=0.8)
                if draw_start_and_end_points:
                    ax.scatter(pred_np[0, 0], pred_np[0, 1],
                           c="green", s=35, marker="o", zorder=5, linewidths=2)
                    ax.scatter(pred_np[-1, 0], pred_np[-1, 1],
                           c="red", s=25, marker="s", zorder=5, linewidths=2)

        if true_trajs is not None:
            colors_true = plt.cm.Greens(np.linspace(0.6, 1.0, len(true_trajs)))
            for i, traj in enumerate(true_trajs):
                label = f"True trajectory {i+1}"
                ax.plot(traj.xs[:, 0], traj.xs[:, 1],
                        color=colors_true[i], linestyle="-", linewidth=2, label=label, alpha=0.8)
                if draw_start_and_end_points:
                    ax.scatter(traj.xs[0, 0], traj.xs[0, 1],
                           c="green", s=35, marker="o", zorder=5, linewidths=2)
                    ax.scatter(traj.xs[-1, 0], traj.xs[-1, 1],
                           c="red", s=25, marker="s", zorder=5, linewidths=2)

        ax.set_xlabel("$x_1$", fontsize=14)
        ax.set_ylabel("$x_2$", fontsize=14)
        ax.set_title("Trajectories and the predicted vector field", fontsize=16, fontweight="bold")
        if show_legend:
            ax.legend(loc="best", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()



def main():
    """Example usage."""
    # Path to checkpoint directory
    checkpoint_dir = Path("models/base_model/checkpoints")
    
    # Load model
    print("Loading model...")
    model = load_odeon_model_from_checkpoint(checkpoint_dir, epoch=None)  # None = best-model
    print(f"Model loaded: {model.title}")
    
    # Example: Create dummy context trajectory
    # In practice, you'd load this from your data
    T = 50  # Number of time points
    D = 2   # State dimension (e.g., for 2D system)
    
    ts = np.linspace(0, 4, T)
    xs = np.stack([np.cos(ts), np.sin(ts)], axis=1)
    context_traj = trajectory(xs=xs, ts=ts)

    ts = np.linspace(-1, 0, T)
    xs = 0.5 * np.stack([np.cos(ts + 0.5), np.sin(ts)], axis=1) + np.stack([ts, 0.2 * np.sin(ts)], axis=1)
    context_traj2 = trajectory(xs=xs, ts=ts)

    
    # Example 1: Predict vector field at specific locations
    print("\nExample 1: Predicting vector field at locations...")
    locations = np.random.randn(5, D)  # 5 random locations
    vector_field = predict_vector_field(model, [context_traj], locations)
    print(f"Vector field predictions shape: {vector_field.shape}")
    print(f"Sample prediction: {vector_field[0]}")
    
    # Example 2: Integrate ODE from initial condition
    print("\nExample 2: Integrating ODE...")
    y0 = np.array([1, -1])
    t_eval = np.linspace(0, 6, 100)  # Time points for solution
    pred_traj = predict_and_integrate_ode(model, [context_traj, context_traj2], y0, t_eval)

    t_eval2 = np.linspace(0, 6, 100)  # Time points for solution
    y0_2 = np.array([0, 0])
    pred_traj2 = predict_and_integrate_ode(model, [context_traj, context_traj2], y0_2, t_eval2)
    
    # Example 3: Plot vector field and trajectories (only works for 2D systems)
    if D == 2:
        print("\nExample 3: Plotting vector field and trajectories...")
        # Can pass single trajectories or lists
        plot_vector_field_and_trajectories(
            model=model,
            context_trajs=[context_traj, context_traj2],  # List of context trajectories
            predicted_trajs=[pred_traj, pred_traj2],  # List of predicted trajectories
            grid_resolution=20,
            #save_path=Path("vector_field_plot.png"),
            show_plot=True,
        )
    else:
        print(f"\nSkipping plotting (only supported for 2D systems, got D={D})")

if __name__ == "__main__":
    main()
