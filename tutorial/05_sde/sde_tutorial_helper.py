import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch


def prepare_sde_data(data_sample, num_paths=5, dt=0.002):
    """
    Convert dataset sample to model input format with custom time spacing.
    Automatically handles both 1D and 2D trajectory dimensions.
    """
    # Extract trajectory data
    trajectories = torch.tensor(data_sample["observations"][:num_paths], dtype=torch.float32)

    # Create evenly spaced times with specified dt
    num_timesteps = trajectories.shape[1]
    time_end = (num_timesteps - 1) * dt
    times_1d = torch.linspace(0, time_end, num_timesteps, dtype=torch.float32)

    # Times should be shared across all paths: [1, 1, num_timesteps, 1]
    times = times_1d.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

    # Reshape for model input: [batch_size, num_paths, time_steps, dimensions]
    trajectories = trajectories.unsqueeze(0)  # Add batch dimension
    dims = trajectories.shape[-1]

    # Dynamically handle 1D vs 2D grids based on data dimensionality
    if dims == 1:
        x_min, x_max = trajectories.min(), trajectories.max()
        locations = torch.linspace(x_min - 0.5, x_max + 0.5, 50).unsqueeze(0).unsqueeze(-1)
    elif dims == 2:
        x_min, x_max = trajectories[:, :, :, 0].min(), trajectories[:, :, :, 0].max()
        y_min, y_max = trajectories[:, :, :, 1].min(), trajectories[:, :, :, 1].max()

        x_grid = torch.linspace(x_min - 0.5, x_max + 0.5, 25)
        y_grid = torch.linspace(y_min - 0.5, y_max + 0.5, 25)

        X, Y = torch.meshgrid(x_grid, y_grid, indexing="ij")
        locations = torch.stack([X.flatten(), Y.flatten()], dim=-1)
        locations = locations.unsqueeze(0)  # Add batch dimension
    else:
        raise ValueError(f"Unsupported data dimension: {dims}D. Expected 1D or 2D.")

    return {"obs_values": trajectories, "obs_times": times, "locations": locations}


def plot_sde_results(input_data, estimated_concepts, sampled_paths, sampled_paths_time_grid, title):
    """
    Unified entry point for plotting. Dispatches to 1D or 2D plotting
    depending on the dimensionality of the observations.
    """
    dims = input_data["obs_values"].shape[-1]

    if dims == 1:
        return plot_1d_sde_results(input_data, estimated_concepts, sampled_paths, sampled_paths_time_grid, title)
    elif dims == 2:
        return plot_2d_sde_results(input_data, estimated_concepts, sampled_paths, title)
    else:
        raise ValueError(f"No plotting layout implemented for {dims}D data.")


def plot_1d_sde_results(input_data, estimated_concepts, sampled_paths, sampled_paths_time_grid, title):
    """
    Plot trajectories, estimated drift, and diffusion functions
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=16)

    # Extract data
    trajectories = input_data["obs_values"][0]  # Remove batch dimension
    times = input_data["obs_times"][0, 0, :, 0]  # Shared times: [num_timesteps]
    locations = estimated_concepts.locations[0, :, 0].cpu()  # Remove batch dimension
    drift = estimated_concepts.drift[0, :, 0].cpu()
    diffusion = estimated_concepts.diffusion[0, :, 0].cpu()

    # Plot trajectories
    axes[0, 0].set_title("Ground-Truth Trajectories")
    # Original trajectories
    for i in range(min(3, trajectories.shape[0])):
        axes[0, 0].plot(times, trajectories[i, :, 0], linewidth=1.5, color="black")

    # Sampled trajectories (if available)
    if sampled_paths is not None:
        axes[0, 1].set_title("Sampled Trajectories")
        sampled_times = sampled_paths_time_grid[0, 0, :, 0]  # Time grid from sampled paths
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["#D55E00", "#56B4E9"], N=3)
        colors = [mcolors.to_hex(cmap(i)) for i in range(3)]
        for i in range(3):
            axes[0, 1].plot(sampled_times, sampled_paths[0, i, :, 0], linewidth=1.5, color=colors[i])
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("X(t)")

    # Plot estimated drift
    axes[1, 0].set_title("Drift")
    axes[1, 0].plot(locations, drift, "-", linewidth=6, label="Estimated Drift", color="#0072B2")
    axes[1, 0].plot(locations, 4 * (locations - locations**3), linestyle="dotted", linewidth=3, label="True Drift", color="black")
    axes[1, 0].set_xlabel("X")
    axes[1, 0].set_ylabel("f(x)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Plot estimated diffusion
    axes[1, 1].set_title("Diffusion")
    axes[1, 1].plot(locations, diffusion, "-", linewidth=6, label="Estimated Diffusion", color="#0072B2")
    axes[1, 1].plot(
        locations,
        torch.sqrt(torch.max(4 - 1.25 * locations**2, torch.tensor(0.0))),
        linestyle="dotted",
        linewidth=3,
        label="True Diffusion",
        color="black",
    )
    axes[1, 1].set_xlabel("X")
    axes[1, 1].set_ylabel("g(x)")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    return fig


def _subsample_2D_locs(locations, size_dim, subsample_factor):
    locations = locations.reshape((size_dim, size_dim, 2))
    locations = locations[::subsample_factor, ::subsample_factor]
    locations = locations.reshape((1, -1, 2))

    return locations


def plot_2d_sde_results(input_data, estimated_concepts, sampled_paths, title, location_subsample_factor=2):
    """
    Plot 2D system trajectories and vector fields
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(title, fontsize=16)

    # Subsample both axes for cleaner visualization
    size_per_axis = int(np.sqrt(estimated_concepts.locations.shape[1]).item())

    drift = _subsample_2D_locs(estimated_concepts.drift[..., :2], size_per_axis, location_subsample_factor)
    diffusion = _subsample_2D_locs(estimated_concepts.diffusion[..., :2], size_per_axis, location_subsample_factor)
    locations = _subsample_2D_locs(estimated_concepts.locations[..., :2], size_per_axis, location_subsample_factor)

    # Extract data, remove batch dimension
    trajectories = input_data["obs_values"][0]
    locations = locations[0].cpu()
    drift = drift[0].cpu()
    diffusion = diffusion[0].cpu()

    # Reshape for vector field plotting
    grid_size = int(np.sqrt(locations.shape[0]))
    X = locations[:, 0].reshape(grid_size, grid_size)
    Y = locations[:, 1].reshape(grid_size, grid_size)
    U = drift[:, 0].reshape(grid_size, grid_size)
    U_true = Y
    V = drift[:, 1].reshape(grid_size, grid_size)
    V_true = -(X**3 - X + 0.35 * Y)

    # Plot trajectories
    axes[0, 0].set_title("Ground-Truth Phase Space Trajectories")
    for i in range(trajectories.shape[0]):
        axes[0, 0].plot(trajectories[i, :, 0], trajectories[i, :, 1], linewidth=1, color="black")
        axes[0, 0].scatter(trajectories[i, 0, 0], trajectories[i, 0, 1], s=50, marker="o", color="black")
    axes[0, 0].set_xlabel("X")
    axes[0, 0].set_ylabel("Y")

    axes[0, 1].set_title("Sampled Phase Space Trajectories")
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["#D55E00", "#56B4E9"], N=5)
    colors = [mcolors.to_hex(cmap(i)) for i in range(5)]
    for i in range(5):
        axes[0, 1].plot(sampled_paths[0, i, :, 0], sampled_paths[0, i, :, 1], linewidth=1, color=colors[i])
        axes[0, 1].scatter(sampled_paths[0, i, 0, 0], sampled_paths[0, i, 0, 1], s=50, marker="o", color=colors[i])
    axes[0, 1].set_xlabel("X")
    axes[0, 1].set_ylabel("Y")

    # Plot drift vector field
    axes[1, 0].set_title("Drift")
    axes[1, 0].quiver(X, Y, U_true, V_true, label="True Drift", color="black")
    axes[1, 0].quiver(X, Y, U, V, label="Estimated Drift", color="#0072B2")

    axes[1, 0].set_xlabel("X")
    axes[1, 0].set_ylabel("Y")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    U = diffusion[:, 0].reshape(grid_size, grid_size)
    U_true = torch.ones_like(U)
    V = diffusion[:, 1].reshape(grid_size, grid_size)
    V_true = torch.ones_like(V)

    # Plot diffusion vector field
    axes[1, 1].set_title("Diffusion")
    axes[1, 1].quiver(X, Y, U, V, label="Estimated Diffusion", color="black")
    axes[1, 1].quiver(X, Y, U_true, V_true, label="True Diffusion", color="#0072B2")

    axes[1, 1].set_xlabel("X")
    axes[1, 1].set_ylabel("Y")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    return fig
