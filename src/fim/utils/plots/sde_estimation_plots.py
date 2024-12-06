import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from torch import Tensor

from fim.data.datasets import FIMSDEDatabatchTuple
from fim.pipelines.sde_pipelines import FIMSDEPipelineOutput
from fim.utils.helper import select_dimension_for_plot


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def plot_1d_vf_real_and_estimation(
    locations: Tensor,
    drift_at_locations_real: Tensor,
    drift_at_locations_estimation: Tensor,
    diffusion_at_locations_real: Tensor,
    diffusion_at_locations_estimation: Tensor,
    show=True,
):
    """
    Plots estimated drift and diffusion along real parts and returns the figure.
    Selected 1D data already.
    """
    fig, ax1 = plt.subplots()
    color = "tab:red"
    ax1.set_xlabel("State")
    ax1.set_ylabel("Drift", color=color)
    ax1.plot(locations, drift_at_locations_real, color="r", label="Real Drift")
    ax1.plot(locations, drift_at_locations_estimation, color="black", label="Estimator Drift")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    color = "tab:blue"
    ax2.set_ylabel("Diffusion", color=color)  # we already handled the x-label with ax1
    ax2.plot(locations, diffusion_at_locations_real, color="tab:blue", linestyle="--", label="Real Diffusion")
    ax2.plot(locations, diffusion_at_locations_estimation, color="black", linestyle="--", label="Estimator Diffusion")
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # Place the legend above the plot
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=4)
    if show:
        plt.show()
    else:
        return fig


def plot_2d_vf_real_and_estimation(
    locations: Tensor,
    drift_at_locations_real: Tensor,
    drift_at_locations_estimation: Tensor,
    diffusion_at_locations_real: Tensor,
    diffusion_at_locations_estimation: Tensor,
    show: bool = True,
):
    """
    Plots estimated drift and diffusion along real parts and returns the figure.
    Selected 2D data already.
    """
    # Extract grid points (x, y)
    x, y = locations[:, 0], locations[:, 1]

    # Real vector fields
    u_real_drift, v_real_drift = drift_at_locations_real[:, 0], drift_at_locations_real[:, 1]
    u_real_diffusion, v_real_diffusion = diffusion_at_locations_real[:, 0], diffusion_at_locations_real[:, 1]

    # Estimated vector fields
    u_estimated_drift, v_estimated_drift = drift_at_locations_estimation[:, 0], drift_at_locations_estimation[:, 1]
    u_estimated_diffusion, v_estimated_diffusion = diffusion_at_locations_estimation[:, 0], diffusion_at_locations_estimation[:, 1]

    # Create a figure and 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    # Plot real drift
    axs[0, 0].quiver(x, y, u_real_drift, v_real_drift)
    axs[0, 0].set_title("Real Drift")

    # Plot real diffusion
    axs[0, 1].quiver(x, y, u_real_diffusion, v_real_diffusion)
    axs[0, 1].set_title("Real Diffusion")

    # Plot estimated drift
    axs[1, 0].quiver(x, y, u_estimated_drift, v_estimated_drift)
    axs[1, 0].set_title("Estimated Drift")

    # Plot estimated diffusion
    axs[1, 1].quiver(x, y, u_estimated_diffusion, v_estimated_diffusion)
    axs[1, 1].set_title("Estimated Diffusion")

    # Adjust layout
    plt.tight_layout()

    if show:
        plt.show()
    else:
        return fig  # Return the figure to log


def plot_3d_vf_real_and_estimation(
    locations: Tensor,
    drift_at_locations_real: Tensor,
    drift_at_locations_estimation: Tensor,
    diffusion_at_locations_real: Tensor,
    diffusion_at_locations_estimation: Tensor,
    your_fixed_x_value: float = -1.0,
    show: bool = True,
):
    """
    Plots estimated drift and diffusion along real parts and returns the figure.
    Selected 3D data already.
    """
    # Assuming `locations` is a NumPy array of shape (P, 3), and `estimated_drift`, `estimated_diffusion`, `ground_truth_drift`, and `ground_truth_diffusion` are also NumPy arrays of shape (P, 3).
    tolerance = 0.1  # tolerance for finding points close to the desired x_0 value

    # Define your tolerance, locations, and fixed x value as before
    slice_indices = np.where(np.abs(locations[:, 0] - your_fixed_x_value) < tolerance)[0]

    # Calculate the largest perfect square that is less than or equal to the length of slice_indices
    closest_square = int(np.floor(np.sqrt(len(slice_indices))) ** 2)

    # Select only the closest_square number of indices for a perfect grid
    slice_indices = slice_indices[:closest_square]

    # Now you can reshape into a square grid for visualization
    grid_size = int(np.sqrt(closest_square))

    # Select corresponding values for both model estimates and ground truth
    drift_slice = drift_at_locations_estimation[slice_indices]
    diffusion_slice = diffusion_at_locations_estimation[slice_indices]

    ground_truth_drift_slice = drift_at_locations_real[slice_indices]
    ground_truth_diffusion_slice = diffusion_at_locations_real[slice_indices]

    drift_slice_reshaped = drift_slice.reshape(grid_size, grid_size, -1)
    diffusion_slice_reshaped = diffusion_slice.reshape(grid_size, grid_size, -1)

    ground_truth_drift_reshaped = ground_truth_drift_slice.reshape(grid_size, grid_size, -1)
    ground_truth_diffusion_reshaped = ground_truth_diffusion_slice.reshape(grid_size, grid_size, -1)

    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(4, 3, figure=fig, wspace=0.005, hspace=0.05)  # Adjust wspace and hspace

    fig.suptitle(f"Slice at x_0 = {your_fixed_x_value:.5f}")

    axs = []
    for i in range(4):
        row = []
        for j in range(3):
            ax = fig.add_subplot(gs[i, j])
            row.append(ax)
        axs.append(row)

    for i in range(3):  # Loop over dimensions 0, 1, 2
        # Ground-truth drift and diffusion
        axs[0][i].imshow(ground_truth_drift_reshaped[..., i], origin="lower", cmap="viridis")
        axs[0][i].set_title(f"Dimension {i}")
        axs[0][i].set_ylabel("Ground-Truth Drift")

        # Model drift
        axs[1][i].imshow(drift_slice_reshaped[..., i], origin="lower", cmap="viridis")
        axs[1][i].set_ylabel("Model Drift")

        # Ground-truth diffusion
        axs[2][i].imshow(ground_truth_diffusion_reshaped[..., i], origin="lower", cmap="viridis")
        axs[2][i].set_ylabel("Ground-Truth Diffusion")

        # Model diffusion
        axs[3][i].imshow(diffusion_slice_reshaped[..., i], origin="lower", cmap="viridis")
        axs[3][i].set_ylabel("Model Diffusion")

        # Remove axis labels and ticks from the last two columns
        if i > 0:
            for j in range(4):  # Iterate over rows
                axs[j][i].set_yticks([])
                axs[j][i].set_ylabel("")
        # Remove ticks from all axes
        for j in range(4):
            axs[j][i].set_xticks([])
            axs[j][i].set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the suptitle
    if show:
        plt.show()
    else:
        return fig


def images_log_3D(databatch_target: FIMSDEDatabatchTuple, pipeline_output: FIMSDEPipelineOutput):
    selected_data = select_dimension_for_plot(
        3,
        databatch_target.dimension_mask,
        databatch_target.locations,
        pipeline_output.drift_at_locations_estimator,
        pipeline_output.diffusion_at_locations_estimator,
        databatch_target.drift_at_locations,
        databatch_target.diffusion_at_locations,
        index_to_select=0,
    )

    locations, drift_at_locations_real, diffusion_at_locations_real, drift_at_locations_estimation, diffusion_at_locations_estimation = (
        selected_data
    )
    fig = plot_3d_vf_real_and_estimation(
        locations,
        drift_at_locations_real,
        drift_at_locations_estimation,
        diffusion_at_locations_real,
        diffusion_at_locations_estimation,
        your_fixed_x_value=0.1,
        show=False,
    )
    return fig


def images_log_2D(databatch_target: FIMSDEDatabatchTuple, pipeline_output: FIMSDEPipelineOutput):
    selected_data = select_dimension_for_plot(
        2,
        databatch_target.dimension_mask,
        databatch_target.locations,
        pipeline_output.drift_at_locations_estimator,
        pipeline_output.diffusion_at_locations_estimator,
        databatch_target.drift_at_locations,
        databatch_target.diffusion_at_locations,
        index_to_select=0,
    )
    locations, drift_at_locations_real, diffusion_at_locations_real, drift_at_locations_estimation, diffusion_at_locations_estimation = (
        selected_data
    )
    fig = plot_2d_vf_real_and_estimation(
        locations,
        drift_at_locations_real,
        drift_at_locations_estimation,
        diffusion_at_locations_real,
        diffusion_at_locations_estimation,
        show=False,
    )
    return fig


def images_log_1D(databatch_target: FIMSDEDatabatchTuple, pipeline_output: FIMSDEPipelineOutput):
    selected_data = select_dimension_for_plot(
        1,
        databatch_target.dimension_mask,
        databatch_target.locations,
        pipeline_output.drift_at_locations_estimator,
        pipeline_output.diffusion_at_locations_estimator,
        databatch_target.drift_at_locations,
        databatch_target.diffusion_at_locations,
        index_to_select=0,
    )

    locations, drift_at_locations_real, diffusion_at_locations_real, drift_at_locations_estimation, diffusion_at_locations_estimation = (
        selected_data
    )
    fig = plot_1d_vf_real_and_estimation(
        locations,
        drift_at_locations_real,
        drift_at_locations_estimation,
        diffusion_at_locations_real,
        diffusion_at_locations_estimation,
        show=False,
    )
    return fig
