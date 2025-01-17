

from pathlib import Path
import pickle
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from torch import Tensor
import torch


from fim.data.utils import load_h5s_in_folder
from fim.utils.plots.sde_data_exploration_plots import plot_paths_in_axis
from fim.utils.plots.sde_estimation_plots import plot_2d_vf_real_and_estimation_axes


model_data_pickle_path = Path("evaluations/synthetic_datasets/01151152_testing/model_evaluations/1000_threshold_linear_softmax_attn-2_layers_GNOT_repeated_4_layers/wang_two_d_80000_points/default1000_threshold_linear_softmax_attn-2_layers_GNOT_repeated_4_layers_wang_two_d_80000_points.pickle")
ground_truth_data_folder_path = Path("/cephfs_projects/foundation_models/data/SDE/test/20241223_opper_and_wang_cut_to_128_lenght_paths/two_d_wang_80000_points/")

with open(model_data_pickle_path, "rb") as f:
    model_data = pickle.load(f).results["estimated_concepts"]
ground_truth_data = load_h5s_in_folder(ground_truth_data_folder_path)

model_locations, model_drift, model_diffusion = model_data.locations, model_data.drift[:,:,:2], model_data.diffusion[:,:,:2]
ground_truth_locations, ground_truth_drift, ground_truth_diffusion = ground_truth_data["locations"], ground_truth_data["drift_at_locations"], ground_truth_data["diffusion_at_locations"]

assert model_drift.shape == ground_truth_drift.shape, f"Drifts have different shapes between model and ground truth data: {model_drift.shape} vs {ground_truth_drift.shape}"
assert model_diffusion.shape == ground_truth_diffusion.shape, f"Diffusions have different shapes between model and ground truth data: {model_diffusion.shape} vs {ground_truth_diffusion.shape}"

# Only consider every nth point
n = 10
model_locations, model_drift, model_diffusion = model_locations[:,::n], model_drift[:,::n], model_diffusion[:,::n]
ground_truth_locations, ground_truth_drift, ground_truth_diffusion = ground_truth_locations[:,::n], ground_truth_drift[:,::n], ground_truth_diffusion[:,::n]




def create_2D_quiver_plot(
    locations: Tensor,  # [B, G, D]
    ground_truth_drift: Tensor,  # [B, G, D]
    ground_truth_diffusion: Tensor,  # [B, G, D]
    estimated_drift: Tensor,  # [B, G, D]
    estimated_diffusion: Tensor,  # [B, G, D]
    **kwargs,
):
    """
    Plot B equations of dimension 2 as a grid of figures.
    Each row contains ground-truth and (single) model estimation drift and diffusion, and P 2D paths from data and sampled from model.

    Args: Ground-truth and estimation of paths and vector fields. Shape: [B, G, D] or [B, P, T, D] for inputs and [B, P, L, D] model paths.

    Returns: Figure with data from B equations.
    """
    ncols = 2
    nrows = locations.shape[0]

    figsize_per_col = kwargs.get("figsize_per_col", 4)
    figsize_per_row = kwargs.get("figsize_per_row", 4)
    figsize = (ncols * figsize_per_col, nrows * figsize_per_row)

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)

    if nrows == 1:
        axs = axs.reshape(1, -1)

    for row in range(nrows):
        plot_2d_vf_real_and_estimation_axes(
            axs[row, 0],
            axs[row, 1],
            locations[row, ..., :2],
            ground_truth_drift[row, ..., :2],
            estimated_drift[row, ..., :2],
            ground_truth_diffusion[row, ..., :2],
            estimated_diffusion[row, ..., :2],
        )

    return fig


def plot_2d_vf_real_and_estimation_axes(
    axis_drift,
    axis_diffusion,
    locations: Tensor,
    drift_at_locations_real: Tensor,
    drift_at_locations_estimation: Tensor,
    diffusion_at_locations_real: Tensor,
    diffusion_at_locations_estimation: Tensor,
):
    # Extract grid points (x, y)
    x, y = locations[:, 0], locations[:, 1]

    # Real vector fields
    u_real_drift, v_real_drift = drift_at_locations_real[:, 0], drift_at_locations_real[:, 1]
    u_real_diffusion, v_real_diffusion = diffusion_at_locations_real[:, 0], diffusion_at_locations_real[:, 1]

    # Estimated vector fields
    u_estimated_drift, v_estimated_drift = drift_at_locations_estimation[:, 0], drift_at_locations_estimation[:, 1]
    u_estimated_diffusion, v_estimated_diffusion = diffusion_at_locations_estimation[:, 0], diffusion_at_locations_estimation[:, 1]

    # Plot drift
    real_drift_quiver = axis_drift.quiver(
        x,
        y, 
        u_real_drift, 
        v_real_drift, 
        color="red"
    )

    axis_drift.quiver(
        x,
        y,
        u_estimated_drift,
        v_estimated_drift,
        scale=real_drift_quiver.scale,
        color="black",
    )
    axis_drift.set_title("Drift")

    # Plot diffusion
    real_diffusion_quiver = axis_diffusion.quiver(
        x,
        y,
        u_real_diffusion,
        v_real_diffusion,
        color="red",
    )

    axis_diffusion.quiver(
        x,
        y,
        u_estimated_diffusion,
        v_estimated_diffusion,
        scale=real_diffusion_quiver.scale,
        color="black",
    )
    axis_diffusion.set_title("Diffusion")

    # Plot diffusion
    real_diffusion_quiver = axis_diffusion.quiver(
        x,
        y,
        u_real_diffusion,
        v_real_diffusion,
        color="red",
    )

    axis_diffusion.quiver(
        x,
        y,
        u_estimated_diffusion,
        v_estimated_diffusion,
        scale=real_diffusion_quiver.scale,
        color="black",
    )
    axis_diffusion.set_title("Diffusion")


fig = create_2D_quiver_plot(model_locations, ground_truth_drift, ground_truth_diffusion, model_drift, model_diffusion)

plt.savefig("2D_quiver_plot.png")
