

from pathlib import Path
import pickle
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np
from torch import Tensor
import torch


from fim.data.utils import load_h5s_in_folder
from fim.utils.plots.sde_data_exploration_plots import plot_paths_in_axis
from fim.utils.plots.sde_estimation_plots import plot_2d_vf_real_and_estimation_axes


model_data_pickle_path = Path("evaluations/synthetic_datasets/01221033_30k_deg_3_ablation_studies_only_learn_scale/model_evaluations/11M_params/wang_two_d_80000_points/default11M_params_wang_two_d_80000_points.pickle")
ground_truth_data_folder_path = Path("/cephfs_projects/foundation_models/data/SDE/test/20241223_opper_and_wang_cut_to_128_lenght_paths/two_d_wang_80000_points")
comparison_model_data_folder_path = Path("data/processed/test/20250117_wang_estimated_equations/bisde_est_2D_synth_80000_points_split_128_length")

with open(model_data_pickle_path, "rb") as f:
    model_data = pickle.load(f).results["estimated_concepts"]
ground_truth_data = load_h5s_in_folder(ground_truth_data_folder_path)
comparison_model_data = load_h5s_in_folder(comparison_model_data_folder_path)

model_locations, model_drift, model_diffusion = model_data.locations[:,:,:2], model_data.drift[:,:,:2], model_data.diffusion[:,:,:2]
ground_truth_locations, ground_truth_drift, ground_truth_diffusion = ground_truth_data["locations"], ground_truth_data["drift_at_locations"], ground_truth_data["diffusion_at_locations"]
comparison_model_locations, comparison_model_drift, comparison_model_diffusion = comparison_model_data["locations"], comparison_model_data["drift_at_locations"], comparison_model_data["diffusion_at_locations"]

assert model_drift.shape == ground_truth_drift.shape, f"Drifts have different shapes between model and ground truth data: {model_drift.shape} vs {ground_truth_drift.shape}"
assert model_drift.shape == comparison_model_drift.shape, f"Drifts have different shapes between model and comparison model data: {model_drift.shape} vs {comparison_model_drift.shape}"
assert model_diffusion.shape == ground_truth_diffusion.shape, f"Diffusions have different shapes between model and ground truth data: {model_diffusion.shape} vs {ground_truth_diffusion.shape}"
assert model_diffusion.shape == comparison_model_diffusion.shape, f"Diffusions have different shapes between model and comparison model data: {model_diffusion.shape} vs {comparison_model_diffusion.shape}"

# Only consider every nth point
n = 10
model_locations, model_drift, model_diffusion = model_locations[:,::n], model_drift[:,::n], model_diffusion[:,::n]
ground_truth_locations, ground_truth_drift, ground_truth_diffusion = ground_truth_locations[:,::n], ground_truth_drift[:,::n], ground_truth_diffusion[:,::n]
comparison_model_locations, comparison_model_drift, comparison_model_diffusion = comparison_model_locations[:,::n], comparison_model_drift[:,::n], comparison_model_diffusion[:,::n]



## Some code to restrict the plot to a certain region
# too_large_mask = (torch.abs(model_locations) > 1)
# # Compute the logical or between the last two dimensions
# too_large_mask = too_large_mask.any(dim=-1)[:,:,None]
# too_large_mask = too_large_mask.repeat(1, 1, 2)

# model_locations[too_large_mask] = 0
# model_drift[too_large_mask] = 0
# model_diffusion[too_large_mask] = 0
# ground_truth_drift[too_large_mask] = 0
# ground_truth_diffusion[too_large_mask] = 0




def create_2D_quiver_plot(
    locations: Tensor,  # [B, G, D]
    ground_truth_drift: Tensor,  # [B, G, D]
    ground_truth_diffusion: Tensor,  # [B, G, D]
    estimated_drift: Tensor,  # [B, G, D]
    estimated_diffusion: Tensor,  # [B, G, D]
    comparison_model_drift: Tensor = None,  # [B, G, D]
    comparison_model_diffusion: Tensor = None,  # [B, G, D]
    **kwargs,
):
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
            ground_truth_diffusion[row, ..., :2],
            estimated_drift[row, ..., :2],
            estimated_diffusion[row, ..., :2],
            comparison_model_drift=comparison_model_drift[row, ..., :2] if comparison_model_drift is not None else None,
            comparison_model_diffusion=comparison_model_diffusion[row, ..., :2] if comparison_model_diffusion is not None else None,
        )

    return fig

def plot_2d_vf_real_and_estimation_axes(
    axis_drift,
    axis_diffusion,
    locations: Tensor,
    drift_at_locations_real: Tensor,
    diffusion_at_locations_real: Tensor,
    drift_at_locations_estimation: Tensor,
    diffusion_at_locations_estimation: Tensor,
    comparison_model_drift: Tensor = None,
    comparison_model_diffusion: Tensor = None,
    **kwargs
):
    use_comparison_model = comparison_model_drift is not None and comparison_model_diffusion is not None
    ground_truth_color = kwargs.get("ground_truth_color", "#0072B2")
    fim_color = kwargs.get("fim_color", "#CC79A7")
    comparison_color = kwargs.get("comparison_color", "#009E73")
    
    # Extract grid points (x, y)
    x, y = locations[:, 0], locations[:, 1]

    # Real vector fields
    u_real_drift, v_real_drift = drift_at_locations_real[:, 0], drift_at_locations_real[:, 1]
    u_real_diffusion, v_real_diffusion = diffusion_at_locations_real[:, 0], diffusion_at_locations_real[:, 1]

    # Estimated vector fields
    u_estimated_drift, v_estimated_drift = drift_at_locations_estimation[:, 0], drift_at_locations_estimation[:, 1]
    u_estimated_diffusion, v_estimated_diffusion = diffusion_at_locations_estimation[:, 0], diffusion_at_locations_estimation[:, 1]
    
    # Comparison model vector fields
    if use_comparison_model:
        u_comparison_model_drift, v_comparison_model_drift = comparison_model_drift[:, 0], comparison_model_drift[:, 1]
        u_comparison_model_diffusion, v_comparison_model_diffusion = comparison_model_diffusion[:, 0], comparison_model_diffusion[:, 1]

    # Plot drift
    real_drift_quiver = axis_drift.quiver(
        x,
        y, 
        u_real_drift, 
        v_real_drift, 
        color=ground_truth_color,
    )

    axis_drift.quiver(
        x,
        y,
        u_estimated_drift,
        v_estimated_drift,
        scale=real_drift_quiver.scale,
        color=fim_color,
    )
    
    if use_comparison_model:
        axis_drift.quiver(
            x,
            y,
            u_comparison_model_drift,
            v_comparison_model_drift,
            scale=real_drift_quiver.scale,
            color=comparison_color,
        )
    
    axis_drift.set_title("Drift")

    # Plot diffusion
    real_diffusion_quiver = axis_diffusion.quiver(
        x,
        y,
        u_real_diffusion,
        v_real_diffusion,
        color=ground_truth_color,
    )

    axis_diffusion.quiver(
        x,
        y,
        u_estimated_diffusion,
        v_estimated_diffusion,
        scale=real_diffusion_quiver.scale,
        color=fim_color,
    )
    
    if use_comparison_model:
        axis_diffusion.quiver(
            x,
            y,
            u_comparison_model_diffusion,
            v_comparison_model_diffusion,
            scale=real_diffusion_quiver.scale,
            color=comparison_color,
        )
    
    axis_diffusion.set_title("Diffusion")
    
    # Create custom legend handles with arrows using Line2D
    legend_elements = [
        Line2D([0], [0], color=ground_truth_color, lw=2, label='Ground Truth', marker=r'$\rightarrow$', markersize=10, linestyle='None'),
        Line2D([0], [0], color=fim_color, lw=2, label='FIM', marker=r'$\rightarrow$', markersize=10, linestyle='None')
    ]
    if use_comparison_model:
        legend_elements.append(Line2D([0], [0], color=comparison_color, lw=2, label='Comparison Model', marker=r'$\rightarrow$', markersize=10, linestyle='None'))

    # Add legends
    axis_drift.legend(handles=legend_elements)
    axis_diffusion.legend(handles=legend_elements)

fig = create_2D_quiver_plot(model_locations, ground_truth_drift, ground_truth_diffusion, model_drift, model_diffusion, comparison_model_drift=comparison_model_drift, comparison_model_diffusion=comparison_model_diffusion)

plt.savefig("2D_quiver_plot.png")