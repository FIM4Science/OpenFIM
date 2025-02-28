import pickle
from pathlib import Path

import torch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from torch import Tensor

from fim.data.utils import load_h5s_in_folder


model_data_pickle_path = Path(
    "evaluations/synthetic_datasets/01221033_30k_deg_3_ablation_studies_only_learn_scale/model_evaluations/11M_params/wang_double_well_25000_points/default11M_params_wang_double_well_25000_points.pickle"
)
ground_truth_data_folder_path = Path(
    "/cephfs_projects/foundation_models/data/SDE/test/20241223_opper_and_wang_cut_to_128_lenght_paths/double_well_wang_25000_points"
)
comparison_model_data_folder_path = Path(
    "data/processed/test/20250117_wang_estimated_equations/bisde_est_double_well_25000_points_split_128_length"
)

with open(model_data_pickle_path, "rb") as f:
    model_data = pickle.load(f).results["estimated_concepts"]
ground_truth_data = load_h5s_in_folder(ground_truth_data_folder_path)
comparison_model_data = load_h5s_in_folder(comparison_model_data_folder_path)

model_values, model_drift, model_diffusion = model_data.locations[:, :, :1], model_data.drift[:, :, :1], model_data.diffusion[:, :, :1]
ground_truth_values, ground_truth_drift, ground_truth_diffusion = (
    ground_truth_data["locations"],
    ground_truth_data["drift_at_locations"],
    ground_truth_data["diffusion_at_locations"],
)
comparison_model_values, comparison_model_drift, comparison_model_diffusion = (
    comparison_model_data["locations"],
    comparison_model_data["drift_at_locations"],
    comparison_model_data["diffusion_at_locations"],
)

assert model_drift.shape == ground_truth_drift.shape, (
    f"Drifts have different shapes between model and ground truth data: {model_drift.shape} vs {ground_truth_drift.shape}"
)
assert model_drift.shape == comparison_model_drift.shape, (
    f"Drifts have different shapes between model and comparison model data: {model_drift.shape} vs {comparison_model_drift.shape}"
)
assert model_diffusion.shape == ground_truth_diffusion.shape, (
    f"Diffusions have different shapes between model and ground truth data: {model_diffusion.shape} vs {ground_truth_diffusion.shape}"
)
assert model_diffusion.shape == comparison_model_diffusion.shape, (
    f"Diffusions have different shapes between model and comparison model data: {model_diffusion.shape} vs {comparison_model_diffusion.shape}"
)
assert torch.allclose(model_values, ground_truth_values), "Values are not the same between model and ground truth data"
assert torch.allclose(model_values, comparison_model_values), "Values are not the same between model and comparison model data"

# Only consider every nth point
n = 10
model_values, model_drift, model_diffusion = model_values[:, ::n], model_drift[:, ::n], model_diffusion[:, ::n]
ground_truth_values, ground_truth_drift, ground_truth_diffusion = (
    ground_truth_values[:, ::n],
    ground_truth_drift[:, ::n],
    ground_truth_diffusion[:, ::n],
)
comparison_model_values, comparison_model_drift, comparison_model_diffusion = (
    comparison_model_values[:, ::n],
    comparison_model_drift[:, ::n],
    comparison_model_diffusion[:, ::n],
)


def create_1D_plot(
    values: Tensor,  # [B, G]
    ground_truth_drift: Tensor,  # [B, G]
    ground_truth_diffusion: Tensor,  # [B, G]
    estimated_drift: Tensor,  # [B, G]
    estimated_diffusion: Tensor,  # [B, G]
    comparison_model_drift: Tensor = None,  # [B, G]
    comparison_model_diffusion: Tensor = None,  # [B, G]
    zoom_areas_drift: list = [dict(xlim=(-4, -3.8), ylim=(50, 60))],  # List of dicts with 'xlim' and 'ylim' for each plot # noqa: C408
    zoom_areas_diffusion: list = [dict(xlim=(-4, -3.5), ylim=(3.7, 4.5))],  # Same as above for diffusion # noqa: C408
    **kwargs,
):
    ncols = 2
    nrows = values.shape[0]

    figsize_per_col = kwargs.get("figsize_per_col", 6)
    figsize_per_row = kwargs.get("figsize_per_row", 4)
    figsize = (ncols * figsize_per_col, nrows * figsize_per_row)

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)

    if nrows == 1:
        axs = axs.reshape(1, -1)

    for row in range(nrows):
        plot_1D_axes(
            axs[row, 0],
            axs[row, 1],
            values[row],
            ground_truth_drift[row],
            ground_truth_diffusion[row],
            estimated_drift[row],
            estimated_diffusion[row],
            comparison_model_drift=comparison_model_drift[row] if comparison_model_drift is not None else None,
            comparison_model_diffusion=comparison_model_diffusion[row] if comparison_model_diffusion is not None else None,
            zoom_area_drift=zoom_areas_drift[row] if zoom_areas_drift else None,
            zoom_area_diffusion=zoom_areas_diffusion[row] if zoom_areas_diffusion else None,
            **kwargs,
        )

    return fig


def plot_1D_axes(
    ax_drift,
    ax_diffusion,
    values: Tensor,
    drift_real: Tensor,
    diffusion_real: Tensor,
    drift_estimated: Tensor,
    diffusion_estimated: Tensor,
    comparison_model_drift: Tensor = None,
    comparison_model_diffusion: Tensor = None,
    zoom_area_drift: dict = None,
    zoom_area_diffusion: dict = None,
    **kwargs,
):
    use_comparison = comparison_model_drift is not None and comparison_model_diffusion is not None
    ground_truth_color = kwargs.get("ground_truth_color", "#0072B2")
    fim_color = kwargs.get("fim_color", "#CC79A7")
    comparison_color = kwargs.get("comparison_color", "#009E73")

    values_np = values.cpu().numpy()
    drift_real_np = drift_real.cpu().numpy()
    diffusion_real_np = diffusion_real.cpu().numpy()
    drift_estimated_np = drift_estimated.cpu().numpy()
    diffusion_estimated_np = diffusion_estimated.cpu().numpy()

    if use_comparison:
        comparison_model_drift_np = comparison_model_drift.cpu().numpy()
        comparison_model_diffusion_np = comparison_model_diffusion.cpu().numpy()

    # Plot drift
    ax_drift.plot(values_np, drift_real_np, color=ground_truth_color, label="Ground Truth")
    ax_drift.plot(values_np, drift_estimated_np, color=fim_color, label="FIM")
    if use_comparison:
        ax_drift.plot(values_np, comparison_model_drift_np, color=comparison_color, label="Comparison Model")
    ax_drift.set_title("Drift")
    ax_drift.legend(loc="upper right")

    # Plot diffusion
    ax_diffusion.plot(values_np, diffusion_real_np, color=ground_truth_color, label="Ground Truth")
    ax_diffusion.plot(values_np, diffusion_estimated_np, color=fim_color, label="FIM")
    if use_comparison:
        ax_diffusion.plot(values_np, comparison_model_diffusion_np, color=comparison_color, label="Comparison Model")
    ax_diffusion.set_title("Diffusion")
    ax_diffusion.legend(loc="upper right")

    # Add insets if zoom areas are specified
    if zoom_area_drift:
        inset_ax_drift = inset_axes(ax_drift, width="30%", height="30%", loc=zoom_area_drift.get("loc", "lower left"), borderpad=2)
        ax_drift_zoom = inset_ax_drift
        ax_drift_zoom.plot(values_np, drift_real_np, color=ground_truth_color)
        ax_drift_zoom.plot(values_np, drift_estimated_np, color=fim_color)
        if use_comparison:
            ax_drift_zoom.plot(values_np, comparison_model_drift_np, color=comparison_color)
        ax_drift_zoom.set_xlim(zoom_area_drift["xlim"])
        ax_drift_zoom.set_ylim(zoom_area_drift["ylim"])
        ax_drift_zoom.set_xticks([])
        ax_drift_zoom.set_yticks([])
        mark_inset(
            ax_drift,
            inset_ax_drift,
            loc1=zoom_area_drift.get("mark_loc1", 2),
            loc2=zoom_area_drift.get("mark_loc2", 4),
            fc="none",
            ec="0.5",
        )

    if zoom_area_diffusion:
        inset_ax_diffusion = inset_axes(
            ax_diffusion, width="30%", height="30%", loc=zoom_area_diffusion.get("loc", "lower left"), borderpad=2
        )
        ax_diffusion_zoom = inset_ax_diffusion
        ax_diffusion_zoom.plot(values_np, diffusion_real_np, color=ground_truth_color)
        ax_diffusion_zoom.plot(values_np, diffusion_estimated_np, color=fim_color)
        if use_comparison:
            ax_diffusion_zoom.plot(values_np, comparison_model_diffusion_np, color=comparison_color)
        ax_diffusion_zoom.set_xlim(zoom_area_diffusion["xlim"])
        ax_diffusion_zoom.set_ylim(zoom_area_diffusion["ylim"])
        ax_diffusion_zoom.set_xticks([])
        ax_diffusion_zoom.set_yticks([])
        mark_inset(
            ax_diffusion,
            inset_ax_diffusion,
            loc1=zoom_area_diffusion.get("mark_loc1", 2),
            loc2=zoom_area_diffusion.get("mark_loc2", 4),
            fc="none",
            ec="0.5",
        )


fig = create_1D_plot(
    model_values,
    ground_truth_drift,
    ground_truth_diffusion,
    model_drift,
    model_diffusion,
    comparison_model_drift=comparison_model_drift,
    comparison_model_diffusion=comparison_model_diffusion,
)

plt.savefig("1D_plot.png")
