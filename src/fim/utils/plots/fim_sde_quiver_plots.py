

from pathlib import Path
import pickle
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np
from torch import Tensor
import torch


from fim.data.utils import load_h5s_in_folder
from fim.utils.plots.sde_data_exploration_plots import plot_paths_in_axis
from fim.utils.plots.sde_estimation_plots import plot_2d_vf_real_and_estimation_axes


def create_2D_quiver_plot(
    locations: Tensor,  # [B, G, D]
    ground_truth_drift: Tensor,  # [B, G, D]
    ground_truth_diffusion: Tensor,  # [B, G, D]
    estimated_drift: Tensor,  # [B, G, D]
    estimated_diffusion: Tensor,  # [B, G, D]
    comparison_model_drift: Tensor = None,  # [B, G, D]
    comparison_model_diffusion: Tensor = None,  # [B, G, D]
    zoom_area_drift=dict(xlim=(2, 2.8), ylim=(-1.1, -0.9)),
    zoom_area_diffusion=dict(xlim=(1.2, 2.2), ylim=(-2, -1)),
    zoom_position_drift='lower left',
    zoom_position_diffusion='lower left',
    inset_scale_drift=0.2,
    inset_scale_diffusion=0.5,
    title=None,
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
            zoom_area_drift=zoom_area_drift,
            zoom_area_diffusion=zoom_area_diffusion,
            zoom_position_drift=zoom_position_drift,
            zoom_position_diffusion=zoom_position_diffusion,
            inset_scale_drift=inset_scale_drift,
            inset_scale_diffusion=inset_scale_diffusion,
            **kwargs,
        )
        
    # Set title
    if title is not None:
        fig.suptitle(title)
        
    # Save figure
    plt.savefig(f"{title}.png")

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
    zoom_area_drift=dict(xlim=(2, 2.8), ylim=(-1.1, -0.9)),
    zoom_area_diffusion=dict(xlim=(1.2, 2.2), ylim=(-2, -1)),
    zoom_position_drift='lower left',
    zoom_position_diffusion='lower left',
    inset_scale_drift=0.2,
    inset_scale_diffusion=0.5,
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
    axis_drift.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 0.9))
    axis_diffusion.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 0.9))
    
    # Define arrow width for insets
    inset_arrow_width = kwargs.get("inset_arrow_width", 0.03)
    
    # Add zoom inset to drift plot
    axins_drift = inset_axes(axis_drift, width="30%", height="30%", loc=zoom_position_drift, borderpad=3)
    real_drift_scale = real_drift_quiver.scale if real_drift_quiver.scale is not None else 1
    axins_drift.quiver(
        x,
        y,
        u_real_drift * inset_scale_drift,
        v_real_drift * inset_scale_drift,
        color=ground_truth_color,
        scale=real_drift_scale / inset_scale_drift,
        width=inset_arrow_width
    )
    axins_drift.quiver(
        x,
        y,
        u_estimated_drift * inset_scale_drift,
        v_estimated_drift * inset_scale_drift,
        scale=real_drift_scale / inset_scale_drift,
        color=fim_color,
        width=inset_arrow_width
    )
    if use_comparison_model:
        axins_drift.quiver(
            x,
            y,
            u_comparison_model_drift * inset_scale_drift,
            v_comparison_model_drift * inset_scale_drift,
            scale=real_drift_scale / inset_scale_drift,
            color=comparison_color,
            width=inset_arrow_width
        )
    axins_drift.set_xlim(zoom_area_drift['xlim'])
    axins_drift.set_ylim(zoom_area_drift['ylim'])
    axins_drift.set_xticks([])
    axins_drift.set_yticks([])
    
    mark_inset(axis_drift, axins_drift, loc1=2, loc2=4, fc="none", ec="0.5")
    
    # Add zoom inset to diffusion plot
    axins_diffusion = inset_axes(axis_diffusion, width="30%", height="30%", loc=zoom_position_diffusion, borderpad=3)
    real_diffusion_scale = real_diffusion_quiver.scale if real_diffusion_quiver.scale is not None else 1
    axins_diffusion.quiver(
        x,
        y,
        u_real_diffusion * inset_scale_diffusion,
        v_real_diffusion * inset_scale_diffusion,
        color=ground_truth_color,
        scale=real_diffusion_scale / inset_scale_diffusion,
        width=inset_arrow_width
    )
    axins_diffusion.quiver(
        x,
        y,
        u_estimated_diffusion * inset_scale_diffusion,
        v_estimated_diffusion * inset_scale_diffusion,
        scale=real_diffusion_scale / inset_scale_diffusion,
        color=fim_color,
        width=inset_arrow_width
    )
    if use_comparison_model:
        axins_diffusion.quiver(
            x,
            y,
            u_comparison_model_diffusion * inset_scale_diffusion,
            v_comparison_model_diffusion * inset_scale_diffusion,
            scale=real_diffusion_scale / inset_scale_diffusion,
            color=comparison_color,
            width=inset_arrow_width
        )
    axins_diffusion.set_xlim(zoom_area_diffusion['xlim'])
    axins_diffusion.set_ylim(zoom_area_diffusion['ylim'])
    axins_diffusion.set_xticks([])
    axins_diffusion.set_yticks([])
    
    mark_inset(axis_diffusion, axins_diffusion, loc1=2, loc2=4, fc="none", ec="0.5")