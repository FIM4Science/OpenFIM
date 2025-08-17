import json
from copy import copy
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import optree

from fim import project_path
from fim.utils.sde.evaluation import save_fig


def _load_from_json(path_to_json: Path, system_name: str, tau: float, noise: float, exp: int, apply_sqrt_to_diffusion: bool):
    all_results: list[dict] = json.load(open(path_to_json))

    results_with_system = [result for result in all_results if result["name"] == system_name]
    results_with_tau = [result for result in results_with_system if result["tau"] == tau]
    results_with_noise = [result for result in results_with_tau if result["noise"] == noise]

    assert len(results_with_noise) == 1, f"Got {len(results_with_noise)}."

    all_exp_results = results_with_noise[0]
    all_exp_results.pop("name", None)
    all_exp_results.pop("tau", None)
    all_exp_results.pop("noise", None)
    all_exp_results.pop("equations", None)  # BISDE results has extra key

    all_exp_results = {k: np.array(v) for k, v in all_exp_results.items()}

    all_exp_results = optree.tree_map(lambda x: x[exp], all_exp_results)

    if apply_sqrt_to_diffusion is True:
        all_exp_results["diffusion_at_locations"] = np.sqrt(np.clip(all_exp_results["diffusion_at_locations"], a_min=0, a_max=np.inf))

    return all_exp_results


def _plot_1D_vfs(axs, locations, drift, diffusion, color, linewidth, linestyle, label):
    axs[0].plot(locations.squeeze(), drift.squeeze(), color=color, linewidth=linewidth, linestyle=linestyle, label=label)
    axs[1].plot(locations.squeeze(), diffusion.squeeze(), color=color, linewidth=linewidth, linestyle=linestyle)


def _plot_2D_vf(ax, locations, vf, color, linestyle, scale=None, label=None):
    return ax.quiver(
        locations.squeeze()[:, 0],
        locations.squeeze()[:, 1],
        vf.squeeze()[:, 0],
        vf.squeeze()[:, 1],
        color=color,
        scale=scale,
        linestyle=linestyle,
        linewidth=linewidth_vf,
        width=0.011,
        headwidth=2.5,
        label=label,
    )


def _subsample_2D_locs(locations, size_dim, subsample_factor):
    locations = locations.reshape(size_dim, size_dim, 2)
    locations = locations[::subsample_factor, ::subsample_factor]
    locations = locations.reshape(1, -1, 2)

    return locations


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    global_description = "synthetic_systems_figure_double_well_and_2D_synth_vfs"

    # current_description = "neurips_search_for_decent_results_tau_0_002_noise_0_exp_4"
    # current_description = "found_tau_0_002_noise_0_results_for_all_models"
    current_description = "post_neurips_color_update_and_zero_shot_in_brackets"

    # data and results to load
    path_to_vector_fields_data_json = Path(
        "/cephfs_projects/foundation_models/data/SDE/external_evaluations_and_data/20250325_synthetic_systems_5000_points_with_additive_noise/data/systems_ground_truth_drift_diffusion.json"
    )
    path_to_ksig_paths_json = Path(
        "/cephfs_projects/foundation_models/data/SDE/external_evaluations_and_data/20250325_synthetic_systems_5000_points_with_additive_noise/data/systems_ksig_reference_paths.json"
    )
    path_to_gp_json = Path(
        "/cephfs_projects/foundation_models/data/SDE/external_evaluations_and_data/20250505_sparse_gp_model_results_with_noise/20250505_sparse_gp_experiments_mai.json"
    )
    path_to_bisde_json = Path(
        "/cephfs_projects/foundation_models/data/SDE/external_evaluations_and_data/20250510_bisde_results_with_noise/20250510_bisde_results_with_multiple_diffusion_summands_no_diffusion_clipping/20250510_bisde_results_no_diffusion_clipping.json"
    )
    path_to_fimsde_json = Path(
        "/cephfs_projects/foundation_models/data/SDE/saved_evaluation_results/20250329_neurips_submission_evaluations/synthetic_systems_vf_and_paths/05141437_fim_fixed_softmax_dim_05-03-2033_epoch_139/model_paths.json"
    )

    # results to plot
    systems_to_plot = ["Wang", "Double Well"]

    select_gp_results = {
        "Double Well": {"tau": 0.002, "exp": 0, "noise": 0},
        "Wang": {"tau": 0.002, "exp": 2, "noise": 0},
    }
    select_bisde_results = {
        "Double Well": {"tau": 0.002, "exp": 1, "noise": 0},
        "Wang": {"tau": 0.002, "exp": 0, "noise": 0},
    }
    select_fimsde_results = {
        "Double Well": {"tau": 0.02, "exp": 4, "noise": 0},
        "Wang": {"tau": 0.02, "exp": 1, "noise": 0},
    }
    # --------------------------------------------------------------------------------------------------------------------------------- #

    # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
    evaluation_path = Path(project_path) / "evaluations"
    time: str = str(datetime.now().strftime("%m%d%H%M"))
    evaluation_dir: Path = evaluation_path / global_description / (time + "_" + current_description)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # load data of double well and wang 2D synth
    all_data: list[dict] = json.load(open(path_to_vector_fields_data_json))
    all_systems_data: dict = {all_data[i].get("name"): copy(all_data[i]) for i in range(len(all_data))}
    data = {k: v for k, v in all_systems_data.items() if k in systems_to_plot}
    data = optree.tree_map(lambda x: np.array(x), data, is_leaf=lambda x: isinstance(x, list))
    # data is the same in all experiments
    # keys: obs_times, obs_values, locations, drift_at_locations, diffusion_at_locations per system

    gp_results = {
        "Wang": _load_from_json(path_to_gp_json, "Wang", **select_gp_results["Wang"], apply_sqrt_to_diffusion=True),
        "Double Well": _load_from_json(path_to_gp_json, "Double Well", **select_gp_results["Double Well"], apply_sqrt_to_diffusion=True),
    }
    bisde_results = {
        "Wang": _load_from_json(path_to_bisde_json, "Wang", **select_bisde_results["Wang"], apply_sqrt_to_diffusion=True),
        "Double Well": _load_from_json(
            path_to_bisde_json, "Double Well", **select_bisde_results["Double Well"], apply_sqrt_to_diffusion=True
        ),
    }
    fimsde_results = {
        "Wang": _load_from_json(path_to_fimsde_json, "Wang", **select_fimsde_results["Wang"], apply_sqrt_to_diffusion=False),
        "Double Well": _load_from_json(
            path_to_fimsde_json, "Double Well", **select_fimsde_results["Double Well"], apply_sqrt_to_diffusion=False
        ),
    }

    # create figure with separating gap between the two systems
    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(7, 1.5), gridspec_kw={"width_ratios": [1, 1, 0.02, 1, 1]}, dpi=300)
    axs[2].axis("off")

    # configure axes general
    for ax in axs:
        [x.set_linewidth(0.3) for x in ax.spines.values()]
        ax.tick_params(axis="both", direction="out", labelsize=5, width=0.5, length=2, pad=0.8)

    # configure double well
    axs[0].xaxis.set_major_locator(plticker.MultipleLocator(base=1))
    axs[1].xaxis.set_major_locator(plticker.MultipleLocator(base=1))

    axs[0].yaxis.set_major_locator(plticker.MultipleLocator(base=20))  # drift
    axs[1].yaxis.set_major_locator(plticker.MultipleLocator(base=1))  # diffusion

    # configure wang 2D synthetic
    axs[3].xaxis.set_major_locator(plticker.MultipleLocator(base=2))
    axs[4].xaxis.set_major_locator(plticker.MultipleLocator(base=2))

    axs[3].yaxis.set_major_locator(plticker.MultipleLocator(base=2))  # drift
    axs[4].yaxis.set_major_locator(plticker.MultipleLocator(base=2))  # diffusion

    # write vector field as title
    axs[0].set_title("Drift", fontsize=5, pad=2)
    axs[1].set_title("Diffusion", fontsize=5, pad=2)
    axs[3].set_title("Drift", fontsize=5, pad=2)
    axs[4].set_title("Diffusion", fontsize=5, pad=2)

    # general plot config
    linewidth_vf = 1
    loc_size_per_dim = 32
    loc_subsample_factor = 4

    gt_color = "black"
    gp_color = "#D55E00"
    bisde_color = "#CC79A7"
    fimsde_color = "#0072B2"

    gt_linestyle = "dotted"
    gp_linestyle = "dashed"
    bisde_linestyle = "dashdot"
    fimsde_linestyle = "solid"

    gt_label = "Ground-Truth"
    gp_label = "SparseGP"
    bisde_label = "BISDE"
    fimsde_label = "FIM-SDE (Zero-Shot)"

    # plot ground-truth
    dw_data = data["Double Well"]
    dw_data["locations"] = dw_data["locations"][0]
    dw_data["drift_at_locations"] = dw_data["drift_at_locations"][0]
    dw_data["diffusion_at_locations"] = dw_data["diffusion_at_locations"][0]
    two_d_data = data["Wang"]
    two_d_data["locations"] = two_d_data["locations"][0]
    two_d_data["drift_at_locations"] = two_d_data["drift_at_locations"][0]
    two_d_data["diffusion_at_locations"] = two_d_data["diffusion_at_locations"][0]

    dw_locs = dw_data["locations"]
    dw_drift = dw_data["drift_at_locations"]
    dw_diff = dw_data["diffusion_at_locations"]
    _plot_1D_vfs(axs, dw_locs, dw_drift, dw_diff, color=gt_color, linewidth=linewidth_vf, linestyle=gt_linestyle, label=gt_label)

    two_d_locs = _subsample_2D_locs(two_d_data["locations"], size_dim=loc_size_per_dim, subsample_factor=loc_subsample_factor)
    two_d_drift = _subsample_2D_locs(two_d_data["drift_at_locations"], size_dim=loc_size_per_dim, subsample_factor=loc_subsample_factor)
    two_d_diff = _subsample_2D_locs(two_d_data["diffusion_at_locations"], size_dim=loc_size_per_dim, subsample_factor=loc_subsample_factor)

    drift_quiver = _plot_2D_vf(axs[3], two_d_locs, two_d_drift, color=gt_color, linestyle=gt_linestyle)
    diff_quiver = _plot_2D_vf(axs[4], two_d_locs, two_d_diff, color=gt_color, linestyle=gt_linestyle)

    # plot GP results
    dw_gp_results = gp_results["Double Well"]
    dw_gp_drift = dw_gp_results["drift_at_locations"]
    dw_gp_diff = dw_gp_results["diffusion_at_locations"]
    _plot_1D_vfs(axs, dw_locs, dw_gp_drift, dw_gp_diff, color=gp_color, linewidth=linewidth_vf, linestyle=gp_linestyle, label=gp_label)

    two_d_gp_results = gp_results["Wang"]
    two_d_gp_drift = _subsample_2D_locs(
        two_d_gp_results["drift_at_locations"], size_dim=loc_size_per_dim, subsample_factor=loc_subsample_factor
    )
    two_d_gp_diff = _subsample_2D_locs(
        two_d_gp_results["diffusion_at_locations"], size_dim=loc_size_per_dim, subsample_factor=loc_subsample_factor
    )
    _plot_2D_vf(axs[3], two_d_locs, two_d_gp_drift, color=gp_color, linestyle=gp_linestyle, scale=drift_quiver.scale)
    _plot_2D_vf(axs[4], two_d_locs, two_d_gp_diff, color=gp_color, linestyle=gp_linestyle, scale=diff_quiver.scale)

    # plot bisde results
    dw_bisde_results = bisde_results["Double Well"]
    dw_bisde_drift = dw_bisde_results["drift_at_locations"]
    dw_bisde_diff = dw_bisde_results["diffusion_at_locations"]
    _plot_1D_vfs(
        axs, dw_locs, dw_bisde_drift, dw_bisde_diff, color=bisde_color, linewidth=linewidth_vf, linestyle=bisde_linestyle, label=bisde_label
    )

    two_d_bisde_results = bisde_results["Wang"]
    two_d_bisde_drift = _subsample_2D_locs(
        two_d_bisde_results["drift_at_locations"], size_dim=loc_size_per_dim, subsample_factor=loc_subsample_factor
    )
    two_d_bisde_diff = _subsample_2D_locs(
        two_d_bisde_results["diffusion_at_locations"], size_dim=loc_size_per_dim, subsample_factor=loc_subsample_factor
    )
    _plot_2D_vf(
        axs[3], two_d_locs, two_d_bisde_drift, color=bisde_color, linestyle=bisde_linestyle, scale=drift_quiver.scale, label=bisde_label
    )
    _plot_2D_vf(axs[4], two_d_locs, two_d_bisde_diff, color=bisde_color, linestyle=bisde_linestyle, scale=diff_quiver.scale)

    # plot our results
    dw_fimsde_results = fimsde_results["Double Well"]
    dw_fimsde_drift = dw_fimsde_results["drift_at_locations"]
    dw_fimsde_diff = dw_fimsde_results["diffusion_at_locations"]
    _plot_1D_vfs(
        axs,
        dw_locs,
        dw_fimsde_drift,
        dw_fimsde_diff,
        color=fimsde_color,
        linewidth=linewidth_vf,
        linestyle=fimsde_linestyle,
        label=fimsde_label,
    )

    two_d_fimsde_results = fimsde_results["Wang"]
    two_d_fimsde_drift = _subsample_2D_locs(
        two_d_fimsde_results["drift_at_locations"], size_dim=loc_size_per_dim, subsample_factor=loc_subsample_factor
    )
    two_d_fimsde_diff = _subsample_2D_locs(
        two_d_fimsde_results["diffusion_at_locations"], size_dim=loc_size_per_dim, subsample_factor=loc_subsample_factor
    )
    _plot_2D_vf(axs[3], two_d_locs, two_d_fimsde_drift, color=fimsde_color, linestyle=fimsde_linestyle, scale=drift_quiver.scale)
    _plot_2D_vf(axs[4], two_d_locs, two_d_fimsde_diff, color=fimsde_color, linestyle=fimsde_linestyle, scale=diff_quiver.scale)

    # place right legend directly on top of the plot
    plt.draw()
    handles, labels = axs[0].get_legend_handles_labels()

    legend_fontsize = 5
    bbox_x = axs[2].get_position().x0 + 0.5 * (axs[2].get_position().x1 - axs[2].get_position().x0)
    bbox_y = axs[2].get_position().y1 * 1.07

    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=[bbox_x, bbox_y],
        fontsize=legend_fontsize,
        ncols=4,
    )

    # save
    save_dir: Path = evaluation_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    file_name = "vector_field_plot"
    save_fig(fig, save_dir, file_name)
