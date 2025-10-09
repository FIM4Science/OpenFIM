import json
from copy import copy, deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import optree

from fim import project_path
from fim.utils.sde.evaluation import save_fig


def load_system_results(all_results: list[dict], system_name: str, tau: float, exp: int, noise: float, apply_sqrt_to_diffusion: bool):
    """
    Extract results from one model on one system, defined by (system_name, tau, exp), from a list[dict] with all results from that model.

    Args:
        all_results (list[dict]): Loaded from some model_results.json.
        (system_name, tau, exp, noise): defines the results to extract.
        apply_sqrt_to_diffusion (bool): Some models (SparseGP, BISDE) return diffusion value under sqrt. Adapt it here for comparison.

    Return:
        all_exp_results (dict[np.ndarray]): Keys: locations, drift_at_locations, diffusion_at_locations, synthetic_paths.
    """
    print(system_name, tau, exp, noise)
    results_with_system = [result for result in all_results if result["name"] == system_name]
    results_with_tau = [result for result in results_with_system if result["tau"] == tau]
    results_with_noise = [result for result in results_with_tau if result["noise"] == noise]
    assert len(results_with_noise) == 1, f"Failed with {system_name=}, {tau=}, {exp=}, {noise=}. Got {len(results_with_noise)}"

    all_exp_results = results_with_noise[0]
    all_exp_results.pop("name", None)
    all_exp_results.pop("tau", None)
    all_exp_results.pop("noise", None)
    all_exp_results.pop("equations", None)

    all_exp_results = {k: np.array(v) for k, v in all_exp_results.items()}
    all_exp_results = optree.tree_map(lambda x: x[exp], all_exp_results)

    if apply_sqrt_to_diffusion is True:
        all_exp_results["diffusion_at_locations"] = np.sqrt(np.clip(all_exp_results["diffusion_at_locations"], a_min=0, a_max=np.inf))

    return all_exp_results


def subsample_2D_grid(grid: np.ndarray, size_dim: int, stride_length: int):
    """
    Regular subsampling of a 2D grid.

    Args:
        grid (np.ndarray): Flattened 2D, regular grid. Shape: [size_dim * size_dim, 2]
        stride_length (int): Subsampling lenght along each dimension.

    Returns:
        subsampled_grid (np.ndarray): Shape: [(size_dim / stride_length) * (size_dim / stride_length), 2]
    """
    grid = grid.reshape(size_dim, size_dim, 2)
    grid = grid[::stride_length, ::stride_length]
    grid = grid.reshape(1, -1, 2)

    return grid


def plot_2D_vf(
    ax, locations: np.ndarray, vf: np.ndarray, color: str, linestyle: str, scale: Optional[float] = None, label: Optional[str] = None
):
    """
    Plot 2D vector field value at some locations as quivers.

    Args:
        ax: Axis to plot quivers into.
        locations + vf (np.ndarray): Define vector field to visualize.
        scale (Optional[float]): Optional scaling of quivers by other vector field scale.
        plot configs: ...

    Returns:
        quiver object for reference
    """
    return ax.quiver(
        locations.squeeze()[:, 0],
        locations.squeeze()[:, 1],
        vf.squeeze()[:, 0],
        vf.squeeze()[:, 1],
        color=color,
        scale=scale,
        linestyle=linestyle,
        linewidth=1,
        width=0.011,
        headwidth=2.5,
        label=label,
    )


def plot_1D_paths(ax, times: np.ndarray, values: np.ndarray, color: str, linestyle: str, linewidth: float, label: str):
    """
    Plot multiple paths into an axis.

    Args:
        times, values: Paths to plot. Shape: [T, 1], [P, T, 1]
        plot configs: ...
    """
    P = values.shape[0]
    for path in range(P):
        ax.plot(
            times.reshape(-1),
            values[path].reshape(-1),
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label if path == 0 else None,
        )


def plot_2D_paths(ax, values: np.ndarray, color: str, linestyle: str, linewidth: float, label: str):
    """
    Plot multiple paths into an axis.

    Args:
        values: Paths to plot. Shape: [P, T, 2]
        plot configs: ...
    """
    P = values.shape[0]
    for path in range(P):
        ax.plot(
            values[path, :, 0].reshape(-1),
            values[path, :, 1].reshape(-1),
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label if path == 0 else None,
        )


def plot_row(
    axs,
    locations: np.ndarray,
    drift: np.ndarray,
    diffusion: np.ndarray,
    times: np.ndarray,
    paths: np.ndarray,
    color: str,
    linestyle: str,
    label: str,
    linewidth_vf: float,
    linewidth_paths: float,
    drift_quiver_scale: Optional[float] = None,
    diff_quiver_scale: Optional[float] = None,
):
    """
    Plot results of one model (or ground-truth data) of one system into one row of the grid.

    Args:
        axs: len(axs) == 3, drift, diffusion, paths
        locations, drift, diffusion, times, paths: data to plot
        plot configs: ...
        ...scale: Quiver scales from other models / data for consistent scaling.
    """
    D = locations.shape[-1]

    if D == 1:
        axs[0].plot(locations.squeeze(), drift.squeeze(), color=color, linewidth=linewidth_vf, linestyle=linestyle, label=label)
        axs[1].plot(locations.squeeze(), diffusion.squeeze(), color=color, linewidth=linewidth_vf, linestyle=linestyle, label=label)

        if label in ["Ground-Truth", "FIM-SDE (Zero-Shot)"]:
            plot_1D_paths(
                axs[2],
                times,
                paths,
                color=color,
                linewidth=linewidth_paths,
                linestyle="solid",  # linestyle,
                label=label,
            )

        return None, None

    else:
        drift_quiver = plot_2D_vf(axs[0], locations, drift, color=color, linestyle=linestyle, scale=drift_quiver_scale, label=label)
        diff_quiver = plot_2D_vf(axs[1], locations, diffusion, color=color, linestyle=linestyle, scale=diff_quiver_scale, label=label)

        if label in ["Ground-Truth", "FIM-SDE (Zero-Shot)"]:
            plot_2D_paths(
                axs[2],
                paths,
                color=color,
                linewidth=linewidth_paths,
                linestyle="solid",  # linestyle,
                label=label,
            )

    return drift_quiver.scale, diff_quiver.scale


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    global_description = "synthetic_systems_figure_all_vfs_and_paths"
    current_description = "post_neurips_rebuttal_selected_exps_candidate_tighter_grid"

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
    systems_to_plot = ["Double Well", "Wang", "Damped Linear", "Damped Cubic", "Duffing", "Glycosis", "Hopf"]
    experiment_to_plot = 0  # we repeat each experiment 5 times, but reference paths stay the same, so choose first copy

    # choose visual best experiment / tau per model
    select_gp_results = {
        "Double Well": {"tau": 0.002, "exp": 0, "noise": 0},  # 0 or 1
        "Wang": {"tau": 0.002, "exp": 2, "noise": 0},  # they all look bad, take 2
        "Damped Linear": {"tau": 0.002, "exp": 0, "noise": 0},  # 3 or 0
        "Damped Cubic": {"tau": 0.002, "exp": 0, "noise": 0},  # they all look good
        "Duffing": {"tau": 0.002, "exp": 4, "noise": 0},  # 4 or 0
        "Glycosis": {"tau": 0.002, "exp": 0, "noise": 0},  # 0
        "Hopf": {"tau": 0.002, "exp": 2, "noise": 0},  # they all look bad, take 2
    }
    select_bisde_results = {
        "Double Well": {"tau": 0.002, "exp": 1, "noise": 0},  # 1 very good, 0 very bad
        "Wang": {"tau": 0.002, "exp": 0, "noise": 0},  # 1 or 0
        "Damped Linear": {"tau": 0.002, "exp": 0, "noise": 0},  # all good, take 0
        "Damped Cubic": {"tau": 0.002, "exp": 2, "noise": 0},  # all good, take 2
        "Duffing": {"tau": 0.002, "exp": 2, "noise": 0},  # 2
        "Glycosis": {"tau": 0.002, "exp": 0, "noise": 0},  # 0 or 3
        "Hopf": {"tau": 0.002, "exp": 4, "noise": 0},  # 4 or 1
    }
    select_fimsde_results = {
        "Double Well": {"tau": 0.02, "exp": 4, "noise": 0},
        "Wang": {"tau": 0.02, "exp": 1, "noise": 0},
        "Damped Linear": {"tau": 0.002, "exp": 1, "noise": 0},
        "Damped Cubic": {"tau": 0.02, "exp": 1, "noise": 0},
        "Duffing": {"tau": 0.02, "exp": 0, "noise": 0},
        "Glycosis": {"tau": 0.02, "exp": 0, "noise": 0},
        "Hopf": {"tau": 0.002, "exp": 2, "noise": 0},  # 1 or 2
    }

    # tau = 0.02
    # exp = 4
    # noise = 0.0
    #
    # select_gp_results = {
    #     "Double Well": {"tau": tau, "exp": exp, "noise": noise},
    #     "Wang": {"tau": tau, "exp": exp, "noise": noise},
    #     "Damped Linear": {"tau": tau, "exp": exp, "noise": noise},
    #     "Damped Cubic": {"tau": tau, "exp": exp, "noise": noise},
    #     "Duffing": {"tau": tau, "exp": exp, "noise": noise},
    #     "Glycosis": {"tau": tau, "exp": exp, "noise": noise},
    #     "Hopf": {"tau": tau, "exp": exp, "noise": noise},
    # }
    # select_bisde_results = {
    #     "Double Well": {"tau": tau, "exp": exp, "noise": noise},
    #     "Wang": {"tau": tau, "exp": exp, "noise": noise},
    #     "Damped Linear": {"tau": tau, "exp": exp, "noise": noise},
    #     "Damped Cubic": {"tau": tau, "exp": exp, "noise": noise},
    #     "Duffing": {"tau": tau, "exp": exp, "noise": noise},
    #     "Glycosis": {"tau": tau, "exp": exp, "noise": noise},
    #     "Hopf": {"tau": tau, "exp": exp, "noise": noise},
    # }
    # select_fimsde_results = {
    #     "Double Well": {"tau": tau, "exp": exp, "noise": noise},
    #     "Wang": {"tau": tau, "exp": exp, "noise": noise},
    #     "Damped Linear": {"tau": tau, "exp": exp, "noise": noise},
    #     "Damped Cubic": {"tau": tau, "exp": exp, "noise": noise},
    #     "Duffing": {"tau": tau, "exp": exp, "noise": noise},
    #     "Glycosis": {"tau": tau, "exp": exp, "noise": noise},
    #     "Hopf": {"tau": tau, "exp": exp, "noise": noise},
    # }

    # paths setup
    plot_num_paths = 10

    # general plot config
    linewidth_vf = 1
    # linewidth_paths = 0.2
    loc_size_per_dim = 32
    loc_stride_length = 4

    gt_plot_config = {
        "color": "black",
        "linestyle": "dotted",
        # "linestyle": "solid",
        "label": "Ground-Truth",
        "linewidth_paths": 0.15,
    }

    gp_plot_config = {
        "color": "#D55E00",
        "linestyle": "dashed",
        # "linestyle": "solid",
        "label": "SparseGP",
        "linewidth_paths": 0.25,
    }

    bisde_plot_config = {
        "color": "#CC79A7",
        "linestyle": "dashdot",
        # "linestyle": "solid",
        "label": "BISDE",
        "linewidth_paths": 0.25,
    }

    fimsde_plot_config = {
        "color": "#0072B2",
        "linestyle": "solid",
        "label": "FIM-SDE (Zero-Shot)",
        "linewidth_paths": 0.25,
    }

    # select the models to show
    models_to_plot = [gp_plot_config, bisde_plot_config, fimsde_plot_config]

    tick_base_per_system = {
        "Double Well": {
            "drift": {"x": 1, "y": 20},
            "diffusion": {"x": 1, "y": 1},
            "paths": {"x": 1, "y": 1},
        },
        "Wang": {
            "drift": {"x": 2, "y": 2},
            "diffusion": {"x": 2, "y": 2},
            "paths": {"x": 2, "y": 2},
        },
        "Damped Linear": {
            "drift": {"x": 1, "y": 1},
            "diffusion": {"x": 1, "y": 1},
            "paths": {"x": 2, "y": 2},
        },
        "Damped Cubic": {
            "drift": {"x": 1, "y": 1},
            "diffusion": {"x": 1, "y": 1},
            "paths": {"x": 1, "y": 1},
        },
        "Duffing": {
            "drift": {"x": 2, "y": 2},
            "diffusion": {"x": 2, "y": 2},
            "paths": {"x": 2, "y": 2},
        },
        "Glycosis": {
            "drift": {"x": 1, "y": 1},
            "diffusion": {"x": 1, "y": 1},
            "paths": {"x": 1, "y": 1},
        },
        "Hopf": {
            "drift": {"x": 1, "y": 1},
            "diffusion": {"x": 1, "y": 1},
            "paths": {"x": 1, "y": 1},
        },
    }
    # --------------------------------------------------------------------------------------------------------------------------------- #

    # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
    evaluation_path = Path(project_path) / "evaluations"
    time: str = str(datetime.now().strftime("%m%d%H%M"))
    evaluation_dir: Path = evaluation_path / global_description / (time + "_" + current_description)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # load ground-truth locations, drift, diffusion
    all_data: list[dict] = json.load(open(path_to_vector_fields_data_json))
    all_systems_data: dict = {all_data[i].get("name"): copy(all_data[i]) for i in range(len(all_data))}
    all_systems_data = {k: v for k, v in all_systems_data.items() if k in systems_to_plot}
    all_systems_data = optree.tree_map(lambda x: np.array(x), all_systems_data, is_leaf=lambda x: isinstance(x, list))
    # all_systems_data is the same in all experiments
    # keys: obs_times, obs_values, locations, drift_at_locations, diffusion_at_locations per system

    # load ground-truth paths
    all_paths: list[dict] = json.load(open(path_to_ksig_paths_json))
    all_systems_paths: dict = {all_paths[i].get("name"): copy(all_paths[i]) for i in range(len(all_paths))}
    all_systems_paths = {k: v for k, v in all_systems_paths.items() if k in systems_to_plot}
    all_systems_paths = optree.tree_map(lambda x: np.array(x), all_systems_paths, is_leaf=lambda x: isinstance(x, list))

    # # load results from all models and select systems to plot
    all_gp_results: list[dict] = json.load(open(path_to_gp_json))
    gp_results = {
        system: load_system_results(deepcopy(all_gp_results), system, **select_gp_results[system], apply_sqrt_to_diffusion=True)
        for system in systems_to_plot
    }
    gp_plot_config.update({"results": gp_results})

    bisde_results = {
        system: load_system_results(
            json.load(open(path_to_bisde_json)), system, **select_bisde_results[system], apply_sqrt_to_diffusion=True
        )
        for system in systems_to_plot
    }
    bisde_plot_config.update({"results": bisde_results})

    all_fimsde_results: list[dict] = json.load(open(path_to_fimsde_json))
    fimsde_results = {
        system: load_system_results(deepcopy(all_fimsde_results), system, **select_fimsde_results[system], apply_sqrt_to_diffusion=False)
        for system in systems_to_plot
    }
    fimsde_plot_config.update({"results": fimsde_results})

    # create figure with 3 subplots per system: drift, diffusion, paths
    fig, axs = plt.subplots(
        nrows=len(systems_to_plot),
        ncols=3,
        figsize=(4.5, 1.5 * len(systems_to_plot)),
        gridspec_kw={"width_ratios": [1, 1, 1], "hspace": 0.17, "wspace": 0.3},
        dpi=300,
        # tight_layout=True,
    )

    # configure axes
    for ax in axs.reshape(-1):
        [x.set_linewidth(0.3) for x in ax.spines.values()]
        ax.tick_params(axis="both", direction="out", labelsize=4, width=0.5, length=2)

    for i, system_name in enumerate(systems_to_plot):
        axs[i, 0].xaxis.set_major_locator(plticker.MultipleLocator(base=tick_base_per_system[system_name]["drift"]["x"]))
        axs[i, 0].yaxis.set_major_locator(plticker.MultipleLocator(base=tick_base_per_system[system_name]["drift"]["y"]))

        axs[i, 1].xaxis.set_major_locator(plticker.MultipleLocator(base=tick_base_per_system[system_name]["diffusion"]["x"]))
        axs[i, 1].yaxis.set_major_locator(plticker.MultipleLocator(base=tick_base_per_system[system_name]["diffusion"]["y"]))

        axs[i, 2].xaxis.set_major_locator(plticker.MultipleLocator(base=tick_base_per_system[system_name]["paths"]["x"]))
        axs[i, 2].yaxis.set_major_locator(plticker.MultipleLocator(base=tick_base_per_system[system_name]["paths"]["y"]))

    # vector field as column title
    axs[0, 0].set_title("Drift", fontsize=6, pad=2)
    axs[0, 1].set_title("Diffusion", fontsize=6, pad=2)
    axs[0, 2].set_title("Paths", fontsize=6, pad=2)

    # 1D data (double well) requires time for plotting paths
    times = 0.002 * np.arange(all_systems_paths["Double Well"]["real_paths"][0].shape[1]).reshape(1, -1, 1)

    # plot ground-truth data and model results; 2D quiver scale determined by ground-truth
    for system_num, system in enumerate(systems_to_plot):
        print(system)

        if system == "Wang":
            axs[system_num, 0].set_ylabel("2D Synthetic", fontsize=6)
        else:
            axs[system_num, 0].set_ylabel(system, fontsize=6)

        # ground-truth
        system_data = all_systems_data[system]
        locations = system_data["locations"][experiment_to_plot]
        drift = system_data["drift_at_locations"][experiment_to_plot]
        diffusion = system_data["diffusion_at_locations"][experiment_to_plot]

        D = locations.shape[-1]
        if D == 2:  # for clarity: reduce number of quivers
            locations = subsample_2D_grid(locations, size_dim=loc_size_per_dim, stride_length=loc_stride_length)
            drift = subsample_2D_grid(drift, size_dim=loc_size_per_dim, stride_length=loc_stride_length)
            diffusion = subsample_2D_grid(diffusion, size_dim=loc_size_per_dim, stride_length=loc_stride_length)

        system_paths = all_systems_paths[system]
        paths = system_paths["real_paths"][experiment_to_plot]

        drift_quiver_scale, diff_quiver_scale = plot_row(
            axs[system_num],
            locations,
            drift,
            diffusion,
            times[:plot_num_paths],
            paths[:plot_num_paths],
            gt_plot_config["color"],
            gt_plot_config["linestyle"],
            gt_plot_config["label"],
            linewidth_vf,
            gt_plot_config["linewidth_paths"],
        )

        # models
        for model_plot_config in models_to_plot:
            drift = model_plot_config["results"][system]["drift_at_locations"]
            diffusion = model_plot_config["results"][system]["diffusion_at_locations"]

            D = locations.shape[-1]
            if D == 2:  # for clarity: reduce number of quivers
                drift = subsample_2D_grid(drift, size_dim=loc_size_per_dim, stride_length=loc_stride_length)
                diffusion = subsample_2D_grid(diffusion, size_dim=loc_size_per_dim, stride_length=loc_stride_length)

            plot_row(
                axs[system_num],
                locations,
                drift,
                diffusion,
                times[:plot_num_paths],
                model_plot_config["results"][system]["synthetic_paths"][:plot_num_paths],
                model_plot_config["color"],
                model_plot_config["linestyle"],
                model_plot_config["label"],
                linewidth_vf,
                model_plot_config["linewidth_paths"],
                drift_quiver_scale=drift_quiver_scale,
                diff_quiver_scale=diff_quiver_scale,
            )

    # place right legend on top of the plot
    plt.draw()
    handles, labels = axs[0, 0].get_legend_handles_labels()

    legend_fontsize = 6
    bbox_x = axs[0, 1].get_position().x0 + 0.5 * (axs[0, 1].get_position().x1 - axs[0, 1].get_position().x0)
    bbox_y = axs[0, 1].get_position().y1 * 1.01  # * 1.12

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
    file_name = "vector_fields_and_samples_plot"
    save_fig(fig, save_dir, file_name)
