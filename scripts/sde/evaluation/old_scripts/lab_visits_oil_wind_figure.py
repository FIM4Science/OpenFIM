import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np

from fim import project_path
from fim.utils.evaluation_sde import save_fig


def _load_from_json(path_to_json: Path, dataset_name: str, apply_sqrt_to_diffusion: bool):
    all_results: list[dict] = json.load(open(path_to_json))

    results_with_dataset = [result for result in all_results if result["name"] == dataset_name]
    results_with_dataset = results_with_dataset[0]

    results_with_dataset.pop("name")
    results_with_dataset = {k: np.array(v) for k, v in results_with_dataset.items()}

    if apply_sqrt_to_diffusion is True:
        results_with_dataset["diffusion_at_locations"] = np.sqrt(
            np.clip(results_with_dataset["diffusion_at_locations"], a_min=0, a_max=np.inf)
        )

    return results_with_dataset


def _load_location_data(path_to_json: Path, dataset_name: str):
    all_results: list[dict] = json.load(open(path_to_json))
    results_with_dataset = [result for result in all_results if result["name"] == dataset_name]

    locations = results_with_dataset[0]["locations"]
    locations = np.array(locations)
    return locations


def _load_gt_paths(path_to_json: Path, dataset_name: str):
    all_paths: list[dict] = json.load(open(path_to_json))
    paths_of_dataset = [result for result in all_paths if result["name"] == dataset_name]

    paths = paths_of_dataset[0]["real_paths"]
    paths = np.array(paths)
    return paths


def _plot_1D_vfs(axs, locations, drift, diffusion, color, linewidth, linestyle, label):
    axs[0].plot(locations.squeeze(), drift.squeeze(), color=color, linewidth=linewidth, linestyle=linestyle, label=label)
    axs[1].plot(locations.squeeze(), diffusion.squeeze(), color=color, linewidth=linewidth, linestyle=linestyle)


def _plot_1D_paths(ax, times, values, color, linestyle, linewidth, label, alpha=1):
    """
    times: [P, T, 1]
    values: [P, T, 1]
    """
    P = times.shape[0]
    for path in range(P):
        ax.plot(
            times[path].reshape(-1),
            values[path].reshape(-1),
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=label if path == 0 else None,
            alpha=alpha,
        )


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    global_description = "lab_visits_figure_oil_wind"

    current_description = "first_iteration_thin_path_lines"

    # data and results to load
    path_to_locations_json = Path(
        "/cephfs_projects/foundation_models/data/SDE/external_evaluations_and_data/20250130_bisde_real_world_oil_wind/data_for_inference.json"
    )
    path_to_ksig_paths_json = Path(
        "/cephfs_projects/foundation_models/data/SDE/external_evaluations_and_data/20250130_bisde_real_world_oil_wind/ksig_reference_paths.json"
    )
    path_to_bisde_json = Path("/home/seifner/repos/FIM/data/raw/SDE_bisde_on_bisde_oil_wind/bisde_vector_fields.json")
    path_to_fimsde_json = Path(
        "/cephfs_projects/foundation_models/data/SDE/external_evaluations_and_data/20250130_bisde_real_world_oil_wind/model_paths.json"
    )

    # --------------------------------------------------------------------------------------------------------------------------------- #

    # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
    evaluation_path = Path(project_path) / "evaluations"
    time: str = str(datetime.now().strftime("%m%d%H%M"))
    evaluation_dir: Path = evaluation_path / global_description / (time + "_" + current_description)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    location_data = {
        "oil": _load_location_data(path_to_locations_json, "oil"),
        "wind": _load_location_data(path_to_locations_json, "wind"),
    }
    gt_paths = {
        "oil": _load_gt_paths(path_to_ksig_paths_json, "oil"),
        "wind": _load_gt_paths(path_to_ksig_paths_json, "wind"),
    }
    bisde_results = {
        "oil": _load_from_json(path_to_bisde_json, "oil", apply_sqrt_to_diffusion=True),
        "wind": _load_from_json(path_to_bisde_json, "wind", apply_sqrt_to_diffusion=True),
    }
    fimsde_results = {
        "oil": _load_from_json(path_to_fimsde_json, "oil", apply_sqrt_to_diffusion=False),
        "wind": _load_from_json(path_to_fimsde_json, "wind", apply_sqrt_to_diffusion=False),
    }

    # create figure with separating gap between the two systems
    fig, axs = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(4.5, 3),
        gridspec_kw={"width_ratios": [1, 1, 1]},
        dpi=300,
        tight_layout=True,
    )

    # configure axes general
    for ax in axs.reshape(-1):
        [x.set_linewidth(0.3) for x in ax.spines.values()]
        ax.tick_params(axis="both", direction="out", labelsize=5, width=0.5, length=2)

    # configure oil
    axs[0, 0].xaxis.set_major_locator(plticker.MultipleLocator(base=10))  # drift
    axs[0, 1].xaxis.set_major_locator(plticker.MultipleLocator(base=10))  # diffusion
    axs[0, 2].xaxis.set_major_locator(plticker.MultipleLocator(base=1000))  # paths

    axs[0, 0].yaxis.set_major_locator(plticker.MultipleLocator(base=20))  # drift
    axs[0, 1].yaxis.set_major_locator(plticker.MultipleLocator(base=10))  # diffusion
    axs[0, 2].yaxis.set_major_locator(plticker.MultipleLocator(base=5))  # paths

    # configure wind
    axs[1, 0].xaxis.set_major_locator(plticker.MultipleLocator(base=5))  # drift
    axs[1, 1].xaxis.set_major_locator(plticker.MultipleLocator(base=5))  # diffusion
    axs[1, 2].xaxis.set_major_locator(plticker.MultipleLocator(base=2000))  # paths

    axs[1, 0].yaxis.set_major_locator(plticker.MultipleLocator(base=40))  # drift
    axs[1, 1].yaxis.set_major_locator(plticker.MultipleLocator(base=10))  # diffusion
    axs[1, 2].yaxis.set_major_locator(plticker.MultipleLocator(base=4))  # paths

    # vector field as title
    axs[0, 0].set_title("Drift", fontsize=5, pad=2)
    axs[0, 1].set_title("Diffusion", fontsize=5, pad=2)
    axs[0, 2].set_title("Paths", fontsize=5, pad=2)

    # # describe x axis
    axs[0, 0].set_ylabel("Oil price fluctuation", fontsize=5)
    axs[1, 0].set_ylabel("Wind speed fluctuation", fontsize=5)

    # general plot config
    linewidth = 1
    linewidth_paths = 0.2
    loc_size_per_dim = 32
    loc_subsample_factor = 4

    gt_color = "black"
    bisde_color = "#CC79A7"
    fimsde_color = "#0072B2"

    gt_linestyle = "solid"
    # bisde_linestyle = "dashdot"
    bisde_linestyle = "solid"
    fimsde_linestyle = "solid"

    gt_label = "Observations"
    bisde_label = "BISDE"
    fimsde_label = "FIM-SDE"

    # plot ground-truh paths
    tau_oil = 1
    times_oil = np.arange(gt_paths["oil"].size)
    _plot_1D_paths(
        axs[0, 2],
        times_oil.reshape(1, -1),
        gt_paths["oil"].reshape(1, -1),
        color=gt_color,
        linewidth=0.1,
        linestyle=gt_linestyle,
        label=gt_label,
        alpha=0.7,
    )
    tau_wind = 1 / 6
    times_wind = np.arange(gt_paths["wind"].size)
    _plot_1D_paths(
        axs[1, 2],
        times_wind.reshape(1, -1),
        gt_paths["wind"].reshape(1, -1),
        color=gt_color,
        linewidth=0.075,
        linestyle=gt_linestyle,
        label=gt_label,
        alpha=0.7,
    )

    # plot bisde results
    oil_bisde_results = bisde_results["oil"]
    oil_bisde_drift = oil_bisde_results["drift_at_locations"]
    oil_bisde_diff = oil_bisde_results["diffusion_at_locations"]
    _plot_1D_vfs(
        axs[0, :2],
        location_data["oil"],
        oil_bisde_drift,
        oil_bisde_diff,
        color=bisde_color,
        linewidth=linewidth,
        linestyle=bisde_linestyle,
        label=bisde_label,
    )

    wind_bisde_results = bisde_results["wind"]
    wind_bisde_drift = wind_bisde_results["drift_at_locations"]
    wind_bisde_diff = wind_bisde_results["diffusion_at_locations"]
    _plot_1D_vfs(
        axs[1, :2],
        location_data["wind"],
        wind_bisde_drift,
        wind_bisde_diff,
        color=bisde_color,
        linewidth=linewidth,
        linestyle=bisde_linestyle,
        label=bisde_label,
    )

    # plot fimsde results
    oil_fimsde_results = fimsde_results["oil"]
    oil_fimsde_drift = oil_fimsde_results["drift_at_locations"]
    oil_fimsde_diff = oil_fimsde_results["diffusion_at_locations"]
    oil_fimsde_paths = oil_fimsde_results["synthetic_paths"]
    _plot_1D_vfs(
        axs[0, :2],
        location_data["oil"],
        oil_fimsde_drift,
        oil_fimsde_diff,
        color=fimsde_color,
        linewidth=linewidth,
        linestyle=fimsde_linestyle,
        label=fimsde_label,
    )
    _plot_1D_paths(
        axs[0, 2],
        times_oil.reshape(1, -1),
        oil_fimsde_paths.reshape(1, -1),
        color=fimsde_color,
        linewidth=linewidth_paths,
        linestyle=fimsde_linestyle,
        label=None,
    )

    wind_fimsde_results = fimsde_results["wind"]
    wind_fimsde_drift = wind_fimsde_results["drift_at_locations"]
    wind_fimsde_diff = wind_fimsde_results["diffusion_at_locations"]
    wind_fimsde_paths = wind_fimsde_results["synthetic_paths"]
    _plot_1D_vfs(
        axs[1, :2],
        location_data["wind"],
        wind_fimsde_drift,
        wind_fimsde_diff,
        color=fimsde_color,
        linewidth=linewidth,
        linestyle=fimsde_linestyle,
        label=fimsde_label,
    )
    _plot_1D_paths(
        axs[1, 2],
        times_wind.reshape(1, -1),
        wind_fimsde_paths.reshape(1, -1),
        color=fimsde_color,
        linewidth=linewidth_paths,
        linestyle=fimsde_linestyle,
        label=None,
    )

    # place right legend directly on top of the plot
    plt.draw()

    # legend_fontsize = 6
    legend_fontsize = 5
    bbox_x = axs[0, 1].get_position().x0 + 0.5 * (axs[0, 1].get_position().x1 - axs[0, 1].get_position().x0)
    bbox_y = axs[0, 1].get_position().y1 * 1.03

    handles_vfs, labels_vfs = axs[0, 0].get_legend_handles_labels()
    handles_paths, labels_paths = axs[0, 2].get_legend_handles_labels()

    handles = handles_paths + handles_vfs
    labels = labels_paths + labels_vfs

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
