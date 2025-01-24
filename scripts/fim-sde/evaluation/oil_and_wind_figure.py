import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np

from fim import project_path
from fim.utils.evaluation_sde import save_fig


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    global_description = "paper_figure_oil_and_wind_vf"

    current_description = "develop"

    # data and results to load
    path_to_locations_json = Path(
        "/cephfs_projects/foundation_models/data/SDE/evaluation/20250130_bisde_real_world_oil_wind/data_for_inference.json"
    )
    path_to_bisde_json = Path("/home/seifner/repos/FIM/data/raw/SDE_bisde_on_bisde_oil_wind/bisde_vector_fields.json")
    path_to_fimsde_json = Path("/cephfs_projects/foundation_models/data/SDE/evaluation/20250130_bisde_real_world_oil_wind/model_paths.json")

    # --------------------------------------------------------------------------------------------------------------------------------- #

    # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
    evaluation_path = Path(project_path) / "evaluations"
    time: str = str(datetime.now().strftime("%m%d%H%M"))
    evaluation_dir: Path = evaluation_path / global_description / (time + "_" + current_description)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    def _load_from_json(path_to_json: Path, dataset_name: str, apply_sqrt_to_diffusion: bool):
        all_results: list[dict] = json.load(open(path_to_json))

        results_with_dataset = [result for result in all_results if result["name"] == dataset_name]
        # if len(results_with_dataset) != 1:
        #     breakpoint()
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
        # if len(results_with_dataset) != 1:
        #     breakpoint()

        locations = results_with_dataset[0]["locations"]
        locations = np.array(locations)
        return locations

    location_data = {
        "oil": _load_location_data(path_to_locations_json, "oil"),
        "wind": _load_location_data(path_to_locations_json, "wind"),
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
    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(7, 1.5), gridspec_kw={"width_ratios": [1, 1, 0.02, 1, 1]}, dpi=300)
    axs[2].axis("off")

    # configure axes general
    for ax in axs:
        [x.set_linewidth(0.3) for x in ax.spines.values()]
        ax.tick_params(axis="both", direction="out", labelsize=5, width=0.5, length=2)

    # configure oil
    axs[0].xaxis.set_major_locator(plticker.MultipleLocator(base=10))
    axs[1].xaxis.set_major_locator(plticker.MultipleLocator(base=10))

    axs[0].yaxis.set_major_locator(plticker.MultipleLocator(base=20))  # drift
    axs[1].yaxis.set_major_locator(plticker.MultipleLocator(base=5))  # diffusion

    # configure wind
    axs[3].xaxis.set_major_locator(plticker.MultipleLocator(base=5))
    axs[4].xaxis.set_major_locator(plticker.MultipleLocator(base=5))

    axs[3].yaxis.set_major_locator(plticker.MultipleLocator(base=40))  # drift
    axs[4].yaxis.set_major_locator(plticker.MultipleLocator(base=5))  # diffusion

    # vector field as title
    axs[0].set_title("Drift", fontsize=5, pad=2)
    axs[1].set_title("Diffusion", fontsize=5, pad=2)
    axs[3].set_title("Drift", fontsize=5, pad=2)
    axs[4].set_title("Diffusion", fontsize=5, pad=2)

    # describe x axis
    axs[0].set_xlabel("Oil price fluctuation", fontsize=5)
    axs[1].set_xlabel("Oil price fluctuation", fontsize=5)
    axs[3].set_xlabel("Wind speed fluctuation", fontsize=5)
    axs[4].set_xlabel("Wind speed fluctuation", fontsize=5)

    # general plot config
    linewidth = 1
    loc_size_per_dim = 32
    loc_subsample_factor = 4

    bisde_color = "#CC79A7"
    fimsde_color = "#0072B2"

    bisde_linestyle = "dashdot"
    fimsde_linestyle = "solid"

    bisde_label = "BISDE"
    fimsde_label = "FIM-SDE"

    def _plot_1D_vfs(axs, locations, drift, diffusion, color, linewidth, linestyle, label):
        axs[0].plot(locations.squeeze(), drift.squeeze(), color=color, linewidth=linewidth, linestyle=linestyle, label=label)
        axs[1].plot(locations.squeeze(), diffusion.squeeze(), color=color, linewidth=linewidth, linestyle=linestyle)

    # plot bisde results
    oil_bisde_results = bisde_results["oil"]
    oil_bisde_drift = oil_bisde_results["drift_at_locations"]
    oil_bisde_diff = oil_bisde_results["diffusion_at_locations"]
    _plot_1D_vfs(
        axs[:2],
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
        axs[3:],  # axs[2] divides between oil and wind
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
    _plot_1D_vfs(
        axs[:2],
        location_data["oil"],
        oil_fimsde_drift,
        oil_fimsde_diff,
        color=fimsde_color,
        linewidth=linewidth,
        linestyle=fimsde_linestyle,
        label=fimsde_label,
    )

    wind_fimsde_results = fimsde_results["wind"]
    wind_fimsde_drift = wind_fimsde_results["drift_at_locations"]
    wind_fimsde_diff = wind_fimsde_results["diffusion_at_locations"]
    _plot_1D_vfs(
        axs[3:],  # axs[2] divides between oil and wind
        location_data["wind"],
        wind_fimsde_drift,
        wind_fimsde_diff,
        color=fimsde_color,
        linewidth=linewidth,
        linestyle=fimsde_linestyle,
        label=fimsde_label,
    )

    # place right legend directly on top of the plot
    plt.draw()
    handles, labels = axs[0].get_legend_handles_labels()
    # legend_fontsize = 6
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
