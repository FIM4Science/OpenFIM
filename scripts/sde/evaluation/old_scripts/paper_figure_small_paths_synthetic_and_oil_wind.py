import json
from copy import copy
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import optree
from lab_visits_oil_wind_figure import _load_from_json, _load_gt_paths, _plot_1D_paths
from lab_visits_synthetic_equations_figure import load_system_results, plot_1D_paths, plot_2D_paths
from matplotlib.lines import Line2D

from fim import project_path
from fim.utils.sde.evaluation import save_fig


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    global_description = "paper_small_paths_figure"

    current_description = "first_iteration"

    # real world data and results to load
    path_to_real_world_paths_json = Path(
        # "/cephfs_projects/foundation_models/data/SDE/external_evaluations_and_data/20250130_bisde_real_world_oil_wind/ksig_reference_paths.json"
        "/Users/patrickseifner/repos/FIM/saved_evaluations/20250203_icml_submission_evaluations/bisde_real_world_evaluation_for_figure_and_table/data_jsons/ksig_reference_paths.json"
    )
    path_to_bisde_real_world_json = Path(
        "/Users/patrickseifner/repos/FIM/data/raw/SDE_bisde_on_bisde_oil_wind/bisde_vector_fields.json",
    )
    path_to_fimsde_real_world_json = Path(
        # "/cephfs_projects/foundation_models/data/SDE/external_evaluations_and_data/20250130_bisde_real_world_oil_wind/model_paths.json"
        "/Users/patrickseifner/repos/FIM/saved_evaluations/20250203_icml_submission_evaluations/bisde_real_world_evaluation_for_figure_and_table/data_jsons/model_paths.json"
    )

    # synthetic data and results to load
    path_to_synthetic_paths_json = Path(
        "/Users/patrickseifner/repos/FIM/saved_evaluations/20250203_icml_submission_evaluations/synthetic_equations_stride_1_5_10_for_table/data_jsons/ksig_reference_paths.json"
    )
    path_to_bisde_synthetic_json = Path(
        "/Users/patrickseifner/sciebo/sde_data_transfer/coarse_obs_systems_data_5000_points/bisde_experiments_friday_full.json"
    )
    path_to_fimsde_synthetic_json = Path(
        "/Users/patrickseifner/repos/FIM/saved_evaluations/20250203_icml_submission_evaluations/synthetic_equations_stride_1_5_10_for_table/data_jsons/model_paths.json"
    )

    # select visually best experiment per model and system
    select_bisde_synthetic_results = {
        "Double Well": {"tau": 0.002, "exp": 2},
        "Wang": {"tau": 0.002, "exp": 3},
        "Damped Linear": {"tau": 0.002, "exp": 1},
        "Damped Cubic": {"tau": 0.002, "exp": 0},
        "Duffing": {"tau": 0.002, "exp": 3},
        "Glycosis": {"tau": 0.002, "exp": 3},
        "Hopf": {"tau": 0.002, "exp": 2},
    }
    select_fimsde_synthetic_results = {
        "Double Well": {"tau": 0.02, "exp": 4},
        "Wang": {"tau": 0.02, "exp": 1},
        "Damped Linear": {"tau": 0.01, "exp": 1},
        "Damped Cubic": {"tau": 0.02, "exp": 1},
        "Duffing": {"tau": 0.02, "exp": 0},
        "Glycosis": {"tau": 0.02, "exp": 0},
        "Hopf": {"tau": 0.01, "exp": 2},
    }

    # configure axis tick spacing per system
    tick_base_per_synthetic_system = {
        "Double Well": {"x": 1, "y": 1},
        "Wang": {"x": 2, "y": 2},
        "Damped Linear": {"x": 2, "y": 2},
        "Damped Cubic": {"x": 1, "y": 1},
        "Duffing": {"x": 2, "y": 2},
        "Glycosis": {"x": 1, "y": 1},
        "Hopf": {"x": 1, "y": 1},
    }

    # general plot config
    real_world_system = "oil"
    synthetic_system = "Double Well"

    plot_num_paths = 10

    linewidth = 1
    linewidth_paths = 0.2

    gt_plot_config = {
        "color": "black",
        # "linestyle": "dotted",
        "linestyle": "solid",
        "label": "Observations",
    }

    bisde_plot_config = {
        "color": "#CC79A7",
        # "linestyle": "dashdot",
        "linestyle": "solid",
        "label": "BISDE",
    }

    fimsde_plot_config = {
        "color": "#0072B2",
        "linestyle": "solid",
        "label": "FIM-SDE",
    }

    plot_num_synthetic_paths = 10

    # --------------------------------------------------------------------------------------------------------------------------------- #

    # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
    evaluation_path = Path(project_path) / "evaluations"
    time: str = str(datetime.now().strftime("%m%d%H%M"))
    evaluation_dir: Path = evaluation_path / global_description / (time + "_" + current_description)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # load ground-truth paths of selected synthetic system
    all_paths: list[dict] = json.load(open(path_to_synthetic_paths_json))
    all_systems_paths: dict = {all_paths[i].get("name"): copy(all_paths[i]) for i in range(len(all_paths))}
    system_paths = all_systems_paths[synthetic_system]
    gt_system_paths: np.ndarray = optree.tree_map(lambda x: np.array(x), system_paths, is_leaf=lambda x: isinstance(x, list))

    # load results on selected synthetic system
    bisde_results = load_system_results(
        json.load(open(path_to_bisde_synthetic_json)),
        synthetic_system,
        **select_bisde_synthetic_results[synthetic_system],
        apply_sqrt_to_diffusion=True,
    )
    bisde_synthetic_paths = bisde_results["synthetic_paths"]

    fimsde_results = load_system_results(
        json.load(open(path_to_fimsde_synthetic_json)),
        synthetic_system,
        **select_fimsde_synthetic_results[synthetic_system],
        apply_sqrt_to_diffusion=False,
    )
    fimsde_synthetic_paths = fimsde_results["synthetic_paths"]

    # load observations of real world systems
    gt_real_world_paths = {
        "oil": _load_gt_paths(path_to_real_world_paths_json, "oil"),
        "wind": _load_gt_paths(path_to_real_world_paths_json, "wind"),
    }

    # load observations of real world systems
    bisde_real_world_results = {
        "oil": _load_from_json(path_to_bisde_real_world_json, "oil", apply_sqrt_to_diffusion=True),
        "wind": _load_from_json(path_to_bisde_real_world_json, "wind", apply_sqrt_to_diffusion=True),
    }
    fimsde_real_world_results = {
        "oil": _load_from_json(path_to_fimsde_real_world_json, "oil", apply_sqrt_to_diffusion=False),
        "wind": _load_from_json(path_to_fimsde_real_world_json, "wind", apply_sqrt_to_diffusion=False),
    }

    # create figure
    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(3, 1.5),
        gridspec_kw={"width_ratios": [1, 1]},
        dpi=300,
        tight_layout=True,
    )

    # configure axes general
    for ax in axs.reshape(-1):
        [x.set_linewidth(0.3) for x in ax.spines.values()]
        ax.tick_params(axis="both", direction="out", labelsize=5, width=0.5, length=2)

    if real_world_system == "oil":
        axs[1].xaxis.set_major_locator(plticker.MultipleLocator(base=1000))  # paths
        axs[1].yaxis.set_major_locator(plticker.MultipleLocator(base=5))  # paths
        axs[1].set_ylabel("Oil price fluctuation", fontsize=5)

    else:
        assert real_world_system == "wind", f"Got {real_world_system}"
        axs[1].xaxis.set_major_locator(plticker.MultipleLocator(base=2000))  # paths
        axs[1].yaxis.set_major_locator(plticker.MultipleLocator(base=4))  # paths
        axs[1].set_ylabel("Wind speed fluctuation", fontsize=5)

    axs[0].xaxis.set_major_locator(plticker.MultipleLocator(base=tick_base_per_synthetic_system[synthetic_system]["x"]))
    axs[0].yaxis.set_major_locator(plticker.MultipleLocator(base=tick_base_per_synthetic_system[synthetic_system]["y"]))
    axs[0].set_ylabel(synthetic_system, fontsize=5)

    # plot ground-truh real world paths
    if real_world_system == "oil":
        tau = 1
        real_world_paths = gt_real_world_paths["oil"].reshape(1, -1)
        real_world_times = np.arange(real_world_paths.shape[1]).reshape(1, -1)
        linewidth_gt_real_world_paths = 0.1
        alpha = 0.7

    else:
        assert real_world_system == "wind", f"Got {real_world_system}"
        tau = 1 / 6
        real_world_paths = gt_real_world_paths["wind"].reshape(1, -1)
        real_world_times = np.arange(real_world_paths.shape[1]).reshape(1, -1)
        linewidth_gt_real_world_paths = 0.075
        alpha = 0.7

    _plot_1D_paths(
        axs[1],
        real_world_times,
        real_world_paths,
        color=gt_plot_config["color"],
        linewidth=linewidth_gt_real_world_paths,
        linestyle=gt_plot_config["linestyle"],
        label=gt_plot_config["label"],
        alpha=alpha,
    )

    # # plot bisde real world results
    # bisde_paths = bisde_real_world_results[real_world_system]["synthetic_paths"]
    # _plot_1D_paths(
    #     axs[1],
    #     real_world_times.reshape(1, -1),
    #     bisde_paths.reshape(1, -1),
    #     color=bisde_plot_config["color"],
    #     linewidth=linewidth_paths,
    #     linestyle=bisde_plot_config["linestyle"],
    #     label=None,
    # )

    # plot fimsde real world results
    fimsde_paths = fimsde_real_world_results[real_world_system]["synthetic_paths"][:, 0, :, :]
    _plot_1D_paths(
        axs[1],
        real_world_times.reshape(1, -1),
        fimsde_paths.reshape(1, -1),
        color=fimsde_plot_config["color"],
        linewidth=linewidth_paths,
        linestyle=fimsde_plot_config["linestyle"],
        label=None,
    )

    if synthetic_system == "Double Well":
        # double well requires time for plotting paths
        times = (0.002 * np.arange(gt_system_paths["real_paths"].shape[2])).reshape(1, -1, 1)
        plot_1D_paths(
            axs[0],
            times,
            gt_system_paths["real_paths"][0, :plot_num_paths],
            color=gt_plot_config["color"],
            linewidth=linewidth_paths,
            linestyle=gt_plot_config["linestyle"],
            label=gt_plot_config["label"],
        )
        # plot_1D_paths(
        #     axs[0],
        #     times.squeeze(),
        #     bisde_synthetic_paths,
        #     color=bisde_plot_config["color"],
        #     linewidth=linewidth_paths,
        #     linestyle=bisde_plot_config["linestyle"],
        #     label=bisde_plot_config["label"],
        # )
        plot_1D_paths(
            axs[0],
            times.squeeze(),
            fimsde_synthetic_paths[:plot_num_paths],
            color=fimsde_plot_config["color"],
            linewidth=linewidth_paths,
            linestyle=fimsde_plot_config["linestyle"],
            label=fimsde_plot_config["label"],
        )

    else:
        plot_2D_paths(
            axs[0],
            gt_system_paths["real_paths"][0, :plot_num_paths],
            color=gt_plot_config["color"],
            linewidth=linewidth_paths,
            linestyle=gt_plot_config["linestyle"],
            label=gt_plot_config["label"],
        )
        # plot_2D_paths(
        #     axs[0],
        #     bisde_synthetic_paths[0],
        #     color=bisde_plot_config["color"],
        #     linewidth=linewidth_paths,
        #     linestyle=bisde_plot_config["linestyle"],
        #     label=bisde_plot_config["label"],
        # )
        plot_2D_paths(
            axs[0],
            fimsde_synthetic_paths[0][:plot_num_paths],
            color=fimsde_plot_config["color"],
            linewidth=linewidth_paths,
            linestyle=fimsde_plot_config["linestyle"],
            label=fimsde_plot_config["label"],
        )

    # place right legend directly on top of the plot
    plt.draw()

    # legend_fontsize = 6
    legend_fontsize = 5
    bbox_x = axs[0].get_position().x0 + 0.5 * (axs[1].get_position().x1 - axs[0].get_position().x0)
    bbox_y = axs[0].get_position().y1 * 1.03

    # handles, labels = axs[0].get_legend_handles_labels()
    # larger lines in legend for clarity
    handles = [
        Line2D([0], [0], color=gt_plot_config["color"], linestyle="solid", linewidth=0.8),
        Line2D([0], [0], color=fimsde_plot_config["color"], linestyle="solid", linewidth=0.8),
    ]
    labels = [gt_plot_config["label"], fimsde_plot_config["label"]]

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
    file_name = "paths_plot"
    save_fig(fig, save_dir, file_name)
