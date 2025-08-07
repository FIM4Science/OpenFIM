import json
from copy import copy
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import optree
from lab_visits_synthetic_equations_figure import plot_1D_paths, plot_2D_paths
from paper_figure_double_well_2D_synth import _load_from_json

from fim import project_path
from fim.utils.sde.evaluation import save_fig


def _load_real_world_from_json(path_to_json: Path, name: str, split_num: int):
    all_results: list[dict] = json.load(open(path_to_json))

    results_with_name = [result for result in all_results if result["name"] == name]
    results_with_split_num = [result for result in results_with_name if result["split"] == split_num]

    assert len(results_with_split_num) == 1, f"Got {len(results_with_split_num)}."

    results = results_with_split_num[0]
    # results.pop("name", None)
    # results.pop("tau", None)
    # results.pop("noise", None)
    # results.pop("equations", None)  # BISDE results has extra key

    results = {k: np.array(v) if isinstance(v, list) else v for k, v in results.items()}

    return results


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    global_description = "combined_figure_sample_paths_dw_wang_fb_tsla"

    # current_description = "neurips_search_for_decent_results_tau_0_002_noise_0_exp_0"
    current_description = "okay_split_for_each_neurips_submission"

    # data and results to load
    path_to_synthetic_data_json = Path(
        "/cephfs_projects/foundation_models/data/SDE/external_evaluations_and_data/20250325_synthetic_systems_5000_points_with_additive_noise/data/systems_ksig_reference_paths.json"
    )
    path_to_synthetic_fimsde_json = Path(
        "/cephfs_projects/foundation_models/data/SDE/saved_evaluation_results/20250329_neurips_submission_evaluations/synthetic_systems_vf_and_paths/05141437_fim_fixed_softmax_dim_05-03-2033_epoch_139/model_paths.json"
    )

    path_to_real_world_cross_validation_data_json = Path(
        "/cephfs_projects/foundation_models/data/SDE/test/20250726_real_world_with_5_fold_cross_validation/cross_val_ksig_reference_paths.json"
    )
    path_to_real_world_cross_validation_fimsde_json = Path(
        "/cephfs_projects/foundation_models/data/SDE/saved_evaluation_results/20250329_neurips_submission_evaluations/real_world_cross_validation_vf_and_paths_evaluation/05140056_fim_fixed_softmax_05-03-2033_epoch_138/model_paths.json"
    )

    # results to plot
    systems_to_plot = ["Double Well", "Wang"]
    noise = 0.0
    synthetic_experiment_to_plot = 0

    select_synthetic_fimsde_results = {
        "Double Well": {"tau": 0.002, "noise": noise, "exp": 1},  # any is okay
        "Wang": {"tau": 0.002, "noise": noise, "exp": 0},  # good: 0, 1
    }

    real_world_to_plot = ["fb", "tsla"]
    select_real_world_fimsde_results = {
        "fb": {"split_num": 0},  # good: 0, 3, 4
        "tsla": {"split_num": 4},  # good: 3, 4
    }

    # plot config
    num_paths_synthetic = 10

    fimsde_plot_config = {
        "color": "#0072B2",
        "linestyle": "solid",
        "label": "FIM-SDE",
        "linewidth": 0.2,
    }

    gt_plot_config = {
        "color": "black",
        "linestyle": "solid",
        "label": "Observations",
        "linewidth": 0.2,
    }

    # linewidth_paths = 0.2

    # --------------------------------------------------------------------------------------------------------------------------------- #

    # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
    evaluation_path = Path(project_path) / "evaluations"
    time: str = str(datetime.now().strftime("%m%d%H%M"))
    evaluation_dir: Path = evaluation_path / global_description / (time + "_" + current_description)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # load ground-truth synthetic paths
    all_paths: list[dict] = json.load(open(path_to_synthetic_data_json))
    all_systems_paths: dict = {all_paths[i].get("name"): copy(all_paths[i]) for i in range(len(all_paths))}
    all_systems_paths = {k: v for k, v in all_systems_paths.items() if k in systems_to_plot}
    all_systems_paths = optree.tree_map(lambda x: np.array(x), all_systems_paths, is_leaf=lambda x: isinstance(x, list))
    # data is the same in all experiments
    # keys: obs_times, obs_values, locations, drift_at_locations, diffusion_at_locations per system

    # load synthetic results from select systems to plot
    synthetic_fimsde_results = {
        system: _load_from_json(
            path_to_synthetic_fimsde_json, system, **select_synthetic_fimsde_results[system], apply_sqrt_to_diffusion=False
        )
        for system in systems_to_plot
    }

    # load real world data
    all_real_world_paths = {
        label: _load_real_world_from_json(path_to_real_world_cross_validation_data_json, label, **select_real_world_fimsde_results[label])
        for label in real_world_to_plot
    }

    # load real world results from selected datasets to plot
    real_world_fimsde_results = {
        label: _load_real_world_from_json(path_to_real_world_cross_validation_fimsde_json, label, **select_real_world_fimsde_results[label])
        for label in real_world_to_plot
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
    axs[0].yaxis.set_major_locator(plticker.MultipleLocator(base=1))

    # configure wang 2D synthetic
    axs[1].xaxis.set_major_locator(plticker.MultipleLocator(base=1))
    axs[1].yaxis.set_major_locator(plticker.MultipleLocator(base=2))

    # configure fb
    # axs[3].xaxis.set_major_locator(plticker.MultipleLocator(base=0.02))
    axs[3].yaxis.set_major_locator(plticker.MultipleLocator(base=10))

    # configure tsla
    # axs[4].xaxis.set_major_locator(plticker.MultipleLocator(base=0.02))
    # axs[4].yaxis.set_major_locator(plticker.MultipleLocator(base=20))

    # write as title
    axs[0].set_title("Double Well", fontsize=5, pad=2)
    axs[1].set_title("Synthetic 2D", fontsize=5, pad=2)
    axs[3].set_title("Facebook Stock", fontsize=5, pad=2)
    axs[4].set_title("Tesla Stock", fontsize=5, pad=2)

    # 1D data (double well) requires time for plotting paths
    dw_times = 0.002 * np.arange(all_systems_paths["Double Well"]["real_paths"][0].shape[1]).reshape(1, -1, 1)

    # plot ground-truth
    dw_paths = all_systems_paths["Double Well"]["real_paths"][synthetic_experiment_to_plot]  # [100, 500, 1]
    wang_paths = all_systems_paths["Wang"]["real_paths"][synthetic_experiment_to_plot]  # [100, 500, 2]

    fb_times = all_real_world_paths["fb"]["obs_times"][0, 0]  # prev: [1, 10, T, 1], but we reset all 10 paths to start at 0
    fb_times = fb_times - fb_times.min()
    fb_paths = all_real_world_paths["fb"]["obs_values"][0]  # prev: [1, 10, T, 1]

    tsla_times = all_real_world_paths["tsla"]["obs_times"][0, 0]  # prev: [1, 10, T, 1], but we reset all 10 paths to start at 0
    tsla_times = tsla_times - tsla_times.min()
    tsla_paths = all_real_world_paths["tsla"]["obs_values"][0]  # prev: [1, 10, T, 1]

    plot_1D_paths(axs[0], dw_times, dw_paths[:num_paths_synthetic], **gt_plot_config)
    plot_2D_paths(axs[1], wang_paths[:num_paths_synthetic], **gt_plot_config)
    plot_1D_paths(axs[3], fb_times, np.exp(fb_paths), **gt_plot_config)
    plot_1D_paths(axs[4], tsla_times, np.exp(tsla_paths), **gt_plot_config)

    # plot fimsde paths
    dw_paths = synthetic_fimsde_results["Double Well"]["synthetic_paths"]  # [100, 500, 1]
    wang_paths = synthetic_fimsde_results["Wang"]["synthetic_paths"]  # [100, 500, 1]

    fb_paths = real_world_fimsde_results["fb"]["synthetic_paths"][0]  # prev: [1, 10, T, 1]
    tsla_paths = real_world_fimsde_results["tsla"]["synthetic_paths"][0]  # prev: [1, 10, T, 1]

    plot_1D_paths(axs[0], dw_times, dw_paths[:num_paths_synthetic], **fimsde_plot_config)
    plot_2D_paths(axs[1], wang_paths[:num_paths_synthetic], **fimsde_plot_config)
    plot_1D_paths(axs[3], fb_times, np.exp(fb_paths), **fimsde_plot_config)
    plot_1D_paths(axs[4], tsla_times, np.exp(tsla_paths), **fimsde_plot_config)

    # place right legend directly on top of the plot
    plt.draw()
    handles, labels = axs[0].get_legend_handles_labels()

    # # because bise is not on double well
    # quiver_handles, quiver_labels = axs[3].get_legend_handles_labels()
    # bisde_handle = [mlines.Line2D([], [], color=bisde_color, linewidth=linewidth, linestyle="dashdot")]
    # bisde_label = quiver_labels
    #
    # handles = handles[:2] + bisde_handle + handles[2:]
    # labels = labels[:2] + bisde_label + labels[2:]
    #
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
