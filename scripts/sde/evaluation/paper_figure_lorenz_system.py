import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fim import project_path
from fim.utils.evaluation_sde import save_fig


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
    global_description = "paper_figure_lorenz_sample_paths"

    # current_description = "neurips_search_for_decent_results_tau_0_002_noise_0_exp_0"
    current_description = "develop"

    reference_paths_json = Path(
        "/cephfs_projects/foundation_models/data/SDE/external_evaluations_and_data/20250512_lorenz_data_from_neural_sde_github/20250514221957_lorenz_system_mmd_reference_paths/20250514221957_lorenz_mmd_reference_data.json"
    )
    fim_no_finetune_paths_json = Path(
        "/cephfs_projects/foundation_models/data/SDE/saved_evaluation_results/20250329_neurips_submission_evaluations/lorenz_system_vf_and_paths_evaluation/05160055_neurips_model_no_finetuning/model_paths/fim_model_C_at_139_epochs_no_finetuning_train_data_linear_diffusion_num_context_paths_1024.json"  # data and results to load
    )
    fim_finetune_paths_json = Path(
        "/cephfs_projects/foundation_models/data/SDE/saved_evaluation_results/20250329_neurips_submission_evaluations/lorenz_system_vf_and_paths_evaluation/05160031_neurips_model_finetuning_on_128_paths_up_to_500_epochs/model_paths/fim_model_C_at_139_epochs_finetuned_on_128_paths_all_points_lr_1e-6_epochs_200_train_data_linear_diffusion_num_context_paths_1024.json"
    )
    latent_sde_paths_json = Path(
        "/cephfs_projects/foundation_models/data/SDE/external_evaluations_and_data/20250512_lorenz_data_from_neural_sde_github/05161005_latent_dim_3_paths_sampled_from_prior_lorenz-16_paths.json"
    )

    diffusion_label = "linear"
    initial_state_label = "sampled_normal_mean_0_std_1"

    num_plot_paths = 128

    # results to plot
    # systems_to_plot = ["Double Well", "Wang"]
    # noise = 0.0
    # synthetic_experiment_to_plot = 0
    #
    # select_synthetic_fimsde_results = {
    #     "Double Well": {"tau": 0.002, "noise": noise, "exp": 1},  # any is okay
    #     "Wang": {"tau": 0.002, "noise": noise, "exp": 0},  # good: 0, 1
    # }
    #
    # real_world_to_plot = ["fb", "tsla"]
    # select_real_world_fimsde_results = {
    #     "fb": {"split_num": 0},  # good: 0, 3, 4
    #     "tsla": {"split_num": 4},  # good: 3, 4
    # }
    #
    # # plot config
    # num_paths_synthetic = 10
    #
    fim_plot_config = {
        "color": "#0072B2",
        "linestyle": "solid",
        "label": "FIM-SDE",
        "linewidth": 0.2,
    }

    fim_finetuned_plot_config = {
        "color": "blue",
        "linestyle": "solid",
        "label": "FIM-SDE finetuned",
        "linewidth": 0.2,
    }

    gt_plot_config = {
        "color": "black",
        "linestyle": "solid",
        "label": "Observations",
        "linewidth": 0.2,
    }

    latent_sde_plot_config = {
        "color": "#CC79A7",
        "linestyle": "solid",
        "label": "Latent SDE",
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
    all_paths: list[dict] = json.load(open(reference_paths_json, "r"))
    reference_paths: list[dict] = [
        d for d in all_paths if d["diffusion_label"] == diffusion_label and d["initial_state_label"] == initial_state_label
    ]
    assert len(reference_paths) == 1
    reference_paths = np.array(reference_paths[0]["paths"]).squeeze()

    # no finetuned
    all_fim_no_finetuned_paths: list[dict] = json.load(open(fim_no_finetune_paths_json, "r"))
    fim_no_finetuned_paths = [
        d
        for d in all_fim_no_finetuned_paths
        if d["train_data_diffusion_label"] == diffusion_label and d["initial_state_label"] == initial_state_label
    ]
    assert len(fim_no_finetuned_paths) == 1
    fim_no_finetuned_paths = np.array(fim_no_finetuned_paths[0]["synthetic_path"]).squeeze()

    # finetuned
    all_fim_finetuned_paths: list[dict] = json.load(open(fim_finetune_paths_json, "r"))
    fim_finetuned_paths = [
        d
        for d in all_fim_finetuned_paths
        if d["train_data_diffusion_label"] == diffusion_label and d["initial_state_label"] == initial_state_label
    ]
    assert len(fim_finetuned_paths) == 1
    fim_finetuned_paths = np.array(fim_finetuned_paths[0]["synthetic_path"]).squeeze()

    # latent sde
    all_latent_sde_paths: list[dict] = json.load(open(latent_sde_paths_json, "r"))
    latent_sde_paths = [
        d
        for d in all_latent_sde_paths
        if d["train_data_diffusion_label"] == diffusion_label and d["initial_state_label"] == initial_state_label
    ]
    assert len(latent_sde_paths) == 1
    latent_sde_paths = np.array(latent_sde_paths[0]["synthetic_path"]).squeeze()

    # data is the same in all experiments
    # keys: obs_times, obs_values, locations, drift_at_locations, diffusion_at_locations per system

    # create figure with separating gap between the two systems
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(7, 1.5), dpi=300, subplot_kw={"projection": "3d"})

    # configure axes general
    for ax in axs:
        ax.set_axis_off()

    for i in range(num_plot_paths):
        for a in range(3):
            axs[a].plot(reference_paths[i, :, 0], reference_paths[i, :, 1], reference_paths[i, :, 2], **gt_plot_config)

        axs[0].plot(latent_sde_paths[i, :, 0], latent_sde_paths[i, :, 1], latent_sde_paths[i, :, 2], **latent_sde_plot_config)
        axs[1].plot(fim_no_finetuned_paths[i, :, 0], fim_no_finetuned_paths[i, :, 1], fim_no_finetuned_paths[i, :, 2], **fim_plot_config)
        axs[2].plot(fim_finetuned_paths[i, :, 0], fim_finetuned_paths[i, :, 1], fim_finetuned_paths[i, :, 2], **fim_finetuned_plot_config)

    axs[1].scatter(fim_no_finetuned_paths[:, 0, 0], fim_no_finetuned_paths[:, 0, 1], fim_no_finetuned_paths[:, 0, 2], marker="o", c="red")

    # # place right legend directly on top of the plot
    # plt.draw()
    # handles, labels = axs[0].get_legend_handles_labels()
    #
    # # # because bise is not on double well
    # # quiver_handles, quiver_labels = axs[3].get_legend_handles_labels()
    # # bisde_handle = [mlines.Line2D([], [], color=bisde_color, linewidth=linewidth, linestyle="dashdot")]
    # # bisde_label = quiver_labels
    # #
    # # handles = handles[:2] + bisde_handle + handles[2:]
    # # labels = labels[:2] + bisde_label + labels[2:]
    # #
    # # legend_fontsize = 6
    # legend_fontsize = 5
    # bbox_x = axs[2].get_position().x0 + 0.5 * (axs[2].get_position().x1 - axs[2].get_position().x0)
    # bbox_y = axs[2].get_position().y1 * 1.07
    #
    # legend = fig.legend(
    #     handles,
    #     labels,
    #     loc="lower center",
    #     bbox_to_anchor=[bbox_x, bbox_y],
    #     fontsize=legend_fontsize,
    #     ncols=4,
    # )
    #
    # save
    save_dir: Path = evaluation_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    file_name = "vector_field_plot"
    save_fig(fig, save_dir, file_name)
