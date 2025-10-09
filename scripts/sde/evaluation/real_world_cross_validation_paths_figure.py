import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optree
from matplotlib.lines import Line2D
from synthetic_systems_figure_all_vfs_and_paths import plot_1D_paths

from fim import project_path
from fim.utils.sde.evaluation import save_fig


def extract_results_of_split(all_results: list[dict], dataset_label: str, split: int) -> dict:
    results_of_label = [d for d in all_results if d["name"] == dataset_label]
    results_of_split = [d for d in results_of_label if d["split"] == split]
    assert len(results_of_split) == 1, f"Got {len(results_of_split)}."
    return results_of_split[0]


reference_data_json = Path(
    "/cephfs_projects/foundation_models/data/SDE/test/20250726_real_world_with_5_fold_cross_validation/cross_val_ksig_reference_paths.json"
)
rebuttal_base = Path(
    "/cephfs_projects/foundation_models/data/SDE/saved_evaluation_results/20250808_neurips_rebuttal_evaluations/real_world_cross_validation_vf_and_paths_evaluation/"
)
latent_sde_NLL_train_subsplit_10_base = rebuttal_base / "latent_sde_latent_dim_4_context_dim_100_decoder_NLL_train_subsplits_10/"
finetune_neurips_rebuttal_512_points_base = (
    rebuttal_base / "finetune_for_neurips_rebuttal_one_step_ahead_one_em_step_nll_512_points_500_epochs"
)

latent_sde_json = latent_sde_NLL_train_subsplit_10_base / "combined_outputs_epoch_4999.json"
fim_no_finetune_json = Path(
    "/cephfs_projects/foundation_models/data/SDE/saved_evaluation_results/20250329_neurips_submission_evaluations/real_world_cross_validation_vf_and_paths_evaluation/05140056_fim_fixed_softmax_05-03-2033_epoch_138/model_paths.json",
)
fim_finetune_json = finetune_neurips_rebuttal_512_points_base / "combined_outputs_epoch_99.json"

selected_split = {
    "wind": 4,
    "oil": 1,
    "fb": 2,
    "tsla": 4,
}

gt_plot_config = {
    "label": "Ground-truth",
    "color": "black",
    "linestyle": "solid",
    # "linewidth": 0.1,
}

latent_sde_plot_config = {
    "label": "LatentSDE",
    "color": "#D55E00",
    "linestyle": "solid",
    # "linewidth": 0.1,
}

fim_no_finetune_plot_config = {
    "label": "FIM-SDE Zero-Shot",
    "color": "#0072B2",
    "linestyle": "solid",
    # "linewidth": 0.1,
}

fim_finetune_plot_config = {
    "label": "FIM-SDE Finetuned",
    "color": "#56B4E9",
    "linestyle": "solid",
    # "linewidth": 0.1,
}

linewidths = {
    "wind": 0.2,  # 0.05,
    "oil": 0.2,  # 0.05,
    "fb": 0.3,
    "tsla": 0.3,
}

fix_ylim = {
    "wind": {"ymin": -5.5, "ymax": 6.5},
    "oil": {"ymin": -4, "ymax": 4},
    "fb": {"ymin": 245, "ymax": 285},
    "tsla": {"ymin": 330, "ymax": 490},
}

num_obs = {
    "wind": 80,
    "oil": 80,
    "fb": None,
    "tsla": None,
}


def plot_paths_subplot(axs, row, col, model_paths, plot_config, gt_plot_config, dataset_label, selected_split, num_obs):
    if fix_ylim.get(dataset_label) is not None:
        axs[row, col].set_ylim(**fix_ylim.get(dataset_label))

    axs[row, col].xaxis.set_major_locator(plt.MaxNLocator(3))
    axs[row, col].yaxis.set_major_locator(plt.MaxNLocator(3))

    selcted_gt_paths = extract_results_of_split(reference_paths, dataset_label, selected_split[dataset_label])
    selected_model_paths = extract_results_of_split(model_paths, dataset_label, selected_split[dataset_label])

    obs_times = np.array(selcted_gt_paths["obs_times"])[0, 0]  # [T, 1]
    obs_times = obs_times - obs_times[0]
    ref_values = np.array(selcted_gt_paths["obs_values"]).squeeze(0)  # [P, T, 1]
    model_values = np.array(selected_model_paths["synthetic_paths"]).squeeze(0)  # [P, T, 1]

    if (n := num_obs.get(dataset_label)) is not None:
        obs_times = obs_times[:n]
        ref_values = ref_values[:, :n]
        model_values = model_values[:, :n]

    if dataset_label in ["fb", "tsla"]:
        ref_values = np.exp(ref_values)
        model_values = np.exp(model_values)

    plot_1D_paths(axs[row, col], obs_times, ref_values, linewidth=linewidths[dataset_label], **gt_plot_config)
    plot_1D_paths(axs[row, col], obs_times, model_values, linewidth=linewidths[dataset_label], **plot_config)


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    dataset_descr = "real_world_cross_validation_paths_figure"

    # How to name experiments
    experiment_descr = "with_main_text_figure"

    # --------------------------------------------------------------------------------------------------------------------------------- #

    # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
    evaluation_path = Path(project_path) / "evaluations"
    time: str = str(datetime.now().strftime("%m%d%H%M"))
    evaluation_dir: Path = evaluation_path / dataset_descr / (time + "_" + experiment_descr)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    reference_paths: list[dict] = json.load(open(reference_data_json, "r"))  # "obs_times" "obs_values":  [1, 10, T, 1]
    latent_sde_paths, fim_no_finetune_paths, fim_finetune_paths = optree.tree_map(  # "synthetic_paths": [1, 10, T, 1]
        lambda x: json.load(open(x, "r")), (latent_sde_json, fim_no_finetune_json, fim_finetune_json)
    )

    ### All systems, including latentsde
    fig, axs = plt.subplots(
        nrows=4,
        ncols=3,
        figsize=(4.5, 6),
        dpi=300,
        tight_layout=True,
        gridspec_kw={
            "width_ratios": [1, 1, 1],
            "height_ratios": [1, 1, 1, 1],
        },
    )

    # configure axes general
    for ax in axs.reshape(-1):
        [x.set_linewidth(0.3) for x in ax.spines.values()]
        ax.tick_params(axis="both", direction="out", labelsize=4, width=0.5, length=2, pad=0.8)

    # plot ground-truth and models paths for each dataset, from a selected split
    for col, (model_paths, plot_config) in enumerate(
        zip(
            [latent_sde_paths, fim_no_finetune_paths, fim_finetune_paths],
            [latent_sde_plot_config, fim_no_finetune_plot_config, fim_finetune_plot_config],
        )
    ):
        for row, (dataset_label, dataset_name) in enumerate(
            zip(
                ["wind", "oil", "fb", "tsla"],
                ["Wind", "Oil", "Facebook", "Tesla"],
            )
        ):
            plot_paths_subplot(axs, row, col, model_paths, plot_config, gt_plot_config, dataset_label, selected_split, num_obs)

            if col == 0:
                axs[row, 0].set_ylabel(dataset_name, fontsize=6)

    # place right legend directly on top of the plot
    plt.draw()

    obs_handle = Line2D([0], [0], color=gt_plot_config["color"], label="Ground-Truth", linewidth=1)
    latentsde_handle = Line2D([0], [0], color=latent_sde_plot_config["color"], label="LatentSDE", linewidth=1)
    fim_handle = Line2D([0], [0], color=fim_no_finetune_plot_config["color"], label="FIM-SDE (Zero-Shot)", linewidth=1)
    fine_handle = Line2D([0], [0], color=fim_finetune_plot_config["color"], label="FIM-SDE (Finetuned)", linewidth=1)

    handles = [obs_handle, latentsde_handle, fim_handle, fine_handle]

    legend_fontsize = 5
    bbox_x = axs[0, 1].get_position().x0 + 0.5 * (axs[0, 1].get_position().x1 - axs[0, 1].get_position().x0)
    bbox_y = axs[0, 1].get_position().y1 * 1.02

    legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=[bbox_x, bbox_y],
        fontsize=legend_fontsize,
        ncols=4,
    )

    save_dir: Path = evaluation_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    save_fig(fig, save_dir, "paths_all_models_all_datasets")

    plt.close(fig)

    ### Oil Tesla, only FIM and FIM finetuned
    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(7, 1.5), gridspec_kw={"width_ratios": [1, 1, 0.02, 1, 1]}, dpi=300)
    axs[2].axis("off")
    axs = axs.reshape(1, -1)

    # configure axes general
    for ax in axs.reshape(-1):
        [x.set_linewidth(0.3) for x in ax.spines.values()]
        ax.tick_params(axis="both", direction="out", labelsize=4, width=0.5, length=2, pad=0.8)

    # plot ground-truth and models paths for each dataset, from a selected split
    for col, model_paths, plot_config, dataset_label in [
        (0, fim_no_finetune_paths, fim_no_finetune_plot_config, "oil"),
        (1, fim_finetune_paths, fim_finetune_plot_config, "oil"),
        (3, fim_no_finetune_paths, fim_no_finetune_plot_config, "tsla"),
        (4, fim_finetune_paths, fim_finetune_plot_config, "tsla"),
    ]:
        row = 0
        plot_paths_subplot(axs, row, col, model_paths, plot_config, gt_plot_config, dataset_label, selected_split, num_obs)

    # place dataset names atop center of each 2 subplots
    left_title_x = axs[0, 0].get_position().x0 + 0.5 * (axs[0, 1].get_position().x1 - axs[0, 0].get_position().x0)
    left_title_y = axs[0, 1].get_position().y1 * 1.013
    plt.figtext(left_title_x, left_title_y, "Oil", va="bottom", ha="center", size=7)

    right_title_x = axs[0, 3].get_position().x0 + 0.5 * (axs[0, 4].get_position().x1 - axs[0, 3].get_position().x0)
    right_title_y = axs[0, 4].get_position().y1 * 1.013
    plt.figtext(right_title_x, right_title_y, "Tesla", va="bottom", ha="center", size=7)

    # place right legend directly on top of the plot
    plt.draw()

    obs_handle = Line2D([0], [0], color=gt_plot_config["color"], label="Ground-Truth", linewidth=1)
    fim_handle = Line2D([0], [0], color=fim_no_finetune_plot_config["color"], label="FIM-SDE (Zero-Shot)", linewidth=1)
    fine_handle = Line2D([0], [0], color=fim_finetune_plot_config["color"], label="FIM-SDE (Finetuned)", linewidth=1)

    handles = [obs_handle, fim_handle, fine_handle]

    legend_fontsize = 6
    bbox_x = axs[0, 2].get_position().x0 + 0.5 * (axs[0, 2].get_position().x1 - axs[0, 2].get_position().x0)
    bbox_y = axs[0, 2].get_position().y1 * 1.08

    legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=[bbox_x, bbox_y],
        fontsize=legend_fontsize,
        ncols=4,
    )

    save_dir: Path = evaluation_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    save_fig(fig, save_dir, "oil_tesla_fim_fim_finetuned_main_text")

    plt.close(fig)
