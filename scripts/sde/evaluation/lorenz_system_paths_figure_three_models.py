import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optree
from matplotlib.lines import Line2D

from fim import project_path
from fim.utils.sde.evaluation import save_fig


rebuttal_base_path = Path(
    "/cephfs_projects/foundation_models/data/SDE/saved_evaluation_results/20250808_neurips_rebuttal_evaluations/lorenz_system_vf_and_paths_evaluation/"
)

fim_no_training_base_path = rebuttal_base_path / "07011430_fim_no_finetune_or_train_from_scratch/model_paths"
convergence_base_path = rebuttal_base_path / "20250715_fim_finetune_vs_retrain_convergence_comparison_mse_and_nll/model_paths"
lat_sde_context_100_base_path = rebuttal_base_path / "07011436_latent_sde_context_100_with_vector_fields/model_paths"


def plot_3D_paths(ax, paths: np.ndarray, label, **plot_config):
    for i in range(paths.shape[0]):
        ax.plot(paths[i, :, 0], paths[i, :, 1], paths[i, :, 2], label=label if i == 0 else None, **plot_config)


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    global_description = "lorenz_system_paths_figure_three_models"

    current_description = "neurips_figure_with_convergence_for_1500_iterations"

    neural_sde_paper_path = Path("/cephfs_projects/foundation_models/data/SDE/test/20250629_lorenz_systems/neural_sde_paper/set_0/")
    reference_paths_jsons = neural_sde_paper_path / "N(0,1)_reference_data.json"

    latent_sde_json = lat_sde_context_100_base_path / "lat_sde_context_100_train_data_neural_sde_paper.json"
    fim_no_finetune_json = fim_no_training_base_path / "fim_model_C_at_139_epochs_no_finetuning_train_data_neural_sde_paper.json"
    fim_finetune_json = convergence_base_path / "fim_finetune_sample_mse_epoch_100_train_data_neural_sde_paper.json"

    # contains dict with keys: epochs and values: mse at that epoch
    convergence_base_path = Path(
        "/cephfs_projects/foundation_models/data/SDE/saved_evaluation_results/20250808_neurips_rebuttal_evaluations/lorenz_convergence_speed"
    )
    latent_sde_prior_eq_convergence_json = convergence_base_path / "lorenz_convergence_latent_sde_every_5_epochs_solve_prior_eq.json"
    fim_custom_trained_convergence_json = convergence_base_path / "lorenz_convergence_fim_retrain_every_5_epochs.json"
    fim_finetune_convergence_json = convergence_base_path / "lorenz_convergence_fim_finetune_ever_5_epochs.json"

    convergence_metric = "mmd"  # "mmd" or "paths_mse"
    convergenc_iterations = 1500
    use_fim_view = True
    convergence_log_scale = False

    num_paths = 128  # 128 max

    reference_plot_config = {
        "color": "black",
        "linestyle": "solid",
        "linewidth": 0.05,
    }

    latent_sde_prior_eq_plot_config = {
        "color": "#D55E00",
        "linestyle": "solid",
        # "linewidth": 0.15,
        "linewidth": 0.05,
    }

    fim_no_finetune_plot_config = {
        "color": "#0072B2",
        "linestyle": "solid",
        "linewidth": 0.05,
        # "linewidth": 0.15,
    }

    fim_finetune_plot_config = {
        "color": "#56B4E9",
        "linestyle": "solid",
        "linewidth": 0.05,
        # "linewidth": 0.15,
    }

    fim_custom_trained_plot_config = {
        "color": "#CC79A7",
        "linestyle": "solid",
        # "linewidth": 0.15,
        "linewidth": 0.05,
    }

    # --------------------------------------------------------------------------------------------------------------------------------- #

    # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
    evaluation_path = Path(project_path) / "evaluations"
    time: str = str(datetime.now().strftime("%m%d%H%M"))
    evaluation_dir: Path = evaluation_path / global_description / (time + "_" + current_description)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # load results
    def _extract_paths(json_path: Path, num_paths: int):
        content: list = json.load(open(json_path, "r"))

        if "initial_state_label" in content[0].keys():
            content = [d for d in content if d["initial_state_label"] == "Posterior Init. Cond. from Ref. Set."]

        content = [
            d["synthetic_path"]
            for d in content
            if ((d["train_data_label"] == "neural_sde_paper") and (d["inference_data_label"] == "N(0,1)"))
        ]
        assert len(content) == 1, f"Got {len(content)}"
        return np.array(content[0])[:num_paths]

    reference_paths = np.array(json.load(open(reference_paths_jsons, "r"))["noisy_obs_values"])[:num_paths]
    latent_sde_paths, fim_no_finetune_paths, fim_finetune_paths = optree.tree_map(
        lambda x: _extract_paths(x, num_paths), (latent_sde_json, fim_no_finetune_json, fim_finetune_json)
    )

    fig_grid = plt.Figure(figsize=(6, 1.5), dpi=300, tight_layout=True)
    axs_grid = [fig_grid.add_subplot(1, 4, i + 1, projection="3d" if i < 3 else None) for i in range(4)]

    fig_single, axs_single = [], []
    for i in range(4):
        fig = plt.Figure(figsize=(1.5, 1.5), dpi=300, tight_layout=True)
        ax = fig.add_subplot(1, 1, 1, projection="3d" if i < 3 else None)
        fig_single.append(fig)
        axs_single.append(ax)

    ### Paths of figures
    for axs in [axs_grid, axs_single]:
        for ax in axs[:3]:
            ax.set_axis_off()

        plot_3D_paths(axs[0], reference_paths, label="Ground-truth", **reference_plot_config)
        plot_3D_paths(axs[1], reference_paths, label=None, **reference_plot_config)
        plot_3D_paths(axs[2], reference_paths, label=None, **reference_plot_config)

        plot_3D_paths(axs[0], latent_sde_paths, label="LatentSDE", **latent_sde_prior_eq_plot_config)
        plot_3D_paths(axs[1], fim_no_finetune_paths, label="FIM-SDE", **fim_no_finetune_plot_config)
        plot_3D_paths(axs[2], fim_finetune_paths, label="FIM-SDE Finetuned", **fim_finetune_plot_config)

        if use_fim_view is True:
            axs[0].view_init(azim=axs[1].azim, elev=axs[1].elev)
            axs[0].view_init(azim=axs[1].azim, elev=axs[1].elev)

            axs[0].set_xlim3d(axs[1].get_xlim3d())
            axs[0].set_ylim3d(axs[1].get_ylim3d())
            axs[0].set_zlim3d(axs[1].get_zlim3d())

            axs[2].set_xlim3d(axs[1].get_xlim3d())
            axs[2].set_ylim3d(axs[1].get_ylim3d())
            axs[2].set_zlim3d(axs[1].get_zlim3d())

    ### Convergence speed
    def _load_convergence(
        json_path: Path,
    ):  # contains dict with keys: epoch (value is list of epochs), "mmd" (value is list of mmds), "paths_mse"
        result = json.load(open(json_path, "r"))
        result["epoch"] = np.array(result["epoch"])  # [T]
        result[convergence_metric] = np.array(result[convergence_metric])  # [T]

        sort_indices = np.argsort(result["epoch"])
        result["epoch"] = result["epoch"][sort_indices]
        result[convergence_metric] = result[convergence_metric][sort_indices]

        # truncate iterations
        mask = result["epoch"] <= convergenc_iterations
        result["epoch"] = result["epoch"][mask]
        result[convergence_metric] = result[convergence_metric][mask]

        return result

    latent_sde_prior_eq_convergence, fim_custom_trained_convergence, fim_finetune_convergence = optree.tree_map(
        _load_convergence,
        (latent_sde_prior_eq_convergence_json, fim_custom_trained_convergence_json, fim_finetune_convergence_json),
    )

    for ax in [axs_single[-1], axs_grid[-1]]:
        [x.set_linewidth(0.3) for x in ax.spines.values()]
        ax.tick_params(axis="both", direction="out", labelsize=4, width=0.5, length=2, pad=0.8)
        ax.set_xlim(-50, convergenc_iterations + 50)
        ax.set_xticks([0, convergenc_iterations // 2, convergenc_iterations])

        latent_sde_prior_eq_plot_config["linewidth"] *= 2
        fim_custom_trained_plot_config["linewidth"] *= 2
        fim_finetune_plot_config["linewidth"] *= 2

        if convergence_log_scale is True:
            latent_sde_prior_eq_convergence[convergence_metric] = np.clip(
                latent_sde_prior_eq_convergence[convergence_metric], a_min=1e-8, a_max=np.inf
            )
            fim_custom_trained_convergence[convergence_metric] = np.clip(
                fim_custom_trained_convergence[convergence_metric], a_min=1e-8, a_max=np.inf
            )
            fim_finetune_convergence[convergence_metric] = np.clip(fim_finetune_convergence[convergence_metric], a_min=1e-8, a_max=np.inf)

            ax.set_yscale("log")

        latent_sde_prior_eq_plot_config["linewidth"] = latent_sde_prior_eq_plot_config["linewidth"] * 1.5
        fim_custom_trained_plot_config["linewidth"] = fim_custom_trained_plot_config["linewidth"] * 1.5
        fim_finetune_plot_config["linewidth"] = fim_finetune_plot_config["linewidth"] * 1.5

        ax.plot(
            latent_sde_prior_eq_convergence["epoch"],
            latent_sde_prior_eq_convergence[convergence_metric],
            label="Lat.SDE",
            **latent_sde_prior_eq_plot_config,
        )
        ax.plot(
            fim_custom_trained_convergence["epoch"],
            fim_custom_trained_convergence[convergence_metric],
            label="Retrain",
            **fim_custom_trained_plot_config,
        )
        ax.plot(
            fim_finetune_convergence["epoch"],
            fim_finetune_convergence[convergence_metric],
            label="Finetune",
            **fim_finetune_plot_config,
        )

        ax.set_xlabel("Iteration", fontsize=5)
        ax.set_ylabel("MMD" if convergence_metric == "mmd" else "MSE" if convergence_metric == "paths_mse" else None, fontsize=5)

    # place right legend on top of the plot
    plt.draw()

    obs_handle = Line2D([0], [0], color=reference_plot_config["color"], label="Ground-Truth", linewidth=1)
    latentsde_handle = Line2D([0], [0], color=latent_sde_prior_eq_plot_config["color"], label="LatentSDE", linewidth=1)
    fim_handle = Line2D([0], [0], color=fim_no_finetune_plot_config["color"], label="FIM-SDE (Zero-Shot)", linewidth=1)
    fine_handle = Line2D([0], [0], color=fim_finetune_plot_config["color"], label="FIM-SDE (Finetuned)", linewidth=1)
    fim_custom_handle = Line2D([0], [0], color=fim_custom_trained_plot_config["color"], label="FIM-SDE (Custom Trained)", linewidth=1)

    handles = [obs_handle, latentsde_handle, fim_handle, fine_handle, fim_custom_handle]

    legend_fontsize = 5
    bbox_x = axs_grid[1].get_position().x1 + 0.5 * (axs_grid[2].get_position().x0 - axs_grid[1].get_position().x1)
    bbox_y = axs[1].get_position().y1 * 1.02
    # bbox_y = axs_grid[1].get_position().y1 * 0.8

    legend = fig_grid.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=[bbox_x, bbox_y],
        fontsize=legend_fontsize,
        ncols=5,
    )

    save_dir: Path = evaluation_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    save_fig(fig_grid, save_dir, "all_models_and_convergence")
    plt.close(fig_grid)

    save_fig(fig_single[0], save_dir, "latent_sde_paths")
    plt.close(fig_single[0])

    save_fig(fig_single[1], save_dir, "fim_paths")
    plt.close(fig_single[1])

    save_fig(fig_single[2], save_dir, "fim_finetuned_paths")
    plt.close(fig_single[2])

    save_fig(fig_single[3], save_dir, "convergence_speed_only")
    plt.close(fig_single[3])
