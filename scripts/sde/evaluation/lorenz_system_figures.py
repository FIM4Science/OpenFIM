import json
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optree

from fim import project_path
from fim.utils.sde.evaluation import save_fig


def plot_3D_quiver(
    ax,
    locations: np.ndarray,
    vector_field: np.ndarray,
    min_distance_to_obs: np.ndarray,
    quiver_length_scale: float,
    cmap,
    quiver_color: str,
    **quiver_config,
) -> None:
    """
    Plot vector field as 3D quiver plot into axis.

    locations (np.ndarray): locations of quivers. Shape: [G, 3]
    vector_fields (np.ndarray): vector field values to plot. Shape: [G, 3]
    min_distance_to_obs (np.ndarray): Per location, minimum distance to a point on some set of paths. Shape: [G, 3]
    quiver_length_scale (float): Lenght of each quiver is proportional to norm of vector field at a location.
    cmap: Cmap for coloring quivers.
    quiver_color (str): Str identifying how to color each quiver.
    quiver_config (dict): kwargs passed to ax.quiver
    """

    l2_norm = np.sqrt((vector_field**2).sum(axis=1))
    quiver_length = l2_norm * quiver_length_scale

    if quiver_color == "location_norm":
        loc_norm = np.sqrt((locations**2).sum(axis=1))
        loc_norm = (loc_norm - np.amin(loc_norm)) / (np.amax(loc_norm) - np.amin(loc_norm))
        color_by = loc_norm

    elif quiver_color == "min_distance_to_obs":
        color_by = min_distance_to_obs

    for i in range(locations.shape[0]):
        c = cmap(color_by[i])
        ax.quiver(
            locations[i, 0],
            locations[i, 1],
            locations[i, 2],
            vector_field[i, 0],
            vector_field[i, 1],
            vector_field[i, 2],
            length=quiver_length[i],
            colors=c,
            alpha=1 - min_distance_to_obs[i],
            **quiver_config,
        )


def subsample_locations(locations: np.ndarray, stride_length: int = 2, points_per_axis: int = 20) -> None:
    """
    Regular subsampling of a flattened 3D grid.

    locations (np.ndarray): Flattened 3D grid. Shape: [points_per_axis * points_per_axis * points_per_axis, 3]
    stride_length (int): Subsampling strides.
    """
    locations = locations.reshape(points_per_axis, points_per_axis, points_per_axis, 3)
    locations = locations[::stride_length, ::stride_length, ::stride_length, :]
    return locations.reshape(-1, 3)


def vector_fields_plot(
    reference_locations: np.ndarray,
    reference_drift: np.ndarray,
    reference_diffusion: np.ndarray,
    reference_paths: np.ndarray,
    model_drift: np.ndarray,
    model_diffusion: np.ndarray,
    model_paths: np.ndarray,
    quiver_color,
):
    """
    Save one plot comparing reference vector fields to model vector field estimations.

    reference_locations: Shape: [G, 3]
    reference/model_/drift/diffusion (np.ndarray): Shape: [G, 3]
    reference/model_paths (np.ndarray): Shape: [P, T, 3]
    """
    model_drift = subsample_locations(model_drift)
    model_diffusion = subsample_locations(model_diffusion)
    reference_locations = subsample_locations(reference_locations)
    reference_drift = subsample_locations(reference_drift)
    reference_diffusion = subsample_locations(reference_diffusion)

    difference_drift = model_drift - reference_drift
    difference_diffusion = model_diffusion - reference_diffusion

    distance_to_obs = np.sqrt(np.sum((reference_paths.reshape(1, -1, 3) - reference_locations.reshape(-1, 1, 3)) ** 2, axis=-1))
    min_distance_to_obs = np.amin(distance_to_obs, axis=1)

    normalized_min_distance_to_obs = (min_distance_to_obs - np.amin(min_distance_to_obs)) / (
        np.amax(min_distance_to_obs) - np.amin(min_distance_to_obs)
    )

    num_rows = 3
    num_cols = 3
    fig, axs = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(num_cols * 2, num_rows * 2),
        dpi=300,
        subplot_kw={"projection": "3d"},
        tight_layout=True,
    )
    fig.suptitle(figure_setup["model_label"], fontsize=7)

    for ax in axs.reshape(-1):
        ax.set_axis_off()

    cmap = mpl.colormaps["viridis"]
    colorbar_label = "NDistance to nearest Observation" if quiver_color == "min_distance_to_obs" else "Location Norm"

    axs[0, 0].set_title("Ground-Truth")
    axs[0, 1].set_title("Model")
    axs[0, 2].set_title("Diff: Model - G.T.")

    plot_3D_quiver(
        axs[0, 0], reference_locations, reference_drift, normalized_min_distance_to_obs, scale_drift, cmap, quiver_color, **quiver_config
    )
    plot_3D_quiver(
        axs[1, 0],
        reference_locations,
        reference_diffusion,
        normalized_min_distance_to_obs,
        scale_diffusion,
        cmap,
        quiver_color,
        **quiver_config,
    )

    for path in range(reference_paths.shape[0]):
        axs[2, 0].plot(reference_paths[path, :, 0], reference_paths[path, :, 1], reference_paths[path, :, 2], linewidth=0.2, color="black")

    plot_3D_quiver(
        axs[0, 1], reference_locations, model_drift, normalized_min_distance_to_obs, scale_drift, cmap, quiver_color, **quiver_config
    )
    plot_3D_quiver(
        axs[1, 1],
        reference_locations,
        model_diffusion,
        normalized_min_distance_to_obs,
        scale_diffusion,
        cmap,
        quiver_color,
        **quiver_config,
    )

    for path in range(reference_paths.shape[0]):
        axs[2, 1].plot(model_paths[path, :, 0], model_paths[path, :, 1], model_paths[path, :, 2], linewidth=0.2, color="blue")

    plot_3D_quiver(
        axs[0, 2], reference_locations, difference_drift, normalized_min_distance_to_obs, scale_drift, cmap, quiver_color, **quiver_config
    )
    plot_3D_quiver(
        axs[1, 2],
        reference_locations,
        difference_diffusion,
        normalized_min_distance_to_obs,
        scale_diffusion,
        cmap,
        quiver_color,
        **quiver_config,
    )

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(0, 1))
    cbar = plt.colorbar(sm, ax=axs[1, 0], ticks=np.linspace(0, 1, 2))
    cbar.set_label(label=colorbar_label, size=7)
    cbar.ax.tick_params(labelsize=7)

    save_dir: Path = evaluation_dir / "vector_fields" / figure_setup["train_data_label"] / ("color_by_" + quiver_color)
    save_dir.mkdir(parents=True, exist_ok=True)
    file_name = figure_setup["model_label"]
    save_fig(fig, save_dir, file_name)

    plt.close(fig)


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    global_description = "lorenz_system_figures"

    # current_description = "latent_sde_vs_fim_finetuning_vs_fim_from_scratch_bit_longer_quivers"
    # current_description = "fim_finetune_on_NLL_and_from_scratch_with_weight_decay"
    current_description = "fim_finetune_on_sampling"

    neural_sde_paper_path = Path(
        "/home/seifner/repos/FIM/data/processed/test/20250629_lorenz_system_with_vector_fields_at_locations/neural_sde_paper/set_0/"
    )
    neural_sde_github_path = Path(
        "/home/seifner/repos/FIM/data/processed/test/20250629_lorenz_system_with_vector_fields_at_locations/neural_sde_github/set_0/"
    )

    reference_paths_jsons = {
        "neural_sde_paper": {
            "(1,1,1)": neural_sde_paper_path / "(1,1,1)_reference_data.json",
            "N(0,1)": neural_sde_paper_path / "N(0,1)_reference_data.json",
            "N(0,2)": neural_sde_paper_path / "N(0,2)_reference_data.json",
        },
        "neural_sde_github": {
            "(1,1,1)": neural_sde_github_path / "(1,1,1)_reference_data.json",
            "N(0,1)": neural_sde_github_path / "N(0,1)_reference_data.json",
            "N(0,2)": neural_sde_github_path / "N(0,2)_reference_data.json",
        },
    }

    #### Finetuned on NLL

    # base_path = Path(
    #     "/cephfs/users/seifner/repos/FIM/saved_evaluations/lorenz_system_vf_and_paths_evaluation/20250701_latent_sde_and_fim_with_vector_fields"
    # )

    # lat_sde_context_1_base_path = base_path / "07011432_latent_sde_context_1_with_vector_fields/model_paths"
    # lat_sde_context_100_base_path = base_path / "07011436_latent_sde_context_100_with_vector_fields/model_paths"
    # fim_epochs_200_500_base_path = base_path / "07011423_fim_finetune_epochs_200_500_with_vector_fields/model_paths"
    # fim_epochs_1000_2000_base_path = base_path / "07011427_fim_finetune_epochs_1000_2000_with_vector_fields/model_paths"
    # fim_no_training_base_path = base_path / "07011430_fim_no_finetune_or_train_from_scratch/model_paths"
    #
    # models_jsons = {
    #     "fim_no_finetuning": fim_no_training_base_path / "fim_model_C_at_139_epochs_no_finetuning_train_data_neural_sde_paper.json",
    #     "fim_train_from_scratch_epochs_5000_lr_1e-5": fim_no_training_base_path
    #     / "fim_train_from_scratch_lr_1e-5_train_data_neural_sde_paper.json",
    #     "fim_finetune_epochs_200_lr_1e-5": fim_epochs_200_500_base_path
    #     / "fim_finetune_200_epochs_lr_1e-5_train_data_neural_sde_paper.json",
    #     "fim_finetune_epochs_500_lr_1e-5": fim_epochs_200_500_base_path
    #     / "fim_finetune_500_epochs_lr_1e-5_train_data_neural_sde_paper.json",
    #     "fim_finetune_epochs_1000_lr_1e-5": fim_epochs_1000_2000_base_path
    #     / "fim_finetune_1000_epochs_lr_1e-5_train_data_neural_sde_paper.json",
    #     "fim_finetune_epochs_2000_lr_1e-5": fim_epochs_1000_2000_base_path
    #     / "fim_finetune_2000_epochs_lr_1e-5_train_data_neural_sde_paper.json",
    #     "fim_finetune_epochs_200_lr_1e-6": fim_epochs_200_500_base_path
    #     / "fim_finetune_200_epochs_lr_1e-6_train_data_neural_sde_paper.json",
    #     "fim_finetune_epochs_500_lr_1e-6": fim_epochs_200_500_base_path
    #     / "fim_finetune_500_epochs_lr_1e-6_train_data_neural_sde_paper.json",
    #     "fim_finetune_epochs_1000_lr_1e-6": fim_epochs_1000_2000_base_path
    #     / "fim_finetune_1000_epochs_lr_1e-6_train_data_neural_sde_paper.json",
    #     "fim_finetune_epochs_2000_lr_1e-6": fim_epochs_1000_2000_base_path
    #     / "fim_finetune_2000_epochs_lr_1e-6_train_data_neural_sde_paper.json",
    #     "lat_sde_context_1": lat_sde_context_1_base_path / "lat_sde_context_1_train_data_neural_sde_paper.json",
    #     "lat_sde_context_100": lat_sde_context_100_base_path / "lat_sde_context_100_train_data_neural_sde_paper.json",
    #     "lat_sde_latent_3_context_1": lat_sde_context_1_base_path / "lat_sde_latent_3_context_1_train_data_neural_sde_paper.json",
    #     "lat_sde_latent_3_context_100": lat_sde_context_100_base_path / "lat_sde_latent_3_context_100_train_data_neural_sde_paper.json",
    #     "lat_sde_latent_3_no_proj_context_1": lat_sde_context_1_base_path
    #     / "lat_sde_latent_3_no_proj_context_1_train_data_neural_sde_paper.json",
    #     "lat_sde_latent_3_no_proj_context_100": lat_sde_context_100_base_path
    #     / "lat_sde_latent_3_no_proj_context_100_train_data_neural_sde_paper.json",
    # }

    ### With Weight decay
    # evaluation_base_path = Path("/cephfs/users/seifner/repos/FIM/saved_evaluations/lorenz_system_vf_and_paths_evaluation")
    # weigh_decay_base_path = evaluation_base_path / "20250701_latent_sde_and_fim_with_vector_fields"
    # lat_sde_context_100_base_path = (
    #     evaluation_base_path
    #     / "20250701_latent_sde_and_fim_with_vector_fields"
    #     / "07011436_latent_sde_context_100_with_vector_fields/model_paths"
    # )
    # fim_epochs_1000_2000_base_path = (
    #     evaluation_base_path
    #     / "20250701_latent_sde_and_fim_with_vector_fields"
    #     / "07011427_fim_finetune_epochs_1000_2000_with_vector_fields/model_paths"
    # )
    # fim_no_training_base_path = (
    #     evaluation_base_path
    #     / "20250701_latent_sde_and_fim_with_vector_fields"
    #     / "07011430_fim_no_finetune_or_train_from_scratch/model_paths"
    # )
    #
    # models_jsons = {
    #     "fim_no_finetuning": fim_no_training_base_path / "fim_model_C_at_139_epochs_no_finetuning_train_data_neural_sde_paper.json",
    #     "fim_finetune_epochs_1000_lr_1e-5": fim_epochs_1000_2000_base_path
    #     / "fim_finetune_1000_epochs_lr_1e-5_train_data_neural_sde_paper.json",
    #     "lat_sde_latent_3_no_proj_context_100": lat_sde_context_100_base_path
    #     / "lat_sde_latent_3_no_proj_context_100_train_data_neural_sde_paper.json",
    #     "fim_finetune_1000_epochs_lr_1e-5_weight_decay_1e-4": weigh_decay_base_path
    #     / "07081716_fim_finetune_1024_paths_512_points_1000_epochs_lr_1e-5_weight_decay_1e-4/model_paths/fim_finetune_1024_paths_512_points_1000_epochs_lr_1e-5_weight_decay_1e-4_train_data_neural_sde_paper.json",
    #     "fim_finetune_1000_epochs_lr_1e-5_weight_decay_1e-3": weigh_decay_base_path
    #     / "07081731_fim_finetune_1024_paths_512_points_1000_epochs_lr_1e-5_weight_decay_1e-3/model_paths/fim_finetune_1024_paths_512_points_1000_epochs_lr_1e-5_weight_decay_1e-3_train_data_neural_sde_paper.json",
    #     "fim_finetune_5000_epochs_lr_1e-5_weight_decay_1e-4": weigh_decay_base_path
    #     / "07081731_fim_finetune_1024_paths_512_points_1000_epochs_lr_1e-5_weight_decay_1e-3/model_paths/fim_finetune_1024_paths_512_points_1000_epochs_lr_1e-5_weight_decay_1e-3_train_data_neural_sde_paper.json",
    #     "fim_retrain_from_scratch_5000_epochs_lr_1e-5_weight_decay_1e-4": weigh_decay_base_path
    #     / "07081913_fim_finetune_1024_paths_512_points_5000_epochs_lr_1e-5_weight_decay_1e-4_from_scratch/model_paths/fim_finetune_1024_paths_512_points_5000_epochs_lr_1e-5_weight_decay_1e-4_from_scratch_train_data_neural_sde_paper.json",
    #     "fim_retrain_from_scratch_5000_epochs_lr_1e-5_weight_decay_1e-5": weigh_decay_base_path
    #     / "07081922_fim_finetune_1024_paths_512_points_5000_epochs_lr_1e-5_weight_decay_1e-5_from_scratch/model_paths/fim_finetune_1024_paths_512_points_5000_epochs_lr_1e-5_weight_decay_1e-5_from_scratch_train_data_neural_sde_paper.json",
    # }

    ### Finetuned on Sampling
    evaluation_base_path = Path("/cephfs/users/seifner/repos/FIM/saved_evaluations/lorenz_system_vf_and_paths_evaluation")
    sampling_base_path = evaluation_base_path / "20250709_fim_finetune_on_sampling"
    lat_sde_context_100_base_path = (
        evaluation_base_path
        / "20250701_latent_sde_and_fim_with_vector_fields"
        / "07011436_latent_sde_context_100_with_vector_fields/model_paths"
    )
    fim_epochs_1000_2000_base_path = (
        evaluation_base_path
        / "20250701_latent_sde_and_fim_with_vector_fields"
        / "07011427_fim_finetune_epochs_1000_2000_with_vector_fields/model_paths"
    )
    fim_no_training_base_path = (
        evaluation_base_path
        / "20250701_latent_sde_and_fim_with_vector_fields"
        / "07011430_fim_no_finetune_or_train_from_scratch/model_paths"
    )

    models_jsons = {
        "fim_no_finetuning": fim_no_training_base_path / "fim_model_C_at_139_epochs_no_finetuning_train_data_neural_sde_paper.json",
        "fim_currently_training": sampling_base_path / "fim_currently_training_train_data_neural_sde_paper.json",
        "fim_finetune_on_NLL_epochs_1000_lr_1e-5": fim_epochs_1000_2000_base_path
        / "fim_finetune_1000_epochs_lr_1e-5_train_data_neural_sde_paper.json",
        "lat_sde_latent_3_no_proj_context_100": lat_sde_context_100_base_path
        / "lat_sde_latent_3_no_proj_context_100_train_data_neural_sde_paper.json",
        "fim_finetune_on_sampling_500_epochs_1_sample_1_step_ahead_1_em_step": sampling_base_path
        / "07071245_fim_finetune_on_sampling_1024_points_500_epochs_lr_1e-5_32_points_1_sample_1_step_ahead_1_em_step/model_paths/fim_finetune_on_sampling_1024_points_500_epochs_lr_1e-5_32_points_1_sample_1_step_ahead_1_em_step_train_data_neural_sde_paper.json",
        "fim_finetune_on_sampling_500_epochs_10_samples_1_step_ahead_1_em_step": sampling_base_path
        / "07071252_fim_finetune_on_sampling_1024_points_500_epochs_lr_1e-5_32_points_10_samples_1_step_ahead_1_em_step/model_paths/fim_finetune_on_sampling_1024_points_500_epochs_lr_1e-5_32_points_10_samples_1_step_ahead_1_em_step_train_data_neural_sde_paper.json",
        "fim_finetune_on_sampling_500_epochs_1_sample_10_step_ahead_1_em_step": sampling_base_path
        / "07071308_fim_finetune_on_sampling_1024_points_500_epochs_lr_1e-5_32_points_1_sample_10_step_ahead_1_em_step/model_paths/fim_finetune_on_sampling_1024_points_500_epochs_lr_1e-5_32_points_1_sample_10_step_ahead_1_em_step_train_data_neural_sde_paper.json",
        "fim_finetune_on_sampling_500_epochs_1_sample_1_step_ahead_10_em_step": sampling_base_path
        / "07071324_fim_finetune_on_sampling_1024_points_500_epochs_lr_1e-5_32_points_1_sample_1_step_ahead_10_em_step/model_paths/fim_finetune_on_sampling_1024_points_500_epochs_lr_1e-5_32_points_1_sample_1_step_ahead_10_em_step_train_data_neural_sde_paper.json",
        "fim_finetune_on_sampling_500_epochs_1_sample_5_step_ahead_5_em_step": sampling_base_path
        / "07071357_fim_finetune_on_sampling_1024_points_500_epochs_lr_1e-5_32_points_1_sample_5_step_ahead_5_em_step/model_paths/fim_finetune_on_sampling_1024_points_500_epochs_lr_1e-5_32_points_1_sample_5_step_ahead_5_em_step_train_data_neural_sde_paper.json",
        "fim_finetune_on_sampling_500_epochs_1_sample_5_step_ahead_5_em_step_detach_diffusion": sampling_base_path
        / "07071420_fim_finetune_on_sampling_1024_points_500_epochs_lr_1e-5_32_points_1_sample_5_step_ahead_5_em_step_detach_diffusion/model_paths/fim_finetune_on_sampling_1024_points_500_epochs_lr_1e-5_32_points_1_sample_5_step_ahead_5_em_step_detach_diffusion_train_data_neural_sde_paper.json",
        "fim_retrain_from_scratch_on_sampling_1_sample_1_step_ahead_10_em_step": sampling_base_path
        / "07081819_fim_finetune_on_sampling_1024_points_5000_epochs_lr_1e-5_32_points_1_sample_1_step_ahead_10_em_step_retrain_from_scratch/model_paths/fim_finetune_on_sampling_1024_points_5000_epochs_lr_1e-5_32_points_1_sample_1_step_ahead_10_em_step_retrain_from_scratch_train_data_neural_sde_paper.json",
        "fim_finetune_on_sampling_100_epochs_1_sample_5_step_ahead_5_em_step": sampling_base_path
        / "fim_1_sample_5_steps_ahead_5_em_steps_epoch_100_train_data_neural_sde_paper.json",
        "fim_finetune_on_sampling_200_epochs_1_sample_5_step_ahead_5_em_step": sampling_base_path
        / "fim_1_sample_5_steps_ahead_5_em_steps_epoch_200_train_data_neural_sde_paper.json",
    }

    plot_paths = True
    plot_vector_fields = True

    # paths
    gt_plot_config = {
        "color": "black",
        "linestyle": "solid",
        "linewidth": 0.2,
    }

    model_plot_config = {
        "color": "#0072B2",
        "linestyle": "solid",
        "linewidth": 0.2,
    }

    # vector fields
    quiver_config = {
        "linewidths": 0.4,
        "pivot": "middle",
    }
    scale_drift = 0.0003
    scale_diffusion = 1

    # --------------------------------------------------------------------------------------------------------------------------------- #

    # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
    evaluation_path = Path(project_path) / "evaluations"
    time: str = str(datetime.now().strftime("%m%d%H%M"))
    evaluation_dir: Path = evaluation_path / global_description / (time + "_" + current_description)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # load ground-truth synthetic paths
    print("Loading ground-truth and model data.")
    reference_data: dict = optree.tree_map(lambda x: json.load(open(x, "r")), reference_paths_jsons)
    models_data: dict = optree.tree_map(lambda x: json.load(open(x, "r")), models_jsons)

    # per train_data_label, inference_data_label, model_label show (cols) sampling, (rows), initial_state_label
    figure_setups: list[dict] = [
        {
            "model_label": model_label,
            # "sampling_label": model_paths.get("sampling_label"),
            # "initial_state_label": model_paths.get("initial_state_label"),
            "train_data_label": model_paths["train_data_label"],
            "inference_data_label": model_paths["inference_data_label"],
        }
        for model_label, model_data in models_data.items()
        for model_paths in model_data
    ]

    if plot_paths is True:
        for figure_setup in figure_setups:
            print(f"Processing {figure_setup}")
            figure_model_data: list = [
                d
                for d in models_data[figure_setup["model_label"]]
                if (
                    d["train_data_label"] == figure_setup["train_data_label"]
                    and d["inference_data_label"] == figure_setup["inference_data_label"]
                )
            ]

            sampling_labels = list({d.get("sampling_label") for d in figure_model_data})
            initial_state_labels = list({d.get("initial_state_label") for d in figure_model_data})

            # create figure with separating gap between the two systems
            num_rows = len(initial_state_labels)
            num_cols = len(sampling_labels)
            fig, axs = plt.subplots(
                nrows=num_rows,
                ncols=num_cols,
                figsize=(num_cols * 2, num_rows * 2),
                dpi=300,
                subplot_kw={"projection": "3d"},
                tight_layout=True,
            )
            fig.suptitle(figure_setup["model_label"], fontsize=7)

            if num_rows == 1 and num_cols == 1:
                axs = np.array([[axs]])

            elif num_rows == 1:
                axs = axs.reshape(1, -1)

            elif num_cols == 1:
                axs = axs.reshape(-1, 1)

            # configure axes general
            for ax in axs.reshape(-1):
                ax.set_axis_off()

            for row in range(num_rows):
                for col in range(num_cols):
                    initial_state_label = initial_state_labels[row]
                    sampling_label = sampling_labels[col]

                    axs[row, col].set_title(f"Init. State Sampling: {initial_state_label} \n Equation: {sampling_label}", fontsize=6)

                    reference_paths = reference_data[figure_setup["train_data_label"]][figure_setup["inference_data_label"]]
                    reference_paths = np.array(reference_paths["clean_obs_values"])

                    model_paths = [
                        d
                        for d in figure_model_data
                        if d.get("initial_state_label") == initial_state_label and d.get("sampling_label") == sampling_label
                    ]
                    if len(model_paths) == 0:
                        print(
                            f"Model path {initial_state_label=}, {sampling_label} not found in {[d['initial_state_label'] for d in figure_model_data]}, {[d['sampling_label'] for d in figure_model_data]}."
                        )
                    elif len(model_paths) > 1:
                        raise ValueError(
                            f"Found multiple paths of figure setup {optree.tree_map(lambda x: x.shape if isinstance(x, np.ndarray) else x, figure_setup)}"
                        )
                    else:
                        model_paths = np.array(model_paths[0]["synthetic_path"])

                        assert reference_paths.shape == model_paths.shape
                        for i in range(reference_paths.shape[0]):
                            axs[row, col].plot(
                                reference_paths[i, :, 0],
                                reference_paths[i, :, 1],
                                reference_paths[i, :, 2],
                                **gt_plot_config,
                                label="Reference" if i == 0 else None,
                            )
                            axs[row, col].plot(
                                model_paths[i, :, 0],
                                model_paths[i, :, 1],
                                model_paths[i, :, 2],
                                **model_plot_config,
                                label="Model" if i == 0 else None,
                            )

            plt.draw()
            handles, labels = axs[0, 0].get_legend_handles_labels()

            legend = fig.legend(fontsize=5)

            save_dir: Path = evaluation_dir / "paths" / figure_setup["train_data_label"] / figure_setup["inference_data_label"]
            save_dir.mkdir(parents=True, exist_ok=True)
            file_name = figure_setup["model_label"]
            save_fig(fig, save_dir, file_name)

            plt.close(fig)

    if plot_vector_fields is True:
        # per model (including train data label) vector fields for all inference_data_label (i.e. initial states for paths) are the same
        figure_setups = [d for d in figure_setups if d["inference_data_label"] == "N(0,1)"]

        for figure_setup in figure_setups:
            figure_model_data: list = [
                d
                for d in models_data[figure_setup["model_label"]]
                if (
                    d["train_data_label"] == figure_setup["train_data_label"]
                    and d["inference_data_label"] == figure_setup["inference_data_label"]
                )
            ]

            if len(figure_model_data) == 0:
                print(
                    f"Model vector fields {initial_state_label=}, {sampling_label} not found in {[d['initial_state_label'] for d in figure_model_data]}, {[d['sampling_label'] for d in figure_model_data]}."
                )
            else:
                # multiple sampling strategies for Latent SDE, but vector fields (if available) are all the same -> take 0th
                if figure_model_data[0]["drift_at_locations"] is not None:
                    model_drift = np.array(figure_model_data[0]["drift_at_locations"])  # [G, 3]
                    model_diffusion = np.array(figure_model_data[0]["diffusion_at_locations"])  # [G, 3]
                    model_paths = np.array(figure_model_data[0]["synthetic_path"])  # [B, T, 3]

                    print(figure_setup)

                else:
                    continue  # can only process certain latent sdes

            reference_data_for_figure = reference_data[figure_setup["train_data_label"]][figure_setup["inference_data_label"]]
            reference_locations = np.array(reference_data_for_figure["locations"])
            reference_drift = np.array(reference_data_for_figure["drift_at_locations"])
            reference_diffusion = np.array(reference_data_for_figure["diffusion_at_locations"])
            reference_paths = np.array(reference_data_for_figure["clean_obs_values"])

            vector_fields_plot(
                reference_locations,
                reference_drift,
                reference_diffusion,
                reference_paths,
                model_drift,
                model_diffusion,
                model_paths,
                "min_distance_to_obs",
            )
            vector_fields_plot(
                reference_locations,
                reference_drift,
                reference_diffusion,
                reference_paths,
                model_drift,
                model_diffusion,
                model_paths,
                "location_norm",
            )
