import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optree

from fim import project_path
from fim.utils.sde.evaluation import save_fig


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    global_description = "lorenz_system_figure_paths"

    current_description = "latent_sde_vs_fim_finetuning_vs_fim_from_scratch"

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

    base_path = Path(
        "/cephfs/users/seifner/repos/FIM/saved_evaluations/lorenz_system_vf_and_paths_evaluation/20250701_latent_sde_and_fim_with_vector_fields"
    )

    lat_sde_context_1_base_path = base_path / "07011432_latent_sde_context_1_with_vector_fields/model_paths"
    lat_sde_context_100_base_path = base_path / "07011436_latent_sde_context_100_with_vector_fields/model_paths"
    fim_epochs_200_500_base_path = base_path / "07011423_fim_finetune_epochs_200_500_with_vector_fields/model_paths"
    fim_epochs_1000_2000_base_path = base_path / "07011427_fim_finetune_epochs_1000_2000_with_vector_fields/model_paths"
    fim_no_training_base_path = base_path / "07011430_fim_no_finetune_or_train_from_scratch/model_paths"

    models_jsons = {
        "fim_no_finetuning": fim_no_training_base_path / "fim_model_C_at_139_epochs_no_finetuning_train_data_neural_sde_paper.json",
        "fim_train_from_scratch_epochs_5000_lr_1e-5": fim_no_training_base_path
        / "fim_train_from_scratch_lr_1e-5_train_data_neural_sde_paper.json",
        "fim_finetune_epochs_200_lr_1e-5": fim_epochs_200_500_base_path
        / "fim_finetune_200_epochs_lr_1e-5_train_data_neural_sde_paper.json",
        "fim_finetune_epochs_500_lr_1e-5": fim_epochs_200_500_base_path
        / "fim_finetune_500_epochs_lr_1e-5_train_data_neural_sde_paper.json",
        "fim_finetune_epochs_1000_lr_1e-5": fim_epochs_1000_2000_base_path
        / "fim_finetune_1000_epochs_lr_1e-5_train_data_neural_sde_paper.json",
        "fim_finetune_epochs_2000_lr_1e-5": fim_epochs_1000_2000_base_path
        / "fim_finetune_2000_epochs_lr_1e-5_train_data_neural_sde_paper.json",
        "fim_finetune_epochs_200_lr_1e-6": fim_epochs_200_500_base_path
        / "fim_finetune_200_epochs_lr_1e-6_train_data_neural_sde_paper.json",
        "fim_finetune_epochs_500_lr_1e-6": fim_epochs_200_500_base_path
        / "fim_finetune_500_epochs_lr_1e-6_train_data_neural_sde_paper.json",
        "fim_finetune_epochs_1000_lr_1e-6": fim_epochs_1000_2000_base_path
        / "fim_finetune_1000_epochs_lr_1e-6_train_data_neural_sde_paper.json",
        "fim_finetune_epochs_2000_lr_1e-6": fim_epochs_1000_2000_base_path
        / "fim_finetune_2000_epochs_lr_1e-6_train_data_neural_sde_paper.json",
        "lat_sde_context_1": lat_sde_context_1_base_path / "lat_sde_context_1_train_data_neural_sde_paper.json",
        "lat_sde_context_100": lat_sde_context_100_base_path / "lat_sde_context_100_train_data_neural_sde_paper.json",
        "lat_sde_latent_3_context_1": lat_sde_context_1_base_path / "lat_sde_latent_3_context_1_train_data_neural_sde_paper.json",
        "lat_sde_latent_3_context_100": lat_sde_context_100_base_path / "lat_sde_latent_3_context_100_train_data_neural_sde_paper.json",
        "lat_sde_latent_3_no_proj_context_1": lat_sde_context_1_base_path
        / "lat_sde_latent_3_no_proj_context_1_train_data_neural_sde_paper.json",
        "lat_sde_latent_3_no_proj_context_100": lat_sde_context_100_base_path
        / "lat_sde_latent_3_no_proj_context_100_train_data_neural_sde_paper.json",
    }

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

        save_dir: Path = evaluation_dir / figure_setup["train_data_label"] / figure_setup["inference_data_label"]
        save_dir.mkdir(parents=True, exist_ok=True)
        file_name = figure_setup["model_label"]
        save_fig(fig, save_dir, file_name)

        plt.close(fig)
