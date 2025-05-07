import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fim import project_path
from fim.utils.evaluation_sde import save_fig


def get_model_results_on_splits(
    models_results: dict[str, list], dataset: str, apply_sqrt_to_diffusion: list, expected_num_total_splits: int
) -> dict[str, list]:
    """
    Extracts results of all models on a particular dataset.
    Converts loaded lists into arrays.
    If inference was on log-transformed data, reverse the transform.
    """
    models_results_of_dataset = {}

    for model_label, results in models_results.items():
        # should extract expected number of split result for each model
        results_of_dataset = [r for r in results if r["name"] == dataset]
        assert len(results_of_dataset) == expected_num_total_splits, f"Expected {expected_num_total_splits}, got {len(results_of_dataset)}."

        transformed_split_results = []

        for split_result in results_of_dataset:
            # for sanity, check loaded data has expected number of splits
            assert split_result["num_total_splits"] == expected_num_total_splits, (
                f"Expected {expected_num_total_splits}, got {split_result['num_total_splits']}."
            )

            # convert to arrays
            split_result["locations"] = np.array(split_result["locations"])
            split_result["synthetic_paths"] = np.array(split_result["synthetic_paths"])
            split_result["drift_at_locations"] = np.array(split_result["drift_at_locations"])
            split_result["diffusion_at_locations"] = np.array(split_result["diffusion_at_locations"])

            if model_label in apply_sqrt_to_diffusion:
                split_result["diffusion_at_locations"] = np.sqrt(np.max(split_result["diffusion_at_locations"], 0))

            # reverse log-transform
            if split_result.get("transform") == "log":
                split_result["locations"] = np.exp(split_result["locations"])
                split_result["synthetic_paths"] = np.exp(split_result["synthetic_paths"])

                # ito formula applied to f(x) =df(x) = ddf(x) = exp(x)
                split_result["drift_at_locations"] = split_result["locations"] * (
                    split_result["drift_at_locations"] + 1 / 2 * split_result["diffusion_at_locations"] ** 2
                )
                split_result["diffusion_at_locations"] = split_result["locations"] * split_result["diffusion_at_locations"]

            transformed_split_results.append(split_result)

        models_results_of_dataset.update({model_label: transformed_split_results})

    return models_results_of_dataset


def get_reference_data_splits(data_paths: list, dataset: str, expected_num_total_splits: int) -> list[dict]:
    """
    From all data, extract all splits of the dataset.
    Converts loaded lists into arrays.
    If log-transformed data, reverse the transform.
    """
    # should extract exactly one set of data
    reference_data = [d for d in data_paths if d["name"] == dataset]
    assert len(reference_data) == expected_num_total_splits, f"Expected {expected_num_total_splits}, got {len(reference_data)}."

    transformed_reference_data = []

    for split_data in reference_data:
        # for sanity, check loaded data has expected number of splits
        assert split_data["num_total_splits"] == expected_num_total_splits, (
            f"Expected {expected_num_total_splits}, got {split_data['num_total_splits']}."
        )

        # convert to arrays
        split_data["locations"] = np.array(split_data["locations"])
        split_data["obs_times"] = np.array(split_data["obs_times"])
        split_data["obs_values"] = np.array(split_data["obs_values"])

        # reverse log-transform
        if split_data.get("transform") == "log":
            split_data["locations"] = np.exp(split_data["locations"])
            split_data["obs_values"] = np.exp(split_data["obs_values"])

        transformed_reference_data.append(split_data)

    return transformed_reference_data


def get_split_from_reference_data(reference_data: list[dict], split: int) -> dict:
    """
    From all splits of one dataset, extract particular split.
    """
    reference_split_data = [d for d in reference_data if d["split"] == split]
    assert len(reference_split_data) == 1, f"Expected 1, got {len(reference_split_data)}."
    reference_split_data = reference_split_data[0]

    return reference_split_data


def get_split_from_models_results(models_results: dict[str, list], split: int) -> dict[str, dict]:
    """
    From all models results on one dataset, extract particular split for each of them
    """
    models_split_result = {}

    for model_label, model_results in models_results.items():
        split_results = [d for d in model_results if d["split"] == split]
        assert len(split_results) == 1, f"Expected 1, got {len(split_results)}."
        split_results = split_results[0]

        models_split_result.update({model_label: split_results})

    return models_split_result


def get_sample_paths_figure(
    reference_data: list[dict],
    models_results: dict[str, list],
    models_color: dict,
    plot_at_obs_times: bool,
    title: str = "",
    figsize=(10, 10),
    linewidth=0.5,
    reference_label="Reference Paths",
    initial_state_color="red",
):
    """
    Plot (set of) reference sample paths against paths from all models.
    """

    num_total_splits = len(reference_data)

    fig, ax = plt.subplots(nrows=num_total_splits, ncols=len(models_results) + 1, figsize=figsize, dpi=300)

    for split in range(num_total_splits):
        ax[split, 0].set_ylabel(f"Split: {split}")

        # extract reference data
        reference_split_data: dict = get_split_from_reference_data(reference_data, split)
        obs_times = reference_split_data["obs_times"]  # [1, P, T, 1]
        ref_obs_values = reference_split_data["obs_values"]  # [1, P, T, 1]
        assert obs_times.shape == ref_obs_values.shape
        assert obs_times.ndim == 4

        # optionally reset times per path to 0
        if plot_at_obs_times:
            times = obs_times

        else:
            times = obs_times - obs_times[:, :, 0, :][:, :, None, :]

        P = ref_obs_values.shape[1]

        # plot reference paths (with same initial states) in first column, each split into separate rows
        for p in range(P):
            ax[split, 0].plot(times[:, p].squeeze(), ref_obs_values[:, p].squeeze(), color="black", linewidth=linewidth)

        # optionally mark initial states for clarity
        if plot_at_obs_times is True:
            ax[split, 0].scatter(
                times[:, :, 0, :].squeeze(), ref_obs_values[:, :, 0, :].squeeze(), color=initial_state_color, s=(4 * linewidth)
            )

        # extract results of current split per model
        models_split_results: dict[str, dict] = get_split_from_models_results(models_results, split)

        # plot model paths into columns, each split into separate rows
        for model_num, (label, result) in enumerate(models_split_results.items()):
            color = models_color[label]
            synthetic_paths = result["synthetic_paths"]  # [1, P, T, 1]

            assert times.shape == synthetic_paths.shape  # [1, P, T, 1]

            for p in range(P):
                ax[split, model_num + 1].plot(
                    times[:, p].squeeze(), synthetic_paths[:, p].squeeze(), label=label, color=color, linewidth=linewidth
                )

            if split == 0:
                ax[0, model_num + 1].set_title(label)

            # optionally mark initial states for clarity
            if plot_at_obs_times is True:
                ax[split, model_num + 1].scatter(
                    times[:, :, 0, :].squeeze(), synthetic_paths[:, :, 0, :].squeeze(), color=initial_state_color, s=(4 * linewidth)
                )

    ax[0, 0].set_title(reference_label)
    fig.suptitle(title)

    return fig


def get_vector_fields_figure(
    models_results: dict[str, list],
    models_color: dict,
    title="",
    figsize=(10, 10),
    dpi=300,
    linewidth=2,
):
    """
    Plot infered vector fields from all models and all splits.
    """
    num_total_splits = len(list(models_results.values())[0])

    fig, ax = plt.subplots(num_total_splits, 2, figsize=figsize, dpi=dpi)

    ax[0, 0].set_title("Drift")
    ax[0, 1].set_title("Diffusion")

    # plot models vector fields, each split into separate rows
    for split in range(num_total_splits):
        ax[split, 0].set_ylabel(f"Split: {split}")

        models_split_results: dict[str, dict] = get_split_from_models_results(models_results, split)

        for label, model_results in models_split_results.items():
            color = models_color[label]

            locations = model_results["locations"]  # [1, L, 1]
            drift = model_results["drift_at_locations"]  # [1, L, 1]
            diffusion = model_results["diffusion_at_locations"]  # [1, L, 1]

            ax[split, 0].plot(locations.squeeze(), drift.squeeze(), label=label, color=color, linewidth=linewidth)
            ax[split, 1].plot(locations.squeeze(), diffusion.squeeze(), label=label, color=color, linewidth=linewidth)

    ax[0, 0].legend()

    fig.suptitle(title)

    return fig


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    dataset_descr = "real_world_cross_validation_figures"

    # How to name experiments
    # experiment_descr = "fim_checkpoints_vs_BISDE"
    # experiment_descr = "fim_and_BISDE_vs_ablation_models"
    experiment_descr = "fim_fixed_linear_vs_fixed_softmax_different_epochs"

    reference_data_json = Path(
        "/home/seifner/repos/FIM/data/processed/test/20250506_real_world_with_5_fold_cross_validation/cross_val_ksig_reference_paths.json"
    )

    expected_num_total_splits = 5

    datasets_to_evaluate: list[str] = [
        "wind",
        "oil",
        "fb",
        "tsla",
    ]

    # models_jsons = {
    #     "FIM (05-03-2033)": Path(
    #         "/home/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/real_world_cross_validation_vf_and_paths_evaluation/05102037_fim_fixed_attn_fixed_softmax_05-03-2033/model_paths.json",
    #     ),
    #     "FIM (05-06-2300)": Path(
    #         "/home/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/real_world_cross_validation_vf_and_paths_evaluation/05102124_fim_fixed_attn_fixed_softmax_05-06-2300/model_paths.json",
    #     ),
    # "BISDE(20250514, BISDE Library Functions)": Path(
    #     "/cephfs_projects/foundation_models/data/SDE/evaluation/20250506_real_world_with_5_fold_cross_validation/20250514_bisde_5_fold_cross_validation_paths_no_nans/bisde_real_world_cv_results.json"
    # ),
    # "BISDE(20250514, (u^(0,..,3), exp(u), sin(u)) Library Functions)": Path(
    #     "/cephfs_projects/foundation_models/data/SDE/evaluation/20250506_real_world_with_5_fold_cross_validation/20250514_bisde_5_fold_cross_validation_paths_no_nans/bisde_real_world_cv_our_basis_results.json"
    # ),
    #     # "BISDE(20250510, BISDE Library Functions)": Path(
    #     #     "/cephfs_projects/foundation_models/data/SDE/evaluation/20250506_real_world_with_5_fold_cross_validation/20250510_bisde_5_fold_cross_validation_function_library_from_bisde_paper/bisde_real_world_cv_results.json"
    #     # ),
    #     # "BISDE(20250510, (u^(0,..,3), exp(u), sin(u)) Library Functions)": Path(
    #     #     "/cephfs_projects/foundation_models/data/SDE/evaluation/20250506_real_world_with_5_fold_cross_validation/20250510_bisde_5_fold_cross_validation_function_library_with_exps_and_sins/bisde_real_world_cv_our_basis_results.json"
    #     # ),
    #     # "Ablation: train size 30k, 5M params": Path(
    #     #     "/home/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/real_world_cross_validation_vf_and_paths_evaluation/05121921_ablation_model_train_size_30k/model_paths.json"
    #     # ),
    #     # "Ablation: train size 100k, 10M params": Path(
    #     #     "/home/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/real_world_cross_validation_vf_and_paths_evaluation/05121914_ablation_model_train_size_100k/model_paths.json"
    #     # ),
    #     "Ablation: train size 600k, 20M params": Path(
    #         "/home/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/real_world_cross_validation_vf_and_paths_evaluation/05121902_ablation_model_train_size_600k/model_paths.json"
    #     ),
    #     # "Ablation: train size 30k with degree 4 drift, 5M params": Path(
    #     #     "/home/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/real_world_cross_validation_vf_and_paths_evaluation/05121927_ablation_model_degree_4_drift/model_paths.json"
    #     # ),
    # }
    models_jsons = {
        "BISDE(20250514, BISDE Library Functions)": Path(
            "/cephfs_projects/foundation_models/data/SDE/evaluation/20250506_real_world_with_5_fold_cross_validation/20250514_bisde_5_fold_cross_validation_paths_no_nans/bisde_real_world_cv_results.json"
        ),
        "BISDE(20250514, (u^(0,..,3), exp(u), sin(u)) Library Functions)": Path(
            "/cephfs_projects/foundation_models/data/SDE/evaluation/20250506_real_world_with_5_fold_cross_validation/20250514_bisde_5_fold_cross_validation_paths_no_nans/bisde_real_world_cv_our_basis_results.json"
        ),
        "FIM fixed linear Attn., 04-28-0941, Epoch 040": Path(
            "/home/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/real_world_cross_validation_vf_and_paths_evaluation/05132202_fim_fixed_linear_attn_04-28-0941_epoch_040/model_paths.json",
        ),
        "FIM fixed linear Attn., 04-28-0941, Epoch 070": Path(
            "/home/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/real_world_cross_validation_vf_and_paths_evaluation/05132215_fim_fixed_linear_attn_04-28-0941_epoch_070/model_paths.json",
        ),
        "FIM fixed linear Attn., 04-28-0941, Epoch 100": Path(
            "/home/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/real_world_cross_validation_vf_and_paths_evaluation/05132227_fim_fixed_linear_attn_04-28-0941_epoch_100/model_paths.json",
        ),
        "FIM fixed linear Attn., 04-28-0941, Epoch 125": Path(
            "/home/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/real_world_cross_validation_vf_and_paths_evaluation/05132353_fim_fixed_linear_attn_04-28-0941_epoch_125/model_paths.json",
        ),
        "FIM fixed Softmax dim., 05-03-2033, Epoch 040": Path(
            "/home/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/real_world_cross_validation_vf_and_paths_evaluation/05140004_fim_fixed_softmax_05-03-2033_epoch_040/model_paths.json",
        ),
        "FIM fixed Softmax dim., 05-03-2033, Epoch 070": Path(
            "/home/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/real_world_cross_validation_vf_and_paths_evaluation/05140019_fim_fixed_softmax_05-03-2033_epoch_070/model_paths.json",
        ),
        "FIM fixed Softmax dim., 05-03-2033, Epoch 100": Path(
            "/home/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/real_world_cross_validation_vf_and_paths_evaluation/05140033_fim_fixed_softmax_05-03-2033_epoch_100/model_paths.json",
        ),
        "FIM fixed Softmax dim., 05-03-2033, Epoch 125": Path(
            "/home/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/real_world_cross_validation_vf_and_paths_evaluation/05140045_fim_fixed_softmax_05-03-2033_epoch_125/model_paths.json",
        ),
        "FIM fixed Softmax dim., 05-03-2033, Epoch 138": Path(
            "/home/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/real_world_cross_validation_vf_and_paths_evaluation/05140056_fim_fixed_softmax_05-03-2033_epoch_138/model_paths.json",
        ),
    }

    # models_color = {
    #     "FIM (05-03-2033)": "#0072B2",
    #     "FIM (05-06-2300)": "blue",
    #     "BISDE(20250510, BISDE Library Functions)": "#CC79A7",
    #     "BISDE(20250510, (u^(0,..,3), exp(u), sin(u)) Library Functions)": "magenta",
    #     "Ablation: train size 30k, 5M params": "green",
    #     "Ablation: train size 100k, 10M params": "yellow",
    #     "Ablation: train size 600k, 20M params": "brown",
    #     "Ablation: train size 30k with degree 4 drift, 5M params": "red",
    # }

    models_color = {
        "FIM fixed linear Attn., 04-28-0941, Epoch 040": "#0072B2",
        "FIM fixed linear Attn., 04-28-0941, Epoch 070": "blue",
        "FIM fixed linear Attn., 04-28-0941, Epoch 100": "#CC79A7",
        "FIM fixed linear Attn., 04-28-0941, Epoch 125": "magenta",
        "FIM fixed Softmax dim., 05-03-2033, Epoch 040": "green",
        "FIM fixed Softmax dim., 05-03-2033, Epoch 070": "yellow",
        "FIM fixed Softmax dim., 05-03-2033, Epoch 100": "brown",
        "FIM fixed Softmax dim., 05-03-2033, Epoch 125": "red",
        "FIM fixed Softmax dim., 05-03-2033, Epoch 138": "grey",
        "BISDE(20250514, BISDE Library Functions)": "lightblue",
        "BISDE(20250514, (u^(0,..,3), exp(u), sin(u)) Library Functions)": "orange",
    }

    apply_sqrt_to_diffusion = [
        "BISDE(20250510, BISDE Library Functions)",
        "BISDE(20250510, (u^(0,..,3), exp(u), sin(u)) Library Functions)",
        "BISDE(20250514, BISDE Library Functions)",
        "BISDE(20250514, (u^(0,..,3), exp(u), sin(u)) Library Functions)",
    ]

    # --------------------------------------------------------------------------------------------------------------------------------- #

    # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
    evaluation_path = Path(project_path) / "evaluations"
    time: str = str(datetime.now().strftime("%m%d%H%M"))
    evaluation_dir: Path = evaluation_path / dataset_descr / (time + "_" + experiment_descr)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # load models results from jsons
    models_results = {model_label: json.load(open(model_json, "r")) for model_label, model_json in models_jsons.items()}

    # load reference data paths
    data_paths: list[dict] = json.load(open(reference_data_json, "r"))

    for dataset in datasets_to_evaluate:
        reference_data_splits: list[dict] = get_reference_data_splits(deepcopy(data_paths), dataset, expected_num_total_splits)
        models_results_of_splits: dict[str, list] = get_model_results_on_splits(
            deepcopy(models_results), dataset, apply_sqrt_to_diffusion, expected_num_total_splits
        )

        # sample paths figure with consecutive obs times
        fig = get_sample_paths_figure(reference_data_splits, models_results_of_splits, models_color, plot_at_obs_times=True, title=dataset)

        save_dir: Path = evaluation_dir / "figures_sample_paths_all_splits_consecutive_obs_times"
        save_dir.mkdir(parents=True, exist_ok=True)
        file_name = dataset
        save_fig(fig, save_dir, file_name)

        plt.close(fig)

        # sample paths figure with obs times reset to 0
        fig = get_sample_paths_figure(reference_data_splits, models_results_of_splits, models_color, plot_at_obs_times=False, title=dataset)

        save_dir: Path = evaluation_dir / "figures_sample_paths_all_splits_reset_obs_times"
        save_dir.mkdir(parents=True, exist_ok=True)
        file_name = dataset
        save_fig(fig, save_dir, file_name)

        plt.close(fig)

        # vector fields figure
        fig = get_vector_fields_figure(models_results_of_splits, models_color, title=dataset)

        save_dir: Path = evaluation_dir / "figures_vector_fields_all_splits"
        save_dir.mkdir(parents=True, exist_ok=True)
        file_name = dataset
        save_fig(fig, save_dir, file_name)
