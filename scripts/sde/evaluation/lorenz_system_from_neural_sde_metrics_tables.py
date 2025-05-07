import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import optree
import pandas as pd
from metrics_helpers import (
    MetricEvaluation,
    load_metric_evaluations_from_dirs,
    nans_to_stars,
    save_table,
)
from mmd import compute_mmd
from tqdm import tqdm


def get_lorenz_setup_paths(all_paths: list[dict], system_setup: str, initial_state_setup: int) -> np.ndarray:
    """
    Extract paths from one experiment of lorenz data, loaded from json file.

    Args:
        all_paths (list[dict]): List of all datasets, including all setups.
        system_setup, initial_state_setup: Identifying information for paths to extract from all_paths.

    Returns:
        paths_of_dataset (dict): Unique element in all_paths with identifier (system_setup, initial_state_setup).
    """
    paths_of_setup = [d for d in all_paths if d["diffusion_label"] == system_setup and d["initial_state_label"] == initial_state_setup]

    # should contain exactly one set of paths for each setup
    if len(paths_of_setup) == 1:
        return np.array(paths_of_setup[0]["paths"])

    elif len(paths_of_setup) == 0:
        raise ValueError(f"Could not find reference paths of setup {initial_state_setup=}, {system_setup=}.")

    else:
        raise ValueError(f"Found {len(paths_of_setup)} sets of paths for setup {initial_state_setup=}, {system_setup=}.")


def get_model_paths(all_model_data: list[dict], train_data_setup: str, initial_state_setup: str) -> np.ndarray:
    """
    Extract result of one experiment from one model from a loaded json file.

    Args:
        all_model_data (list[dict]): Results of one model on all datasets.
        train_data_setup, initial_state_setup: Identifying information for result to extract from all_model_data.

    Returns:
        model_paths_value (np.ndarray): Extracted model result.
    """
    model_data_of_setup = [
        d
        for d in all_model_data
        if (d["train_data_diffusion_label"] == train_data_setup) and (d["initial_state_label"] == initial_state_setup)
    ]

    # should contain exactly one set of data for each setup
    if len(model_data_of_setup) == 1:
        return np.array(model_data_of_setup[0]["synthetic_path"])

    elif len(model_data_of_setup) == 0:
        return None

    else:
        raise Warning(
            f"Found {len(model_data_of_setup)} sets of data for setup {train_data_setup} with initial_state_setup {initial_state_setup}."
        )


def lorenz_metric_table(
    all_evaluations: list[MetricEvaluation],
    metric: str,
    models_order: list[str],
    initial_state_order: list[str],
    train_data_order: list[str],
    precision: int,
    handle_negative_values: str,
):
    """
    Turn all results of one metric into pandas dataframes, depicting them as mean, standard deviation, mean + std and mean(std).
    """
    assert handle_negative_values in [None, "clip", "abs"]

    # all evaluations with metric to dataframe
    rows = [
        {
            "model": eval.model_id,
            "train_data_diffusion_label": eval.data_id[0],
            "initial_state_label": eval.data_id[1],
            "metric_value": eval.metric_value,
        }
        for eval in all_evaluations
        if eval.metric_id == metric
    ]

    cols = optree.tree_map(lambda *x: x, *rows)
    df = pd.DataFrame.from_dict(cols)

    # models have custom sorting
    df["model"] = pd.Categorical(df["model"], models_order)

    # count Nans and depict them as stars
    df_count_nans = deepcopy(df[["train_data_diffusion_label", "initial_state_label", "model", "metric_value"]])
    df_count_nans["metric_is_nan"] = np.isnan(df["metric_value"])
    df_count_nans = df_count_nans.drop("metric_value", axis=1)

    df_star_nans = df_count_nans.groupby(["train_data_diffusion_label", "initial_state_label", "model"]).agg(nans_to_stars)
    df_star_nans = df_star_nans["metric_is_nan"].unstack(0).unstack(0)

    # sometimes mmd can be negative, if the paths are really good
    if handle_negative_values == "clip":
        df["metric_value"] = np.clip(df["metric_value"], a_min=0.0, a_max=np.inf)

    elif handle_negative_values == "abs":
        df["metric_value"] = np.abs(df["metric_value"])

    # mean without Nans
    df_mean = df.groupby(["train_data_diffusion_label", "initial_state_label", "model"]).agg(
        lambda x: str(x.dropna().mean().round(precision)) if (len(x.dropna()) != 0) else "-"
    )
    df_mean = df_mean["metric_value"].unstack(0).unstack(0)

    # add number of Nan experiments as stars to each cell
    df_mean = df_mean + " " + df_star_nans

    # reorder columns
    column_order = [(t, i) for t in train_data_order for i in initial_state_order]
    df_mean = df_mean[column_order]

    return df_mean


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    dataset_descr = "lorenz_system_metrics_tables"

    # How to name experiments
    experiment_descr = "posterior_of_reference_paths_for_latent_initial_condition_sampled_latent_sdes"
    # experiment_descr = "fim_vs_lorenz_sampled_from_prior_initial_condition"

    project_path = "/cephfs/users/seifner/repos/FIM"

    reference_paths_json = Path(
        "/cephfs_projects/foundation_models/data/SDE/evaluation/20250512_lorenz_data_from_neural_sde_github/20250514221957_lorenz_system_mmd_reference_paths/20250514221957_lorenz_mmd_reference_data.json"
    )

    base_path_latent_sde_latent_dim_3 = Path(
        "/cephfs_projects/foundation_models/data/SDE/evaluation/20250512_lorenz_data_from_neural_sde_github/"
    )
    base_path_no_finetuning = Path(
        "/cephfs/users/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/lorenz_system_vf_and_paths_evaluation/05160055_neurips_model_no_finetuning/model_paths/"
    )
    base_path_finetuned = Path(
        "/cephfs/users/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/lorenz_system_vf_and_paths_evaluation/05160031_neurips_model_finetuning_on_128_paths_up_to_500_epochs/model_paths/"
    )

    models_jsons = {
        # "latent_sde_latent_dim_3_context_16": base_path_latent_sde_latent_dim_3 / "05151624_lorenz-16_paths.json",
        # "latent_sde_latent_dim_3_context_32": base_path_latent_sde_latent_dim_3 / "05151624_lorenz-32_paths.json",
        # "latent_sde_latent_dim_3_context_64": base_path_latent_sde_latent_dim_3 / "05151624_lorenz-64_paths.json",
        "fim_no_finetuning": [
            base_path_no_finetuning / "fim_model_C_at_139_epochs_no_finetuning_train_data_linear_diffusion_num_context_paths_1024.json",
            base_path_no_finetuning / "fim_model_C_at_139_epochs_no_finetuning_train_data_constant_diffusion_num_context_paths_1024.json",
        ],
        # "fim_finetune_on_128_paths_all_points_lr_1e-6_epochs_010": [
        #     base_path_finetuned
        #     / "fim_model_C_at_139_epochs_finetuned_on_128_paths_all_points_lr_1e-6_epochs_010_train_data_linear_diffusion_num_context_paths_1024.json",
        #     base_path_finetuned
        #     / "fim_model_C_at_139_epochs_finetuned_on_128_paths_all_points_lr_1e-6_epochs_010_train_data_constant_diffusion_num_context_paths_1024.json",
        # ],
        # "fim_finetune_on_128_paths_all_points_lr_1e-6_epochs_050": [
        #     base_path_finetuned
        #     / "fim_model_C_at_139_epochs_finetuned_on_128_paths_all_points_lr_1e-6_epochs_050_train_data_linear_diffusion_num_context_paths_1024.json",
        #     base_path_finetuned
        #     / "fim_model_C_at_139_epochs_finetuned_on_128_paths_all_points_lr_1e-6_epochs_050_train_data_constant_diffusion_num_context_paths_1024.json",
        # ],
        # "fim_finetune_on_128_paths_all_points_lr_1e-6_epochs_100": [
        #     base_path_finetuned
        #     / "fim_model_C_at_139_epochs_finetuned_on_128_paths_all_points_lr_1e-6_epochs_100_train_data_linear_diffusion_num_context_paths_1024.json",
        #     base_path_finetuned
        #     / "fim_model_C_at_139_epochs_finetuned_on_128_paths_all_points_lr_1e-6_epochs_100_train_data_constant_diffusion_num_context_paths_1024.json",
        # ],
        "fim_finetune_on_128_paths_all_points_lr_1e-6_epochs_200": [
            base_path_finetuned
            / "fim_model_C_at_139_epochs_finetuned_on_128_paths_all_points_lr_1e-6_epochs_200_train_data_linear_diffusion_num_context_paths_1024.json",
            base_path_finetuned
            / "fim_model_C_at_139_epochs_finetuned_on_128_paths_all_points_lr_1e-6_epochs_200_train_data_constant_diffusion_num_context_paths_1024.json",
        ],
        # "fim_finetune_on_128_paths_all_points_lr_1e-6_epochs_500": [
        #     base_path_finetuned
        #     / "fim_model_C_at_139_epochs_finetuned_on_128_paths_all_points_lr_1e-6_epochs_500_train_data_linear_diffusion_num_context_paths_1024.json",
        #     base_path_finetuned
        #     / "fim_model_C_at_139_epochs_finetuned_on_128_paths_all_points_lr_1e-6_epochs_500_train_data_constant_diffusion_num_context_paths_1024.json",
        # ],
        #
        #
        # "latent_sde_latent_dim_3_context_64_sample_from_prior": base_path_latent_sde_latent_dim_3
        # / "05161005_latent_dim_3_paths_sampled_from_prior_lorenz-64_paths.json",
        # "latent_sde_latent_dim_3_context_32_sample_from_prior": base_path_latent_sde_latent_dim_3
        # / "05161005_latent_dim_3_paths_sampled_from_prior_lorenz-32_paths.json",
        # "latent_sde_latent_dim_3_context_16_sample_from_prior": base_path_latent_sde_latent_dim_3
        # / "05161005_latent_dim_3_paths_sampled_from_prior_lorenz-16_paths.json",
        # "latent_sde_latent_dim_4_context_64_sample_from_prior": base_path_latent_sde_latent_dim_3
        # / "05161006_latent_dim_4_paths_sampled_from_prior_lorenz_64_paths.json",
        # "latent_sde_latent_dim_4_context_32_sample_from_prior": base_path_latent_sde_latent_dim_3
        # / "05161006_latent_dim_4_paths_sampled_from_prior_lorenz_32_paths.json",
        #
        #
        "latent_sde_latent_dim_3_context_64_sample_from_posterior": base_path_latent_sde_latent_dim_3
        / "05161229_latent_dim_3_paths_sampled_from_posterior_initial_condition_from_reference_paths_lorenz-16_paths.json",
        "latent_sde_latent_dim_3_context_32_sample_from_posterior": base_path_latent_sde_latent_dim_3
        / "05161229_latent_dim_3_paths_sampled_from_posterior_initial_condition_from_reference_paths_lorenz-32_paths.json",
        "latent_sde_latent_dim_3_context_16_sample_from_posterior": base_path_latent_sde_latent_dim_3
        / "05161229_latent_dim_3_paths_sampled_from_posterior_initial_condition_from_reference_paths_lorenz-64_paths.json",
        "latent_sde_latent_dim_4_context_64_sample_from_posterior": base_path_latent_sde_latent_dim_3
        / "05161230_latent_dim_4_paths_sampled_from_posterior_initial_condition_from_reference_paths_lorenz_64_paths.json",
        "latent_sde_latent_dim_4_context_32_sample_from_posterior": base_path_latent_sde_latent_dim_3
        / "05161230_latent_dim_4_paths_sampled_from_posterior_initial_condition_from_reference_paths_lorenz_32_paths.json",
    }

    initial_states_setups_to_evaluate = {
        "sampled_normal_mean_0_std_1": r"$\mathcal{N}(0, 1)$",
        "sampled_normal_mean_0_std_2": r"$\mathcal{N}(0, 2)$",
        "fixed_at_1_1_1": r"$(1, 1, 1)$",
    }

    train_data_setups_to_evaluate = {
        "linear": "Linear Diffusion",
        "constant": "Constant Diffusion",
    }
    only_use_loaded_evaluations = False

    metric_evaluations_to_load: list[Path] = [
        Path(
            "/cephfs/users/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/lorenz_system_metrics_tables/05160100_fim_model_C_fixed_softmax_dim_epoch_139_no_finetuning/metric_evaluations_jsons"
        ),
        Path(
            "/cephfs/users/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/lorenz_system_metrics_tables/05160042_fim_model_C_fixed_softmax_dim_epoch_139_finetune_on_128_paths_lr_1e-6_eval_context_1024/metric_evaluations_jsons"
        ),
        # Path(
        #     "/cephfs/users/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/lorenz_system_metrics_tables/05160105_latent_sde_latent_dim_3_context_16-64/metric_evaluations_jsons"
        # ),
        # Path(
        #     "/cephfs/users/seifner/repos/FIM/evaluations/lorenz_system_metrics_tables/05161014_prior_sampled_latent_sdes/metric_evaluations_jsons"
        # ),
    ]

    # tables config
    models_order = tuple(models_jsons.keys())
    initial_state_order = tuple(initial_states_setups_to_evaluate.values())
    train_data_order = tuple(train_data_setups_to_evaluate.values())
    precision = 3

    # --------------------------------------------------------------------------------------------------------------------------------- #

    # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
    evaluation_path = Path(project_path) / "evaluations"
    time: str = str(datetime.now().strftime("%m%d%H%M"))
    evaluation_dir: Path = evaluation_path / dataset_descr / (time + "_" + experiment_descr)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ground-truth and model data.")
    reference_data_paths: list[dict] = json.load(open(reference_paths_json, "r"))

    # some model data is split into multiple jsons
    models_data: dict = {}
    for model_name, model_json_list in models_jsons.items():
        if not isinstance(model_json_list, list):
            model_json_list = [model_json_list]

        model_data = []

        for model_json in model_json_list:
            model_data = model_data + json.load(open(model_json, "r"))

        models_data.update({model_name: model_data})

    # models_data: list[dict] = {model_name: json.load(open(model_json, "r")) for model_name, model_json in models_jsons.items()}

    # prepare all evaluations to be done
    print("Preparing all evaluations.")
    all_evaluations: list[MetricEvaluation] = [
        MetricEvaluation(
            model_id=model,
            model_json=models_jsons[model],
            data_id=(train_data_setup, initial_state_setup),
            data_paths_json=reference_paths_json,
            data_vector_fields_json=None,
            metric_id="mmd",
            metric_value=None,  # input later
        )
        for initial_state_setup in initial_states_setups_to_evaluate.keys()
        for train_data_setup in train_data_setups_to_evaluate
        for model in models_jsons.keys()
    ]

    print("Loading saved evaluations.")
    # load saved evaluations; remove those not already contained in all_evaluations
    loaded_evaluations: list[MetricEvaluation] = load_metric_evaluations_from_dirs(metric_evaluations_to_load)
    loaded_evaluations: list[MetricEvaluation] = [eval for eval in loaded_evaluations if eval in all_evaluations]

    if only_use_loaded_evaluations is False:
        to_evaluate: list[MetricEvaluation] = [eval for eval in all_evaluations if eval not in loaded_evaluations]

    else:
        to_evaluate: list[MetricEvaluation] = []

    # prepare directory to save evaluations in save
    json_save_dir: Path = evaluation_dir / "metric_evaluations_jsons"
    json_save_dir.mkdir(exist_ok=True, parents=True)

    for eval in loaded_evaluations:
        train_data_setup, initial_state_setup = eval.data_id
        file_name = f"mod_{eval.model_id.replace(' ', '_')}_met_{eval.metric_id}_train_data_{train_data_setup}_init_state_{initial_state_setup}.json"
        eval.to_json(json_save_dir / file_name)

    # compute metric for missing evaluations
    all_evaluations = loaded_evaluations
    mmd_ground_truth_cache = {}  # K_xx for the ground truth data is the same for all models, so we cache it

    if len(to_evaluate) > 0:
        print(
            f"Data paths keys (Initial States, Diffusion Label): {[(d['initial_state_label'], d['diffusion_label']) for d in reference_data_paths]}"
        )
        print("Models keys: ")
        for model_name, model_data in models_data.items():
            print(
                f"Model: {model_name}, (Initial States, Train Data): ",
                [(setup["initial_state_label"], setup["train_data_diffusion_label"]) for setup in model_data],
            )

        for eval in (pbar := tqdm(to_evaluate, leave=False)):
            train_data_setup, initial_state_setup = eval.data_id
            pbar.set_description(f"Processing metric={'mmd'}, model={eval.model_id}, {train_data_setup=}, {initial_state_setup=}.")

            start_time = datetime.now()

            reference_paths: np.ndarray = get_lorenz_setup_paths(reference_data_paths, train_data_setup, initial_state_setup)
            model_paths: np.ndarray = get_model_paths(models_data[eval.model_id], train_data_setup, initial_state_setup)

            if model_paths is None or np.isnan(model_paths).any() or np.isnan(reference_paths).any():
                eval.metric_value = np.nan

            else:
                # print("\n\n")
                # print(f"{model_paths.min()=}, {model_paths.max()=}")
                # print(f"{reference_paths.min()=}, {reference_paths.max()=}")
                # print("\n\n")
                print(model_paths.shape)
                eval.metric_value = compute_mmd(reference_paths, model_paths, kernel_cache=mmd_ground_truth_cache)

            # record computation time
            end_time = datetime.now()
            computation_time = (end_time - start_time).total_seconds()

            print(
                f"Last result: metric={eval.metric_id}, value={eval.metric_value}, model={eval.model_id}, {train_data_setup=}, {initial_state_setup=}, {computation_time=} seconds."
            )
            print("\n")

            # save results as json
            file_name = f"mod_{eval.model_id.replace(' ', '_')}_met_{eval.metric_id}_train_data_{train_data_setup}_init_state_{initial_state_setup}.json"
            eval.to_json(json_save_dir / file_name)

            all_evaluations.append(eval)

    # replace initial states and training data labels
    renamed_evaluations = []
    for eval in all_evaluations:
        train_data_setup, initial_state_setup = eval.data_id

        train_data_setup = train_data_setups_to_evaluate[train_data_setup]
        initial_state_setup = initial_states_setups_to_evaluate[initial_state_setup]

        eval.data_id = (train_data_setup, initial_state_setup)

        renamed_evaluations.append(eval)

    for handle_negative_values in [None, "clip", "abs"]:
        df_mean = lorenz_metric_table(
            renamed_evaluations, "mmd", models_order, initial_state_order, train_data_order, precision, handle_negative_values
        )

        subdir_name = "tables_mmd"

        if handle_negative_values is not None:
            subdir_name = subdir_name + "_" + handle_negative_values

        metric_save_dir: Path = evaluation_dir / subdir_name
        metric_save_dir.mkdir(exist_ok=True, parents=True)

        # save_table(df_count_exps, metric_save_dir, "count_experiments")
        save_table(df_mean, metric_save_dir, "mean")
