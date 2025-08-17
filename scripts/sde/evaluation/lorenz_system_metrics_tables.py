import json
from datetime import datetime
from pathlib import Path

import numpy as np
import optree
import pandas as pd
from metrics_helpers import (
    MetricEvaluation,
    load_metric_evaluations_from_dirs,
    save_table,
)
from mmd import compute_mmd
from tqdm import tqdm


def get_mse(target: np.ndarray, estimation: np.ndarray):
    """
    Return MSE between estimation and target.

    Args: target, estimation (np.ndarray): Shape: [G, D]
    Returns: mse np.ndarrayTensor): Shape: []
    """
    assert estimation.ndim == target.ndim == 2

    se = (estimation - target) ** 2
    mse = np.mean(se)

    return mse


def get_lorenz_data(all_data, train_data_label, inference_data_label) -> np.ndarray:
    """
    Extract data from one experiment of lorenz data, loaded from json file.

    Args:
        all_data (dict): all datasets, including all setups.
        ... label: Identifying information for paths to extract from all_paths.

    Returns:
        paths, drift, diffusion of setup
    """
    data_of_setup = all_data[train_data_label][inference_data_label]
    return (
        np.array(data_of_setup["clean_obs_values"]),
        np.array(data_of_setup["drift_at_locations"]),
        np.array(data_of_setup["diffusion_at_locations"]),
    )


def get_model_data(
    all_model_data: list[dict], train_data_label: str, inference_data_label: str, sampling_label: str, initial_state_label: str | None
) -> np.ndarray:
    """
    Extract result of one experiment from one model from a loaded json file.

    Args:
        all_model_data (list[dict]): Results of one model on all datasets.
        ...label (str): Identifying information for result to extract from all_model_data.

    Returns:
        model_paths_value (np.ndarray): Extracted model sampled paths.
        model_drift (np.ndarray): Extracted model drift at passed locations.
        model_diffusion (np.ndarray): Extracted model diffusion at passed locations.
    """
    model_data = [
        d
        for d in all_model_data
        if (
            d["train_data_label"] == train_data_label
            and (d["inference_data_label"] == inference_data_label)
            and (d["sampling_label"] == sampling_label)
            and (d.get("initial_state_label") == initial_state_label)
        )
    ]

    if len(model_data) == 1:
        return model_data[0]

    elif len(model_data) == 0:
        return None

    else:
        raise Warning(
            f"Found {len(model_data)} sets of data for {train_data_label=}, {inference_data_label=}, {sampling_label=}, {initial_state_label=}."
        )


def get_model_hyperparams(
    *keys,
    model_json: Path,
    train_data_label: str,
    inference_data_label: str,
    sampling_label: str,
    initial_state_label: str | None,
):
    model_data: dict = json.load(open(model_json, "r"))
    model_data = get_model_data(model_data, train_data_label, inference_data_label, sampling_label, initial_state_label)
    return {key: model_data.get(key) for key in keys}


def lorenz_metric_table(
    all_evaluations: list[MetricEvaluation],
    metric: str,
    precision: int,
    handle_negative_values: str,
):
    """
    Turn all results of one metric into pandas dataframes, depicting them as mean, standard deviation, mean + std and mean(std).
    """
    assert handle_negative_values in [None, "clip", "abs"]

    rows = [
        {
            "model": eval.model_id[0],
            "sampling_label": eval.model_id[1],
            "initial_state_label": eval.model_id[2],
            "train_data_label": eval.data_id[0],
            "inference_data_label": eval.data_id[1],
            "metric_value": eval.metric_value,
        }
        | get_model_hyperparams(
            "learn_projection",
            "latent_size",
            "hidden_size",
            "context_size",
            "activation",
            model_json=eval.model_json,
            train_data_label=eval.data_id[0],
            inference_data_label=eval.data_id[1],
            sampling_label=eval.model_id[1],
            initial_state_label=eval.model_id[2],
        )
        for eval in all_evaluations
        if eval.metric_id == metric
    ]

    cols = optree.tree_map(lambda *x: x, *rows, none_is_leaf=True)
    df = pd.DataFrame.from_dict(cols)

    # sometimes mmd can be negative, if the paths are really good
    if handle_negative_values == "clip":
        df["metric_value"] = np.clip(df["metric_value"], a_min=0.0, a_max=np.inf)

    elif handle_negative_values == "abs":
        df["metric_value"] = np.abs(df["metric_value"])

    df = df.groupby(
        [
            "learn_projection",
            "activation",
            "latent_size",
            "hidden_size",
            "context_size",
            "model",
            "sampling_label",
            "initial_state_label",
            "train_data_label",
            "inference_data_label",
        ],
        dropna=False,
        as_index=False,
    ).mean()

    return df


####### Rebuttal experiments

# newly trained FIM on additional locations at observations
rebuttal_base_path = Path(
    "/cephfs_projects/foundation_models/data/SDE/saved_evaluation_results/20250808_neurips_rebuttal_evaluations/lorenz_system_vf_and_paths_evaluation/"
)
NEW_FIM_MODELS_JSONS = {
    "fim_locs_at_obs_epoch_139_no_finetuning": rebuttal_base_path
    / "20250716_fim_location_at_obs_epoch_139_no_finetuning/model_paths/fim_locs_at_obs_no_finetuning_train_data_neural_sde_paper.json"
}

# Sampling + NLL based finetuning, repeated 10 times
sampling_10_seeds_base_path = Path(
    "/cephfs_projects/foundation_models/data/SDE/saved_evaluation_results/20250808_not_used_in_neurips_rebuttal/lorenz_system_vf_and_paths_evaluation/20250717_fim_finetune_on_sampling_nll_10_seeds/"
)

SAMPLING_NLL_10_SEEDS_MODELS_JSONS = {
    "fim_finetune_sampling_nll_500_epochs_10_em_steps_seed_0": sampling_10_seeds_base_path
    / "07161833_fim_finetune_on_sampling_nll_500_epochs_lr_1e-5_32_points_1_sample_1_step_ahead_10_em_step_seed_0/model_paths/fim_finetune_on_sampling_nll_500_epochs_lr_1e-5_32_points_1_sample_1_step_ahead_10_em_step_seed_0_train_data_neural_sde_paper.json",
    "fim_finetune_sampling_nll_500_epochs_10_em_steps_seed_1": sampling_10_seeds_base_path
    / "07161851_fim_finetune_on_sampling_nll_500_epochs_lr_1e-5_32_points_1_sample_1_step_ahead_10_em_step_seed_1/model_paths/fim_finetune_on_sampling_nll_500_epochs_lr_1e-5_32_points_1_sample_1_step_ahead_10_em_step_seed_1_train_data_neural_sde_paper.json",
    "fim_finetune_sampling_nll_500_epochs_10_em_steps_seed_2": sampling_10_seeds_base_path
    / "07161908_fim_finetune_on_sampling_nll_500_epochs_lr_1e-5_32_points_1_sample_1_step_ahead_10_em_step_seed_2/model_paths/fim_finetune_on_sampling_nll_500_epochs_lr_1e-5_32_points_1_sample_1_step_ahead_10_em_step_seed_2_train_data_neural_sde_paper.json",
    "fim_finetune_sampling_nll_500_epochs_10_em_steps_seed_3": sampling_10_seeds_base_path
    / "07161925_fim_finetune_on_sampling_nll_500_epochs_lr_1e-5_32_points_1_sample_1_step_ahead_10_em_step_seed_3/model_paths/fim_finetune_on_sampling_nll_500_epochs_lr_1e-5_32_points_1_sample_1_step_ahead_10_em_step_seed_3_train_data_neural_sde_paper.json",
    "fim_finetune_sampling_nll_500_epochs_10_em_steps_seed_4": sampling_10_seeds_base_path
    / "07161942_fim_finetune_on_sampling_nll_500_epochs_lr_1e-5_32_points_1_sample_1_step_ahead_10_em_step_seed_4/model_paths/fim_finetune_on_sampling_nll_500_epochs_lr_1e-5_32_points_1_sample_1_step_ahead_10_em_step_seed_4_train_data_neural_sde_paper.json",
    "fim_finetune_sampling_nll_500_epochs_10_em_steps_seed_5": sampling_10_seeds_base_path
    / "07162000_fim_finetune_on_sampling_nll_500_epochs_lr_1e-5_32_points_1_sample_1_step_ahead_10_em_step_seed_5/model_paths/fim_finetune_on_sampling_nll_500_epochs_lr_1e-5_32_points_1_sample_1_step_ahead_10_em_step_seed_5_train_data_neural_sde_paper.json",
    "fim_finetune_sampling_nll_500_epochs_10_em_steps_seed_6": sampling_10_seeds_base_path
    / "07162017_fim_finetune_on_sampling_nll_500_epochs_lr_1e-5_32_points_1_sample_1_step_ahead_10_em_step_seed_6/model_paths/fim_finetune_on_sampling_nll_500_epochs_lr_1e-5_32_points_1_sample_1_step_ahead_10_em_step_seed_6_train_data_neural_sde_paper.json",
    "fim_finetune_sampling_nll_500_epochs_10_em_steps_seed_7": sampling_10_seeds_base_path
    / "07162034_fim_finetune_on_sampling_nll_500_epochs_lr_1e-5_32_points_1_sample_1_step_ahead_10_em_step_seed_7/model_paths/fim_finetune_on_sampling_nll_500_epochs_lr_1e-5_32_points_1_sample_1_step_ahead_10_em_step_seed_7_train_data_neural_sde_paper.json",
    "fim_finetune_sampling_nll_500_epochs_10_em_steps_seed_8": sampling_10_seeds_base_path
    / "07162051_fim_finetune_on_sampling_nll_500_epochs_lr_1e-5_32_points_1_sample_1_step_ahead_10_em_step_seed_8/model_paths/fim_finetune_on_sampling_nll_500_epochs_lr_1e-5_32_points_1_sample_1_step_ahead_10_em_step_seed_8_train_data_neural_sde_paper.json",
    "fim_finetune_sampling_nll_500_epochs_10_em_steps_seed_9": sampling_10_seeds_base_path
    / "07162108_fim_finetune_on_sampling_nll_500_epochs_lr_1e-5_32_points_1_sample_1_step_ahead_10_em_step_seed_9/model_paths/fim_finetune_on_sampling_nll_500_epochs_lr_1e-5_32_points_1_sample_1_step_ahead_10_em_step_seed_9_train_data_neural_sde_paper.json",
    "fim_finetune_sampling_nll_500_epochs_10_em_steps_seed_10": sampling_10_seeds_base_path
    / "07162125_fim_finetune_on_sampling_nll_500_epochs_lr_1e-5_32_points_1_sample_1_step_ahead_10_em_step_seed_10/model_paths/fim_finetune_on_sampling_nll_500_epochs_lr_1e-5_32_points_1_sample_1_step_ahead_10_em_step_seed_10_train_data_neural_sde_paper.json",
}

# Finetune vs retrain convergence
conv_base_path = rebuttal_base_path / "20250715_fim_finetune_vs_retrain_convergence_comparison_mse_and_nll/model_paths"

CONVERGENCE_MODELS_JSONS = {
    "finetune_mse_epoch_0050": conv_base_path / "fim_finetune_sample_mse_epoch_50_train_data_neural_sde_paper.json",
    "finetune_mse_epoch_0100": conv_base_path / "fim_finetune_sample_mse_epoch_100_train_data_neural_sde_paper.json",
    "finetune_mse_epoch_0200": conv_base_path / "fim_finetune_sample_mse_epoch_200_train_data_neural_sde_paper.json",
    "finetune_mse_epoch_0500": conv_base_path / "fim_finetune_sample_mse_epoch_500_train_data_neural_sde_paper.json",
    "finetune_mse_epoch_1000": conv_base_path / "fim_finetune_sample_mse_epoch_1000_train_data_neural_sde_paper.json",
    "finetune_mse_epoch_2000": conv_base_path / "fim_finetune_sample_mse_epoch_2000_train_data_neural_sde_paper.json",
    "finetune_mse_epoch_5000": conv_base_path / "fim_finetune_sample_mse_epoch_5000_train_data_neural_sde_paper.json",
    "finetune_nll_epoch_0050": conv_base_path / "fim_finetune_sample_nll_epoch_50_train_data_neural_sde_paper.json",
    "finetune_nll_epoch_0100": conv_base_path / "fim_finetune_sample_nll_epoch_100_train_data_neural_sde_paper.json",
    "finetune_nll_epoch_0200": conv_base_path / "fim_finetune_sample_nll_epoch_200_train_data_neural_sde_paper.json",
    "finetune_nll_epoch_0500": conv_base_path / "fim_finetune_sample_nll_epoch_500_train_data_neural_sde_paper.json",
    "finetune_nll_epoch_1000": conv_base_path / "fim_finetune_sample_nll_epoch_1000_train_data_neural_sde_paper.json",
    "finetune_nll_epoch_2000": conv_base_path / "fim_finetune_sample_nll_epoch_2000_train_data_neural_sde_paper.json",
    "finetune_nll_epoch_5000": conv_base_path / "fim_finetune_sample_nll_epoch_5000_train_data_neural_sde_paper.json",
    "retrain_mse_epoch_0050": conv_base_path / "fim_retrain_sample_mse_epoch_50_train_data_neural_sde_paper.json",
    "retrain_mse_epoch_0100": conv_base_path / "fim_retrain_sample_mse_epoch_100_train_data_neural_sde_paper.json",
    "retrain_mse_epoch_0200": conv_base_path / "fim_retrain_sample_mse_epoch_200_train_data_neural_sde_paper.json",
    "retrain_mse_epoch_0500": conv_base_path / "fim_retrain_sample_mse_epoch_500_train_data_neural_sde_paper.json",
    "retrain_mse_epoch_1000": conv_base_path / "fim_retrain_sample_mse_epoch_1000_train_data_neural_sde_paper.json",
    "retrain_mse_epoch_2000": conv_base_path / "fim_retrain_sample_mse_epoch_2000_train_data_neural_sde_paper.json",
    "retrain_mse_epoch_5000": conv_base_path / "fim_retrain_sample_mse_epoch_5000_train_data_neural_sde_paper.json",
    "retrain_nll_epoch_0050": conv_base_path / "fim_retrain_sample_nll_epoch_50_train_data_neural_sde_paper.json",
    "retrain_nll_epoch_0100": conv_base_path / "fim_retrain_sample_nll_epoch_100_train_data_neural_sde_paper.json",
    "retrain_nll_epoch_0200": conv_base_path / "fim_retrain_sample_nll_epoch_200_train_data_neural_sde_paper.json",
    "retrain_nll_epoch_0500": conv_base_path / "fim_retrain_sample_nll_epoch_500_train_data_neural_sde_paper.json",
    "retrain_nll_epoch_1000": conv_base_path / "fim_retrain_sample_nll_epoch_1000_train_data_neural_sde_paper.json",
    "retrain_nll_epoch_2000": conv_base_path / "fim_retrain_sample_nll_epoch_2000_train_data_neural_sde_paper.json",
    "retrain_nll_epoch_5000": conv_base_path / "fim_retrain_sample_nll_epoch_5000_train_data_neural_sde_paper.json",
}

# Preliminary Neurips rebuttal comparisons
evaluation_base_path = Path(
    "/cephfs_projects/foundation_models/data/SDE/saved_evaluation_results/20250808_not_used_in_neurips_rebuttal/lorenz_system_vf_and_paths_evaluation/"
)
lat_sde_context_100_base_path = (
    rebuttal_base_path / "20250701_latent_sde_and_fim_with_vector_fields/07011436_latent_sde_context_100_with_vector_fields/model_paths"
)
fim_no_training_base_path = rebuttal_base_path / "07011430_fim_no_finetune_or_train_from_scratch/model_paths"
fim_epochs_200_500_base_path = rebuttal_base_path / "07011423_fim_finetune_epochs_200_500_with_vector_fields/model_paths"

PRELIMINARY_REBUTTAL_MODELS_JSONS = {
    "lat_sde_context_100": lat_sde_context_100_base_path / "lat_sde_context_100_train_data_neural_sde_paper.json",
    "lat_sde_latent_3_no_proj_context_100": lat_sde_context_100_base_path
    / "lat_sde_latent_3_no_proj_context_100_train_data_neural_sde_paper.json",
    "fim_no_finetuning": fim_no_training_base_path / "fim_model_C_at_139_epochs_no_finetuning_train_data_neural_sde_paper.json",
    "fim_locs_at_obs_no_finetuning": rebuttal_base_path
    / "20250716_fim_location_at_obs_epoch_139_no_finetuning/model_paths/fim_locs_at_obs_no_finetuning_train_data_neural_sde_paper.json",
    "fim_finetune_epochs_500_lr_1e-5_one_step_ahead_NLL": fim_epochs_200_500_base_path
    / "fim_finetune_500_epochs_lr_1e-5_train_data_neural_sde_paper.json",
    "fim_finetune_epochs_100_lr_1e-5_one_step_ahead_MSE_10_em_steps": conv_base_path
    / "fim_finetune_sample_mse_epoch_100_train_data_neural_sde_paper.json",
    "fim_finetune_epochs_500_lr_1e-5_one_step_ahead_MSE_10_em_steps": conv_base_path
    / "fim_finetune_sample_mse_epoch_500_train_data_neural_sde_paper.json",
    "fim_finetune_epochs_100_lr_1e-5_one_step_ahead_NLL_10_em_steps": conv_base_path
    / "fim_finetune_sample_nll_epoch_100_train_data_neural_sde_paper.json",
    "fim_finetune_epochs_500_lr_1e-5_one_step_ahead_NLL_10_em_steps": conv_base_path
    / "fim_finetune_sample_nll_epoch_500_train_data_neural_sde_paper.json",
}


#### Post neurips convergence speed comparison for figure in 100 stepsize grid
convergence_base_path = Path(
    "/cephfs_projects/foundation_models/data/SDE/saved_evaluation_results/20250808_neurips_rebuttal_evaluations/lorenz_system_vf_and_paths_evaluation/20250820_fim_finetune_vs_retrain_convergence_comparison_mse_every_100_epochs/"
)

conv_lat_sde_path = (
    convergence_base_path / "08200856_latent_sde_paper_setup_record_every_100_epochs_for_convergence_speed_comparison/model_paths"
)
conv_fim_finetune_base_path = (
    convergence_base_path
    / "08201339_fim_finetune_on_sampling_mse_5000_epochs_lr_1e-5_32_points_1_sample_1_step_ahead_10_em_step_convergence_speed_post_neurips/model_paths"
)
conv_fim_retrain_base_path = (
    convergence_base_path
    / "08201722_fim_from_scratch_on_sampling_mse_5000_epochs_lr_1e-5_32_points_1_sample_1_step_ahead_10_em_step_convergence_speed_post_neurips/model_paths"
)

epochs = [99 + i * 100 for i in range(49)]
epochs = [str(e).zfill(4) for e in epochs]  # 0099 til 4899

CONVERGENCE_EVERY_100_LATENT_SDE_JSONS = {
    f"latent_sde_epoch_{epoch}": conv_lat_sde_path / f"develop_sampling_epoch_{epoch}_train_data_neural_sde_paper.json" for epoch in epochs
}

CONVERGENCE_EVERY_100_FIM_FINETUNE_JSONS = {
    f"fim_finetune_epoch_{epoch}": conv_fim_finetune_base_path
    / f"fim_finetune_on_sampling_mse_5000_epochs_lr_1e-5_32_points_1_sample_1_step_ahead_10_em_step_convergence_speed_post_neurips_epoch_{epoch}_train_data_neural_sde_paper.json"
    for epoch in epochs
}

CONVERGENCE_EVERY_100_FIM_RETRAIN_JSONS = {
    f"fim_retrain_epoch_{epoch}": conv_fim_retrain_base_path
    / f"fim_from_scratch_on_sampling_mse_5000_epochs_lr_1e-5_32_points_1_sample_1_step_ahead_10_em_step_convergence_speed_post_neurips_epoch_{epoch}_train_data_neural_sde_paper.json"
    for epoch in epochs
}

CONVERGENCE_EVERY_100_MODEL_JSONS = (
    CONVERGENCE_EVERY_100_LATENT_SDE_JSONS | CONVERGENCE_EVERY_100_FIM_FINETUNE_JSONS | CONVERGENCE_EVERY_100_FIM_RETRAIN_JSONS
)

#### Post neurips convergence speed comparison for figure in 5 stepsize grid
convergence_base_path = Path(
    "/cephfs_projects/foundation_models/data/SDE/saved_evaluation_results/20250808_neurips_rebuttal_evaluations/lorenz_convergence_speed/sampled_paths"
)
conv_lat_sde_prior_eq_path = (
    convergence_base_path / "08210828_latent_sde_paper_setup_record_every_5_epochs_convergence_speed_post_neurips/model_paths"
)
conv_lat_sde_posterior_eq_path = (
    convergence_base_path
    / "08221217_latent_sde_paper_setup_posterior_equation_record_every_5_epochs_convergence_speed_post_neurips/model_paths"
)
conv_fim_finetune_base_path = (
    convergence_base_path
    / "08210827_fim_finetune_on_sampling_mse_5000_epochs_lr_1e-5_32_points_1_sample_1_step_ahead_10_em_step_record_every_5_epochs_convergence_speed_post_neurips/model_paths"
)
conv_fim_retrain_base_path = (
    convergence_base_path
    / "08210825_fim_from_scratch_on_sampling_mse_5000_epochs_lr_1e-5_32_points_1_sample_1_step_ahead_10_em_step_every_5_epochs_convergence_speed_post_neurips/model_paths"
)

epochs = [4 + i * 5 for i in range(999)]
epochs = [str(e).zfill(4) for e in epochs]  # 0004 til 4899

CONVERGENCE_EVERY_5_LATENT_SDE_PRIOR_EQ_JSONS = {
    f"latent_sde_prior_eq_epoch_{epoch}": conv_lat_sde_prior_eq_path
    / f"latent_sde_paper_setup_record_every_5_epochs_convergence_speed_post_neurips_epoch_{epoch}_train_data_neural_sde_paper.json"
    for epoch in epochs
}

CONVERGENCE_EVERY_5_LATENT_SDE_POSTERIOR_EQ_JSONS = {
    f"latent_sde_posterior_eq_epoch_{epoch}": conv_lat_sde_posterior_eq_path
    / f"latent_sde_paper_setup_posterior_equation_record_every_5_epochs_convergence_speed_post_neurips_epoch_{epoch}_train_data_neural_sde_paper.json"
    for epoch in epochs
}

CONVERGENCE_EVERY_5_FIM_FINETUNE_JSONS = {
    f"fim_finetune_epoch_{epoch}": conv_fim_finetune_base_path
    / f"fim_finetune_on_sampling_mse_5000_epochs_lr_1e-5_32_points_1_sample_1_step_ahead_10_em_step_record_every_5_epochs_convergence_speed_post_neurips_epoch_{epoch}_train_data_neural_sde_paper.json"
    for epoch in epochs
}

CONVERGENCE_EVERY_5_FIM_RETRAIN_JSONS = {
    f"fim_retrain_epoch_{epoch}": conv_fim_retrain_base_path
    / f"fim_from_scratch_on_sampling_mse_5000_epochs_lr_1e-5_32_points_1_sample_1_step_ahead_10_em_step_every_5_epochs_convergence_speed_post_neurips_epoch_{epoch}_train_data_neural_sde_paper.json"
    for epoch in epochs
}

CONVERGENCE_EVERY_5_MODEL_JSONS = (
    CONVERGENCE_EVERY_5_LATENT_SDE_PRIOR_EQ_JSONS
    | CONVERGENCE_EVERY_5_LATENT_SDE_POSTERIOR_EQ_JSONS
    | CONVERGENCE_EVERY_5_FIM_FINETUNE_JSONS
    | CONVERGENCE_EVERY_5_FIM_RETRAIN_JSONS
)

if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    dataset_descr = "lorenz_system_metrics_tables"

    # How to name experiments
    # experiment_descr = "fim_finetune_vs_retrain_convergence_speed"
    # experiment_descr = "fim_locs_at_obs_epoch_139_no_finetuning"
    # experiment_descr = "fim_finetune_on_sampling_nll_10_seeds"
    # experiment_descr = "preliminary_rebuttal_comparison_all_models"
    experiment_descr = "convergence_speed_every_5_steps_latent_sde_posterior_and_prior_eq_fim_finetuned_and_retrained"

    models_jsons = CONVERGENCE_EVERY_5_MODEL_JSONS

    metrics = ["mmd", "mse_drift", "mse_diffusion", "mse_paths"]

    project_path = "/cephfs/users/seifner/repos/FIM"
    neural_sde_paper_path = Path("/cephfs_projects/foundation_models/data/SDE/test/20250629_lorenz_systems/neural_sde_paper/set_0/")
    neural_sde_github_path = Path("/cephfs_projects/foundation_models/data/SDE/test/20250629_lorenz_systems/neural_sde_github/set_0/")

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

    only_use_loaded_evaluations = False

    metric_evaluations_to_load: list[Path] = [
        # Path(
        #     "/cephfs/users/seifner/repos/FIM/evaluations/lorenz_system_metrics_tables/06171745_latent_sde_vs_fim_finetuning_first_comparisons/metric_evaluations_jsons"
        # ),
    ]

    # tables config
    precision = 10

    # --------------------------------------------------------------------------------------------------------------------------------- #

    # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
    evaluation_path = Path(project_path) / "evaluations"
    time: str = str(datetime.now().strftime("%m%d%H%M"))
    evaluation_dir: Path = evaluation_path / dataset_descr / (time + "_" + experiment_descr)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    print("Loading ground-truth and model data.")
    reference_data: dict = optree.tree_map(lambda x: json.load(open(x, "r")), reference_paths_jsons)
    models_data: dict = optree.tree_map(lambda x: json.load(open(x, "r")), models_jsons)

    # prepare all evaluations to be done
    print("Preparing all evaluations.")
    all_evaluations: list[MetricEvaluation] = [
        MetricEvaluation(
            model_id=(model_label, model_paths.get("sampling_label"), model_paths.get("initial_state_label")),
            model_json=models_jsons[model_label],
            data_id=(model_paths["train_data_label"], model_paths["inference_data_label"]),
            data_paths_json=None,  # reference_paths_jsons[model_paths["train_data_label"]][model_paths["inference_data_label"]],
            data_vector_fields_json=None,
            metric_id=metric,
            metric_value=None,  # input later
        )
        for metric in metrics
        for model_label, model_data in models_data.items()
        for model_paths in model_data
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
        train_data_label, inference_data_label = eval.data_id
        file_name = f"model_{str(eval.model_id).replace(' ', '_')}_data_{str(eval.data_id).replace(' ', '_')}_met_{eval.metric_id}.json"
        eval.to_json(json_save_dir / file_name)

    all_evaluations = loaded_evaluations

    if len(to_evaluate) > 0:
        print(
            f"Data paths keys (Train Data Label, Inference Labels): {[(train_data_label, inference_data.keys()) for train_data_label, inference_data in reference_data.items()]}"
        )
        print("Models keys: ")

        evaluated = []

        for model_label, model_data in models_data.items():
            for model_paths in model_data:
                print(
                    f"Model: {model_label, model_paths.get('sampling_label'), model_paths.get('initial_state_label')}, (Train Data Label, Inference Label): ",
                    (model_paths["train_data_label"], model_paths["inference_data_label"]),
                )

        for eval in (pbar := tqdm(to_evaluate, leave=False)):
            train_data_setup, initial_state_setup = eval.data_id
            pbar.set_description(f"Processing metric={eval.metric_id}, model={eval.model_id}, {train_data_setup=}, {initial_state_setup=}.")

            start_time = datetime.now()

            model_label, sampling_label, initial_state_label = eval.model_id
            train_data_label, inference_data_label = eval.data_id

            model_data = get_model_data(
                models_data[model_label], train_data_label, inference_data_label, sampling_label, initial_state_label
            )

            model_paths = np.array(model_data["synthetic_path"])
            model_drift = model_data.get("drift_at_locations")
            model_diffusion = model_data.get("diffusion_at_locations")

            if model_drift is not None:
                model_drift = np.array(model_drift)
            if model_diffusion is not None:
                model_diffusion = np.array(model_diffusion)

            reference_paths, reference_drift, reference_diffusion = get_lorenz_data(reference_data, train_data_label, inference_data_label)

            if eval.metric_id == "mmd":
                if model_paths is None or np.isnan(model_paths).any() or np.isnan(reference_paths).any():
                    eval.metric_value = np.nan

                else:
                    eval.metric_value = compute_mmd(reference_paths, model_paths)

            elif eval.metric_id == "mse_drift" and model_drift is not None:
                eval.metric_value = get_mse(reference_drift, model_drift)

            elif eval.metric_id == "mse_diffusion" and model_drift is not None:
                eval.metric_value = get_mse(reference_diffusion, model_diffusion)

            elif eval.metric_id == "mse_paths":
                assert reference_paths.shape == model_paths.shape
                eval.metric_value = get_mse(reference_paths.reshape(-1, 3), model_paths.reshape(-1, 3))

            end_time = datetime.now()
            computation_time = (end_time - start_time).total_seconds()

            # in case no model drift or diffusion was available
            if eval.metric_value is not None:
                print(
                    f"Last result: metric={eval.metric_id}, value={eval.metric_value}, model={eval.model_id}, data={eval.data_id}, {computation_time=} seconds."
                )
                print("\n")

                file_name = (
                    f"model_{str(eval.model_id).replace(' ', '_')}_data_{str(eval.data_id).replace(' ', '_')}_met_{eval.metric_id}.json"
                )
                eval.to_json(json_save_dir / file_name)

                all_evaluations.append(eval)

    for metric in metrics:
        handle_negative_values = [None, "clip", "abs"] if metric == "mmd" else [None]
        for handle in handle_negative_values:
            df_mean = lorenz_metric_table(all_evaluations, metric, precision, handle)

            subdir_name = f"tables_{metric}"

            if handle is not None:
                subdir_name = subdir_name + "_" + handle

            metric_save_dir: Path = evaluation_dir / subdir_name
            metric_save_dir.mkdir(exist_ok=True, parents=True)

            # save_table(df_count_exps, metric_save_dir, "count_experiments")
            save_table(df_mean, metric_save_dir, "mean")
