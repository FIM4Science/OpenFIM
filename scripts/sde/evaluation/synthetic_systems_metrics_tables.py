import json
from copy import deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import optree
import pandas as pd
from metrics_helpers import (
    MetricEvaluation,
    load_metric_evaluations_from_dirs,
    mean_bracket_std_agg,
    mean_plus_std_agg,
    nans_to_stars,
    save_table,
)
from mmd import compute_mmd
from tqdm import tqdm


def get_ground_truth_system_paths(all_paths: list[dict], system: str, experiment_num: int) -> np.ndarray:
    """
    Extract paths from one experiment of ground-truth system data, loaded from json file.

    Args:
        all_paths (list[dict]): List all datasets, including all experiments, tau, noise.
        system, experiment_num: Identifying information for paths to extract from all_paths.

    Returns:
        paths_of_dataset (dict): Unique element in all_paths with identifier (dataset, split, total_splits).
    """
    paths_of_system = [d for d in all_paths if d["name"] == system]

    # should contain exactly one set of paths for each system
    if len(paths_of_system) == 1:
        return np.array(paths_of_system[0]["real_paths"][experiment_num])

    elif len(paths_of_system) == 0:
        raise ValueError(f"Could not find ground-truth paths of system {system}.")

    else:
        raise ValueError(f"Found {len(paths_of_system)} sets of paths for system {system}")


def get_ground_truth_vector_field(all_vector_fields: list[dict], vector_field_key: str, system: str, experiment_num: int) -> np.ndarray:
    """
    Extract vector field from one experiment of ground-truth system data, loaded from json file.

    Args:
        all_vector_fields (list[dict]): List all vector fields.
        system, experiment_num: Identifying information for vector fields to extract.

    Returns:
        paths_of_dataset (dict): Unique element in all_vector_fields with identifier (dataset, split, total_splits).
    """
    vector_fields_of_system = [d for d in all_vector_fields if d["name"] == system]

    # should contain exactly one pair of vector fields for each system
    if len(vector_fields_of_system) == 1:
        return np.array(vector_fields_of_system[0][vector_field_key][experiment_num])

    elif len(vector_fields_of_system) == 0:
        raise ValueError(f"Could not find ground-truth {vector_field_key} of system {system}.")

    else:
        raise ValueError(f"Found {len(vector_fields_of_system)} vector fields {vector_field_key} for system {system}")


def get_model_data(all_model_data: list[dict], data_key: str, system: str, tau: float, noise: float, experiment_num: int) -> np.ndarray:
    """
    Extract result of one experiment from one model from a loaded json file.

    Args:
        all_model_data (list[dict]): Results of one model on all datasets.
        data_key (str): Dict key of result to extract.
        system, tau, noise, experiment_num: Identifying information for result to extract from all_model_data.

    Returns:
        model_paths_value (np.ndarray): Extracted model result.
    """
    # backward compatability: if model data does not contain noise key, it has been evaluated with 0 noise
    for i in range(len(all_model_data)):
        model_data: dict = all_model_data[i]
        if "noise" not in model_data:
            model_data["noise"] = 0.0

    model_data_of_system = [d for d in all_model_data if (d["name"] == system) and (d["tau"] == tau) and (d["noise"] == noise)]

    # should contain exactly one set of data for each system
    if len(model_data_of_system) == 1:
        return np.array(model_data_of_system[0][data_key][experiment_num])

    elif len(model_data_of_system) == 0:
        # raise Warning(f"Could not find {data_key} of system {system} with tau {tau} and noise {noise}.")
        return None

    else:
        raise Warning(f"Found {len(model_data_of_system)} sets of {data_key} for system {system} with tau {tau} and noise {noise}.")


def synthetic_systems_metric_table(
    all_evaluations: list[MetricEvaluation],
    metric: str,
    models_order: list[str],
    systems_order: list[str],
    precision: int,
    handle_negative_values: Optional[str] = "clip",
):
    """
    Turn all results of one metric into pandas dataframes, depicting them as mean, standard deviation, mean + std and mean(std).

    Args:
        all_evaluations (list[MetricEvaluation]): Results from all models on all datasets and all splits.
        metric (str): Metric to create tables for.
        models_order (list[str]): Specify order of rows the models appear in.
        datasets_order (list[str]): Specify order of columns the datasets appear in.
        precision (int): Rounding precision (used only in some tables).
        handle_negative_values (Optional[str] = "clip"): Sometimes mmd can be negative, if the paths are really good.

    Returns:
        Dataframes with metric, averaged over some splits, with different formatting:
        mean, std, mean + std, mean(std)
        Also a table that counts (non) NaN splits.
    """
    assert handle_negative_values in [None, "clip", "abs"]

    # all evaluations with metric to dataframe
    rows = [
        {
            "model": eval.model_id,
            "system": eval.data_id[0],
            "tau": eval.data_id[1],
            "noise": eval.data_id[2],
            "exp": eval.data_id[3],
            "metric_value": eval.metric_value,
        }
        for eval in all_evaluations
        if eval.metric_id == metric
    ]

    cols = optree.tree_map(lambda *x: x, *rows)
    df = pd.DataFrame.from_dict(cols)

    # models have custom sorting
    df["model"] = pd.Categorical(df["model"], models_order)

    # count number of experiments
    df_count_exps = deepcopy(df[["system", "noise", "tau", "model", "metric_value"]])
    df_count_exps = df_count_exps.groupby(["system", "noise", "tau", "model"]).size()
    df_count_exps = df_count_exps.unstack(0)

    # count number of experiments_without Nans
    df_count_non_nans = df.groupby(["system", "noise", "tau", "model"]).count().drop("exp", axis=1)
    df_count_non_nans = df_count_non_nans["metric_value"].unstack(0)

    # sometimes mmd can be negative, if the paths are really good
    if handle_negative_values == "clip":
        df["metric_value"] = np.clip(df["metric_value"], a_min=0.0, a_max=np.inf)

    elif handle_negative_values == "abs":
        df["metric_value"] = np.abs(df["metric_value"])

    # mean without Nans
    df_mean = (
        df.groupby(["system", "noise", "tau", "model"])
        .agg(lambda x: str(x.dropna().mean()) if (len(x.dropna()) != 0) else "-")
        .drop("exp", axis=1)
    )
    df_mean = df_mean["metric_value"].unstack(0)

    # std without Nans
    df_std = (
        df.groupby(["system", "noise", "tau", "model"])
        .agg(lambda x: str(x.dropna().std()) if (len(x.dropna()) != 0) else "-")
        .drop("exp", axis=1)
    )
    df_std = df_std["metric_value"].unstack(0)

    # mean and std in one cell; formatted as "mean $\pm$ std"
    df_mean_plus_std = (
        df.groupby(["system", "noise", "tau", "model"]).agg(partial(mean_plus_std_agg, precision=precision)).drop("exp", axis=1)
    )
    df_mean_plus_std = df_mean_plus_std["metric_value"].unstack(0)

    # mean and std in one cell; formatted with bracket notation
    df_mean_bracket_std = (
        df.groupby(["system", "noise", "tau", "model"]).agg(partial(mean_bracket_std_agg, precision=precision)).drop("exp", axis=1)
    )
    df_mean_bracket_std = df_mean_bracket_std["metric_value"].unstack(0)

    # mean and std in one cell; formatted with bracket notation; multiply by 10 first
    df_mean_bracket_std_times_10 = deepcopy(df)
    df_mean_bracket_std_times_10["metric_value"] = df_mean_bracket_std_times_10["metric_value"] * 10
    df_mean_bracket_std_times_10 = (
        df_mean_bracket_std_times_10.groupby(["system", "noise", "tau", "model"])
        .agg(partial(mean_bracket_std_agg, precision=precision))
        .drop("exp", axis=1)
    )
    df_mean_bracket_std_times_10 = df_mean_bracket_std_times_10["metric_value"].unstack(0)

    # # drop rows with all Nans
    # df_mean = df_mean.map(lambda x: np.nan if x == "-" else x).dropna()
    # df_std = df_std.map(lambda x: np.nan if x == "-" else x).dropna()
    # df_mean_bracket_std = df_mean_bracket_std.map(lambda x: np.nan if x == "-" else x).dropna()
    # df_mean_bracket_std_times_10 = df_mean_bracket_std_times_10.map(lambda x: np.nan if x == "-" else x).dropna()

    # count Nans and depict them as stars
    df_star_nans = deepcopy(df[["system", "noise", "tau", "model", "metric_value"]])
    df_star_nans["metric_is_nan"] = np.isnan(df["metric_value"])
    df_star_nans = df_star_nans.drop("metric_value", axis=1)

    df_star_nans = df_star_nans.groupby(["system", "noise", "tau", "model"]).agg(nans_to_stars)
    df_star_nans = df_star_nans["metric_is_nan"].unstack(0)

    # # add number of Nan experiments as stars to each cell
    df_mean = df_mean + " " + df_star_nans[df_star_nans.index.isin(df_mean.index)]
    df_std = df_std + " " + df_star_nans[df_star_nans.index.isin(df_std.index)]
    df_mean_plus_std = df_mean_plus_std + " " + df_star_nans[df_star_nans.index.isin(df_mean_plus_std.index)]
    df_mean_bracket_std = df_mean_bracket_std + " " + df_star_nans[df_star_nans.index.isin(df_mean_bracket_std.index)]
    df_mean_bracket_std_times_10 = (
        df_mean_bracket_std_times_10 + " " + df_star_nans[df_star_nans.index.isin(df_mean_bracket_std_times_10.index)]
    )

    # reorder columns
    df_count_exps = df_count_exps[systems_order]
    df_count_non_nans = df_count_non_nans[systems_order]
    df_mean = df_mean[systems_order]
    df_std = df_std[systems_order]
    df_mean_plus_std = df_mean_plus_std[systems_order]
    df_mean_bracket_std = df_mean_bracket_std[systems_order]
    df_mean_bracket_std_times_10 = df_mean_bracket_std_times_10[systems_order]

    return df_count_exps, df_count_non_nans, df_mean, df_std, df_mean_plus_std, df_mean_bracket_std_times_10, df_mean_bracket_std


def mse(target: np.ndarray, prediction: np.ndarray) -> float:
    """
    MSE

    target: [B, L, D]
    prediction: [B, L, D]
    """
    assert target.shape == prediction.shape

    return ((target - prediction) ** 2).mean().item()


def nmse(target: np.ndarray, prediction: np.ndarray, cutoff: float = 0.0) -> float:
    """
    Normalized MSE by target norm, when target norm is above cutoff.

    target: [B, L, D]
    prediction: [B, L, D]
    """
    assert target.shape == prediction.shape

    error_norm = ((target - prediction) ** 2).mean(axis=-1)
    target_norm = (target**2).mean(axis=-1)

    nmse = np.where(target_norm > cutoff, error_norm / target_norm, np.nan)

    return np.nanmean(nmse).item()


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    dataset_descr = "synthetic_systems_metrics_tables"

    # How to name experiments
    # experiment_descr = "fim_fixed_attn_fixed_softmax_05-06-2300"
    # experiment_descr = "fim_vs_bisde_vs_sparsegp"
    # experiment_descr = "fim_with_ablations_vs_bisde_vs_sparsegp"
    # experiment_descr = "ablation_600k_train_size_500k_steps_using_ICML_model_no_delta_tau"
    # experiment_descr = "fim_fixed_softmax_dim_05-03-2033_epoch_139"
    # experiment_descr = "ablations_training_epochs_of_single_model"
    # experiment_descr = "comparing_checkpoints_of_fim_fixed_linear_attn_and_fixed_softmax_dim"
    experiment_descr = "neurips_table_1_fim_sparsegp_bisde"

    project_path = "/cephfs/users/seifner/repos/FIM"

    data_paths_json = Path(
        "/cephfs_projects/foundation_models/data/SDE/evaluation/20250325_synthetic_systems_5000_points_with_additive_noise/data/systems_ksig_reference_paths.json"
    )
    data_vector_fields_json = Path(
        "/cephfs_projects/foundation_models/data/SDE/evaluation/20250325_synthetic_systems_5000_points_with_additive_noise/data/systems_ground_truth_drift_diffusion.json"
    )

    models_jsons = {
        # "20M_params_600k_polys_delta_tau_fixed_attn_fixed_softmax (checkpoint 05-03-2033)": Path(
        #     "/home/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/synthetic_systems_vf_and_paths/05062310_fim_fixed_linear_attn_fixed_softmax_delta_tau_05-03-2033/model_paths.json"
        # ),
        # "20M_params_600k_polys_delta_tau_fixed_attn_fixed_softmax (checkpoint 05-06-2300)": Path(
        #     "/home/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/synthetic_systems_vf_and_paths/05092348_fim_fixed_linear_attn_fixed_softmax_delta_tau_05-06-2300/model_paths.json"
        # ),
        "SparseGP(Reevaluation, May 5th 25)": Path(
            "/cephfs_projects/foundation_models/data/SDE/evaluation/20250505_sparse_gp_model_results_with_noise/20250505_sparse_gp_experiments_mai.json"
        ),
        # "Ablation: 30k, degree 4 drift, 500k steps": Path(
        #     "/home/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/synthetic_systems_vf_and_paths/05021706_ablation_30k_deg_4_drift_500k_steps/model_paths.json"
        # ),
        # "Ablation: 30k train size, 5M parameters, 500k steps": Path(
        #     "/home/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/synthetic_systems_vf_and_paths/05021712_ablation_30k_train_size_500k_steps/model_paths.json"
        # ),
        # "Ablation: 100k train size, 10M parameters, 500k steps": Path(
        #     "/home/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/synthetic_systems_vf_and_paths/05021720_ablation_100k_train_size_500k_steps/model_paths.json"
        # ),
        # "Ablation: 600k train size, 20M parameters, 500k steps": Path(  # DO NOT USE; TRAIN DATA WAS DELTA TAU (should be named "FIM (fixed linear Attn.) (04-28-0941) Epoch 70")
        #     "/home/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/synthetic_systems_vf_and_paths/05021730_ablation_600k_train_size_500k_steps/model_paths.json"
        # ),
        # "Ablation: 600k train size, 20M parameters, 500k steps (ICML model, no delta tau, epoch 70)": Path(
        #     "/home/seifner/repos/FIM/evaluations/synthetic_systems_vf_and_paths_evaluation/05131859_ablation_600k_train_size_500k_steps_using_ICML_model_no_delta_tau/model_paths.json",
        # ),
        "BISDE(Reevaluation, May 10th 25) (no diffusion clipping)": Path(
            "/cephfs_projects/foundation_models/data/SDE/evaluation/20250510_bisde_results_with_noise/20250510_bisde_results_with_multiple_diffusion_summands_no_diffusion_clipping/20250510_bisde_results_no_diffusion_clipping.json"
        ),
        # "BISDE(Reevaluation, May 11th 25) (with diffusion clipping)": Path(
        #     "/cephfs_projects/foundation_models/data/SDE/evaluation/20250510_bisde_results_with_noise/20250511_bisde_results_with_multiple_diffusion_summands_with_diffusion_clipping/20250511_bisde_results_with_diffusion_clipping.json"
        # ),
        # "FIM (fixed linear Attn.) (04-28-0941) Epoch 125": Path(
        #     "/home/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/synthetic_systems_vf_and_paths/05141243_fim_fixed_linear_attn_04-28-0941_epoch_125/model_paths.json"
        # ),
        # "FIM (fixed Softmax dim.) (05-03-2033) Epoch 40": Path(
        #     "/home/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/synthetic_systems_vf_and_paths/05141331_fim_fixed_softmax_dim_05-03-2033_epoch_040/model_paths.json"
        # ),
        # "FIM (fixed Softmax dim.) (05-03-2033) Epoch 70": Path(
        #     "/home/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/synthetic_systems_vf_and_paths/05141352_fim_fixed_softmax_dim_05-03-2033_epoch_070/model_paths.json"
        # ),
        # "FIM (fixed Softmax dim.) (05-03-2033) Epoch 100": Path(
        #     "/home/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/synthetic_systems_vf_and_paths/05141418_fim_fixed_softmax_dim_05-03-2033_epoch_100/model_paths.json"
        # ),
        "FIM (fixed Softmax dim.) (05-03-2033) Epoch 139": Path(
            "/home/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/synthetic_systems_vf_and_paths/05141437_fim_fixed_softmax_dim_05-03-2033_epoch_139/model_paths.json"
        ),
    }
    apply_sqrt_to_diffusion = [
        "BISDE",
        "BISDE(Reevaluation, May 10th 25) (no diffusion clipping)",
        "BISDE(Reevaluation, May 11th 25) (with diffusion clipping)",
        "SparseGP(Reevaluation, May 5th 25)",
    ]

    models_to_evaluate = [
        # "20M_params_600k_polys_delta_tau_fixed_attn_fixed_softmax (checkpoint 05-03-2033)",
        # "20M_params_600k_polys_delta_tau_fixed_attn_fixed_softmax (checkpoint 05-06-2300)",
        "SparseGP(Reevaluation, May 5th 25)",
        # "Ablation: 30k, degree 4 drift, 500k steps",
        # "Ablation: 30k train size, 5M parameters, 500k steps",
        # "Ablation: 100k train size, 10M parameters, 500k steps",
        # "Ablation: 600k train size, 20M parameters, 500k steps",  # DO NOT USE THIS, WAS TRAINED ON DELTA TAU DATA (should be named "FIM (fixed linear Attn.) (04-28-0941) Epoch 70")
        # "Ablation: 600k train size, 20M parameters, 500k steps (ICML model, no delta tau, epoch 70)",
        "BISDE(Reevaluation, May 10th 25) (no diffusion clipping)",
        # "BISDE(Reevaluation, May 11th 25) (with diffusion clipping)",
        # "FIM (fixed linear Attn.) (04-28-0941) Epoch 125",
        # "FIM (fixed Softmax dim.) (05-03-2033) Epoch 40",
        # "FIM (fixed Softmax dim.) (05-03-2033) Epoch 70",
        # "FIM (fixed Softmax dim.) (05-03-2033) Epoch 100",
        "FIM (fixed Softmax dim.) (05-03-2033) Epoch 139",
    ]

    systems_to_evaluate = [
        "Double Well",
        "Wang",
        "Damped Linear",
        "Damped Cubic",
        "Duffing",
        "Glycosis",
        "Hopf",
    ]

    taus_to_evaluate = [
        0.002,
        # 0.01,
        0.02,
        # 0.2,
    ]
    noise_to_evaluate = [
        0.0,
        0.05,
        # 0.1,
    ]
    experiment_count = 5
    mmd_max_num_paths = 100

    only_use_loaded_evaluations = True

    metrics_to_evaluate = [
        "nmse_drift",
        "nmse_diffusion",
        "nmse_drift_and_diffusion_average",
        "mse_drift",
        "mse_diffusion",
        "mse_drift_and_diffusion_average",
        "nmse_above_1e-6_drift",
        "nmse_above_1e-6_diffusion",
        "nmse_above_1e-6_drift_and_diffusion_average",
        "mmd",
    ]

    metric_evaluations_to_load: list[Path] = [
        # Path(
        #     "/cephfs/users/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/synthetic_systems_metrics_tables/05091823_fim_fixed_attn_fixed_softmax_05-03-2033/metric_evaluations_jsons"
        # ),
        # Path(
        #     "/cephfs/users/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/synthetic_systems_metrics_tables/05100018_fim_fixed_attn_fixed_softmax_05-06-2300/metric_evaluations_jsons"
        # ),
        Path(
            "/cephfs/users/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/synthetic_systems_metrics_tables/05110044_bisde_reevaluation_may_10th_25_no_diffusion_clipping/metric_evaluations_jsons"
        ),
        # Path(
        #     "/cephfs/users/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/synthetic_systems_metrics_tables/05112004_bisde_reevaluation_may_11th_25_with_diffusion_clipping/metric_evaluations_jsons"
        # ),
        Path(
            "/cephfs/users/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/synthetic_systems_metrics_tables/05091855_sparsegp_reevaluation_may_5th/metric_evaluations_jsons"
        ),
        # Path(
        #     "/cephfs/users/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/synthetic_systems_metrics_tables/05122300_ablation_degree_4/metric_evaluations_jsons"
        # ),
        # Path(
        #     "/cephfs/users/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/synthetic_systems_metrics_tables/05091903_ablation_train_size_30k/metric_evaluations_jsons"
        # ),
        # Path(
        #     "/cephfs/users/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/synthetic_systems_metrics_tables/05110804_ablation_train_size_100k/metric_evaluations_jsons"
        # ),
        # Path(
        #     "/cephfs/users/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/synthetic_systems_metrics_tables/05110806_ablation_train_size_600k/metric_evaluations_jsons"
        # ),
        # Path(
        #     "/cephfs/users/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/synthetic_systems_metrics_tables/05131923_ablation_600k_train_size_500k_steps_using_ICML_model_no_delta_tau/metric_evaluations_jsons"
        # ),
        # Path(
        #     "/cephfs/users/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/synthetic_systems_metrics_tables/05141535_fim_fixed_linear_attn_04-28-0941_epoch_125/metric_evaluations_jsons"
        # ),
        # Path(
        #     "/cephfs/users/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/synthetic_systems_metrics_tables/05141538_fim_fixed_softmax_dim_05-03-2033_epoch_40/metric_evaluations_jsons"
        # ),
        # Path(
        #     "/cephfs/users/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/synthetic_systems_metrics_tables/05141540_fim_fixed_softmax_dim_05-03-2033_epoch_70/metric_evaluations_jsons"
        # ),
        # Path(
        #     "/cephfs/users/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/synthetic_systems_metrics_tables/05141547_fim_fixed_softmax_dim_05-03-2033_epoch_100/metric_evaluations_jsons"
        # ),
        Path(
            "/cephfs/users/seifner/repos/FIM/saved_evaluations/20250329_neurips_submission_preparations/synthetic_systems_metrics_tables/05141550_fim_fixed_softmax_dim_05-03-2033_epoch_139/metric_evaluations_jsons"
        ),
    ]

    # tables config
    models_order = [
        # "20M_params_600k_polys_delta_tau_fixed_attn_fixed_softmax (checkpoint 05-03-2033)",
        # "20M_params_600k_polys_delta_tau_fixed_attn_fixed_softmax (checkpoint 05-06-2300)",
        # "Ablation: 30k, degree 4 drift, 500k steps",
        # "Ablation: 30k train size, 5M parameters, 500k steps",
        # "Ablation: 100k train size, 10M parameters, 500k steps",
        # "Ablation: 600k train size, 20M parameters, 500k steps",  # DO NOT USE THIS, (should be named "FIM (fixed linear Attn.) (04-28-0941) Epoch 70")
        # "Ablation: 600k train size, 20M parameters, 500k steps (ICML model, no delta tau, epoch 70)",
        "BISDE(Reevaluation, May 10th 25) (no diffusion clipping)",
        # "BISDE(Reevaluation, May 11th 25) (with diffusion clipping)",
        "SparseGP(Reevaluation, May 5th 25)",
        # "FIM (fixed linear Attn.) (04-28-0941) Epoch 125",
        # "FIM (fixed Softmax dim.) (05-03-2033) Epoch 40",
        # "FIM (fixed Softmax dim.) (05-03-2033) Epoch 70",
        # "FIM (fixed Softmax dim.) (05-03-2033) Epoch 100",
        "FIM (fixed Softmax dim.) (05-03-2033) Epoch 139",
    ]
    systems_order = systems_to_evaluate
    precision = 3

    # --------------------------------------------------------------------------------------------------------------------------------- #

    # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
    evaluation_path = Path(project_path) / "evaluations"
    time: str = str(datetime.now().strftime("%m%d%H%M"))
    evaluation_dir: Path = evaluation_path / dataset_descr / (time + "_" + experiment_descr)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # add average of drift and diffusion mse as metric
    if ("mse_drift" in metrics_to_evaluate) and ("mse_diffusion" in metrics_to_evaluate):
        metrics_to_evaluate.append("mse_drift_and_diffusion_average")

    # prepare all evaluations to be done
    print("Preparing all evaluations.")
    all_evaluations: list[MetricEvaluation] = [
        MetricEvaluation(
            model_id=model,
            model_json=models_jsons[model],
            data_id=(system, tau, noise, experiment_num, mmd_max_num_paths),
            data_paths_json=data_paths_json,
            data_vector_fields_json=data_vector_fields_json,
            metric_id=metric,
            metric_value=None,  # input later
        )
        for model in models_to_evaluate
        for system in systems_to_evaluate
        for tau in taus_to_evaluate
        for noise in noise_to_evaluate
        for experiment_num in range(experiment_count)
        for metric in metrics_to_evaluate
    ]

    print("Loading saved evaluations.")
    # load saved evaluations; remove those not already contained in all_evaluations
    loaded_evaluations: list[MetricEvaluation] = load_metric_evaluations_from_dirs(metric_evaluations_to_load)

    for eval in loaded_evaluations:
        if len(eval.data_id) == 4:  # backward compatability for evaluations without noise
            system, tau, experiment_num, mmd_max_num_paths = eval.data_id
            eval.data_id = system, tau, 0.0, experiment_num, mmd_max_num_paths

    loaded_evaluations: list[MetricEvaluation] = [eval for eval in loaded_evaluations if eval in all_evaluations]

    if only_use_loaded_evaluations is False:
        to_evaluate: list[MetricEvaluation] = [eval for eval in all_evaluations if eval not in loaded_evaluations]

    else:
        to_evaluate: list[MetricEvaluation] = []

    # prepare directory to save evaluations in save
    json_save_dir: Path = evaluation_dir / "metric_evaluations_jsons"
    json_save_dir.mkdir(exist_ok=True, parents=True)

    for eval in loaded_evaluations:
        system, tau, noise, exp, max_num_paths = eval.data_id
        file_name = f"mod_{eval.model_id.replace(' ', '_')}_met_{eval.metric_id}_sys_{system}_tau_{tau}_noise_{noise}_exp_{exp}.json"
        eval.to_json(json_save_dir / file_name)

    # compute metric for missing evaluations
    all_evaluations = loaded_evaluations
    mmd_ground_truth_cache = {}  # K_xx for the ground truth data is the same for all models, so we cache it

    if len(to_evaluate) > 0:
        print("Loading ground-truth and model data.")

        data_paths: list[dict] = json.load(open(data_paths_json, "r"))
        data_vector_fields: list[dict] = json.load(open(data_vector_fields_json, "r"))
        models_data: list[dict] = {model_name: json.load(open(model_json, "r")) for model_name, model_json in models_jsons.items()}

        print(f"Data paths keys: {[d['name'] for d in data_paths]}")
        print(f"Data vector fields keys: {[d['name'] for d in data_vector_fields]}")
        print("Models keys: ")
        for model_name, model_systems in models_data.items():
            print(f"Model: {model_name}, Systems: {[system['name'] for system in model_systems]}")

        for eval in (pbar := tqdm(to_evaluate, leave=False)):
            system, tau, noise, exp, mmd_max_num_paths = eval.data_id
            pbar.set_description(f"Processing metric={eval.metric_id}, model={eval.model_id}, {system=}, {tau=}, {noise=}, {exp=}.")

            start_time = datetime.now()

            if eval.metric_id == "mmd":
                ground_truth_paths: np.ndarray = get_ground_truth_system_paths(data_paths, system, exp)[:mmd_max_num_paths]
                model_paths: np.ndarray = get_model_data(models_data[eval.model_id], "synthetic_paths", system, tau, noise, exp)[
                    :mmd_max_num_paths
                ]

                if model_paths is None or np.isnan(model_paths).any():
                    eval.metric_value = np.nan

                else:
                    eval.metric_value = compute_mmd(ground_truth_paths, model_paths, kernel_cache=mmd_ground_truth_cache)

            elif eval.metric_id in [
                "mse_drift",
                "mse_diffusion",
                "nmse_drift",
                "nmse_diffusion",
                "nmse_above_1e-6_drift",
                "nmse_above_1e-6_diffusion",
                "mse_drift_and_diffusion_average",
                "nmse_drift_and_diffusion_average",
                "nmse_above_1e-6_drift_and_diffusion_average",
            ]:
                ground_truth_drift: np.ndarray = get_ground_truth_vector_field(data_vector_fields, "drift_at_locations", system, exp)
                model_drift: np.ndarray = get_model_data(models_data[eval.model_id], "drift_at_locations", system, tau, noise, exp)

                ground_truth_diff: np.ndarray = get_ground_truth_vector_field(data_vector_fields, "diffusion_at_locations", system, exp)
                model_diff: np.ndarray = get_model_data(models_data[eval.model_id], "diffusion_at_locations", system, tau, noise, exp)

                if ground_truth_diff.shape != model_diff.shape:
                    raise ValueError(
                        f"Ground-Truth vf {ground_truth_diff.shape} and estimation {model_diff.shape} must have same shape. Evaluation {eval}."
                    )

                if ground_truth_drift.shape != model_drift.shape:
                    raise ValueError(
                        f"Ground-Truth vf {ground_truth_drift.shape} and estimation {model_drift.shape} must have same shape. Evaluation {eval}."
                    )

                # adjust for different convention of comparison models
                if eval.model_id in apply_sqrt_to_diffusion:
                    model_diff = np.sqrt(np.clip(model_diff, a_min=0.0, a_max=np.inf))

                if eval.metric_id == "mse_drift":
                    eval.metric_value = mse(ground_truth_drift, model_drift)

                elif eval.metric_id == "mse_diffusion":
                    eval.metric_value = mse(ground_truth_diff, model_diff)

                elif eval.metric_id == "nmse_drift":
                    eval.metric_value = nmse(ground_truth_drift, model_drift)

                elif eval.metric_id == "nmse_diffusion":
                    eval.metric_value = nmse(ground_truth_diff, model_diff)

                elif eval.metric_id == "nmse_above_1e-6_drift":
                    eval.metric_value = nmse(ground_truth_drift, model_drift, cutoff=1e-6)

                elif eval.metric_id == "nmse_above_1e-6_diffusion":
                    eval.metric_value = nmse(ground_truth_diff, model_diff, cutoff=1e-6)

                elif eval.metric_id == "mse_drift_and_diffusion_average":
                    mse_drift = mse(ground_truth_drift, model_drift)
                    mse_diff = mse(ground_truth_diff, model_diff)
                    eval.metric_value = (1 / 2) * (mse_drift + mse_diff)

                elif eval.metric_id == "nmse_drift_and_diffusion_average":
                    nmse_drift = nmse(ground_truth_drift, model_drift)
                    nmse_diff = nmse(ground_truth_diff, model_diff)
                    eval.metric_value = (1 / 2) * (nmse_drift + nmse_diff)

                elif eval.metric_id == "nmse_above_1e-6_drift_and_diffusion_average":
                    nmse_drift = nmse(ground_truth_drift, model_drift, cutoff=1.0e-6)
                    nmse_diff = nmse(ground_truth_diff, model_diff, cutoff=1.0e-6)
                    eval.metric_value = (1 / 2) * (nmse_drift + nmse_diff)

            else:
                valid_metric_ids = [
                    "nmse_drift",
                    "nmse_diffusion",
                    "nmse_drift_and_diffusion_average",
                    "mse_drift",
                    "mse_diffusion",
                    "mse_drift_and_diffusion_average",
                    "nmse_above_1e-6_drift",
                    "nmse_above_1e-6_diffusion",
                    "nmse_above_1e-6_drift_and_diffusion_average",
                    "mmd",
                ]
                raise ValueError(f"Valid metrics are {str(valid_metric_ids)}, got {eval.metric_id} from requested evaluation {eval}.")

            # record computation time
            end_time = datetime.now()
            computation_time = (end_time - start_time).total_seconds()

            print(
                f"Last result: metric={eval.metric_id}, model={eval.model_id}, {system=}, {tau=}, {exp=}, value={eval.metric_value}, {computation_time=} seconds."
            )

            # save results as json
            file_name = f"mod_{eval.model_id.replace(' ', '_')}_met_{eval.metric_id}_sys_{system}_tau_{tau}_noise_{noise}_exp_{exp}.json"
            eval.to_json(json_save_dir / file_name)

            all_evaluations.append(eval)

    for metric in metrics_to_evaluate:
        negative_values = [None, "clip", "abs"] if metric == "mmd" else [None]

        for handle_negative_values in negative_values:
            df_count_exps, df_count_non_nans, df_mean, df_std, df_mean_plus_std, df_mean_bracket_std_times_10, df_mean_bracket_std = (
                synthetic_systems_metric_table(all_evaluations, metric, models_order, systems_order, precision, handle_negative_values)
            )

            subdir_name = "tables_" + metric

            if handle_negative_values is not None:
                subdir_name = subdir_name + "_" + handle_negative_values

            metric_save_dir: Path = evaluation_dir / subdir_name
            metric_save_dir.mkdir(exist_ok=True, parents=True)

            # save_table(df_count_exps, metric_save_dir, "count_experiments")
            save_table(df_count_non_nans, metric_save_dir, "count_non_nans")
            save_table(df_mean, metric_save_dir, "mean")
            save_table(df_std, metric_save_dir, "std")
            save_table(df_mean_plus_std, metric_save_dir, "mean_plus_std")
            save_table(df_mean_bracket_std, metric_save_dir, "mean_bracket_std")
            save_table(df_mean_bracket_std_times_10, metric_save_dir, "mean_bracket_std_times_10")
