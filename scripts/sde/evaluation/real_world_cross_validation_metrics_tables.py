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


def get_reference_paths(all_paths: list[dict], dataset: str, split: int, total_splits: int) -> np.ndarray:
    """
    Extract paths from one split of reference data, loaded from json file.

    Args:
        all_paths (list[dict]): List of all splits from all datasets.
        dataset, split, total_splits: Identifying information for paths to extract from all_paths.

    Returns:
        paths_of_dataset (dict): Unique element in all_paths with identifier (dataset, split, total_splits).
    """
    paths_of_dataset = [d for d in all_paths if d["name"] == dataset and d["split"] == split]

    # should contain exactly one set of paths for each dataset and split
    if len(paths_of_dataset) == 1:
        paths_of_dataset = paths_of_dataset[0]

        assert paths_of_dataset["num_total_splits"] == total_splits, f"Expected {total_splits}, got {paths_of_dataset['num_total_splits']}."

        if paths_of_dataset["transform"] == "log":
            paths_of_dataset["obs_values"] = np.exp(paths_of_dataset["obs_values"])

        return np.array(paths_of_dataset["obs_values"])

    elif len(paths_of_dataset) == 0:
        raise ValueError(f"Could not find reference paths of dataset {dataset}.")

    else:
        raise ValueError(f"Found {len(paths_of_dataset)} sets of paths for dataset {dataset}")


def get_model_data(all_model_data: list[dict], data_key: str, dataset: str, split: int, total_splits: int) -> np.ndarray:
    """
    Extract result on dataset split from one model from a loaded json file.

    Args:
        all_model_data (list[dict]): Results of one model on all datasets.
        data_key (str): Dict key of result to extract.
        dataset, split, total_splits: Identifying information for result to extract from all_model_data.

    Returns:
        model_paths_value (np.ndarray): Extracted model result.
    """
    model_paths_of_dataset = [d for d in all_model_data if (d["name"] == dataset) and (d["split"] == split)]

    # should contain exactly one set of model paths for each dataset
    if len(model_paths_of_dataset) == 1:
        model_paths_of_dataset = model_paths_of_dataset[0]

        assert model_paths_of_dataset["num_total_splits"] == total_splits, (
            f"Expected {total_splits}, got {model_paths_of_dataset['num_total_splits']}."
        )

        model_paths_value = np.array(model_paths_of_dataset[data_key])

        if model_paths_of_dataset["transform"] == "log":
            model_paths_value = np.exp(model_paths_value)

        return model_paths_value

    elif len(model_paths_of_dataset) == 0:
        return None

    else:
        raise Warning(f"Found {len(model_paths_of_dataset)} sets of {data_key} for dataset {dataset} with split {split}.")


def real_world_metric_table(
    all_evaluations: list[MetricEvaluation],
    metric: str,
    models_order: list[str],
    datasets_order: list[str],
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
            "dataset": eval.data_id[0],
            "split": eval.data_id[1],
            "metric_value": eval.metric_value,
        }
        for eval in all_evaluations
        if eval.metric_id == metric
    ]

    cols = optree.tree_map(lambda *x: x, *rows)
    df = pd.DataFrame.from_dict(cols)

    # models have custom sorting
    df["model"] = pd.Categorical(df["model"], models_order)

    # count number of splits
    df_count_exps = deepcopy(df[["dataset", "model", "metric_value"]])
    df_count_exps = df_count_exps.groupby(["dataset", "model"]).size()
    df_count_exps = df_count_exps.unstack(0)

    # count number of experiments_without Nans
    df_count_non_nans = df.groupby(["dataset", "model"]).count().drop("split", axis=1)
    df_count_non_nans = df_count_non_nans["metric_value"].unstack(0)

    # handle negative values
    if handle_negative_values == "clip":
        df["metric_value"] = np.clip(df["metric_value"], a_min=0.0, a_max=np.inf)

    elif handle_negative_values == "abs":
        df["metric_value"] = np.abs(df["metric_value"])

    # mean without Nans
    df_mean = (
        df.groupby(["dataset", "model"]).agg(lambda x: str(x.dropna().mean()) if (len(x.dropna()) != 0) else "-").drop("split", axis=1)
    )
    df_mean = df_mean["metric_value"].unstack(0)

    # std without Nans
    df_std = df.groupby(["dataset", "model"]).agg(lambda x: str(x.dropna().std()) if (len(x.dropna()) != 0) else "-").drop("split", axis=1)
    df_std = df_std["metric_value"].unstack(0)

    # mean and std in one cell; formatted as "mean $\pm$ std"
    df_mean_plus_std = df.groupby(["dataset", "model"]).agg(partial(mean_plus_std_agg, precision=precision)).drop("split", axis=1)
    df_mean_plus_std = df_mean_plus_std["metric_value"].unstack(0)

    # mean and std in one cell; formatted with bracket notation
    df_mean_bracket_std = df.groupby(["dataset", "model"]).agg(partial(mean_bracket_std_agg, precision=precision)).drop("split", axis=1)
    df_mean_bracket_std = df_mean_bracket_std["metric_value"].unstack(0)

    # mean and std in one cell; formatted with bracket notation; multiply by 10 first
    df_mean_bracket_std_times_10 = deepcopy(df)
    df_mean_bracket_std_times_10["metric_value"] = df_mean_bracket_std_times_10["metric_value"] * 10
    df_mean_bracket_std_times_10 = (
        df_mean_bracket_std_times_10.groupby(["dataset", "model"])
        .agg(partial(mean_bracket_std_agg, precision=precision))
        .drop("split", axis=1)
    )
    df_mean_bracket_std_times_10 = df_mean_bracket_std_times_10["metric_value"].unstack(0)

    # # drop rows with all Nans
    # df_mean = df_mean.map(lambda x: np.nan if x == "-" else x).dropna()
    # df_std = df_std.map(lambda x: np.nan if x == "-" else x).dropna()
    # df_mean_bracket_std = df_mean_bracket_std.map(lambda x: np.nan if x == "-" else x).dropna()
    # df_mean_bracket_std_times_10 = df_mean_bracket_std_times_10.map(lambda x: np.nan if x == "-" else x).dropna()

    # count Nans and depict them as stars
    df_star_nans = deepcopy(df[["dataset", "model", "metric_value"]])
    df_star_nans["metric_is_nan"] = np.isnan(df["metric_value"])
    df_star_nans = df_star_nans.drop("metric_value", axis=1)

    df_star_nans = df_star_nans.groupby(["dataset", "model"]).agg(nans_to_stars)
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
    df_count_exps = df_count_exps[datasets_order]
    df_count_non_nans = df_count_non_nans[datasets_order]
    df_mean = df_mean[datasets_order]
    df_std = df_std[datasets_order]
    df_mean_plus_std = df_mean_plus_std[datasets_order]
    df_mean_bracket_std = df_mean_bracket_std[datasets_order]
    df_mean_bracket_std_times_10 = df_mean_bracket_std_times_10[datasets_order]

    return df_count_exps, df_count_non_nans, df_mean, df_std, df_mean_plus_std, df_mean_bracket_std_times_10, df_mean_bracket_std


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    dataset_descr = "real_world_cross_validation_metrics_tables"

    # How to name experiments
    # experiment_descr = "fim_checkpoints_vs_BISDE"
    # experiment_descr = "fim_and_BISDE_vs_ablation_models"
    # experiment_descr = "fim_vs_bisde_neurips_table"
    # experiment_descr = "post_neurips_table_vm_vs_fim_location_at_obs_vs_bisde"
    # experiment_descr = "finetuning_on_one_step_ahead_one_em_step_sampling_nll_vs_mse"
    # experiment_descr = "finetuning_sampling_nll_512_points"
    # experiment_descr = "finetuning_sampling_nll_512_points_seed_1"
    # experiment_descr = "finetune_sampling_nll_one_step_ahead_one_em_step_32_and_512_points_every_10_epochs"
    # experiment_descr = "finetune_sampling_nll_one_step_ahead_one_em_step_nll_512_points_lr_1e_6"
    # experiment_descr = "finetune_sampling_nll_one_step_ahead_one_em_step_nll_512_points_lr_1e_6_every_10_epochs"
    # experiment_descr = "latentsde_MSE_10_subsplits_vs_NLL_100_subsplits"
    experiment_descr = "finetune_sampling_nll_neurips_rebuttal_1_step_ahead_1_em_step_512_points"

    project_path = "/cephfs/users/seifner/repos/FIM"

    reference_paths_json = Path(
        "/cephfs_projects/foundation_models/data/SDE/test/20250726_real_world_with_5_fold_cross_validation/cross_val_ksig_reference_paths.json"
    )

    # models_jsons = {
    #     "FIM (05-03-2033)": Path(
    #         "/cephfs_projects/foundation_models/data/SDE/saved_evaluation_results/20250329_neurips_submission_evaluations/real_world_cross_validation_vf_and_paths_evaluation/05102037_fim_fixed_attn_fixed_softmax_05-03-2033/model_paths.json",
    #     ),
    #     "FIM (05-06-2300)": Path(
    #         "/cephfs_projects/foundation_models/data/SDE/saved_evaluation_results/20250329_neurips_submission_evaluations/real_world_cross_validation_vf_and_paths_evaluation/05102124_fim_fixed_attn_fixed_softmax_05-06-2300/model_paths.json",
    #     ),
    #     "BISDE(20250510, BISDE Library Functions)": Path(
    #         "/cephfs_projects/foundation_models/data/SDE/external_evaluations_and_data/20250726_real_world_with_5_fold_cross_validation/20250510_bisde_5_fold_cross_validation_function_library_from_bisde_paper/bisde_real_world_cv_results.json"
    #     ),
    #     "BISDE(20250510, (u^(0,..,3), exp(u), sin(u)) Library Functions)": Path(
    #         "/cephfs_projects/foundation_models/data/SDE/external_evaluations_and_data/20250726_real_world_with_5_fold_cross_validation/20250510_bisde_5_fold_cross_validation_function_library_with_exps_and_sins/bisde_real_world_cv_our_basis_results.json"
    #     ),
    #     "Ablation: train size 30k, 5M params": Path(
    #         "/cephfs_projects/foundation_models/data/SDE/saved_evaluation_results/20250329_neurips_submission_evaluations/real_world_cross_validation_vf_and_paths_evaluation/05121921_ablation_model_train_size_30k/model_paths.json"
    #     ),
    #     "Ablation: train size 100k, 10M params": Path(
    #         "/cephfs_projects/foundation_models/data/SDE/saved_evaluation_results/20250329_neurips_submission_evaluations/real_world_cross_validation_vf_and_paths_evaluation/05121914_ablation_model_train_size_100k/model_paths.json"
    #     ),
    #     "Ablation: train size 600k, 20M params": Path(
    #         "/cephfs_projects/foundation_models/data/SDE/saved_evaluation_results/20250329_neurips_submission_evaluations/real_world_cross_validation_vf_and_paths_evaluation/05121902_ablation_model_train_size_600k/model_paths.json"
    #     ),
    #     "Ablation: train size 30k with degree 4 drift, 5M params": Path(
    #         "/cephfs_projects/foundation_models/data/SDE/saved_evaluation_results/20250329_neurips_submission_evaluations/real_world_cross_validation_vf_and_paths_evaluation/05121927_ablation_model_degree_4_drift/model_paths.json"
    #     ),
    # }

    # models_jsons = {
    #     "FIM (05-03-2033)": Path(
    #         "/cephfs_projects/foundation_models/data/SDE/saved_evaluation_results/20250329_neurips_submission_evaluations/real_world_cross_validation_vf_and_paths_evaluation/05102037_fim_fixed_attn_fixed_softmax_05-03-2033/model_paths.json",
    #     ),
    #     "FIM (05-06-2300)": Path(
    #         "/cephfs_projects/foundation_models/data/SDE/saved_evaluation_results/20250329_neurips_submission_evaluations/real_world_cross_validation_vf_and_paths_evaluation/05102124_fim_fixed_attn_fixed_softmax_05-06-2300/model_paths.json",
    #     ),
    #     "BISDE(20250510, BISDE Library Functions)": Path(
    #         "/cephfs_projects/foundation_models/data/SDE/external_evaluations_and_data/20250726_real_world_with_5_fold_cross_validation/20250510_bisde_5_fold_cross_validation_function_library_from_bisde_paper/bisde_real_world_cv_results.json"
    #     ),
    #     "BISDE(20250510, (u^(0,..,3), exp(u), sin(u)) Library Functions)": Path(
    #         "/cephfs_projects/foundation_models/data/SDE/external_evaluations_and_data/20250726_real_world_with_5_fold_cross_validation/20250510_bisde_5_fold_cross_validation_function_library_with_exps_and_sins/bisde_real_world_cv_our_basis_results.json"
    #     ),
    #     "BISDE(20250514, BISDE Library Functions)": Path(
    #         "/cephfs_projects/foundation_models/data/SDE/external_evaluations_and_data/20250726_real_world_with_5_fold_cross_validation/20250514_bisde_5_fold_cross_validation_paths_no_nans/bisde_real_world_cv_results.json"
    #     ),
    #     "BISDE(20250514, (u^(0,..,3), exp(u), sin(u)) Library Functions)": Path(
    #         "/cephfs_projects/foundation_models/data/SDE/external_evaluations_and_data/20250726_real_world_with_5_fold_cross_validation/20250514_bisde_5_fold_cross_validation_paths_no_nans/bisde_real_world_cv_our_basis_results.json"
    #     ),
    #     "FIM fixed linear Attn., 04-28-0941, Epoch 040": Path(
    #         "/cephfs_projects/foundation_models/data/SDE/saved_evaluation_results/20250329_neurips_submission_evaluations/real_world_cross_validation_vf_and_paths_evaluation/05132202_fim_fixed_linear_attn_04-28-0941_epoch_040/model_paths.json",
    #     ),
    #     "FIM fixed linear Attn., 04-28-0941, Epoch 070": Path(
    #         "/cephfs_projects/foundation_models/data/SDE/saved_evaluation_results/20250329_neurips_submission_evaluations/real_world_cross_validation_vf_and_paths_evaluation/05132215_fim_fixed_linear_attn_04-28-0941_epoch_070/model_paths.json",
    #     ),
    #     "FIM fixed linear Attn., 04-28-0941, Epoch 100": Path(
    #         "/cephfs_projects/foundation_models/data/SDE/saved_evaluation_results/20250329_neurips_submission_evaluations/real_world_cross_validation_vf_and_paths_evaluation/05132227_fim_fixed_linear_attn_04-28-0941_epoch_100/model_paths.json",
    #     ),
    #     "FIM fixed linear Attn., 04-28-0941, Epoch 125": Path(
    #         "/cephfs_projects/foundation_models/data/SDE/saved_evaluation_results/20250329_neurips_submission_evaluations/real_world_cross_validation_vf_and_paths_evaluation/05132353_fim_fixed_linear_attn_04-28-0941_epoch_125/model_paths.json",
    #     ),
    #     "FIM fixed Softmax dim., 05-03-2033, Epoch 040": Path(
    #         "/cephfs_projects/foundation_models/data/SDE/saved_evaluation_results/20250329_neurips_submission_evaluations/real_world_cross_validation_vf_and_paths_evaluation/05140004_fim_fixed_softmax_05-03-2033_epoch_040/model_paths.json",
    #     ),
    #     "FIM fixed Softmax dim., 05-03-2033, Epoch 070": Path(
    #         "/cephfs_projects/foundation_models/data/SDE/saved_evaluation_results/20250329_neurips_submission_evaluations/real_world_cross_validation_vf_and_paths_evaluation/05140019_fim_fixed_softmax_05-03-2033_epoch_070/model_paths.json",
    #     ),
    #     "FIM fixed Softmax dim., 05-03-2033, Epoch 100": Path(
    #         "/cephfs_projects/foundation_models/data/SDE/saved_evaluation_results/20250329_neurips_submission_evaluations/real_world_cross_validation_vf_and_paths_evaluation/05140033_fim_fixed_softmax_05-03-2033_epoch_100/model_paths.json",
    #     ),
    #     "FIM fixed Softmax dim., 05-03-2033, Epoch 125": Path(
    #         "/cephfs_projects/foundation_models/data/SDE/saved_evaluation_results/20250329_neurips_submission_evaluations/real_world_cross_validation_vf_and_paths_evaluation/05140045_fim_fixed_softmax_05-03-2033_epoch_125/model_paths.json",
    #     ),
    #     "FIM fixed Softmax dim., 05-03-2033, Epoch 138": Path(
    #         "/cephfs_projects/foundation_models/data/SDE/saved_evaluation_results/20250329_neurips_submission_evaluations/real_world_cross_validation_vf_and_paths_evaluation/05140056_fim_fixed_softmax_05-03-2033_epoch_138/model_paths.json",
    #     ),
    # }

    # models_jsons = {
    #     "BISDE(20250514, BISDE Library Functions)": Path(
    #         "/cephfs_projects/foundation_models/data/SDE/external_evaluations_and_data/20250726_real_world_with_5_fold_cross_validation/20250514_bisde_5_fold_cross_validation_paths_no_nans/bisde_real_world_cv_results.json"
    #     ),
    #     "BISDE(20250514, (u^(0,..,3), exp(u), sin(u)) Library Functions)": Path(
    #         "/cephfs_projects/foundation_models/data/SDE/external_evaluations_and_data/20250726_real_world_with_5_fold_cross_validation/20250514_bisde_5_fold_cross_validation_paths_no_nans/bisde_real_world_cv_our_basis_results.json"
    #     ),
    #     "FIM fixed Softmax dim., 05-03-2033, Epoch 138": Path(
    #         "/cephfs_projects/foundation_models/data/SDE/saved_evaluation_results/20250329_neurips_submission_evaluations/real_world_cross_validation_vf_and_paths_evaluation/05140056_fim_fixed_softmax_05-03-2033_epoch_138/model_paths.json",
    #     ),
    #     "FIM (half locations at observations) (07-14-1850) Epoch 139": Path(
    #         "/cephfs_projects/foundation_models/data/SDE/saved_evaluation_results/20250716_post_neurips_evaluations/real_world_cross_validation_vf_and_paths_evaluation/07161232_fim_location_at_obs_no_finetuning/model_paths.json"
    #     ),
    # }

    non_rebuttal_base = Path(
        "/cephfs_projects/foundation_models/data/SDE/saved_evaluation_results/20250808_not_used_in_neurips_rebuttal/real_world_cross_validation_vf_and_paths_evaluation/"
    )
    rebuttal_base = Path(
        "/cephfs_projects/foundation_models/data/SDE/saved_evaluation_results/20250808_neurips_rebuttal_evaluations/real_world_cross_validation_vf_and_paths_evaluation/"
    )

    finetune_mse_base = non_rebuttal_base / "finetune_one_step_ahead_one_em_step_mse"
    finetune_nll_base = non_rebuttal_base / "finetune_one_step_ahead_one_em_step_nll"
    finetune_nll_seed_1_base = non_rebuttal_base / "finetune_one_step_ahead_one_em_step_nll_seed_1"
    finetune_nll_5_em_step_base = non_rebuttal_base / "finetune_one_step_ahead_five_em_step_nll"
    finetune_nll_5_step_ahead_base = non_rebuttal_base / "finetune_five_step_ahead_one_em_step_nll"
    finetune_nll_512_base = non_rebuttal_base / "finetune_one_step_ahead_one_em_step_nll_512_points"
    finetune_nll_512_seed_1_base = non_rebuttal_base / "finetune_one_step_ahead_one_em_step_nll_512_points_seed_1"
    finetune_nll_512_every_10_epochs_base = non_rebuttal_base / "finetune_one_step_ahead_one_em_step_nll_512_points_every_10_epochs"
    finetune_nll_32_every_10_epochs_base = non_rebuttal_base / "finetune_one_step_ahead_one_em_step_nll_32_points_every_10_epochs"
    finetune_nll_512_lr_1e_6 = non_rebuttal_base / "finetune_one_step_ahead_one_em_step_nll_512_points_lr_1e_6"
    finetune_nll_512_lr_1e_6_every_10_epochs = (
        non_rebuttal_base / "finetune_one_step_ahead_one_em_step_nll_512_points_lr_1e_6_every_10_epochs"
    )

    latent_sde_MSE_train_subsplit_100_base = non_rebuttal_base / "latent_sde_latent_dim_4_context_dim_100_decoder_MSE_train_subsplits_100/"
    latent_sde_NLL_train_subsplit_10_base = rebuttal_base / "latent_sde_latent_dim_4_context_dim_100_decoder_NLL_train_subsplits_10/"
    latent_sde_NLL_train_len_40_base = non_rebuttal_base / "latent_sde_latent_dim_4_context_dim_100_decoder_NLL_len_train_subsplits_40/"

    finetune_neurips_rebuttal_512_points_base = (
        rebuttal_base / "finetune_for_neurips_rebuttal_one_step_ahead_one_em_step_nll_512_points_500_epochs"
    )

    models_jsons = {
        "BISDE": Path(
            "/cephfs_projects/foundation_models/data/SDE/external_evaluations_and_data/20250726_real_world_with_5_fold_cross_validation/20250514_bisde_5_fold_cross_validation_paths_no_nans/bisde_real_world_cv_our_basis_results.json"
        ),
        "No Finetune": Path(
            "/cephfs_projects/foundation_models/data/SDE/saved_evaluation_results/20250329_neurips_submission_evaluations/real_world_cross_validation_vf_and_paths_evaluation/05140056_fim_fixed_softmax_05-03-2033_epoch_138/model_paths.json",
        ),
        # "Finetune Sample MSE, Epoch 50": finetune_mse_base / "combined_outputs_epoch_49.json",
        # "Finetune Sample MSE, Epoch 100": finetune_mse_base / "combined_outputs_epoch_99.json",
        # "Finetune Sample MSE, Epoch 200": finetune_mse_base / "combined_outputs_epoch_199.json",
        # "Finetune Sample MSE, Epoch 500": finetune_mse_base / "combined_outputs_epoch_499.json",
        # "Finetune Sample MSE, Epoch Best": finetune_mse_base / "combined_outputs_best_model.json",
        # "Finetune Sample NLL, Epoch 50": finetune_nll_base / "combined_outputs_epoch_49.json",
        # "Finetune Sample NLL, Epoch 100": finetune_nll_base / "combined_outputs_epoch_99.json",
        # "Finetune Sample NLL, Epoch 200": finetune_nll_base / "combined_outputs_epoch_199.json",
        # "Finetune Sample NLL, Epoch 500": finetune_nll_base / "combined_outputs_epoch_499.json",
        # "Finetune Sample NLL, Epoch Best": finetune_nll_base / "combined_outputs_best_model.json",
        # "Finetune Sample NLL, Seed 1, Epoch 50": finetune_nll_seed_1_base / "combined_outputs_epoch_49.json",
        # "Finetune Sample NLL, Seed 1, Epoch 100": finetune_nll_seed_1_base / "combined_outputs_epoch_99.json",
        # "Finetune Sample NLL, Seed 1, Epoch 200": finetune_nll_seed_1_base / "combined_outputs_epoch_199.json",
        # "Finetune Sample NLL, Seed 1, Epoch 500": finetune_nll_seed_1_base / "combined_outputs_epoch_499.json",
        # "Finetune Sample NLL, Seed 1, Epoch Best": finetune_nll_seed_1_base / "combined_outputs_best_model.json",
        # "Finetune Sample NLL, 5 EM Step, Epoch 50": finetune_nll_5_em_step_base / "combined_outputs_epoch_49.json",
        # "Finetune Sample NLL, 5 EM Step, Epoch 100": finetune_nll_5_em_step_base / "combined_outputs_epoch_99.json",
        # "Finetune Sample NLL, 5 EM Step, Epoch 200": finetune_nll_5_em_step_base / "combined_outputs_epoch_199.json",
        # "Finetune Sample NLL, 5 EM Step, Epoch 500": finetune_nll_5_em_step_base / "combined_outputs_epoch_499.json",
        # "Finetune Sample NLL, 5 EM Step, Epoch Best": finetune_nll_5_em_step_base / "combined_outputs_best_model.json",
        # "Finetune Sample NLL, 5 Step Ahead, Epoch 50": finetune_nll_5_step_ahead_base / "combined_outputs_epoch_49.json",
        # "Finetune Sample NLL, 5 Step Ahead, Epoch 100": finetune_nll_5_step_ahead_base / "combined_outputs_epoch_99.json",
        # "Finetune Sample NLL, 5 Step Ahead, Epoch 200": finetune_nll_5_step_ahead_base / "combined_outputs_epoch_199.json",
        # "Finetune Sample NLL, 5 Step Ahead, Epoch 500": finetune_nll_5_step_ahead_base / "combined_outputs_epoch_499.json",
        # "Finetune Sample NLL, 5 Step Ahead, Epoch Best": finetune_nll_5_step_ahead_base / "combined_outputs_best_model.json",
        # "Finetune Sample NLL, 512 Points, Epoch 50": finetune_nll_512_base / "combined_outputs_epoch_49.json",
        # "Finetune Sample NLL, 512 Points, Epoch 100": finetune_nll_512_base / "combined_outputs_epoch_99.json",
        # "Finetune Sample NLL, 512 Points, Epoch 200": finetune_nll_512_base / "combined_outputs_epoch_199.json",
        # "Finetune Sample NLL, 512 Points, Epoch 500": finetune_nll_512_base / "combined_outputs_epoch_499.json",
        # "Finetune Sample NLL, 512 Points, Epoch Best": finetune_nll_512_base / "combined_outputs_best_model.json",
        # "Finetune Sample NLL, 512 Points, Seed 1, Epoch 50": finetune_nll_seed_1_base / "combined_outputs_epoch_49.json",
        # "Finetune Sample NLL, 512 Points, Seed 1, Epoch 100": finetune_nll_seed_1_base / "combined_outputs_epoch_99.json",
        # "Finetune Sample NLL, 512 Points, Seed 1, Epoch 200": finetune_nll_seed_1_base / "combined_outputs_epoch_199.json",
        # "Finetune Sample NLL, 512 Points, Seed 1, Epoch 500": finetune_nll_seed_1_base / "combined_outputs_epoch_499.json",
        # "Finetune Sample NLL, 512 Points, Seed 1, Epoch Best": finetune_nll_seed_1_base / "combined_outputs_best_model.json",
        # "Finetune Sample NLL, 512 Points, Short, Epoch 10": finetune_nll_512_every_10_epochs_base / "combined_outputs_epoch_9.json",
        # "Finetune Sample NLL, 512 Points, Short, Epoch 20": finetune_nll_512_every_10_epochs_base / "combined_outputs_epoch_19.json",
        # "Finetune Sample NLL, 512 Points, Short, Epoch 30": finetune_nll_512_every_10_epochs_base / "combined_outputs_epoch_29.json",
        # "Finetune Sample NLL, 512 Points, Short, Epoch 40": finetune_nll_512_every_10_epochs_base / "combined_outputs_epoch_39.json",
        # "Finetune Sample NLL, 512 Points, Short, Epoch 50": finetune_nll_512_every_10_epochs_base / "combined_outputs_epoch_49.json",
        # "Finetune Sample NLL, 512 Points, Short, Epoch 60": finetune_nll_512_every_10_epochs_base / "combined_outputs_epoch_59.json",
        # "Finetune Sample NLL, 512 Points, Short, Epoch 70": finetune_nll_512_every_10_epochs_base / "combined_outputs_epoch_69.json",
        # "Finetune Sample NLL, 512 Points, Short, Epoch 80": finetune_nll_512_every_10_epochs_base / "combined_outputs_epoch_79.json",
        # "Finetune Sample NLL, 512 Points, Short, Epoch 90": finetune_nll_512_every_10_epochs_base / "combined_outputs_epoch_89.json",
        # "Finetune Sample NLL, 512 Points, Short, Epoch 100": finetune_nll_512_every_10_epochs_base / "combined_outputs_epoch_99.json",
        # "Finetune Sample NLL, 32 Points, Short, Epoch 10": finetune_nll_32_every_10_epochs_base / "combined_outputs_epoch_9.json",
        # "Finetune Sample NLL, 32 Points, Short, Epoch 20": finetune_nll_32_every_10_epochs_base / "combined_outputs_epoch_19.json",
        # "Finetune Sample NLL, 32 Points, Short, Epoch 30": finetune_nll_32_every_10_epochs_base / "combined_outputs_epoch_29.json",
        # "Finetune Sample NLL, 32 Points, Short, Epoch 40": finetune_nll_32_every_10_epochs_base / "combined_outputs_epoch_39.json",
        # "Finetune Sample NLL, 32 Points, Short, Epoch 50": finetune_nll_32_every_10_epochs_base / "combined_outputs_epoch_49.json",
        # "Finetune Sample NLL, 32 Points, Short, Epoch 60": finetune_nll_32_every_10_epochs_base / "combined_outputs_epoch_59.json",
        # "Finetune Sample NLL, 32 Points, Short, Epoch 70": finetune_nll_32_every_10_epochs_base / "combined_outputs_epoch_69.json",
        # "Finetune Sample NLL, 32 Points, Short, Epoch 80": finetune_nll_32_every_10_epochs_base / "combined_outputs_epoch_79.json",
        # "Finetune Sample NLL, 32 Points, Short, Epoch 90": finetune_nll_32_every_10_epochs_base / "combined_outputs_epoch_89.json",
        # "Finetune Sample NLL, 32 Points, Short, Epoch 100": finetune_nll_32_every_10_epochs_base / "combined_outputs_epoch_99.json",
        # "Finetune Sample NLL, 512 Points, lr 1e-6, Epoch 50": finetune_nll_512_lr_1e_6 / "combined_outputs_epoch_49.json",
        # "Finetune Sample NLL, 512 Points, lr 1e-6, Epoch 100": finetune_nll_512_lr_1e_6 / "combined_outputs_epoch_99.json",
        # "Finetune Sample NLL, 512 Points, lr 1e-6, Epoch 200": finetune_nll_512_lr_1e_6 / "combined_outputs_epoch_199.json",
        # "Finetune Sample NLL, 512 Points, lr 1e-6, Epoch 500": finetune_nll_512_lr_1e_6 / "combined_outputs_epoch_499.json",
        # "Finetune Sample NLL, 512 Points, lr 1e-6, Epoch Best": finetune_nll_512_lr_1e_6 / "combined_outputs_best_model.json",
        # "Finetune Sample NLL, 512 Points, Short, lr 1e-6, Epoch 10": finetune_nll_512_lr_1e_6_every_10_epochs
        # / "combined_outputs_epoch_9.json",
        # "Finetune Sample NLL, 512 Points, Short, lr 1e-6, Epoch 20": finetune_nll_512_lr_1e_6_every_10_epochs
        # / "combined_outputs_epoch_19.json",
        # "Finetune Sample NLL, 512 Points, Short, lr 1e-6, Epoch 30": finetune_nll_512_lr_1e_6_every_10_epochs
        # / "combined_outputs_epoch_29.json",
        # "Finetune Sample NLL, 512 Points, Short, lr 1e-6, Epoch 40": finetune_nll_512_lr_1e_6_every_10_epochs
        # / "combined_outputs_epoch_39.json",
        # "Finetune Sample NLL, 512 Points, Short, lr 1e-6, Epoch 50": finetune_nll_512_lr_1e_6_every_10_epochs
        # / "combined_outputs_epoch_49.json",
        # "Finetune Sample NLL, 512 Points, Short, lr 1e-6, Epoch 60": finetune_nll_512_lr_1e_6_every_10_epochs
        # / "combined_outputs_epoch_59.json",
        # "Finetune Sample NLL, 512 Points, Short, lr 1e-6, Epoch 70": finetune_nll_512_lr_1e_6_every_10_epochs
        # / "combined_outputs_epoch_69.json",
        # "Finetune Sample NLL, 512 Points, Short, lr 1e-6, Epoch 80": finetune_nll_512_lr_1e_6_every_10_epochs
        # / "combined_outputs_epoch_79.json",
        # "Finetune Sample NLL, 512 Points, Short, lr 1e-6, Epoch 90": finetune_nll_512_lr_1e_6_every_10_epochs
        # / "combined_outputs_epoch_89.json",
        # "Finetune Sample NLL, 512 Points, Short, lr 1e-6, Epoch 100": finetune_nll_512_lr_1e_6_every_10_epochs
        # / "combined_outputs_epoch_99.json",
        # "LatentSDE, MSE objective, 100 train subsplits, Epoch 500": latent_sde_MSE_train_subsplit_100_base
        # / "combined_outputs_epoch_499.json",
        # "LatentSDE, MSE objective, 100 train subsplits, Epoch 1000": latent_sde_MSE_train_subsplit_100_base
        # / "combined_outputs_epoch_999.json",
        # "LatentSDE, MSE objective, 100 train subsplits, Epoch 2000": latent_sde_MSE_train_subsplit_100_base
        # / "combined_outputs_epoch_1999.json",
        # "LatentSDE, MSE objective, 100 train subsplits, Epoch 5000": latent_sde_MSE_train_subsplit_100_base
        # / "combined_outputs_epoch_4999.json",
        # "LatentSDE, NLL objective, 10 train subsplits, Epoch 500": latent_sde_NLL_train_subsplit_10_base
        # / "combined_outputs_epoch_499.json",
        # "LatentSDE, NLL objective, 10 train subsplits, Epoch 1000": latent_sde_NLL_train_subsplit_10_base
        # / "combined_outputs_epoch_999.json",
        # "LatentSDE, NLL objective, 10 train subsplits, Epoch 2000": latent_sde_NLL_train_subsplit_10_base
        # / "combined_outputs_epoch_1999.json",
        # "LatentSDE, NLL objective, 10 train subsplits, Epoch 5000": latent_sde_NLL_train_subsplit_10_base
        # / "combined_outputs_epoch_4999.json",
        "Finetune Sample NLL, NeurIPS Rebuttal, 512 Points, Epoch 10": finetune_neurips_rebuttal_512_points_base
        / "combined_outputs_epoch_9.json",
        "Finetune Sample NLL, NeurIPS Rebuttal, 512 Points, Epoch 20": finetune_neurips_rebuttal_512_points_base
        / "combined_outputs_epoch_19.json",
        "Finetune Sample NLL, NeurIPS Rebuttal, 512 Points, Epoch 30": finetune_neurips_rebuttal_512_points_base
        / "combined_outputs_epoch_29.json",
        "Finetune Sample NLL, NeurIPS Rebuttal, 512 Points, Epoch 40": finetune_neurips_rebuttal_512_points_base
        / "combined_outputs_epoch_39.json",
        "Finetune Sample NLL, NeurIPS Rebuttal, 512 Points, Epoch 50": finetune_neurips_rebuttal_512_points_base
        / "combined_outputs_epoch_49.json",
        "Finetune Sample NLL, NeurIPS Rebuttal, 512 Points, Epoch 60": finetune_neurips_rebuttal_512_points_base
        / "combined_outputs_epoch_59.json",
        "Finetune Sample NLL, NeurIPS Rebuttal, 512 Points, Epoch 70": finetune_neurips_rebuttal_512_points_base
        / "combined_outputs_epoch_69.json",
        "Finetune Sample NLL, NeurIPS Rebuttal, 512 Points, Epoch 80": finetune_neurips_rebuttal_512_points_base
        / "combined_outputs_epoch_79.json",
        "Finetune Sample NLL, NeurIPS Rebuttal, 512 Points, Epoch 90": finetune_neurips_rebuttal_512_points_base
        / "combined_outputs_epoch_89.json",
        "Finetune Sample NLL, NeurIPS Rebuttal, 512 Points, Epoch 100": finetune_neurips_rebuttal_512_points_base
        / "combined_outputs_epoch_99.json",
        "Finetune Sample NLL, NeurIPS Rebuttal, 512 Points, Epoch 200": finetune_neurips_rebuttal_512_points_base
        / "combined_outputs_epoch_199.json",
        "Finetune Sample NLL, NeurIPS Rebuttal, 512 Points, Epoch 500": finetune_neurips_rebuttal_512_points_base
        / "combined_outputs_epoch_499.json",
    }

    datasets_to_evaluate: list[str] = [
        "wind",
        "oil",
        "fb",
        "tsla",
    ]

    expected_num_total_splits = 5

    only_use_loaded_evaluations = False

    metrics_to_evaluate = [
        "mmd",
    ]

    metric_evaluations_to_load: list[Path] = [
        # Path(
        #     "/cephfs/users/seifner/repos/FIM/evaluations/real_world_cross_validation_metrics_tables/07231113_finetuning_samplings/metric_evaluations_jsons/"
        #     "/cephfs/users/seifner/repos/FIM/evaluations/real_world_cross_validation_metrics_tables/07231447_finetuning_sampling_nll_512_points/metric_evaluations_jsons/"
        # ),
    ]

    # models_order = [
    #     # "FIM (05-03-2033)",
    #     # "FIM (05-06-2300)",
    #     # "BISDE(20250510, BISDE Library Functions)",
    #     # "BISDE(20250510, (u^(0,..,3), exp(u), sin(u)) Library Functions)",
    #     "BISDE(20250514, BISDE Library Functions)",
    #     "BISDE(20250514, (u^(0,..,3), exp(u), sin(u)) Library Functions)",
    #     # "FIM fixed linear Attn., 04-28-0941, Epoch 040",
    #     # "FIM fixed linear Attn., 04-28-0941, Epoch 070",
    #     # "FIM fixed linear Attn., 04-28-0941, Epoch 100",
    #     # "FIM fixed linear Attn., 04-28-0941, Epoch 125",
    #     # "FIM fixed Softmax dim., 05-03-2033, Epoch 040",
    #     # "FIM fixed Softmax dim., 05-03-2033, Epoch 070",
    #     # "FIM fixed Softmax dim., 05-03-2033, Epoch 100",
    #     # "FIM fixed Softmax dim., 05-03-2033, Epoch 125",
    #     "FIM fixed Softmax dim., 05-03-2033, Epoch 138",
    #     "FIM (half locations at observations) (07-14-1850) Epoch 139",
    # ]

    models_order = [
        "BISDE",
        "No Finetune",
        # "Finetune Sample MSE, Epoch 50",
        # "Finetune Sample MSE, Epoch 100",
        # "Finetune Sample MSE, Epoch 200",
        # "Finetune Sample MSE, Epoch 500",
        # "Finetune Sample MSE, Epoch Best",
        # "Finetune Sample NLL, Epoch 50",
        # "Finetune Sample NLL, Epoch 100",
        # "Finetune Sample NLL, Epoch 200",
        # "Finetune Sample NLL, Epoch 500",
        # "Finetune Sample NLL, Epoch Best",
        # "Finetune Sample NLL, Seed 1, Epoch 50",
        # "Finetune Sample NLL, Seed 1, Epoch 100",
        # "Finetune Sample NLL, Seed 1, Epoch 200",
        # "Finetune Sample NLL, Seed 1, Epoch 500",
        # "Finetune Sample NLL, Seed 1, Epoch Best",
        # "Finetune Sample NLL, 5 EM Step, Epoch 50",
        # "Finetune Sample NLL, 5 EM Step, Epoch 100",
        # "Finetune Sample NLL, 5 EM Step, Epoch 200",
        # "Finetune Sample NLL, 5 EM Step, Epoch 500",
        # "Finetune Sample NLL, 5 EM Step, Epoch Best",
        # "Finetune Sample NLL, 5 Step Ahead, Epoch 50",
        # "Finetune Sample NLL, 5 Step Ahead, Epoch 100",
        # "Finetune Sample NLL, 5 Step Ahead, Epoch 200",
        # "Finetune Sample NLL, 5 Step Ahead, Epoch 500",
        # "Finetune Sample NLL, 5 Step Ahead, Epoch Best",
        # "Finetune Sample NLL, 512 Points, Epoch 50",
        # "Finetune Sample NLL, 512 Points, Epoch 100",
        # "Finetune Sample NLL, 512 Points, Epoch 200",
        # "Finetune Sample NLL, 512 Points, Epoch 500",
        # "Finetune Sample NLL, 512 Points, Epoch Best",
        # "Finetune Sample NLL, 512 Points, Seed 1, Epoch 50",
        # "Finetune Sample NLL, 512 Points, Seed 1, Epoch 100",
        # "Finetune Sample NLL, 512 Points, Seed 1, Epoch 200",
        # "Finetune Sample NLL, 512 Points, Seed 1, Epoch 500",
        # "Finetune Sample NLL, 512 Points, Seed 1, Epoch Best",
        # "Finetune Sample NLL, 512 Points, Short, Epoch 10",
        # "Finetune Sample NLL, 512 Points, Short, Epoch 20",
        # "Finetune Sample NLL, 512 Points, Short, Epoch 30",
        # "Finetune Sample NLL, 512 Points, Short, Epoch 40",
        # "Finetune Sample NLL, 512 Points, Short, Epoch 50",
        # "Finetune Sample NLL, 512 Points, Short, Epoch 60",
        # "Finetune Sample NLL, 512 Points, Short, Epoch 70",
        # "Finetune Sample NLL, 512 Points, Short, Epoch 80",
        # "Finetune Sample NLL, 512 Points, Short, Epoch 90",
        # "Finetune Sample NLL, 512 Points, Short, Epoch 100",
        # "Finetune Sample NLL, 32 Points, Short, Epoch 10",
        # "Finetune Sample NLL, 32 Points, Short, Epoch 20",
        # "Finetune Sample NLL, 32 Points, Short, Epoch 30",
        # "Finetune Sample NLL, 32 Points, Short, Epoch 40",
        # "Finetune Sample NLL, 32 Points, Short, Epoch 50",
        # "Finetune Sample NLL, 32 Points, Short, Epoch 60",
        # "Finetune Sample NLL, 32 Points, Short, Epoch 70",
        # "Finetune Sample NLL, 32 Points, Short, Epoch 80",
        # "Finetune Sample NLL, 32 Points, Short, Epoch 90",
        # "Finetune Sample NLL, 32 Points, Short, Epoch 100",
        # "Finetune Sample NLL, 512 Points, lr 1e-6, Epoch 50",
        # "Finetune Sample NLL, 512 Points, lr 1e-6, Epoch 100",
        # "Finetune Sample NLL, 512 Points, lr 1e-6, Epoch 200",
        # "Finetune Sample NLL, 512 Points, lr 1e-6, Epoch 500",
        # "Finetune Sample NLL, 512 Points, lr 1e-6, Epoch Best",
        # "Finetune Sample NLL, 512 Points, Short, lr 1e-6, Epoch 10",
        # "Finetune Sample NLL, 512 Points, Short, lr 1e-6, Epoch 20",
        # "Finetune Sample NLL, 512 Points, Short, lr 1e-6, Epoch 30",
        # "Finetune Sample NLL, 512 Points, Short, lr 1e-6, Epoch 40",
        # "Finetune Sample NLL, 512 Points, Short, lr 1e-6, Epoch 50",
        # "Finetune Sample NLL, 512 Points, Short, lr 1e-6, Epoch 60",
        # "Finetune Sample NLL, 512 Points, Short, lr 1e-6, Epoch 70",
        # "Finetune Sample NLL, 512 Points, Short, lr 1e-6, Epoch 80",
        # "Finetune Sample NLL, 512 Points, Short, lr 1e-6, Epoch 90",
        # "Finetune Sample NLL, 512 Points, Short, lr 1e-6, Epoch 100",
        # "LatentSDE, MSE objective, 100 train subsplits, Epoch 500",
        # "LatentSDE, MSE objective, 100 train subsplits, Epoch 1000",
        # "LatentSDE, MSE objective, 100 train subsplits, Epoch 2000",
        # "LatentSDE, MSE objective, 100 train subsplits, Epoch 5000",
        # "LatentSDE, NLL objective, 10 train subsplits, Epoch 500",
        # "LatentSDE, NLL objective, 10 train subsplits, Epoch 1000",
        # "LatentSDE, NLL objective, 10 train subsplits, Epoch 2000",
        # "LatentSDE, NLL objective, 10 train subsplits, Epoch 5000",
        "Finetune Sample NLL, NeurIPS Rebuttal, 512 Points, Epoch 10",
        "Finetune Sample NLL, NeurIPS Rebuttal, 512 Points, Epoch 20",
        "Finetune Sample NLL, NeurIPS Rebuttal, 512 Points, Epoch 30",
        "Finetune Sample NLL, NeurIPS Rebuttal, 512 Points, Epoch 40",
        "Finetune Sample NLL, NeurIPS Rebuttal, 512 Points, Epoch 50",
        "Finetune Sample NLL, NeurIPS Rebuttal, 512 Points, Epoch 60",
        "Finetune Sample NLL, NeurIPS Rebuttal, 512 Points, Epoch 70",
        "Finetune Sample NLL, NeurIPS Rebuttal, 512 Points, Epoch 80",
        "Finetune Sample NLL, NeurIPS Rebuttal, 512 Points, Epoch 90",
        "Finetune Sample NLL, NeurIPS Rebuttal, 512 Points, Epoch 100",
        "Finetune Sample NLL, NeurIPS Rebuttal, 512 Points, Epoch 200",
        "Finetune Sample NLL, NeurIPS Rebuttal, 512 Points, Epoch 500",
    ]
    datasets_order = datasets_to_evaluate
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
            data_id=(dataset, split, expected_num_total_splits),
            data_paths_json=reference_paths_json,
            data_vector_fields_json=None,
            metric_id=metric,
            metric_value=None,  # input later
        )
        for model in models_jsons.keys()
        for dataset in datasets_to_evaluate
        for split in range(expected_num_total_splits)
        for metric in metrics_to_evaluate
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
        dataset, split, total_splits = eval.data_id
        file_name = (
            f"mod_{eval.model_id.replace(' ', '_')}_met_{eval.metric_id}_dataset_{dataset}_split_{split}_total_splits_{total_splits}.json"
        )
        eval.to_json(json_save_dir / file_name)

    # compute metric for missing evaluations
    all_evaluations = loaded_evaluations

    if len(to_evaluate) > 0:
        print("Loading ground-truth and model data.")

        data_paths: list[dict] = json.load(open(reference_paths_json, "r"))
        models_data: dict = {model_name: json.load(open(model_json, "r")) for model_name, model_json in models_jsons.items()}

        print(f"Data paths keys: {[d['name'] for d in data_paths]}")
        print("Models keys: ")
        for model_name, model_systems in models_data.items():
            print(f"Model: {model_name}, Dataset: {[system['name'] for system in model_systems]}")

        for eval in (pbar := tqdm(to_evaluate, leave=False)):
            dataset, split, total_splits = eval.data_id
            pbar.set_description(f"Processing metric={eval.metric_id}, model={eval.model_id}, {dataset=}, {split=}, {total_splits=}.")

            start_time = datetime.now()

            if eval.metric_id == "mmd":
                reference_paths: np.ndarray = get_reference_paths(deepcopy(data_paths), dataset, split, total_splits)
                model_paths: np.ndarray = get_model_data(
                    deepcopy(models_data[eval.model_id]), "synthetic_paths", dataset, split, total_splits
                )

                if model_paths is None or np.isnan(model_paths).any():
                    eval.metric_value = np.nan

                else:
                    assert reference_paths.shape == model_paths.shape
                    assert reference_paths.ndim == 4  # expect [1, P, T, 1], 1 set of P paths of length T
                    assert reference_paths.shape[0] == 1

                    # ksig interface expects ndim == 3
                    reference_paths = reference_paths[0]
                    model_paths = model_paths[0]

                    eval.metric_value = compute_mmd(reference_paths, model_paths)

            # record computation time
            end_time = datetime.now()
            computation_time = (end_time - start_time).total_seconds()

            print(
                f"Last result: metric={eval.metric_id}, model={eval.model_id}, {dataset=}, {split=}, {total_splits=}, value={eval.metric_value}, {computation_time=} seconds."
            )

            # save results as json
            file_name = f"mod_{eval.model_id.replace(' ', '_')}_met_{eval.metric_id}_dataset_{dataset}_split_{split}_total_splits_{total_splits}.json"
            eval.to_json(json_save_dir / file_name)

            all_evaluations.append(eval)

    for metric in metrics_to_evaluate:
        negative_values = [None, "clip", "abs"] if metric == "mmd" else [None]

        for handle_negative_values in negative_values:
            df_count_exps, df_count_non_nans, df_mean, df_std, df_mean_plus_std, df_mean_bracket_std_times_10, df_mean_bracket_std = (
                real_world_metric_table(all_evaluations, metric, models_order, datasets_order, precision, handle_negative_values)
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
