import itertools
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
    """
    vector_fields_of_system = [d for d in all_vector_fields if d["name"] == system]

    # should contain exactly one pair of vector fields for each system
    if len(vector_fields_of_system) == 1:
        return np.array(vector_fields_of_system[0][vector_field_key][experiment_num])

    elif len(vector_fields_of_system) == 0:
        raise ValueError(f"Could not find ground-truth {vector_field_key} of system {system}.")

    else:
        raise ValueError(f"Found {len(vector_fields_of_system)} vector fields {vector_field_key} for system {system}")


def get_model_data(all_model_data: list[dict], data_key: str, system: str, tau: str, experiment_num: int) -> np.ndarray:
    """
    Extract result of one experiment from one model from a loaded json file.
    """
    model_data_of_system = [d for d in all_model_data if (d["name"] == system) and (d["tau"] == tau)]

    # should contain exactly one set of data for each system
    if len(model_data_of_system) == 1:
        return np.array(model_data_of_system[0][data_key][experiment_num])

    elif len(model_data_of_system) == 0:
        raise ValueError(f"Could not find {data_key} of system {system} with tau {tau}.")

    else:
        raise ValueError(f"Found {len(model_data_of_system)} sets of {data_key} for system {system} with tau {tau}.")


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
    """
    assert handle_negative_values in [None, "clip", "abs"]

    # all evaluations with metric to dataframe
    rows = [
        {
            "model": eval.model_id,
            "system": eval.data_id[0],
            "tau": eval.data_id[1],
            "exp": eval.data_id[2],
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
    df_count_exps = deepcopy(df[["system", "tau", "model", "metric_value"]])
    df_count_exps = df_count_exps.groupby(["system", "tau", "model"]).size()
    df_count_exps = df_count_exps.unstack(0)

    # count number of experiments_without Nans
    df_count_non_nans = df.groupby(["system", "tau", "model"]).count().drop("exp", axis=1)
    df_count_non_nans = df_count_non_nans["metric_value"].unstack(0)

    # count Nans and depict them as stars
    df_star_nans = deepcopy(df[["system", "tau", "model", "metric_value"]])
    df_star_nans["metric_is_nan"] = np.isnan(df["metric_value"])
    df_star_nans = df_star_nans.drop("metric_value", axis=1)

    df_star_nans = df_star_nans.groupby(["system", "tau", "model"]).agg(nans_to_stars)
    df_star_nans = df_star_nans["metric_is_nan"].unstack(0)

    # sometimes mmd can be negative, if the paths are really good
    if handle_negative_values == "clip":
        df["metric_value"] = np.clip(df["metric_value"], a_min=0.0, a_max=np.inf)

    elif handle_negative_values == "abs":
        df["metric_value"] = np.abs(df["metric_value"])

    # mean without Nans
    df_mean = (
        df.groupby(["system", "tau", "model"]).agg(lambda x: str(x.dropna().mean()) if (len(x.dropna()) != 0) else "-").drop("exp", axis=1)
    )
    df_mean = df_mean["metric_value"].unstack(0)

    # std without Nans
    df_std = (
        df.groupby(["system", "tau", "model"]).agg(lambda x: str(x.dropna().std()) if (len(x.dropna()) != 0) else "-").drop("exp", axis=1)
    )
    df_std = df_std["metric_value"].unstack(0)

    # mean and std in one cell; formatted as "mean $\pm$ std"
    df_mean_plus_std = df.groupby(["system", "tau", "model"]).agg(partial(mean_plus_std_agg, precision=precision)).drop("exp", axis=1)
    df_mean_plus_std = df_mean_plus_std["metric_value"].unstack(0)

    # mean and std in one cell; formatted with bracket notation
    df_mean_bracket_std = df.groupby(["system", "tau", "model"]).agg(partial(mean_bracket_std_agg, precision=precision)).drop("exp", axis=1)
    df_mean_bracket_std = df_mean_bracket_std["metric_value"].unstack(0)

    # mean and std in one cell; formatted with bracket notation; multiply by 10 first
    df_mean_bracket_std_times_10 = deepcopy(df)
    df_mean_bracket_std_times_10["metric_value"] = df_mean_bracket_std_times_10["metric_value"] * 10
    df_mean_bracket_std_times_10 = (
        df_mean_bracket_std_times_10.groupby(["system", "tau", "model"])
        .agg(partial(mean_bracket_std_agg, precision=precision))
        .drop("exp", axis=1)
    )
    df_mean_bracket_std_times_10 = df_mean_bracket_std_times_10["metric_value"].unstack(0)

    # add number of Nan experiments as stars to each cell
    df_mean = df_mean + " " + df_star_nans
    df_std = df_std + " " + df_star_nans
    df_mean_plus_std = df_mean_plus_std + " " + df_star_nans
    df_mean_bracket_std = df_mean_bracket_std + " " + df_star_nans

    # reorder columns
    df_count_exps = df_count_exps[systems_order]
    df_count_non_nans = df_count_non_nans[systems_order]
    df_mean = df_mean[systems_order]
    df_std = df_std[systems_order]
    df_mean_plus_std = df_mean_plus_std[systems_order]
    df_mean_bracket_std = df_mean_bracket_std[systems_order]
    df_mean_bracket_std_times_10 = df_mean_bracket_std_times_10[systems_order]

    return df_count_exps, df_count_non_nans, df_mean, df_std, df_mean_plus_std, df_mean_bracket_std_times_10, df_mean_bracket_std


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    dataset_descr = "synthetic_systems_metrics_tables"

    # How to name experiments
    # experiment_descr = "combined_table_fim_fim_cont_bisde_sparsegp"
    experiment_descr = "develop"

    project_path = "/cephfs/users/seifner/repos/FIM"

    data_paths_json = Path(
        "/cephfs_projects/foundation_models/data/SDE/evaluation/20250129_coarse_synthetic_systems_5000_points_data/ksig_reference_paths.json"
    )
    data_vector_fields_json = Path(
        "/cephfs_projects/foundation_models/data/SDE/evaluation/20250129_coarse_synthetic_systems_5000_points_data/ground_truth_drift_diffusion.json"
    )

    models_jsons = {
        "SparseGP": Path(
            "/cephfs_projects/foundation_models/data/SDE/evaluation/20250129_coarse_synthetic_systems_5000_points_data/opper_sparse_gp_evaluations_at_locations_paths_coarse_synth_data.json",
        ),
        "SparseGP(ICML Submission)": Path(
            "/cephfs_projects/foundation_models/data/SDE/table_sanity_check_data/sparse_gp_sparse_observations_many_paths.json"
        ),
        "BISDE": Path(
            "/cephfs_projects/foundation_models/data/SDE/evaluation/20250129_coarse_synthetic_systems_5000_points_data/bisde_experiments_friday_full.json"
        ),
        "BISDE(ICML Submission)": Path(
            "/cephfs_projects/foundation_models/data/SDE/table_sanity_check_data/bisde_experiments_friday_full.json"
        ),
        "FIM(Paper)": Path(
            "/cephfs_projects/foundation_models/data/SDE/evaluation/20250129_coarse_synthetic_systems_5000_points_data/20M_trained_even_longer_synthetic_paths.json"
        ),
        "FIM(Cont. with unary-binary)": Path(
            "/cephfs_projects/foundation_models/data/SDE/evaluation/20250129_coarse_synthetic_systems_5000_points_data/20M_cont_train_icml_model_with_unary_binary_data_mixed_in.json"
        ),
    }
    apply_sqrt_to_diffusion = [
        "BISDE",
        "BISDE(ICML Submission)",
        "SparseGP",
        "SparseGP(ICML Submission)",
    ]

    models_to_evaluate = [
        "SparseGP",
        "SparseGP(ICML Submission)",
        "BISDE",
        "BISDE(ICML Submission)",
        "FIM(Paper)",
        "FIM(Cont. with unary-binary)",
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
        0.01,
        0.02,
    ]
    experiment_count = 5
    mmd_max_num_paths = 100

    metrics_to_evaluate = [
        "mse_drift",
        "mse_diffusion",
        "mmd",
    ]

    metric_evaluations_to_load: list[Path] = [
        Path(
            "/cephfs/users/seifner/repos/FIM/saved_evaluations/20250311_synthetic_systems_metrics_tables/03100928_sparsegp_evaluation/metric_evaluations_jsons"
        ),
        Path(
            "/cephfs/users/seifner/repos/FIM/saved_evaluations/20250311_synthetic_systems_metrics_tables/03100931_bisde_evaluation/metric_evaluations_jsons"
        ),
        Path(
            "/cephfs/users/seifner/repos/FIM/saved_evaluations/20250311_synthetic_systems_metrics_tables/03100934_fim_paper_model_evaluation/metric_evaluations_jsons"
        ),
        Path(
            "/cephfs/users/seifner/repos/FIM/saved_evaluations/20250311_synthetic_systems_metrics_tables/03100939_fim_paper_model_cont_unary_binary_evaluation/metric_evaluations_jsons"
        ),
        Path(
            "/cephfs/users/seifner/repos/FIM/saved_evaluations/20250311_synthetic_systems_metrics_tables/03141800_bisde_from_icml_submission/metric_evaluations_jsons"
        ),
        Path(
            "/cephfs/users/seifner/repos/FIM/saved_evaluations/20250311_synthetic_systems_metrics_tables/03141807_sparsegp_from_icml_submission/metric_evaluations_jsons"
        ),
    ]

    # tables config
    models_order = [
        "SparseGP",
        "SparseGP(ICML Submission)",
        "BISDE",
        "BISDE(ICML Submission)",
        "FIM(Paper)",
        "FIM(Cont. with unary-binary)",
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
            data_id=(system, tau, experiment_num, mmd_max_num_paths),
            data_paths_json=data_paths_json,
            data_vector_fields_json=data_vector_fields_json,
            metric_id=metric,
            metric_value=None,  # input later
        )
        for model, system, tau, experiment_num, metric in itertools.product(
            models_to_evaluate, systems_to_evaluate, taus_to_evaluate, range(experiment_count), metrics_to_evaluate
        )
    ]

    print("Loading saved evaluations.")
    # load saved evaluations; remove those not already contained in all_evaluations
    loaded_evaluations: list[MetricEvaluation] = load_metric_evaluations_from_dirs(metric_evaluations_to_load)
    loaded_evaluations: list[MetricEvaluation] = [eval for eval in loaded_evaluations if eval in all_evaluations]
    to_evaluate: list[MetricEvaluation] = [eval for eval in all_evaluations if eval not in loaded_evaluations]

    # prepare directory to save evaluations in save
    json_save_dir: Path = evaluation_dir / "metric_evaluations_jsons"
    json_save_dir.mkdir(exist_ok=True, parents=True)

    for eval in loaded_evaluations:
        system, tau, exp, max_num_paths = eval.data_id
        file_name = f"mod_{eval.model_id.replace(' ', '_')}_met_{eval.metric_id}_sys_{system}_tau_{tau}_exp_{exp}.json"
        eval.to_json(json_save_dir / file_name)

    # compute metric for missing evaluations
    all_evaluations = loaded_evaluations
    mmd_ground_truth_cache = {}  # K_xx for the ground truth data is the same for all models, so we cache it

    if len(to_evaluate) > 0:
        print("Loading ground-truth and model data.")

        data_paths: dict = json.load(open(data_paths_json, "r"))
        data_vector_fields: dict = json.load(open(data_vector_fields_json, "r"))
        models_data: dict = {model_name: json.load(open(model_json, "r")) for model_name, model_json in models_jsons.items()}

        print(f"Data paths keys: {[d['name'] for d in data_paths]}")
        print(f"Data vector fields keys: {[d['name'] for d in data_vector_fields]}")
        print("Models keys: ")
        for model_name, model_systems in models_data.items():
            print(f"Model: {model_name}, Systems: {[system['name'] for system in model_systems]}")

        for eval in (pbar := tqdm(to_evaluate, leave=False)):
            system, tau, exp, mmd_max_num_paths = eval.data_id
            pbar.set_description(f"Processing metric={eval.metric_id}, model={eval.model_id}, {system=}, {tau=}, {exp=}.")

            start_time = datetime.now()

            if eval.metric_id == "mmd":
                ground_truth_paths: np.ndarray = get_ground_truth_system_paths(data_paths, system, exp)[:mmd_max_num_paths]
                model_paths: np.ndarray = get_model_data(models_data[eval.model_id], "synthetic_paths", system, tau, exp)[
                    :mmd_max_num_paths
                ]

                if not np.isnan(model_paths).any():
                    eval.metric_value = compute_mmd(ground_truth_paths, model_paths, kernel_cache=mmd_ground_truth_cache)

                else:
                    eval.metric_value = np.nan

            elif eval.metric_id in ["mse_drift", "mse_diffusion"]:
                vector_field_key = "drift_at_locations" if eval.metric_id == "mse_drift" else "diffusion_at_locations"
                ground_truth_vf: np.ndarray = get_ground_truth_vector_field(data_vector_fields, vector_field_key, system, exp)
                model_vf: np.ndarray = get_model_data(models_data[eval.model_id], vector_field_key, system, tau, exp)

                # adjust for different convention of comparison models
                if vector_field_key == "diffusion_at_location" and eval.model_id in apply_sqrt_to_diffusion:
                    model_vf = np.sqrt(np.clip(model_vf, a_min=0.0, a_max=np.inf))

                if ground_truth_vf.shape != model_vf.shape:
                    raise ValueError(
                        f"Ground-Truth vf {ground_truth_vf.shape} and estimation {model_vf.shape} must have same shape. Evaluation {eval}."
                    )

                eval.metric_value = ((ground_truth_vf - model_vf) ** 2).mean().item()

            elif eval.metric_id == "mse_drift_and_diffusion_average":
                ground_truth_drift: np.ndarray = get_ground_truth_vector_field(data_vector_fields, "drift_at_locations", system, exp)
                model_drift: np.ndarray = get_model_data(models_data[eval.model_id], "drift_at_locations", system, tau, exp)

                ground_truth_diff: np.ndarray = get_ground_truth_vector_field(data_vector_fields, "diffusion_at_locations", system, exp)
                model_diff: np.ndarray = get_model_data(models_data[eval.model_id], "diffusion_at_locations", system, tau, exp)

                # adjust for different convention of comparison models
                if eval.model_id in apply_sqrt_to_diffusion:
                    model_diff = np.sqrt(np.clip(model_diff, a_min=0.0, a_max=np.inf))

                if (ground_truth_drift.shape != model_drift.shape) or (ground_truth_diff.shape != model_diff.shape):
                    raise ValueError(
                        f"Vector fields must have same shape, got: {ground_truth_drift.shape}, {model_drift.shape}, {ground_truth_diff.shape}, {model_diff.shape}."
                    )

                mse_drift = ((ground_truth_drift - model_drift) ** 2).mean().item()
                mse_diff = ((ground_truth_diff - model_diff) ** 2).mean().item()

                eval.metric_value = (1 / 2) * (mse_drift + mse_diff)

            else:
                valid_metric_ids = ["mmd", "mse_drift", "mse_diffusion", "mse_drift_and_diffusion_average"]
                raise ValueError(f"Valid metrics are {str(valid_metric_ids)}, got {eval.metric_id} from requested evaluation {eval}.")

            # record computation time
            end_time = datetime.now()
            computation_time = (end_time - start_time).total_seconds()

            print(
                f"Last result: metric={eval.metric_id}, model={eval.model_id}, {system=}, {tau=}, {exp=}, value={eval.metric_value}, {computation_time=} seconds."
            )

            # save results as json
            file_name = f"mod_{eval.model_id.replace(' ', '_')}_met_{eval.metric_id}_sys_{system}_tau_{tau}_exp_{exp}.json"
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

            save_table(df_count_exps, metric_save_dir, "count_experiments")
            save_table(df_count_non_nans, metric_save_dir, "count_non_nans")
            save_table(df_mean, metric_save_dir, "mean")
            save_table(df_std, metric_save_dir, "std")
            save_table(df_mean_plus_std, metric_save_dir, "mean_plus_std")
            save_table(df_mean_bracket_std, metric_save_dir, "mean_bracket_std")
            save_table(df_mean_bracket_std_times_10, metric_save_dir, "mean_bracket_std_times_10")
