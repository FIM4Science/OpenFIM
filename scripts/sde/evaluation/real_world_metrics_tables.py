import itertools
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import optree
import pandas as pd
from metrics_helpers import MetricEvaluation, load_metric_evaluations_from_dirs, nans_to_stars, save_table
from mmd import compute_mmd
from tqdm import tqdm


def split_path(long_path: np.ndarray, split_into_num_paths: Optional[int]) -> np.ndarray:
    """
    Split one long path into smaller paths (e.g. for MMD computation)

    long_path: Shape [T, D]
    return: Shape [split_into_num_paths, T // split_into_num_paths, D]
    """
    split_path = np.split(long_path, indices_or_sections=split_into_num_paths, axis=0)
    return np.stack(split_path, axis=0)


def get_observed_paths(all_paths: list[dict], system: str, split_into_num_paths: Optional[int]) -> np.ndarray:
    """
    Extract paths real world system, loaded from json file.
    Split one long path into `split_into_num_paths` many paths (e.g. for MMD computation).
    """
    paths_of_system = [d for d in all_paths if d["name"] == system]

    # should contain exactly one set of paths for each system
    if len(paths_of_system) == 1:
        long_path = np.array(paths_of_system[0]["real_paths"][0], dtype=np.float64).reshape(-1, 1)  # shape [T, 1]
        return split_path(long_path, split_into_num_paths)

    elif len(paths_of_system) == 0:
        raise ValueError(f"Could not find ground-truth paths of system {system}.")

    else:
        raise ValueError(f"Found {len(paths_of_system)} sets of paths for system {system}")


def get_model_paths(all_model_data: list[dict], system: str, split_into_num_paths: Optional[int]) -> np.ndarray:
    """
    Extract paths of one system from one model from a loaded json file.
    Split one long path into `split_into_num_paths` many paths (e.g. for MMD computation).
    """
    model_data_of_system = [d for d in all_model_data if d["name"] == system]

    # should contain exactly one set of data for each system
    if len(model_data_of_system) == 1:
        long_path = np.array(model_data_of_system[0]["synthetic_paths"], dtype=np.float64).reshape(-1, 1)  # shape [T, 1]]
        return split_path(long_path, split_into_num_paths)

    elif len(model_data_of_system) == 0:
        raise ValueError(f"Could not find system {system} with tau .")

    else:
        raise ValueError(f"Found {len(model_data_of_system)} sets of system {system}.")


def real_world_metric_table(
    all_evaluations: list[MetricEvaluation],
    metric: str,
    models_order: list[str],
    systems_order: list[str],
    handle_negative_values: Optional[bool] = False,
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
    df_count_nans = deepcopy(df[["system", "model", "metric_value"]])
    df_count_nans["metric_is_nan"] = np.isnan(df["metric_value"])
    df_count_nans = df_count_nans.drop("metric_value", axis=1)

    df_star_nans = df_count_nans.groupby(["system", "model"]).agg(nans_to_stars)
    df_star_nans = df_star_nans["metric_is_nan"].unstack(0)

    # sometimes mmd can be negative, if the paths are really good
    if handle_negative_values == "clip":
        df["metric_value"] = np.clip(df["metric_value"], a_min=0.0, a_max=np.inf)

    elif handle_negative_values == "abs":
        df["metric_value"] = np.abs(df["metric_value"])

    # mean without Nans
    df_mean = df.groupby(["system", "model"]).agg(lambda x: str(x.dropna().mean()) if (len(x.dropna()) != 0) else "-")
    df_mean = df_mean["metric_value"].unstack(0)

    # add number of Nan experiments as stars to each cell
    df_mean = df_mean + " " + df_star_nans

    # reorder columns
    df_mean = df_mean[systems_order]

    return df_mean


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    dataset_descr = "real_world_metrics_tables"

    # How to name experiments
    experiment_descr = "develop"

    project_path = "/cephfs/users/seifner/repos/FIM"

    data_paths_json = Path(
        "/cephfs_projects/foundation_models/data/SDE/evaluation/20250130_bisde_real_world_oil_wind/ksig_reference_paths.json"
    )

    models_jsons = {
        "FIM(Paper)": Path("/cephfs_projects/foundation_models/data/SDE/evaluation/20250130_bisde_real_world_oil_wind/model_paths.json"),
        "FIM(Cont. with unary-binary)": Path(
            "/cephfs/users/seifner/repos/FIM/evaluations/oil_wind_tsla_fb_model_evaluation/03102335_model_cont_train_on_unary_binary_trees_with_polynomials_mixed_in/data_jsons/model_paths.json"
        ),
        "BISDE": Path(
            "/cephfs_projects/foundation_models/data/SDE/evaluation/20250130_bisde_real_world_oil_wind/bisde_inferred_vector_fields_and_paths.json"
            # "/Users/patrickseifner/repos/FIM/data/raw/SDE_bisde_on_bisde_oil_wind/bisde_vector_fields.json", # same file as the above in cephfs
        ),
        "BISDE(ICML Submission)": Path(
            "/cephfs_projects/foundation_models/data/SDE/table_sanity_check_data/bisde_empirical_experiments.json"
        ),
    }
    models_to_evaluate = [
        "FIM(Paper)",
        "FIM(Cont. with unary-binary)",
        "BISDE",
        "BISDE(ICML Submission)",
    ]
    systems_to_evaluate = ["oil", "wind"]

    split_into_num_paths = 10

    metrics_to_evaluate = [
        "mmd",
    ]

    metric_evaluations_to_load: list[Path] = [
        # Path("/home/seifner/repos/FIM/evaluations/synthetic_systems_metrics_tables/03092259_sparsegp_evaluation/metric_evaluations_jsons"),
    ]

    # tables config
    models_order = [
        "BISDE",
        "BISDE(ICML Submission)",
        "FIM(Paper)",
        "FIM(Cont. with unary-binary)",
    ]
    systems_order = systems_to_evaluate

    # --------------------------------------------------------------------------------------------------------------------------------- #

    # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
    evaluation_path = Path(project_path) / "evaluations"
    time: str = str(datetime.now().strftime("%m%d%H%M"))
    evaluation_dir: Path = evaluation_path / dataset_descr / (time + "_" + experiment_descr)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # prepare all evaluations to be done
    print("Preparing all evaluations.")
    all_evaluations: list[MetricEvaluation] = [
        MetricEvaluation(
            model_id=model,
            model_json=models_jsons[model],
            data_id=(system, split_into_num_paths),
            data_paths_json=data_paths_json,
            data_vector_fields_json=None,
            metric_id=metric,
            metric_value=None,  # input later
        )
        for model, system, metric in itertools.product(models_to_evaluate, systems_to_evaluate, metrics_to_evaluate)
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
        system, _ = eval.data_id
        file_name = f"mod_{eval.model_id.replace(' ', '_')}_met_{eval.metric_id}_sys_{system}.json"
        eval.to_json(json_save_dir / file_name)

    # compute metric for missing evaluations
    all_evaluations = loaded_evaluations
    mmd_observations_cache = {}  # K_xx for the observed data is the same for all models, so we cache it

    if len(to_evaluate) > 0:
        print("Loading observation and model data.")

        data_paths: dict = json.load(open(data_paths_json, "r"))
        models_data: dict = {model_name: json.load(open(model_json, "r")) for model_name, model_json in models_jsons.items()}

        print(f"Data paths keys: {[d['name'] for d in data_paths]}")
        print("Models keys: ")
        for model_name, model_systems in models_data.items():
            print(f"Model: {model_name}, Systems: {[system['name'] for system in model_systems]}")

        for eval in (pbar := tqdm(to_evaluate, leave=False)):
            system, split_into_num_paths = eval.data_id
            pbar.set_description(f"Processing metric={eval.metric_id}, model={eval.model_id}, {system=}.")

            start_time = datetime.now()

            if eval.metric_id == "mmd":
                mmd_comparison_paths: np.ndarray = get_observed_paths(
                    data_paths, system, split_into_num_paths
                )  # [split_into_num_paths, T, 1]
                model_paths: np.ndarray = get_model_paths(
                    models_data[eval.model_id], system, split_into_num_paths
                )  # [split_into_num_paths, T, 1]

                assert model_paths.shape == mmd_comparison_paths.shape

                if not np.isnan(model_paths).any():
                    eval.metric_value = compute_mmd(mmd_comparison_paths, model_paths, kernel_cache=mmd_observations_cache)

                else:
                    eval.metric_value = np.nan

            else:
                valid_metric_ids = ["mmd"]
                raise ValueError(f"Valid metrics are {str(valid_metric_ids)}, got {eval.metric_id} from requested evaluation {eval}.")

            # record computation time
            end_time = datetime.now()
            computation_time = (end_time - start_time).total_seconds()

            print(
                f"Last result: metric={eval.metric_id}, model={eval.model_id}, {system=}, value={eval.metric_value}, {computation_time=} seconds."
            )

            # save results as json
            file_name = f"mod_{eval.model_id.replace(' ', '_')}_met_{eval.metric_id}_sys_{system}.json"
            eval.to_json(json_save_dir / file_name)

            all_evaluations.append(eval)

    for metric in metrics_to_evaluate:
        negative_values = [None, "clip", "abs"] if metric == "mmd" else [None]

        for handle_negative_values in negative_values:
            df_mean = real_world_metric_table(all_evaluations, metric, models_order, systems_order, handle_negative_values)

            subdir_name = "tables_" + metric

            if handle_negative_values is not None:
                subdir_name = subdir_name + "_" + handle_negative_values

            metric_save_dir: Path = evaluation_dir / subdir_name
            metric_save_dir.mkdir(exist_ok=True, parents=True)

            save_table(df_mean, metric_save_dir, "mean")
