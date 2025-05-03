import itertools
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import optree
import torch
from model_dicts.models_trained_on_600k_deg_3_drift_deg_2_diffusion import (
    get_model_dicts_600k_fixed_linear_attn,
)
from real_world_cross_validation_vf_and_paths_evaluation import _check_finite, _pprint_dict_with_shapes, run_evaluations

from fim import project_path
from fim.utils.evaluation_sde import (
    ModelEvaluation,
    NumpyEncoder,
    load_evaluations,
    model_map_from_dict,
    save_evaluations,
)


def get_real_world_data(all_data: list[dict], dataset: str) -> dict:
    """
     From a list of all real world data, extract data of particular dataset

    Args:
        all_data (list[dict]): List of splits of all complete trajectories, each of which is a dict.
        dataset (str): Name of dataset data to extract.

    Return:
        data (dict): Keys: name, delta_tau, transform, obs_times, obs_values, locations, initial_state, path_length_to_generate
    """
    data = [d for d in all_data if d["name"] == dataset]

    # should contain exactly one dataset
    if len(data) == 1:
        data = data[0]
        data = {k: np.array(v) if isinstance(v, list) else v for k, v in data.items()}

        # loaded data contains expected keys
        return data

    elif len(data) == 0:
        raise ValueError(f"Could not find dataset {dataset}.")

    else:
        raise ValueError(f"Found {len(data)} sets of data for dataset.")


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    dataset_descr = "real_world_complete_trajectory_vf_and_paths_evaluation"

    # How to name experiments
    experiment_descr = "develop"

    model_dicts, models_display_ids = get_model_dicts_600k_fixed_linear_attn()

    results_to_load: list[str] = [
        # "/home/seifner/repos/FIM/saved_evaluations/20250203_icml_submission_evaluations/synthetic_equations_stride_1_5_10_for_table/model_evaluations/20M_params_trained_even_longer"
    ]

    # systems in table of paper
    path_to_complete_trajectory_data_json = Path(
        "/home/seifner/repos/FIM/data/processed/test/20250506_real_world_with_5_fold_cross_validation/complete_paths.json"
    )

    datasets_to_load: list[str] = [
        "wind",
        "oil",
        "fb",
        "tsla",
    ]

    save_model_data = True
    # --------------------------------------------------------------------------------------------------------------------------------- #

    # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
    evaluation_path = Path(project_path) / "evaluations"
    time: str = str(datetime.now().strftime("%m%d%H%M"))
    evaluation_dir: Path = evaluation_path / dataset_descr / (time + "_" + experiment_descr)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    all_data = json.load(open(path_to_complete_trajectory_data_json, "r"))

    datasets: dict = {dataset: get_real_world_data(all_data, dataset) for dataset in datasets_to_load}

    # Setup inits for models and dataloaders
    model_map = model_map_from_dict(model_dicts)

    # Load previous evaluations that don't need to be evaluated anymore
    loaded_evaluations: list[ModelEvaluation] = load_evaluations(results_to_load)

    # Evaluate all models on all datasets
    all_evaluations: list[ModelEvaluation] = [
        ModelEvaluation(model_id, dataset_id) for model_id, dataset_id in itertools.product(model_dicts.keys(), datasets.keys())
    ]
    to_evaluate: list[ModelEvaluation] = [evaluation for evaluation in all_evaluations if evaluation not in loaded_evaluations]

    # Create, run and save evaluations
    evaluated: list[ModelEvaluation] = run_evaluations(to_evaluate, model_map, datasets)

    # remove loaded evaluations not needed
    loaded_evaluations = [
        eval for eval in loaded_evaluations if (eval.model_id, eval.dataloader_id) in itertools.product(model_dicts.keys(), datasets.keys())
    ]

    # Add loaded evaluations
    all_evaluations: list[ModelEvaluation] = loaded_evaluations + evaluated
    save_evaluations(all_evaluations, evaluation_dir / "model_evaluations")

    # save sampled paths and vector field inferences
    model_outputs = []
    for model_evaluation in all_evaluations:
        name = model_evaluation.dataloader_id
        name = name.replace("_", " ")

        dataset = datasets[model_evaluation.dataloader_id]

        model_outputs.append(
            {
                "name": name,
                "delta_tau": dataset["delta_tau"],
                "transform": dataset["transform"],
                "synthetic_paths": model_evaluation.results["sample_paths"],
                "locations": dataset["locations"],
                "drift_at_locations": model_evaluation.results["estimated_concepts"].drift,
                "diffusion_at_locations": model_evaluation.results["estimated_concepts"].diffusion,
            }
        )

    print("Model outputs to save:")
    _pprint_dict_with_shapes(model_outputs)

    # save drift and diffusion values
    if save_model_data is True:
        assert len(model_dicts) == 1, "Only works for a single model right now"

        _check_finite(model_outputs)
        model_outputs = optree.tree_map(lambda x: x.detach().to("cpu").numpy() if isinstance(x, torch.Tensor) else x, model_outputs)

        # Convert to JSON
        json_data = json.dumps(model_outputs, cls=NumpyEncoder)

        # Write JSON data to a file
        json_dir = evaluation_dir
        json_dir.mkdir(exist_ok=True, parents=True)

        file: Path = json_dir / "model_paths.json"
        with open(file, "w") as file:
            file.write(json_data)
