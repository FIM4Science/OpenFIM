import itertools
import json
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Optional

import numpy as np
import optree
import torch
from tqdm import tqdm

from fim import project_path
from fim.models.sde import FIMSDE
from fim.sampling.sde_path_samplers import fimsde_sample_paths_on_masked_grid
from fim.utils.sde.evaluation import (
    ModelEvaluation,
    ModelMap,
    NumpyEncoder,
    load_evaluations,
    model_map_from_dict,
    save_evaluations,
)


def get_real_world_data(
    all_data: list[dict],
    dataset: str,
    split: int,
    expected_num_total_splits: int,
    obs_times_key: str = "obs_times_separate",
    obs_values_key: str = "obs_values_separate",
    obs_mask_key: str = "obs_mask_separate",
) -> dict:
    """
     From a list of all real world data, extract dataset split.

    Args:
        all_data (list[dict]): List of splits of cross validation data, each of which is a dict.
        dataset (str): Name of dataset data to extract.
        split (int): The label of split to extract.
        expected_num_total_splits (int): For sanity that expected data fits loaded data.

    Return:
        data (dict): Keys: name, split, num_total_splits, delta_tau, transform, path_length_to_generate, obs_times, obs_values, locations, initial_states
    """
    data = [d for d in all_data if (d["name"] == dataset and d["split"] == split)]

    # should contain exactly one data for each dataset and split
    if len(data) == 1:
        data = data[0]

        assert data["num_total_splits"] == expected_num_total_splits, (
            f"Expected {expected_num_total_splits}, got {data['num_total_splits']}."
        )

        data = {k: np.array(v) if isinstance(v, list) else v for k, v in data.items()}

        # extract relevant keys from data
        return_data = {
            "name": data["name"],
            "split": data["split"],
            "num_total_splits": data["num_total_splits"],
            "delta_tau": data["delta_tau"],
            "transform": data["transform"],
            "path_length_to_generate": data.get("path_length_to_generate"),
            # FIM can be evaluated with (potentially) two paths, before and after the reference split
            "obs_times": data[obs_times_key],
            "obs_values": data[obs_values_key],
            "obs_mask": data.get(obs_mask_key),
            "locations": data.get("locations"),
            "initial_states": data.get("initial_states"),
        }

        return return_data

    elif len(data) == 0:
        raise ValueError(f"Could not find dataset {dataset} and split {split}.")

    else:
        raise ValueError(f"Found {len(data)} sets of data for dataset {dataset} and split {split}.")


def evaluate_model(
    model: FIMSDE,
    dataset: dict,
    device: Optional[str] = None,
):
    model.eval()

    # sample on device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    dataset = optree.tree_map(
        lambda x: torch.from_numpy(x).to(torch.float32).to(device) if isinstance(x, np.ndarray) else x, dataset, namespace="fimsde"
    )

    # extract initial states for sampling
    initial_states = dataset.get("initial_states")
    _, sample_paths_count, D = initial_states.shape

    # set up sampling grid based on delta_tau and path_length_to_generate from dataset
    path_length_to_generate = dataset["path_length_to_generate"]
    delta_tau = dataset["delta_tau"]
    grid = (torch.arange(path_length_to_generate) * delta_tau).view(1, 1, -1, 1)
    grid = torch.broadcast_to(grid, (1, sample_paths_count, path_length_to_generate, 1)).to(device)

    # sample paths
    sample_paths, sample_paths_grid = fimsde_sample_paths_on_masked_grid(
        model,
        dataset,
        grid=grid,
        mask=torch.ones_like(grid),
        initial_states=initial_states,
        solver_granularity=5,
    )

    print("Sample path shape: ", sample_paths.shape)

    # get vector fields at locations
    estimated_concepts = model(dataset, training=False, return_losses=False)

    # reduce outputs to dimensionality of original problem
    estimated_concepts.drift = estimated_concepts.drift[..., :D]
    estimated_concepts.diffusion = estimated_concepts.diffusion[..., :D]
    sample_paths = sample_paths[..., :D]

    # gather results
    results = {}
    results.update({"estimated_concepts": estimated_concepts})
    results.update(
        {
            "sample_paths": sample_paths,
            "sample_paths_grid": sample_paths_grid,
        }
    )

    results = optree.tree_map(lambda x: x.detach().to("cpu"), results, namespace="fimsde")

    return results


def run_evaluations(
    to_evaluate: list[ModelEvaluation],
    model_map: ModelMap,
    datasets: dict,
    device: Optional[str] = None,
) -> list[ModelEvaluation]:
    """
    Evaluate models on datasets from dataloaders.

    Args:
        evaluations (list[ModelEvaluation]): List of evaluations to run, specified by their model_id and dataloader_id
        model map: Returning required models
        datasets: Returning required dataset.

    Return:
        evaluations (list[ModelEvaluation]): Input evaluations.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    evaluations_with_results: list[ModelEvaluation] = []

    for evaluation in (pbar := tqdm(to_evaluate, total=len(to_evaluate), leave=False)):
        pbar.set_description(f"Model: {str(evaluation.model_id)}. Overall progress")

        model: FIMSDE = model_map[evaluation.model_id]().to(torch.float)
        dataset = datasets[evaluation.dataloader_id]

        evaluation.results = evaluate_model(model, dataset, device=device)
        evaluations_with_results.append(evaluation)

    return evaluations_with_results


def _check_finite(x) -> None:
    """
    Helper function to check finiteness of arrays in a nested structure
    """
    if isinstance(x, np.ndarray):
        assert np.isfinite(x).all().item()


def _pprint_dict_with_shapes(d: dict) -> None:
    pprint(optree.tree_map(lambda x: x.shape if isinstance(x, np.ndarray) or isinstance(x, torch.Tensor) else x, d))


if __name__ == "__main__":
    from model_dicts.models_trained_on_600k_deg_3_drift_deg_2_diffusion import (
        get_model_dicts_post_neurips_submission_checkpoint,
    )

    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    dataset_descr = "real_world_cross_validation_vf_and_paths_evaluation"

    # How to name experiments
    # experiment_descr = "fim_fixed_attn_fixed_softmax_05-06-2300"
    # experiment_descr = "fim_fixed_softmax_05-03-2033_epoch_138"
    experiment_descr = "fim_location_at_obs_no_finetuning"

    model_dicts, models_display_ids = get_model_dicts_post_neurips_submission_checkpoint()

    results_to_load: list[str] = []

    # systems in table of paper
    path_to_inference_data_json = Path(
        "/cephfs_projects/foundation_models/data/SDE/test/20250506_real_world_with_5_fold_cross_validation/cross_val_inference_paths.json"
    )

    datasets_to_load: list[str] = [
        "wind",
        "oil",
        "fb",
        "tsla",
    ]

    expected_num_total_splits = 5

    save_model_data = True
    # --------------------------------------------------------------------------------------------------------------------------------- #

    # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
    evaluation_path = Path(project_path) / "evaluations"
    time: str = str(datetime.now().strftime("%m%d%H%M"))
    evaluation_dir: Path = evaluation_path / dataset_descr / (time + "_" + experiment_descr)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    all_data = json.load(open(path_to_inference_data_json, "r"))

    datasets: dict = {
        (dataset, split): get_real_world_data(all_data, dataset, split, expected_num_total_splits)
        for dataset in datasets_to_load
        for split in range(expected_num_total_splits)
    }

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
        name, split = model_evaluation.dataloader_id
        name = name.replace("_", " ")

        dataset = datasets[model_evaluation.dataloader_id]

        model_outputs.append(
            {
                "name": name,
                "split": split,
                "num_total_splits": dataset["num_total_splits"],
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
