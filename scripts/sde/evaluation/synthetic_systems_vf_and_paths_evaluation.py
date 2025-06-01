import itertools
import json
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Optional

import numpy as np
import optree
import torch
from model_dicts.models_trained_on_600k_deg_3_drift_deg_2_diffusion import (
    get_model_dicts_neurips_submission_checkpoint,
)
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


def get_system_data(all_systems_data: list[dict], system: str, tau: float, noise: float) -> dict:
    """
     From a list of all systems data, extract data of system with tau inter-observation times and relative noise.

    Args:
        all_systems_data (list[dict]): List of system data, each of which is a dict.
        system (str): Name of system data to extract.
        tau (float): Inter-observation time of data to extract.
        noise (float): Relative additive noise added to observations of trajectories.

    Return:
        data_of_system (dict): Keys: name, tau, obs_times, obs_values, locations, initial_states
    """
    data_of_system = [d for d in all_systems_data if (d["name"] == system and d["tau"] == tau and d["noise"] == noise)]

    # should contain exactly one data for each system and tau
    if len(data_of_system) == 1:
        data_of_system = {k: np.array(v) if isinstance(v, list) else v for k, v in data_of_system[0].items()}

        # rename for easier inference for model
        if "observations" in data_of_system:
            data_of_system["obs_values"] = data_of_system.pop("observations")

        # add obs times based_on_tau
        B, M, T, _ = data_of_system["obs_values"].shape
        data_of_system["obs_times"] = tau * np.ones((B, M, 1, 1)) * np.arange(T).reshape(1, 1, T, 1)

        return data_of_system

    elif len(data_of_system) == 0:
        raise ValueError(f"Could not find data of system {system} and tau {tau} and noise perc {noise}.")

    else:
        raise ValueError(f"Found {len(data_of_system)} sets of data for system {system} and tau {tau}.")


def evaluate_model(
    model: FIMSDE,
    dataset: dict,
    sample_paths: Optional[bool],
    sample_paths_count: int,
    dt: float,
    sample_path_steps: int,
    device: Optional[str] = None,
):
    model.eval()

    results = {}

    # sample on device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    dataset = optree.tree_map(
        lambda x: torch.from_numpy(x).to(torch.float32).to(device) if isinstance(x, np.ndarray) else x, dataset, namespace="fimsde"
    )

    initial_states = dataset.get("initial_states")
    D = dataset["initial_states"].shape[-1]

    grid = (torch.arange(sample_path_steps) * dt).view(1, 1, -1, 1)
    grid = torch.broadcast_to(grid, (initial_states.shape[0], sample_paths_count, sample_path_steps, 1)).to(device)

    if sample_paths is True:
        sample_paths, sample_paths_grid = fimsde_sample_paths_on_masked_grid(
            model,
            dataset,
            grid=grid,
            mask=torch.ones_like(grid),
            initial_states=initial_states,
            solver_granularity=1,
        )

        results.update(
            {
                "sample_paths": sample_paths,
                "sample_paths_grid": sample_paths_grid,
            }
        )

        print("Sample path shape: ", sample_paths.shape)

    # get vector fields at locations
    estimated_concepts = model(dataset, training=False, return_losses=False)
    results.update({"estimated_concepts": estimated_concepts})

    # reduce outputs to dimensionality of original problem
    estimated_concepts.drift = estimated_concepts.drift[..., :D]
    estimated_concepts.diffusion = estimated_concepts.diffusion[..., :D]
    if "sample_paths" in results.keys():
        results["sample_paths"] = results["sample_paths"][..., :D]

    results = optree.tree_map(lambda x: x.detach().to("cpu"), results, namespace="fimsde")

    return results


def run_evaluations(
    to_evaluate: list[ModelEvaluation],
    model_map: ModelMap,
    dataloaders: dict,
    sample_paths: bool,
    sample_paths_count: int,
    dt: float,
    sample_path_steps: int,
    device: Optional[str] = None,
) -> list[ModelEvaluation]:
    """
    Evaluate models on datasets from dataloaders.

    Args:
        evaluations (list[ModelEvaluation]): List of evaluations to run, specified by their model_id and dataloader_id
        model map: Returning required models
        dataloaders: Returning required dataloaders.
        sample_paths (bool): If True, sample a path for each model.

    Return:
        evaluations (list[ModelEvaluation]): Input evaluations.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    evaluations_with_results: list[ModelEvaluation] = []

    for evaluation in (pbar := tqdm(to_evaluate, total=len(to_evaluate), leave=False)):
        pbar.set_description(f"Model: {str(evaluation.model_id)}. Overall progress")

        model: FIMSDE = model_map[evaluation.model_id]().to(torch.float)
        dataset = dataloaders[evaluation.dataloader_id]

        evaluation.results = evaluate_model(model, dataset, sample_paths, sample_paths_count, dt, sample_path_steps, device=device)
        evaluations_with_results.append(evaluation)

    return evaluations_with_results


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    dataset_descr = "synthetic_systems_vf_and_paths_evaluation"

    # How to name experiments
    # experiment_descr = "fim_fixed_linear_attn_fixed_softmax_delta_tau_05-06-2300"
    # experiment_descr = "fim_fixed_linear_attn_fixed_softmax_delta_tau_05-06-2300"
    experiment_descr = "fim_fixed_softmax_dim_05-03-2033_epoch_139"

    model_dicts, models_display_ids = get_model_dicts_neurips_submission_checkpoint()

    results_to_load: list[str] = []

    # systems in table of paper
    path_to_inference_data_json = Path(
        "/cephfs_projects/foundation_models/data/SDE/external_evaluations_and_data/20250325_synthetic_systems_5000_points_with_additive_noise/data/systems_observations_and_locations.json"
    )

    systems_to_load: list[str] = [
        "Damped Linear",
        "Damped Cubic",
        "Duffing",
        "Glycosis",
        "Hopf",
        "Double Well",
        "Wang",
        "Syn Drift",
    ]

    taus = [0.002, 0.01, 0.02, 0.2]
    noises = [0.0, 0.05, 0.1]
    sample_paths = True
    sample_paths_count = 100
    dt = 0.002
    sample_path_steps = 500

    save_model_data = True
    # --------------------------------------------------------------------------------------------------------------------------------- #

    # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
    evaluation_path = Path(project_path) / "evaluations"
    time: str = str(datetime.now().strftime("%m%d%H%M"))
    evaluation_dir: Path = evaluation_path / dataset_descr / (time + "_" + experiment_descr)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    data = json.load(open(path_to_inference_data_json, "r"))

    datasets: dict = {
        (system, tau, noise): get_system_data(data, system, tau, noise) for system in systems_to_load for tau in taus for noise in noises
    }

    # Setup inits for models and dataloaders
    model_map = model_map_from_dict(model_dicts)

    # Load previous evaluations that don't need to be evaluated anymore
    loaded_evaluations: list[ModelEvaluation] = load_evaluations(results_to_load)

    ######################################################################
    for eval in loaded_evaluations:  # backward compatability for evaluations without noise
        system, stride, mmd_max_num_paths = eval.dataloader_id
        eval.dataloader_id = (system.replace("_", " "), stride * 0.002, 0.0)
    ######################################################################

    # Evaluate all models on all datasets
    all_evaluations: list[ModelEvaluation] = [
        ModelEvaluation(model_id, dataloader_id) for model_id, dataloader_id in itertools.product(model_dicts.keys(), datasets.keys())
    ]
    to_evaluate: list[ModelEvaluation] = [evaluation for evaluation in all_evaluations if evaluation not in loaded_evaluations]

    # Create, run and save evaluations
    evaluated: list[ModelEvaluation] = run_evaluations(
        to_evaluate, model_map, datasets, sample_paths, sample_paths_count, dt, sample_path_steps
    )

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
        name, tau, noise = model_evaluation.dataloader_id
        name = name.replace("_", " ")

        model_outputs.append(
            {
                "name": name,
                "tau": tau,
                "noise": noise,
                "synthetic_paths": model_evaluation.results.get("sample_paths"),
                "drift_at_locations": model_evaluation.results["estimated_concepts"].drift,
                "diffusion_at_locations": model_evaluation.results["estimated_concepts"].diffusion,
            }
        )

    print("Model outputs to save")
    pprint(optree.tree_map(lambda x: x.shape if isinstance(x, torch.Tensor) else x, model_outputs))

    # save ground-truth drift and diffusion values
    if save_model_data is True:
        assert len(model_dicts) == 1, "Only works for a single model right now"

        def _check_finite(x):
            if isinstance(x, torch.Tensor):
                assert torch.torch.isfinite(x).all().item()

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

    # # Figures with subplot grid containing results from multiple equations per dataset
    # if sample_paths is True:
    #     for model_evaluation in (pbar := tqdm(all_evaluations, total=len(all_evaluations), leave=False)):
    #         pbar.set_description(
    #             f"Saving figure grids for model {model_evaluation.model_id} and dataloader {model_evaluation.dataloader_id}."
    #         )
    #
    #         dataset: dict = datasets[model_evaluation.dataloader_id]
    #         dim = dataset["initial_states"].shape[-1]
    #
    #         if dim == 1:
    #             grid_plot_func = plot_1D_synthetic_data_figure_grid
    #         if dim == 2:
    #             grid_plot_func = plot_2D_synthetic_data_figure_grid
    #
    #         fig = grid_plot_func(
    #             dataset["locations"],
    #             dataset["drift_at_locations"],
    #             dataset["diffusion_at_locations"],
    #             model_evaluation.results["estimated_concepts"].drift,
    #             model_evaluation.results["estimated_concepts"].diffusion,
    #             dataset["obs_times"],
    #             dataset["obs_values"],
    #             dataset["obs_mask"].bool(),
    #             model_evaluation.results.get("sample_paths_grid"),
    #             model_evaluation.results.get("sample_paths"),
    #         )
    #
    #         # save
    #         save_dir: Path = (
    #             evaluation_dir
    #             / "figure_grid"
    #             / model_evaluation.dataloader_id[0]
    #             / ("length_" + str(model_evaluation.dataloader_id[2]))
    #             / ("stride_" + str(model_evaluation.dataloader_id[1]))
    #         )
    #         save_dir.mkdir(parents=True, exist_ok=True)
    #         file_name = (
    #             f"data_{model_evaluation.dataloader_id[0]}_stride_{model_evaluation.dataloader_id[1]}_model_{model_evaluation.model_id}"
    #         )
    #         save_fig(fig, save_dir, file_name)
    #
    #         plt.close(fig)
    #
    #     pbar.close()
