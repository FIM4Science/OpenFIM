import itertools
import json
from copy import copy
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Optional

import matplotlib.pyplot as plt
import optree
import torch
from model_dicts.models_trained_on_600k_deg_3_drift_deg_2_diffusion import get_model_dicts_600k_post_submission_models
from tqdm import tqdm

from fim import project_path
from fim.data.utils import load_h5
from fim.models.blocks import AModel
from fim.pipelines.sde_sampling_from_model import fimsde_sample_paths_on_masked_grid
from fim.utils.evaluation_sde import (
    ModelEvaluation,
    ModelMap,
    NumpyEncoder,
    load_evaluations,
    model_map_from_dict,
    save_evaluations,
    save_fig,
)


def evaluate_model(model: AModel, dataset: dict, samples_per_initial_state: int, process_log: bool, device: Optional[str] = None):
    model.eval()

    results = {}

    if process_log is True:
        dataset["obs_values"] = torch.log(dataset["obs_values"])

    # evaluate vector field on some location grid
    # evaluate on cpu, as it sometimes runs out of memory due to large time series
    if "locations" in dataset.keys():
        if process_log is True:
            dataset["locations"] = torch.log(dataset["locations"])
        estimated_concepts = model(dataset, training=False, return_losses=False)
        estimated_concepts = optree.tree_map(lambda x: x.to("cpu"), estimated_concepts, namespace="fimsde")
        estimated_concepts = optree.tree_map(lambda x: x[..., : dataset["obs_values"].shape[-1]], estimated_concepts, namespace="fimsde")

        if process_log is True:
            locations = copy(estimated_concepts.locations)
            drift = copy(estimated_concepts.drift)
            diffusion = copy(estimated_concepts.diffusion)

            exp_x = torch.exp(locations)

            estimated_concepts.locations = exp_x
            estimated_concepts.drift = exp_x * (drift + 1 / 2 * (diffusion**2))
            estimated_concepts.diffusion = exp_x * diffusion

        results.update({"estimated_concepts": estimated_concepts})

    # sample on device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    dataset = optree.tree_map(lambda x: x.to(device), dataset, namespace="fimsde")

    # sample paths on whole observed time interval
    if "initial_states" in dataset.keys():
        initial_states = dataset["initial_states"]
    else:
        initial_states = dataset["obs_values"][:, :, 0]  # [1, 1, 1]

    ksig_sampling_grid = dataset["ksig_sampling_grid"]

    # repat initial condition to sample more paths
    ksig_sampling_grid = torch.repeat_interleave(ksig_sampling_grid, repeats=samples_per_initial_state, dim=1)

    initial_states = torch.repeat_interleave(initial_states, repeats=samples_per_initial_state, dim=1)

    sample_paths, sample_paths_grid = fimsde_sample_paths_on_masked_grid(
        model,
        dataset,
        ksig_sampling_grid,
        mask=torch.ones_like(ksig_sampling_grid),
        initial_states=initial_states,
        solver_granularity=5,
    )

    results.update(
        {
            "sample_paths": sample_paths,
            "sample_paths_grid": sample_paths_grid,
        }
    )

    if process_log is True:
        results["sample_paths"] = torch.exp(results["sample_paths"])

    results = optree.tree_map(lambda x: x.detach().to("cpu"), results, namespace="fimsde")

    return results


def run_wang_evaluations(
    to_evaluate: list[ModelEvaluation],
    model_map: ModelMap,
    datasets: dict,
    samples_per_initial_state: int,
    device: Optional[str] = None,
) -> list[ModelEvaluation]:
    """
    Evaluate model on real world data from BISDE (Wang 2022).

    Args:
        evaluations (list[ModelEvaluation]): List of evaluations to run, specified by their model_id and dataloader_id
        model map: Returning required models
        datasets:

    Return:
        evaluations (list[ModelEvaluation]): Input evaluations.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    evaluations_with_results: list[ModelEvaluation] = []

    for evaluation in (pbar := tqdm(to_evaluate, total=len(to_evaluate), leave=False)):
        pbar.set_description(f"Model: {str(evaluation.model_id)}. Overall progress")

        model: AModel = model_map[evaluation.model_id]().to(torch.float)
        dataset: dict = datasets[evaluation.dataloader_id]

        evaluation.results = evaluate_model(
            model, copy(dataset), samples_per_initial_state, process_log=evaluation.dataloader_id in ["fb", "tsla"], device=device
        )
        evaluations_with_results.append(evaluation)

    return evaluations_with_results


def get_wang_real_world_data(data_dir: Path, name: str, split_ksig_length: int, load_fluctuations: Optional[bool] = False) -> dict:
    """
    Return data for real world data from BISDE (Wang 2022).

    Args:
        data_dir (Path): Absolute path to dir containing h5 files

    Returns:
        data (dict[str, Tensor]): Keys: obs_times, obs_values, locations
    """
    obs_times = load_h5(data_dir / "obs_times.h5") if load_fluctuations is False else load_h5(data_dir / "obs_times_fluctuations.h5")
    obs_values = load_h5(data_dir / "obs_values.h5") if load_fluctuations is False else load_h5(data_dir / "obs_values_fluctuations.h5")
    locations = torch.linspace(start=torch.amin(obs_values), end=torch.amax(obs_values), steps=1024).view(1, -1, 1)  # for now

    tau = obs_times[:, :, 1, :] - obs_times[:, :, 0, :]
    tau = tau.squeeze().item()

    inference_data_to_share = {
        "name": name,
        "tau": tau,
        "observations": obs_values[:, :, :-split_ksig_length, :],  # [1,1,T,1],
        "locations": locations,  # [num_locations,1]
        "initial_states": obs_values[:, :, -split_ksig_length, :],  # [1, 1, 1]
    }

    ksig_data_to_share = {
        "name": name,
        "real_paths": obs_values[:, :, -split_ksig_length:, :],  # [1, 1, T_ksig, 1],
    }

    our_model_input = {  # same as inference_data_to_share, with other keys
        "obs_times": obs_times[:, :, :-split_ksig_length, :],
        "obs_values": inference_data_to_share["observations"],
        "locations": locations,
        "initial_states": inference_data_to_share["initial_states"],
        "ksig_sampling_grid": torch.arange(split_ksig_length).reshape(1, 1, -1, 1) * tau,
    }

    return inference_data_to_share, ksig_data_to_share, our_model_input


def get_wang_results(bisde_results_dir: Path) -> tuple[dict]:
    """
    Return results for real world data from BISDE (Wang 2022) model.

    Args:
        bisde_results_dir (Path): Absolute path to dir containing h5 files.

    Returns:
        data (dict[str, Tensor]): locations, drift, diffusion, sample_paths_grid, sample_paths
    """
    bisde_results = {
        "locations": load_h5(bisde_results_dir / "locations.h5"),
        "drift": load_h5(bisde_results_dir / "drift_at_locations.h5"),
        "diffusion": load_h5(bisde_results_dir / "diffusion_at_locations.h5"),
        "sample_paths_grid": load_h5(bisde_results_dir / "obs_times.h5"),
        "sample_paths": load_h5(bisde_results_dir / "obs_values.h5"),
    }

    return bisde_results


def get_wang_sample_paths_figure(
    obs_times,
    obs_values,
    model_sample_paths_times,
    model_sample_paths_values,
    bisde_sample_paths_times,
    bisde_sample_paths_values,
    title: str = None,
    figsize=(5, 5),
    obs_color="black",
    model_color="#0072B2",
    bisde_color="#CC79A7",
    alpha=0.75,
    linewidth=0.5,
):
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    ax.plot(obs_times.squeeze(), obs_values.squeeze(), label="Observations", color=obs_color, linewidth=linewidth)
    ax.plot(
        model_sample_paths_times.squeeze(),
        model_sample_paths_values.squeeze(),
        label="Our Model",
        color=model_color,
        alpha=alpha,
        linewidth=linewidth,
    )
    ax.plot(
        bisde_sample_paths_times.squeeze(),
        bisde_sample_paths_values.squeeze(),
        label="BISDE Eq. from Paper",
        color=bisde_color,
        alpha=alpha,
        linewidth=linewidth,
    )

    ax.legend()
    ax.set_title(title)

    return fig


def figure_sample_paths(model_evaluation: ModelEvaluation, datasets: dict, bisde_results: dict, evaluation_dir: Path):
    data_id = model_evaluation.dataloader_id

    data: dict = datasets[data_id]
    bisde_results: dict = bisde_results[data_id]
    model_results = model_evaluation.results

    fig = get_wang_sample_paths_figure(
        data["obs_times"],
        data["obs_values"],
        model_results["sample_paths_grid"],
        model_results["sample_paths"],
        bisde_results["sample_paths_grid"],
        bisde_results["sample_paths"],
        title=data_id,
    )

    # save
    save_dir: Path = evaluation_dir / "figure_sample_paths" / model_evaluation.dataloader_id / model_evaluation.model_id
    save_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"model_{model_evaluation.model_id}"
    save_fig(fig, save_dir, file_name)

    plt.close(fig)


def get_wang_vector_fields_figure(
    model_locations,
    model_drift,
    model_diffusion,
    bisde_locations,
    bisde_drift,
    bisde_diffusion,
    title=None,
    figsize=(5, 5),
    model_color="#0072B2",
    bisde_color="#CC79A7",
):
    fig, ax = plt.subplots(1, 2, figsize=figsize, dpi=300)

    ax[0].set_title("Drift")
    ax[1].set_title("Diffusion")

    ax[0].plot(model_locations.squeeze(), model_drift.squeeze(), label="Our Model", color=model_color)
    ax[0].plot(bisde_locations.squeeze(), bisde_drift.squeeze(), label="BISDE", color=bisde_color)
    ax[1].plot(model_locations.squeeze(), model_diffusion.squeeze(), label="Our Model", color=model_color)
    ax[1].plot(bisde_locations.squeeze(), bisde_diffusion.squeeze(), label="BISDE", color=bisde_color)

    ax[0].legend()

    fig.suptitle(title)

    return fig


def figure_vector_fields(model_evaluation: ModelEvaluation, bisde_results: dict, evaluation_dir: Path):
    data_id = model_evaluation.dataloader_id

    bisde_results: dict = bisde_results[data_id]
    model_results = model_evaluation.results

    fig = get_wang_vector_fields_figure(
        model_results["estimated_concepts"].locations.detach(),
        model_results["estimated_concepts"].drift.detach(),
        model_results["estimated_concepts"].diffusion.detach(),
        bisde_results["locations"],
        bisde_results["drift"],
        bisde_results["diffusion"],
        title=data_id,
    )

    # save
    save_dir: Path = evaluation_dir / "figure_vector_fields" / model_evaluation.dataloader_id / model_evaluation.model_id
    save_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"model_{model_evaluation.model_id}"
    save_fig(fig, save_dir, file_name)

    plt.close(fig)


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    dataset_descr = "oil_wind_tsla_fb_model_evaluation"

    # How to name experiments
    experiment_descr = "model_trained_on_delta_tau_1e-1_to_1e-03-23-1747"

    model_dicts, models_display_ids = get_model_dicts_600k_post_submission_models()

    results_to_load: list[str] = [
        "/home/seifner/repos/FIM/evaluations/oil_wind_tsla_fb_model_evaluation/03102255_model_cont_train_on_unary_binary_trees_with_polynomials_mixed_in/model_evaluations",
    ]

    base_data_dir = Path("/home/seifner/repos/FIM/data/processed/test/20250125_preprocessed_wang_real_world")
    data_fb_subdir = "fb_stock/full_traj_single_path"
    data_tsla_subdir = "tsla_stock/full_traj_single_path"
    data_wind_subdir = "wind/full_traj_single_path"
    data_oil_subdir = "oil/full_traj_single_path"

    bisde_results_dir = Path("/home/seifner/repos/FIM/data/processed/test/20250126_wang_estimated_equations")
    bisde_fb_subdir = "bisde_est_facebook"
    bisde_tsla_subdir = "bisde_est_tesla"
    bisde_wind_subdir = "bisde_est_wind"
    bisde_oil_subdir = "bisde_est_oil"

    # samples_per_initial_state = 100
    samples_per_initial_state = 1
    # --------------------------------------------------------------------------------------------------------------------------------- #

    # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
    evaluation_path = Path(project_path) / "evaluations"
    time: str = str(datetime.now().strftime("%m%d%H%M"))
    evaluation_dir: Path = evaluation_path / dataset_descr / (time + "_" + experiment_descr)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # Get datasets
    datasets = {
        "fb": get_wang_real_world_data(base_data_dir / data_fb_subdir, "fb", 1, load_fluctuations=False),
        "tsla": get_wang_real_world_data(base_data_dir / data_tsla_subdir, "tsla", 1, load_fluctuations=False),
        "wind": get_wang_real_world_data(base_data_dir / data_wind_subdir, "wind", 5000, load_fluctuations=True),
        "oil": get_wang_real_world_data(base_data_dir / data_oil_subdir, "oil", 2000, load_fluctuations=True),
    }
    bisde_results = {
        "fb": get_wang_results(bisde_results_dir / bisde_fb_subdir),
        "tsla": get_wang_results(bisde_results_dir / bisde_tsla_subdir),
        "wind": get_wang_results(bisde_results_dir / bisde_wind_subdir),
        "oil": get_wang_results(bisde_results_dir / bisde_oil_subdir),
    }

    # split inference_data_to_share and ksig_data_to_share from data
    inference_data_to_share = {name: dataset[0] for name, dataset in datasets.items()}
    ksig_data_to_share = {name: dataset[1] for name, dataset in datasets.items()}
    datasets = {name: dataset[2] for name, dataset in datasets.items()}

    # Get model_map to load models when they are needed
    model_map = model_map_from_dict(model_dicts)

    # Load previous evaluations that don't need to be evaluated anymore
    loaded_evaluations: list[ModelEvaluation] = load_evaluations(results_to_load)

    # Evaluate all models on all datasets
    all_evaluations: list[ModelEvaluation] = [
        ModelEvaluation(model_id, dataset_id) for model_id, dataset_id in itertools.product(model_dicts.keys(), datasets.keys())
    ]
    to_evaluate: list[ModelEvaluation] = [evaluation for evaluation in all_evaluations if evaluation not in loaded_evaluations]

    # Create, run and save EvaluationConfig
    evaluated: list[ModelEvaluation] = run_wang_evaluations(to_evaluate, model_map, datasets, samples_per_initial_state)
    all_evaluations: list[ModelEvaluation] = loaded_evaluations + evaluated
    save_evaluations(all_evaluations, evaluation_dir / "model_evaluations")

    # save data
    inference_data = []
    for name, data in inference_data_to_share.items():
        data.update({"name": name})
        inference_data.append(data)

    print("Inference datasets to save")
    pprint(optree.tree_map(lambda x: x.shape if isinstance(x, torch.Tensor) else x, inference_data))

    ksig_data = []
    for name, data in ksig_data_to_share.items():
        data.update({"name": name})
        ksig_data.append(data)

    print("Ksig reference to save")
    pprint(optree.tree_map(lambda x: x.shape if isinstance(x, torch.Tensor) else x, ksig_data))

    model_outputs = []
    for model_evaluation in all_evaluations:
        name = model_evaluation.dataloader_id
        model_outputs.append(
            {
                "name": name,
                "synthetic_paths": model_evaluation.results["sample_paths"],
                "drift_at_locations": model_evaluation.results["estimated_concepts"].drift,
                "diffusion_at_locations": model_evaluation.results["estimated_concepts"].diffusion,
            }
        )

    print("Model outputs to save")
    pprint(optree.tree_map(lambda x: x.shape if isinstance(x, torch.Tensor) else x, model_outputs))

    # Figure
    for model_evaluation in (pbar := tqdm(all_evaluations, total=len(all_evaluations), leave=False)):
        pbar.set_description(f"Sample paths: {model_evaluation.model_id}.")
        figure_sample_paths(model_evaluation, datasets, bisde_results, evaluation_dir)
        figure_vector_fields(model_evaluation, bisde_results, evaluation_dir)

    pbar.close()

    # Save data
    assert len(model_dicts) == 1, "Only works for a single model right now"

    def _check_finite(x):
        if isinstance(x, torch.Tensor):
            assert torch.torch.isfinite(x).all().item()

    # optree.tree_map(_check_finite, (model_outputs, ksig_data, inference_data), namespace="fimsde")
    print("Check finiteness: ")
    pprint(
        optree.tree_map(
            lambda x: torch.torch.isfinite(x).all().item() if isinstance(x, torch.Tensor) else x,
            (model_outputs, ksig_data, inference_data),
            namespace="fimsde",
        )
    )

    model_outputs, ksig_data, inference_data = optree.tree_map(
        lambda x: x.detach().to("cpu").numpy() if isinstance(x, torch.Tensor) else x,
        (model_outputs, ksig_data, inference_data),
    )

    for data, filename in zip(
        [model_outputs, ksig_data, inference_data],
        ["model_paths.json", "ksig_reference_paths.json", "data_for_inference.json"],
    ):
        # Convert to JSON
        json_data = json.dumps(data, cls=NumpyEncoder)

        # Write JSON data to a file
        json_dir = evaluation_dir / "data_jsons"
        json_dir.mkdir(exist_ok=True, parents=True)

        file: Path = json_dir / filename
        with open(file, "w") as file:
            file.write(json_data)
