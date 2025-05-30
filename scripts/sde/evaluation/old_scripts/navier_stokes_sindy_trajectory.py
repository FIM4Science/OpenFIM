from copy import copy
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import optree
import torch
from model_dicts.models_trained_on_600k_deg_3_drift_deg_2_diffusion import get_model_dicts_600k_deg_3_drift_deg_2_diff
from tqdm import tqdm

from fim import project_path
from fim.data.utils import load_h5
from fim.models.blocks import AModel
from fim.pipelines.sde_sampling_from_model import fimsde_sample_paths_on_masked_grid
from fim.utils.evaluation_sde import (
    ModelEvaluation,
    ModelMap,
    load_evaluations,
    model_map_from_dict,
    save_evaluations,
    save_fig,
)


def evaluate_model(model: AModel, dataset: dict, device: Optional[str] = None):
    model.eval()

    results = {}

    # sample on device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    dataset = optree.tree_map(lambda x: x.to(device), dataset, namespace="fimsde")

    # sample paths on whole observed time interval
    sample_paths, sample_paths_grid = fimsde_sample_paths_on_masked_grid(
        model,
        dataset,
        grid=dataset["obs_times"],
        mask=dataset["obs_mask"],
        initial_states=dataset["obs_values"][..., 0, :],
        solver_granularity=5,
    )

    results.update(
        {
            "sample_paths": sample_paths,
            "sample_paths_grid": sample_paths_grid,
        }
    )

    # only first trajectory as input
    sample_paths, sample_paths_grid = fimsde_sample_paths_on_masked_grid(
        model,
        optree.tree_map(lambda x: x[:, 0][:, None], dataset),
        grid=dataset["obs_times"],
        mask=dataset["obs_mask"],
        initial_states=dataset["obs_values"][..., 0, :],
        solver_granularity=5,
    )
    results.update(
        {
            "sample_paths_from_first_traj": sample_paths,
            "sample_paths_grid_from_first_traj": sample_paths_grid,
        }
    )

    results = optree.tree_map(lambda x: x.detach().to("cpu"), results, namespace="fimsde")

    return results


def run_navier_stokes_evaluations(
    to_evaluate: list[ModelEvaluation],
    model_map: ModelMap,
    dataset: dict,
    device: Optional[str] = None,
) -> list[ModelEvaluation]:
    """
    Evaluate model on navier stokes trajectory from SINDy.

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

        evaluation.results = evaluate_model(model, copy(dataset), device=device)
        evaluations_with_results.append(evaluation)

    return evaluations_with_results


def get_dataset(data_dir: Path) -> dict:
    """
    Return preprocessed data of navier stokes trajectory from SINDy.

    Args:
        data_dir (Path): Absolute path to dir containing h5 files

    Returns:
        data (dict[str, Tensor]): Keys: obs_times, obs_values, obs_mask
    """
    data = {
        "obs_times": load_h5(data_dir / "obs_times.h5"),
        "obs_values": load_h5(data_dir / "obs_values.h5"),
        "obs_mask": load_h5(data_dir / "obs_mask.h5").bool(),
    }

    return data


def get_3D_figure_grid(
    obs_values,  # [1, 2, 5000, 3]
    obs_mask,  # [1, 2, 5000, 1]
    model_sample_paths,  # [1, 2, 5000, 3]
    model_sample_paths_first_traj,  # [1, 1, 5000, 3]
    figsize=(7, 7),
    obs_color="black",
    model_color="#0072B2",
    linewidth=1,
):
    fig, axs = plt.subplots(2, 2, figsize=figsize, dpi=300, subplot_kw={"projection": "3d"})

    # run 1
    obs_values_run_1 = obs_values[0, 0][obs_mask[0, 0].squeeze()]
    model_sample_paths_run_1 = model_sample_paths[0, 0][obs_mask[0, 0].squeeze()]
    axs[0, 0].plot(
        obs_values_run_1[:, 0], obs_values_run_1[:, 1], obs_values_run_1[:, 2], linewidth=linewidth, color=obs_color, label="Observations"
    )
    axs[0, 1].plot(obs_values_run_1[:, 0], obs_values_run_1[:, 1], obs_values_run_1[:, 2], linewidth=linewidth, color=obs_color, alpha=0.5)
    axs[0, 1].plot(
        model_sample_paths_run_1[:, 0],
        model_sample_paths_run_1[:, 1],
        model_sample_paths_run_1[:, 2],
        linewidth=linewidth,
        color=model_color,
        label="Our Model both traj",
    )
    axs[0, 1].plot(
        model_sample_paths_first_traj[0, 0, :, 0],
        model_sample_paths_first_traj[0, 0, :, 1],
        model_sample_paths_first_traj[0, 0, :, 2],
        linewidth=linewidth,
        color="#CC79A7",
        label="Our Model first traj",
    )
    axs[0, 0].scatter(obs_values_run_1[0, 0], obs_values_run_1[0, 1], obs_values_run_1[0, 2], color="red")
    axs[0, 1].scatter(obs_values_run_1[0, 0], obs_values_run_1[0, 1], obs_values_run_1[0, 2], color="red")

    # run 2
    obs_values_run_2 = obs_values[0, 1][obs_mask[0, 1].squeeze()]
    model_sample_paths_run_2 = model_sample_paths[0, 1][obs_mask[0, 1].squeeze()]
    axs[1, 0].plot(
        obs_values_run_2[:, 0], obs_values_run_2[:, 1], obs_values_run_2[:, 2], linewidth=linewidth, color=obs_color, label="Observations"
    )
    axs[1, 1].plot(obs_values_run_2[:, 0], obs_values_run_2[:, 1], obs_values_run_2[:, 2], linewidth=linewidth, color=obs_color, alpha=0.5)
    axs[1, 1].plot(
        model_sample_paths_run_2[:, 0],
        model_sample_paths_run_2[:, 1],
        model_sample_paths_run_2[:, 2],
        linewidth=linewidth,
        color=model_color,
        label="Our Model",
    )
    axs[1, 1].plot(
        model_sample_paths_first_traj[0, 1, :, 0],
        model_sample_paths_first_traj[0, 1, :, 1],
        model_sample_paths_first_traj[0, 1, :, 2],
        linewidth=linewidth,
        color="#CC79A7",
        label="Our Model first traj",
    )
    axs[1, 0].scatter(obs_values_run_2[0, 0], obs_values_run_2[0, 1], obs_values_run_2[0, 2], color="red")
    axs[1, 1].scatter(obs_values_run_2[0, 0], obs_values_run_2[0, 1], obs_values_run_2[0, 2], color="red")

    # titles
    axs[0, 0].set_title("Trajectory 1")
    axs[1, 0].set_title("Trajectory 2")

    axs[0, 1].legend()
    return fig


def figure_grid_3D(model_evaluation: ModelEvaluation, dataset: dict, evaluation_dir: Path):
    model_results = model_evaluation.results

    fig = get_3D_figure_grid(
        dataset["obs_values"], dataset["obs_mask"], model_results["sample_paths"], model_results["sample_paths_from_first_traj"]
    )

    # save
    save_dir: Path = evaluation_dir / "figure_3D_sample_paths"
    save_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"model_{model_evaluation.model_id}"
    save_fig(fig, save_dir, file_name)

    plt.close(fig)


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    dataset_descr = "navier_stokes_sindy_trajectory"

    # How to name experiments
    experiment_descr = "large_models_comparison"

    model_dicts, models_display_ids = get_model_dicts_600k_deg_3_drift_deg_2_diff()

    results_to_load: list[str] = [
        "/home/seifner/repos/FIM/evaluations/navier_stokes_sindy_trajectory/01261810_develop/model_evaluations",
    ]

    data_dir = Path("")  # full path to a dir
    # --------------------------------------------------------------------------------------------------------------------------------- #

    # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
    evaluation_path = Path(project_path) / "evaluations"
    time: str = str(datetime.now().strftime("%m%d%H%M"))
    evaluation_dir: Path = evaluation_path / dataset_descr / (time + "_" + experiment_descr)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # Get dataset
    dataset: dict = get_dataset(data_dir)

    # Get model_map to load models when they are needed
    model_map = model_map_from_dict(model_dicts)

    # Load previous evaluations that don't need to be evaluated anymore
    loaded_evaluations: list[ModelEvaluation] = load_evaluations(results_to_load)

    # Evaluate all models on all datasets
    all_evaluations: list[ModelEvaluation] = [ModelEvaluation(model_id, None) for model_id in model_dicts.keys()]
    to_evaluate: list[ModelEvaluation] = [evaluation for evaluation in all_evaluations if evaluation not in loaded_evaluations]

    # Create, run and save EvaluationConfig
    evaluated: list[ModelEvaluation] = run_navier_stokes_evaluations(to_evaluate, model_map, dataset)
    all_evaluations: list[ModelEvaluation] = loaded_evaluations + evaluated
    save_evaluations(all_evaluations, evaluation_dir / "model_evaluations")

    # Figure
    for model_evaluation in (pbar := tqdm(all_evaluations, total=len(all_evaluations), leave=False)):
        pbar.set_description(f"Sample paths: {model_evaluation.model_id}.")
        figure_grid_3D(model_evaluation, dataset, evaluation_dir)

    pbar.close()
