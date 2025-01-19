from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import optree
import torch
from model_dicts.models_trained_on_600k_deg_3_drift_deg_2_diffusion import get_model_dicts_600k_deg_3_drift_deg_2_diff
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from fim import data_path, project_path
from fim.data.datasets import PaddedFIMSDEDataset
from fim.models.blocks import AModel
from fim.pipelines.sde_sampling_from_model import fimsde_sample_paths
from fim.utils.evaluation_sde import (
    ModelEvaluation,
    ModelMap,
    load_evaluations,
    model_map_from_dict,
    save_evaluations,
    save_fig,
)


def get_navier_stokes_dataloader(data_dir: Path) -> DataLoader:
    if not data_dir.is_absolute():
        data_dir = Path(data_path) / data_dir

    files_to_load = {
        "obs_times": "obs_times.h5",
        "obs_values": "obs_values_pca.h5",
        "obs_values_high_dim": "obs_values_high_dim.h5",
        "pca_data_mean": "pca_data_mean.h5",
        "pca_eigenvalues": "pca_eigenvalues.h5",
        "pca_left_eigenvectors": "pca_left_eigenvectors.h5",
        "pca_right_eigenvectors": "pca_right_eigenvectors.h5",
    }

    dataset = PaddedFIMSDEDataset(
        data_dirs=[data_dir],
        files_to_load=files_to_load,
        max_dim=3,
        batch_size=1,
        shuffle_locations=False,
        shuffle_paths=False,
        shuffle_elements=False,
    )

    return DataLoader(dataset, shuffle=False, batch_size=None)


def evaluate_navier_stokes_model(model: AModel, train_dl: DataLoader, test_dl: DataLoader, device):
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    results = {}

    # train and test are only one element
    train_set: dict = next(iter(train_dl))
    test_set: dict = next(iter(test_dl))

    train_set = optree.tree_map(lambda x: x.to(device), train_set, namespace="fimsde")

    # evaluate vector field on train set
    if "locations" in train_set.keys():
        estimated_concepts = model(train_set, training=False, return_losses=False)
        estimated_concepts = optree.tree_map(lambda x: x.to("cpu"), estimated_concepts, namespace="fimsde")
        results.update({"estimated_concepts": estimated_concepts})

    # sample paths starting from the first state of the test grid
    test_set = optree.tree_map(lambda x: x.to(device), test_set, namespace="fimsde")

    initial_time = test_set["obs_times"][:, :, 0]
    end_time = test_set["obs_times"][:, :, -1]
    grid_size = test_set["obs_times"].shape[-2]
    sample_paths, sample_paths_grid = fimsde_sample_paths(
        model,
        train_set,
        initial_states=test_set["obs_values"][:, :, 0],
        initial_time=initial_time,
        end_time=end_time,
        grid_size=grid_size,
        solver_granularity=20,
    )
    sample_paths, sample_paths_grid = optree.tree_map(lambda x: x.to("cpu"), (sample_paths, sample_paths_grid), namespace="fimsde")

    results.update({"test_sample_paths": sample_paths, "test_sample_paths_grid": sample_paths_grid})

    return results


def run_navier_stokes_evaluations(
    evaluations: list[ModelEvaluation], model_map: ModelMap, train_dl: DataLoader, test_dl: DataLoader, device=None
) -> list[ModelEvaluation]:
    """
    Evaluate model on navier stokes data.

    Args:
        evaluations (list[ModelEvaluation]): List of evaluations to run, specified by their model_id and dataloader_id
        train_dl / test_dl

    Return:
        evaluations (list[ModelEvaluation]): Input evaluations with results from EvaluationTask.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    evaluations_with_results: list[ModelEvaluation] = []

    for evaluation in (pbar := tqdm(evaluations, total=len(evaluations), leave=False)):
        pbar.set_description(f"Model: {str(evaluation.model_id)}. Overall progress")

        model: AModel = model_map[evaluation.model_id]().to(torch.float)

        evaluation.results = evaluate_navier_stokes_model(model, train_dl, test_dl, device)
        evaluations_with_results.append(evaluation)

    return evaluations_with_results


def axes_navier_stokes_latent_individual_dims(
    axs: list[plt.axis],
    data_grid: Tensor,
    data_path: Tensor,
    model_sample_paths_grid: Tensor,
    model_sample_paths: Tensor,
    obs_color: Optional[str] = "black",
    obs_label: Optional[str] = "Observation",
    obs_linestyle: Optional[str] = "dashed",
    model_color: Optional[str] = "#0072B2",
    model_label: Optional[str] = "Model Sample",
) -> None:
    """
    Plot first 3 latent dimensions of navier stokes paths into three axes.

    Args:
        axs (list[plt.axis]): Length 3, one for each dim.
        grids: Shape: [T] paths: Shape: [T, 3]
        + plot config
    """
    for dim in range(3):
        axs[dim].plot(data_grid, data_path[..., dim], label=obs_label if dim == 0 else None, color=obs_color, linestyle=obs_linestyle)
        axs[dim].plot(model_sample_paths_grid, model_sample_paths[..., dim], label=model_label if dim == 0 else None, color=model_color)
        axs[dim].set_ylabel("x_" + str(dim))

    axs[0].legend(loc="upper right", bbox_to_anchor=(1, 1))
    axs[2].set_xlabel("t (s)")


def axis_navier_stokes_latent_3D(
    ax: plt.axis,
    data_path: Tensor,
    model_sample_paths: Tensor,
    obs_color: Optional[str] = "black",
    obs_label: Optional[str] = "Observation",
    obs_linestyle: Optional[str] = "dashed",
    model_color: Optional[str] = "#0072B2",
    model_label: Optional[str] = "Model Sample",
) -> None:
    """
    Plot first 3 latent dimensions of navier stokes paths into a 3D axis.

    Args:
        axs (list[plt.axis]): Length 3, one for each dim.
        paths: Shape: [T, 3]
        + plot config
    """
    ax.plot(data_path[..., 0], data_path[..., 1], data_path[..., 2], label=obs_label, color=obs_color, linestyle=obs_linestyle)
    ax.plot(model_sample_paths[..., 0], model_sample_paths[..., 1], model_sample_paths[..., 2], label=model_label, color=model_color)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1))


def figures_navier_stokes_latent_individual_dims(
    data_grid: Tensor, data_path: Tensor, model_sample_paths_grid: Tensor, model_sample_paths: Tensor, figsize: Optional[tuple] = (10, 5)
):
    """
    Plot results for navier stokes latent dimensions as individual subplots.
    Return two figures: with full test trajectory and just the beginning, like (Course 2023)

    Args:
        grids: Shape: [T] paths: Shape: [T, 3]
        + figures config
    """
    fig_full, axs_full = plt.subplots(3, 1, figsize=figsize, dpi=300, tight_layout=True)
    axes_navier_stokes_latent_individual_dims(axs_full, data_grid / 10, data_path, model_sample_paths_grid / 10, model_sample_paths)
    # / 10 to get same time scale as Course

    fig_course, axs_course = plt.subplots(3, 1, figsize=figsize, dpi=300, tight_layout=True)
    axes_navier_stokes_latent_individual_dims(
        axs_course, data_grid[:40] / 10, data_path[:40], model_sample_paths_grid[:40] / 10, model_sample_paths[:40]
    )

    return fig_full, fig_course


def figures_navier_stokes_latent_3D(data_path: Tensor, model_sample_paths: Tensor, figsize: Optional[tuple] = (5, 5)):
    """
    Plot results for navier stokes latent dimensions as 3D plots.
    Return two figures: with full test trajectory and just the beginning, like (Course 2023)

    Args:
        grids: Shape: [T] paths: Shape: [T, 3]
        + figures config
    """
    # create figures
    fig_full = plt.figure(figsize=figsize, dpi=300)
    ax_full = fig_full.add_axes(111, projection="3d")
    axis_navier_stokes_latent_3D(ax_full, data_path, model_sample_paths)

    fig_course = plt.figure(figsize=figsize, dpi=300)
    ax_course = fig_course.add_axes(111, projection="3d")
    axis_navier_stokes_latent_3D(ax_course, data_path[:40], model_sample_paths[:40])

    return fig_full, fig_course


def latent_sample_path_individual(model_evaluation: ModelEvaluation, test_dl: DataLoader, evaluation_dir: Path):
    """
    Plot latent sample paths of one model against the test set as individual subplot per dimension.

    Args:
        model_evaluation (ModelEvaluation): Results to plot.
        test_dl (DataLoader): Contains test trajectory
        evaluation_dir (Path): Base dir to save Figures in.
    """
    # extract data
    test_set: dict = next(iter(test_dl))
    test_grid = test_set.get("obs_times").squeeze()  # [T]
    test_path = test_set.get("obs_values").squeeze()  # [T, 3]

    # extract results
    model_sample_paths_grid = model_evaluation.results.get("test_sample_paths_grid").squeeze()  # [T]
    model_sample_paths = model_evaluation.results.get("test_sample_paths").squeeze()  # [T, 3]

    # plot
    fig_full, fig_course = figures_navier_stokes_latent_individual_dims(test_grid, test_path, model_sample_paths_grid, model_sample_paths)

    # save
    save_dir: Path = evaluation_dir / "latent_path_individual_dimensions_full" / model_evaluation.model_id
    save_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"model_{model_evaluation.model_id}"
    save_fig(fig_full, save_dir, file_name)

    save_dir: Path = evaluation_dir / "latent_path_individual_dimensions_course" / model_evaluation.model_id
    save_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"model_{model_evaluation.model_id}"
    save_fig(fig_course, save_dir, file_name)

    plt.close(fig_full)
    plt.close(fig_course)


def latent_sample_path_3D(model_evaluation: ModelEvaluation, test_dl: DataLoader, evaluation_dir: Path):
    """
    Plot latent sample paths of one model against the test set in 3D.

    Args:
        model_evaluation (ModelEvaluation): Results to plot.
        test_dl (DataLoader): Contains test trajectory
        evaluation_dir (Path): Base dir to save Figures in.
    """
    # extract data
    test_set: dict = next(iter(test_dl))
    test_path = test_set.get("obs_values").squeeze()  # [T, 3]

    # extract results
    model_sample_paths = model_evaluation.results.get("test_sample_paths").squeeze()  # [T, 3]

    # plot
    fig_full, fig_course = figures_navier_stokes_latent_3D(test_path, model_sample_paths)

    # save
    save_dir: Path = evaluation_dir / "latent_path_3D_full" / model_evaluation.model_id
    save_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"model_{model_evaluation.model_id}"
    save_fig(fig_full, save_dir, file_name)

    save_dir: Path = evaluation_dir / "latent_path_3D_course" / model_evaluation.model_id
    save_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"model_{model_evaluation.model_id}"
    save_fig(fig_course, save_dir, file_name)

    plt.close(fig_full)
    plt.close(fig_course)


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    dataset_descr = "navier_stokes"

    # How to name experiments
    experiment_descr = "develop"

    model_dicts, models_display_ids = get_model_dicts_600k_deg_3_drift_deg_2_diff()

    results_to_load: list[str] = [
        # "/home/seifner/repos/FIM/evaluations/navier_stokes/01192207_develop/model_evaluations",
    ]

    train_path = Path("processed/test/20250116_preprocessed_navier_stokes/train/")
    test_path = Path("processed/test/20250116_preprocessed_navier_stokes/test/")
    # --------------------------------------------------------------------------------------------------------------------------------- #

    # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
    evaluation_path = Path(project_path) / "evaluations"
    time: str = str(datetime.now().strftime("%m%d%H%M"))
    evaluation_dir: Path = evaluation_path / dataset_descr / (time + "_" + experiment_descr)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # Get dataloaders inits and their display ids (for ModelEvaluation)
    train_dl = get_navier_stokes_dataloader(train_path)
    test_dl = get_navier_stokes_dataloader(test_path)

    # Get model_map to load models when they are needed
    model_map = model_map_from_dict(model_dicts)

    # Load previous evaluations that don't need to be evaluated anymore
    loaded_evaluations: list[ModelEvaluation] = load_evaluations(results_to_load)

    # Evaluate all models on all datasets
    all_evaluations: list[ModelEvaluation] = [ModelEvaluation(model_id, None) for model_id in model_dicts.keys()]
    to_evaluate: list[ModelEvaluation] = [evaluation for evaluation in all_evaluations if evaluation not in loaded_evaluations]

    # Create, run and save EvaluationConfig
    evaluated: list[ModelEvaluation] = run_navier_stokes_evaluations(to_evaluate, model_map, train_dl, test_dl)

    # Add loaded evaluations
    all_evaluations: list[ModelEvaluation] = loaded_evaluations + evaluated

    save_evaluations(all_evaluations, evaluation_dir / "model_evaluations")

    # Figure of sample paths on test set
    for model_evaluation in (pbar := tqdm(all_evaluations, total=len(all_evaluations), leave=False)):
        pbar.set_description(f"3 x 1D sample paths on test, model: {model_evaluation.model_id}.")
        latent_sample_path_individual(model_evaluation, test_dl, evaluation_dir)

    pbar.close()

    for model_evaluation in (pbar := tqdm(all_evaluations, total=len(all_evaluations), leave=False)):
        pbar.set_description(f"3D sample paths on test, model: {model_evaluation.model_id}.")
        latent_sample_path_3D(model_evaluation, test_dl, evaluation_dir)

    pbar.close()
