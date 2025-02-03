import itertools
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import optree
import pandas as pd
import torch
from model_dicts.models_trained_on_600k_deg_3_drift_deg_2_diffusion import get_model_dicts_600k_deg_3_drift_deg_2_diff
from torch.utils.data import DataLoader
from tqdm import tqdm

from fim import project_path
from fim.data.datasets import PaddedFIMSDEDataset
from fim.data.utils import load_h5
from fim.models.sde import FIMSDE, SDEConcepts
from fim.pipelines.sde_sampling_from_model import fimsde_sample_paths_on_masked_grid
from fim.utils.evaluation_sde import (
    ModelEvaluation,
    ModelMap,
    NumpyEncoder,
    load_evaluations,
    model_map_from_dict,
    save_evaluations,
    save_fig,
    save_table,
)
from fim.utils.evaluation_sde_synthetic_datasets import (
    plot_1D_synthetic_data_figure_grid,
    plot_2D_synthetic_data_figure_grid,
)


def get_dataset_from_opper_generated_data(data_dir: Path, stride: int, num_initial_states: int) -> tuple:
    """
    Subsample data on a fine grid of 5001 points. That means higher strides means fewer data.
    """
    # load and pad data to max dim
    files_to_load = {
        "obs_times": "obs_times.h5",
        "obs_values": "obs_values.h5",
        "locations": "locations.h5",
        "drift_at_locations": "drift_at_locations.h5",
        "diffusion_at_locations": "diffusion_at_locations.h5",
    }

    dataset = PaddedFIMSDEDataset(
        data_dirs=data_dir,
        batch_size=1,
        files_to_load=files_to_load,
        max_dim=3,
        shuffle_locations=False,
        shuffle_paths=False,
        shuffle_elements=False,
        load_data_at_init=True,
    )

    dataset: dict = dataset.data

    # sample initial states from observations of the path
    obs_values = dataset["obs_values"][0, 0]  # [5000, 1 / 2]
    indices = np.random.choice(obs_values.shape[0], size=num_initial_states)
    initial_states = obs_values[indices][None]  # [1, num_initial_states, 1 / 2 ]
    dataset["initial_states"] = initial_states

    # subsample observations by strides of provided length
    dataset["obs_values"] = dataset["obs_values"][:, :, ::stride, :]
    dataset["obs_times"] = dataset["obs_times"][:, :, ::stride, :]
    dataset["obs_mask"] = dataset["obs_mask"][:, :, ::stride, :]

    # record stride
    dataset["stride_length"] = stride

    return dataset


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

    dataset = optree.tree_map(lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, dataset, namespace="fimsde")

    # dataloader can pad data to dim 3
    D = dataset["dimension_mask"][0, 0].sum()
    initial_states = dataset.get("initial_states")

    # if initial_states is None:  # default to standard normal
    #     initial_states = torch.randn(1, sample_paths_count, D).to(device)
    #
    #   if D < 3:
    #       initial_states = torch.concatenate([initial_states, torch.zeros(1, sample_paths_count, 3 - D).to(device)], dim=-1)

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

    # dataloader can pad data to dim 3
    D = dataset["dimension_mask"][0, 0].sum()
    estimated_concepts.drift = estimated_concepts.drift[..., :D]
    estimated_concepts.diffusion = estimated_concepts.diffusion[..., :D]
    if "sample_paths" in results.keys():
        results["sample_paths"] = results["sample_paths"][..., :D]

    # mse at locations
    gt_drift = dataset["drift_at_locations"]
    gt_drift = gt_drift[..., :D]
    est_drift = estimated_concepts.drift
    assert gt_drift.shape == est_drift.shape
    drift_mse_mean = ((gt_drift - est_drift) ** 2).mean()
    drift_mse_std = ((gt_drift - est_drift) ** 2).std()

    gt_diffusion = dataset["diffusion_at_locations"]
    gt_diffusion = gt_diffusion[..., :D]
    est_diffusion = estimated_concepts.diffusion
    assert gt_diffusion.shape == est_diffusion.shape
    diffusion_mse_mean = ((gt_diffusion - est_diffusion) ** 2).mean()
    diffusion_mse_std = ((gt_diffusion - est_diffusion) ** 2).std()

    mse = {"drift": (drift_mse_mean, drift_mse_std), "diffusion": (diffusion_mse_mean, diffusion_mse_std)}
    results.update({"mse": mse, "est_drift": est_drift, "est_diffusion": est_diffusion})

    results = optree.tree_map(lambda x: x.detach().to("cpu"), results, namespace="fimsde")

    return results


def load_gp_model_evaluations(path_to_gp_results: Path, systems_to_load: list[str], dataloaders: dict[str, DataLoader]):
    model_evaluations = []

    for system_name in systems_to_load:
        # load vector fields
        est_drift = load_h5(path_to_gp_results / system_name / "estimated_drift_at_locations.h5")
        est_diffusion = load_h5(path_to_gp_results / system_name / "estimated_diffusion_at_locations.h5")

        # load gt data
        dataset = dataloaders[(system_name, 1)]
        gt_drift = dataset["drift_at_locations"]
        D = dataset["dimension_mask"][0, 0].sum()  # dataloader can pad data to dim 3
        gt_drift = gt_drift[..., :D]
        est_drift = est_drift[..., :D]
        assert gt_drift.shape == est_drift.shape
        drift_mse_mean = ((gt_drift - est_drift) ** 2).mean()
        drift_mse_std = ((gt_drift - est_drift) ** 2).std()

        gt_diffusion = dataset["diffusion_at_locations"]
        gt_diffusion = gt_diffusion[..., :D]
        est_diffusion = est_diffusion[..., :D]
        assert gt_diffusion.shape == est_diffusion.shape
        diffusion_mse_mean = ((gt_diffusion - est_diffusion) ** 2).mean()
        diffusion_mse_std = ((gt_diffusion - est_diffusion) ** 2).std()

        # get versions of MSEs
        mse = {"drift": (drift_mse_mean, drift_mse_std), "diffusion": (diffusion_mse_mean, diffusion_mse_std)}
        estimated_concepts = SDEConcepts(locations=None, drift=est_drift, diffusion=est_diffusion)
        results = {"mse": mse, "estimated_concepts": estimated_concepts}

        model_evaluations.append(ModelEvaluation(model_id="GP_based", dataloader_id=(system_name, 1), results=results))

    return model_evaluations


def table_of_metrics(model_evaluations: list[ModelEvaluation], evaluation_dir: Path, metrics: list[str] = ["mse"], precision: int = 2):
    def _get_row_from_evaluation(model_evaluation: ModelEvaluation, vector_field: str):
        row = {"model": model_evaluation.model_id, "data": model_evaluation.dataloader_id[0], "stride": model_evaluation.dataloader_id[1]}
        for metric in metrics:
            mean, std = model_evaluation.results[metric][vector_field]  # Tensors
            mean = round(mean.item(), precision)
            std = round(std.item(), precision)
            row.update({metric: str(mean) + r" $\pm$ " + str(std), "vector_field": vector_field})

        return row

    all_rows = [_get_row_from_evaluation(eval, vector_field) for eval in model_evaluations for vector_field in ["drift", "diffusion"]]
    all_cols = optree.tree_map(lambda *x: x, *all_rows)  # concatenate each value of rows to columns

    df = pd.DataFrame.from_dict(all_cols)

    # all models, data and metrics in one
    df_all = df.set_index(["data", "stride", "model"])
    save_table(df_all, evaluation_dir, "all_metrics_all_models")

    # one table per metric, rows are data and vector fields, columns are models
    for metric in metrics:
        df_metric = df[["model", "data", "vector_field", "stride", metric]]
        df_metric = df_metric.set_index(["model", "data", "vector_field", "stride"])[metric]
        df_metric = df_metric.unstack("model")

        save_table(df_metric, evaluation_dir / "tables_per_metric", metric)


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    dataset_descr = "2D_dense_data_from_wang_opper_svise"

    # How to name experiments
    experiment_descr = "1000_paths_500_length_initial_states_from_paths_subsampled_1_or_10"

    model_dicts, models_display_ids = get_model_dicts_600k_deg_3_drift_deg_2_diff()

    results_to_load: list[str] = [
        # "/home/seifner/repos/FIM/evaluations/2D_dense_data_from_wang_opper_svise/01281104_1000_paths_500_length_initial_states_from_paths_subsampled_1_or_10/model_evaluations"
    ]

    # data and results to load
    path_to_data = Path("/home/seifner/repos/FIM/data/processed/test/20250126_2D_dense_data_from_wang_opper_svise/systems_data/")
    path_to_gp_results = Path("/home/seifner/repos/FIM/data/processed/test/20250126_2D_dense_data_from_wang_opper_svise/gp_results")
    systems_to_load: list[str] = [
        "Damped_Cubic",
        "Damped_Linear",
        "Double_Well_Max_Diffusion",
        "Duffing",
        "Glycosis",
        "Hopf",
        "Syn_Drift",
        "Wang",
    ]

    metrics = ["mse"]
    metrics_precision = 3
    strides = [1, 10]

    sample_model_paths = True
    sample_paths_count = 1000
    dt = 0.002
    sample_path_steps = 500
    # --------------------------------------------------------------------------------------------------------------------------------- #

    # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
    evaluation_path = Path(project_path) / "evaluations"
    time: str = str(datetime.now().strftime("%m%d%H%M"))
    evaluation_dir: Path = evaluation_path / dataset_descr / (time + "_" + experiment_descr)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # Get dataloaders inits and their display ids (for ModelEvaluation)
    dataloaders = {
        (system, stride): get_dataset_from_opper_generated_data(path_to_data / system, stride, num_initial_states=sample_paths_count)
        for system in systems_to_load
        for stride in strides
    }
    dataloaders_display_ids = {system: system for system in systems_to_load}  # for now

    # Setup inits for models and dataloaders
    model_map = model_map_from_dict(model_dicts)

    # Load previous evaluations that don't need to be evaluated anymore
    loaded_evaluations: list[ModelEvaluation] = load_evaluations(results_to_load)

    # Evaluate all models on all datasets
    all_evaluations: list[ModelEvaluation] = [
        ModelEvaluation(model_id, dataloader_id) for model_id, dataloader_id in itertools.product(model_dicts.keys(), dataloaders.keys())
    ]
    to_evaluate: list[ModelEvaluation] = [evaluation for evaluation in all_evaluations if evaluation not in loaded_evaluations]

    # Add model evaluations from GP model
    gp_model_evaluations: list[ModelEvaluation] = load_gp_model_evaluations(path_to_gp_results, systems_to_load, dataloaders)

    # Create, run and save evaluations
    evaluated: list[ModelEvaluation] = run_evaluations(
        to_evaluate, model_map, dataloaders, sample_model_paths, sample_paths_count, dt, sample_path_steps
    )

    # Add loaded evaluations
    all_evaluations: list[ModelEvaluation] = loaded_evaluations + evaluated
    save_evaluations(all_evaluations, evaluation_dir / "model_evaluations")
    all_evaluations = all_evaluations + gp_model_evaluations

    # metrics tables
    table_of_metrics(all_evaluations, evaluation_dir, metrics, metrics_precision)

    # Figures with subplot grid containing results from multiple equations per dataset
    if sample_model_paths is True:
        for model_evaluation in (pbar := tqdm(all_evaluations, total=len(all_evaluations), leave=False)):
            pbar.set_description(
                f"Saving figure grids for model {model_evaluation.model_id} and dataloader {model_evaluation.dataloader_id}."
            )

            dataset: dict = dataloaders[model_evaluation.dataloader_id]
            dim = dataset["dimension_mask"][0, 0, :].sum().item()

            if dim == 1:
                grid_plot_func = plot_1D_synthetic_data_figure_grid
            if dim == 2:
                grid_plot_func = plot_2D_synthetic_data_figure_grid

            fig = grid_plot_func(
                dataset["locations"],
                dataset["drift_at_locations"],
                dataset["diffusion_at_locations"],
                model_evaluation.results["estimated_concepts"].drift,
                model_evaluation.results["estimated_concepts"].diffusion,
                dataset["obs_times"],
                dataset["obs_values"],
                dataset["obs_mask"].bool(),
                model_evaluation.results.get("sample_paths_grid"),
                model_evaluation.results.get("sample_paths"),
            )

            # save
            save_dir: Path = (
                evaluation_dir
                / "figure_grid"
                / (model_evaluation.dataloader_id[0] + "_stride_" + str(model_evaluation.dataloader_id[1]))
                / model_evaluation.model_id
            )
            save_dir.mkdir(parents=True, exist_ok=True)
            file_name = (
                f"data_{model_evaluation.dataloader_id[0]}_stride_{model_evaluation.dataloader_id[1]}_model_{model_evaluation.model_id}"
            )
            save_fig(fig, save_dir, file_name)

            plt.close(fig)

        pbar.close()

    # check if sample paths contain Nans
    print("Sampled paths contain Nans or infs:")
    for model_evaluation in all_evaluations:
        if model_evaluation.results.get("sample_paths") is not None:
            paths = model_evaluation.results["sample_paths"]
            is_finite = torch.isfinite(paths).all().item()
            if is_finite is False:
                print(model_evaluation.model_id, model_evaluation.dataloader_id, " contains Nans or infs")

    # for each model and stride, save sample paths in json in dict
    for model_name in model_dicts.keys():
        for stride in strides:
            selected_evals = [model_eval for model_eval in all_evaluations if model_eval.model_id == model_name]
            selected_evals = [model_eval for model_eval in selected_evals if model_eval.dataloader_id[1] == stride]

            paths_per_system = {  # keys: system name, values: paths
                model_eval.dataloader_id[0].replace("_", " "): np.array(model_eval.results["sample_paths"].squeeze(0))
                for model_eval in selected_evals
            }

            # Convert to JSON
            json_data = json.dumps(paths_per_system, cls=NumpyEncoder)

            # Write JSON data to a file
            json_dir = evaluation_dir / "json_paths_per_system"
            json_dir.mkdir(exist_ok=True, parents=True)

            file: Path = json_dir / ("model_" + model_name + "_stride_" + str(stride) + ".json")
            with open(file, "w") as file:
                file.write(json_data)
