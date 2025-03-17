import itertools
import json
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import Optional

import matplotlib.pyplot as plt
import optree
import pandas as pd
import torch
from model_dicts.models_trained_on_600k_deg_3_drift_deg_2_diffusion import get_model_dicts_600k_deg_3_drift_deg_2_diff
from tqdm import tqdm

from fim import project_path
from fim.data.datasets import PaddedFIMSDEDataset
from fim.data.utils import load_h5
from fim.models.sde import FIMSDE
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
from fim.utils.evaluation_sde_synthetic_datasets import plot_1D_synthetic_data_figure_grid, plot_2D_synthetic_data_figure_grid


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


def get_dataset_from_opper_generated_data(data_dir: Path, stride: int) -> tuple:
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

    # subsample observations by strides of provided length
    dataset: dict = dataset.data
    dataset["obs_values"] = dataset["obs_values"][:, :, ::stride, :]
    dataset["obs_times"] = dataset["obs_times"][:, :, ::stride, :]
    dataset["obs_mask"] = dataset["obs_mask"][:, :, ::stride, :]

    # record stride
    dataset["stride_length"] = stride

    return dataset


def get_dataset_from_self_generated_data(
    data_dir: Path, ksig_reference_obs_values: torch.Tensor, stride: int, dt: float, target_length: int, num_initial_states: int
) -> tuple:
    """
    Subsample data from a large fine grid by strides. truncate to a target length.
    i.e. stride does not imply fewer points in trajectories.
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

    # subsample observations by strides of provided length
    dataset: dict = dataset.data
    dataset["obs_values"] = dataset["obs_values"][:, :, ::stride, :]
    dataset["obs_times"] = dataset["obs_times"][:, :, ::stride, :]
    dataset["obs_mask"] = dataset["obs_mask"][:, :, ::stride, :]

    # take ksig paths initial states as initial states for sampling paths
    # B, P, T, D = ksig_reference_obs_values.shape
    dataset["initial_states"] = ksig_reference_obs_values[:, :, 0, :]

    # truncate observations to target length
    dataset["obs_values"] = dataset["obs_values"][:, :, :target_length, :]
    dataset["obs_times"] = dataset["obs_times"][:, :, :target_length, :]
    dataset["obs_mask"] = dataset["obs_mask"][:, :, :target_length, :]

    # record stride
    dataset["stride_length"] = stride

    # record time between two observations
    dataset["tau"] = stride * dt

    return dataset


# def run_evaluations(
#     to_evaluate: list[ModelEvaluation], model_map: ModelMap, dataloaders: dict, device: Optional[str] = None
# ) -> list[ModelEvaluation]:
#     """
#     Evaluate models on datasets from dataloaders.
#
#     Args:
#         evaluations (list[ModelEvaluation]): List of evaluations to run, specified by their model_id and dataloader_id
#         model map: Returning required models
#         dataloaders: Returning required dataloaders.
#         sample_paths (bool): If True, sample a path for each model.
#
#     Return:
#         evaluations (list[ModelEvaluation]): Input evaluations.
#     """
#
#     if device is None:
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#
#     evaluations_with_results: list[ModelEvaluation] = []
#
#     for evaluation in (pbar := tqdm(to_evaluate, total=len(to_evaluate), leave=False)):
#         pbar.set_description(f"Model: {str(evaluation.model_id)}. Overall progress")
#
#         model: FIMSDE = model_map[evaluation.model_id]().to(torch.float)
#         dataset = dataloaders[evaluation.dataloader_id]
#
#         evaluation.results = evaluate_model(model, dataset, device=device)
#         evaluations_with_results.append(evaluation)
#
#     return evaluations_with_results
#
#
# def evaluate_model(model: FIMSDE, dataset: dict, device: Optional[str] = None):
#     model.eval()
#
#     results = {}
#
#     # sample on device
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model.to(device)
#
#     dataset = optree.tree_map(lambda x: x.to(device) if isinstance(x, torch.Tensor) else x, dataset, namespace="fimsde")
#
#     # get vector fields at locations
#     estimated_concepts = model(dataset, training=False, return_losses=False)
#     results.update({"estimated_concepts": estimated_concepts})
#
#     # mse at locations
#     gt_drift = dataset["drift_at_locations"]
#     D = dataset["dimension_mask"][0, 0].sum()  # dataloader can pad data to dim 3
#     gt_drift = gt_drift[..., :D]
#     est_drift = estimated_concepts.drift[..., :D]
#     assert gt_drift.shape == est_drift.shape
#     drift_mse_mean = ((gt_drift - est_drift) ** 2).mean()
#     drift_mse_std = ((gt_drift - est_drift) ** 2).std()
#
#     gt_diffusion = dataset["diffusion_at_locations"]
#     gt_diffusion = gt_diffusion[..., :D]
#     est_diffusion = estimated_concepts.diffusion[..., :D]
#     assert gt_diffusion.shape == est_diffusion.shape
#     diffusion_mse_mean = ((gt_diffusion - est_diffusion) ** 2).mean()
#     diffusion_mse_std = ((gt_diffusion - est_diffusion) ** 2).std()
#
#     mse = {"drift": (drift_mse_mean, drift_mse_std), "diffusion": (diffusion_mse_mean, diffusion_mse_std)}
#     results.update({"mse": mse, "est_drift": est_drift, "est_diffusion": est_diffusion})
#
#     results = optree.tree_map(lambda x: x.detach().to("cpu"), results, namespace="fimsde")
#
#     return results


def table_of_metrics(model_evaluations: list[ModelEvaluation], evaluation_dir: Path, precision: int = 2):
    def _get_row_from_evaluation(model_evaluation: ModelEvaluation, vector_field: str):
        dataset, stride, length = model_evaluation.dataloader_id
        row = {"model": model_evaluation.model_id, "data": dataset, "stride": stride, "length": length}

        mean, std = model_evaluation.results["mse"][vector_field]  # Tensors
        mean = round(mean.item(), precision)
        std = round(std.item(), precision)
        row.update({"mse": str(mean) + r" $\pm$ " + str(std), "vector_field": vector_field})

        return row

    all_rows = [_get_row_from_evaluation(eval, vector_field) for eval in model_evaluations for vector_field in ["drift", "diffusion"]]
    all_cols = optree.tree_map(lambda *x: x, *all_rows)  # concatenate each value of rows to columns

    df = pd.DataFrame.from_dict(all_cols)

    # one table per dataset and model, rows are strides, columns are length
    dfs_by_data_and_model: dict = df.groupby(["data", "model"])

    for group_key in dfs_by_data_and_model.groups.keys():
        data_name, model_name = group_key
        df = dfs_by_data_and_model.get_group(group_key)
        df = df.sort_values(["stride", "length"])
        df = df.set_index(["vector_field", "stride", "length"])["mse"]
        df = df.unstack(["length"])

        save_table(df, evaluation_dir / data_name, data_name + "_" + model_name)


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    dataset_descr = "synthetic_systems_vf_and_paths_evaluation"

    # How to name experiments
    # experiment_descr = "large_models_comparison"
    # experiment_descr = "finding_best_scale_for_our_model"
    # experiment_descr = "data_for_table"
    experiment_descr = "develop_old_script"

    model_dicts, models_display_ids = get_model_dicts_600k_deg_3_drift_deg_2_diff()

    results_to_load: list[str] = [
        # "/home/seifner/repos/FIM/evaluations/data_density_ablation_study/01291612_data_for_table/model_evaluations"
    ]

    # data and results to load
    # # previous examles
    # path_to_data = Path("/home/seifner/repos/FIM/data/processed/test/20250126_2D_dense_data_from_wang_opper_svise/systems_data/")
    # # path_to_gp_results = Path("/home/seifner/repos/FIM/data/processed/test/20250126_2D_dense_data_from_wang_opper_svise/gp_results")
    # systems_to_load: list[str] = [
    #     "Damped_Cubic",
    #     "Damped_Linear",
    #     "Double_Well_Max_Diffusion",
    #     "Duffing",
    #     "Glycosis",
    #     "Hopf",
    #     "Syn_Drift",
    #     "Wang",
    # ]

    # # previous examles
    # path_to_data = Path("/home/seifner/repos/FIM/data/processed/test/20250127_opper_svise_wang_long_dense_for_density_ablation/")
    # systems_to_load: list[str] = [
    #     "damped_cubic_oscillator",
    #     "damped_linear_oscillator",
    #     "duffing_oscillator",
    #     "hopf_bifurcation",
    #     "opper_double_well_constant_diff",
    #     "opper_double_well_state_dep_diff",
    #     "opper_two_d_synthetic",
    #     "selkov_glycolysis",
    #     "wang_double_well",
    #     "wang_two_d_synthetic",
    # ]

    # systems in table of paper
    path_to_data = Path(
        "/home/seifner/repos/FIM/data/processed/test/20250129_opper_svise_wang_long_dense_for_density_ablation_5_realizations/"
    )
    path_to_ksig_reference_data = Path(
        "/home/seifner/repos/FIM/data/processed/test/20250129_opper_svise_wang_long_dense_for_density_ablation_5_realizations_KSIG_reference_paths"
    )
    systems_to_load: list[str] = [
        "Damped_Linear",
        "Damped_Cubic",
        "Duffing",
        "Glycosis",
        "Hopf",
        "Double_Well",
        "Wang",
        "Syn_Drift",
    ]

    metrics_precision = 3
    # subsampling_strides = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100]
    # target_length = 5001
    # subsampling_strides = [1, 3, 5, 8, 10]
    # target_lengths = [2500, 5000, 8000, 12800]
    subsampling_strides = [1, 5, 10]
    target_lengths = [5000]
    sample_paths = True
    sample_paths_count = 100
    dt = 0.002
    sample_path_steps = 500

    save_reference_data = True
    # --------------------------------------------------------------------------------------------------------------------------------- #

    # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
    evaluation_path = Path(project_path) / "evaluations"
    time: str = str(datetime.now().strftime("%m%d%H%M"))
    evaluation_dir: Path = evaluation_path / dataset_descr / (time + "_" + experiment_descr)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # get datasets, load ksig obs values for initial states
    ksig_ref_obs_values = {system: load_h5(path_to_ksig_reference_data / system / "obs_values.h5") for system in systems_to_load}
    ksig_ref_obs_values = {
        system: ksig_values[:, :, : sample_paths_count * sample_path_steps, :] for system, ksig_values in ksig_ref_obs_values.items()
    }  # reshape 1 into many path
    ksig_ref_obs_values = {
        system: ksig_values.reshape(-1, sample_paths_count, sample_path_steps, ksig_values.shape[-1])
        for system, ksig_values in ksig_ref_obs_values.items()
    }

    datasets = {
        (system, stride, target_length): get_dataset_from_self_generated_data(
            path_to_data / system, ksig_ref_obs_values[system], stride, dt, target_length, sample_paths_count
        )
        for system in systems_to_load
        for stride in subsampling_strides
        for target_length in target_lengths
    }

    # Setup inits for models and dataloaders
    model_map = model_map_from_dict(model_dicts)

    # Load previous evaluations that don't need to be evaluated anymore
    loaded_evaluations: list[ModelEvaluation] = load_evaluations(results_to_load)

    # Evaluate all models on all datasets
    all_evaluations: list[ModelEvaluation] = [
        ModelEvaluation(model_id, dataloader_id) for model_id, dataloader_id in itertools.product(model_dicts.keys(), datasets.keys())
    ]
    to_evaluate: list[ModelEvaluation] = [evaluation for evaluation in all_evaluations if evaluation not in loaded_evaluations]

    # Create, run and save evaluations
    evaluated: list[ModelEvaluation] = run_evaluations(
        to_evaluate, model_map, datasets, sample_paths, sample_paths_count, dt, sample_path_steps
    )

    # Add loaded evaluations
    all_evaluations: list[ModelEvaluation] = loaded_evaluations + evaluated
    save_evaluations(all_evaluations, evaluation_dir / "model_evaluations")

    # metrics tables
    table_of_metrics(all_evaluations, evaluation_dir, metrics_precision)

    # save inference part of datasets for reproducibility
    reduced_datasets = []
    for dataset_id, data in datasets.items():
        # reverse padding of dataloader
        D = data["dimension_mask"][0, 0, :].sum().item()
        data["obs_values"] = data["obs_values"][..., :D]
        data["locations"] = data["locations"][..., :D]
        reduced_datasets.append(
            {
                "name": dataset_id[0].replace("_", " "),
                "tau": data["tau"],
                "observations": data["obs_values"],
                "locations": data["locations"],
                "initial_states": data["initial_states"],
            }
        )

    print("Reduced datasets to save")
    pprint(optree.tree_map(lambda x: x.shape if isinstance(x, torch.Tensor) else x, reduced_datasets))

    # save reference paths for e.g. KSIG of same size as sampled paths
    # ksig_datasets = {  # per realization, one long path (without strides), that is cut into sample_paths_count paths
    #     system: get_dataset_from_self_generated_data(
    #         path_to_ksig_reference_data / system,
    #         stride=1,
    #         dt=dt,
    #         target_length=sample_paths_count * sample_path_steps,
    #         num_initial_states=1,  # does not matter here
    #     )
    #     for system in systems_to_load
    # }
    #
    reference_datasets = []
    for system, data in ksig_ref_obs_values.items():
        reference_datasets.append(
            {
                "name": system.replace("_", " "),
                "real_paths": data,
            }
        )

    print("Reference datasets to save")
    pprint(optree.tree_map(lambda x: x.shape if isinstance(x, torch.Tensor) else x, reference_datasets))

    # save sampled paths and vector field inferences
    model_outputs = []
    for model_evaluation in all_evaluations:
        name = model_evaluation.dataloader_id[0].replace("_", " ")
        stride = model_evaluation.dataloader_id[1]

        model_outputs.append(
            {
                "name": name,
                "tau": stride * dt,
                "synthetic_paths": model_evaluation.results["sample_paths"],
                "drift_at_locations": model_evaluation.results["estimated_concepts"].drift,
                "diffusion_at_locations": model_evaluation.results["estimated_concepts"].diffusion,
            }
        )

    print("Model outputs to save")
    pprint(optree.tree_map(lambda x: x.shape if isinstance(x, torch.Tensor) else x, model_outputs))

    ground_truth_drift_diffusion = []
    for dataset_id, data in datasets.items():
        system = dataset_id[0]
        stride = dataset_id[1]
        tau = data["tau"]
        if tau == 0.002 and stride == 1:  # save only once
            D = data["dimension_mask"][0, 0, :].sum().item()
            ground_truth_drift_diffusion.append(
                {
                    "name": system.replace("_", " "),
                    "locations": data["locations"][..., :D],
                    "drift_at_locations": data["drift_at_locations"][..., :D],
                    "diffusion_at_locations": data["diffusion_at_locations"][..., :D],
                }
            )

    print("Ground truth drift and diffusion to save")
    pprint(optree.tree_map(lambda x: x.shape if isinstance(x, torch.Tensor) else x, ground_truth_drift_diffusion))

    # save ground-truth drift and diffusion values
    if save_reference_data is True:
        assert len(model_dicts) == 1, "Only works for a single model right now"

        def _check_finite(x):
            if isinstance(x, torch.Tensor):
                assert torch.torch.isfinite(x).all().item()

        optree.tree_map(_check_finite, (model_outputs, reference_datasets, reduced_datasets), namespace="fimsde")

        model_outputs, reference_datasets, reduced_datasets, ground_truth_drift_diffusion = optree.tree_map(
            lambda x: x.detach().to("cpu").numpy() if isinstance(x, torch.Tensor) else x,
            (model_outputs, reference_datasets, reduced_datasets, ground_truth_drift_diffusion),
        )

        for data, filename in zip(
            [model_outputs, reference_datasets, reduced_datasets],
            ["model_paths.json", "ksig_reference_paths.json", "systems_coarse_observations.json"],
            # [reference_datasets, reduced_datasets, ground_truth_drift_diffusion],
            # ["ksig_reference_paths.json", "systems_coarse_observations.json", "ground_truth_drift_diffusion.json"],
        ):
            # Convert to JSON
            json_data = json.dumps(data, cls=NumpyEncoder)

            # Write JSON data to a file
            json_dir = evaluation_dir / "data_jsons"
            json_dir.mkdir(exist_ok=True, parents=True)

            file: Path = json_dir / filename
            with open(file, "w") as file:
                file.write(json_data)

    # Figures with subplot grid containing results from multiple equations per dataset
    if sample_paths is True:
        for model_evaluation in (pbar := tqdm(all_evaluations, total=len(all_evaluations), leave=False)):
            pbar.set_description(
                f"Saving figure grids for model {model_evaluation.model_id} and dataloader {model_evaluation.dataloader_id}."
            )

            dataset: dict = datasets[model_evaluation.dataloader_id]
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
                / model_evaluation.dataloader_id[0]
                / ("length_" + str(model_evaluation.dataloader_id[2]))
                / ("stride_" + str(model_evaluation.dataloader_id[1]))
            )
            save_dir.mkdir(parents=True, exist_ok=True)
            file_name = (
                f"data_{model_evaluation.dataloader_id[0]}_stride_{model_evaluation.dataloader_id[1]}_model_{model_evaluation.model_id}"
            )
            save_fig(fig, save_dir, file_name)

            plt.close(fig)

        pbar.close()
