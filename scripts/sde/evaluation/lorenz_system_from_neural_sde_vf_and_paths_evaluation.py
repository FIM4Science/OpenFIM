import itertools
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import optree
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fim import project_path
from fim.data.datasets import PaddedFIMSDEDataset
from fim.models.blocks import AModel
from fim.pipelines.sde_sampling_from_model import fimsde_sample_paths_on_masked_grid
from fim.utils.evaluation_sde import (
    DataLoaderMap,
    EvaluationConfig,
    ModelEvaluation,
    ModelMap,
    NumpyEncoder,
    dataloader_map_from_dict,
    get_data_from_model_evaluation,
    load_evaluations,
    model_map_from_dict,
    save_evaluations,
    save_fig,
)


def evaluate_lorenz_model(model: AModel, dataloader: DataLoader, device: Optional[str] = None):
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    dataset: dict = next(iter(dataloader))
    dataset = optree.tree_map(lambda x: x.to(device), dataset, namespace="fimsde")

    # # evaluate vector field on some location grid
    # if "locations" in dataset.keys():
    #     estimated_concepts = model(dataset, training=False, return_losses=False)
    #     estimated_concepts = optree.tree_map(lambda x: x.to("cpu"), estimated_concepts, namespace="fimsde")
    #     results.update({"estimated_concepts": estimated_concepts})

    inference_grid = dataset["inference_grid"]  # [1, L, T, 1]
    initial_states = dataset["initial_states"]  # [1, L, 3]

    sample_paths, sample_paths_grid = fimsde_sample_paths_on_masked_grid(
        model,
        dataset,
        initial_states=initial_states,
        grid=inference_grid,
        mask=torch.ones_like(inference_grid),
        solver_granularity=10,
    )  # [1 num_initial_states, T, 3]

    results = {"synthetic_path": sample_paths}

    results = optree.tree_map(lambda x: x.to("cpu"), results)

    return results


def run_lorenz_evaluation(
    to_evaluate: list[ModelEvaluation],
    model_map: ModelMap,
    dataloader_map: DataLoaderMap,
    device: Optional[str] = None,
) -> list[ModelEvaluation]:
    """
    Evaluate model on mocap data.

    Args:
        evaluations (list[ModelEvaluation]): List of evaluations to run, specified by their model_id and dataloader_id
        model / dataloader map: Returning required dataloaders

    Return:
        evaluations (list[ModelEvaluation]): Input evaluations.
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    evaluations_with_results: list[ModelEvaluation] = []

    for evaluation in (pbar := tqdm(to_evaluate, total=len(to_evaluate), leave=False)):
        pbar.set_description(f"Model: {str(evaluation.model_id)}. Overall progress")

        model: AModel = model_map[evaluation.model_id]().to(torch.float)
        dataloader: DataLoader = dataloader_map[evaluation.dataloader_id]()

        evaluation.results = evaluate_lorenz_model(model, dataloader, device)
        evaluations_with_results.append(evaluation)

    return evaluations_with_results


def example_lorenz_paths_figure(obs_values: torch.Tensor):
    """
    obs_values: [1, B, T, 3]
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300, subplot_kw={"projection": "3d"})

    for i in range(obs_values.shape[1]):
        ax.plot(
            obs_values[0, i, :, 0],
            obs_values[0, i, :, 1],
            obs_values[0, i, :, 2],
            color="black",
            linewidth=0.5,
            label="Observed Paths" if i == 0 else None,
        )

    fig.legend()

    return fig


def get_lorenz_data(lorenz_data_pth: str) -> dict:
    """
    Unpacks and pre-processes Lorenz data from NeuralSDE.

    Args:
        lorenz_data_pth (str): Path to pth with data of NeuralSDE Lorenz system.

    Returns:
        obs_times (torch.Tensor): Observation times. Shape: [1, P, T, 1]
        obs_values (torch.Tensor): Observation values. Shape: [1, P, T, 3]
    """
    lorenz_data: dict = torch.load(lorenz_data_pth, map_location=torch.device("cpu"))

    obs_times = lorenz_data["ts"]  # [T]
    obs_values = lorenz_data["xs"]  # [T, P, 3]

    # preprocess to our convention of [batch, paths, T, D]
    T, P, _ = obs_values.shape
    obs_times = torch.broadcast_to(obs_times.reshape(1, 1, T, 1), (1, P, T, 1))  # [1, P, T, 1]
    obs_values = torch.swapaxes(obs_values, 0, 1).reshape(1, P, T, 3)

    return {"obs_times": obs_times, "obs_values": obs_values}


def get_lorenz_dataloaders(
    constant_diffusion_training_data_json: str,
    linear_diffusion_training_data_json: str,
    inference_data_json: str,
    num_context_paths: list[int],
):
    """ """
    training_data: dict = {
        "constant_diffusion": get_lorenz_data(constant_diffusion_training_data_json),
        "linear_diffusion": get_lorenz_data(linear_diffusion_training_data_json),
    }

    inference_data: list[dict] = json.load(open(inference_data_json, "r"))
    inference_data: list[dict] = [
        {key: torch.tensor(value) if isinstance(value, list) else value for key, value in d.items()} for d in inference_data
    ]

    def _get_dataloader(train_data: dict, inf_data: dict, num_paths: int):
        obs_times = train_data["obs_times"]  # [1, P, T, 1]
        obs_values = train_data["obs_values"]  # [1, P, T, 3]

        obs_times = obs_times[:, :num_paths]
        obs_values = obs_values[:, :num_paths]

        initial_states = inf_data["initial_states"]  # [L, 3]
        L = initial_states.shape[0]

        inference_grid = inf_data["grid"]  # [M]
        M = inference_grid.shape[0]

        initial_states = initial_states.reshape(1, L, 3)  # [1, L, 3]
        inference_grid = torch.broadcast_to(inference_grid.reshape(1, 1, M, 1), (1, L, M, 1))  # [1, L, M, 1]

        dataset = PaddedFIMSDEDataset(
            data={
                "obs_times": obs_times,
                "obs_values": obs_values,
                "inference_grid": inference_grid,
                "initial_states": initial_states,
            },
            batch_size=1,
            max_dim=3,
            shuffle_locations=False,
            shuffle_paths=False,
            shuffle_elements=False,
        )

        dataloader = DataLoader(
            dataset,
            drop_last=False,
            shuffle=False,
            batch_size=None,  # handled by iterable dataset
            num_workers=0,
        )

        return dataloader

    dataloaders = {}
    for inf_data in inference_data:
        for train_data_label, train_data in training_data.items():
            for num_paths in num_context_paths:
                dataloader = _get_dataloader(train_data, inf_data, num_paths)
                dataloaders.update({(inf_data["initial_state_label"], train_data_label, num_paths): dataloader})

    return dataloaders


def model_sample_paths_figure(model_evaluation: ModelEvaluation, evaluation_config: EvaluationConfig):
    dataset: dict = get_data_from_model_evaluation(model_evaluation, evaluation_config)

    obs_values = dataset["obs_values"].squeeze()  # [P, T, 3]

    model_sample_paths = model_evaluation.results["synthetic_path"].squeeze().squeeze()  # [L, T, 3]

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300, subplot_kw={"projection": "3d"})

    for i in range(obs_values.shape[0]):
        ax.plot(
            obs_values[i, :, 0],
            obs_values[i, :, 1],
            obs_values[i, :, 2],
            color="black",
            linewidth=0.5,
            label="Observed Paths" if i == 0 else None,
        )

    for i in range(model_sample_paths.shape[0]):
        ax.plot(
            model_sample_paths[i, :, 0],
            model_sample_paths[i, :, 1],
            model_sample_paths[i, :, 2],
            color="red",
            linewidth=0.5,
            label="Model" if i == 0 else None,
        )

    fig.legend()

    # save
    save_dir: Path = evaluation_dir / "model_sample_paths_figure" / model_evaluation.dataloader_id / model_evaluation.model_id
    save_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"model_{model_evaluation.model_id}"
    save_fig(fig, save_dir, file_name)

    plt.close(fig)


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    dataset_descr = "lorenz_system_vf_and_paths_evaluation"

    # How to name experiments
    experiment_descr = "neurips_model_no_finetuning"

    model_dicts = {
        ("fim_model_C_at_139_epochs_no_finetuning", "linear_diffusion"): {
            "checkpoint_dir": "/cephfs_projects/foundation_models/models/FIMSDE/NeurIPS_submission_models/600k_drift_deg_3_diff_deg_2_delta_tau_fixed_linear_attn_softmax_no_extra_normalization_and_fix_in_residual_layer_05-06-2300/checkpoints/",
            "checkpoint_name": "epoch-139",
        },
        ("fim_model_C_at_139_epochs_no_finetuning", "constant_diffusion"): {
            "checkpoint_dir": "/cephfs_projects/foundation_models/models/FIMSDE/NeurIPS_submission_models/600k_drift_deg_3_diff_deg_2_delta_tau_fixed_linear_attn_softmax_no_extra_normalization_and_fix_in_residual_layer_05-06-2300/checkpoints/",
            "checkpoint_name": "epoch-139",
        },
    }

    constant_diffusion_training_data_pth = "/cephfs_projects/foundation_models/data/SDE/external_evaluations_and_data/20250512_lorenz_data_from_neural_sde_github/20250514_lorenz_data_constant_diffusion_from_neural_sde_paper_setup_generated_with_adapted_github_code.pth"
    linear_diffusion_training_data_pth = "/cephfs_projects/foundation_models/data/SDE/external_evaluations_and_data/20250512_lorenz_data_from_neural_sde_github/20250513_lorenz_data_linear_diffusion_from_neural_sde_github_setup_40_path_length.pth"
    inference_data_json = "/cephfs_projects/foundation_models/data/SDE/external_evaluations_and_data/20250512_lorenz_data_from_neural_sde_github/20250514221957_lorenz_system_mmd_reference_paths/20250514221957_lorenz_mmd_inference_data.json"

    num_context_paths = [128, 1024]
    # num_context_paths = [128]

    results_to_load = []

    # --------------------------------------------------------------------------------------------------------------------------------- #

    # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
    evaluation_path = Path(project_path) / "evaluations"
    time: str = str(datetime.now().strftime("%m%d%H%M"))
    evaluation_dir: Path = evaluation_path / dataset_descr / (time + "_" + experiment_descr)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    dataloaders = get_lorenz_dataloaders(
        constant_diffusion_training_data_pth,
        linear_diffusion_training_data_pth,
        inference_data_json,
        num_context_paths,
    )

    # Get model_map to load models when they are needed
    model_map = model_map_from_dict(model_dicts)
    dataloader_map = dataloader_map_from_dict(dataloaders)

    # Load previous evaluations that don't need to be evaluated anymore
    loaded_evaluations: list[ModelEvaluation] = load_evaluations(results_to_load)

    # Evaluate models using the data they were trained on
    ids_to_evaluate = []
    for model_id, dataloader_id in itertools.product(model_dicts.keys(), dataloaders.keys()):
        model_train_data_label = model_id[1]
        dataloader_train_data_label = dataloader_id[1]

        if model_train_data_label == dataloader_train_data_label:
            ids_to_evaluate.append((model_id, dataloader_id))

    all_evaluations: list[ModelEvaluation] = [ModelEvaluation(model_id, dataloader_id) for model_id, dataloader_id in ids_to_evaluate]
    to_evaluate: list[ModelEvaluation] = [evaluation for evaluation in all_evaluations if evaluation not in loaded_evaluations]

    # Create, run and save EvaluationConfig
    evaluated: list[ModelEvaluation] = run_lorenz_evaluation(to_evaluate, model_map, dataloader_map)
    all_evaluations: list[ModelEvaluation] = loaded_evaluations + evaluated
    save_evaluations(all_evaluations, evaluation_dir / "model_evaluations")

    # save results in json per model
    for model_id in model_map.keys():
        model_evals = [eval for eval in all_evaluations if eval.model_id == model_id]

        def _check_finite(x):
            if isinstance(x, torch.Tensor):
                assert torch.torch.isfinite(x).all().item()

        model_label, train_data_label = model_id

        for num_paths in num_context_paths:
            model_outputs = []

            # save one json per model, train_data_label, context size
            for eval in model_evals:
                if eval.dataloader_id[2] == num_paths:
                    initial_state_label = eval.dataloader_id[0]

                    model_outputs.append(
                        {
                            "model_label": model_label + "_context_path_num_" + str(num_paths),
                            "initial_state_label": initial_state_label,
                            "train_data_diffusion_label": train_data_label.split("_")[0],  # remove "_diffusion"
                            "synthetic_path": eval.results["synthetic_path"].squeeze(),  # [P, T, 3]
                        }
                    )

            # save outputs as json
            _check_finite(model_outputs)
            model_outputs = optree.tree_map(lambda x: x.detach().to("cpu").numpy() if isinstance(x, torch.Tensor) else x, model_outputs)
            json_data = json.dumps(model_outputs, cls=NumpyEncoder)

            # Write JSON data to a file
            json_dir = evaluation_dir / "model_paths"
            json_dir.mkdir(exist_ok=True, parents=True)

            file: Path = json_dir / (model_label + "_train_data_" + train_data_label + "_num_context_paths_" + str(num_paths) + ".json")
            with open(file, "w") as file:
                file.write(json_data)
