import itertools
from datetime import datetime
from pathlib import Path
from typing import Optional

import optree
import torch
from dataloader_inits.synthetic_test_equations import get_svise_dataloaders_inits
from torch.utils.data import DataLoader
from tqdm import tqdm

from fim import project_path
from fim.models.blocks import AModel
from fim.pipelines.sde_sampling_from_model import fimsde_sample_paths
from fim.utils.evaluation_sde import (
    DataLoaderMap,
    EvaluationConfig,
    ModelEvaluation,
    ModelMap,
    dataloader_map_from_dict,
    model_map_from_dict,
    save_evaluations,
)


def evaluate_model(model: AModel, dataloader: DataLoader, num_sample_paths: int, device: Optional[str] = None):
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    results = {}

    dataset: dict = next(iter(dataloader))
    dataset = optree.tree_map(lambda x: x.to(device), dataset, namespace="fimsde")

    # # evaluate vector field on some location grid
    # if "locations" in dataset.keys():
    #     estimated_concepts = model(dataset, training=False, return_losses=False)
    #     estimated_concepts = optree.tree_map(lambda x: x.to("cpu"), estimated_concepts, namespace="fimsde")
    #     results.update({"estimated_concepts": estimated_concepts})

    grid = dataset["obs_times"]  # [..., 43, T, 1]
    initial_states = dataset["obs_values"][:, :, 0]

    # Repeat grid and initial states for num_sample_paths on dim 1
    grid = grid.repeat(1, num_sample_paths, 1, 1)[:, :num_sample_paths]
    initial_states = initial_states.repeat(1, num_sample_paths, 1)[:, :num_sample_paths]

    sample_paths, sample_paths_grid = fimsde_sample_paths(
        model,
        dataset,
        grid=grid,
        solver_granularity=20,
        num_paths=num_sample_paths,
        initial_states=initial_states,
    )  # [..., 43 * num_sample_paths, D]

    results.update({"sample_paths": sample_paths, "sample_paths_grid": sample_paths_grid})

    return results


def run_evaluations(
    to_evaluate: list[ModelEvaluation],
    model_map: ModelMap,
    dataloader_map: DataLoaderMap,
    num_sample_paths: int,
    device: Optional[str] = None,
) -> list[ModelEvaluation]:
    """
    Evaluate model on mocap data.

    Args:
        evaluations (list[ModelEvaluation]): List of evaluations to run, specified by their model_id and dataloader_id
        model / dataloader map: Returning required dataloaders
        num_sample_paths (int): number of model sample paths per trajectory

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

        evaluation.results = evaluate_model(model, dataloader, num_sample_paths, device)
        evaluations_with_results.append(evaluation)

    return evaluations_with_results


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    dataset_descr = "nature_datasets"

    BATCH_SIZE = 128
    NUM_WORKERS = 8

    # How to name experiments
    experiment_descr = "develop"

    model_dicts = {
        "11M_params": "/cephfs_projects/foundation_models/models/FIMSDE/600k_drift_deg_3_diff_deg_2_embed_size_256_cont_with_grad_clip_lr_1e-5_weight_decay_1e-4_01-19-2228/checkpoints",
    }

    models_display_ids = {
        "11M_params": "11M Parameters",
    }

    results_to_load: list[str] = [
        # "/home/seifner/repos/FIM/evaluations/motion_capture/01221310_develop/model_evaluations",
    ]

    num_sample_paths = 1000
    # --------------------------------------------------------------------------------------------------------------------------------- #

    # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
    evaluation_path = Path(project_path) / "evaluations"
    time: str = str(datetime.now().strftime("%m%d%H%M"))
    evaluation_dir: Path = evaluation_path / dataset_descr / (time + "_" + experiment_descr)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # Get dataloaders inits and their display ids (for ModelEvaluation)
    dataloader_dicts, dataloader_display_ids = get_svise_dataloaders_inits(
        Path("/cephfs_projects/foundation_models/data/SDE/test/20250124_svise/ode"), batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    # Get model_map to load models when they are needed
    model_map = model_map_from_dict(model_dicts)
    dataloader_map = dataloader_map_from_dict(dataloader_dicts)

    # Evaluate all models on all datasets
    all_evaluations: list[ModelEvaluation] = [
        ModelEvaluation(model_id, dataloader_id)
        for model_id, dataloader_id in itertools.product(model_dicts.keys(), dataloader_dicts.keys())
    ]
    to_evaluate: list[ModelEvaluation] = list(all_evaluations)

    # Create, run and save EvaluationConfig
    evaluated: list[ModelEvaluation] = run_evaluations(to_evaluate, model_map, dataloader_map, num_sample_paths)
    save_evaluations(evaluated, evaluation_dir / "model_evaluations")

    evaluation_config = EvaluationConfig(model_map, dataloader_map, evaluation_dir, models_display_ids, dataloader_display_ids)
