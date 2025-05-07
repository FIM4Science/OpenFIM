import itertools
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import optree
import pandas as pd
import torch
from motion_capture_43 import figure_grid_motion_capture_latent
from torch import Tensor
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
    dataloader_map_from_dict,
    get_data_from_model_evaluation,
    load_evaluations,
    model_map_from_dict,
    save_evaluations,
    save_fig,
    save_table,
)


def get_high_dim_reconstr_mocap_35(paths: Tensor, eigenvalues: Tensor, eigenvectors: Tensor):
    """
    Project latent paths to high dimensional space.

    Args:
        paths (Tensor): Shape: ([1, B, num_paths, 300, 3]
        eigenvectors (Tensor): Shape: ([1, 50, 50])
        eigenvalues (Tensor): Shape: ([1, 50])

    Returns:
        high_dim_reconst (Tensor): Shape: ([1, B, num_paths, 300, 50]
    """
    assert paths.ndim == 5

    _, _, num_paths, T, _ = paths.shape

    eigenvalues = eigenvalues[:, None, None, None, :3]
    eigenvectors = eigenvectors[:, None, None, None, :, :3].expand(-1, -1, num_paths, T, -1, -1)

    # reconstruct from first 3 pca dimensions
    high_dim_reconst = torch.matmul((paths * torch.sqrt(eigenvalues))[..., None, :], torch.transpose(eigenvectors, -2, -1)).squeeze(-2)

    assert high_dim_reconst.ndim == 5
    return high_dim_reconst


def evaluate_mocap_35_model(model: AModel, dataloader: DataLoader, num_sample_paths: int, device: Optional[str] = None):
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    results = {}

    dataset: dict = next(iter(dataloader))
    dataset = optree.tree_map(lambda x: x.to(device), dataset, namespace="fimsde")

    # evaluate vector field on some location grid
    if "locations" in dataset.keys():
        estimated_concepts = model(dataset, training=False, return_losses=False)
        estimated_concepts = optree.tree_map(lambda x: x.to("cpu"), estimated_concepts, namespace="fimsde")
        results.update({"estimated_concepts": estimated_concepts})

    # sample multiple paths starting from last observed value on test sets
    # "inference_mask": includes last context point
    last_context_value = dataset["last_context_value"]  # [1, 4, 3]
    inference_grid = dataset["paths_inference_grid"]  # [1, 4, 298-300, 1]

    # sample multiple paths per trajectory
    last_context_value = torch.repeat_interleave(last_context_value, dim=-2, repeats=num_sample_paths)
    inference_grid = torch.repeat_interleave(inference_grid, dim=-3, repeats=num_sample_paths)

    sample_paths, sample_paths_grid = fimsde_sample_paths_on_masked_grid(
        model,
        dataset,
        initial_states=last_context_value,
        grid=inference_grid,
        mask=torch.ones_like(inference_grid),
        solver_granularity=5,
    )  # [..., 4 * num_sample_paths, 298-300, 3]

    # Return paths per trajectory, [1, 4, num_sample_paths, 298-300, 3]
    sample_paths = torch.stack(torch.split(sample_paths, num_sample_paths, dim=-3), dim=-4)
    sample_paths_grid = torch.stack(torch.split(sample_paths_grid, num_sample_paths, dim=-3), dim=-4)

    # return mean of sample paths
    mean_sample_paths = torch.mean(sample_paths, dim=-3, keepdim=True)
    mean_sample_paths_grid = sample_paths_grid[..., 0, :, :]  # sample path grids will be the same

    # add high dimensional reconstruction
    eigenvectors = dataset["eigenvectors"]
    eigenvalues = dataset["eigenvalues"]
    high_dim_from_samples = get_high_dim_reconstr_mocap_35(sample_paths, eigenvalues, eigenvectors)  # [1, 4, num_sample_paths, 298-300, 50]
    high_dim_from_mean = get_high_dim_reconstr_mocap_35(mean_sample_paths, eigenvalues, eigenvectors)  # [1, 4, 1, 298-300, 50]

    # add MSE in high dimensional space at forecasting
    reference_paths = dataset["paths_metrics_high_dim_reference_values"]  # to compute metrics gainst [1, 4, 298-300, 50]
    metric_reference_mask = dataset["paths_metrics_mask"].bool()  # True where metric should be computed [1, 4, 298-300, 1]

    def _mse_at_forecasting(targets: Tensor, prediction: Tensor, mask: Tensor):
        """
        Return MSE (across time and all 50 dimensions, as Yıldız 2019) at (mask == True) points between targets and predictions.

        Args:
            targets (Tensor): Shape: [1, 4, 298-300, 50]
            prediction (Tensor): Shape: [1, 4, P, 298-300, 50]
            mask (Tensor): MSE computed at True in time dim. Shape: [1, 4, 298-300, 50]

        returns:
            mse_per_path (Tensor): Shape: [..., 4, P]

        """
        targets = torch.broadcast_to(targets[..., None, :, :], prediction.shape)
        se = torch.mean((targets - prediction) ** 2, dim=-1)

        mask = torch.broadcast_to(mask[..., None, :, 0], se.shape)
        mse = torch.nanmean(torch.where(mask.squeeze(-1), se, torch.nan), dim=-1)  # [..., 4, P]
        return mse

    mse_from_samples = _mse_at_forecasting(reference_paths, high_dim_from_samples, metric_reference_mask)
    mse_from_mean = _mse_at_forecasting(reference_paths, high_dim_from_mean, metric_reference_mask)
    mse_from_gt_pca = _mse_at_forecasting(
        reference_paths, dataset["reconst_high_dim_from_pca_test"][..., None, :, :], metric_reference_mask
    )

    mse_from_samples_mean = mse_from_samples.mean()
    mse_from_samples_std = mse_from_samples.std()
    mse_from_mean_mean = mse_from_mean.mean()
    mse_from_mean_std = mse_from_mean.std()
    mse_from_gt_pca_mean = mse_from_gt_pca.mean()
    mse_from_gt_pca_std = mse_from_gt_pca.std()

    results.update(
        {
            "sample_paths": sample_paths,
            "sample_paths_grid": sample_paths_grid,
            "high_dim_reconst_per_path": high_dim_from_samples,
            "high_dim_reconst_mean_path": high_dim_from_mean,
            "mean_sample_paths": mean_sample_paths,
            "mean_sample_paths_grid": mean_sample_paths_grid,
            "mse_from_samples_mean": mse_from_samples_mean,
            "mse_from_samples_std": mse_from_samples_std,
            "mse_from_mean_mean": mse_from_mean_mean,
            "mse_from_mean_std": mse_from_mean_std,
            "mse_from_gt_pca_mean": mse_from_gt_pca_mean,
            "mse_from_gt_pca_std": mse_from_gt_pca_std,
        }
    )

    results = optree.tree_map(lambda x: x.to("cpu"), results)

    return results


def run_mocap_35_evaluations(
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

        evaluation.results = evaluate_mocap_35_model(model, dataloader, num_sample_paths, device)
        evaluations_with_results.append(evaluation)

    return evaluations_with_results


def table_forecasting_mse(model_evaluations: list[ModelEvaluation], evaluation_config: EvaluationConfig):
    """
    Table with all MSE results from all models, and baselines.
    """
    precision = 2
    rows = []

    # Our models
    for evaluation in model_evaluations:
        mse_from_samples_mean = round(evaluation.results["mse_from_samples_mean"].item(), precision)
        mse_from_samples_std = round(evaluation.results["mse_from_samples_std"].item(), precision)
        mse_from_mean_mean = round(evaluation.results["mse_from_mean_mean"].item(), precision)
        mse_from_mean_std = round(evaluation.results["mse_from_mean_std"].item(), precision)

        mse_from_samples = str(mse_from_samples_mean) + r"$\pm$" + str(mse_from_samples_std)
        mse_from_mean = str(mse_from_mean_mean) + r"$\pm$" + str(mse_from_mean_std)

        rows.append(
            {
                "model": evaluation_config.model_display_id_map[evaluation.model_id],
                "data": evaluation_config.dataloader_display_id_map[evaluation.dataloader_id],
                "MSE from mean path": mse_from_mean,
                "MSE from individual paths": mse_from_samples,
            }
        )

    # baselines
    rows = rows + [
        {"model": "DTSBN-S", "data": "-", "MSE from mean path": r"34.86 $\pm$ 0.02 ", "MSE from individual paths": "-"},
        {"model": "NPODE", "data": "-", "MSE from mean path": r"22.96 ", "MSE from individual paths": "-"},
        {"model": "NEURALODE", "data": "-", "MSE from mean path": r"22.49 $\pm$ 0.88 ", "MSE from individual paths": "-"},
        {"model": "ODE2VAE", "data": "-", "MSE from mean path": r"10.06 $\pm$ 1.4 ", "MSE from individual paths": "-"},
        {"model": "ODE2VAE-KL", "data": "-", "MSE from mean path": r"8.09 $\pm$ 1.95 ", "MSE from individual paths": "-"},
        {"model": "LATENTSDE", "data": "-", "MSE from mean path": r"4.03 $\pm$ 0.20", "MSE from individual paths": "-"},
        {"model": "ARCTA", "data": "-", "MSE from mean path": r"7.62 $\pm$ 0.93", "MSE from individual paths": "-"},
        {"model": "SDE MATCHING", "data": "-", "MSE from mean path": r"4.50 $\pm$ 0.32", "MSE from individual paths": "-"},
    ]

    # sanity check for MSE from ground truth pca projection back to high dim.
    mse_from_gt_pca_mean = round(model_evaluations[0].results["mse_from_gt_pca_mean"].item(), precision)
    mse_from_gt_pca_std = round(model_evaluations[0].results["mse_from_gt_pca_std"].item(), precision)
    mse_from_gt_pca = str(mse_from_gt_pca_mean) + r"$\pm$" + str(mse_from_gt_pca_std)
    rows.append({"model": "From PCA directly", "data": "-", "MSE from mean path": mse_from_gt_pca, "MSE from individual paths": "-"})

    cols = optree.tree_map(lambda *x: x, *rows)

    df = pd.DataFrame.from_dict(cols)
    df = df.sort_values(["data", "model"])
    df.set_index(["model", "data"])

    save_table(df, evaluation_config.save_dir, "mse_all_models")


def get_mocap_35_dataloaders_inits(mocap_dir: Optional[Path] = None) -> tuple[dict]:
    """
    Return DataLoaderInitializer for mocap 35 forecasting task.

    Args:
        mocap_dir (Path): Absolute path to dir with subdirs of preprocessed data.

    Returns:
        dataloder_dict, dataloader_map
    """
    files_to_load = {
        "obs_times": "obs_grid.h5",
        "obs_values": "obs_values.h5",
        "obs_grid_test": "obs_grid_test.h5",
        "obs_values_test": "obs_values_test.h5",
        "last_context_value": "last_context_value_test.h5",
        "paths_inference_grid": "paths_inference_grid.h5",
        "paths_metrics_high_dim_reference_values": "paths_metrics_high_dim_reference_values.h5",
        "paths_metrics_mask": "paths_metrics_mask.h5",
        "reconst_high_dim_from_pca_test": "reconst_high_dim_from_pca_test.h5",
        "eigenvectors": "eigenvectors.h5",
        "eigenvalues": "eigenvalues.h5",
        "forecasting_mask_test": "forecasting_mask_test.h5",
    }

    def _get_dataloader(data_subdir: str):
        path = mocap_dir / data_subdir

        dataset = PaddedFIMSDEDataset(
            data_dirs=path,
            batch_size=19,
            files_to_load=files_to_load,
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

    dataloader_dict = {f"mocap_35_last_test_context_index_{index}": _get_dataloader(f"last_context_index_{index}") for index in range(1)}
    dataloader_display_ids = {f"mocap_35_last_test_context_index_{index}": f"Solve from text index {index}" for index in range(1)}

    return dataloader_dict, dataloader_display_ids


def latent_forecasting_grid(model_evaluation: ModelEvaluation, evaluation_config: EvaluationConfig):
    """
    Plot forecasting sample paths of one model for multiple mocap trajectories

    Args:
        model_evaluation (ModelEvaluation): Results to plot.
        evaluation_config (EvaluationConfig): Providing access to dataloader and model.
    """

    dataset: dict = get_data_from_model_evaluation(model_evaluation, evaluation_config)

    # extract data
    obs_times = dataset["obs_grid_test"].squeeze()  # [4, 300]
    obs_values = dataset["obs_values_test"].squeeze()  # [4, 300, 3]
    forecasting_mask_test = dataset["forecasting_mask_test"].squeeze().bool()  # [4, 300]

    model_sample_paths_grid = model_evaluation.results["sample_paths_grid"].squeeze()  # [4, num_sample_paths, 298-300]
    model_sample_paths = model_evaluation.results["sample_paths"].squeeze()  # [4, num_sample_paths, 298-300, 3]

    mean_model_sample_paths_grid = model_evaluation.results["mean_sample_paths_grid"].squeeze()  # [4, num_sample_paths,298-300]
    mean_model_sample_paths = model_evaluation.results["mean_sample_paths"].squeeze()  # [4, num_sample_paths, 298-300, 3]

    # plot
    fig_grid = figure_grid_motion_capture_latent(
        obs_times,
        obs_values,
        np.logical_not(forecasting_mask_test).bool(),
        forecasting_mask_test,
        model_sample_paths_grid,
        model_sample_paths,
        mean_model_sample_paths_grid,
        mean_model_sample_paths,
        num_plot_traj=4,
        figsize=(10, 10),
    )

    # save
    save_dir: Path = evaluation_dir / "figure_grid_latent_dimension" / model_evaluation.dataloader_id / model_evaluation.model_id
    save_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"model_{model_evaluation.model_id}"
    save_fig(fig_grid, save_dir, file_name)

    plt.close(fig_grid)
    plt.close(fig_grid)


def latent_forecasting_grid_3D(model_evaluation: ModelEvaluation, evaluation_config: EvaluationConfig):
    """
    Plot forecasting sample paths of one model for multiple mocap trajectories

    Args:
        model_evaluation (ModelEvaluation): Results to plot.
        evaluation_config (EvaluationConfig): Providing access to dataloader and model.
    """

    dataset: dict = get_data_from_model_evaluation(model_evaluation, evaluation_config)

    # extract data
    obs_values = dataset["obs_values_test"].squeeze()  # [4, T, 3]
    mean_model_sample_paths = model_evaluation.results["mean_sample_paths"].squeeze()  # [4, num_sample_paths, 300, 3]

    # plot
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), subplot_kw={"projection": "3d"})

    for i in range(4):
        axs[i].plot(obs_values[i, :, 0], obs_values[i, :, 1], obs_values[i, :, 2], color="black")
        axs[i].plot(mean_model_sample_paths[i, :, 0], mean_model_sample_paths[i, :, 1], mean_model_sample_paths[i, :, 2], color="#CC79A7")

    # save
    save_dir: Path = evaluation_dir / "3D_latent_dimension" / model_evaluation.dataloader_id / model_evaluation.model_id
    save_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"model_{model_evaluation.model_id}"
    save_fig(fig, save_dir, file_name)

    plt.close(fig)
    plt.close(fig)


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    dataset_descr = "motion_capture_35"

    # How to name experiments
    # experiment_descr = "train_or_finetune_on_test_split"
    # experiment_descr = "train_or_finetune_on_train_split"
    experiment_descr = "train_or_finetune_on_train_split_with_likelihood"

    # model_dicts, models_display_ids = get_model_dicts_600k_fixed_linear_attn()

    # model_dicts = {
    #     "trained_test_split_lr_1e-4_epoch_10": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/mocap_35_train_on_test_split_lr_e-4_05-11-2324/checkpoints/",
    #         "checkpoint_name": "epoch-10",
    #     },
    #     "trained_test_split_lr_1e-4_epoch_50": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/mocap_35_train_on_test_split_lr_e-4_05-11-2324/checkpoints/",
    #         "checkpoint_name": "epoch-50",
    #     },
    #     "trained_test_split_lr_1e-4_epoch_200": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/mocap_35_train_on_test_split_lr_e-4_05-11-2324/checkpoints/",
    #         "checkpoint_name": "epoch-200",
    #     },
    #     "trained_test_split_lr_1e-4_epoch_500": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/mocap_35_train_on_test_split_lr_e-4_05-11-2324/checkpoints/",
    #         "checkpoint_name": "epoch-500",
    #     },
    #     "finetune_05-03-2033_test_split_lr_1e-5_epoch_10": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/mocap_35_finetune_05-03-2033_on_test_split_lr_e-5_05-11-2337/checkpoints/",
    #         "checkpoint_name": "epoch-110",
    #     },
    #     "finetune_05-03-2033_test_split_lr_1e-5_epoch_50": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/mocap_35_finetune_05-03-2033_on_test_split_lr_e-5_05-11-2337/checkpoints/",
    #         "checkpoint_name": "epoch-160",
    #     },
    #     "finetune_05-03-2033_test_split_lr_1e-5_epoch_200": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/mocap_35_finetune_05-03-2033_on_test_split_lr_e-5_05-11-2337/checkpoints/",
    #         "checkpoint_name": "epoch-300",
    #     },
    #     "finetune_05-03-2033_test_split_lr_1e-5_epoch_500": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/mocap_35_finetune_05-03-2033_on_test_split_lr_e-5_05-11-2337/checkpoints/",
    #         "checkpoint_name": "epoch-599",
    #     },
    #     "finetune_05-03-2033_test_split_lr_1e-3_epoch_10": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/mocap_35_finetune_05-03-2033_on_test_split_lr_e-3_05-12-0000/checkpoints/",
    #         "checkpoint_name": "epoch-110",
    #     },
    #     "finetune_05-03-2033_test_split_lr_1e-3_epoch_50": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/mocap_35_finetune_05-03-2033_on_test_split_lr_e-3_05-12-0000/checkpoints/",
    #         "checkpoint_name": "epoch-160",
    #     },
    #     "finetune_05-03-2033_test_split_lr_1e-3_epoch_200": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/mocap_35_finetune_05-03-2033_on_test_split_lr_e-3_05-12-0000/checkpoints/",
    #         "checkpoint_name": "epoch-300",
    #     },
    #     "finetune_05-03-2033_test_split_lr_1e-3_epoch_500": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/mocap_35_finetune_05-03-2033_on_test_split_lr_e-3_05-12-0000/checkpoints/",
    #         "checkpoint_name": "epoch-599",
    #     },
    #     "finetune_05-03-2033_test_split_detach_diffusion_epoch_10": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/mocap_35_finetune_05-03-2033_detach_diffusion_lr_e-4_05-12-1035/checkpoints/",
    #         "checkpoint_name": "epoch-110",
    #     },
    #     "finetune_05-03-2033_test_split_detach_diffusion_epoch_50": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/mocap_35_finetune_05-03-2033_detach_diffusion_lr_e-4_05-12-1035/checkpoints/",
    #         "checkpoint_name": "epoch-160",
    #     },
    #     "finetune_05-03-2033_test_split_detach_diffusion_epoch_200": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/mocap_35_finetune_05-03-2033_detach_diffusion_lr_e-4_05-12-1035/checkpoints/",
    #         "checkpoint_name": "epoch-300",
    #     },
    #     "finetune_05-03-2033_test_split_detach_diffusion_epoch_500": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/mocap_35_finetune_05-03-2033_detach_diffusion_lr_e-4_05-12-1035/checkpoints/",
    #         "checkpoint_name": "epoch-599",
    #     },
    #     "finetune_05-06-2300_test_split_detach_diffusion_epoch_10": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/mocap_35_finetune_05-06-2300_detach_diffusion_lr_e-4_05-12-1046/checkpoints/",
    #         "checkpoint_name": "epoch-110",
    #     },
    #     "finetune_05-06-2300_test_split_detach_diffusion_epoch_50": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/mocap_35_finetune_05-06-2300_detach_diffusion_lr_e-4_05-12-1046/checkpoints/",
    #         "checkpoint_name": "epoch-160",
    #     },
    #     "finetune_05-06-2300_test_split_detach_diffusion_epoch_200": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/mocap_35_finetune_05-06-2300_detach_diffusion_lr_e-4_05-12-1046/checkpoints/",
    #         "checkpoint_name": "epoch-300",
    #     },
    #     "finetune_05-06-2300_test_split_detach_diffusion_epoch_500": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/mocap_35_finetune_05-06-2300_detach_diffusion_lr_e-4_05-12-1046/checkpoints/",
    #         "checkpoint_name": "epoch-599",
    #     },
    #     "finetune_05-06-2300_test_split_on_likelihood_epoch_10": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/mocap_35_finetune_05-03-2033_on_likelihood_lr_e-4_05-12-1109/checkpoints/",
    #         "checkpoint_name": "epoch-110",
    #     },
    #     "finetune_05-06-2300_test_split_on_likelihood_epoch_50": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/mocap_35_finetune_05-03-2033_on_likelihood_lr_e-4_05-12-1109/checkpoints/",
    #         "checkpoint_name": "epoch-160",
    #     },
    #     "finetune_05-06-2300_test_split_on_likelihood_epoch_200": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/mocap_35_finetune_05-03-2033_on_likelihood_lr_e-4_05-12-1109/checkpoints/",
    #         "checkpoint_name": "epoch-300",
    #     },
    #     "finetune_05-06-2300_test_split_on_likelihood_epoch_500": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/mocap_35_finetune_05-03-2033_on_likelihood_lr_e-4_05-12-1109/checkpoints/",
    #         "checkpoint_name": "epoch-599",
    #     },
    # }
    #
    # models_display_ids = {
    #     "trained_test_split_lr_1e-4_epoch_10": "Train on test split, lr 1e-4, epoch 010",
    #     "trained_test_split_lr_1e-4_epoch_50": "Train on test split, lr 1e-4, epoch 050",
    #     "trained_test_split_lr_1e-4_epoch_200": "Train on test split, lr 1e-4, epoch 200",
    #     "trained_test_split_lr_1e-4_epoch_500": "Train on test split, lr 1e-4, epoch 500",
    #     "finetune_05-03-2033_test_split_lr_1e-5_epoch_10": "Finetune FIM 05-03-2033 on test split, lr 1e-5, epoch 010",
    #     "finetune_05-03-2033_test_split_lr_1e-5_epoch_50": "Finetune FIM 05-03-2033 on test split, lr 1e-5, epoch 050",
    #     "finetune_05-03-2033_test_split_lr_1e-5_epoch_200": "Finetune FIM 05-03-2033 on test split, lr 1e-5, epoch 200",
    #     "finetune_05-03-2033_test_split_lr_1e-5_epoch_500": "Finetune FIM 05-03-2033 on test split, lr 1e-5, epoch 500",
    #     "finetune_05-03-2033_test_split_lr_1e-3_epoch_10": "Finetune FIM 05-03-2033 on test split, lr 1e-3, epoch 010",
    #     "finetune_05-03-2033_test_split_lr_1e-3_epoch_50": "Finetune FIM 05-03-2033 on test split, lr 1e-3, epoch 050",
    #     "finetune_05-03-2033_test_split_lr_1e-3_epoch_200": "Finetune FIM 05-03-2033 on test split, lr 1e-3, epoch 200",
    #     "finetune_05-03-2033_test_split_lr_1e-3_epoch_500": "Finetune FIM 05-03-2033 on test split, lr 1e-3, epoch 500",
    #     "finetune_05-03-2033_test_split_detach_diffusion_epoch_10": "Finetune FIM 05-03-2033 on test split, detach diffusion, lr 1e-4, epoch 010",
    #     "finetune_05-03-2033_test_split_detach_diffusion_epoch_50": "Finetune FIM 05-03-2033 on test split, detach diffusion, lr 1e-4, epoch 050",
    #     "finetune_05-03-2033_test_split_detach_diffusion_epoch_200": "Finetune FIM 05-03-2033 on test split, detach diffusion, lr 1e-4, epoch 200",
    #     "finetune_05-03-2033_test_split_detach_diffusion_epoch_500": "Finetune FIM 05-03-2033 on test split, detach diffusion, lr 1e-4, epoch 500",
    #     "finetune_05-06-2300_test_split_detach_diffusion_epoch_10": "Finetune FIM 05-06-2300 on test split, detach diffusion, lr 1e-4, epoch 010",
    #     "finetune_05-06-2300_test_split_detach_diffusion_epoch_50": "Finetune FIM 05-06-2300 on test split, detach diffusion, lr 1e-4, epoch 050",
    #     "finetune_05-06-2300_test_split_detach_diffusion_epoch_200": "Finetune FIM 05-06-2300 on test split, detach diffusion, lr 1e-4, epoch 200",
    #     "finetune_05-06-2300_test_split_detach_diffusion_epoch_500": "Finetune FIM 05-06-2300 on test split, detach diffusion, lr 1e-4, epoch 500",
    #     "finetune_05-06-2300_test_split_on_likelihood_epoch_10": "Finetune FIM 05-03-2033 on test split, likelihood, lr 1e-4, epoch 010",
    #     "finetune_05-06-2300_test_split_on_likelihood_epoch_50": "Finetune FIM 05-03-2033 on test split, likelihood, lr 1e-4, epoch 050",
    #     "finetune_05-06-2300_test_split_on_likelihood_epoch_200": "Finetune FIM 05-03-2033 on test split, likelihood, lr 1e-4, epoch 200",
    #     "finetune_05-06-2300_test_split_on_likelihood_epoch_500": "Finetune FIM 05-03-2033 on test split, likelihood, lr 1e-4, epoch 500",
    # }

    # model_dicts = {
    #     "trained_on_train_split_lr_1e-4_epoch_10": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/train_on_train_set_lr_1e-4_05-12-1305/checkpoints/",
    #         "checkpoint_name": "epoch-10",
    #     },
    #     "trained_on_train_split_lr_1e-4_epoch_20": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/train_on_train_set_lr_1e-4_05-12-1305/checkpoints/",
    #         "checkpoint_name": "epoch-20",
    #     },
    #     "trained_on_train_split_lr_1e-4_epoch_50": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/train_on_train_set_lr_1e-4_05-12-1305/checkpoints/",
    #         "checkpoint_name": "epoch-50",
    #     },
    #     "trained_on_train_split_lr_1e-4_epoch_200": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/train_on_train_set_lr_1e-4_05-12-1305/checkpoints/",
    #         "checkpoint_name": "epoch-200",
    #     },
    #     "trained_on_train_split_lr_1e-4_epoch_500": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/train_on_train_set_lr_1e-4_05-12-1305/checkpoints/",
    #         "checkpoint_name": "epoch-500",
    #     },
    #     "finetune_on_train_split_lr_1e-4_epoch_10": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/finetune_fim_05-03-2033_on_train_set_lr_1e-4_05-12-1319/checkpoints/",
    #         "checkpoint_name": "epoch-110",
    #     },
    #     "finetune_on_train_split_lr_1e-4_epoch_20": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/finetune_fim_05-03-2033_on_train_set_lr_1e-4_05-12-1319/checkpoints/",
    #         "checkpoint_name": "epoch-120",
    #     },
    #     "finetune_on_train_split_lr_1e-4_epoch_50": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/finetune_fim_05-03-2033_on_train_set_lr_1e-4_05-12-1319/checkpoints/",
    #         "checkpoint_name": "epoch-160",
    #     },
    #     "finetune_on_train_split_lr_1e-4_epoch_200": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/finetune_fim_05-03-2033_on_train_set_lr_1e-4_05-12-1319/checkpoints/",
    #         "checkpoint_name": "epoch-300",
    #     },
    #     "finetune_on_train_split_lr_1e-4_epoch_500": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/finetune_fim_05-03-2033_on_train_set_lr_1e-4_05-12-1319/checkpoints/",
    #         "checkpoint_name": "epoch-599",
    #     },
    #     "finetune_on_train_split_lr_1e-5_epoch_10": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/finetune_fim_05-03-2033_on_train_set_lr_1e-5_05-12-1340/checkpoints/",
    #         "checkpoint_name": "epoch-110",
    #     },
    #     "finetune_on_train_split_lr_1e-5_epoch_20": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/finetune_fim_05-03-2033_on_train_set_lr_1e-5_05-12-1340/checkpoints/",
    #         "checkpoint_name": "epoch-120",
    #     },
    #     "finetune_on_train_split_lr_1e-5_epoch_50": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/finetune_fim_05-03-2033_on_train_set_lr_1e-5_05-12-1340/checkpoints/",
    #         "checkpoint_name": "epoch-160",
    #     },
    #     "finetune_on_train_split_lr_1e-5_epoch_200": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/finetune_fim_05-03-2033_on_train_set_lr_1e-5_05-12-1340/checkpoints/",
    #         "checkpoint_name": "epoch-300",
    #     },
    #     "finetune_on_train_split_lr_1e-5_epoch_500": {
    #         "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/finetune_fim_05-03-2033_on_train_set_lr_1e-5_05-12-1340/checkpoints/",
    #         "checkpoint_name": "epoch-599",
    #     },
    # }
    #
    # models_display_ids = {
    #     "trained_on_train_split_lr_1e-4_epoch_10": "Train on train split, lr 1e-4, epoch 010",
    #     "trained_on_train_split_lr_1e-4_epoch_20": "Train on train split, lr 1e-4, epoch 020",
    #     "trained_on_train_split_lr_1e-4_epoch_50": "Train on train split, lr 1e-4, epoch 050",
    #     "trained_on_train_split_lr_1e-4_epoch_200": "Train on train split, lr 1e-4, epoch 200",
    #     "trained_on_train_split_lr_1e-4_epoch_500": "Train on train split, lr 1e-4, epoch 500",
    #     "finetune_on_train_split_lr_1e-4_epoch_10": "Finetune on train split, lr 1e-4, epoch 010",
    #     "finetune_on_train_split_lr_1e-4_epoch_20": "Finetune on train split, lr 1e-4, epoch 020",
    #     "finetune_on_train_split_lr_1e-4_epoch_50": "Finetune on train split, lr 1e-4, epoch 050",
    #     "finetune_on_train_split_lr_1e-4_epoch_200": "Finetune on train split, lr 1e-4, epoch 200",
    #     "finetune_on_train_split_lr_1e-4_epoch_500": "Finetune on train split, lr 1e-4, epoch 500",
    #     "finetune_on_train_split_lr_1e-5_epoch_10": "Finetune on train split, lr 1e-5, epoch 010",
    #     "finetune_on_train_split_lr_1e-5_epoch_20": "Finetune on train split, lr 1e-5, epoch 020",
    #     "finetune_on_train_split_lr_1e-5_epoch_50": "Finetune on train split, lr 1e-5, epoch 050",
    #     "finetune_on_train_split_lr_1e-5_epoch_200": "Finetune on train split, lr 1e-5, epoch 200",
    #     "finetune_on_train_split_lr_1e-5_epoch_500": "Finetune on train split, lr 1e-5, epoch 500",
    # }

    model_dicts = {
        "trained_on_train_split_log_likelihood_lr_1e-5_epoch_010": {
            "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/train_on_train_set_with_likelihood_lr_1e-5_05-12-1512/checkpoints/",
            "checkpoint_name": "epoch-10",
        },
        "trained_on_train_split_log_likelihood_lr_1e-5_epoch_020": {
            "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/train_on_train_set_with_likelihood_lr_1e-5_05-12-1512/checkpoints/",
            "checkpoint_name": "epoch-20",
        },
        "trained_on_train_split_log_likelihood_lr_1e-5_epoch_050": {
            "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/train_on_train_set_with_likelihood_lr_1e-5_05-12-1512/checkpoints/",
            "checkpoint_name": "epoch-50",
        },
        "trained_on_train_split_log_likelihood_lr_1e-5_epoch_200": {
            "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/train_on_train_set_with_likelihood_lr_1e-5_05-12-1512/checkpoints/",
            "checkpoint_name": "epoch-200",
        },
        "trained_on_train_split_log_likelihood_lr_1e-5_epoch_500": {
            "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/train_on_train_set_with_likelihood_lr_1e-5_05-12-1512/checkpoints/",
            "checkpoint_name": "epoch-500",
        },
        "trained_on_train_split_log_likelihood_lr_1e-4_epoch_010": {
            "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/train_on_train_set_with_likelihood_lr_1e-4_05-12-1523/checkpoints/",
            "checkpoint_name": "epoch-10",
        },
        "trained_on_train_split_log_likelihood_lr_1e-4_epoch_020": {
            "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/train_on_train_set_with_likelihood_lr_1e-4_05-12-1523/checkpoints/",
            "checkpoint_name": "epoch-20",
        },
        "trained_on_train_split_log_likelihood_lr_1e-4_epoch_050": {
            "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/train_on_train_set_with_likelihood_lr_1e-4_05-12-1523/checkpoints/",
            "checkpoint_name": "epoch-50",
        },
        "trained_on_train_split_log_likelihood_lr_1e-4_epoch_200": {
            "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/train_on_train_set_with_likelihood_lr_1e-4_05-12-1523/checkpoints/",
            "checkpoint_name": "epoch-200",
        },
        "trained_on_train_split_log_likelihood_lr_1e-4_epoch_500": {
            "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/train_on_train_set_with_likelihood_lr_1e-4_05-12-1523/checkpoints/",
            "checkpoint_name": "epoch-500",
        },
        "finetuned_fim_05-03-2033_on_train_split_log_likelihood_lr_1e-5_epoch_010": {
            "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/finetune_fim_05-03-2033_on_train_set_with_likelihood_lr_1e-5_05-12-1549/checkpoints/",
            "checkpoint_name": "epoch-110",
        },
        "finetuned_fim_05-03-2033_on_train_split_log_likelihood_lr_1e-5_epoch_020": {
            "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/finetune_fim_05-03-2033_on_train_set_with_likelihood_lr_1e-5_05-12-1549/checkpoints/",
            "checkpoint_name": "epoch-120",
        },
        "finetuned_fim_05-03-2033_on_train_split_log_likelihood_lr_1e-5_epoch_050": {
            "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/finetune_fim_05-03-2033_on_train_set_with_likelihood_lr_1e-5_05-12-1549/checkpoints/",
            "checkpoint_name": "epoch-150",
        },
        "finetuned_fim_05-03-2033_on_train_split_log_likelihood_lr_1e-5_epoch_200": {
            "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/finetune_fim_05-03-2033_on_train_set_with_likelihood_lr_1e-5_05-12-1549/checkpoints/",
            "checkpoint_name": "epoch-200",
        },
        "finetuned_fim_05-03-2033_on_train_split_log_likelihood_lr_1e-5_epoch_500": {
            "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/finetune_fim_05-03-2033_on_train_set_with_likelihood_lr_1e-5_05-12-1549/checkpoints/",
            "checkpoint_name": "epoch-500",
        },
        "finetuned_fim_05-03-2033_on_train_split_log_likelihood_lr_1e-4_epoch_010": {
            "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/finetune_fim_05-03-2033_on_train_set_with_likelihood_lr_1e-4_05-12-1602/checkpoints/",
            "checkpoint_name": "epoch-110",
        },
        "finetuned_fim_05-03-2033_on_train_split_log_likelihood_lr_1e-4_epoch_020": {
            "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/finetune_fim_05-03-2033_on_train_set_with_likelihood_lr_1e-4_05-12-1602/checkpoints/",
            "checkpoint_name": "epoch-120",
        },
        "finetuned_fim_05-03-2033_on_train_split_log_likelihood_lr_1e-4_epoch_050": {
            "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/finetune_fim_05-03-2033_on_train_set_with_likelihood_lr_1e-4_05-12-1602/checkpoints/",
            "checkpoint_name": "epoch-150",
        },
        "finetuned_fim_05-03-2033_on_train_split_log_likelihood_lr_1e-4_epoch_200": {
            "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/finetune_fim_05-03-2033_on_train_set_with_likelihood_lr_1e-4_05-12-1602/checkpoints/",
            "checkpoint_name": "epoch-200",
        },
        "finetuned_fim_05-03-2033_on_train_split_log_likelihood_lr_1e-4_epoch_500": {
            "checkpoint_dir": "/cephfs/users/seifner/repos/FIM/results/finetune_fim_05-03-2033_on_train_set_with_likelihood_lr_1e-4_05-12-1602/checkpoints/",
            "checkpoint_name": "epoch-500",
        },
    }

    models_display_ids = {
        "trained_on_train_split_log_likelihood_lr_1e-5_epoch_010": "Trained on train split, log-likelihood, lr 1e-5, epochs 010",
        "trained_on_train_split_log_likelihood_lr_1e-5_epoch_020": "Trained on train split, log-likelihood, lr 1e-5, epochs 020",
        "trained_on_train_split_log_likelihood_lr_1e-5_epoch_050": "Trained on train split, log-likelihood, lr 1e-5, epochs 050",
        "trained_on_train_split_log_likelihood_lr_1e-5_epoch_200": "Trained on train split, log-likelihood, lr 1e-5, epochs 200",
        "trained_on_train_split_log_likelihood_lr_1e-5_epoch_500": "Trained on train split, log-likelihood, lr 1e-5, epochs 500",
        "trained_on_train_split_log_likelihood_lr_1e-4_epoch_010": "Trained on train split, log-likelihood, lr 1e-4, epochs 010",
        "trained_on_train_split_log_likelihood_lr_1e-4_epoch_020": "Trained on train split, log-likelihood, lr 1e-4, epochs 020",
        "trained_on_train_split_log_likelihood_lr_1e-4_epoch_050": "Trained on train split, log-likelihood, lr 1e-4, epochs 050",
        "trained_on_train_split_log_likelihood_lr_1e-4_epoch_200": "Trained on train split, log-likelihood, lr 1e-4, epochs 200",
        "trained_on_train_split_log_likelihood_lr_1e-4_epoch_500": "Trained on train split, log-likelihood, lr 1e-4, epochs 500",
        "finetuned_fim_05-03-2033_on_train_split_log_likelihood_lr_1e-5_epoch_010": "FIM 05-03-2033 Finetuned on train split, log-likelihood, lr 1e-5, epochs 010",
        "finetuned_fim_05-03-2033_on_train_split_log_likelihood_lr_1e-5_epoch_020": "FIM 05-03-2033 Finetuned on train split, log-likelihood, lr 1e-5, epochs 020",
        "finetuned_fim_05-03-2033_on_train_split_log_likelihood_lr_1e-5_epoch_050": "FIM 05-03-2033 Finetuned on train split, log-likelihood, lr 1e-5, epochs 050",
        "finetuned_fim_05-03-2033_on_train_split_log_likelihood_lr_1e-5_epoch_200": "FIM 05-03-2033 Finetuned on train split, log-likelihood, lr 1e-5, epochs 200",
        "finetuned_fim_05-03-2033_on_train_split_log_likelihood_lr_1e-5_epoch_500": "FIM 05-03-2033 Finetuned on train split, log-likelihood, lr 1e-5, epochs 500",
        "finetuned_fim_05-03-2033_on_train_split_log_likelihood_lr_1e-4_epoch_010": "FIM 05-03-2033 Finetuned on train split, log-likelihood, lr 1e-4, epochs 010",
        "finetuned_fim_05-03-2033_on_train_split_log_likelihood_lr_1e-4_epoch_020": "FIM 05-03-2033 Finetuned on train split, log-likelihood, lr 1e-4, epochs 020",
        "finetuned_fim_05-03-2033_on_train_split_log_likelihood_lr_1e-4_epoch_050": "FIM 05-03-2033 Finetuned on train split, log-likelihood, lr 1e-4, epochs 050",
        "finetuned_fim_05-03-2033_on_train_split_log_likelihood_lr_1e-4_epoch_200": "FIM 05-03-2033 Finetuned on train split, log-likelihood, lr 1e-4, epochs 200",
        "finetuned_fim_05-03-2033_on_train_split_log_likelihood_lr_1e-4_epoch_500": "FIM 05-03-2033 Finetuned on train split, log-likelihood, lr 1e-4, epochs 500",
    }

    results_to_load: list[str] = [
        # "/home/seifner/repos/FIM/evaluations/motion_capture_35/05121159_train_or_finetune_on_test_split/model_evaluations",
        # "/home/seifner/repos/FIM/evaluations/motion_capture_35/05121333_train_or_finetune_on_train_split/model_evaluations",
    ]

    num_sample_paths = 100
    # --------------------------------------------------------------------------------------------------------------------------------- #

    # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
    evaluation_path = Path(project_path) / "evaluations"
    time: str = str(datetime.now().strftime("%m%d%H%M"))
    evaluation_dir: Path = evaluation_path / dataset_descr / (time + "_" + experiment_descr)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # Get dataloaders inits and their display ids (for ModelEvaluation)
    # mocap_dir = Path("/home/seifner/repos/FIM/data/processed/test/20250123_preprocessed_mocap35/")
    mocap_dir = Path("/home/seifner/repos/FIM/data/processed/test/20250510_preprocessed_mocap35/")
    dataloader_dicts, dataloader_display_ids = get_mocap_35_dataloaders_inits(mocap_dir)

    # Get model_map to load models when they are needed
    model_map = model_map_from_dict(model_dicts)
    dataloader_map = dataloader_map_from_dict(dataloader_dicts)

    # Load previous evaluations that don't need to be evaluated anymore
    loaded_evaluations: list[ModelEvaluation] = load_evaluations(results_to_load)

    # Evaluate all models on all datasets
    all_evaluations: list[ModelEvaluation] = [
        ModelEvaluation(model_id, dataloader_id)
        for model_id, dataloader_id in itertools.product(model_dicts.keys(), dataloader_dicts.keys())
    ]
    to_evaluate: list[ModelEvaluation] = [evaluation for evaluation in all_evaluations if evaluation not in loaded_evaluations]

    # Create, run and save EvaluationConfig
    evaluated: list[ModelEvaluation] = run_mocap_35_evaluations(to_evaluate, model_map, dataloader_map, num_sample_paths)
    all_evaluations: list[ModelEvaluation] = loaded_evaluations + evaluated
    save_evaluations(all_evaluations, evaluation_dir / "model_evaluations")

    evaluation_config = EvaluationConfig(model_map, dataloader_map, evaluation_dir, models_display_ids, dataloader_display_ids)

    # Figure of sample paths on test set
    for model_evaluation in (pbar := tqdm(all_evaluations, total=len(all_evaluations), leave=False)):
        pbar.set_description(f"Individual latent forecasting grid: {model_evaluation.model_id}.")
        latent_forecasting_grid(model_evaluation, evaluation_config)

    pbar.close()

    # # 3D Figure of sample paths on test set
    # for model_evaluation in (pbar := tqdm(all_evaluations, total=len(all_evaluations), leave=False)):
    #     pbar.set_description(f"3D latent forecasting grid: {model_evaluation.model_id}.")
    #     latent_forecasting_grid_3D(model_evaluation, evaluation_config)
    #
    # pbar.close()

    # MSE per dataloader for all models, all dataloaders
    table_forecasting_mse(all_evaluations, evaluation_config)
