import itertools
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import optree
import pandas as pd
import torch
from model_dicts.models_trained_on_600k_deg_3_drift_deg_2_diffusion import get_model_dicts_600k_post_submission_models
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
from fim.utils.grids import define_regular_surrounding_cube


def get_high_dim_reconstr_mocap(paths: Tensor, eigenvalues: Tensor, eigenvectors: Tensor):
    """
    Project latent paths to high dimensional space.

    Args:
        paths (Tensor): Shape: ([1, 43] or [43, 1])[num_paths, T, 3]
        eigenvectors (Tensor): Shape: ([1, 43] or [43, 1])[50, 50])
        eigenvalues (Tensor): Shape: ([1, 43] or [43, 1])[50])

    Returns:
        high_dim_reconst (Tensor): Shape: ([1, 43] or [43, 1])[num_paths, T, 50]
    """
    assert paths.ndim == 5

    _, _, num_paths, T, _ = paths.shape

    eigenvalues = eigenvalues[:, :, None, None, :3]
    eigenvectors = eigenvectors[:, :, None, None, :, :3].expand(-1, -1, num_paths, T, -1, -1)

    # reconstruct from first 3 pca dimensions
    high_dim_reconst = torch.matmul((paths * torch.sqrt(eigenvalues))[..., None, :], torch.transpose(eigenvectors, -2, -1)).squeeze(-2)

    assert high_dim_reconst.ndim == 5
    return high_dim_reconst


def evaluate_mocap_model(model: AModel, dataloader: DataLoader, num_sample_paths: int, zs: list[float], device: Optional[str] = None):
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    results = {}

    dataset: dict = next(iter(dataloader))

    # evaluate vector field location grid with specified z values, surrounding observations
    # get regular grid in first 2 dimensions
    obs_values = dataset["obs_values"]
    obs_values = obs_values[..., :2]  # [..., 43, 128, 2]

    L = 256
    two_d_locations = define_regular_surrounding_cube(num_points=L, paths_values=obs_values, extension_perc=0.2)  # [1, L, 2]

    locations = []
    for z in zs:
        z_location = torch.ones_like(two_d_locations[..., 0][..., None]) * z
        locations.append(torch.concatenate([two_d_locations, z_location], dim=-1))  # [B, L, 3]

    locations = torch.concatenate(locations, dim=-2).expand(obs_values.shape[0], -1, -1)  # [B, zs * L, 3]

    dataset = optree.tree_map(lambda x: x.to(device), dataset, namespace="fimsde")
    dataset["locations"] = locations.to(device)

    estimated_concepts = model(dataset, training=False, return_losses=False)
    results.update({"estimated_concepts": estimated_concepts})

    # sample multiple paths starting from last observed value
    # "inference_mask": np.logical_or(last_obs_mask, forecasting_masks),  # [B, T, 1], solve from last oservation
    grid = dataset["obs_times"]  # [..., 43, T, 1]
    inference_mask = dataset["inference_mask"]  # [..., 43, T, 1]
    last_context_value = dataset["last_context_value"]  # [..., 43, 3]

    # sample multiple paths per trajectory
    last_context_value = torch.repeat_interleave(last_context_value, dim=-2, repeats=num_sample_paths)
    grid, inference_mask = optree.tree_map(lambda x: torch.repeat_interleave(x, dim=-3, repeats=num_sample_paths), (grid, inference_mask))

    sample_paths, sample_paths_grid = fimsde_sample_paths_on_masked_grid(
        model,
        dataset,
        initial_states=last_context_value,
        grid=grid,
        mask=inference_mask,
        solver_granularity=20,
    )  # [..., 43 * num_sample_paths, D]

    # Return paths per trajectory
    sample_paths = torch.stack(torch.split(sample_paths, num_sample_paths, dim=-3), dim=-4)  # [.., 43, num_sample_paths, T, D]
    sample_paths_grid = torch.stack(torch.split(sample_paths_grid, num_sample_paths, dim=-3), dim=-4)  # [.., 43, num_sample_paths, T, D]

    # return mean of sample paths
    mean_sample_paths = torch.mean(sample_paths, dim=-3, keepdim=True)
    mean_sample_paths_grid = sample_paths_grid[..., 0, :, :]  # sample path grids will be the same

    # add high dimensional reconstruction
    eigenvectors = dataset["eigenvectors"]
    eigenvalues = dataset["eigenvalues"]
    high_dim_from_samples = get_high_dim_reconstr_mocap(sample_paths, eigenvalues, eigenvectors)  # [..., 43, 40, 125, 50]
    high_dim_from_mean = get_high_dim_reconstr_mocap(mean_sample_paths, eigenvalues, eigenvectors)  # [...,1, 43, 1, 125, 50]

    # add MSE in high dimensional space at forecasting
    forecasting_mask = dataset["forecasting_mask"].bool()  # [..., 43, 125, 1]
    high_dim_trajectory = dataset["high_dim_trajectory"]  # [..., 43, 125, 50]

    def _mse_at_forecasting(targets: Tensor, prediction: Tensor, mask: Tensor):
        """
        Return MSE (across time and all 50 dimensions, as Yıldız 2019) at (mask == True) points between targets and predictions.

        Args:
            targets (Tensor): Shape: [..., 43, 125, 50]
            prediction (Tensor): Shape: [..., 43, P, 125, 50]
            mask (Tensor): MSE computed at True in time dim. Shape: [..., 43, 125, 1]

        returns:
            mse_per_path (Tensor): Shape: [..., 43, P]

        """
        targets = torch.broadcast_to(targets[..., None, :, :], prediction.shape)
        se = torch.mean((targets - prediction) ** 2, dim=-1)

        mask = torch.broadcast_to(mask[..., None, :, 0], se.shape)
        mse = torch.nanmean(torch.where(mask.squeeze(-1), se, torch.nan), dim=-1)  # [..., 43, P]
        return mse

    # mean prediction: report std over the 43 trajectories
    mse_from_mean = _mse_at_forecasting(high_dim_trajectory, high_dim_from_mean, forecasting_mask)  # [..., 43, 1]
    mse_from_mean_mean = mse_from_mean.mean()
    mse_from_mean_std = mse_from_mean.std()

    # best possible prediction (using PCA directly): report std over the 43 trajectories
    mse_from_gt_pca = _mse_at_forecasting(high_dim_trajectory, dataset["high_dim_reconst_from_3_pca"][..., None, :, :], forecasting_mask)
    mse_from_gt_pca_mean = mse_from_gt_pca.mean()
    mse_from_gt_pca_std = mse_from_gt_pca.std()

    # prediction from samples: mean and standard deviation *over the paths* per trajectory; average both over 43 trajectories
    mse_from_samples = _mse_at_forecasting(high_dim_trajectory, high_dim_from_samples, forecasting_mask)  # [..., 43, P]
    mse_from_samples_mean_paths = mse_from_samples.mean(dim=-1)  # [..., 43]
    mse_from_samples_std_paths = mse_from_samples.std(dim=-1)  # [..., 43]
    mse_from_samples_mean = mse_from_samples_mean_paths.mean(dim=-1)
    mse_from_samples_std = mse_from_samples_std_paths.mean(dim=-1)

    mse_from_samples_mean = mse_from_samples.mean()
    mse_from_samples_std = mse_from_samples.std()

    results.update(
        {
            "locations": locations,
            "estimated_concepts": estimated_concepts,
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

    results = optree.tree_map(lambda x: x.to("cpu").detach(), results, namespace="fimsde")

    return results


def run_mocap_evaluations(
    to_evaluate: list[ModelEvaluation],
    model_map: ModelMap,
    dataloader_map: DataLoaderMap,
    num_sample_paths: int,
    zs: list[float],
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

        evaluation.results = evaluate_mocap_model(model, dataloader, num_sample_paths, zs, device)
        evaluations_with_results.append(evaluation)

    return evaluations_with_results


def axis_motion_capture_single_latent(
    ax: plt.axis,
    obs_times: Tensor,  # [T]
    obs_values: Tensor,  # [T]
    obs_mask: Tensor,  # [T]
    forecasting_mask: Tensor,  # [T]
    model_sample_paths_grid: Tensor,  # [num_sample_paths, T]
    model_sample_paths: Tensor,  # [num_sample_paths, T, D]
    mean_model_sample_paths_grid: Optional[Tensor] = None,  # [T]
    mean_model_sample_paths: Optional[Tensor] = None,  # [T, D]
):
    # plot in (artificial) time range [0,1]
    max_forecast_time = obs_times[forecasting_mask].max()
    obs_times = obs_times / max_forecast_time
    model_sample_paths_grid = model_sample_paths_grid / max_forecast_time
    mean_model_sample_paths_grid = mean_model_sample_paths_grid / max_forecast_time

    ax.plot(
        obs_times[obs_mask],
        obs_values[obs_mask],
        label="Context",
        marker="o",
        markerfacecolor="None",
        markersize=3,
        markeredgewidth=0.6,
        linestyle="None",
        color="black",
    )
    ax.plot(
        obs_times[forecasting_mask],
        obs_values[forecasting_mask],
        label="Targets",
        marker="^",
        markerfacecolor="None",
        markersize=3,
        markeredgewidth=0.6,
        linestyle="None",
        c="#D55E00",
        zorder=2,
    )

    num_sample_paths = model_sample_paths_grid.shape[0]

    for i in range(num_sample_paths):
        ax.plot(
            model_sample_paths_grid[i],
            model_sample_paths[i],
            color="#0072B2",
            label=r"FIM-SDE samples" if i == 0 else None,
            alpha=0.2,
            linewidth=0.5,
            zorder=0,
        )

    if mean_model_sample_paths is not None:
        ax.plot(
            mean_model_sample_paths_grid,
            mean_model_sample_paths,
            color="#0072B2",
            label=r"FIM-SDE mean",
            linewidth=1.5,
            zorder=10,
        )

    # axis config
    [x.set_linewidth(0.3) for x in ax.spines.values()]
    ax.yaxis.set_major_locator(plticker.MultipleLocator(base=1.5))
    ax.set_xlim([0, 1])
    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.25))
    ax.set_xlabel(r"$Time$", fontsize=6, labelpad=0.75)

    # tick config
    ax.tick_params(axis="both", direction="out", labelsize=5, width=0.5, length=2)

    # grey region for observations
    last_time_before_imputation = obs_times[obs_mask].max()
    vspan_config = {"facecolor": "gainsboro", "alpha": 0.3, "zorder": 0}
    ax.axvspan(0, last_time_before_imputation, **vspan_config)


def figure_grid_motion_capture_latent(
    obs_times: Tensor,  # [43, T]
    obs_values: Tensor,  # [43, T, 3]
    obs_mask: Tensor,  # [43, T]
    forecasting_mask: Tensor,  # [43, T]
    model_sample_paths_grid: Tensor,  # [43, num_sample_paths, T]
    model_sample_paths: Tensor,  # [43, num_sample_paths, T, 3]
    mean_model_sample_paths_grid: Optional[Tensor] = None,  # [43, T]
    mean_model_sample_paths: Optional[Tensor] = None,  # [43, T, D]
    num_plot_traj: Optional[int] = 43,
    figsize: Optional[tuple] = (7, 43 * 2),
):
    fig, axs = plt.subplots(num_plot_traj, 3, figsize=figsize, dpi=300, tight_layout=True)

    for dim in range(3):
        axs[0, dim].set_title("Latent Dim. " + str(dim))

    for traj in range(num_plot_traj):
        axs[traj, 0].set_ylabel("Trajector Nr. " + str(traj))

        for dim in range(3):
            axis_motion_capture_single_latent(
                axs[traj, dim],
                obs_times[traj],
                obs_values[traj, :, dim],
                obs_mask[traj],
                forecasting_mask[traj],
                model_sample_paths_grid[traj],
                model_sample_paths[traj, :, :, dim],
                mean_model_sample_paths_grid[traj] if mean_model_sample_paths_grid is not None else None,
                mean_model_sample_paths[traj, :, dim] if mean_model_sample_paths is not None else None,
            )

    return fig


def figure_single_motion_capture_latent(
    obs_times: Tensor,  # [T]
    obs_values: Tensor,  # [T, 3]
    obs_mask: Tensor,  # [T]
    forecasting_mask: Tensor,  # [T]
    model_sample_paths_grid: Tensor,  # [num_sample_paths, T]
    model_sample_paths: Tensor,  # [num_sample_paths, T, 3]
    mean_model_sample_paths_grid: Optional[Tensor] = None,  # [T]
    mean_model_sample_paths: Optional[Tensor] = None,  # [T, D]
    figsize: Optional[tuple] = (7, 2),
):
    fig, axs = plt.subplots(1, 3, figsize=figsize, dpi=300, tight_layout=True)

    for dim in range(3):
        axis_motion_capture_single_latent(
            axs[dim],
            obs_times,
            obs_values[:, dim],
            obs_mask,
            forecasting_mask,
            model_sample_paths_grid,
            model_sample_paths[:, :, dim],
            mean_model_sample_paths_grid if mean_model_sample_paths_grid is not None else None,
            mean_model_sample_paths[:, dim] if mean_model_sample_paths is not None else None,
        )

    plt.draw()

    # place right legend directly on top of the plot
    handles, labels = axs[0].get_legend_handles_labels()
    legend_fontsize = 6
    bbox_x = axs[1].get_position().x0 + 0.5 * (axs[1].get_position().x1 - axs[1].get_position().x0)
    bbox_y = axs[1].get_position().y1

    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=[bbox_x, bbox_y],
        fontsize=legend_fontsize,
        ncols=4,
    )

    # line of samples broader in legend
    legend.get_lines()[2].set_linewidth(1.5)

    return fig


def figure_single_vector_fields(
    locations: Tensor,  # [L, 3]
    drift: Tensor,  # [L, 3]
    diffusion: Tensor,  # [L, 3]
    zs: list[float],
    figsize: Optional[tuple] = (7, 7),
):
    fig, axs = plt.subplots(2, len(zs), figsize=figsize, dpi=300, tight_layout=True)
    axs[0, 0].set_ylabel("Drift")
    axs[1, 0].set_ylabel("Diffusion")

    # only plot first two dimensions
    locations = locations[:, :2]
    drift = drift[:, :2]
    diffusion = diffusion[:, :2]

    # split results evenly, each chunk is associated to another z
    locations_per_z = torch.chunk(locations, chunks=len(zs), dim=0)
    drift_per_z = torch.chunk(drift, chunks=len(zs), dim=0)
    diffusion_per_z = torch.chunk(diffusion, chunks=len(zs), dim=0)

    for i in range(len(zs)):
        z = zs[i]
        locations = locations_per_z[i]
        drift = drift_per_z[i]
        diffusion = diffusion_per_z[i]

        axs[0, i].set_title(f"Slice at z={str(z)}")

        axs[0, i].quiver(locations[:, 0], locations[:, 1], drift[:, 0], drift[:, 1], color="#0072B2")
        axs[1, i].quiver(locations[:, 0], locations[:, 1], diffusion[:, 0], diffusion[:, 1], color="#0072B2")

    return fig


def latent_forecasting_grid(model_evaluation: ModelEvaluation, evaluation_config: EvaluationConfig, zs: list[float]):
    """
    Plot forecasting sample paths of one model for multiple mocap trajectories

    Args:
        model_evaluation (ModelEvaluation): Results to plot.
        evaluation_config (EvaluationConfig): Providing access to dataloader and model.
    """

    dataset: dict = get_data_from_model_evaluation(model_evaluation, evaluation_config)

    # extract data
    obs_times = dataset["obs_times"].squeeze()  # [43, T]
    obs_values = dataset["obs_values"].squeeze()  # [43, T, 3]
    obs_mask = dataset["obs_mask"].squeeze().bool()  # [43, T]
    forecasting_mask = dataset["forecasting_mask"].squeeze().bool()  # [43, T]

    model_sample_paths_grid = model_evaluation.results["sample_paths_grid"].squeeze()  # [43, num_sample_paths, 128]
    model_sample_paths = model_evaluation.results["sample_paths"].squeeze()  # [43, num_sample_paths, 128, 3]

    mean_model_sample_paths_grid = model_evaluation.results["mean_sample_paths_grid"].squeeze()  # [43, num_sample_paths, 128]
    mean_model_sample_paths = model_evaluation.results["mean_sample_paths"].squeeze()  # [43, num_sample_paths, 128, 3]

    # plot
    fig_grid = figure_grid_motion_capture_latent(
        obs_times,
        obs_values,
        obs_mask,
        forecasting_mask,
        model_sample_paths_grid,
        model_sample_paths,
        mean_model_sample_paths_grid,
        mean_model_sample_paths,
    )

    # save
    save_dir: Path = evaluation_dir / "figure_grid_latent_dimension" / model_evaluation.dataloader_id / model_evaluation.model_id
    save_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"model_{model_evaluation.model_id}"
    save_fig(fig_grid, save_dir, file_name)

    plt.close(fig_grid)

    # plot only selected trajectory
    selected_trajectory = 12

    # plot
    fig_single = figure_single_motion_capture_latent(
        obs_times[selected_trajectory],
        obs_values[selected_trajectory],
        obs_mask[selected_trajectory],
        forecasting_mask[selected_trajectory],
        model_sample_paths_grid[selected_trajectory],
        model_sample_paths[selected_trajectory],
        mean_model_sample_paths_grid[selected_trajectory],
        mean_model_sample_paths[selected_trajectory],
    )

    # save
    save_dir: Path = evaluation_dir / "figure_single_latent_dimension" / model_evaluation.dataloader_id / model_evaluation.model_id
    save_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"model_{model_evaluation.model_id}"
    save_fig(fig_single, save_dir, file_name)

    # # plot
    # estimated_concepts: SDEConcepts = model_evaluation.results["estimated_concepts"]
    # locations = estimated_concepts.locations
    # drift = estimated_concepts.drift
    # diffusion = estimated_concepts.diffusion
    #
    # if locations.shape[0] > 1:  # individual paths case
    #     locations = locations[selected_trajectory]
    #     drift = drift[selected_trajectory]
    #     diffusion = diffusion[selected_trajectory]
    #
    # else:
    #     locations = locations[0]
    #     drift = drift[0]
    #     diffusion = diffusion[0]
    #
    # fig_single = figure_single_vector_fields(locations, drift, diffusion, zs)
    #
    # # save
    # save_dir: Path = evaluation_dir / "figure_single_vector_field" / model_evaluation.dataloader_id / model_evaluation.model_id
    # save_dir.mkdir(parents=True, exist_ok=True)
    # file_name = f"model_{model_evaluation.model_id}"
    # save_fig(fig_single, save_dir, file_name)


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
        {"model": "GPDM", "data": "-", "MSE from mean path": r"126.46 $\pm$ 34", "MSE from individual paths": "-"},
        {"model": "VGPLVM", "data": "-", "MSE from mean path": r"142.18 $\pm$ 1.92", "MSE from individual paths": "-"},
        {"model": "DTSBN-S", "data": "-", "MSE from mean path": r"80.21 $\pm$ 0.04", "MSE from individual paths": "-"},
        {"model": "NPODE", "data": "-", "MSE from mean path": "45.74", "MSE from individual paths": "-"},
        {"model": "NEURALODE", "data": "-", "MSE from mean path": r"87.23 $\pm$ 0.02", "MSE from individual paths": "-"},
        {"model": "ODE2VAE", "data": "-", "MSE from mean path": r"93.07 $\pm$ 0.72", "MSE from individual paths": "-"},
        {"model": "ODE2VAE-KL", "data": "-", "MSE from mean path": r"15.99 $\pm$ 4.16", "MSE from individual paths": "-"},
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


def get_mocap_dataloaders_inits(mocap_dir: Optional[Path] = None) -> tuple[dict]:
    """
    Return DataLoaderInitializer for mocap forecasting data.

    Args:
        mocap_dir (Path): Absolute path to dir with subdirs of data.

    Returns:
        dataloder_dict, dataloader_map
    """
    files_to_load = {
        "obs_times": "observation_grid.h5",
        "obs_values": "observation_values.h5",
        "obs_mask": "observation_mask.h5",
        "forecasting_mask": "forecasting_mask.h5",
        "inference_mask": "inference_mask.h5",
        "last_context_value": "last_context_value.h5",
        "eigenvectors": "eigenvectors.h5",
        "eigenvalues": "eigenvalues.h5",
        "high_dim_trajectory": "high_dim_trajectory.h5",
        "high_dim_reconst_from_3_pca": "high_dim_reconst_from_3_pca.h5",
    }

    def _get_dataloader(data_subdir: Path):
        path = mocap_dir / data_subdir

        dataset = PaddedFIMSDEDataset(
            data_dirs=path,
            batch_size=43,
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

    dataloader_dict = {
        "forecasting_50_perc_all_paths": _get_dataloader("forecasting_0.5_perc/all_paths_one_element"),
        "forecasting_50_perc_single_path": _get_dataloader("forecasting_0.5_perc/single_path_per_element"),
    }

    dataloader_display_ids = {
        "forecasting_50_perc_all_paths": "All Paths Input, 50% forecast",
        "forecasting_50_perc_single_path": "Single Path Input, 50% forecast",
    }

    return dataloader_dict, dataloader_display_ids


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    dataset_descr = "motion_capture_43"

    # How to name experiments
    experiment_descr = "model_trained_on_delta_tau_1e-1_to_1e-3_1.3M_steps"

    model_dicts, models_display_ids = get_model_dicts_600k_post_submission_models()

    results_to_load: list[str] = [
        # "/home/seifner/repos/FIM/evaluations/motion_capture/01281819_develop_vector_field_figure/model_evaluations",
    ]

    num_sample_paths = 50
    zs = [-1, 0, 1]
    # --------------------------------------------------------------------------------------------------------------------------------- #

    # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
    evaluation_path = Path(project_path) / "evaluations"
    time: str = str(datetime.now().strftime("%m%d%H%M"))
    evaluation_dir: Path = evaluation_path / dataset_descr / (time + "_" + experiment_descr)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # Get dataloaders inits and their display ids (for ModelEvaluation)
    mocap_dir = Path("/home/seifner/repos/FIM/data/processed/test/20250115_preprocessed_mocap43/")
    dataloader_dicts, dataloader_display_ids = get_mocap_dataloaders_inits(mocap_dir)

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
    evaluated: list[ModelEvaluation] = run_mocap_evaluations(to_evaluate, model_map, dataloader_map, num_sample_paths, zs)
    all_evaluations: list[ModelEvaluation] = loaded_evaluations + evaluated
    save_evaluations(all_evaluations, evaluation_dir / "model_evaluations")

    evaluation_config = EvaluationConfig(model_map, dataloader_map, evaluation_dir, models_display_ids, dataloader_display_ids)

    # Figure of sample paths on test set
    for model_evaluation in (pbar := tqdm(all_evaluations, total=len(all_evaluations), leave=False)):
        pbar.set_description(f"Latent forecasting grid: {model_evaluation.model_id}.")
        latent_forecasting_grid(model_evaluation, evaluation_config, zs)

    pbar.close()

    # MSE per dataloader for all models, all dataloaders
    table_forecasting_mse(all_evaluations, evaluation_config)
