from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import optree
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from fim.data.datasets import PaddedFIMSDEDataset
from fim.models.sde import MinMaxNormalization, SDEConcepts, Standardization
from fim.utils.evaluation_sde import (
    EvaluationConfig,
    ModelEvaluation,
    dataloader_get_all_elements,
    find_indices_of_dim,
    get_data_from_model_evaluation,
    get_regression_metrics,
    save_df_per_dataloader_id,
    save_df_per_metric,
    save_df_per_model_id,
    save_fig,
    save_table,
)
from fim.utils.plots.sde_data_exploration_plots import plot_paths_in_axis
from fim.utils.plots.sde_estimation_plots import plot_1d_vf_real_and_estimation_axes, plot_2d_vf_real_and_estimation_axes


def _plot_1D_synthetic_data_figure_grid(
    locations: Tensor,  # [B, G, D]
    ground_truth_drift: Tensor,  # [B, G, D]
    ground_truth_diffusion: Tensor,  # [B, G, D]
    estimated_drift: Tensor,  # [B, G, D]
    estimated_diffuion: Tensor,  # [B, G, D]
    obs_paths_times: Tensor,  # [B, P, T, 1]
    obs_paths_values: Tensor,  # [B, P, T, D]
    obs_paths_mask: Tensor,  # [B, P, T, 1]
    model_paths_times: Tensor,  # [B, P, L, 1]
    model_paths_values: Tensor,  # [B, P, L, D]
    **kwargs,
) -> plt.Figure:
    """
    Plot B equations of dimension 1 as a grid of figures.
    Each row contains ground-truth and (single) model estimation drift and diffusion, and P 1D paths from data and sampled from model.

    Args: Ground-truth and estimation of paths and vector fields. Shape: [B, G, D] or [B, P, T, D] for inputs and [B, P, L, ] model paths.

    Returns: Figure with data from B equations.
    """
    ncols = 2
    nrows = obs_paths_times.shape[0]

    figsize_per_col = kwargs.get("figsize_per_col", 4)
    figsize_per_row = kwargs.get("figsize_per_row", 4)
    figsize = (ncols * figsize_per_col, nrows * figsize_per_row)

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)

    if nrows == 1:
        axs = axs.reshape(1, -1)

    for row in range(nrows):
        ax_diff = axs[row, 0].twinx()
        plot_1d_vf_real_and_estimation_axes(
            axs[row, 0],
            ax_diff,
            locations[row, ..., :1],
            ground_truth_drift[row, ..., :1],
            estimated_drift[row, ..., :1],
            ground_truth_diffusion[row, ..., :1],
            estimated_diffuion[row, ..., :1],
        )

        plot_paths_in_axis(
            axs[row, 1],
            obs_paths_times[row],
            obs_paths_values[row, ..., :1],
            obs_paths_mask[row],
            color=kwargs.get("obs_paths_color", "red"),
            paths_label=kwargs.get("obs_paths_label", "Observed paths"),
            initial_states_label=kwargs.get("initial_states_label", "Initial states"),
        )
        plot_paths_in_axis(
            axs[row, 1],
            model_paths_times[row],
            model_paths_values[row, ..., :1],
            color=kwargs.get("model_paths_color", "black"),
            paths_label=kwargs.get("model_paths_label", "Model samples"),
        )

    return fig


def _plot_2D_synthetic_data_figure_grid(
    locations: Tensor,  # [B, G, D]
    ground_truth_drift: Tensor,  # [B, G, D]
    ground_truth_diffusion: Tensor,  # [B, G, D]
    estimated_drift: Tensor,  # [B, G, D]
    estimated_diffuion: Tensor,  # [B, G, D]
    obs_paths_times: Tensor,  # [B, P, T, 1]
    obs_paths_values: Tensor,  # [B, P, T, D]
    obs_paths_mask: Tensor,  # [B, P, T, 1]
    model_paths_times: Tensor,  # [B, P, L, 1]
    model_paths_values: Tensor,  # [B, P, L, D]
    **kwargs,
):
    """
    Plot B equations of dimension 2 as a grid of figures.
    Each row contains ground-truth and (single) model estimation drift and diffusion, and P 2D paths from data and sampled from model.

    Args: Ground-truth and estimation of paths and vector fields. Shape: [B, G, D] or [B, P, T, D] for inputs and [B, P, L, D] model paths.

    Returns: Figure with data from B equations.
    """
    ncols = 5
    nrows = obs_paths_times.shape[0]

    figsize_per_col = kwargs.get("figsize_per_col", 4)
    figsize_per_row = kwargs.get("figsize_per_row", 4)
    figsize = (ncols * figsize_per_col, nrows * figsize_per_row)

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)

    if nrows == 1:
        axs = axs.reshape(1, -1)

    for row in range(nrows):
        plot_2d_vf_real_and_estimation_axes(
            axs[row, 0],
            axs[row, 1],
            axs[row, 2],
            axs[row, 3],
            locations[row, ..., :2],
            ground_truth_drift[row, ..., :2],
            estimated_drift[row, ..., :2],
            ground_truth_diffusion[row, ..., :2],
            estimated_diffuion[row, ..., :2],
        )

        plot_paths_in_axis(
            axs[row, 4],
            obs_paths_times[row],
            obs_paths_values[row, ..., :2],
            obs_paths_mask[row],
            color=kwargs.get("obs_paths_color", "red"),
            paths_label=kwargs.get("obs_paths_label", "Observed paths"),
            initial_states_label=kwargs.get("initial_states_label", "Initial states"),
        )
        plot_paths_in_axis(
            axs[row, 4],
            model_paths_times[row],
            model_paths_values[row, ..., :2],
            color=kwargs.get("model_paths_color", "black"),
            paths_label=kwargs.get("model_paths_label", "Model samples"),
        )

    return fig


def _plot_3D_synthetic_data_figure_grid(
    locations: Tensor,  # [B, G, D]
    ground_truth_drift: Tensor,  # [B, G, D]
    ground_truth_diffusion: Tensor,  # [B, G, D]
    estimated_drift: Tensor,  # [B, G, D]
    estimated_diffuion: Tensor,  # [B, G, D]
    obs_paths_times: Tensor,  # [B, P, T, 1]
    obs_paths_values: Tensor,  # [B, P, T, D]
    obs_paths_mask: Tensor,  # [B, P, T, 1]
    model_paths_times: Tensor,  # [B, P, T, 1]
    model_paths_values: Tensor,  # [B, P, T, D]
    **kwargs,
):
    """
    Plot B equations of dimension : as a grid of figures.
    Each row contains ground-truth and (single) model estimation drift and diffusion, and P 3D paths from data and sampled from model.

    Args: Ground-truth and estimation of paths and vector fields. Shape: [B, G, D] or [B, P, T, D] for inputs and [B, P, L, D] model paths.

    Returns: Figure with data from B equations.
    """
    ncols = 1
    nrows = obs_paths_times.shape[0]

    figsize_per_col = kwargs.get("figsize_per_col", 4)
    figsize_per_row = kwargs.get("figsize_per_row", 4)
    figsize = (ncols * figsize_per_col, nrows * figsize_per_row)

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, subplot_kw={"projection": "3d"})

    if nrows > 1:
        axs = axs.reshape(nrows, ncols)
    else:
        axs = np.array(axs).reshape(1, 1)

    for row in range(nrows):
        plot_paths_in_axis(
            axs[row, 0],
            obs_paths_times[row],
            obs_paths_values[row, ..., :3],
            obs_paths_mask[row],
            color=kwargs.get("obs_paths_color", "red"),
            paths_label=kwargs.get("obs_paths_label", "Observed paths"),
            initial_states_label=kwargs.get("initial_states_label", "Initial states"),
        )
        plot_paths_in_axis(
            axs[row, 0],
            model_paths_times[row],
            model_paths_values[row, ..., :3],
            color=kwargs.get("model_paths_color", "black"),
            paths_label=kwargs.get("model_paths_label", "Model samples"),
        )

    return fig


def synthetic_data_plots(
    evaluation_config: EvaluationConfig,
    model_evaluation: ModelEvaluation,
    plot_indices: Optional[int | Tensor] = 20,
    max_ground_truth_paths: Optional[int] = None,
    max_model_paths: Optional[int] = None,
    random_indices=False,
    **kwargs,
) -> None:
    """
    Find equations of dimension 1, 2 or 3 in dataloader from evaluation_config. Plot plot_indices of data and model evaluations in a
    grid of subplots, where each row contains ground-truth and estimated vector fields, and observed and model sample paths.
    Save figures per dimension in subdirs of evaluation_config.save_dir.
    """
    # get data from dataloader
    dataloder = evaluation_config.get_dataloader(model_evaluation)
    data = dataloader_get_all_elements(dataloder)

    for dim in range(1, 4):
        # select elements of dimension from data and model_evaluation
        if isinstance(plot_indices, int):
            indices = find_indices_of_dim(dim, data["dimension_mask"], indices_count=plot_indices, random_indices=random_indices)

        else:
            indices = plot_indices

        data_of_dim, model_output_of_dim = optree.tree_map(lambda x: x[indices], (data, model_evaluation.results), namespace="fimsde")

        if max_ground_truth_paths is not None and data_of_dim["obs_times"].size(1) > max_ground_truth_paths:
            data_of_dim["obs_times"] = data_of_dim["obs_times"][:, :max_ground_truth_paths]
            data_of_dim["obs_values"] = data_of_dim["obs_values"][:, :max_ground_truth_paths]
            data_of_dim["obs_mask"] = data_of_dim["obs_mask"][:, :max_ground_truth_paths]

        if max_model_paths is not None and model_output_of_dim["sample_paths"].size(1) > max_model_paths:
            model_output_of_dim["sample_paths_grid"] = model_output_of_dim["sample_paths_grid"][:, :max_model_paths]
            model_output_of_dim["sample_paths"] = model_output_of_dim["sample_paths"][:, :max_model_paths]

        if data_of_dim["locations"].shape[0] != 0:
            # plot with method for each dimension
            if dim == 1:
                grid_plot_func = _plot_1D_synthetic_data_figure_grid
            if dim == 2:
                grid_plot_func = _plot_2D_synthetic_data_figure_grid
            if dim == 3:
                grid_plot_func = _plot_3D_synthetic_data_figure_grid

            fig = grid_plot_func(
                data_of_dim["locations"],
                data_of_dim["drift_at_locations"],
                data_of_dim["diffusion_at_locations"],
                model_output_of_dim["estimated_concepts"].drift,
                model_output_of_dim["estimated_concepts"].diffusion,
                data_of_dim["obs_times"],
                data_of_dim["obs_values"],
                data_of_dim["obs_mask"].bool(),
                model_output_of_dim["sample_paths_grid"],
                model_output_of_dim["sample_paths"],
                **kwargs,
            )

            # save
            save_dir: Path = evaluation_config.save_dir / "figure_grid" / model_evaluation.dataloader_id / model_evaluation.model_id
            save_dir.mkdir(parents=True, exist_ok=True)
            file_name = f"{dim}D_plots_data_{model_evaluation.dataloader_id}_model_{model_evaluation.model_id}"
            save_fig(fig, save_dir, file_name)

            plt.close(fig)


def _get_statistics_from_tensor(tensor: Tensor, label: str, mask: Optional[Tensor] = None) -> dict[str, float]:
    """
    Collect some stats from tensor (min, max, median) as dict, where keys include label for tensor.

    Args:
        tensor (Tensor) + mask (Tensor): Tensor to compute stats of values where mask == True.
        label (str): Label for tensor.

    Return:
        stats_dict (dict[str, Tensor]): Maps label + statistic to statistic about tensor: label + "_" + stat -> stat_value
    """
    if mask is None:
        mask = torch.ones_like(tensor).bool()

    mask = mask.bool()

    min_ = torch.min(torch.where(mask, tensor, torch.inf))
    max_ = torch.max(torch.where(mask, tensor, -torch.inf))
    median_ = torch.nanmedian(torch.where(mask, tensor, torch.nan))

    stats_dict = {"min": min_, "max": max_, "median": median_}
    return {label + "_" + key: value.item() for key, value in stats_dict.items()}


def _get_statistics_from_synthetic_dataset(
    obs_times: Tensor,  # [B, P, T, 1]
    obs_values: Tensor,  # [B, P, T, D]
    obs_mask: Tensor | None,  # [B, P, T, 1]
    locations: Tensor,  # [B, G, D]
    drift_at_locations: Tensor,  # [B, G, D]
    diffusion_at_locations: Tensor,  # [B, G, D]
    dimension_mask: Optional[Tensor],  # [B, G, D]
) -> tuple[dict[str, Tensor]]:
    """
    Return statistics from generated datasets.

    Returns:
        general_stats (dict[str, Tensor]): Shapes and sizes of tensors.
        tensors_stats (dict[str, Tensor]): E.g. min and max values.
        drift_threshold_stats (dict[str, Tensor]): Norm of drift exceeding threshold values.
        diffusion_threshold_stats (dict[str, Tensor]): Norm of diffusion exceeding threshold values.
    """
    if dimension_mask is None:
        dimension_mask = torch.ones_like(locations)

    num_elements, num_paths, ts_length, max_dim = obs_times.shape
    num_locations = locations.shape[1]

    # shapes and sizes
    general_stats = {
        "num_elements": num_elements,
        "num_paths": num_paths,
        "ts_length": ts_length,
        "max_dim": max_dim,
        "num_locations": num_locations,
    }

    # get statistics per tensor
    obs_times_stats: dict = _get_statistics_from_tensor(obs_times, "obs_times", obs_mask)

    obs_norm = torch.linalg.norm(obs_values, dim=-1)
    obs_values_stats: dict = _get_statistics_from_tensor(obs_norm, "obs_values", obs_mask[:, :, :, 0])

    locations_norm = torch.linalg.norm(locations, dim=-1)
    locations_norm_stats: dict = _get_statistics_from_tensor(locations_norm, "locations_norm", dimension_mask[:, :, 0])

    drift_norm = torch.linalg.norm(drift_at_locations, dim=-1)
    drift_norm_stats: dict = _get_statistics_from_tensor(drift_norm, "drift_norm", dimension_mask[:, :, 0])

    diffusion_norm = torch.linalg.norm(diffusion_at_locations, dim=-1)
    diffusion_norm_stats: dict = _get_statistics_from_tensor(diffusion_norm, "diffusion_norm", dimension_mask[:, :, 0])

    tensors_stats: dict = obs_times_stats | obs_values_stats | locations_norm_stats | drift_norm_stats | diffusion_norm_stats

    # norm below thresholds for vector fields
    thresholds = [1, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]

    drift_threshold_stats = {}
    diffusion_threshold_stats = {}

    for threshold in thresholds:
        drift_norm_perc = (drift_norm < threshold).mean(dtype=torch.float32).item()
        drift_threshold_stats.update({f"drift_norm_<_{threshold}_perc": drift_norm_perc})

        diffusion_norm_perc = (diffusion_norm < threshold).mean(dtype=torch.float32).item()
        diffusion_threshold_stats.update({f"diffusion_norm_<_{threshold}_perc": diffusion_norm_perc})

    return general_stats, tensors_stats, drift_threshold_stats, diffusion_threshold_stats


def _get_statistics_from_multiple_synthetic_datasets(
    evaluation_config: EvaluationConfig,
    dataloader_ids: list[str],
    precision: Optional[float] = None,
    normalization: Optional[str] = "unnormalized",
    **kwargs,
) -> tuple[pd.DataFrame]:
    """
    For multiple dataloaders (specified by their id), collect their statistics in DataFrames.

    Args:
        evaluation_config (EvaluationConfig): Needed for the .dataloder_map.
        dataloader_ids (list[str]): Ids of dataloaders (in evaluation_config) to collect statistics from.
        precision (Optional[float]): Precision of values in DataFrames.
        normalization (Optional[str]): Specifies if and with what data normalization the statistics should be computed.

    Returns:
        Output from _get_statistics_from_synthetic_dataset as Dataframes with one dataset per row.
    """
    dataloaders_general_stats = []
    dataloaders_tensors_stats = []
    dataloaders_drift_threshold_stats = []
    dataloaders_diffusion_threshold_stats = []

    for dataloader_id in (pbar := tqdm(dataloader_ids, total=len(dataloader_ids), leave=False)):
        pbar.set_description_str(f"Metrics from dataloader {dataloader_id}")
        data: dict = dataloader_get_all_elements(evaluation_config.dataloader_map[dataloader_id]())

        if normalization == "unnormalized":
            pass

        elif normalization in ["min_max_normalized", "standardized"]:
            if normalization == "min_max_normalized":
                values_norm = MinMaxNormalization(normalized_min=-1, normalized_max=1)
            else:
                values_norm = Standardization()

            values_norm_stats = values_norm.get_norm_stats(data["obs_values"], data.get("obs_mask"))
            obs_values = values_norm.normalization_map(data["obs_values"], values_norm_stats)

            times_norm = MinMaxNormalization(normalized_min=0, normalized_max=1)
            times_norm_stats = times_norm.get_norm_stats(data["obs_times"], data.get("obs_mask"))
            obs_times = times_norm.normalization_map(data["obs_times"], times_norm_stats)

            sde_concepts = SDEConcepts.from_dict(data)
            sde_concepts.normalize(values_norm, values_norm_stats, times_norm, times_norm_stats)

            data.update({"obs_times": obs_times, "obs_values": obs_values})
        else:
            raise ValueError(f"normalization mode {normalization} not recognized")

        general_stats, tensors_stats, drift_threshold_stats, diffusion_threshold_stats = _get_statistics_from_synthetic_dataset(
            data.get("obs_times"),
            data.get("obs_values"),
            data.get("obs_mask"),
            data.get("locations"),
            data.get("drift_at_locations"),
            data.get("diffusion_at_locations"),
            data.get("dimension_mask"),
        )
        dataloaders_tensors_stats.append(pd.DataFrame(tensors_stats, index=[dataloader_id]))
        dataloaders_general_stats.append(pd.DataFrame(general_stats, index=[dataloader_id]))
        dataloaders_drift_threshold_stats.append(pd.DataFrame(drift_threshold_stats, index=[dataloader_id]))
        dataloaders_diffusion_threshold_stats.append(pd.DataFrame(diffusion_threshold_stats, index=[dataloader_id]))

    def _combine_dfs(dfs: list[pd.DataFrame]) -> pd.DataFrame:
        df = pd.concat(dfs)
        df.index = df.index.map(lambda x: evaluation_config.dataloader_display_id_map[x])
        df: pd.DataFrame = df.sort_index(axis=0)
        if precision is not None:
            df: pd.DataFrame = df.round(precision)

        return df

    tensors_stats_df: pd.DataFrame = _combine_dfs(dataloaders_tensors_stats)
    general_stats_df: pd.DataFrame = _combine_dfs(dataloaders_general_stats)
    drift_threshold_stats_df: pd.DataFrame = _combine_dfs(dataloaders_drift_threshold_stats)
    diffusion_threshold_stats_df: pd.DataFrame = _combine_dfs(dataloaders_diffusion_threshold_stats)

    return tensors_stats_df, general_stats_df, drift_threshold_stats_df, diffusion_threshold_stats_df


def synthetic_dataset_statistics(
    evaluation_config: EvaluationConfig, model_evaluations: list[ModelEvaluation], precision: Optional[float] = None, **kwargs
):
    """
    Collect and save statistics from all dataloaders used in model_evaluations.
    """
    unique_dataloader_ids = list({evaluation.dataloader_id for evaluation in model_evaluations})

    normalization_modes = ["unnormalized", "min_max_normalized", "standardized"]

    for normalization_mode in (pbar := tqdm(normalization_modes, total=len(normalization_modes), leave=False)):
        pbar.set_description(f"Normalization: {normalization_mode}")

        tensors_stats_df, general_stats_df, drift_threshold_stats_df, diffusion_threshold_stats_df = (
            _get_statistics_from_multiple_synthetic_datasets(
                evaluation_config,
                unique_dataloader_ids,
                precision,
                normalization_mode,
                **kwargs,
            )
        )

        dataset_stats_dir: Path = evaluation_config.save_dir / "datasets_statistics" / normalization_mode
        save_table(tensors_stats_df, dataset_stats_dir, "min_max_median_of_tensors")
        save_table(general_stats_df, dataset_stats_dir, "general_stats_about_tensors")
        save_table(drift_threshold_stats_df, dataset_stats_dir, "drift_norm_below_threshold_stats")
        save_table(diffusion_threshold_stats_df, dataset_stats_dir, "diffusion_norm_below_threshold_stats")


def get_synthetic_dataloader(
    data_dirs: list[str],
    batch_size: Optional[int] = 16,
    num_workers: Optional[int] = 8,
    max_dim: Optional[int] = 3,
    shuffle_locations: Optional[bool] = False,
    shuffle_paths: Optional[bool] = False,
    shuffle_elements: Optional[bool] = False,
) -> DataLoader:
    """
    Return Dataloader (with FIMSDEDataloader.fimsde_collate_fn collate function) of some generated dynamical system dataset.
    Load data from files, without permutation or randomness, using num_workers for batch_size per batch

    Args:
        data_dirs (list[str]): Paths to directories containing generated data.
        batch_size, num_workers (int): Passed to torch.DataLoader.

    Returns:
        dataloader (Dataloader): Dataloader returning data as dicts.
    """
    files_to_load = {
        "obs_times": "obs_times.h5",
        "obs_values": "obs_values.h5",
        "obs_mask": "obs_mask.h5",
        "locations": "locations.h5",
        "drift_at_locations": "drift_at_locations.h5",
        "diffusion_at_locations": "diffusion_at_locations.h5",
    }
    dataset = PaddedFIMSDEDataset(
        data_dirs=data_dirs,
        batch_size=batch_size,
        files_to_load=files_to_load,
        max_dim=max_dim,
        shuffle_locations=shuffle_locations,
        shuffle_paths=shuffle_paths,
        shuffle_elements=shuffle_elements,
    )

    dataloader: DataLoader = DataLoader(
        dataset,
        drop_last=False,
        shuffle=False,
        batch_size=None,  # handled by iterable dataset
        num_workers=num_workers,
    )

    return dataloader


def add_vector_field_metrics(
    evaluation_config: EvaluationConfig,
    model_evaluation: ModelEvaluation,
    metrics: list[str],
    estimated_concepts_key: Optional[str] = "estimated_concepts",
) -> ModelEvaluation:
    """
    If not already computed, add set of metrics to model_evaluation.results.

    Args:
        evaluation_config, model_evaluation: Specify model results and data to compute metrics on.
        metrics (list[str]): Metrics to compute for drift and diffusion.
        estimated_concepts_key (Optional[str]): Key for model output in model_evaluation.results.
    """
    if "metrics" not in model_evaluation.results.keys():
        # get data and model results on that data
        estimated_concepts: SDEConcepts = model_evaluation.results.get(estimated_concepts_key)
        data: dict = get_data_from_model_evaluation(model_evaluation, evaluation_config)

        # compute metrics
        metrics_for_drift: dict[str, Tensor] = get_regression_metrics(estimated_concepts.drift, data["drift_at_locations"], metrics)
        metrics_for_diffusion: dict[str, Tensor] = get_regression_metrics(
            estimated_concepts.diffusion, data["diffusion_at_locations"], metrics
        )
        metrics = {"drift": metrics_for_drift, "diffusion": metrics_for_diffusion}

        # add metrics to result
        model_evaluation.results.update({"metrics": metrics})

    return model_evaluation


def get_df_with_metrics_per_model_and_data(
    metrics_per_model_dataloader: list[dict], model_ids: list[str], dataloader_ids: list[str], precision: Optional[int] = 2
) -> pd.DataFrame:
    """
    Combine dict of metrics for each evaluation to a DataFrame, compute their mean and standard deviation and concatenate all dfs,
    using model_id and dataloder_id as labels.

    Args:
        metrics_per_model_dataloader (list[dict]): Metrics for one model elaluation: {"mse": [B], "rmse": ...}
        model_ids (list[dict]): Associated model_id.
        dataloader_ids (list[dict]): Associated dataloader_id.
        precision (int): Precision for string representation of floats.

    Returns:
        df (pd.DataFrame): Columns: [model_id, dataloader_id, metrics...]
    """
    dfs_per_model_eval: list[dict] = []

    # columns: model_id, dataloader_id, metrics per batch element
    for metrics, model_id, dataloader_id in tqdm(
        zip(metrics_per_model_dataloader, model_ids, dataloader_ids),
        total=len(model_ids),
        desc="Metrics per Model and Dataloader",
        leave=False,
    ):
        metrics_df: pd.DataFrame = pd.DataFrame.from_dict(metrics)
        metrics_df["model_id"] = model_id
        metrics_df["dataloader_id"] = dataloader_id

        dfs_per_model_eval.append(metrics_df)

    df_all_model_evals: pd.DataFrame = pd.concat(dfs_per_model_eval)

    # mean and std per model eval, has multicolumn (metric, mean or std)
    df_mean_std_metrics = df_all_model_evals.groupby(["model_id", "dataloader_id"]).agg(["mean", "std"])
    df_mean_std_metrics = df_mean_std_metrics.round(precision)

    # get string representation per metric
    all_metrics = list({metric for (metric, _) in df_mean_std_metrics.columns})

    for metric in all_metrics:
        df_mean_std_metrics[(metric, "repr")] = (
            df_mean_std_metrics[(metric, "mean")].astype(str) + " $\\pm$ " + df_mean_std_metrics[(metric, "std")].astype(str)
        )

    # reduce to only string representation and make model_id + dataloader_id into columns
    df_mean_std_metrics = df_mean_std_metrics[[(metric, "repr") for metric in all_metrics]]
    df_mean_std_metrics = df_mean_std_metrics.droplevel(level=1, axis=1)
    df_mean_std_metrics = df_mean_std_metrics.reset_index()

    return df_mean_std_metrics


def synthetic_data_metrics(
    evaluation_config: EvaluationConfig,
    model_evaluations: list[ModelEvaluation],
    precision: Optional[int] = 2,
) -> None:
    """
    Save tables of metrics on vector fields for synthetic datasets.

    Args:
        evaluation_config, model_evaluations: Define evaluations with metrics to save. Metrics are at model_evaluation.results["metrics"]
        precision (int): Precision for displaying floats in tables.
    """
    # extract metrics per model, dataloader and vector field
    drift_metrics = [model_eval.results["metrics"]["drift"] for model_eval in model_evaluations]
    diffusion_metrics = [model_eval.results["metrics"]["diffusion"] for model_eval in model_evaluations]
    model_ids = [model_eval.model_id for model_eval in model_evaluations]
    dataloader_ids = [dataloader_eval.dataloader_id for dataloader_eval in model_evaluations]

    # drift: df with mean and std of all models, dataloaders and metrics
    drift_metrics_df = get_df_with_metrics_per_model_and_data(drift_metrics, model_ids, dataloader_ids, precision)

    # rename model_ids and dataloader_ids
    drift_metrics_df["dataloader_id"] = drift_metrics_df["dataloader_id"].map(lambda x: evaluation_config.dataloader_display_id_map[x])
    drift_metrics_df["model_id"] = drift_metrics_df["model_id"].map(lambda x: evaluation_config.model_display_id_map[x])

    # save different views of table
    save_df_per_metric(drift_metrics_df, evaluation_config.save_dir / "metrics" / "drift" / "per_metric")
    save_df_per_model_id(drift_metrics_df, evaluation_config.save_dir / "metrics" / "drift" / "per_model")
    save_df_per_dataloader_id(drift_metrics_df, evaluation_config.save_dir / "metrics" / "drift" / "per_dataloader")

    # diffusion: df with mean and std of all models, dataloaders and metrics
    diffusion_metrics_df = get_df_with_metrics_per_model_and_data(diffusion_metrics, model_ids, dataloader_ids, precision)

    # rename model_ids and dataloader_ids
    diffusion_metrics_df["dataloader_id"] = diffusion_metrics_df["dataloader_id"].map(
        lambda x: evaluation_config.dataloader_display_id_map[x]
    )
    diffusion_metrics_df["model_id"] = diffusion_metrics_df["model_id"].map(lambda x: evaluation_config.model_display_id_map[x])

    # save different views of table
    save_df_per_metric(diffusion_metrics_df, evaluation_config.save_dir / "metrics" / "diffusion" / "per_metric")
    save_df_per_model_id(diffusion_metrics_df, evaluation_config.save_dir / "metrics" / "diffusion" / "per_model")
    save_df_per_dataloader_id(diffusion_metrics_df, evaluation_config.save_dir / "metrics" / "diffusion" / "per_dataloader")
