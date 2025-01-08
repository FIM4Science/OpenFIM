from abc import ABC, abstractmethod
from typing import Optional

import matplotlib.pyplot as plt
import optree
import torch
from torch import Tensor

from fim.data.dataloaders import BaseDataLoader
from fim.models.blocks import AModel
from fim.models.sde import SDEConcepts
from fim.pipelines.sde_sampling_from_model import fimsde_sample_paths
from fim.trainers.utils import TrainLossTracker
from fim.utils.plots.sde_estimation_plots import (
    plot_1d_vf_real_and_estimation,
    plot_2d_vf_real_and_estimation,
    plot_3d_vf_real_and_estimation,
    plot_paths,
)


class EvaluationEpoch(ABC):
    model: AModel
    dataloader: BaseDataLoader
    loss_tracker: TrainLossTracker
    debug_mode: bool

    # accelerator handling
    local_rank: int
    accel_type: str
    use_mixeprecision: bool
    is_accelerator: bool
    auto_cast_type: torch.dtype

    def __init__(
        self,
        model: AModel,
        dataloader: BaseDataLoader,
        loss_tracker: TrainLossTracker,
        debug_mode: bool,
        local_rank: int,
        accel_type: str,
        use_mixeprecision: bool,
        is_accelerator: bool,
        auto_cast_type: torch.dtype,
        **kwargs,
    ):
        self.model: AModel = model
        self.dataloader: BaseDataLoader = dataloader
        self.loss_tracker: TrainLossTracker = loss_tracker

        # accelerator handling
        self.debug_mode: bool = debug_mode
        self.local_rank: int = local_rank
        self.accel_type: str = accel_type
        self.use_mixeprecision: bool = use_mixeprecision
        self.is_accelerator: bool = is_accelerator
        self.auto_cast_type: torch.dtype = auto_cast_type

    @abstractmethod
    def __call__(self, epoch: int) -> dict:
        """
        Run evaluation epoch and return a dict with stats to log.
        Returned dict looks something like {"losses": losses_dict, "figures": figures_dict}, where:
            losses_dict: maps labels to zero-dim. tensors
            figures: maps labels to plt.figures
        """
        raise NotImplementedError("The __call__ method is not implemented in your class!")


class TestEvaluationEpoch(EvaluationEpoch):
    def __init__(
        self,
        model,
        dataloader,
        loss_tracker,
        debug_mode,
        local_rank,
        accel_type,
        use_mixeprecision,
        is_accelerator,
        auto_cast_type,
        **kwargs,
    ):
        super().__init__(
            model, dataloader, loss_tracker, debug_mode, local_rank, accel_type, use_mixeprecision, is_accelerator, auto_cast_type
        )

    def __call__(self, epoch: int) -> dict:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(torch.linspace(0, 1, 10), torch.linspace(-2, 1, 10))

        return {"losses": {"dumm_loss": torch.sum(torch.zeros(1))}, "figures": {"dummy": fig}}


class SDEEvaluationPlots(EvaluationEpoch):
    def __init__(
        self,
        model,
        dataloader,
        loss_tracker,
        debug_mode,
        local_rank,
        accel_type,
        use_mixeprecision,
        is_accelerator,
        auto_cast_type,
        **kwargs,
    ):
        super().__init__(
            model, dataloader, loss_tracker, debug_mode, local_rank, accel_type, use_mixeprecision, is_accelerator, auto_cast_type
        )

        # which dataloader to take plotting data from
        iterator_name: str = kwargs.get("iterator_name", "test")
        if iterator_name == "validation":
            self.dataloader = dataloader.validation_it
        elif iterator_name == "test":
            self.dataloader = dataloader.test_it
        elif iterator_name == "evaluation":
            self.dataloader = dataloader.evaluation_it

        # how often to plot
        self.plot_frequency: int = kwargs.get("plot_frequency", 1)

        # how many paths to show maximal
        self.plot_paths_count: int = kwargs.get("plot_paths_count")

    @staticmethod
    def plot_example_of_dim(
        data: dict,
        estimated_concepts: SDEConcepts,
        sample_paths: Tensor,
        sample_grid: Tensor,
        dimension: int,
        plot_paths_count: Optional[int],
    ) -> tuple[plt.figure]:
        """
        Plot data and model estimates of random example from data.
        """
        # select batch elements of dimension
        has_dim: Tensor[bool] = data["dimension_mask"].sum(dim=-1)[:, 0].long() == dimension  # [B]
        if not torch.any(has_dim).item():
            return None, None

        input_data: tuple = (data, estimated_concepts, sample_paths, sample_grid)
        input_data = optree.tree_map(lambda x: x[has_dim], input_data, namespace="fimsde")

        # extract example at random index
        B = optree.tree_flatten(input_data)[0][0].shape[0]
        index = torch.randint(size=(1,), low=0, high=B).item()
        input_data = optree.tree_map(lambda x: x[index], input_data, namespace="fimsde")

        # truncate last dimension to dimension
        input_data = optree.tree_map(lambda x: x[..., :dimension] if x.shape[-1] >= dimension else x, input_data, namespace="fimsde")

        # prepare for plotting
        input_data = optree.tree_map(lambda x: x.detach().cpu(), input_data, namespace="fimsde")

        if dimension == 1:
            vf_plotting_func = plot_1d_vf_real_and_estimation
        elif dimension == 2:
            vf_plotting_func = plot_2d_vf_real_and_estimation
        elif dimension == 3:
            vf_plotting_func = plot_3d_vf_real_and_estimation
        else:

            def vf_plotting_func(*args, **kwargs):
                return None

        data, estimated_concepts, sample_paths, sample_grid = input_data

        fig_vf = vf_plotting_func(
            data.get("locations"),
            data.get("drift_at_locations"),
            estimated_concepts.drift,
            data.get("diffusion_at_locations"),
            estimated_concepts.diffusion,
            show=False,
        )

        # select paths to plot
        P = data["obs_times"].shape[0]
        perm = torch.randperm(P)
        plot_paths_count = P if plot_paths_count is None else min(plot_paths_count, P)

        fig_paths = plot_paths(
            dimension,
            data["obs_times"][perm][:plot_paths_count],
            data["obs_values"][perm][:plot_paths_count],
            sample_paths[perm][:plot_paths_count],
            sample_grid[perm][:plot_paths_count],
        )

        return fig_vf, fig_paths

    @torch.no_grad()
    def __call__(self, epoch: int) -> dict:
        """
        Plot ground-truth and estimated vector fields and paths for all available dimensions.
        """
        self.model.eval()

        if epoch % self.plot_frequency != 0:
            return {}

        else:
            # find example for all dimensions
            example_batch: dict = next(iter(self.dataloader))
            max_dim = example_batch["obs_values"].shape[-1]

            all_dims: tuple[int] = []
            batch_with_all_dims: tuple[dict] = []

            for dim in range(1, max_dim + 1):
                for batch in iter(self.dataloader):
                    batch = optree.tree_map(lambda x: x.to(self.model.device), batch, namespace="sde")

                    dim_in_batch = (batch["dimension_mask"].sum(dim=-1) == dim)[:, 0]  # [B]

                    # select first element in batch with dim
                    if dim_in_batch.any().item() is True:
                        batch_with_dim: dict = optree.tree_map(lambda x: x[dim_in_batch][0], batch)
                        batch_with_all_dims.append(batch_with_dim)
                        all_dims.append(dim)

                        break

            # combine found elements to a single dict that can be evaluated by pipeline
            batch_with_all_dims: dict = optree.tree_map(lambda *x: torch.stack(x, dim=0), *batch_with_all_dims)

            # get concepts and samples from batch_with_all_dims
            with torch.no_grad():
                estimated_concepts: SDEConcepts = self.model(batch_with_all_dims, training=False)
                sample_paths, sample_paths_grid = fimsde_sample_paths(
                    self.model, batch_with_all_dims, grid=batch_with_all_dims["obs_times"]
                )

            # Create figures from model outputs and samples
            figures = {}
            for dim in all_dims:
                fig_vf, fig_paths = self.plot_example_of_dim(
                    batch_with_all_dims,
                    estimated_concepts,
                    sample_paths,
                    sample_paths_grid,
                    dimension=dim,
                    plot_paths_count=self.plot_paths_count,
                )

                dim_figures = {f"Vector_Field_{str(dim)}D": fig_vf, f"Paths_{str(dim)}D": fig_paths}
                figures.update(dim_figures)

            return {"figures": figures}
