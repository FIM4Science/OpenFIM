from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import optree
import torch

from fim.data.dataloaders import BaseDataLoader
from fim.data.datasets import FIMSDEDatabatchTuple
from fim.models.blocks import AModel
from fim.pipelines.sde_pipelines import FIMSDEPipeline
from fim.trainers.utils import TrainLossTracker
from fim.utils.plots.sde_estimation_plots import images_log_1D, images_log_2D, images_log_3D


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
        iterator_name: str = kwargs.get("iterator_name", "validation")
        if iterator_name == "validation":
            self.dataloader = dataloader.validation_it
        elif iterator_name == "test":
            self.dataloader = dataloader.test_it
        elif iterator_name == "evaluation":
            self.dataloader = dataloader.evaluation_it

        # how often to plot
        self.plot_frequency: int = kwargs.get("plot_frequency", 1)

    def __call__(self, epoch: int) -> dict:
        """
        Plot ground-truth and estimated vector fields and paths for all available dimensions.
        """

        if epoch % self.plot_frequency != 0:
            return {}

        else:
            # find example for all dimensions
            example_batch: FIMSDEDatabatchTuple = next(iter(self.dataloader))
            max_dim = example_batch.obs_values.shape[-1]

            all_dims: tuple[int] = []
            batch_with_all_dims: tuple[FIMSDEDatabatchTuple] = []

            for dim in range(1, max_dim + 1):
                for batch in self.dataloader:
                    dim_in_batch = (batch.dimension_mask.sum(dim=-1) == dim)[:, 0]  # [B]

                    # select first element in batch with dim
                    if dim_in_batch.any().item() is True:
                        batch_with_dim: FIMSDEDatabatchTuple = optree.tree_map(lambda x: x[dim_in_batch][0], batch)
                        batch_with_all_dims.append(batch_with_dim)
                        all_dims.append(dim)

                        break

            # combine found elements to a single FIMSDEDatabatchTuple that can be evaluated by pipeline
            batch_with_all_dims: FIMSDEDatabatchTuple = optree.tree_map(lambda *x: torch.stack(x, dim=0), *batch_with_all_dims)

            # For now use pipeline to generate samples
            pipeline = FIMSDEPipeline(self.model)
            pipeline_output = pipeline(batch_with_all_dims)

            # Create figures from model outputs and samples
            figures = {}
            for dim in all_dims:
                if dim == 1:
                    fig_vf, fig_paths = images_log_1D(batch_with_all_dims, pipeline_output)
                elif dim == 2:
                    fig_vf, fig_paths = images_log_2D(batch_with_all_dims, pipeline_output)
                elif dim == 3:
                    fig_vf, fig_paths = images_log_3D(batch_with_all_dims, pipeline_output)

                dim_figures = {f"Vector_Field_{str(dim)}D": fig_vf, f"Paths_{str(dim)}D": fig_paths}
                figures.update(dim_figures)

            return {"figures": figures}
