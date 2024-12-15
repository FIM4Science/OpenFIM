from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import torch

from fim.data.dataloaders import BaseDataLoader
from fim.models.blocks import AModel
from fim.trainers.utils import TrainLossTracker


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
