import json
import logging
import os
import re
import shutil
import sys
from functools import partial
from pathlib import Path
from typing import Literal, Union

import torch
import torch.distributed as dist
from dotenv import load_dotenv
from torch.cuda.amp import GradScaler
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig

from ..models.blocks import AModel
from ..trainers.utils import (
    TrainLogging,
    TrainLossTracker,
    broadcast_state_dict,
    is_distributed,
)
from ..utils.git import latest_commit
from ..utils.helper import GenericConfig
from ..utils.logging import RankLoggerAdapter


logger = RankLoggerAdapter(logging.getLogger(__name__))
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
optim_save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)


load_dotenv()


class TrainCheckpoint:
    """
    Helper class for saving model checkpoints, optimizers, and managing best model tracking.

    Args:
        experiment_dir (Union[Path, str]): Directory where experiment data is stored.
        train_config (GenericConfig): Configuration object containing training parameters.

    Attributes:
        logger: Logger instance for printing status messages.
        checkpoint_dir (Path): Directory for storing checkpoints.
        best_model_flag (dict): Dictionary to track best model performance metrics.
        train_config (GenericConfig): Configuration object containing training parameters.

    Methods:
        save_checkpoint(epoch, model, optimizers, schedulers): Saves model, optimizers, and schedulers as checkpoint.
        check_and_save_best_model(epoch, model, optimizers, schedulers, train_stats, validation_stats): Checks if current
            model is better than previous best and saves if necessary.
    """

    def __init__(
        self,
        experiment_dir: Path | str,
        train_config: GenericConfig,
        model: AModel,
        optimizers: dict,
        schedulers: dict,
        training_logger: TrainLogging,
        train_loss_tracker: TrainLossTracker,
        validation_loss_tracker: TrainLossTracker,
        evaluation_loss_tracker: TrainLossTracker,
        grad_scaler: GradScaler = None,
        rank: int = 0,
        is_peft: bool = False,
        resume_dir: Path | str = None,
        hub_model_id: str = None,
    ):
        """
        Initializes the TrainCheckpoint instance.

        Args:
            experiment_dir (Union[Path, str]): Directory where experiment data is stored.
            train_config (GenericConfig): Configuration object containing training parameters.
        """
        self.rank = rank
        self.train_config = train_config
        self.__logger = RankLoggerAdapter(logging.getLogger(self.__class__.__name__))

        self.model = model
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.grad_scaler = grad_scaler
        self.hub_model_id = hub_model_id
        self.best_model_flag = {
            "train_loss": torch.tensor(float("inf")),
            "val_loss": torch.tensor(float("inf")),
            "train_metric": torch.tensor(float("inf")),
            "val_metric": torch.tensor(float("inf")),
        }
        self.checkpoint_dir = Path(experiment_dir) / "checkpoints"
        self.best_model_dir = self.checkpoint_dir / "best-model"
        self.is_peft = is_peft
        self.resume_dir = resume_dir
        if self.rank != 0:
            return
        self.__logger.info("Initializing Checkpoint Directories ...")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.training_logger = training_logger
        self.train_loss_tracker = train_loss_tracker
        self.validation_loss_tracker = validation_loss_tracker
        self.evaluation_loss_tracker = evaluation_loss_tracker

    def save_checkpoint(self, epoch: int, train_stats: dict, validation_stats: dict) -> None:
        """
        Saves the current model, optimizers, and schedulers as a checkpoint.

        Args:
            epoch (int): Current epoch of the training process.
        """
        is_best_model = False
        best_metric = self.train_config.trainer.best_metric
        current_metric_value = validation_stats["losses"][best_metric]
        if isinstance(current_metric_value, torch.Tensor):
            current_metric_value = current_metric_value.item()

        if current_metric_value < self.best_model_flag["val_metric"]:
            is_best_model = True
            if self.rank == 0:
                msg = f"Current model with {best_metric} of {current_metric_value:0.4f} has better performance than the current best model with {best_metric} of {self.best_model_flag['val_metric']:0.4f}"
                self.__logger.info(msg)
            self._update_best_model_flag(train_stats, validation_stats)

        if (epoch + 1) % self.train_config.trainer.save_every != 0:
            if is_best_model:
                self._save_best_model(epoch)
            return
        if is_distributed():
            dist.barrier()
        save_dir = None
        if self.rank == 0:
            self.__logger.info("Creating Checkpointing Directory for Epoch %s", epoch)
            save_dir = self.checkpoint_dir / f"epoch-{epoch}"
            save_dir.mkdir(exist_ok=True)
        self._save_model_state(epoch, save_dir)
        self._save_optimizers_state(epoch, save_dir)
        if is_distributed():
            dist.barrier()
        if self.rank == 0 and is_best_model:
            self.__logger.info("Saving Best Model ...")
            shutil.copytree(save_dir, self.best_model_dir, dirs_exist_ok=True)

    def check_and_save_best_model(self, epoch: int, train_stats: dict, validation_stats: dict) -> None:
        """
        Checks if the current model performance is better than the previous best and saves if necessary.

        Args:
            epoch (int): Current epoch of the training process.
            train_stats (dict): Training statistics.
            validation_stats (dict): Validation statistics.
        """
        best_metric = self.train_config.trainer.best_metric
        current_metric_value = validation_stats["losses"][best_metric]
        if isinstance(current_metric_value, torch.Tensor):
            current_metric_value = current_metric_value.item()

        if current_metric_value < self.best_model_flag["val_metric"]:
            if self.rank == 0:
                msg = f"Current model with {best_metric} of {current_metric_value:0.4f} has better performance than the current best model with {best_metric} of {self.best_model_flag['val_metric']:0.4f}"
                self.__logger.info(msg)

            self._save_best_model(epoch)
            self._update_best_model_flag(train_stats, validation_stats)

    def _save_model_state(self, epoch: int, save_dir: Union[Path, str], push_to_hub: bool = False):
        """
        Saves the model's state as part of the checkpoint.

        Args:
            epoch (int): Current epoch of the training process.
            save_dir (Union[Path, str]): Directory where the model checkpoint will be saved.
        """
        # TODO: The state of the schedulers is not saved
        train_state = save_dir / "train-state-checkpoint.pth"
        self.__logger.info("Saving Model State: %s ...", save_dir)

        # FIX: Ensure model_type is preserved in config before saving
        if hasattr(self.model.config, "model_type") and self.model.config.model_type:
            original_model_type = self.model.config.model_type
        else:
            # Fallback: extract from class name or training config
            original_model_type = getattr(
                self.model.config, "model_type", self.train_config.model.get("model_type", self.model.__class__.__name__.lower())
            )

        try:
            # Try standard save_pretrained first
            self.model.save_pretrained(save_directory=save_dir)
        except Exception as e:
            self.__logger.warning(f"save_pretrained failed: {e}")
            self.__logger.info("Using fallback config saving method...")

            # Fallback: Save config and weights separately (like FSDP version)
            self.model.config.save_pretrained(save_dir)

            # Save model weights as safetensors
            from safetensors.torch import save_file

            save_file(self.model.state_dict(), save_dir / "model.safetensors")

        # Verify and fix config.json if needed
        config_path = Path(save_dir) / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                saved_config = json.load(f)

            # Ensure model_type is present
            if "model_type" not in saved_config:
                self.__logger.info(f"Adding missing model_type '{original_model_type}' to saved config")
                saved_config["model_type"] = original_model_type
                with open(config_path, "w") as f:
                    json.dump(saved_config, f, indent=2)

        if push_to_hub and self.hub_model_id:
            self.__logger.info(f"Pushing Best Model to Huggingface Hub in Repo: {self.hub_model_id}")
            self.model.push_to_hub(self.hub_model_id, private=True, token=os.getenv("HF_TOKEN", None))
        # Save model state dict for checkpoint loading
        torch.save(self.model.state_dict(), Path(save_dir) / "model-checkpoint.pth")
        state = {
            "model_type": original_model_type,
            "last_epoch": epoch,
            "params": self.train_config.to_dict(),
            "checkpointer_state": self.state_dict(),
            "training_logger": self.training_logger.state_dict(),
            "loss_trackers": {
                "training": self.train_loss_tracker.state_dict(),
                "validation": self.validation_loss_tracker.state_dict(),
                "evaluation": self.evaluation_loss_tracker.state_dict(),
            },
            "commit": latest_commit(),
        }
        torch.save(state, train_state)

    def _save_optimizers_state(self, epoch: int, save_dir: Union[Path, str]):
        """
        Saves the state of optimizers and schedulers as part of the checkpoint.

        Args:
            epoch (int): Current epoch of the training process.
            save_dir (Union[Path, str]): Directory where the optimizer checkpoint will be saved.
        """
        file_name = save_dir / "optimizers-checkpoint.pth"
        self.__logger.info("Saving Optimimizers State: %s ...", file_name)
        state = {
            "commit": latest_commit(),
            "last_epoch": epoch,
            "grad_scaler": None if self.grad_scaler is None else self.grad_scaler.state_dict(),
        }
        for name, optimizer in self.optimizers.items():
            schedulers_state = (
                [scheduler.state_dict() for _, scheduler in optimizer["schedulers"]] if optimizer["schedulers"] is not None else None
            )
            state[name] = {"opt": optimizer["opt"].state_dict(), "schedulers": schedulers_state}
        torch.save(state, file_name)

    def _load_optimizers_state(self, load_dir: Union[Path, str]):
        """
        Loads the state of optimizers from a checkpoint.

        Args:
            load_dir (Union[Path, str]): Directory where the optimizer checkpoint is saved.
        """
        file_name = load_dir / "optimizers-checkpoint.pth"
        if not file_name.exists():
            self.__logger.warning("Optimizer checkpoint file %s not found. Skipping optimizer state loading.", file_name)
            return

        self.__logger.info("Loading Optimizers State from %s ...", file_name)
        checkpoint = torch.load(file_name, map_location=torch.device("cpu"))  # Load checkpoint on CPU
        if self.grad_scaler is not None and checkpoint["grad_scaler"] is not None:
            self.grad_scaler.load_state_dict(checkpoint["grad_scaler"])
        for name, optimizer in self.optimizers.items():
            if name in checkpoint:
                optimizer["opt"].load_state_dict(checkpoint[name]["opt"])
                if optimizer["schedulers"] is not None:
                    for ix, (_, scheduler) in enumerate(optimizer["schedulers"]):
                        scheduler.load_state_dict(checkpoint[name]["schedulers"][ix])
                self.__logger.info("Loaded optimizer state for %s.", name)
            else:
                self.__logger.warning("Optimizer state for %s not found in checkpoint.", name)

    def _save_best_model(self, epoch: int):
        """
        Saves the best model based on validation performance.

        Args:
            epoch (int): Current epoch of the training process.
        """
        best_model_dir = None
        if is_distributed():
            dist.barrier()
        if self.rank == 0:
            best_model_dir = self.best_model_dir
            best_model_dir.mkdir(exist_ok=True)
            self.__logger.info("Saving Best Model ...")
        self._save_model_state(epoch, best_model_dir, True)
        self._save_optimizers_state(epoch, best_model_dir)

        self.__logger.info("Best Model Saved!")

    def _update_best_model_flag(self, train_stats: dict, validation_stats: dict) -> None:
        """
        Updates the best model flag with current performance metrics.

        Args:
            train_stats (dict): Training statistics.
            validation_stats (dict): Validation statistics.
        """
        best_metric = self.train_config.trainer.best_metric

        # Convert tensor values to scalars before storing
        train_loss = train_stats["losses"]["loss"]
        if isinstance(train_loss, torch.Tensor):
            train_loss = train_loss.item()

        val_loss = validation_stats["losses"]["loss"]
        if isinstance(val_loss, torch.Tensor):
            val_loss = val_loss.item()

        train_metric = train_stats["losses"][best_metric]
        if isinstance(train_metric, torch.Tensor):
            train_metric = train_metric.item()

        val_metric = validation_stats["losses"][best_metric]
        if isinstance(val_metric, torch.Tensor):
            val_metric = val_metric.item()

        self.best_model_flag["train_loss"] = train_loss
        self.best_model_flag["val_loss"] = val_loss
        self.best_model_flag["train_metric"] = train_metric
        self.best_model_flag["val_metric"] = val_metric

    def load_checkpoint(self, checkpoint: Union[int, Literal["best-model", "last-epoch"]]) -> int:
        """
        Loads a checkpoint for the trainer.

        Args:
            checkpoint (Union[int, Literal["best-model", "last-epoch"]]):
                The checkpoint to load.
                - If an integer is provided, it represents a specific epoch checkpoint.
                - If "best-model" is provided, it loads the best model checkpoint.
                - If "last-epoch" is provided, it loads the checkpoint from the last epoch.

        Returns:
            None
        """
        if not isinstance(checkpoint, (int, str)) or (isinstance(checkpoint, str) and checkpoint not in {"best-model", "last-epoch"}):
            raise ValueError(
                f"Invalid checkpoint value: {checkpoint}. Supported values are 'best-model', 'last-epoch', or an integer epoch number."
            )
        try:
            checkpoint_path = self._get_checkpoint_path(checkpoint)
            self.__logger.info("Loading Checkpoint: %s ...", checkpoint_path)
            epoch = self._load_model_state(checkpoint_path)
            # self._load_optimizers_state(checkpoint_path)
        except FileNotFoundError as e:
            self.__logger.warning(e)
            if self.is_peft:
                self.__logger.critical("Cannot find chekpoint to resume the training. Start the trainin without '--resume' option!")
                sys.exit(1)
            self.__logger.warning("Starting Training from Scratch")
            return 0

        return epoch

    def _load_model_state(self, checkpoint_path: Path):
        """Load model and training state, with fallbacks for older checkpoint formats."""
        model_ckpt = checkpoint_path / "model-checkpoint.pth"
        train_ckpt = checkpoint_path / "train-state-checkpoint.pth"
        if model_ckpt.exists():
            self.__logger.info("Loading Model State: %s", model_ckpt)
            model_state = torch.load(model_ckpt, weights_only=False)
        else:
            # fallback for safetensors or HF save_pretrained directories
            safetensors_ckpt = checkpoint_path / "model.safetensors"
            if safetensors_ckpt.exists():
                self.__logger.info("Loading Model State from safetensors: %s", safetensors_ckpt)
                from safetensors.torch import load_file as _load_safetensors

                model_state = _load_safetensors(safetensors_ckpt)
            else:
                hf_bin = checkpoint_path / "pytorch_model.bin"
                if hf_bin.exists():
                    self.__logger.info("Loading Model State from Huggingface bin: %s", hf_bin)
                    model_state = torch.load(hf_bin, map_location="cpu")
                else:
                    raise FileNotFoundError(
                        f"No model checkpoint found in {checkpoint_path}; looked for model-checkpoint.pth,"
                        " model.safetensors or pytorch_model.bin"
                    )
        # load training state
        self.__logger.info("Loading Training State: %s", train_ckpt)
        train_state = torch.load(train_ckpt, weights_only=False)

        self.model.load_state_dict(model_state)
        self.training_logger.load_state_dict(train_state["training_logger"])
        self.train_loss_tracker.load_state_dict(train_state["loss_trackers"]["training"])
        self.validation_loss_tracker.load_state_dict(train_state["loss_trackers"]["validation"])
        if "evaluation" in train_state["loss_trackers"]:
            self.evaluation_loss_tracker.load_state_dict(train_state["loss_trackers"]["evaluation"])
        self.load_state_dict(train_state["checkpointer_state"])
        return int(train_state["last_epoch"]) + 1

    def _get_checkpoint_path(self, checkpoint: Union[int, Literal["best-model", "last-epoch"]]) -> Path:
        if checkpoint == "best-model":
            checkpoint_path = self.resume_dir / "best-model"
        elif checkpoint == "last-epoch":
            checkpoint_path = self.__get_last_epoch()
        else:
            checkpoint_path = self.resume_dir / f"epoch-{checkpoint}"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' not found")
        return checkpoint_path

    def __get_last_epoch(self):
        epoch_numbers = []
        for checkpoint_name in [item.name for item in self.resume_dir.iterdir() if item.is_dir()]:
            match = re.match(r"^epoch-(\d+)$", checkpoint_name)
            if match:
                epoch_numbers.append(int(match.group(1)))
        if not epoch_numbers:
            raise FileNotFoundError(f"Checkpoints not found in '{self.resume_dir}' not found")
        return self.resume_dir / f"epoch-{max(epoch_numbers)}"

    def state_dict(self) -> dict:
        return self.best_model_flag

    def load_state_dict(self, state: dict):
        self.best_model_flag = state


class TrainCheckpointFSDPFullStateDict(TrainCheckpoint):
    """
    Helper class for saving model checkpoints, optimizers, and managing best model tracking in a FSDP setting.

    Args:
        experiment_dir (Union[Path, str]): Directory where experiment data is stored.
        train_config (GenericConfig): Configuration object containing training parameters.

    Attributes:
        logger: Logger instance for printing status messages.
        checkpoint_dir (Path): Directory for storing checkpoints.
        best_model_flag (dict): Dictionary to track best model performance metrics.
        train_config (GenericConfig): Configuration object containing training parameters.

    Methods:
        save_checkpoint(epoch, model, optimizers, schedulers): Saves model, optimizers, and schedulers as checkpoint.
        check_and_save_best_model(epoch, model, optimizers, schedulers, train_stats, validation_stats): Checks if current
            model is better than previous best and saves if necessary.
    """

    def __init__(self, **kwargs):
        self.__logger = RankLoggerAdapter(logging.getLogger(self.__class__.__name__))
        super().__init__(**kwargs)

    def _save_model_state(self, epoch: int, save_dir: Path | str):
        """
        Saves the model's state as part of the checkpoint.

        Args:
            epoch (int): Current epoch of the training process.
            save_dir (Union[Path, str]): Directory where the model checkpoint will be saved.
        """

        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, fullstate_save_policy):
            self.__logger.info("Getting Model State dict to rank 0 ...")
            model_state = self.model.state_dict()
            self.__logger.info("Model State dict transferred to rank 0")

        if self.rank == 0:
            train_state_path = save_dir / "train-state-checkpoint.pth"
            self.__logger.info("Saving Model State: %s ...", save_dir)
            model_type = type(self.model).__name__
            state = {
                "model_type": model_type,
                "last_epoch": epoch,
                "params": self.train_config.to_dict(),
                "checkpointer_state": self.state_dict(),
                "training_logger": self.training_logger.state_dict(),
                "loss_trackers": {
                    "training": self.train_loss_tracker.state_dict(),
                    "validation": self.validation_loss_tracker.state_dict(),
                    "evaluation": self.evaluation_loss_tracker.state_dict(),
                },
                "commit": latest_commit(),
            }
            self.model.config.save_pretrained(save_dir)
            torch.save(model_state, save_dir / "model-checkpoint.pth")
            torch.save(state, train_state_path)
            self.__logger.info("Done Saving Model State: %s ...", save_dir)

    def _save_optimizers_state(self, epoch: int, save_dir: Path | str):
        """
        Saves the state of optimizers and schedulers as part of the checkpoint.

        Args:
            epoch (int): Current epoch of the training process.
            save_dir (Union[Path, str]): Directory where the optimizer checkpoint will be saved.
        """

        state = {
            "commit": latest_commit(),
            "last_epoch": epoch,
            "grad_scaler": None if self.grad_scaler is None else self.grad_scaler.state_dict(),
        }
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, optim_state_dict_config=optim_save_policy):
            for name, optimizer in self.optimizers.items():
                self.__logger.info("Getting Optimizer State: %s", name)
                optim_state = FSDP.optim_state_dict(self.model, optimizer["opt"])
                if self.rank == 0:
                    schedulers_state = (
                        [scheduler.state_dict() for _, scheduler in optimizer["schedulers"]]
                        if optimizer["schedulers"] is not None
                        else None
                    )
                    state[name] = {"opt": optim_state, "schedulers": schedulers_state}
        if self.rank == 0:
            file_name = save_dir / "optimizers-checkpoint.pth"
            self.__logger.info("Saving Optimimizers State: %s ...", file_name)
            torch.save(state, file_name)
            self.__logger.info("Done Saving Optimimizers State: %s ...", file_name)

    def load_model_state(self, checkpoint: Union[int, Literal["best-model", "last-epoch"]]) -> int:
        try:
            checkpoint_path = self._get_checkpoint_path(checkpoint)
        except FileNotFoundError as e:
            self.__logger.warning(e)
            self.__logger.warning("Starting Training from Scratch")
            return 0
        last_epoch = None
        checkpointer_state = None
        if self.rank == 0:
            model_checkpoint_path = checkpoint_path / "model-checkpoint.pth"
            train_checkpoint_path = checkpoint_path / "train-state-checkpoint.pth"
            self.__logger.info("Loading Model State: %s ...", model_checkpoint_path)
            model_state = torch.load(model_checkpoint_path, weights_only=False)
            train_checkpoint_state = torch.load(train_checkpoint_path, weights_only=False)

            self.model.load_state_dict(model_state)

            self.training_logger.load_state_dict(train_checkpoint_state["training_logger"])
            self.train_loss_tracker.load_state_dict(train_checkpoint_state["loss_trackers"]["training"])
            self.validation_loss_tracker.load_state_dict(train_checkpoint_state["loss_trackers"]["validation"])
            if "evaluation" in train_checkpoint_state["loss_trackers"].keys():  # old checkpoints might not have evaluation_loss_tracker
                self.evaluation_loss_tracker.load_state_dict(train_checkpoint_state["loss_trackers"]["evaluation"])
            last_epoch = int(train_checkpoint_state["last_epoch"])
            checkpointer_state = train_checkpoint_state["checkpointer_state"]
        dist.barrier()
        last_epoch = broadcast_state_dict(last_epoch, "last_epoch")
        checkpointer_state = broadcast_state_dict(checkpointer_state, "checkpointer_state", True)
        self.load_state_dict(checkpointer_state)
        return last_epoch + 1

    def load_optimizers_state(self, checkpoint: Union[int, Literal["best-model", "last-epoch"]]):
        """
        Loads the state of optimizers from a checkpoint.

        Args:
            load_dir (Union[Path, str]): Directory where the optimizer checkpoint is saved.
        """
        try:
            checkpoint = self._get_checkpoint_path(checkpoint)
        except FileNotFoundError as e:
            self.__logger.warning(e)
            self.__logger.warning("Starting Training from Scratch")
            return
        file_name = checkpoint / "optimizers-checkpoint.pth"
        if not file_name.exists():
            self.__logger.warning("Optimizer checkpoint file %s not found. Skipping optimizer state loading.", file_name)
            return

        checkpoint_ = {}
        grad_scaler_state = None
        if self.rank == 0:
            self.__logger.info("Loading Optimizers State from %s ...", file_name)
            checkpoint_ = torch.load(file_name, weights_only=True)  # Load checkpoint on CPU
            grad_scaler_state = checkpoint_["grad_scaler"]
        grad_scaler_state = broadcast_state_dict(grad_scaler_state, "grad_scaler")
        dist.barrier()
        if self.grad_scaler is not None and grad_scaler_state is not None:
            self.grad_scaler.load_state_dict(grad_scaler_state)
        for name, optimizer in self.optimizers.items():
            schedulers_sd = [None] * len(optimizer["schedulers"]) if optimizer["schedulers"] is not None else None
            full_osd = None
            if name in checkpoint_:
                full_osd = checkpoint_[name]["opt"]
                schedulers_sd = checkpoint_[name]["schedulers"]
            sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, self.model)
            optimizer["opt"].load_state_dict(sharded_osd)
            self.__logger.info("Optimizer %s Loaded!", name)
            if schedulers_sd is not None:
                for ix, scheduler_sd in enumerate(schedulers_sd):
                    scheduler_sd = broadcast_state_dict(scheduler_sd, "scheduler_state_dict")
                    # self.__logger.debug("Received Scheduler state for optimizer '%s': %s!", name, schedulers_sd)
                    optimizer["schedulers"][ix][1].load_state_dict(scheduler_sd)
                self.__logger.info("Optimizer-Schedulers %s Loaded!", name)


non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)


def apply_fsdp_checkpointing(model, check_fn: callable):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    logger.info("Applying FSDP Activation Checkpointing ...")

    apply_activation_checkpointing(model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)
