import gc
import logging
import os
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pylab as plt
import psutil
import torch
import torch.distributed as dist
from peft import PeftModel
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from fim.utils.logging import RankLoggerAdapter


logger = RankLoggerAdapter(logging.getLogger("__main__"))


class TrainLossTracker:
    """
    Class for tracking and calculating losses during training.

    Attributes:
        batch_losses (defaultdict): Dictionary to accumulate batch-level losses.
        batch_losses_counter (defaultdict): Dictionary to keep track of batch loss counts.
        epoch_losses (defaultdict): Dictionary to store epoch-level losses.
    """

    def __init__(self):
        """Initialize a new TrainLossTracker instance."""
        self.batch_losses = defaultdict(float)
        self.batch_losses_counter = defaultdict(int)
        self.batch_histograms = {}
        self.batch_line_plots = []
        self.epoch_losses = defaultdict(list)
        self.epoch_histograms = defaultdict(list)
        self.epoch_line_plots = []
        self.logger = RankLoggerAdapter(logging.getLogger(self.__class__.__name__))
        if is_distributed():
            self.world_size = dist.get_world_size()

    def add_batch_loss(self, name: str, value: float | int):
        """
        Add a batch-level loss value to be accumulated within the epoch.

        Args:
            loss_name (str): The name of the loss.
            loss_value (float or torch.Tensor): The batch-level loss value.

        Raises:
            ValueError: If loss_value is not a valid numeric type.
        """
        if not isinstance(value, (int, float, torch.Tensor)):
            raise ValueError(f"Invalid loss value for '{name}': {value}")
        self.batch_losses[name] += value
        self.batch_losses_counter[name] += 1

    def add_batch_losses(self, losses: dict):
        """
        Add a dictionary of batch-level loss values.

        Args:
            losses (dict): A dictionary of loss names and their corresponding values.
        """
        for loss_name, loss_value in losses.items():
            self.add_batch_loss(loss_name, loss_value)

    def add_batch_histogram(self, name: str, value: torch.Tensor):
        """
        Add a batch-level loss value to be accumulated within the epoch.

        Args:
            loss_name (str): The name of the loss.
            loss_value (float or torch.Tensor): The batch-level loss value.

        Raises:
            ValueError: If loss_value is not a valid numeric type.
        """
        value = torch.sum(value, dim=list(range(value.dim() - 1)))
        if name in self.batch_histograms:
            self.batch_histograms[name] += value
        else:
            self.batch_histograms[name] = value

    def add_batch_histograms(self, histograms: dict):
        """
        Add a dictionary of batch-level loss values.

        Args:
            losses_dict (dict): A dictionary of loss names and their corresponding values.
        """
        for name, value in histograms.items():
            self.add_batch_histogram(name, value)

    def add_batch_line_plots(self, line_plots: dict):
        self.batch_line_plots.append(line_plots)

    def add_batch_stats(self, stats: dict):
        self.add_batch_losses(stats["losses"])
        self.add_batch_histograms(stats["histograms"])
        self.add_batch_line_plots(stats["line_plots"])

    def summarize_epoch(self):
        """
        Calculate and store the average batch loss as epoch loss.
        """
        for name, loss in self.batch_losses.items():
            if is_distributed() and torch.cuda.device_count() > 1:
                try:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                except Exception:
                    self.logger.error("Unable to reduce the '%s' loss from all ranks!", name)
                loss = loss / self.world_size
            avg_loss = loss / self.batch_losses_counter[name]
            self.epoch_losses[name].append(avg_loss)
            self.batch_losses[name] = 0.0
            self.batch_losses_counter[name] = 0
        if "nll-loss" in self.epoch_losses:
            nll_loss = self.epoch_losses["nll-loss"][-1]
            self.epoch_losses["ppl"].append(torch.exp(nll_loss))

        for name, histogram in self.batch_histograms.items():
            if is_distributed() and torch.cuda.device_count() > 1:
                try:
                    dist.all_reduce(histogram, op=dist.ReduceOp.SUM)
                except Exception:
                    self.logger.error("Unable to reduce the '%s' histogram from all ranks!", name)

            self.epoch_histograms[name].append(histogram)
            self.batch_histograms[name] = 0.0

        for line_plot in self.batch_line_plots:
            self.epoch_line_plots.append(line_plot)

    def get_batch_losses(self, loss_name=None):
        """
        Get batch-level losses.

        Args:
            loss_name (str, optional): The name of the specific loss to retrieve. Default is None.

        Returns:
            dict or float: A dictionary of batch losses if loss_name is None, otherwise the specific loss value.
        """
        if loss_name is None:
            return dict(self.batch_losses)
        return self.batch_losses.get(loss_name, 0.0)

    def get_epoch_losses(self, loss_name=None):
        """
        Get epoch-level losses.

        Args:
            loss_name (str, optional): The name of the specific loss to retrieve. Default is None.

        Returns:
            list or float: A list of epoch losses if loss_name is None, otherwise the specific loss values.
        """
        if loss_name is None:
            return dict(self.epoch_losses)
        return self.epoch_losses.get(loss_name, [])

    def get_last_epoch_stats(self) -> dict:
        """
        Get last epoch-level losses.


        Returns:
            dict: A dict of epoch losses where the key is the loss name and the value is the spcific loss value.
        """

        return {
            "losses": dict((k, v[-1]) for k, v in self.epoch_losses.items()),
            "histograms": dict((k, v[-1]) for k, v in self.epoch_histograms.items()),
            "line_plots": self.epoch_line_plots[-1],
        }

    def get_total_batch_loss(self, loss_name):
        """
        Get the total batch-level loss value.

        Args:
            loss_name (str): The name of the loss.

        Returns:
            float or None: The total batch-level loss value, or None if not found.
        """
        return self.batch_losses.get(loss_name, None)

    def get_average_epoch_loss(self, loss_name):
        """
        Calculate the average epoch-level loss value.

        Args:
            loss_name (str): The name of the loss.

        Returns:
            float: The average epoch-level loss value.
        """
        epoch_losses = self.epoch_losses.get(loss_name, [])
        if not epoch_losses:
            return 0.0
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        return avg_loss

    def state_dict(self) -> dict:
        """Returns the dictionary state of the loss tracker."""
        state = {
            "epoch_losses": self.epoch_losses,
        }
        return state

    def load_state_dict(self, state: dict):
        """Load the state of the loss tracker."""
        self.epoch_losses = state["epoch_losses"]

    def __str__(self):
        """
        Generate a string representation of the TrainLossTracker instance.

        Returns:
            str: String containing information about batch-level and epoch-level losses.
        """
        loss_info = "TrainLossTracker Information:\n"

        loss_info += "\nBatch-level Losses:\n"
        for loss_name, loss_value in self.batch_losses.items():
            loss_info += f"{loss_name}: {loss_value}\n"

        loss_info += "\nEpoch-level Losses:\n"
        for loss_name, loss_values in self.epoch_losses.items():
            avg_loss = sum(loss_values) / len(loss_values) if len(loss_values) > 0 else 0.0
            loss_info += f"{loss_name} (Avg): {avg_loss}\n"

        return loss_info


class StepProgressBar:
    """
    A progress bar with enhanced logging capabilities for tracking the progress of a process.

    Args:
        total_steps (int): Total number of steps for the progress bar.
        description (str): Description to be displayed for the progress bar.
        unit (str): Unit of measurement for each step.
        color (str): Color code for the progress bar.
        position (int): Position of the progress bar.
        leave (bool, optional): Whether to leave the progress bar displayed after completion. Default is True.
    """

    def __init__(
        self,
        total_steps: int,
        description: str,
        unit: str,
        color: str,
        position: int,
        leave: bool = True,
        starting_step: int = 0,
        rank: int = 0,
    ):
        if rank == 0:
            self.pbar = tqdm(
                total=total_steps,
                desc=description,
                unit=unit,
                colour=color,
                position=position,
                leave=leave,
                initial=starting_step,
            )
        self.rank = rank

    def update(self, step: int, log_str: Optional[str] = None):
        """
        Update the progress bar with the specified step value and an optional log string.

        Args:
            step (int): Number of steps to update the progress bar.
            log_str (str, optional): Log string to display in the progress bar's postfix. Default is None.
        """
        if self.rank != 0:
            return
        self.pbar.update(step)
        if log_str:
            self.set_postfix(log_str)

    def set_postfix(self, log_str: str):
        """
        Set a log string to be displayed in the progress bar's postfix.

        Args:
            log_str (str): Log string to display in the progress bar's postfix.
        """
        self.pbar.set_postfix_str(log_str)

    def close(self):
        """
        Close the progress bar, indicating the completion of the process.
        """
        if self.rank != 0:
            return
        self.pbar.close()

    def update_and_set_postfix(self, step: int, batch_losses: Dict[str, Union[float, int]], metrics: List[str] = None):
        """
        Update the progress bar with the specified step value and log the batch losses.

        Args:
            step (int): Number of steps to update the progress bar.
            batch_losses (dict): Dictionary containing batch loss values to log.
        """
        self.update(step, self._format_batch_losses(batch_losses, metrics))

    def _format_batch_losses(self, batch_losses: Dict[str, Union[float, int]], metrics: List[str] = None):
        log_str = ""
        for key, value in batch_losses.items():
            if (metrics is None or key in metrics) and not (
                isinstance(value, tuple) or (isinstance(value, torch.Tensor) and len(value.size()) >= 1)
            ):
                log_str += f"{key}: {value.item():4.4g} "
        return log_str


class EpochStepProgressBar(StepProgressBar):
    def __init__(self, total_epochs: int, rank: int, starting_step: int = 0):
        """
        Initialize an epoch step progress bar.

        Args:
            total_epochs (int): Total number of epochs to complete.
            rank (int): Rank of the process.
            leave (bool, optional): Whether to leave the progress bar after completion. Default is True.
        """
        super().__init__(
            total_steps=total_epochs,
            description=f"Rank {rank}, Epoch: ",
            unit="epoch",
            color="green",
            position=0,
            leave=True,
            starting_step=starting_step,
            rank=rank,
        )

    def update_and_set_postfix(
        self,
        step: int,
        train_epoch_losses: Dict[str, Union[float, int]],
        validation_epoch_losses: Dict[str, Union[float, int]],
        metrics: List[str],
    ):
        """
        Update the progress bar with the specified step value and log the batch losses.

        Args:
            step (int): Number of steps to update the progress bar.
            train_epoch_losses (dict): Dictionary containing epoch train loss values to log.
            validation_epoch_losses (dict): Dictionary containing epoch validation loss values to log.
            metrics (List(str)): List containing metrics to be shown on the progress bar.
        """
        train_msg = self._format_batch_losses(train_epoch_losses, metrics)
        validation_msg = self._format_batch_losses(validation_epoch_losses, metrics)
        self.update(step, f"TRAIN: {train_msg} VALIDATION: {validation_msg}")


class TrainStepProgressBar(StepProgressBar):
    """
    A specialized progress bar for training batches.

    Args:
        total_steps (int): Total number of training batches.
        rank (int): Rank of the process.
    """

    def __init__(self, total_steps: int, rank: int):
        super().__init__(
            total_steps=total_steps,
            description=f"Rank {rank}, Training batch: ",
            unit="batch",
            color="blue",
            position=1,
            leave=False,
            rank=rank,
        )


class ValidationStepProgressBar(StepProgressBar):
    """
    A specialized progress bar for validation batches.

    Args:
        total_steps (int): Total number of validation batches.
        rank (int): Rank of the process.
    """

    def __init__(self, total_steps: int, rank: int):
        super().__init__(
            total_steps=total_steps,
            description=f"Rank {rank}, Validation batch: ",
            unit="batch",
            color="yellow",
            position=rank * 2 + 1,
            leave=False,
            rank=rank,
        )


class StepProgressBarFactory:
    """
    A factory for creating instances of specialized progress bars.
    """

    @staticmethod
    def create_train_progress_bar(total_steps: int, rank: int) -> TrainStepProgressBar:
        """
        Create a TrainStepProgressBar instance.

        Args:
            total_steps (int): Total number of training batches.
            rank (int): Rank of the process.

        Returns:
            TrainStepProgressBar: Instance of the TrainStepProgressBar class.
        """
        return TrainStepProgressBar(total_steps, rank)

    @staticmethod
    def create_validation_progress_bar(total_steps: int, rank: int) -> ValidationStepProgressBar:
        """ "
        Create a ValidationStepProgressBar instance.

        Args:
            total_steps (int): Total number of validation batches.
            rank (int): Rank of the process.

        Returns:
            ValidationStepProgressBar: Instance of the ValidationStepProgressBar class.
        """
        return ValidationStepProgressBar(total_steps, rank)

    @staticmethod
    def create_epoch_progress_bar(total_epochs: int, rank: int, starting_step: int = 0) -> EpochStepProgressBar:
        """ "
        Create an instance of EpochStepProgressBar.

        Args:
            total_epochs (int): Total number of epochs.
            rank (int): Rank of the process.

        Returns:
            EpochStepProgressBar: An instance of EpochStepProgressBar.
        """
        return EpochStepProgressBar(total_epochs, rank, starting_step)


class TrainLogging:
    """
    Helper class for managing training logs and tensorboard logging.

    Args:
        experiment_dir (Union[str, Path]): Directory where experiment data is stored.
        logging_fmt (str): Format string for logging messages.

    Attributes:
        __logger: Private logger instance for internal use.
        logging_dir (Path): Directory for storing log files.
        logging_filename (Path): Path to the log file.
        tensorboard_dir (Path): Directory for storing tensorboard logs.
        logger: Logger instance for printing training logs.

    Methods:
        log_train_step(epoch, batch_id, batch_stats): Logs training step details.
        log_validation_step(epoch, batch_id, batch_stats): Logs validation step details.
        log_epoch(epoch, train_stats, validation_stats): Logs summary statistics for an epoch.
    """

    def __init__(self, experiment_dir: Path, logging_fmt: str, rank: int = 0):
        """
        Initializes the TrainLogging instance.

        Args:
            experiment_dir (Union[str, Path]): Directory where experiment data is stored.
            logging_fmt (str): Format string for logging messages.
        """
        self.rank = rank
        self.__logger = RankLoggerAdapter(logging.getLogger(self.__class__.__name__))
        self.__initialize_dirs(experiment_dir)
        self.file_logger = self.__get_train_logger(logging_fmt)
        if self.rank != 0:
            return
        self.__tensorboard_global_step = 0
        self.tensorboard_logger = SummaryWriter(self.tensorboard_dir)

    def __initialize_dirs(self, experiment_dir: Path):
        """
        Initializes logging and tensorboard directories.

        Args:
            experiment_dir (Union[str, Path]): Directory where experiment data is stored.
        """
        self.__logger.info("Initialize Logging Directories ...")
        self.logging_dir = experiment_dir / "logging"
        self.logging_filename = self.logging_dir / "train.log"
        self.tensorboard_dir = self.logging_dir / "tensorboard"

        self.logging_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)

    def __get_train_logger(self, logging_fmt: str):
        """
        Sets up the logger for training logs.

        Args:
            logging_fmt (str): Format string for logging messages.

        Returns:
            Logger: Logger instance for training logs.
        """
        _logger = logging.getLogger("TRAIN-LOGGER")
        _logger.propagate = False
        _logger.setLevel(logging.INFO)
        fh = logging.FileHandler(self.logging_filename)
        formatter = logging.Formatter(logging_fmt)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        _logger.addHandler(fh)
        return RankLoggerAdapter(_logger)

    def log_train_batch(self, epoch: int, batch_id: int, batch_stats: dict) -> None:
        """
        Logs details of a training step.

        Args:
            epoch (int): Current epoch of the training process.
            batch_id (int): Index of the current batch.
            batch_stats (dict): Statistics of the current batch.
        """
        self.__log_batch("TRAIN", epoch, batch_id, batch_stats)

    def __log_batch(self, step_type: str, epoch: int, batch_id: int, batch_stats: dict) -> None:
        """
        Logs details of a training/validation step.

        Args:
            step_type (str): Step type (train/validation).
            epoch (int): Current epoch of the training process.
            batch_id (int): Index of the current batch.
            batch_stats (dict): Statistics of the current batch.
        """
        if "losses" in batch_stats:
            losses = batch_stats.get("losses")
        else:
            losses = batch_stats
        sb = " ".join([f"{k}: {v:.6f}" for k, v in losses.items()])
        self.file_logger.info("Epoch %s - %s - Minibatch %s: %s", epoch, step_type.upper(), batch_id, sb)
        if self.rank == 0:
            self._log_tensorboard("BATCH/" + step_type.upper() + "/", batch_stats)

    def log_epoch(self, epoch: int, train_stats: dict, validation_stats: dict):
        """
        Logs summary statistics for an epoch.

        Args:
            epoch (int): Current epoch of the training process.
            train_stats (dict): Training statistics for the epoch.
            validation_stats (dict): Validation statistics for the epoch.
        """
        if self.rank != 0:
            return

        def generate_log(stats: dict) -> str:
            return " ".join([f"{k}: {v:.6f}" for k, v in stats.items()])

        train_log = generate_log(train_stats["losses"])
        self.file_logger.info("Epoch %d - TRAIN: %s", epoch, train_log)

        validation_log = generate_log(validation_stats["losses"])
        self.file_logger.info("Epoch %d - VALIDATION: %s", epoch, validation_log)

        self._log_tensorboard("EPOCH/TRAIN/", train_stats)
        self._log_tensorboard("EPOCH/VALIDATION/", validation_stats)

    def _log_tensorboard(self, label: str, statistics: dict):
        self._log_tensorboard_scalars(label, statistics["losses"])
        if "histograms" in statistics:
            self._log_tensorboard_histograms(label, statistics["histograms"])
        if "line_plots" in statistics:
            self._log_tensorboard_line_plots(label, statistics["line_plots"])
        self.__tensorboard_global_step += 1

    def _log_tensorboard_scalars(self, label, statistics):
        for k, v in statistics.items():
            # if v is not a scalar: continue
            if v.numel() != 1:
                continue

            self.tensorboard_logger.add_scalar(
                label + k.upper(), v.float(), self.__tensorboard_global_step, new_style=True
            )

    def _log_tensorboard_histograms(self, label, histograms):
        for k, v in histograms.items():
            if k == "paths":
                fig, ax = plt.subplots()
                ax.bar(range(len(v)), v.float().cpu())
                self.tensorboard_logger.add_figure(label + k.upper(), fig, self.__tensorboard_global_step)
            else:
                self.tensorboard_logger.add_histogram(label + k.upper(), v.float(), self.__tensorboard_global_step)

    def _log_tensorboard_line_plots(self, label, line_plot_data):
        """

        Expect data to be detached and on cpu.

        Args:
            label: str: Label for the tensorboard plot
            line_plot_data: dict: Dictionary containing the line plot data for 1 sample
                "observation_times": torch.Tensor: Times for the coarse grid
                "observation_values": torch.Tensor: Sample paths for the coarse grid
                "learnt_solution": torch.Tensor: Learnt solution for the fine grid
                "target_path": torch.Tensor: Target path for the fine grid (sample path)
                "fine_grid_times": torch.Tensor: Times for the fine grid
                "target_drift": torch.Tensor: Target drift for the fine grid
                "learnt_drift": torch.Tensor: Learnt drift for the fine grid
                "learnt_std_drift": torch.Tensor: Learnt std drift for the fine grid = certainty
        """
        assert isinstance(line_plot_data, dict), "line plot data should be a dictionary"
        if len(line_plot_data) == 0:
            return

        # create drift plot
        fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
        axes[0] = axes[0]
        axes[0].plot(line_plot_data["fine_grid_times"], line_plot_data["target_drift"], label="target", color="orange")
        axes[0].plot(line_plot_data["fine_grid_times"], line_plot_data["learnt_drift"], label="inference", color="blue")
        axes[0].fill_between(
            line_plot_data["fine_grid_times"],
            line_plot_data["learnt_drift"] - line_plot_data["learnt_std_drift"],
            line_plot_data["learnt_drift"] + line_plot_data["learnt_std_drift"],
            alpha=0.3,
            color="blue",
            label="certainty",
        )
        axes[0].legend()
        axes[0].set_title("Drift")
        fig.tight_layout()

        axes[1].plot(
            line_plot_data["fine_grid_times"],
            line_plot_data["target_path"],
            label="sample path",
            alpha=0.4,
            color="orange",
        )
        axes[1].scatter(
            line_plot_data["observation_times"],
            line_plot_data["observation_values"],
            label="observations",
            marker="x",
            s=7,
            color="orange",
        )
        # learnt solution
        axes[1].plot(line_plot_data["fine_grid_times"], line_plot_data["learnt_solution"], label="inference", color="blue")
        axes[1].fill_between(
            line_plot_data["fine_grid_times"],
            line_plot_data["learnt_solution"] - line_plot_data["learnt_std_drift"],
            line_plot_data["learnt_solution"] + line_plot_data["learnt_std_drift"],
            alpha=0.3,
            color="blue",
            label="certainty",
        )
        axes[1].legend()
        axes[1].set_title("Solution")
        fig.tight_layout()

        self.tensorboard_logger.add_figure(
            tag=label + "_drift_solution", figure=fig, global_step=self.__tensorboard_global_step
        )

    def add_model_graph(self, model, input: Any) -> None:
        """Writes the model graph in tensorboard.

        Args:
            model (AModel): Model that is logged
            input (Any): Input to the model
        """
        if self.rank == 0:
            for k, v in input.items():
                input[k] = v.to("cuda")
            self.tensorboard_logger.add_graph(model, input, use_strict_trace=False)

    def state_dict(self) -> dict:
        """Get the state of the train logging object."""
        state = {"tensorboard_global_step": self.__tensorboard_global_step}
        return state

    def load_state_dict(self, state: dict):
        """Load the state of the train logging object."""
        self.__tensorboard_global_step = state.get("tensorboard_global_step", 0)
        self.tensorboard_logger = SummaryWriter(self.tensorboard_dir, purge_step=self.__tensorboard_global_step)

    def clear_logging_resources(self) -> None:
        """Frees up resources using for logging."""
        if self.rank != 0:
            return
        self.tensorboard_logger.flush()
        self.tensorboard_logger.close()


def setup():
    dist.init_process_group("nccl")


def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # This flag will help with CUDA memory fragmentations that can lead into OOM in some cases.
    # Note this is only availble in PyTorch Nighlies (as of July 30 2023)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = str(15)
    # if rank == 0:
    #     print("--> Running with torch dist debug set to detail")


def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        logger.info("Clearing GPU cache for all ranks")
    torch.cuda.empty_cache()


def is_distributed() -> bool:
    return dist.is_initialized()


def broadcast_variable(variable: Union[float, int, torch.Tensor]):
    """Broadcasts a variable from rank 0 to all other ranks.

    Args:
        variable (Union[torch.Tensor, Any]): The variable to be broadcast.

    Returns:
        Union[torch.Tensor, Any]: The variable that was received by the current process.
    """

    world_size = dist.get_world_size()
    if world_size == 1:
        return variable

    logger.info("Broadcasting variable %s from rank 0", variable)
    rank = dist.get_rank()
    is_var_tensor = isinstance(variable, torch.Tensor)
    if not is_var_tensor:
        variable = torch.tensor(variable, device=torch.cuda.current_device())
    if rank == 0:
        local_variable = variable
    else:
        local_variable = torch.zeros_like(variable, device=torch.cuda.current_device())
    dist.broadcast(local_variable, src=0)

    return local_variable if is_var_tensor else local_variable.item()


def broadcast_state_dict(state_dict: Optional[dict], state_dict_name: str, move_on_local_gpu: bool = False):
    """
    Broadcasts the state dict from rank 0 to all other ranks.

    Args:
        state_dict (dict): The state dict to be broadcast.
        state_dict_name (str): The name of the state dict.

    Returns:
        dict: The state dict that was received by the current process.
    """

    world_size = dist.get_world_size()
    if world_size == 1:
        logger.info("The world_size is 1, no need to broadcast state dict.")
        return state_dict

    logger.info("Broadcasting state dict {} from rank 0".format(state_dict_name))
    rank = dist.get_rank()

    if rank == 0:
        local_state_dict = [state_dict]
    else:
        local_state_dict = [None]
    dist.broadcast_object_list(local_state_dict, src=0, device=torch.cuda.current_device())
    local_state_dict = local_state_dict[0]
    if move_on_local_gpu:
        for k, v in local_state_dict.items():
            local_state_dict[k] = v.to(torch.cuda.current_device())
        return local_state_dict
    return local_state_dict


def load_peft_pretrained_model(model, path: Path):
    # peft_config = PeftConfig.from_pretrained(path)
    backbone = PeftModel.from_pretrained(model.backbone, path, is_trainable=True)
    model.backbone = backbone
    return model


def byte2gb(x):
    return float(x / 2**30)


# This context manager is used to track the peak memory usage of the process
class GPUMemoryTrace:
    def __init__(self, rank: int = 0):
        gc.collect()
        torch.cuda.empty_cache()
        # torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.reset_peak_memory_stats()  # reset the peak gauge to zero

        self.begin = byte2gb(torch.cuda.memory_allocated())
        self.process = psutil.Process()
        self.cpu_begin = byte2gb(self.cpu_mem_used())
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        self.rank = rank
        self.__logger = RankLoggerAdapter(logging.getLogger(self.__class__.__name__))

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    def print_summary(self):
        self.peak_monitoring = False

        gc.collect()
        torch.cuda.empty_cache()
        end = byte2gb(torch.cuda.memory_allocated())
        peak = byte2gb(torch.cuda.max_memory_allocated())
        cuda_info = torch.cuda.memory_stats()
        peak_active_gb = byte2gb(cuda_info["active_bytes.all.peak"])
        cuda_malloc_retires = cuda_info.get("num_alloc_retries", 0)
        peak_active_gb = byte2gb(cuda_info["active_bytes.all.peak"])
        cuda_info.get("num_ooms", 0)
        byte2gb(end - self.begin)
        byte2gb(peak - self.begin)
        max_reserved = byte2gb(torch.cuda.max_memory_reserved())

        cpu_end = self.cpu_mem_used()
        byte2gb(cpu_end - self.cpu_begin)
        cpu_peaked = byte2gb(self.cpu_peak - self.cpu_begin)

        if self.rank == 0:
            self.__logger.info("Max CUDA memory allocated was %.2f GB", peak)
            self.__logger.info("Max CUDA memory reserved was %.2f GB", max_reserved)
            self.__logger.info("Peak active CUDA memory was %.2f GB", peak_active_gb)
            self.__logger.info("Cuda Malloc retires : %d", cuda_malloc_retires)
            self.__logger.info(
                "CPU Total Peak Memory consumed during the train (max): %d GB", cpu_peaked + self.cpu_begin
            )


class TrainingTimePerformanceTracker:
    def __init__(self, rank: int = 0):
        self.rank = rank
        self.__logger = RankLoggerAdapter(logging.getLogger(self.__class__.__name__))
        self.stage_times = {}
        self.epoch_times = []

    def start_epoch(self):
        self.start_timer("epoch")

    def stop_epoch(self):
        elapsed_time = self.stop_timer("epoch")
        self.epoch_times.append(elapsed_time)

    def start_timer(self, stage_name):
        self.stage_times[stage_name] = time.time()

    def stop_timer(self, stage_name):
        elapsed_time = time.time() - self.stage_times[stage_name]
        self.stage_times[stage_name] = elapsed_time
        return elapsed_time

    def get_elapsed_time(self, stage_name):
        return self.stage_times.get(stage_name, None)

    def print_elapsed_time(self, stage_name):
        elapsed_time = self.get_elapsed_time(stage_name)
        if elapsed_time is not None and self.rank == 0:
            self.__logger.info("Elapsed time for %s: %d seconds", stage_name, elapsed_time)

    def print_epochs_time_statistics(self):
        average_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        min_epoch_time = min(self.epoch_times)
        max_epoch_time = max(self.epoch_times)
        if self.rank == 0:
            self.__logger.info("Average epoch time: %d seconds", average_epoch_time)
            self.__logger.info("Min epoch time: %d seconds", min_epoch_time)
            self.__logger.info("Max epoch time: %d seconds", max_epoch_time)
