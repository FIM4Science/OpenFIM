import copy
import logging
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from peft import LoraConfig, PeftConfig
from transformers import PreTrainedModel

from fim.utils.helper import load_yaml

from ..utils.logging import RankLoggerAdapter


logger = RankLoggerAdapter(logging.getLogger("__main__"))


def get_peft_config(config: dict) -> PeftConfig:
    config = copy.deepcopy(config)
    config.pop("method")
    peft_config = LoraConfig(**config)
    return peft_config


def get_peft_trainable_parameters(model):
    """
    Gets the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    return f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"


def add_peft_adapter(model: PreTrainedModel, config: dict, adapter_name: str = None):
    adapter_config = get_peft_config(config)

    model.add_adapter(adapter_config, adapter_name)

    logger.info("Added PEFT addapter `%s` to model!", adapter_name)
    logger.info(get_peft_trainable_parameters(model))


def freeze_transformer_layers(model: nn.Module, num_layers: int = 0):
    """Freeze the layers of a model.

    Args:
        model (nn.Model): which layers we want to freeze.
        num_layer (int): the first `num_layers` will be frozen.
    """
    if num_layers == 0:
        return
    for i, layer in enumerate(model.model.layers):
        if num_layers == -1 or i < num_layers:
            for param in layer.parameters():
                param.requires_grad = False


class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)


def rk4(
    super_fine_grid_grid: torch.Tensor,
    super_fine_grid_drift: torch.Tensor,
    initial_condition: torch.Tensor,
):
    r"""
    Solve ODE using Runge-Kutta 4th order method.

    Args:
        super_fine_grid_grid (torch.Tensor): grid time points: fine grid and one point in between each fine grid point. Shape: [B*D, 2L-1,1]
        super_fine_grid_drift (torch.Tensor): Drift tensor at super_fine_grid points. Shape: [B, 2L-1, D]
        initial_conditions (torch.Tensor): Initial conditions. Shape: [B, D]

    Returns:
        solution (torch.Tensor): Solution at fine grid points. Shape: [B, L, D]
            (at position i, j: x_0^(i)+\sum_{k=1}^{j} increments^(i)_k)
             with increments^(i)_k = h/3*(k1^(i)_k+4*k2^(i)_k+k3^(i)_k)
             and k1^(i)_k = drift^(i)(t_k), k2^(i)_k = drift^(i)(t_k+h/2), k3^(i)_k = drift^(i)(t_k+h))
    """

    def get_rk4_increments(super_fine_grid_grid: torch.Tensor, super_fine_grid_drift: torch.Tensor):
        """
        Calculate solution increments of the Runge-Kutta method based on drift terms provided at the grid.

        Args:
            super_fine_grid_grid (torch.Tensor): grid time points: fine grid and one point in between each fine grid point. Shape: [B, 2L-1,1]
            super_fine_grid_drift (torch.Tensor): Drift tensor at super_fine_grid points. Shape: [B, 2L-1, D]

        Returns:
            increments (torch.Tensor): Increments of the Runge-Kutta 4th order method. Shape: [B, L-1, D]
        """

        B, LL, D = super_fine_grid_drift.shape  # LL=2L-1
        # reshape drift & grid to [B*D, LL]
        super_fine_grid_drift = super_fine_grid_drift.reshape(B * D, LL)

        super_fine_grid_grid = super_fine_grid_grid.repeat(D, 1, 1).reshape(B * D, LL)

        # for each step want drift at start, intermediate and end point of step
        # get drift at start and intermediate point (in last dim of drift tensor)
        drift = super_fine_grid_drift[..., :-1].reshape(B * D, -1, 2)  # [B*D, L-1, 2]

        # end point of step = start point of next step
        drift_end_of_step = drift[..., 1:, 0]  # [B*D, L-2]
        # concat with drift at end of each sample (final end step)
        drift_final_end_step = super_fine_grid_drift[..., -1].unsqueeze(-1)  # [B*D, 1]
        drift_end_of_step = torch.cat([drift_end_of_step, drift_final_end_step], dim=-1).unsqueeze(-1)  # [B*D, L-1, 1]

        # concatenate drift at start, intermediate and end point of step (in last dim of drift tensor)
        drift = torch.cat([drift, drift_end_of_step], dim=-1)  # [B*D, L-1, 3]

        # reshape grid to get half step size i.e. difference between start and intermediate grid point
        grid = super_fine_grid_grid[..., :-1].reshape(B * D, -1, 2)  # [B*D, L-1, 2]
        half_step_size = grid[..., 1] - grid[..., 0]  # [B*D, L-1]

        # calculate increments: half_step_size * 1/3 * (drift at start + 4*drift at intermediate + drift at end)
        increments = half_step_size * (1 / 3) * (drift[..., 0] + 4 * drift[..., 1] + drift[..., 2])  # [B*D, L-1]

        # reshape to [B, L-1, D]
        increments = increments.reshape(B, -1, D)

        return increments

    increments = get_rk4_increments(super_fine_grid_grid, super_fine_grid_drift)  # [B, L-1, D]
    if initial_condition.dim() == 2:
        initial_condition = initial_condition.unsqueeze(1)
    # concat with initial condition to get summands
    summands = torch.cat([initial_condition, increments], dim=1)  # [B, L, D]
    # calculate solution by cumulative sum over summands
    solution = summands.cumsum(dim=1)  # [B, L, D]

    return solution


def load_model_from_checkpoint(checkpoint_path: Union[str, Path], module: nn.Module, for_eval: bool = True) -> nn.Module:
    """
    Load a model from a checkpoint.

    Args:
        checkpoint_path (Union[str, Path]): Path to the checkpoint. Expects key `model_state`. Further expects a `train_parameters.yaml` file two directories above the checkpoint.
        module (nn.Module): Module to load. e.g `fim.models.FIMODE`.
        for_eval (bool): Whether to set the model to eval mode.

    Returns:
        nn.Module: The loaded model.
    """
    # check in same dir as checkpoint for train_parameters.yaml
    params_dict_dir = Path(checkpoint_path).parent / "train_parameters.yaml"
    if not params_dict_dir.exists():
        # check two directories above
        params_dict_dir = (Path(checkpoint_path).parent / "../../train_parameters.yaml").resolve()
        if not params_dict_dir.exists():
            raise FileNotFoundError(f"Could not find train_parameters.yaml in {params_dict_dir} nor in {Path(checkpoint_path).parent}")
    params_dict = load_yaml(params_dict_dir)
    model_params = params_dict.get("model")

    if (model_name := model_params.pop("name")) != "FIMODE" and model_name != "FIM_imputation":
        logger.warning("Not tested for anything but FIMODE and FIMImputation!")

    model = module(**model_params)

    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    model.load_state_dict(checkpoint["model_state"])
    logger.warn(f"Model loaded. last epoch: {checkpoint['last_epoch']}")

    if for_eval:
        # Ensure all parameters of fim_model do not require gradients
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

    return model


def get_off_diagonal_elements(matrix: torch.Tensor) -> torch.Tensor:
    """
    Get the off-diagonal elements of a square matrix.

    Args:
        matrix (torch.Tensor): Square matrix.

    Returns:
        torch.Tensor: Off-diagonal elements of the matrix.
    """
    assert matrix.size(-1) == matrix.size(-2), "The last two dimensions of the matrix must be square."
    *batch_dims, n, _ = matrix.shape
    eye = torch.eye(n, dtype=bool, device=matrix.device).logical_not()
    off_diagonal_indices = torch.nonzero(eye, as_tuple=True)
    return matrix[..., off_diagonal_indices[0], off_diagonal_indices[1]].reshape(*batch_dims, -1)


def create_matrix_from_off_diagonal(off_diagonal_elements: torch.Tensor, size: int, diagonal_value: float = 0.0) -> torch.Tensor:
    """
    Create a square matrix from its off-diagonal elements with a fixed value on the diagonal.

    Args:
        off_diagonal_elements (torch.Tensor): Flattened off-diagonal elements of the matrix.
        size (int): Size of the square matrix.
        diagonal_value (float): Value to set on the diagonal elements.

    Returns:
        torch.Tensor: The reconstructed square matrix.
    """
    assert off_diagonal_elements.size(-1) == size * (size - 1), "Number of off-diagonal elements does not match the expected size."

    *batch_dims, _ = off_diagonal_elements.shape
    matrix = torch.full((*batch_dims, size, size), diagonal_value, dtype=off_diagonal_elements.dtype, device=off_diagonal_elements.device)
    eye = torch.eye(size, dtype=bool, device=matrix.device).logical_not()
    matrix[..., eye] = off_diagonal_elements

    return matrix