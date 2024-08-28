import copy
import os
from abc import abstractmethod
from typing import List, Optional, Union

import numpy as np
import torch
from scipy.signal import savgol_filter
from torch import nn

from fim.utils.helper import create_class_instance

from ..trainers.utils import is_distributed
from .utils import SinActivation


eps = 1e-6


class Block(nn.Module):
    def __init__(self, resume: bool = False, **kwargs):
        super(Block, self).__init__(**kwargs)

        self.resume = resume

    @property
    def device(self):
        if is_distributed():
            return int(os.environ["LOCAL_RANK"])
        return next(self.parameters()).device

    @property
    def rank(self) -> int:
        if is_distributed():
            return int(os.environ["RANK"])
        return 0

    def param_init(self):
        """
        Parameters initialization.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="leaky_relu")
                if module.bias.data is not None:
                    nn.init.zeros_(module.bias)


class Mlp(Block):
    """
    Implement a multi-layer perceptron (MLP) with optional dropout.

    If defined dropout will be applied after each hidden layer but the final hidden and the output layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_layers: List[int],
        hidden_act: nn.Module | dict = nn.ReLU(),
        output_act: nn.Module | dict = None,
        dropout: float = 0.0,
        **kwargs,
    ):
        super(Mlp, self).__init__(**kwargs)

        if isinstance(hidden_act, dict):
            hidden_act = create_class_instance(hidden_act.pop("name"), hidden_act)

        self.layers = nn.Sequential()
        in_size = in_features
        nr_hidden_layers = len(hidden_layers)
        for i, h_size in enumerate(hidden_layers):
            self.layers.add_module(f"linear_{i}", nn.Linear(in_size, h_size))
            self.layers.add_module(f"activation_{i}", hidden_act)
            if dropout != 0 and i < nr_hidden_layers - 1:
                self.layers.add_module(f"dropout_{i}", nn.Dropout(dropout))
            in_size = h_size

        # if no hidden layers are provided, the output layer is directly connected to the input layer
        if len(hidden_layers) == 0:
            hidden_layers = [in_features]
        self.layers.add_module("output", nn.Linear(hidden_layers[-1], out_features))

        if output_act is not None:
            if isinstance(output_act, dict):
                output_act = create_class_instance(output_act.pop("name"), output_act)
            self.layers.add_module("output_activation", output_act)

    def forward(self, x):
        return self.layers(x)


class TimeEncoding(Block):
    """
    Implements the time encoding as described in "Multi-time attention networks for irregularly sampled time series, Shukla & Marlin, 2020".

    Each time point t is encoded as a vector of dimension d_time:
        - first element: linear embedding of t: w_0*t + b_0
        - remaining elements: sinusoidal embedding of t with different frequencies: sin(w_i*t + b_i) for i in {1, ..., d_time-1}
    w_j and b_j are learnable parameters.
    """

    def __init__(self, dim_time: int):
        """
        Args:
            d_time (int): Dimension of the time representation
        """
        super(TimeEncoding, self).__init__()

        self.d_time = dim_time

        self.linear_embedding = nn.Linear(1, 1, bias=True)
        self.periodic_embedding = nn.Sequential(nn.Linear(1, dim_time - 1, bias=True), SinActivation())

    def forward(self, grid: torch.Tensor):
        """
        Args:
            grid (torch.Tensor): Grid of time points, shape (batch_size, seq_len, 1)

        Returns:
            torch.Tensor: Time encoding, shape (batch_size, seq_len, d_time)
        """
        linear = self.linear_embedding(grid)
        periodic = self.periodic_embedding(grid)

        return torch.cat([linear, periodic], dim=-1)


class Transformer(Block):
    """The encoder block of the transformer model as defined in 'Vaswani, A. et al. Attention is all you need'."""

    def __init__(
        self,
        num_encoder_blocks: int,
        dim_model: int,
        dim_time: int,
        num_heads: int,
        dropout: float,
        residual_mlp: dict,
        batch_first: bool = True,
    ):
        super(Transformer, self).__init__()

        self.num_heads = num_heads

        self.input_projection = nn.Linear(dim_time + 1, dim_model)

        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(dim_model, num_heads, dropout, copy.deepcopy(residual_mlp), batch_first=batch_first)
                for _ in range(num_encoder_blocks - 1)
            ]
        )

        self.final_query_vector = nn.Parameter(torch.randn(1, 1, dim_model))
        self.final_attention = nn.MultiheadAttention(dim_model, num_heads, dropout=dropout, batch_first=batch_first)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        """
        Args:
            x (torch.Tensor): Input tensor, shape (batch_size, seq_len, dim_time)
            mask (torch.Tensor): Mask for the input tensor, shape (batch_size, seq_len) with 1 indicating that the time point is masked out. If None, nothing is masked.
        """
        if key_padding_mask is None:
            key_padding_mask = torch.zeros_like(x[:, :, 0])
        elif key_padding_mask.dim() == 3:
            key_padding_mask = key_padding_mask.squeeze(-1)  #  (batch_size, seq_len)#z

        x = self.input_projection(x)  # (batch_size, seq_len, dim_model)

        # pass through encoder blocks
        for encoder_block in self.encoder_blocks:
            x = encoder_block(
                x,
                key_padding_mask,
            )

        # use learnable query vector to get final embedding
        query = self.final_query_vector.repeat(x.size(0), 1, 1)

        attn_output, _ = self.final_attention(
            query,
            x,
            x,
            key_padding_mask=key_padding_mask,
            is_causal=False,
            need_weights=False,
        )  # (batch_size, 1, dim_model)

        return attn_output

    def _pad2attn_mask(self, key_padding_mask, target_seq_len: Optional[int] = None):
        """
        Args:
            key_padding_mask (torch.Tensor): Mask for the input tensor, shape (batch_size, seq_len) with 1 indicating that the time point is masked out.
            target_seq_len (int): Target sequence length. If None, the sequence length is the same as the input sequence length.

        Returns:
            torch.Tensor: Attention mask, float valued, shape (batch_size * num_heads, seq_len, seq_len)
        """
        batch_size, seq_len = key_padding_mask.size()
        if target_seq_len is None:
            target_seq_len = seq_len

        expanded_mask = (
            key_padding_mask.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, target_seq_len, -1)
        )  # Shape: (B, num_heads, seq_len, seq_len)

        # Reshape to (batch_size * num_heads, target_seq_len, seq_len)
        attention_mask = expanded_mask.reshape(batch_size * self.num_heads, target_seq_len, seq_len)

        # Convert boolean mask to float mask (1 -> -inf)
        attention_mask = attention_mask.float().masked_fill(attention_mask, float("-inf"))

        return attention_mask


class EncoderBlock(Block):
    """The encoder block of the transformer model as defined in 'Vaswani, A. et al. Attention is all you need'."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        residual_mlp: dict,
        batch_first: bool = True,
    ):
        super(EncoderBlock, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=batch_first)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.residual_mlp = create_class_instance(
            residual_mlp.pop("name"),
            residual_mlp,
        )
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x, padding_mask):
        # x shape: B, sequence length, d_model, padding_mask shape: B, sequence length, 1
        attn_out, _ = self.self_attn(
            x,
            x,
            x,
            key_padding_mask=padding_mask,
            is_causal=False,
            need_weights=False,
        )
        x = self.layer_norm1(x + attn_out)

        mlp_out = self.residual_mlp(x)
        x = self.layer_norm2(x + mlp_out)

        return x


### Normalization Blocks ###


class BaseNormalization(Block):
    """Base class for normalization. Need to implement
    forward (normalization of a tensor, optionally with observation mask),
    revert_normalization of tensor,
    revert_normalization_derivative (depending on two normalization steps: time and values).
    """

    def __init__(self):
        super(BaseNormalization, self).__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, tuple]:
        raise NotImplementedError

    @abstractmethod
    def revert_normalization(self, x: torch.Tensor, data_concepts: Union[tuple, torch.Tensor]):
        raise NotImplementedError

    @abstractmethod
    def revert_normalization_drift(self, x: torch.Tensor, data_concepts: Union[tuple, torch.Tensor]):
        raise NotImplementedError


class Standardization(BaseNormalization):
    """Standardization block for normalizing input data via mean and std."""

    def __init__(self, mean_target: float = 0.0, std_target: float = 1.0):
        super(Standardization, self).__init__()
        self.mean_target = mean_target
        self.std_target = std_target if std_target != 0 else eps

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, tuple]:
        """
        Change statistics of given data x to target mean and std (Default: 0 and 1) with `X^ = std_target / std_data * (x - mean_data) + mean_target`.

        Standardization is applied along dim 1 (if x.dim()==3) or along dim=(1,2) (if x.dim()==4).

        Args:
            x: (torch.Tensor), shape [B, w, 1] or [B, wc, wlen, 1].
            mask: (torch.Tensor), shape [B, w, 1] or [B, wc, wlen, 1]. 1 indicating that value is masked out. Default: no values masked out.

        Returns:
            normalized_x, (mean_data, var_data): (torch.Tensor, tuple). Normalized data and statistics of original data.
        """
        if mask is None:
            mask = torch.zeros_like(x, dtype=bool)

        x_dim = x.dim()
        if x_dim == 4:
            B, wc, wlen, D = x.shape
            x = x.view(B, wc * wlen, D)
            mask = mask.view(B, wc * wlen, D)

        # invert mask
        mask_inverted = ~mask

        # get masked mean and std per window
        mean_data = ((mask_inverted * x).sum(dim=1) / mask_inverted.sum(dim=1)).unsqueeze(-1)  # shape [B, 1, 1]
        var_data = ((mask_inverted * (x - mean_data) ** 2).sum(dim=1) / mask_inverted.sum(dim=1)).unsqueeze(
            -1
        )  # shape [B, 1, 1]

        normalized_x = (x - mean_data) / torch.sqrt(var_data + eps) * self.std_target + self.mean_target

        # assert not torch.isnan(normalized_x).any(), "Normalization provoked NaN values. Make eps larger?"
        # assert normalized_x.shape == x.shape

        if x_dim == 4:
            normalized_x = normalized_x.view(B, wc, wlen, D)

        return normalized_x, (mean_data, var_data)  # shape [B, T, 1], ([B, 1, 1], [B, 1, 1])

    def revert_normalization(
        self, x: torch.Tensor, data_concepts: Union[tuple, torch.Tensor], log_scale: bool = False
    ) -> torch.Tensor:
        """
        Revert above's standardization using the formula `X_out = std_data / std_target * (X - mean_target) + mean_data` where `X` is the input tensor.

        Args:
            x: torch.Tensor. Either 3 or 4 dimensional.
            data_concepts: mean and var of original data (statistics of data after applying this function).
            log_scale: bool, if True, x is log-scaled.
                (Note: this will always be log_std, hence no additive term is needed)

        Returns:
            x_renormalized: torch.Tensor. Data with given mean and std.
        """
        if isinstance(data_concepts, tuple):
            mean_data, var = data_concepts
        elif isinstance(data_concepts, torch.Tensor) and data_concepts.shape[-1] == 2:
            mean_data, var = data_concepts.split(1, dim=-1)
        else:
            raise ValueError("Wrong format of data concept for reverting the standardization.")

        x_dim = x.dim()
        if x_dim == 4:
            B, wc, wlen, D = x.shape
            x = x.view(B, wc * wlen, D)
        elif x_dim == 2 and mean_data.dim() == 3:
            mean_data = mean_data.squeeze(-1)
            var = var.squeeze(-1)

        if x.dim() != mean_data.dim():
            raise Warning("Data and normalization parameters have different dimensions. Will be broadcasted. Expected?")

        std_data = torch.sqrt(var + eps)
        # assert (var >= 0).any(), f"var: {(var <0).sum()}"
        # assert not torch.isnan(std_data).any()
        # assert not torch.isnan(x).any()

        if not log_scale:
            x_renormalized = std_data / self.std_target * (x - self.mean_target) + mean_data
        else:
            x_renormalized = x + torch.log(std_data / self.std_target)

        if x_dim == 4:
            # need to reshape back to 4 dim
            x_renormalized = x_renormalized.view(B, wc, wlen, D)

        # assert x_renormalized.shape == x.shape

        return x_renormalized

    def get_reversion_factor(self, data_concepts: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Get reversion factor for the standardization.

        Args:
            data_concepts: mean and var of original data.

        Returns:
            torch.Tensor: Reversion factor.
        """
        _, var = data_concepts
        std_data = torch.sqrt(var + eps)

        return std_data / self.std_target

    def __repr__(self):
        return f"Standardization(mean_target={self.mean_target}, std_target={self.std_target})"


class StandardizationSERIN(Standardization):
    """
    Standardization following "Statistics Embedding: Compensate for the Lost Part of Normalization in Time Series Forecasting" by xyz.

    Idea: Fuse normalized data with embedded statistics (learnable embedding).

    Idea: linear combination of "normal" standardization and embedding of statistics (mean and std of input data).
    Revertion of normalization is same as in Standardization (learnable part is ignored).
    """

    def __init__(
        self,
        mean_target: float = 0.0,
        std_target: float = 1.0,
        lin_factor: float = 0.5,
        network: dict = {},
    ):
        std_target = 3
        super(StandardizationSERIN, self).__init__(mean_target, std_target)

        self.linear_factor = torch.tensor(lin_factor)
        mlp = create_class_instance(network.pop("name"), network)
        layer_norm = nn.LayerNorm(network.get("out_features"), elementwise_affine=False)
        self.statistics_embedder = nn.Sequential(mlp, layer_norm)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, tuple]:
        """
        Standardize data by linear combination of Standardization and linearly embedded data statistics.

        Standardization is applied along dim 1 (if x.dim()==3) or along dim=(1,2) (if x.dim()==4).

        Args:
            x: torch.Tensor. Either 3 or 4 dimensional.
            mask: torch.Tensor, optional, default: no values are masked out. 1 indicates that a value is masked out.

        returns:
            standardized x and statistics of original data.
        """
        # x can either be 3 or 4 dimensional we need 3 dim.
        x_dim = x.dim()
        if x_dim == 4:
            B, wc, wlen, D = x.shape
            x = x.view(B, wc * wlen, D)
            mask = mask.view(B, wc * wlen, D)
        standardized_out, statistics = super().forward(x, mask)  # shape [B, w, 1], ([B,1, 1], [B,1, 1])
        embedded_statistics = self.statistics_embedder(torch.concat(statistics, dim=1).squeeze(-1)).unsqueeze(
            -1
        )  # shape [B, w, 1]

        # Hacky solution to make sure that the embedded statistics have the same length as the standardized output
        # TODO: Think about something better
        if embedded_statistics.size(1) > standardized_out.size(1):
            embedded_statistics = embedded_statistics[:, :standardized_out.size(1), :]
        elif embedded_statistics.size(1) < standardized_out.size(1):
            embedded_statistics = embedded_statistics.repeat(1, standardized_out.size(1), 1)

        # fuse normalized x and embedded statistics
        x_fused = (
            torch.sqrt(1 - self.linear_factor) * standardized_out + torch.sqrt(self.linear_factor) * embedded_statistics
        )
        if x_dim == 4:
            # need to reshape
            x_fused = x_fused.view(B, wc, wlen, D)

        return x_fused, statistics

    def __repr__(self):
        return f"StandardizationSERIN(mean_target={self.mean_target}, std_target={self.std_target}, lin_factor={self.linear_factor}, network={self.statistics_embedder})"


class MinMaxNormalization(BaseNormalization):
    """Min-Max scaling block for linearly scaling the data to [0,1]."""

    def __init__(self):
        super(MinMaxNormalization, self).__init__()

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        norm_params: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple]:
        """
        Normalize values using min-max scaling.

        Args:
            data (torch.Tensor): data to normalized [B, T, 1]
            mask (torch.Tensor): observation mask [B, T, 1]
            norm_params (tuple): min and range of the values applied for normalization. ([B, 1], [B, 1])
                 Default: None i.e. min and range are computed from x.

        Returns:
            torch.Tensor: normalized values [B, T, 1]
            tuple: min and range of the values ([B, 1], [B, 1])
        """
        if norm_params is None:
            if mask is None:
                mask = torch.zeros_like(x, dtype=bool)
            min_data, range_data = self._get_norm_params(x, mask)
        else:
            min_data, range_data = norm_params

        # unsqueeze if necessary to allow broadcasting
        if min_data.dim() == 2:
            min_data = min_data.unsqueeze(1)  # Shape [B, 1, 1]
            range_data = range_data.unsqueeze(1)  # Shape [B, 1, 1]

        normalized_data = (x - min_data) / range_data  # Shape [B, T, 1]

        return normalized_data, (min_data, range_data)

    def _get_norm_params(self, data: torch.Tensor, mask: torch.Tensor) -> tuple:
        """
        Compute normalization parameters for min-max scaling (per sample and dimension/feature).

        Args:
            data (torch.Tensor): data to normalize [B, T, D]
            mask (torch.Tensor): observation mask [B, T, 1]. 1 indicates that value is masked out.
        Returns:
            tuple: min (torch.Tensor, [B, D]), range (torch.Tensor, [B, D])
        """
        # get min and max values for each feature dimension per batch entry
        data_min = torch.amin(data.masked_fill(mask, float("inf")), dim=1)  # Shape [B, D]
        data_max = torch.amax(data.masked_fill(mask, float("-inf")), dim=1)  # Shape [B, D]

        # compute range, add small value to avoid division by zero
        data_range = data_max - data_min + 1e-6

        return data_min, data_range

    def revert_normalization(self, x: torch.Tensor, data_concepts: tuple, log_scale: bool = False) -> torch.Tensor:
        """
        Revert min-max normalization.

        Args:
            x: torch.Tensor, normalized data [B, T, 1] or [B, T]
            data_concepts: min and range of original data.
            log_scale: bool, if True, x is log-scaled.
                (Note: this will always be log_std, hence no additive term is needed)

        Returns:
            torch.Tensor: reverted normalized data [B, T, 1] or [B, T]
        """
        min_data, range_data = data_concepts

        x_dim = x.dim()
        if x_dim == 4:
            x = x.squeeze(-1)
        elif x_dim == 2 and min_data.dim() == 3:
            min_data = min_data.squeeze(-1)
            range_data = range_data.squeeze(-1)

        if x.dim() != min_data.dim():
            raise Warning("Data and normalization parameters have different dimensions. Will be broadcasted. Expected?")

        if not log_scale:
            normalized_x = x * range_data + min_data
        else:
            normalized_x = x + torch.log(range_data)

        # assert normalized_x.shape == x.shape

        if x_dim == 4:
            normalized_x = normalized_x.unsqueeze(-1)

        return normalized_x

    # def revert_normalization_drift(
    #     self, x: tuple[torch.Tensor, torch.Tensor], data_concepts_time: tuple, data_concepts_values: tuple
    # ):
    #     mean, log_std = x
    #     data_shape = mean.shape

    #     _, time_range = data_concepts_time
    #     _, values_range = data_concepts_values

    #     # reshape (and repeat) values_range to match drift_mean
    #     values_range_view = values_range.unsqueeze(1).repeat(1, data_shape[1], 1)  # Shape [B, L, 1]
    #     times_range_view = time_range.unsqueeze(1).repeat(1, data_shape[1], data_shape[2])  # Shape [B, L, 1]

    #     # rescale  mean
    #     drift_mean = mean * values_range_view / times_range_view  # Shape [B, L, 1]

    #     # rescale log std if provided
    #     if log_std is not None:
    #         learnt_drift_log_std = (
    #             log_std + torch.log(values_range_view) - torch.log(times_range_view)
    #         )  # Shape [B, L, 1]
    #         return drift_mean, learnt_drift_log_std

    #     else:
    #         return drift_mean

    def get_reversion_factor(self, data_concepts: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Get reversion factor for the min-max normalization.

        Args:
            data_concepts: min and range of original data.

        Returns:
            torch.Tensor: Reversion factor = range of the data.
        """
        _, range_data = data_concepts
        return range_data


###
# Denoising Blocks
###


class SavGolFilter(Block):
    def __init__(self, window_length: int = 15, polyorder: int = 3):
        super(SavGolFilter, self).__init__()

        self.window_length = window_length
        self.polyorder = polyorder

        self.filter = savgol_filter

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Apply Savitzky-Golay filter to input tensor.

        As scipy's savgol-filter is numpy-based, we need to convert the input tensor to numpy & to cpu and back.
        """
        if mask is None:
            mask = torch.zeros_like(x, dtype=bool)
        x_dim = x.dim()
        if x_dim == 3:
            if x.size(-1) == 1:
                x = x.squeeze(-1)
                mask = mask.squeeze(-1) if mask is not None else None
            else:
                raise ValueError("Input tensor must have shape [B, T, 1] or [B, T].")

        # convert to numpy and cpu
        x_device = x.device
        x = x.cpu().numpy()
        mask = mask.cpu().numpy()

        # apply filter to each sample
        denoised_samples = []
        for sample, mask_sample in zip(x, mask):
            denoised_samples.append(self.apply_savgol_to_sample(sample.copy(), mask_sample))

        x_denoised = np.stack(denoised_samples)

        # convert back to torch tensor and send to original device
        x_denoised = torch.tensor(x_denoised, device=x_device)

        if x_dim == 3:
            x_denoised = x_denoised.unsqueeze(-1)
        return x_denoised

    def apply_savgol_to_sample(self, values: np.array, mask: np.array) -> np.array:
        """
        Apply Savitzky-Golay filter to a single sample.

        Args:
            values: np.array, values to be filtered.
            mask: np.array, mask indicating which values are masked out. 1 indicates that value is masked out..

        Returns:
            np.array: denoised values. Same shape as input.
        """
        obs_values = values[mask == 0]
        window_length = min(self.window_length, len(obs_values) - 1)
        polyorder = min(self.polyorder, window_length - 1)
        denoised_values = savgol_filter(obs_values, window_length, polyorder)
        values[mask == 0] = denoised_values
        return values
