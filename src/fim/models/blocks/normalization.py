from abc import abstractmethod
from typing import Optional, Union

import torch
from torch import nn


from fim.models.blocks.base import Block
from fim.utils.helper import create_class_instance

eps = 1e-6

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


class NoNormalization(BaseNormalization):
    """Dummy class for no normalization."""

    def __init__(self):
        super(NoNormalization, self).__init__()

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, norm_params: Optional[dict] = None
    ) -> tuple[torch.Tensor, tuple]:
        return x, (torch.zeros((x.size(0), 1, 1), device=x.device), torch.ones((x.size(0), 1, 1), device=x.device))

    def revert_normalization(self, x: torch.Tensor, data_concepts: Union[tuple, torch.Tensor], log_scale: bool = False) -> torch.Tensor:
        return x

    def revert_normalization_drift(self, x: torch.Tensor, data_concepts: Union[tuple, torch.Tensor]) -> torch.Tensor:
        return x

    def get_reversion_factor(self, data_concepts: Optional[tuple[torch.Tensor, torch.Tensor]] = None) -> Union[torch.Tensor, int]:
        """If data concepts, return"""
        return data_concepts[1]


class Standardization(BaseNormalization):
    """Standardization block for normalizing input data via mean and std."""

    def __init__(self, mean_target: float = 0.0, std_target: float = 1.0):
        super(Standardization, self).__init__()
        self.mean_target = mean_target
        self.std_target = std_target if std_target != 0 else eps

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        norm_params: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple]:
        """
        Change statistics of given data x to target mean and std (Default: 0 and 1) with `X^ = std_target / std_data * (x - mean_data) + mean_target`.

        Standardization is applied along dim 1 (if x.dim()==3) or along dim=(1,2) (if x.dim()==4).

        Args:
            x: (torch.Tensor), shape [B, w, 1] or [B, wc, wlen, 1].
            mask: (torch.Tensor), shape [B, w, 1] or [B, wc, wlen, 1]. 1 indicating that value is masked out. Default: no values masked out.
            norm_params: (tuple), mean and std of original data. ([B, 1], [B, 1]). Default: None, mean and std are computed from x.
                Used to apply normalization to other data.

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
        elif x_dim == 2:
            x.unsqueeze(-1)

        # invert mask
        mask_inverted = ~mask

        # get masked mean and std per window
        if norm_params is None:
            mean_data = ((mask_inverted * x).sum(dim=1) / mask_inverted.sum(dim=1)).unsqueeze(-1)  # shape [B, 1, 1]
            var_data = ((mask_inverted * (x - mean_data) ** 2).sum(dim=1) / mask_inverted.sum(dim=1)).unsqueeze(-1)  # shape [B, 1, 1]
        else:
            mean_data, var_data = norm_params

        normalized_x = (x - mean_data) / torch.sqrt(var_data + eps) * self.std_target + self.mean_target

        # assert not torch.isnan(normalized_x).any(), "Normalization provoked NaN values. Make eps larger?"
        # assert normalized_x.shape == x.shape

        if x_dim == 4:
            normalized_x = normalized_x.view(B, wc, wlen, D)
        elif x_dim == 2:
            normalized_x = normalized_x.squeeze(-1)

        return normalized_x, (mean_data, var_data)  # shape [B, T, 1], ([B, 1, 1], [B, 1, 1])

    def revert_normalization(self, x: torch.Tensor, data_concepts: Union[tuple, torch.Tensor], log_scale: bool = False) -> torch.Tensor:
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

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        norm_params: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple]:
        """
        Standardize data by linear combination of Standardization and linearly embedded data statistics.

        Standardization is applied along dim 1 (if x.dim()==3) or along dim=(1,2) (if x.dim()==4).

        Args:
            x: torch.Tensor. Either 3 or 4 dimensional.
            mask: torch.Tensor, optional, default: no values are masked out. 1 indicates that a value is masked out.
            norm_params: tuple, mean and std of original data. ([B, 1], [B, 1]). Default: None, mean and std are computed from x.
                Used to apply normalization to other data.

        returns:
            standardized x and statistics of original data.
        """
        # x can either be 3 or 4 dimensional we need 3 dim.
        x_dim = x.dim()
        if x_dim == 4:
            B, wc, wlen, D = x.shape
            x = x.view(B, wc * wlen, D)
            mask = mask.view(B, wc * wlen, D)

        standardized_out, statistics = super().forward(x, mask, norm_params)  # shape [B, w, 1], ([B,1, 1], [B,1, 1])
        if norm_params is not None:
            # i.e. this was only used to normalize the data, no need to embed statistics for additional information
            return standardized_out, statistics

        embedded_statistics = self.statistics_embedder(torch.concat(statistics, dim=1).squeeze(-1)).unsqueeze(-1)  # shape [B, w, 1]

        if embedded_statistics.shape != standardized_out.shape:
            raise ValueError(
                f"The statistics are mapped to {embedded_statistics.size(1)} but expected {standardized_out.size(1)} (length of concatenated input windows)."
            )

        # Hacky solution to make sure that the embedded statistics have the same length as the standardized output
        # if embedded_statistics.size(1) > standardized_out.size(1):
        #     embedded_statistics = embedded_statistics[:, : standardized_out.size(1), :]
        # elif embedded_statistics.size(1) < standardized_out.size(1):
        #     embedded_statistics = embedded_statistics.repeat(1, standardized_out.size(1), 1)

        # fuse normalized x and embedded statistics
        x_fused = torch.sqrt(1 - self.linear_factor) * standardized_out + torch.sqrt(self.linear_factor) * embedded_statistics
        if x_dim == 4:
            # need to reshape
            x_fused = x_fused.view(B, wc, wlen, D)

        return x_fused, statistics

    def __repr__(self):
        return f"StandardizationSERIN(mean_target={self.mean_target}, std_target={self.std_target}, lin_factor={self.linear_factor}, network={self.statistics_embedder})"
