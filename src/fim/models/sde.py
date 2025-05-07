from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, Optional

import optree
import torch
import torch._dynamo
import torch.nn as nn
from torch import Tensor
from transformers import AutoConfig, AutoModel, PretrainedConfig

from fim import results_path
from fim.models.blocks import AModel, ModelFactory
from fim.models.blocks.base import Block
from fim.models.blocks.neural_operators import AttentionOperator, InducedSetTransformerEncoder, ResidualEncoderLayer
from fim.models.utils import SinActivation
from fim.utils.helper import create_class_instance


torch._dynamo.config.suppress_errors = True


def rmse_at_locations(estimated: Tensor, target: Tensor, mask: Tensor, scale_per_dimension: Optional[Tensor] = None) -> Tensor:
    """
    Return RMSE between target and estimated values per location. Mask indicates which values (in last dimension) to include.

    Args:
        estimated (Tensor): estimated of target values. Shape: [B, G, D]
        target (Tensor): Target values. Shape: [B, G, D]
        mask (Tensor): 0 at values to ignore in loss computations. Shape: [B, G, D]
        scale_per_dimension (Tensor): If not None, multiply error per location and dimension by scale.

    Return:
        rmse (Tensor): RMSE at locations. Shape: [B, G]
    """
    assert estimated.ndim == 3, "Got " + str(estimated.ndim)
    assert estimated.shape == target.shape, "Got " + str(estimated.shape) + " and " + str(target.shape)
    assert estimated.shape == mask.shape, "Got " + str(estimated.shape) + " and " + str(mask.shape)

    # squared error at non-masked values
    se = mask * ((estimated - target) ** 2)

    if scale_per_dimension is not None:
        assert se.shape == scale_per_dimension.shape
        se = se * scale_per_dimension

    se = torch.sum(se, dim=-1)  # [B, G]

    # mean over non-masked values
    non_masked_values_count = torch.sum(mask, dim=-1)
    mse = se / torch.clip(non_masked_values_count, min=1)  # [B, G]

    # take root per location
    rmse = torch.sqrt(torch.clip(mse, min=1e-12))  # [B, G]

    assert rmse.ndim == 2, "Got " + str(rmse.ndim)

    return rmse


def mse_at_locations(estimated: Tensor, target: Tensor, mask: Tensor, scale_per_dimension: Optional[Tensor] = None) -> Tensor:
    """
    Return MSE between target and estimated values per location. Mask indicates which values (in last dimension) to include.

    Args:
        estimated (Tensor): estimated of target values. Shape: [B, G, D]
        target (Tensor): Target values. Shape: [B, G, D]
        mask (Tensor): 0 at values to ignore in loss computations. Shape: [B, G, D]
        scale_per_dimension (Tensor): If not None, multiply error per location and dimension by scale.

    Return:
        rmse (Tensor): MSE at locations. Shape: [B, G]
    """
    assert estimated.ndim == 3, "Got " + str(estimated.ndim)
    assert estimated.shape == target.shape, "Got " + str(estimated.shape) + " and " + str(target.shape)
    assert estimated.shape == mask.shape, "Got " + str(estimated.shape) + " and " + str(mask.shape)

    # squared error at non-masked values
    se = mask * ((estimated - target) ** 2)

    if scale_per_dimension is not None:
        assert se.shape == scale_per_dimension.shape
        se = se * scale_per_dimension

    se = torch.sum(se, dim=-1)  # [B, G]

    # mean over non-masked values
    non_masked_values_count = torch.sum(mask, dim=-1)
    mse = se / torch.clip(non_masked_values_count, min=1)  # [B, G]

    # take root per location
    assert mse.ndim == 2, "Got " + str(mse.ndim)

    return mse


def nmse_at_locations(estimated: Tensor, target: Tensor, mask: Tensor, scale_per_dimension: Optional[Tensor] = None) -> Tensor:
    mse = mse_at_locations(estimated, target, mask, scale_per_dimension)
    target_norm = mse_at_locations(target, torch.zeros_like(target), mask, scale_per_dimension)

    eps = 1e-6

    return mse / (target_norm + eps)


def nrmse_at_locations(estimated: Tensor, target: Tensor, mask: Tensor, scale_per_dimension: Optional[Tensor] = None) -> Tensor:
    """
    Return NRMSE (normalized by the target norm) between target and estimated values per location. Mask indicates which values (in last dimension) to include.

    Args:
        estimated (Tensor): estimated of target values. Shape: [B, G, D]
        target (Tensor): Target values. Shape: [B, G, D]
        mask (Tensor): 0 at values to ignore in loss computations. Shape: [B, G, D]
        scale_per_dimension (Tensor): If not None, multiply error per location and dimension by scale.

    Return:
        nrmse (Tensor): NRMSE at locations. Shape: [B, G]
    """
    assert estimated.ndim == 3, "Got " + str(estimated.ndim)
    assert estimated.shape == target.shape, "Got " + str(estimated.shape) + " and " + str(target.shape)
    assert estimated.shape == mask.shape, "Got " + str(estimated.shape) + " and " + str(mask.shape)

    rmse = rmse_at_locations(estimated, target, mask, scale_per_dimension)  # [B, G]

    # this is not the norm currently (division by number of dimensions), but does respect masking
    target_norm = rmse_at_locations(target, torch.zeros_like(target), mask, scale_per_dimension)

    return rmse / (target_norm + 1)


def gaussian_nll_at_locations(estimated: Tensor, log_var_estimated: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """
    Return (diagonal) gaussian NLL of target under estimated distribution. Mask indicates which values (in last dimension) to include.

    Args:
        estimated (Tensor): Mean estimated of target. Shape: [B, G, D]
        log_var_estimated (Tensor): Log of variance of estimated of target. Shape: [B, G, D]
        target (Tensor): Target values to compute the NLL of. Shape: [B, G, D]
        mask (Tensor): 0 at values to ignore in loss computations. Shape: [B, G, D]

    Return:
        nll (Tensor): Gaussian NLL, (regularized) averaged over all batches and grid points. Shape: []
    """
    assert estimated.ndim == 3, "Got " + str(estimated.ndim)
    assert estimated.shape == target.shape, "Got " + str(estimated.shape) + " and " + str(target.shape)
    assert estimated.shape == log_var_estimated.shape, "Got " + str(estimated.shape) + " and " + str(log_var_estimated.shape)
    assert estimated.shape == mask.shape, "Got " + str(estimated.shape) + " and " + str(mask.shape)

    # (diagonal) gaussian NLL per dimension
    var_estimated = torch.exp(log_var_estimated)
    nll_per_dim = (
        (estimated - target) ** 2 / (2 * var_estimated)
        + 1 / 2 * log_var_estimated
        + 1 / 2 * torch.log(2 * torch.pi * torch.ones_like(estimated))
    )  # [B, G, D]

    # sum over non-masked values
    nll = torch.sum(mask * nll_per_dim, dim=-1)  # [B, G]

    assert nll.ndim == 2, "Got " + str(nll.ndim)

    return nll


class SineTimeEncoding(Block):
    """
    Implements the time encoding as described in "Multi-time attention networks for irregularly sampled time series, Shukla & Marlin, 2020".

    Each time point t is encoded as a vector of dimension d_time:
        - first element: linear embedding of t: w_0*t + b_0
        - remaining elements: sinusoidal embedding of t with different frequencies: sin(w_i*t + b_i) for i in {1, ..., d_time-1}
    w_j and b_j are learnable parameters.
    """

    def __init__(self, out_features: int, **kwargs):
        """
        Args:
            d_time (int): Dimension of the time representation
        """
        super(SineTimeEncoding, self).__init__()

        self.in_features = kwargs.get("in_features", 1)
        self.out_features = out_features

        self.linear_embedding = nn.Linear(self.in_features, self.in_features, bias=True)
        self.periodic_embedding = nn.Sequential(
            nn.Linear(self.in_features, self.out_features - self.in_features, bias=True), SinActivation()
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid (torch.Tensor): Grid of time points, shape (batch_size, seq_len, 1)

        Returns:
            torch.Tensor: Time encoding, shape (batch_size, seq_len, d_time)
        """
        linear = self.linear_embedding(grid)
        periodic = self.periodic_embedding(grid)

        return torch.cat([linear, periodic], dim=-1)


# 1. Define your query generation model (a simple linear layer can work)
class QueryGenerator(nn.Module):
    def __init__(self, input_dim, query_dim):
        super(QueryGenerator, self).__init__()
        self.linear = nn.Linear(input_dim, query_dim)

    def forward(self, x):
        return self.linear(x)


# 2. Define a static query matrix as a learnable parameter
class StaticQuery(nn.Module):
    def __init__(self, num_steps, query_dim):
        super(StaticQuery, self).__init__()
        self.queries = nn.Parameter(torch.randn(num_steps, query_dim))  # Learnable queries

    def forward(self):
        return self.queries


@torch.profiler.record_function("forward_fill_masked_values")
def forward_fill_masked_values(x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """
    Fill forward values at masked entries in dimension -2.

    Approach:
        An unmasked observation means the cumsum along dim -2 increases by 1.
        Cummax returns the index, when change happens.
        Take care of edgecase: if first values are masked, fill backward from first observed values.

    Args:
        x (Tensor): Tensor to fill values in. Shape: [..., T, D].
        mask (Tensor): Boolean tensor, True indicates observed, False indicates to fill.  Shape: [..., T, 1].

    Return:
        filled_x (Tensor): x with forward filled values at masked out indices. Shape: [..., T, D].

    """
    if mask is None:
        return x

    else:
        mask = mask.bool()
        mask = torch.broadcast_to(mask, x.shape)  # [..., T, D]

        # change in cumsum indicates new observation
        mask_cumsum = torch.cumsum(mask, dim=-2)

        # extract index of change (* mask is important, s.t. index stays the same if masked out values are hit)
        indices_to_take = torch.cummax(mask_cumsum * mask, dim=-2)[1]  # [1] returns indices, [..., T, D]

        # edge case: if first values are masked, we backward fill with first really observed value
        first_non_masked_index = torch.argmin(torch.where(mask_cumsum == 0, torch.inf, mask_cumsum), dim=-2, keepdim=True)
        indices_to_take = torch.where(mask_cumsum == 0, first_non_masked_index, indices_to_take)

        # gather values at indices
        filled_x = torch.gather(x, dim=-2, index=indices_to_take)

        return filled_x


@torch.profiler.record_function("backward_fill_masked_values")
def backward_fill_masked_values(x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """
    Fill backwards values at masked entries in dimension -2.

    Args:
        x (Tensor): Tensor to fill values in. Shape: [..., T, D].
        mask (Tensor): Boolean tensor, True indicates observed, False indicates to fill.  Shape: [..., T, 1].

    Return:
        filled_x (Tensor): x with backward filled values at masked out indices. Shape: [..., T, D].

    """
    if mask is None:
        return x

    else:
        mask = mask.bool()
        mask = torch.broadcast_to(mask, x.shape)

        # backward fill is just forward fill of flipped tensor
        mask = torch.flip(mask, dims=(-2,))
        x = torch.flip(x, dims=(-2,))

        x = forward_fill_masked_values(x, mask)

        return torch.flip(x, dims=(-2,))


class InstanceNormalization(ABC):
    @abstractmethod
    def get_norm_stats(self, values: Tensor, mask: Optional[Tensor] = None, **kwargs) -> Any:
        """
        Extract stats from values for instance normalization.

        Args:
            values (Tensor): Values to instance normalize. Shape: [B, ..., D]
            mask (Optional[Tensor]): True indicates inclusion in stats computation. Shape: [B, ..., 1]
        """
        raise NotImplementedError("`get_norm_stats` is not implemented in your class!")

    @abstractmethod
    def normalization_map(self, values: Tensor, norm_stats: Any, derivative_num: Optional[int] = 0) -> Tensor:
        """
        Return the (up to second) derivative of the instance normalization to some values.

        Args:
            values (Tensor): Values to instance normalize. Shape: [B, ..., D]
            norm_stats (Any): Stats from `get_norm_stats` defining normalization map.
            derivative_num (int): Derivative of instance normalization to return.

        Returns:
            (derivative) of image of values under instance normalization. Shape: [B, ..., D]
        """
        raise NotImplementedError("`normalization_map` is not implemented in your class!")

    @abstractmethod
    def inverse_normalization_map(self, values: Tensor, norm_stats: Any, derivative_num: Optional[int] = 0) -> Tensor:
        """
        Return the (up to second) derivative of the inverse of instance normalization.

        Args:
            values (Tensor): Values to apply inverse instance normalization to. Shape: [B, ..., D]
            norm_stats (Any): Stats from `get_norm_stats` defining normalization map.
            derivative_num (int): Derivative of inverse instance normalization to return.

        Returns:
            (derivative) of image of values under inverse instance normalization. Shape: [B, ..., D]
        """
        raise NotImplementedError("`inverse_normalization_map` is not implemented in your class!")

    @classmethod
    def from_values(
        cls, values: Tensor, mask: Optional[Tensor], get_norm_stats_kwargs: Optional[dict] = {}, init_kwargs: Optional[dict] = {}
    ):
        """
        Pass kwargs to cls.__init__ and get norm_stats.

        Args:
            values (Tensor): Values to instance normalize. Shape: [B, ..., D]
            mask (Optional[Tensor]): True indicates inclusion in stats computation. Shape: [B, ..., 1]
            get_norm_stats_kwargs (Optional[dict]): passed to cls.get_norm_stats.
            kwargs (Optional[dict]): passed to cls.__init__

        Returns:
            norm_instance (InstanceNormalization): Instance of class.
            norm_stats (Any): Stats from `get_norm_stats` defining normalization map.
        """
        norm_instance = cls(**init_kwargs)
        norm_stats = norm_instance.get_norm_stats(values, mask, **get_norm_stats_kwargs)

        return norm_instance, norm_stats

    @staticmethod
    def squash_intermediate_dims(values: Tensor) -> tuple[Tensor, tuple]:
        """
        Reshape values from [B, ..., D] to [B, *, D] momentarily. Return original shape for later reshaping.

        Args:
            values (Tensor): tensor to reshape. Shape: [B, ..., D]

        Returns:
            reshaped_values (Tensor): Shape: [B, *, D]
            original_shape: original shape of values for further use
        """

        original_shape = values.shape
        B, D = values.shape[0], values.shape[-1]
        reshaped_values = values.reshape(B, -1, D)

        return reshaped_values, original_shape

    @staticmethod
    def expand_norm_stats(shape: tuple, norm_stats: tuple[Tensor]) -> tuple[Tensor]:
        """
        Return normalization statistics expanded to desired shape

        Args:
            shape (tuple): Expected shape. Must be of length 3, specifically (B, *, D), where norm_stats are of [B, D].

        Returns:
            expanded_norm_stats (tuple[Tensor]): norm_stats expanded to (B, *, D)
        """
        assert len(shape) == 3, "Expect 3 dimensions, got " + str(len(shape)) + ". Passed shape: " + str(shape)

        return tuple([x.unsqueeze(-2).expand(shape) for x in norm_stats])


class MinMaxNormalization(InstanceNormalization):
    """
    Linear transformation to values s.t. defining values are in the interval [normalized_min, normalized_max].
    """

    def __init__(self, normalized_min: float, normalized_max: float, **kwargs):
        # values and target interval boundaries
        self.normalized_min, self.normalized_max = normalized_min, normalized_max

        # apply transform map over three axes: batch, time, dimension
        transform_map_grad = torch.func.grad(self.transform_map)
        transform_map_grad_grad = torch.func.grad(transform_map_grad)

        self.batch_transform_map = torch.vmap(torch.vmap(torch.vmap(self.transform_map)))
        self.batch_transform_map_grad = torch.vmap(torch.vmap(torch.vmap(transform_map_grad)))
        self.batch_transform_map_grad_grad = torch.vmap(torch.vmap(torch.vmap(transform_map_grad_grad)))

    @torch.profiler.record_function("minmax_get_stats")
    def get_norm_stats(self, values: Tensor, mask: Optional[Tensor] = None) -> tuple[Tensor]:
        """
        Return min and max of passed values along all dimensions 1 to -2, where mask == True.
        Return self.normalized_min and self.normalized_max as Tensors.

        Args:
            values (Tensor): Shape: [B, ..., D]
            mask (Tensor): Shape: [B, ...., 1]

        Returns:
            ref_min, ref_max (Tensor): Statistics of values along dimensions 1 to -2. Shape: [B, D]
            tar_min, tar_max (Tensor): Normalization targets. Shape: [B, D]
        """
        # Squash intermediate dimensions
        values, _ = self.squash_intermediate_dims(values)

        # reference values
        if mask is None:
            values_min = torch.amin(values, dim=-2)
            values_max = torch.amax(values, dim=-2)

        else:
            mask = mask.bool()

            mask, _ = self.squash_intermediate_dims(mask)
            mask = torch.broadcast_to(mask, values.shape)

            values_min = torch.amin(torch.where(mask, values, torch.inf), dim=-2)
            values_max = torch.amax(torch.where(mask, values, -torch.inf), dim=-2)

        assert values_min.ndim == values_max.ndim == 2, f"Got {values_min.ndim} and {values_max.ndim}."

        # target values
        if isinstance(self.normalized_min, Tensor):
            target_min, target_max = self.normalized_min, self.normalized_max

        else:
            target_min = self.normalized_min * torch.ones_like(values_min)
            target_max = self.normalized_max * torch.ones_like(values_max)

        B, D = values_min.shape
        assert target_min.shape == target_max.shape == (B, D), f"Got {target_min} and {target_max}."

        return values_min, values_max, target_min, target_max

    @staticmethod
    def transform_map(value: Tensor, src_min: Tensor, src_max: Tensor, tar_min: Tensor, tar_max: Tensor) -> Tensor:
        """
        Apply the (linear) transformation of interval [src_min, src_max] to [tar_max, tar_max] to the passed value.
        I.e. evaluate the map x -> (x - src_min) / (src_max - src_min) * (tar_max - tar_min) + tar_min.

        Args:
            value (Tensor): Shape: []
            src_min, src_max (Tensor): Boundaries of source interval. Shape: []
            tar_min, tar_max (Tensor): Boundaries of target interval. Shape: []

        Returns:
            transformed_value (Tensor): Image of value under interval transformation. Shape: []
        """
        assert value.ndim == 0, "Got value.ndim == " + str(value.ndim) + ", expected 0"

        tar_range = tar_max - tar_min
        src_range = src_max - src_min

        src_range = torch.clip(src_range, min=1e-6)

        transformed_value = (value - src_min) * tar_range / src_range + tar_min
        assert transformed_value.ndim == 0, "Got transformed_value.ndim == " + str(transformed_value.ndim) + ", expected 0"

        return transformed_value

    @torch.profiler.record_function("minmax_norm_map")
    def normalization_map(self, values: Tensor, norm_stats: tuple[Tensor], derivative_num: Optional[int] = 0) -> Tensor:
        """
        (Derivative of) normalization based on previously set statistics, i.e. evaluate the map
        x -> (x - unnormalized_min) / (unnormalized_max - unnormalized_min) * (normalized_max - normalized_min) + normalized_min
        at all values.

        Args:
            values (Tensor): Values to normalized based on previously set statistics. Shape: [B, ..., D]
            norm_stats (tuple[Tensor]): stats needed for normalization: (ref_min, ref_max, tar_min, tar_max). Shape: [B, D]
            derivative_num (int): Derivative of normalization map to return.

        Returns:
            (derivative) of image of values under normalization_map: Normalized values. Shape: [B, ..., D]
        """
        # check shapes
        B, D = norm_stats[0].shape
        assert values.ndim >= 2, "Got values.ndim == " + str(values.ndim) + ", expected >=2."
        assert values.shape[0] == B, "Got batch size " + str(values.shape[0]) + ", expected " + str(B)
        assert values.shape[-1] == D, "Got dimension " + str(values.shape[-1]) + ", expected " + str(D)

        # Squash intermediate dimensions from values
        values, original_shape = self.squash_intermediate_dims(values)

        # prepare shapes of norm stats
        expanded_norm_stats: tuple[Tensor] = self.expand_norm_stats(values.shape, norm_stats)
        reference_min, reference_max, normalized_min, normalized_max = expanded_norm_stats

        # apply transformation from unnormalized to normalized
        if derivative_num == 0:
            out = self.batch_transform_map(values, reference_min, reference_max, normalized_min, normalized_max)

        elif derivative_num == 1:
            out = self.batch_transform_map_grad(values, reference_min, reference_max, normalized_min, normalized_max)

        elif derivative_num == 2:
            out = self.batch_transform_map_grad_grad(values, reference_min, reference_max, normalized_min, normalized_max)

        else:
            raise ValueError("Can only return up to second derivative. Got " + str(derivative_num))

        # Reintroduce intermediate dimensions from values
        out = out.reshape(original_shape)

        return out

    def inverse_normalization_map(self, values: Tensor, norm_stats: tuple[Tensor], derivative_num: Optional[int] = 0) -> Tensor:
        """
        (Derivative of) inverse normalization of the passed values based on previously set statistics, i.e. evaluate the map
        x -> (x - normalized_min) / (normalized_max - normalized_min) * (unnormalized_max - unnormalized_min) + unnormalized_min
        at all values.

        Args:
            values (Tensor): Values to apply inverse normalization based on previously set statistics to. Shape: [B, ..., D]
            norm_stats (tuple[Tensor]): stats needed for normalization: (ref_min, ref_max, tar_min, tar_max). Shape: [B, D]
            derivative_num (int): Derivative of inverse normalization map to return.

        Returns:
            renormalized_values: Reormalized values. Shape: [B, ..., D]
        """
        # check shapes
        B, D = norm_stats[0].shape
        assert values.ndim >= 2, "Got values.ndim == " + str(values.ndim) + ", expected >=2."
        assert values.shape[0] == B, "Got batch size " + str(values.shape[0]) + ", expected " + str(B)
        assert values.shape[-1] == D, "Got dimension " + str(values.shape[-1]) + ", expected " + str(D)

        # Squash intermediate dimensions from values
        values, original_shape = self.squash_intermediate_dims(values)

        # prepare shapes of norm stats
        expanded_norm_stats: tuple[Tensor] = self.expand_norm_stats(values.shape, norm_stats)
        reference_min, reference_max, normalized_min, normalized_max = expanded_norm_stats

        # apply transformation from normalized to unnormalized
        if derivative_num == 0:
            out = self.batch_transform_map(values, normalized_min, normalized_max, reference_min, reference_max)

        elif derivative_num == 1:
            out = self.batch_transform_map_grad(values, normalized_min, normalized_max, reference_min, reference_max)

        elif derivative_num == 2:
            out = self.batch_transform_map_grad_grad(values, normalized_min, normalized_max, reference_min, reference_max)

        else:
            raise ValueError("Can only return up to second derivative. Got " + str(derivative_num))

        # Reintroduce intermediate dimensions from values
        out = out.reshape(original_shape)

        return out


class Standardization(InstanceNormalization):
    """
    Standardize defining values to standard normal: x -> (x - mean) / std
    """

    def __init__(self, **kwargs):
        # apply standardization map over three axes: batch, time, dimension
        standardize_map_grad = torch.func.grad(self.standardize_map)
        standardize_map_grad_grad = torch.func.grad(standardize_map_grad)

        self.batch_standardize_map = torch.vmap(torch.vmap(torch.vmap(self.standardize_map)))
        self.batch_standardize_map_grad = torch.vmap(torch.vmap(torch.vmap(standardize_map_grad)))
        self.batch_standardize_map_grad_grad = torch.vmap(torch.vmap(torch.vmap(standardize_map_grad_grad)))

        # apply inverse standardization map over three axes: batch, time, dimension
        inv_standardize_map_grad = torch.func.grad(self.inv_standardize_map)
        inv_standardize_map_grad_grad = torch.func.grad(inv_standardize_map_grad)

        self.batch_inv_standardize_map = torch.vmap(torch.vmap(torch.vmap(self.inv_standardize_map)))
        self.batch_inv_standardize_map_grad = torch.vmap(torch.vmap(torch.vmap(inv_standardize_map_grad)))
        self.batch_inv_standardize_map_grad_grad = torch.vmap(torch.vmap(torch.vmap(inv_standardize_map_grad_grad)))

    @torch.profiler.record_function("stand_get_stats")
    def get_norm_stats(self, values: Tensor, mask: Optional[Tensor] = None) -> tuple[Tensor]:
        """
        Return mean and standard deviation of passed values along all dimensions 1 to -2, where mask == True.

        Args:
            values (Tensor): Shape: [B, ..., D]
            mask (Tensor): Shape: [B, ...., 1]

        Returns:
            ref_mean, ref_std (Tensor): Statistics of values along dimensions 1 to -2. Shape: [B, D]
        """
        # Squash intermediate dimensions
        values, _ = self.squash_intermediate_dims(values)

        # reference values
        if mask is None:
            values_mean = torch.mean(values, dim=-2)
            values_std = torch.std(values, dim=-2)

        else:
            mask = mask.bool()

            mask, _ = self.squash_intermediate_dims(mask)
            mask = torch.broadcast_to(mask, values.shape)

            values_mean = torch.nanmean(torch.where(mask, values, torch.nan), dim=-2, keepdim=True)  # [B, 1, D]

            # masked std
            se = (values - values_mean) ** 2
            masked_se = torch.where(mask, se, torch.nan)
            masked_var = torch.nanmean(masked_se, dim=-2)

            values_std = torch.sqrt(masked_var)
            values_mean = values_mean.squeeze(-2)

        values_std = torch.clip(values_std, min=1e-6)

        assert values_std.ndim == values_mean.ndim == 2, f"Got {values_std.ndim} and {values_mean.ndim}."

        return values_mean, values_std

    @staticmethod
    def standardize_map(value: Tensor, ref_mean: Tensor, ref_std: Tensor) -> Tensor:
        """
        Apply the transformation that standardizes the reference values: x -> (x - ref_mean) / ref_std

        Args:
            value (Tensor): Shape: []
            ref_mean, ref_std (Tensor): Statistics that standardize reference values.

        Returns:
            transformed_value (Tensor): Image of value under standardization transformation. Shape: []
        """
        assert value.ndim == 0, "Got value.ndim == " + str(value.ndim) + ", expected 0"

        transformed_value = (value - ref_mean) / ref_std

        assert transformed_value.ndim == 0, "Got transformed_value.ndim == " + str(transformed_value.ndim) + ", expected 0"

        return transformed_value

    @staticmethod
    def inv_standardize_map(value: Tensor, ref_mean: Tensor, ref_std: Tensor) -> Tensor:
        """
        Apply the transformation that reverse the standardization of reference values: x -> x * ref_std + ref_mean

        Args:
            value (Tensor): Shape: []
            ref_mean, ref_std (Tensor): Statistics that standardize reference values.

        Returns:
            transformed_value (Tensor): Image of value under standardization revision transformation. Shape: []
        """
        assert value.ndim == 0, "Got value.ndim == " + str(value.ndim) + ", expected 0"

        transformed_value = value * ref_std + ref_mean

        assert transformed_value.ndim == 0, "Got transformed_value.ndim == " + str(transformed_value.ndim) + ", expected 0"

        return transformed_value

    def normalization_map(self, values: Tensor, norm_stats: tuple[Tensor], derivative_num: Optional[int] = 0) -> Tensor:
        """
        (Derivative of) normalization based on previously set statistics, i.e. evaluate the map
        x -> (x - ref_mean) / ref_std
        at all values.

        Args:
            values (Tensor): Values to normalize based on previously set statistics. Shape: [B, ..., D]
            norm_stats (tuple[Tensor]): stats needed for normalization: (ref_mean, ref_std). Shape: [B, D]
            derivative_num (int): Derivative of normalization map to return.

        Returns:
            (derivative) of image of values under normalization_map: Normalized values. Shape: [B, ..., D]
        """
        # check shapes
        B, D = norm_stats[0].shape
        assert values.ndim >= 2, "Got values.ndim == " + str(values.ndim) + ", expected >=2."
        assert values.shape[0] == B, "Got batch size " + str(values.shape[0]) + ", expected " + str(B)
        assert values.shape[-1] == D, "Got dimension " + str(values.shape[-1]) + ", expected " + str(D)

        # Squash intermediate dimensions from values
        values, original_shape = self.squash_intermediate_dims(values)

        # prepare shapes of norm stats
        expanded_norm_stats: tuple[Tensor] = self.expand_norm_stats(values.shape, norm_stats)
        ref_mean, ref_std = expanded_norm_stats

        # apply transformation from unnormalized to normalized
        if derivative_num == 0:
            out = self.batch_standardize_map(values, ref_mean, ref_std)

        elif derivative_num == 1:
            out = self.batch_standardize_map_grad(values, ref_mean, ref_std)

        elif derivative_num == 2:
            out = self.batch_standardize_map_grad_grad(values, ref_mean, ref_std)

        else:
            raise ValueError("Can only return up to second derivative. Got " + str(derivative_num))

        # Reintroduce intermediate dimensions from values
        out = out.reshape(original_shape)

        return out

    def inverse_normalization_map(self, values: Tensor, norm_stats: tuple[Tensor], derivative_num: Optional[int] = 0) -> Tensor:
        """
        (Derivative of) inverse normalization based on previously set statistics, i.e. evaluate the map
        x -> x * ref_std + ref_mean
        at all values.

        Args:
            values (Tensor): Values to apply inverse normalization based on previously set statistics to. Shape: [B, ..., D]
            norm_stats (tuple[Tensor]): stats needed for normalization: (ref_mean, ref_std). Shape: [B, D]
            derivative_num (int): Derivative of inverse normalization map to return.

        Returns:
            renormalized_values: Renormalized values. Shape: [B, ..., D]
        """
        # check shapes
        B, D = norm_stats[0].shape
        assert values.ndim >= 2, "Got values.ndim == " + str(values.ndim) + ", expected >=2."
        assert values.shape[0] == B, "Got batch size " + str(values.shape[0]) + ", expected " + str(B)
        assert values.shape[-1] == D, "Got dimension " + str(values.shape[-1]) + ", expected " + str(D)

        # Squash intermediate dimensions from values
        values, original_shape = self.squash_intermediate_dims(values)

        expanded_norm_stats: tuple[Tensor] = self.expand_norm_stats(values.shape, norm_stats)
        ref_mean, ref_std = expanded_norm_stats

        # apply transformation from normalized to unnormalized
        if derivative_num == 0:
            out = self.batch_inv_standardize_map(values, ref_mean, ref_std)

        elif derivative_num == 1:
            out = self.batch_inv_standardize_map_grad(values, ref_mean, ref_std)

        elif derivative_num == 2:
            out = self.batch_inv_standardize_map_grad_grad(values, ref_mean, ref_std)

        else:
            raise ValueError("Can only return up to second derivative. Got " + str(derivative_num))

        # Reintroduce intermediate dimensions from values
        out = out.reshape(original_shape)

        return out


class DeltaLogCentering(InstanceNormalization):
    """
    Center ln(x) at some target value: x -> exp(ln(x) - ln_mean + ln(target_value)) = (x  * target_value) / exp(ln_mean)
    Where ln_mean = torch.log(y).mean() for some reference values y
    """

    def __init__(self, **kwargs):
        self.target_value: float = kwargs.get("target_value", 0.01)
        assert self.target_value > 0

        # apply centering map over three axes: batch, time, dimension
        center_map_grad = torch.func.grad(self.center_map)
        center_map_grad_grad = torch.func.grad(center_map_grad)

        self.batch_center_map = torch.vmap(torch.vmap(torch.vmap(self.center_map)))
        self.batch_center_map_grad = torch.vmap(torch.vmap(torch.vmap(center_map_grad)))
        self.batch_center_map_grad_grad = torch.vmap(torch.vmap(torch.vmap(center_map_grad_grad)))

        # apply inverse standardization map over three axes: batch, time, dimension
        inv_center_map_grad = torch.func.grad(self.inv_center_map)
        inv_center_map_grad_grad = torch.func.grad(inv_center_map_grad)

        self.batch_inv_center_map = torch.vmap(torch.vmap(torch.vmap(self.inv_center_map)))
        self.batch_inv_center_map_grad = torch.vmap(torch.vmap(torch.vmap(inv_center_map_grad)))
        self.batch_inv_center_map_grad_grad = torch.vmap(torch.vmap(torch.vmap(inv_center_map_grad_grad)))

    @torch.profiler.record_function("stand_get_stats")
    def get_norm_stats(self, values: Tensor, mask: Optional[Tensor] = None) -> tuple[Tensor]:
        """
        Return mean of ln(values) along all dimensions 1 to -2, where mask == True.
        Assume masked values have been filled with '_fill_masked_values'.

        Args:
            values (Tensor): Shape: [B, ..., D]
            mask (Tensor): Shape: [B, ...., 1]

        Returns:
            ln_ref_mean (Tensor): Mean of ln(values) along dimensions 1 to -2. Shape: [B, D]
        """
        # Squash intermediate dimensions
        values, _ = self.squash_intermediate_dims(values)

        # reference values
        ln_values = torch.log(values)

        if mask is None:
            mean_ln_values = torch.mean(ln_values, dim=-2)

        else:
            mask = mask.bool()

            mask, _ = self.squash_intermediate_dims(mask)
            mask = torch.broadcast_to(mask, values.shape)

            mask = torch.logical_and(mask, values > 0.0)  # in case of 0 at the end of a trajectory
            mean_ln_values = torch.nanmean(torch.where(mask, ln_values, torch.nan), dim=-2)  # [B, D]

        assert mean_ln_values.ndim == 2, f"Got {mean_ln_values.shape}."

        return mean_ln_values, self.target_value * torch.ones_like(mean_ln_values)

    @staticmethod
    def center_map(value: Tensor, ref_mean_ln: Tensor, target_value: Tensor) -> Tensor:
        """
        Apply the transformation that centers the ln(values) at the target_value:  x -> exp(ln(x) - ref_ln_mean + ln(target_value)) = (x  * target_value) / exp(ln_ref_mean)

        Args:
            value (Tensor): Shape: []
            ref_mean_ln, target_value (Tensor): Statistics that center reference ln(values) at target_value.

        Returns:
            transformed_value (Tensor): Image of value under centering transformation. Shape: []
        """
        assert value.ndim == 0, "Got value.ndim == " + str(value.ndim) + ", expected 0"

        transformed_value = value * target_value * torch.exp(-ref_mean_ln)

        assert transformed_value.ndim == 0, "Got transformed_value.ndim == " + str(transformed_value.ndim) + ", expected 0"

        return transformed_value

    @staticmethod
    def inv_center_map(value: Tensor, ref_mean_ln: Tensor, target_value: Tensor) -> Tensor:
        """
        Apply the transformation that reverse the centering of reference ln(values): x -> x * exp(ln_ref_mean) / target_value

        Args:
            value (Tensor): Shape: []
            ref_mean, ref_std (Tensor): Statistics that center reference values.

        Returns:
            transformed_value (Tensor): Image of value under centering revision transformation. Shape: []
        """
        assert value.ndim == 0, "Got value.ndim == " + str(value.ndim) + ", expected 0"

        transformed_value = value / target_value * torch.exp(ref_mean_ln)

        assert transformed_value.ndim == 0, "Got transformed_value.ndim == " + str(transformed_value.ndim) + ", expected 0"

        return transformed_value

    def normalization_map(self, values: Tensor, norm_stats: tuple[Tensor], derivative_num: Optional[int] = 0) -> Tensor:
        """
        (Derivative of) normalization based on previously set statistics, i.e. evaluate the map
        x -> (x * target_value) / exp(ln_ref_mean)
        at all values.

        Args:
            values (Tensor): Values to normalize based on previously set statistics. Shape: [B, ..., D]
            norm_stats (tuple[Tensor]): stats needed for normalization: (ref_mean, ref_std). Shape: [B, D]
            derivative_num (int): Derivative of normalization map to return.

        Returns:
            (derivative) of image of values under normalization_map: Normalized values. Shape: [B, ..., D]
        """
        # check shapes
        B, D = norm_stats[0].shape
        assert values.ndim >= 2, "Got values.ndim == " + str(values.ndim) + ", expected >=2."
        assert values.shape[0] == B, "Got batch size " + str(values.shape[0]) + ", expected " + str(B)
        assert values.shape[-1] == D, "Got dimension " + str(values.shape[-1]) + ", expected " + str(D)

        # Squash intermediate dimensions from values
        values, original_shape = self.squash_intermediate_dims(values)

        # prepare shapes of norm stats
        expanded_norm_stats: tuple[Tensor] = self.expand_norm_stats(values.shape, norm_stats)
        ref_mean_ln, target_value = expanded_norm_stats

        # apply transformation from unnormalized to normalized
        if derivative_num == 0:
            out = self.batch_center_map(values, ref_mean_ln, target_value)

        elif derivative_num == 1:
            out = self.batch_center_map_grad(values, ref_mean_ln, target_value)

        elif derivative_num == 2:
            out = self.batch_center_map_grad_grad(values, ref_mean_ln, target_value)

        else:
            raise ValueError("Can only return up to second derivative. Got " + str(derivative_num))

        # Reintroduce intermediate dimensions from values
        out = out.reshape(original_shape)

        return out

    def inverse_normalization_map(self, values: Tensor, norm_stats: tuple[Tensor], derivative_num: Optional[int] = 0) -> Tensor:
        """
        (Derivative of) inverse normalization based on previously set statistics, i.e. evaluate the map
        x -> x * exp(ln_ref_mean) / target_value
        at all values.

        Args:
            values (Tensor): Values to apply inverse normalization based on previously set statistics to. Shape: [B, ..., D]
            norm_stats (tuple[Tensor]): stats needed for normalization: (ref_mean, ref_std). Shape: [B, D]
            derivative_num (int): Derivative of inverse normalization map to return.

        Returns:
            renormalized_values: Renormalized values. Shape: [B, ..., D]
        """
        # check shapes
        B, D = norm_stats[0].shape
        assert values.ndim >= 2, "Got values.ndim == " + str(values.ndim) + ", expected >=2."
        assert values.shape[0] == B, "Got batch size " + str(values.shape[0]) + ", expected " + str(B)
        assert values.shape[-1] == D, "Got dimension " + str(values.shape[-1]) + ", expected " + str(D)

        # Squash intermediate dimensions from values
        values, original_shape = self.squash_intermediate_dims(values)

        expanded_norm_stats: tuple[Tensor] = self.expand_norm_stats(values.shape, norm_stats)
        ref_mean_ln, target_value = expanded_norm_stats

        # apply transformation from normalized to unnormalized
        if derivative_num == 0:
            out = self.batch_inv_center_map(values, ref_mean_ln, target_value)

        elif derivative_num == 1:
            out = self.batch_inv_center_map_grad(values, ref_mean_ln, target_value)

        elif derivative_num == 2:
            out = self.batch_inv_center_map_grad_grad(values, ref_mean_ln, target_value)

        else:
            raise ValueError("Can only return up to second derivative. Got " + str(derivative_num))

        # Reintroduce intermediate dimensions from values
        out = out.reshape(original_shape)

        return out


@optree.dataclasses.dataclass(namespace="fimsde", eq=False)
class SDEConcepts:
    """
    Stores SDE concepts, i.e. drift and diffusion, at some locations.
    Optionally store (learned) variances, indicating certainty.
    A flag keeps track of the normalization status of these concepts.
    """

    # all attributes are of shape [B, ..., D]
    locations: Tensor
    drift: Tensor
    diffusion: Tensor
    log_var_drift: Tensor | None = None
    log_var_diffusion: Tensor | None = None
    normalized: bool = optree.dataclasses.field(default=False, pytree_node=False)

    def __eq__(self, other: object) -> bool:
        """
        Define equality by closeness of attributes. If log_var... is only in one, return False.
        """
        rtol: float = 1e-5
        atol: float = 1e-6

        is_equal: bool = True

        is_equal = is_equal and torch.allclose(self.locations, other.locations, atol=atol, rtol=rtol)
        is_equal = is_equal and torch.allclose(self.drift, other.drift, atol=atol, rtol=rtol)
        is_equal = is_equal and torch.allclose(self.diffusion, other.diffusion, atol=atol, rtol=rtol)

        if self.log_var_drift is not None and other.log_var_drift is not None:
            is_equal = is_equal and torch.allclose(self.log_var_drift, other.log_var_drift, atol=atol, rtol=rtol)

        elif self.log_var_drift is None and other.log_var_drift is None:
            pass

        else:
            is_equal = False

        if self.log_var_diffusion is not None and other.log_var_diffusion is not None:
            is_equal = is_equal and torch.allclose(self.log_var_diffusion, other.log_var_diffusion, atol=atol, rtol=rtol)

        elif self.log_var_diffusion is None and other.log_var_diffusion is None:
            pass

        else:
            is_equal = False

        is_equal = is_equal and (self.normalized == other.normalized)

        return is_equal

    @classmethod
    def from_dict(cls, data: dict | None, normalized: Optional[bool] = False):
        """
        Construct SDEConcepts from data dict.

        Args:
            data (dict | None): Data to extract locations and concepts from. Return None if not passed.
            normalized (bool): Flag if data is normalized. Default: False.

        Returns:
            sde_concepts (SDEConcepts): SDEConcepts with locations, drift and diffusion extracted from data dict.
        """
        if data is not None:
            if (
                data.get("locations") is not None
                and data.get("drift_at_locations") is not None
                and data.get("diffusion_at_locations") is not None
            ):
                return cls(
                    locations=data["locations"],
                    drift=data["drift_at_locations"],
                    diffusion=data["diffusion_at_locations"],
                    log_var_drift=None,
                    log_var_diffusion=None,
                    normalized=normalized,
                )

        else:
            return None

    def _assert_shape(self) -> None:
        """
        Assert that all attributes are of same shape.
        """
        broadcasted_shape = torch.broadcast_shapes(self.locations.shape, self.drift.shape, self.diffusion.shape)

        if self.log_var_drift is not None:
            broadcasted_shape = torch.broadcast_shapes(broadcasted_shape, self.log_var_drift.shape)

        if self.log_var_diffusion is not None:
            broadcasted_shape = torch.broadcast_shapes(broadcasted_shape, self.log_var_diffusion.shape)

    def _states_transformation(self, states_norm: InstanceNormalization, states_norm_stats: Any, normalize: bool) -> None:
        """
        Apply the transformation to concepts induced by the transformation of the states from the InstanceNormalization.

        Args:
            states_norm (InstanceNormalization): Underlying transformations of states.
            states_norm_stats (Any): Statistics used by states_norm.
            normalize (bool): If true, applies transformation induced by normalization, else by the inverse of normalization.
        """
        self._assert_shape()

        # evaluate gradient of the normalization map at the respective locations
        if normalize is True:
            grad = states_norm.normalization_map(self.locations, states_norm_stats, derivative_num=1)
            grad_grad = states_norm.normalization_map(self.locations, states_norm_stats, derivative_num=2)

        else:
            grad = states_norm.inverse_normalization_map(self.locations, states_norm_stats, derivative_num=1)
            grad_grad = states_norm.inverse_normalization_map(self.locations, states_norm_stats, derivative_num=2)

        log_grad = torch.log(grad)

        # transform equation by Ito's formula
        self.drift = self.drift * grad + 1 / 2 * self.diffusion**2 * grad_grad
        self.diffusion = self.diffusion * grad

        if self.log_var_drift is not None:
            self.log_var_drift = self.log_var_drift + 2 * log_grad

        if self.log_var_diffusion is not None:
            self.log_var_diffusion = self.log_var_diffusion + 2 * log_grad

        self._assert_shape()

    def _times_transformation(self, times_norm: InstanceNormalization, times_norm_stats: Any, normalize: bool) -> None:
        """
        Apply the transformation to concepts induced by the transformation of time from the InstanceNormalization.

        Args:
            times_norm (InstanceNormalization): Underlying transformations of time.
            times_norm_stats (Any): Statistics used by times_norm.
            normalize (bool): If true, applies transformation induced by normalization, else by the inverse of normalization.
        """
        self._assert_shape()

        # need gradient of reverse map for transformation
        # as concepts are purely state dependent, can pass in dummy value to time normalization
        dummy_times = torch.zeros_like(self.locations[..., 0].unsqueeze(-1))  # [..., 1]

        if normalize is True:
            inverse_grad = times_norm.inverse_normalization_map(dummy_times, times_norm_stats, derivative_num=1)

        else:
            inverse_grad = times_norm.normalization_map(dummy_times, times_norm_stats, derivative_num=1)

        log_inverse_grad = torch.log(inverse_grad)

        # transform equation by Oksendal, Theorem 8.5.7
        self.drift = self.drift * inverse_grad
        self.diffusion = self.diffusion * torch.sqrt(inverse_grad)

        if self.log_var_drift is not None:
            self.log_var_drift = self.log_var_drift + 2 * log_inverse_grad

        if self.log_var_diffusion is not None:
            self.log_var_diffusion = self.log_var_diffusion + log_inverse_grad

        self._assert_shape()

    def _locations_transformation(self, states_norm: InstanceNormalization, states_norm_stats: Any, normalize: bool) -> None:
        """
        Apply transformation of states to the locations at which equation concepts are evaluated at.

        Args:
            states_norm (InstanceNormalization): Specifies transformations of states.
            states_norm_stats (Any): Statistics used by states_norm.
            normalize (bool): If true, applies transformation induced by normalization, else by the inverse of normalization.
        """
        self._assert_shape()

        if normalize is True:
            self.locations = states_norm.normalization_map(self.locations, states_norm_stats)

        else:
            self.locations = states_norm.inverse_normalization_map(self.locations, states_norm_stats)

        self._assert_shape()

    def normalize(
        self, states_norm: InstanceNormalization, states_norm_stats: Any, times_norm: InstanceNormalization, times_norm_stats: Any
    ) -> None:
        """
        Normalize locations and concepts if not already normalized.

        Args:
            states_norm, times_norm (InstanceNormalization): Specifies normalizations to apply.
            states_norm_stats, times_norm_stats (Any): Statistics used by the normalizations.
        """
        if self.normalized is False:
            self._states_transformation(states_norm, states_norm_stats, normalize=True)
            self._locations_transformation(states_norm, states_norm_stats, normalize=True)
            self._times_transformation(times_norm, times_norm_stats, normalize=True)

            self.normalized = True

    def renormalize(
        self, states_norm: InstanceNormalization, states_norm_stats: Any, times_norm: InstanceNormalization, times_norm_stats: Any
    ) -> None:
        """
        Reormalize locations and concepts if not already renormalized.

        Args:
            states_norm, times_norm (InstanceNormalization): Specifies renormalizations to apply.
            states_norm_stats, times_norm_stats (Any): Statistics used by the normalizations.
        """
        if self.normalized is True:
            self._states_transformation(states_norm, states_norm_stats, normalize=False)
            self._locations_transformation(states_norm, states_norm_stats, normalize=False)
            self._times_transformation(times_norm, times_norm_stats, normalize=False)

            self.normalized = False


class FIMSDEConfig(PretrainedConfig):
    """
    FIMSDEConfig is a configuration class for the FIMSDE model.

    Attributes:
        name (str): Name of the configuration. Default is "FIMSDE".
        experiment_name (str): Name of the experiment. Default is "sde".
        experiment_dir (str): Directory for experiment results. Default is results_path.
        max_dimension (int): Maximum input dimensions. Default is 3.
        max_time_steps (int): Maximum time steps. Default is 128.
        max_location_size (int): Maximum location size. Default is 1024.
        max_num_paths (int): Maximum number of paths. Default is 30.
        model_embedding_size (int): Embedding size used throughout model. Default is 64.
        delta_time_only (bool): Only use delta-time as time encoding. Default is False.
        layer_norms_in_phi_0 (bool): Use layer norms in encodings for phi_0. Default is False.
        separate_phi_0_encoders (bool): Use separate copies of encoding modules for phi_0s. Default is False.
        phi_0t (dict): Config for phi_0t. Default is {}.
        phi_0x (dict): Config for phi_0x. Default is {}.
        psi_1 (dict): Config for psi_1. Default is {}.
        phi_1x (dict): Config for phi_1x. Default is {}.
        learn_vf_var (bool): Learn also (log) var of drift and diffusion. default is False
        operator_specificity (str): "all", "per_concept" or "per_head" specifying separate operators per heads. Default is "all".
        operator (dict): Config for operator. Default is {}.
        non_negative_diffusion_by (str): Specify if and how to make estimated diffusion non-negative. Defaults to None.
        states_norm (dict): Config for states instance normalization. Default is MinMaxNormalization.
        times_norm (dict): Config for times instance normalization. Default is MinMaxNormalization.
        times_norm_on_deltas (dict): To calculate times normalization on delta-time instead. Default is False.
        loss_filter_nans (bool): Default is True.
        learnable_loss_scales (dict | None): Config for AttentionOperator defining learnable loss scales per location.
        detach_learnable_loss_scale_heads (bool): Default is True.
        single_learnable_loss_scale_head (bool): Default is False
        learnable_loss_scale_mlp (dict): Default is None
        data_delta_t (float): Fine grid delta t of data generation.
        divide_drift_loss_by_diffusion (bool): Default is True
        num_epochs (int): Number of epochs. Default is 2.
        learning_rate (float): Learning rate. Default is 1.0e-5.
        weight_decay (float): Weight decay. Default is 1.0e-4.
        dropout_rate (float): Dropout rate. Default is 0.1.
        loss_type (str): Type of loss. Default is "rmse".
        log_images_every_n_epochs (int): Log images every n epochs. Default is 2.
        ablation_feature_no_X (bool): Zero the embedding of X. Default is False
        ablation_feature_no_dX (bool): Zero the embedding of dX. Default is False
        ablation_feature_no_dX_2 (bool): Zero the embedding of dX^2. Default is False
        ablation_feature_no_dt (bool): Zero the embedding of dt. Default is False
        train_with_normalized_head (bool): Train with normalized head. Default is True.
        skip_nan_grads (bool): Skip optimizer update if (at least one) gradient is Nan. Default is True.
        dt_pipeline (float): Time step for pipeline. Default is 0.01.
        number_of_time_steps_pipeline (int): Number of time steps in the pipeline. Default is 128.
        evaluate_with_unnormalized_heads (bool): Evaluate with unnormalized heads. Default is True.
        finetune (bool): Indicates fintuning on the observations. Default is False.
        finetune_only_on_one_step_ahead (bool): Indicates finetuning only on one-step-ahead prediction. Default is True.
        finetune_samples_count (int): Number of samples to generate for finetuning. Default is 1.
        finetune_em_steps (int): Number of EM steps between two observations for finetuning. Default is 1.
        finetune_detach_diffusion (bool): Detach diffusion head for finetuning. Default is False
    """

    model_type = "fimsde"

    def __init__(
        self,
        name: str = "FIMSDE",
        experiment_name: str = "sde",  # Todo: remove
        experiment_dir: str = rf"{results_path}",  # Todo: remove
        max_dimension: int = 3,
        max_time_steps: int = 128,  # Todo: remove
        max_location_size: int = 1024,  # Todo: remove
        max_num_paths: int = 30,  # Todo: remove
        model_embedding_size: int = 64,
        delta_time_only: bool = False,  # Todo: remove, we use True
        layer_norms_in_phi_0: bool = False,  # Todo: remove, we use False
        separate_phi_0_encoders: bool = False,  # Todo: remove, True
        phi_0t: dict = {},
        phi_0x: dict = {"hidden_layers": [64]},
        psi_1: dict = {"name": "PathTransformer", "num_layers": 2, "layer": {}},
        phi_1x: dict = {"hidden_layers": [64]},
        learn_vf_var: bool = False,  # Todo: remove, we use False
        operator_specificity: str = "all",  # Todo: remove, we use "per_concept"
        operator: dict = {"attention": {"nhead": 2}, "projection": {"hidden_layers": [64]}},
        non_negative_diffusion_by: Optional[str] = None,
        states_norm: dict = {"name": "fim.models.sde.MinMaxNormalization"},
        times_norm: dict = {"name": "fim.models.sde.MinMaxNormalization", "normalized_min": 0, "normalized_max": 1},
        times_norm_on_deltas: bool = False,
        loss_filter_nans: bool = True,
        learnable_loss_scales: Optional[dict] = None,
        detach_learnable_loss_scale_heads: Optional[bool] = True,
        single_learnable_loss_scale_head: Optional[bool] = False,
        learnable_loss_scale_mlp=None,  # Todo: remove, we use None
        data_delta_t: float = 0.003906,  # 10 / 128 / 20
        num_epochs: int = 2,  # training variables (MAYBE SEPARATED LATER)  Todo: remove
        learning_rate: float = 1.0e-5,  # Todo: remove
        weight_decay: float = 1.0e-4,  # Todo: remove
        dropout_rate: float = 0.1,  # Todo: remove
        loss_type: str = "rmse",
        log_images_every_n_epochs: int = 2,  # Todo: remove
        ablation_feature_no_X: bool = False,
        ablation_feature_no_dX: bool = False,
        ablation_feature_no_dX_2: bool = False,
        ablation_feature_no_dt: bool = False,
        train_with_normalized_head: bool = True,
        skip_nan_grads: bool = True,
        dt_pipeline: float = 0.01,  # Todo: remove
        number_of_time_steps_pipeline: int = 128,  # Todo: remove
        evaluate_with_unnormalized_heads: bool = True,  # Todo: remove
        finetune: bool = False,
        finetune_only_on_one_step_ahead: bool = True,
        finetune_samples_count: int = 1,
        finetune_em_steps: int = 1,
        finetune_detach_diffusion: bool = False,
        finetune_on_likelihood: bool = False,
        **kwargs,
    ):
        self.name = name
        self.experiment_name = experiment_name
        self.experiment_dir = experiment_dir
        self.max_dimension = max_dimension
        self.max_time_steps = max_time_steps
        self.max_location_size = max_location_size
        self.max_num_paths = max_num_paths
        self.model_embedding_size = model_embedding_size
        self.delta_time_only = delta_time_only
        self.layer_norms_in_phi_0 = layer_norms_in_phi_0
        self.separate_phi_0_encoders = separate_phi_0_encoders
        self.phi_0t = phi_0t
        self.phi_0x = phi_0x
        self.psi_1 = psi_1
        self.phi_1x = phi_1x
        self.learn_vf_var = learn_vf_var
        self.operator_specificity = operator_specificity
        self.operator = operator
        self.non_negative_diffusion_by = non_negative_diffusion_by
        # normalization
        self.states_norm = states_norm
        self.times_norm = times_norm
        self.times_norm_on_deltas = times_norm_on_deltas
        # regularization
        self.loss_filter_nans = loss_filter_nans
        self.learnable_loss_scales = learnable_loss_scales
        self.detach_learnable_loss_scale_heads = detach_learnable_loss_scale_heads
        self.single_learnable_loss_scale_head = single_learnable_loss_scale_head
        self.learnable_loss_scale_mlp = learnable_loss_scale_mlp
        self.data_delta_t = data_delta_t
        # training variables
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.loss_type = loss_type
        self.log_images_every_n_epochs = log_images_every_n_epochs
        self.train_with_normalized_head = train_with_normalized_head
        self.skip_nan_grads = skip_nan_grads
        self.dt_pipeline = dt_pipeline
        self.number_of_time_steps_pipeline = number_of_time_steps_pipeline
        self.evaluate_with_unnormalized_heads = evaluate_with_unnormalized_heads
        # ablations
        self.ablation_feature_no_X = ablation_feature_no_X
        self.ablation_feature_no_dX = ablation_feature_no_dX
        self.ablation_feature_no_dX_2 = ablation_feature_no_dX_2
        self.ablation_feature_no_dt = ablation_feature_no_dt
        # finetuning
        self.finetune = finetune
        self.finetune_only_on_one_step_ahead = finetune_only_on_one_step_ahead
        self.finetune_samples_count = finetune_samples_count
        self.finetune_em_steps = finetune_em_steps
        self.finetune_detach_diffusion = finetune_detach_diffusion
        self.finetune_on_likelihood = finetune_on_likelihood
        super().__init__(**kwargs)


# 3. Model Following FIM conventions
# class FIMSDE(AModel):
class FIMSDE(AModel):
    """
    Stochastic Differential Equation Trainining
    """

    config_class = FIMSDEConfig

    def __init__(
        self,
        config: FIMSDEConfig,
        device_map: torch.device = None,
        **kwargs,
    ):
        AModel.__init__(self, config, **kwargs)

        # for backward compatibility
        self.learnable_loss_scales = config.learnable_loss_scales if hasattr(config, "learnable_loss_scales") else None
        self.detach_learnable_loss_scale_heads = (
            config.detach_learnable_loss_scale_heads if hasattr(config, "detach_learnable_loss_scale_heads") else True
        )
        self.single_learnable_loss_scale_head = (
            config.single_learnable_loss_scale_head if hasattr(config, "single_learnable_loss_scale_head") else False
        )
        self.times_norm_on_deltas = config.times_norm_on_deltas if hasattr(config, "times_norm_on_deltas") else False
        self.divide_drift_loss_by_diffusion = (
            config.divide_drift_loss_by_diffusion if hasattr(config, "divide_drift_loss_by_diffusion") else True
        )
        self.learnable_loss_scale_mlp = config.learnable_loss_scale_mlp if hasattr(config, "learnable_loss_scale_mlp") else None
        self.data_delta_t = config.data_delta_t if hasattr(config, "data_delta_t") else None
        self.ablation_feature_no_X = config.ablation_feature_no_X if hasattr(config, "ablation_feature_no_X") else False
        self.ablation_feature_no_dX = config.ablation_feature_no_dX if hasattr(config, "ablation_feature_no_dX") else False
        self.ablation_feature_no_dX_2 = config.ablation_feature_no_dX_2 if hasattr(config, "ablation_feature_no_dX_2") else False
        self.ablation_feature_no_dt = config.ablation_feature_no_dt if hasattr(config, "ablation_feature_no_dt") else False

        self.finetune = config.finetune if hasattr(config, "finetune") else False
        self.finetune_only_on_one_step_ahead = (
            config.finetune_only_on_one_step_ahead if hasattr(config, "finetune_only_on_one_step_ahead") else True
        )
        self.finetune_samples_count = config.finetune_samples_count if hasattr(config, "finetune_samples_count") else 1
        self.finetune_em_steps = config.finetune_em_steps if hasattr(config, "finetune_em_steps") else 1
        self.finetune_detach_diffusion = config.finetune_detach_diffusion if hasattr(config, "finetune_detach_diffusion") else False
        self.finetune_on_likelihood = config.finetune_on_likelihood if hasattr(config, "finetune_on_likelihood") else False

        # # Save hyperparameters
        # self.save_hyperparameters()

        # Set hyperparameters
        if isinstance(config, dict):
            self.config = FIMSDEConfig(**config)
        else:
            self.config = config

        self._create_modules()

        # Set a dataset for fixed evaluation
        # self.target_data = generate_all(self.config.max_time_steps, self.config.max_num_paths)

        if device_map is not None:
            self.to(device_map)

    def _create_modules(self):
        config = deepcopy(self.config)  # model loading won't work without it

        # states and times normalization
        states_norm_config = config.states_norm
        self.states_norm: InstanceNormalization = create_class_instance(states_norm_config.pop("name"), states_norm_config)

        times_norm_config = config.times_norm
        self.times_norm: InstanceNormalization = create_class_instance(times_norm_config.pop("name"), times_norm_config)

        # observation times encoder
        phi_0t_out_features = config.model_embedding_size if config.separate_phi_0_encoders is False else config.model_embedding_size // 4

        phi_0t_module_name = config.phi_0t.get("name", "SineTimeEncoding")
        if "SineTimeEncoding" in phi_0t_module_name:
            phi_0t_encoder = SineTimeEncoding(out_features=phi_0t_out_features)

        else:
            config.phi_0t.update({"in_features": 1, "out_features": phi_0t_out_features})
            phi_0t_encoder = create_class_instance(config.phi_0t.pop("name"), config.phi_0t)

        if config.layer_norms_in_phi_0:
            phi_0t_layer_norm = nn.LayerNorm(phi_0t_out_features, dtype=torch.float32)
            self.phi_0t = nn.Sequential(phi_0t_encoder, phi_0t_layer_norm)

        else:
            self.phi_0t = phi_0t_encoder

        # observation values encoder; encode X, del_X and (del_X)**2
        phi_0x_in_features = 3 * config.max_dimension if config.separate_phi_0_encoders is False else config.max_dimension
        phi_0x_out_features = config.model_embedding_size if config.separate_phi_0_encoders is False else config.model_embedding_size // 4
        config.phi_0x.update({"in_features": phi_0x_in_features, "out_features": phi_0x_out_features})

        if config.separate_phi_0_encoders is False:
            phi_0x_encoder = create_class_instance(config.phi_0x.pop("name"), config.phi_0x)

            if config.layer_norms_in_phi_0:
                phi_0x_layer_norm = nn.LayerNorm(config.model_embedding_size, dtype=torch.float32)
                self.phi_0x = nn.Sequential(phi_0x_encoder, phi_0x_layer_norm)

            else:
                self.phi_0x = phi_0x_encoder

        else:
            phi_0x_x_encoder_config = deepcopy(config.phi_0x)
            phi_0x_dx_encoder_config = deepcopy(config.phi_0x)
            phi_0x_dx2_encoder_config = deepcopy(config.phi_0x)

            self.phi_0x_x = create_class_instance(phi_0x_x_encoder_config.pop("name"), phi_0x_x_encoder_config)
            self.phi_0x_dx = create_class_instance(phi_0x_dx_encoder_config.pop("name"), phi_0x_dx_encoder_config)
            self.phi_0x2_dx = create_class_instance(phi_0x_dx2_encoder_config.pop("name"), phi_0x_dx2_encoder_config)

        # combine times and values embedding to config.model_embedding_size
        if config.delta_time_only is True:  # combine spatial and temporal differences
            phi_0_projection_in_features = phi_0t_out_features + phi_0x_out_features

        else:  # combine spatial, temporal and temporal differences
            phi_0_projection_in_features = 2 * phi_0t_out_features + phi_0x_out_features

        phi_0_projection = nn.Linear(phi_0_projection_in_features, config.model_embedding_size)

        if config.layer_norms_in_phi_0:
            phi_0_layer_norm = nn.LayerNorm(config.model_embedding_size, dtype=torch.float32)
            self.phi_0 = nn.Sequential(phi_0_projection, phi_0_layer_norm)

        else:
            self.phi_0 = phi_0_projection

        # observations transformer encoder
        self.psi_1_module_name = config.psi_1.get("name", "PathTransformer")
        assert self.psi_1_module_name in [
            "PathTransformer",
            "SetTransformer",
            "CombinedPathTransformer",
            "None",
        ], f"Got {self.psi_1_module_name}."

        num_layers: int = config.psi_1.get("num_layers")
        layer_config = config.psi_1.get("layer")

        if self.psi_1_module_name == "PathTransformer":
            psi_1_transformer_layer = nn.TransformerEncoderLayer(d_model=config.model_embedding_size, batch_first=True, **layer_config)
            self.psi_1 = nn.TransformerEncoder(psi_1_transformer_layer, num_layers=num_layers)

        elif self.psi_1_module_name == "SetTransformer":
            self.psi_1 = InducedSetTransformerEncoder(d_model=config.model_embedding_size, num_layers=num_layers, layer=layer_config)
            self.psi_1_layer_norm = nn.LayerNorm(config.model_embedding_size, dtype=torch.float32)

        elif self.psi_1_module_name == "CombinedPathTransformer":
            psi_1_transformer_layer = ResidualEncoderLayer(d_model=config.model_embedding_size, batch_first=True, **layer_config)
            self.psi_1 = nn.TransformerEncoder(psi_1_transformer_layer, num_layers=num_layers)

        else:
            pass

        # locations encoder
        phi_1x_in_features = config.max_dimension
        phi_1x_out_features = config.model_embedding_size
        config.phi_1x.update({"in_features": phi_1x_in_features, "out_features": phi_1x_out_features})

        phi_1x_encoder = create_class_instance(config.phi_1x.pop("name"), config.phi_1x)

        if config.layer_norms_in_phi_0:
            phi_1x_layer_norm = nn.LayerNorm(config.model_embedding_size, dtype=torch.float32)
            self.phi_1x = nn.Sequential(phi_1x_encoder, phi_1x_layer_norm)

        else:
            self.phi_1x = phi_1x_encoder

        # operator(s) evaluated at locations
        assert config.operator_specificity in ["all", "per_concept", "per_head"]
        concepts_outs = 2 if config.learn_vf_var is True else 1
        if config.operator_specificity == "all":
            self.operator = AttentionOperator(
                embed_dim=config.model_embedding_size, out_features=2 * concepts_outs * config.max_dimension, **deepcopy(config.operator)
            )
            self.apply_operator = self.heads_projection_all_functions

        elif config.operator_specificity == "per_concept":
            self.operator_drift = AttentionOperator(
                embed_dim=config.model_embedding_size, out_features=concepts_outs * config.max_dimension, **deepcopy(config.operator)
            )
            self.operator_diffusion = AttentionOperator(
                embed_dim=config.model_embedding_size, out_features=concepts_outs * config.max_dimension, **deepcopy(config.operator)
            )
            self.apply_operator = self.heads_projection_per_concept

        else:
            self.operator_drift = AttentionOperator(
                embed_dim=config.model_embedding_size, out_features=config.max_dimension, **deepcopy(config.operator)
            )
            self.operator_diffusion = AttentionOperator(
                embed_dim=config.model_embedding_size, out_features=config.max_dimension, **deepcopy(config.operator)
            )
            if self.learn_vf_var is True:
                self.operator_log_var_drift = AttentionOperator(
                    embed_dim=config.model_embedding_size, out_features=config.max_dimension, **deepcopy(config.operator)
                )
                self.operator_log_var_diffusion = AttentionOperator(
                    embed_dim=config.model_embedding_size, out_features=config.max_dimension, **deepcopy(config.operator)
                )
            self.apply_operator = self.heads_projection_per_head

        if self.learnable_loss_scales is not None:
            self.operator_loss_scale_drift = AttentionOperator(
                embed_dim=config.model_embedding_size, out_features=1, **deepcopy(self.learnable_loss_scales)
            )
            if self.single_learnable_loss_scale_head is True:
                self.operator_loss_scale_diffusion = self.operator_loss_scale_drift

            else:
                self.operator_loss_scale_diffusion = AttentionOperator(
                    embed_dim=config.model_embedding_size, out_features=1, **deepcopy(self.learnable_loss_scales)
                )

        if self.learnable_loss_scale_mlp is not None:
            self.learnable_loss_scale_mlp.update({"in_features": 2, "out_features": 1})
            self.learnable_loss_scale_mlp = create_class_instance(self.learnable_loss_scale_mlp.pop("name"), self.learnable_loss_scale_mlp)

    def heads_projection_all_functions(
        self, locations_encoding: Tensor, observations_encoding: Tensor, observations_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Combine encodings of locations and observations via attention and projection to an value of size self.config.max_dimension for each head.
        Use single attention and projection network and split the output into the 4 heads.

        Args:
            locations_encoding (Tensor): Encoding for each location on grid. Shape: [B, G, H]
            observations_encoding (Tensor): Encoding per observation and path. Shape: [B, P, T, H]
            observations_padding_mask (Optional[Tensor]): Mask per observation and path. True indicates value is observed. Shape: [B, P, T, 1]

        Returns:
            heads (Tensor): Heads that define SDEConcepts. Shape each: [B, G, self.config.max_dimension]
        """
        operator_output = self.operator(locations_encoding, observations_encoding, observations_padding_mask=observations_padding_mask)
        if self.config.learn_vf_var:
            drift_estimator, diffusion_estimator, log_var_drift_estimator, log_var_diffusion_estimator = torch.chunk(
                operator_output, 4, dim=-1
            )

        else:
            drift_estimator, diffusion_estimator = torch.chunk(operator_output, 2, dim=-1)
            log_var_drift_estimator, log_var_drift_estimator = None, None

        return drift_estimator, diffusion_estimator, log_var_drift_estimator, log_var_diffusion_estimator

    def heads_projection_per_concept(
        self, locations_encoding: Tensor, observations_encoding: Tensor, observations_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Combine encodings of locations and observations via attention and projection to an value of size self.config.max_dimension for each head.
        Use separate attention and projection network for drift and diffusion.

        Args:
            locations_encoding (Tensor): Encoding for each location on grid. Shape: [B, G, H]
            observations_encoding (Tensor): Encoding per observation and path. Shape: [B, P, T, H]
            observations_padding_mask (Optional[Tensor]): Mask per observation and path. True indicates value is observed. Shape: [B, P, T, 1]

        Returns:
            heads (Tensor): Heads that define SDEConcepts. Shape each: [B, G, self.config.max_dimension]
        """
        drift_output = self.operator_drift(locations_encoding, observations_encoding, observations_padding_mask=observations_padding_mask)
        diffusion_output = self.operator_diffusion(
            locations_encoding, observations_encoding, observations_padding_mask=observations_padding_mask
        )

        if self.config.learn_vf_var:
            drift_estimator, log_var_drift_estimator = torch.chunk(drift_output, 2, dim=-1)
            diffusion_estimator, log_var_diffusion_estimator = torch.chunk(diffusion_output, 2, dim=-1)

        else:
            drift_estimator = drift_output
            diffusion_estimator = diffusion_output
            log_var_drift_estimator, log_var_diffusion_estimator = None, None

        return drift_estimator, diffusion_estimator, log_var_drift_estimator, log_var_diffusion_estimator

    def heads_projection_per_head(
        self, locations_encoding: Tensor, observations_encoding: Tensor, observations_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Combine encodings of locations and observations via attention and projection to an value of size self.config.max_dimension for each head.
        Use separate attention and projection network for drift, log_var_drift, diffusion and log_var_diffusion.

        Args:
            locations_encoding (Tensor): Encoding for each location on grid. Shape: [B, G, H]
            observations_encoding (Tensor): Encoding per observation and path. Shape: [B, P, T, H]
            observations_padding_mask (Optional[Tensor]): Mask per observation and path. True indicates value is observed. Shape: [B, P, T, 1]

        Returns:
            heads (Tensor): Heads that define SDEConcepts. Shape each: [B, G, self.config.max_dimension]
        """
        drift_estimator = self.operator_drift(
            locations_encoding, observations_encoding, observations_padding_mask=observations_padding_mask
        )
        diffusion_estimator = self.operator_diffusion(
            locations_encoding, observations_encoding, observations_padding_mask=observations_padding_mask
        )

        if self.config.learn_vf_var:
            log_var_drift_estimator = self.operator_log_var_drift(
                locations_encoding, observations_encoding, observations_padding_mask=observations_padding_mask
            )
            log_var_diffusion_estimator = self.operator_log_var_diffusion(
                locations_encoding, observations_encoding, observations_padding_mask=observations_padding_mask
            )

        else:
            log_var_drift_estimator, log_var_diffusion_estimator = None, None

        return drift_estimator, diffusion_estimator, log_var_drift_estimator, log_var_diffusion_estimator

    @staticmethod
    @torch.profiler.record_function("fimsde_fill_masekd_values")
    def _fill_masked_values(data: dict):
        if "obs_mask" in data.keys() and data["obs_mask"] is not None:
            obs_mask = data["obs_mask"].bool()

            # for sanity, removed masked out values
            obs_times = obs_mask * data["obs_times"]
            obs_values = obs_mask * data["obs_values"]

            # backward fill masked values s.t. values differences and squares respect masks
            obs_times = backward_fill_masked_values(obs_times, obs_mask)
            obs_values = backward_fill_masked_values(obs_values, obs_mask)

        else:
            obs_mask = torch.ones_like(data["obs_times"]).bool()
            obs_times = data["obs_times"]
            obs_values = data["obs_values"]

        return obs_times, obs_values, obs_mask

    @torch.profiler.record_function("preprocess_inputs")
    def preprocess_inputs(self, data: dict, locations: Optional[Tensor] = None) -> tuple[Tensor, Any]:
        """
        Preprocessing of forward inputs. Includes:
            1. Backward fill on masked / padded observations.
            2. Extracting instance normalization statistics from data.
            3. Instance normalization of observations and locations.

        Args: See arguments of self.forward(...).

        Returns:
            Preprocessed inputs: obs_times, obs_values, obs_mask, locations
            Instance normalization statistics: states_norm_stats, times_norm_stats
        """
        assert data["obs_values"].shape[-1] <= self.config.max_dimension, (
            f"Can not process observations of dim >{self.config.max_dimension}. Got {data['obs_values'].shape[-1]}."
        )

        obs_times, obs_values, obs_mask = self._fill_masked_values(data)

        # Default to passed locations, otherwise use data["locations"]
        if locations is None:
            locations = data.get("locations")

        # Expand input tensors to max_dimension for inference (assume train targets have correct shape)
        if obs_values.shape[-1] < self.config.max_dimension:
            missing_dims = self.config.max_dimension - obs_values.shape[-1]

            obs_pad = torch.broadcast_to(torch.zeros_like(obs_values[..., 0][..., None]), obs_values.shape[:-1] + (missing_dims,))
            obs_values = torch.concatenate([obs_values, obs_pad], dim=-1)

        if locations is not None and locations.shape[-1] < self.config.max_dimension:
            missing_dims = self.config.max_dimension - locations.shape[-1]

            locations_pad = torch.broadcast_to(torch.zeros_like(locations[..., 0][..., None]), locations.shape[:-1] + (missing_dims,))
            locations = torch.concatenate([locations, locations_pad], dim=-1)

        # Instance normalization
        states_norm_stats: Any = self.states_norm.get_norm_stats(obs_values, obs_mask)
        obs_values = self.states_norm.normalization_map(obs_values, states_norm_stats)

        if locations is not None:
            locations = self.states_norm.normalization_map(locations, states_norm_stats)

        if self.times_norm_on_deltas is True:
            delta_times = obs_times[:, :, 1:, :] - obs_times[:, :, :-1, :]  # obs_times are backward filled
            delta_mask = obs_mask[:, :, :-1, :]
            times_norm_stats: InstanceNormalization = self.times_norm.get_norm_stats(delta_times, delta_mask)
            # transformation can still be applied to obs_times, as delta_times will be recomputed later

        else:
            times_norm_stats: InstanceNormalization = self.times_norm.get_norm_stats(obs_times, obs_mask)

        obs_times = self.times_norm.normalization_map(obs_times, times_norm_stats)

        return obs_times, obs_values, obs_mask, locations, states_norm_stats, times_norm_stats

    @torch.profiler.record_function("get_paths_encoding")
    def get_paths_encoding(self, obs_times: Tensor, obs_values: Tensor, obs_mask: Optional[Tensor] = None) -> Tensor:
        """
        Obtain embedding of all observed values in all paths.

        Args:
            obs_times (Tensor): observation times of obs_values. Shape: [B, P, T, 1]
            obs_values (Tensor): observation values. optionally with noise. Shape: [B, P, T, D]
            obs_mask (Tensor): mask for padded observations. == 1.0 if observed. Shape: [B, P, T, 1]
            where B: batch size P: number of paths T: number of time steps dimensions

        Returns:
            H (Tensor): Embedding of observations processed by transformer. Shape: [B, P, T-1, psi_1_tokes_dim]
        """
        obs_times = obs_times.to(torch.float32)  # Todo: should be handled better
        obs_values = obs_values.to(torch.float32)  # Todo: should be handled better

        B, P, T, D = obs_values.shape

        # Embedded values; include difference and squared difference to next observation -> drop last observation
        X = obs_values[:, :, :-1, :]
        dX = obs_values[:, :, 1:, :] - obs_values[:, :, :-1, :]
        dX2 = dX**2

        if self.config.separate_phi_0_encoders is False:
            x_full = torch.cat([X, dX, dX2], dim=-1)  # [B, P, T, 3*D]
            spatial_encoding = self.phi_0x(x_full)  # [B, P, T, model_embedding_size]

        else:
            X = self.phi_0x_x(X)
            dX = self.phi_0x_dx(dX)
            dX2 = self.phi_0x2_dx(dX2)

            if self.ablation_feature_no_X is True:
                X = torch.zeros_like(X)

            if self.ablation_feature_no_dX is True:
                dX = torch.zeros_like(dX)

            if self.ablation_feature_no_dX_2 is True:
                dX2 = torch.zeros_like(dX2)

            spatial_encoding = torch.concatenate([X, dX, dX2], dim=-1)

        # separate time encoding; drop last time because dropped for values
        if self.config.delta_time_only:
            delta_times = obs_times[:, :, 1:, :] - obs_times[:, :, :-1, :]
            time_encoding = self.phi_0t(delta_times)  # [B, P, T-1, model_embedding_size]

        else:
            absolute_time_encoding = self.phi_0t(obs_times)  # [B, P, T-1, model_embedding_size]
            delta_time_encoding = absolute_time_encoding[:, :, 1:, :] - absolute_time_encoding[:, :, :-1, :]
            time_encoding = torch.concatenate([absolute_time_encoding[:, :, :-1, :], delta_time_encoding], dim=-1)

        if self.ablation_feature_no_dt is True:
            time_encoding = torch.zeros_like(time_encoding)

        # combine time and value encodings per observation
        phi_0_in_features = torch.concat([time_encoding, spatial_encoding], dim=-1)

        if self.config.separate_phi_0_encoders is False:
            U = self.phi_0(phi_0_in_features)

        else:
            U = phi_0_in_features

        assert U.shape == (
            B,
            P,
            T - 1,
            self.config.model_embedding_size,
        ), f"Expect {(B, P, T - 1, self.config.model_embedding_size)}. Got {U.shape}."

        # Transformer processes sequence of observations
        if obs_mask is None:
            key_padding_mask = None

        else:
            # drop last element because it is dropped for values
            obs_mask = obs_mask[:, :, :-1, :]  # [B, P, T-1, 1]

            # revert mask as attention uses other convention
            key_padding_mask = torch.logical_not(obs_mask.bool())  # [B, P, T-1, 1]

        if self.psi_1_module_name == "PathTransformer":
            # apply transformer to all paths separately
            U = U.view(B * P, T - 1, self.config.model_embedding_size)
            key_padding_mask = key_padding_mask.view(B * P, T - 1)  # [B * P, T-1]

            H = self.psi_1(U, src_key_padding_mask=key_padding_mask)  # [B * P, T-1, model_embedding_size]

        elif self.psi_1_module_name == "SetTransformer":
            U = U.view(B, P * (T - 1), self.config.model_embedding_size)
            key_padding_mask = key_padding_mask.view(B, P * (T - 1), 1)

            H = self.psi_1(U, key_padding_mask)  # [B, P * (T-1), H]
            H = self.psi_1_layer_norm(H + U)

        elif self.psi_1_module_name == "CombinedPathTransformer":
            U = U.view(B, P * (T - 1), self.config.model_embedding_size)
            key_padding_mask = key_padding_mask.view(B, P * (T - 1), 1)

            H = self.psi_1(U, src_key_padding_mask=key_padding_mask)  # [B, P * (T-1), H]

        elif self.psi_1_module_name == "None":  # don't apply any transformer
            H = U

        return H.view(B, P, T - 1, self.config.model_embedding_size), obs_mask

    @torch.profiler.record_function("get_estimated_sde_concepts")
    def get_estimated_sde_concepts(
        self,
        locations: Optional[Tensor] = None,
        obs_times: Optional[Tensor] = None,
        obs_values: Optional[Tensor] = None,
        obs_mask: Optional[Tensor] = None,
        dimension_mask: Optional[Tensor] = None,
        paths_encoding: Optional[Tensor] = None,
    ):
        """
        Applies modules to preprocessed model inputs to estimate SDEConcepts at locations.
        SDEConcepts are returned normalized. Padded dimensions of vector fields are set to 0.

        Args:
            locations (Tensor): Locations to extract SDEConcepts at. Shape: [B, G, D]
            obs_times, obs_values, obs_mask (Tensor): See args of self.forward(...). Shape: [B, P, T, 1 or D]
            dimension_mask (Tensor): 0 at padded dimensions of ground-truth data at locations. Shape: [B, G, D]
            paths_encoding (Optional[Tensor]): Encoding of observed paths. If not passed, it is recalculated. Shape: [B, P, T, model_embedding_size]

        Returns:
            estimated_concepts (SDEConcepts): Estimated concepts at locations. Shape: [B, G, D]
            paths_encoding (Tensor): Encoding of observed paths. Shape: [B, P, T, model_embedding_size]
        """

        # get paths encoding only if it was not passed; otherwise ignore obs_...
        if paths_encoding is None:
            assert obs_times is not None
            assert obs_values is not None

            paths_encoding, obs_mask = self.get_paths_encoding(obs_times, obs_values, obs_mask)  # [B, P, T - 1, embed_dim]

            B, P, T, _ = obs_values.shape
            assert paths_encoding.shape == (
                B,
                P,
                T - 1,
                self.config.model_embedding_size,
            ), f"Expect {(B, P, T - 1, self.config.model_embedding_size)}. Got {paths_encoding.shape}"

        # locations encoding
        if locations is not None:
            locations_encoding = self.phi_1x(locations)  # [B, G, embed_dim]

            B, G, _ = locations.shape
            assert locations_encoding.shape == (
                B,
                G,
                self.config.model_embedding_size,
            ), f"Expect {(B, G, self.config.model_embedding_size)}. Got {locations_encoding.shape}"

            # projection to heads
            observations_padding_mask = torch.logical_not(obs_mask)  # revert convention for neural operator
            heads = self.apply_operator(locations_encoding, paths_encoding, observations_padding_mask=observations_padding_mask)
            drift_estimator, diffusion_estimator, log_var_drift_estimator, log_var_diffusion_estimator = heads  # [B, G, D]

            # Optionally make diffusion non-negative, already during training
            if self.config.non_negative_diffusion_by == "clip":
                diffusion_estimator = torch.clip(diffusion_estimator, min=0.0)
            elif self.config.non_negative_diffusion_by == "abs":
                diffusion_estimator = torch.abs(diffusion_estimator)
            elif self.config.non_negative_diffusion_by == "exp":
                diffusion_estimator = torch.exp(diffusion_estimator)
            elif self.config.non_negative_diffusion_by == "softplus":
                diffusion_estimator = torch.nn.functional.softplus(diffusion_estimator)

            # set values at padded dimensions to 0
            if dimension_mask is not None:
                zeros_ = torch.zeros_like(drift_estimator)
                ninf_ = -torch.inf * torch.ones_like(drift_estimator)

                dimension_mask = dimension_mask.bool()
                drift_estimator = torch.where(dimension_mask, drift_estimator, zeros_)
                diffusion_estimator = torch.where(dimension_mask, diffusion_estimator, zeros_)
                if self.config.learn_vf_var is True:
                    log_var_drift_estimator = torch.where(dimension_mask, log_var_drift_estimator, ninf_)
                    log_var_diffusion_estimator = torch.where(dimension_mask, log_var_diffusion_estimator, ninf_)

            estimated_concepts = SDEConcepts(
                locations=locations,
                drift=drift_estimator,
                diffusion=diffusion_estimator,
                log_var_drift=log_var_drift_estimator,
                log_var_diffusion=log_var_diffusion_estimator,
                normalized=True,
            )

        else:
            estimated_concepts = None

        return estimated_concepts, paths_encoding

    @torch.profiler.record_function("fimsde_forward")
    def forward(
        self,
        data: dict,
        locations: Optional[Tensor] = None,
        training: bool = True,
        return_losses: bool = False,
        schedulers: dict | None = None,
        step: int = 0,
    ) -> dict | tuple[SDEConcepts, dict]:
        """
        Args:
            data (dict):
                Required keys:
                    obs_values (Tensor): observation values. optionally with noise. Shape: [B, P, T, D]
                    obs_times (Tensor): observation times of obs_values. Shape: [B, P, T, 1]
                Optional keys:
                    obs_mask (Tensor): mask for padded observations. == 1.0 if observed. Shape: [B, P, T, 1]
                    locations (Tensor): where to evaluate the drift and diffusion function. Shape: [B, G, D]
                Optional keys for loss calculations:
                    drift/diffusion_at_locations (Tensor): ground-truth concepts at locations. Shape: [B, G, D]
                    dimension_mask (Tensor): 0 at padded dimensions of ground-truth data at locations. Shape: [B, G, D]
                where B: batch size P: number of paths T: number of time steps G: location grid size D: dimensions

            locations (Optional[Tensor]): If passed, is prioritized over data.locations. Shape: [B, G, D]
            training (bool): if True returns only dict with losses, including training objective
            return_losses (bool): is True computes and returns losses, even if training is False

            Returns
                estimated_concepts (SDEConcepts): Estimated concepts at locations. Shape: [B, G, D]
                if training == True or return_losses == True return (additionally):
                    losses (dict): training objective has key "loss", other keys are auxiliary for monitoring
        """
        # Instance normalization and appyling mask to observations
        obs_times, obs_values, obs_mask, locations, states_norm_stats, times_norm_stats = self.preprocess_inputs(data, locations)

        # Apply networks to get estimated concepts at locations
        estimated_concepts, paths_encoding = self.get_estimated_sde_concepts(
            locations, obs_times, obs_values, obs_mask, data.get("dimension_mask")
        )

        # Weighting loss from each location
        if locations is not None:
            if self.learnable_loss_scales is not None:
                locations_encoding = self.phi_1x(locations)  # should return this from get_estimated_sde_concepts

                locations_encoding_ = (
                    locations_encoding.clone().detach() if self.detach_learnable_loss_scale_heads is True else locations_encoding
                )
                paths_encoding_ = paths_encoding.clone().detach() if self.detach_learnable_loss_scale_heads is True else paths_encoding
                observations_padding_mask_ = (
                    torch.logical_not(obs_mask[..., :-1, :].contiguous()).clone().detach()
                    if self.detach_learnable_loss_scale_heads is True
                    else torch.logical_not(obs_mask[..., :-1, :].contiguous())
                )

                drift_log_loss_scale_per_location = self.operator_loss_scale_drift(
                    locations_encoding_,
                    paths_encoding_,
                    observations_padding_mask=observations_padding_mask_,
                )  # [B, G, 1]
                diffusion_log_loss_scale_per_location = self.operator_loss_scale_diffusion(
                    locations_encoding_,
                    paths_encoding_,
                    observations_padding_mask=observations_padding_mask_,
                )  # [B, G, 1]

            elif self.learnable_loss_scale_mlp is not None:
                B, P, T, D = obs_values.shape
                B, G, D = locations.shape

                obs_values = torch.broadcast_to(obs_values.view(B, 1, P * T, D), (B, G, P * T, D))  # [B, G, P*T, D]
                obs_mask = torch.broadcast_to(obs_mask.view(B, 1, P * T), (B, G, P * T))  # [B, G, P*T]
                locations = torch.broadcast_to(locations.view(B, G, 1, D), (B, G, P * T, D))  # [B, G, P*T, D]

                dist_to_obs = torch.linalg.vector_norm(obs_values - locations, dim=-1)  # [B, G, P*T]
                min_dist_to_obs = torch.amin(torch.where(obs_mask, dist_to_obs, torch.inf), dim=-1)  # [B, G]
                mean_dist_to_obs = torch.nanmean(torch.where(obs_mask, dist_to_obs, torch.nan), dim=-1)  # [B, G]

                scale_mlp_input = torch.stack([min_dist_to_obs, mean_dist_to_obs], dim=-1)
                log_scale = self.learnable_loss_scale_mlp(scale_mlp_input)  # [B, G, 1]

                drift_log_loss_scale_per_location = log_scale
                diffusion_log_loss_scale_per_location = log_scale

            else:
                drift_log_loss_scale_per_location = torch.zeros_like(locations[..., 0][..., None])
                diffusion_log_loss_scale_per_location = torch.zeros_like(locations[..., 0][..., None])

        else:
            drift_log_loss_scale_per_location = None
            diffusion_log_loss_scale_per_location = None

        # Losses
        target_concepts: SDEConcepts | None = SDEConcepts.from_dict(data)

        if data.get("dimension_mask") is not None:
            dimension_mask = data["dimension_mask"].bool()

        else:
            if estimated_concepts is not None:
                dimension_mask = torch.ones_like(estimated_concepts.drift, dtype=bool)

            else:
                dimension_mask = torch.ones_like(obs_values[:, 0, 0, :][:, None, :])

        if schedulers is not None:
            if "loss_threshold" in schedulers.keys() and training is True:
                loss_threshold = schedulers.get("loss_threshold")(step)

            else:
                loss_threshold = torch.inf

            if "vector_field_max_norm" in schedulers.keys() and training is True:
                vector_field_max_norm = schedulers.get("vector_field_max_norm")(step)

            else:
                vector_field_max_norm = torch.inf

            if "drift_loss_scale" in schedulers.keys() and training is True:
                drift_loss_scale = schedulers.get("drift_loss_scale")(step)

            else:
                drift_loss_scale = 1.0

            if "diffusion_loss_scale" in schedulers.keys() and training is True:
                diffusion_loss_scale = schedulers.get("diffusion_loss_scale")(step)

            else:
                diffusion_loss_scale = 1.0

            if "kl_loss_scale" in schedulers.keys() and training is True:
                kl_loss_scale = schedulers.get("kl_loss_scale")(step)

            else:
                kl_loss_scale = 0.0

            if "short_time_transition_log_likelihood_loss_scale" in schedulers.keys() and training is True:
                short_time_transition_log_likelihood_loss_scale = schedulers.get("short_time_transition_log_likelihood_loss_scale")(step)

            else:
                short_time_transition_log_likelihood_loss_scale = 0.0

            if "one_step_ahead_loss_scale" in schedulers.keys() and training is True:
                one_step_ahead_loss_scale = schedulers.get("one_step_ahead_loss_scale")(step)

            else:
                one_step_ahead_loss_scale = 1.0

            if "whole_path_loss_scale" in schedulers.keys() and training is True:
                whole_path_loss_scale = schedulers.get("whole_path_loss_scale")(step)

            else:
                whole_path_loss_scale = 1.0

        else:
            loss_threshold = torch.inf
            vector_field_max_norm = torch.inf
            drift_loss_scale = 1.0
            diffusion_loss_scale = 1.0
            kl_loss_scale = 0.0
            short_time_transition_log_likelihood_loss_scale = 0.0
            one_step_ahead_loss_scale = 1.0
            whole_path_loss_scale = 1.0

        # Returns
        if training is True:
            if self.finetune is False:
                losses: dict = self.loss(
                    estimated_concepts,
                    target_concepts,
                    states_norm_stats,
                    times_norm_stats,
                    dimension_mask,
                    loss_threshold,
                    vector_field_max_norm,
                    drift_loss_scale,
                    diffusion_loss_scale,
                    kl_loss_scale,
                    short_time_transition_log_likelihood_loss_scale,
                    drift_log_loss_scale_per_location,
                    diffusion_log_loss_scale_per_location,
                )

            else:
                losses: dict = self.finetune_loss(
                    obs_times,
                    obs_values,
                    obs_mask,
                    paths_encoding,
                    one_step_ahead_loss_scale,
                    whole_path_loss_scale,
                    dimension_mask,
                )

            return {"losses": losses}

        else:
            estimated_concepts.renormalize(self.states_norm, states_norm_stats, self.times_norm, times_norm_stats)

            if return_losses is True:
                if self.finetune is False:
                    losses: dict = self.loss(
                        estimated_concepts,
                        target_concepts,
                        states_norm_stats,
                        times_norm_stats,
                        dimension_mask,
                        loss_threshold,
                        vector_field_max_norm,
                        drift_loss_scale,
                        diffusion_loss_scale,
                        kl_loss_scale,
                        short_time_transition_log_likelihood_loss_scale,
                        drift_log_loss_scale_per_location,
                        diffusion_log_loss_scale_per_location,
                    )
                else:
                    losses: dict = self.finetune_loss(
                        obs_times,
                        obs_values,
                        obs_mask,
                        paths_encoding,
                        one_step_ahead_loss_scale,
                        whole_path_loss_scale,
                        dimension_mask,
                    )

                return estimated_concepts, {"losses": losses}

            else:
                return estimated_concepts

    @staticmethod
    def filter_nans_from_vector_fields(estimated: Tensor, log_var_estimated: Tensor | None, target: Tensor, mask: Tensor) -> tuple[Tensor]:
        """
        Filter locations where either estimate or target is Nan (or infinite)

        Args:
            vector field values (Tensor): Vector fields to filter. Shape: [B, G, D]
            mask (Tensor): 0 masks padded values to ignore in percentage calculation. Shape: [B, G, D]

        Returns
            filtered vector field values (Tensor): Shape: [B, G, D]
        """
        # mask Nans per vector field
        estimated = torch.nan_to_num(estimated)
        target = torch.nan_to_num(target)

        if log_var_estimated is not None:
            log_var_estimated = torch.nan_to_num(log_var_estimated)

        return estimated, log_var_estimated, target

    @staticmethod
    def filter_loss_at_locations(loss_at_locations: Tensor, threshold: Optional[float] = None) -> tuple[Tensor]:
        """
        Return mask that filters losses at locations if they are Nan or (optionally) above a threshold. Record statistics about the filtered locations.

        Args:
            loss_at_locations (Tensor): Single loss value per location. Shape: [B, G]
            threshold (Optional[float]): If passed, filter out locations with loss above threshold.

        Returns:
            filtered_loss_at_locations (Tensor): loss_at_locations without Nans. Shape: [B, G]
            filter_mask (Tensor): Masks Nans or above threshold values with 0. Shape: [B, G]
            filtered_loss_locations_perc (Tensor): Percentage of filtered locations in batch. Shape: []
        """
        # mask locations with non-Nan loss values
        loss_is_finite_mask = torch.isfinite(loss_at_locations)  # [B, G]
        loss_at_locations = torch.nan_to_num(loss_at_locations)

        # mask locations below threshold
        if threshold is not None:
            loss_below_threshold_mask = torch.abs(loss_at_locations) <= threshold

        else:
            loss_below_threshold_mask = torch.ones_like(loss_is_finite_mask).bool()

        # combine locations masks
        loss_at_locations_mask = loss_is_finite_mask * loss_below_threshold_mask  # [B, G]

        assert loss_at_locations.ndim == 2
        assert loss_at_locations_mask.ndim == 2

        # record statistics of locations with nan or above threshold loss
        filtered_loss_locations_perc = torch.logical_not(loss_at_locations_mask).mean(dtype=torch.float32)  # []

        return loss_at_locations, loss_at_locations_mask, filtered_loss_locations_perc

    @staticmethod
    def clip_norms_of_vector_field(vector_field: Tensor, clip_threshold: float) -> tuple[Tensor]:
        """
        If norm of vector field value is above clip_threshold, rescale its norm to the clip_threshold.

        Args:
            vector_field (Tensor): vector field values to clip. Shape: [B, G, D]
            clip_threshold (float): target value after clipping

        Returns:
            clipped_vector_field (Tensor): vector field values, rescaled to have norm <= clip_threshold
        """
        vector_field_norm = torch.linalg.norm(torch.clip(torch.abs(vector_field), min=1.0), dim=-1, keepdim=True)  # [B, G, 1]

        vector_field_clip_mask = vector_field_norm > clip_threshold
        vector_field_clipped_norm = vector_field / vector_field_norm * clip_threshold
        vector_field = torch.where(vector_field_clip_mask, vector_field_clipped_norm, vector_field)

        vector_field_clipped_norm_perc = vector_field_clip_mask.mean(dtype=torch.float)

        return vector_field, vector_field_clipped_norm_perc

    def kl_loss(
        self,
        locations: Tensor,
        estimated_drift: Tensor,
        target_drift: Tensor,
        estimated_diffusion: Tensor,
        target_diffusion: Tensor,
        mask: Tensor,
        data_delta_t: Tensor,
        log_loss_scale_per_location: Optional[Tensor] = None,
    ) -> tuple[Tensor]:
        """
        Compute (regularized) KL loss: 1 / 2 * sum_{i=1}^D (f_gt - f_est) ** 2 / g_gt * delta_t + g_est / g_gt  + log(g_gt) - log(g_est) - 1
        Note that target and estimated diffusion already contain a sqrt, i.e. ..._diffusion = sqrt(g_...).

        Args:
            locations, estimated and target vector fields (Tensor): Vector fields to compute loss with.  Shape: [B, G, D]
            mask (Tensor): 0 masks padded values to ignore in loss calculation. Shape: [B, G, D]
            data_delta_t (Tensor): Time scale of dta generation.
            log_loss_scale_per_location (Optional[Tensor]): Multiply the loss at each location by a (potentially) different scale. Shape: [B, G, 1]

        Returns
            loss (Tensor): Loss of vector field. Shape: []
            target_is_infinite_perc (Tensor): Percentage of locations where target vector field is Nan. Shape: []
        """
        # comparing vector field should have 3 dimensions and equal shape
        assert estimated_drift.ndim == target_drift.ndim == 3
        assert estimated_diffusion.ndim == target_diffusion.ndim == 3

        # filter Nans and infinite values
        if self.config.loss_filter_nans:
            estimated_drift, _, target_drift = self.filter_nans_from_vector_fields(estimated_drift, None, target_drift, mask)
            estimated_diffusion, _, target_diffusion = self.filter_nans_from_vector_fields(
                estimated_diffusion, None, target_diffusion, mask
            )

        eps = 1e-4
        estimated_diffusion = torch.clip(estimated_diffusion, min=eps)
        target_diffusion = torch.clip(target_diffusion, min=eps)

        loss_at_locations = (1 / 2) * (
            ((estimated_drift - target_drift) ** 2) / ((target_diffusion) ** 2) * data_delta_t
            + estimated_diffusion**2 / ((target_diffusion) ** 2)
            - 1
            + 2 * torch.log(target_diffusion)
            - 2 * torch.log(estimated_diffusion)
        )  # [B, G, D]

        loss_at_locations = (loss_at_locations * mask).sum(dim=-1)  # [B, G]

        # weight per locations
        assert log_loss_scale_per_location.ndim == 3, f"Got {log_loss_scale_per_location.ndim}"

        loss_at_locations = loss_at_locations * torch.exp(-log_loss_scale_per_location[..., 0])

        assert loss_at_locations.ndim == 2, f"Got {loss_at_locations.ndim}"

        # filter out Nans or above threshold locations from loss
        loss_at_locations, loss_at_locations_mask, filtered_loss_locations_perc = self.filter_loss_at_locations(loss_at_locations, None)

        loss_per_batch_element = torch.sum(loss_at_locations * loss_at_locations_mask, dim=-1)  # [B]
        assert loss_per_batch_element.ndim == 1

        loss = torch.mean(loss_per_batch_element)

        return loss, filtered_loss_locations_perc

    def short_time_trans_ll_loss(
        self,
        locations: Tensor,
        estimated_drift: Tensor,
        target_drift: Tensor,
        estimated_diffusion: Tensor,
        target_diffusion: Tensor,
        mask: Tensor,
        data_delta_t: Tensor,
        log_loss_scale_per_location: Optional[Tensor] = None,
    ) -> tuple[Tensor]:
        """
        Compute (regularized) short-time transition log-likelihood loss:
        let x^prime = x + delta_t * f_gt(x) + sqrt(delta_t) * sqrt(g_gt) * rnd with rnd ~ N(0,1) be the short term simulation of the ground-truth dynamics.
        then compute:
        1 / 2 * sum_{i=1}^D  (x^prime_i - (x_i + delta_t * f_est_i)) ** 2 / (g_est * delta_t) - log(g_est) * delta_t

        Note that target and estimated diffusion already contain a sqrt, i.e. ..._diffusion = sqrt(g_...).

        Args:
            location, estimated and target vector fields (Tensor): Vector fields to compute loss with.  Shape: [B, G, D]
            mask (Tensor): 0 masks padded values to ignore in loss calculation. Shape: [B, G, D]
            data_delta_t (Tensor): Time scale of dta generation.
            log_loss_scale_per_location (Optional[Tensor]): Multiply the loss at each location by a (potentially) different scale. Shape: [B, G, 1]

        Returns
            loss (Tensor): Loss of vector field. Shape: []
            target_is_infinite_perc (Tensor): Percentage of locations where target vector field is Nan. Shape: []
        """
        # comparing vector field should have 3 dimensions and equal shape
        assert estimated_drift.ndim == target_drift.ndim == 3
        assert estimated_diffusion.ndim == target_diffusion.ndim == 3

        # filter Nans and infinite values
        if self.config.loss_filter_nans:
            estimated_drift, _, target_drift = self.filter_nans_from_vector_fields(estimated_drift, None, target_drift, mask)
            estimated_diffusion, _, target_diffusion = self.filter_nans_from_vector_fields(
                estimated_diffusion, None, target_diffusion, mask
            )

        eps = 1e-4
        estimated_diffusion = torch.clip(estimated_diffusion, min=eps)
        target_diffusion = torch.clip(target_diffusion, min=eps)

        # one step simulation of ground-truth system
        gt_step = locations + data_delta_t * target_drift + torch.sqrt(data_delta_t) * target_diffusion * torch.rand_like(locations)
        mean_est_step = locations + data_delta_t * estimated_drift

        loss_at_locations = (1 / 2) * (
            ((gt_step - mean_est_step) ** 2) / (2 * (estimated_diffusion) ** 2 * data_delta_t) + 2 * torch.log(estimated_diffusion)
        )  # [B, G, D]

        loss_at_locations = (loss_at_locations * mask).sum(dim=-1)  # [B, G]

        # weight per locations
        assert log_loss_scale_per_location.ndim == 3, f"Got {log_loss_scale_per_location.ndim}"

        loss_at_locations = loss_at_locations * torch.exp(-log_loss_scale_per_location[..., 0])

        assert loss_at_locations.ndim == 2, f"Got {loss_at_locations.ndim}"

        # filter out Nans or above threshold locations from loss
        loss_at_locations, loss_at_locations_mask, filtered_loss_locations_perc = self.filter_loss_at_locations(loss_at_locations, None)

        loss_per_batch_element = torch.sum(loss_at_locations * loss_at_locations_mask, dim=-1)  # [B]
        assert loss_per_batch_element.ndim == 1

        loss = torch.mean(loss_per_batch_element)

        return loss, filtered_loss_locations_perc

    def vector_field_loss(
        self,
        estimated: Tensor,
        log_var_estimated: Tensor | None,
        target: Tensor,
        mask: Tensor,
        loss_threshold: Optional[float] = None,
        vector_field_max_norm: Optional[float] = None,
        target_diffusion: Optional[Tensor] = None,
        log_loss_scale_per_location: Optional[Tensor] = None,
    ) -> tuple[Tensor]:
        """
        Compute (regularized) loss of vector field values at locations. Return statistics about regularization for monitoring.
        Regularizations:
            Remove Nans and infinite values in passed vector fields.
            Per location, remove Nans and infinite values from calculated loss.
            Per location, remove losses exceeding a threshold.
            Per location, scale loss by a (learned) value.

        Args:
            vector field values (Tensor): Vector fields to compute loss with.  Shape: [B, G, D]
            mask (Tensor): 0 masks padded values to ignore in loss calculation. Shape: [B, G, D]
            loss_threshold (Optional[float]): If passed, set loss per location to 0 if above threshold.
            vector_field_max_norm (Optional[float]): If passed, clip norm of vector fields to this value
            target_diffusion (Optional[Tensor]): If passed, divide loss by norm of target diffusion per location. Shape: [B, G, D]
            log_loss_scale_per_location (Optional[Tensor]): Multiply the loss at each location by a (potentially) different scale. Shape: [B, G, 1]

        Returns
            loss (Tensor): Loss of vector field. Shape: []
            filtered_loss_locations_perc (Tensor): Percentage of locations where loss is above threshold or Nan. Shape: []
            estimated_is_infinite_perc (Tensor): Percentage of locations where estimated vector field is Nan. Shape: []
            target_is_infinite_perc (Tensor): Percentage of locations where target vector field is Nan. Shape: []
            target_clip_perc (Tensor): Percentage of locations where target vector field exceeds norm clip threshold. Shape: []
        """
        # comparing vector field should have 3 dimensions and equal shape
        assert estimated.ndim == 3
        assert estimated.shape == target.shape
        assert estimated.shape == mask.shape
        if log_var_estimated is not None:
            assert estimated.shape == log_var_estimated.shape

        # filter Nans and infinite values
        if self.config.loss_filter_nans:
            estimated, log_var_estimated, target = self.filter_nans_from_vector_fields(estimated, log_var_estimated, target, mask)

        # clip vector field values for stabilities
        if vector_field_max_norm is not None:
            target, target_clipped_norm_perc = self.clip_norms_of_vector_field(target, vector_field_max_norm)

        # scale drift loss by target diffusion
        if target_diffusion is not None:
            eps = 1e-6
            scale_per_dimension = 1 / (target_diffusion + eps)

        else:
            scale_per_dimension = torch.ones_like(estimated)

        # compute loss per location
        if self.config.loss_type == "rmse":
            loss_at_locations = rmse_at_locations(estimated, target, mask, scale_per_dimension=scale_per_dimension)  # [B, G]

        elif self.config.loss_type == "nrmse":
            loss_at_locations = nrmse_at_locations(estimated, target, mask, scale_per_dimension=scale_per_dimension)

        elif self.config.loss_type == "mse":
            loss_at_locations = mse_at_locations(estimated, target, mask, scale_per_dimension=scale_per_dimension)  # [B, G]

        elif self.config.loss_type == "nll":
            assert log_var_estimated is not None, "Must pass ´log_var_estimated` to compute nll loss."
            loss_at_locations = gaussian_nll_at_locations(estimated, log_var_estimated, target, mask)  # [B, G]

        else:
            raise ValueError("`loss_type` must be `rmse`, `nrmse`, 'mse' or `nll`, got " + self.config.loss_type)

        # weight per locations
        assert log_loss_scale_per_location.ndim == 3, f"Got {log_loss_scale_per_location.ndim}"

        loss_at_locations = loss_at_locations * torch.exp(-log_loss_scale_per_location[..., 0])

        assert loss_at_locations.ndim == 2, f"Got {loss_at_locations.ndim}"

        # filter out Nans or above threshold locations from loss
        loss_at_locations, loss_at_locations_mask, filtered_loss_locations_perc = self.filter_loss_at_locations(
            loss_at_locations, loss_threshold
        )

        loss_per_batch_element = torch.sum(loss_at_locations * loss_at_locations_mask, dim=-1)  # [B]
        assert loss_per_batch_element.ndim == 1

        loss = torch.mean(loss_per_batch_element)

        return loss, filtered_loss_locations_perc, target_clipped_norm_perc

    @torch.profiler.record_function("fimsde_train_loss")
    def loss(
        self,
        estimated_concepts: SDEConcepts,
        target_concepts: SDEConcepts | None,
        states_norm_stats: Any,
        times_norm_stats: Any,
        dimension_mask: Optional[Tensor] = None,
        loss_threshold: Optional[float] = None,
        vector_field_max_norm: Optional[float] = None,
        drift_loss_scale: Optional[float] = 1.0,
        diffusion_loss_scale: Optional[float] = 1.0,
        kl_loss_scale: Optional[float] = 0.0,
        short_time_transition_log_likelihood_loss_scale: Optional[float] = 0.0,
        drift_log_loss_scale_per_location: Optional[Tensor] = None,
        diffusion_log_loss_scale_per_location: Optional[Tensor] = None,
    ):
        """
        Compute supervised losses (RMSE or NLL) of sde concepts at non-padded dimensions.

        Args:
            estimated_concepts (SDEConcepts): Learned SDEConcepts. Shape: [B, G, D]
            target_concepts (SDEConcepts ): Ground-truth, target SDEConcepts. Shape: [B, G, D]
            states_norm_stats (Any): Statistics used by self.states_norm for normalization.
            times_norm_stats (Any): Statistics used by self.times_norm for normalization.
            dimension_mask (Optional[Tensor]): Masks padded dimensions to ignore in loss computations. Shape: [B, G, D]
            loss_threshold (Optional[float]): If passed, set loss per location to 0 if above threshold.
            vector_field_max_norm (Optional[float]): If passed, clip norm of vector fields to this value
            ..._loss_scale (Optional[float]): Scales of each loss term loss. Defaults to 1, for direct vector field losses, or 0, for KL and short term trans. ll.
            drift/diffusion_log_loss_scale_per_location (Optional[Tensor]): Log of cale of loss per location. Shape: [B, G, 1]

        Returns:
            losses (dict):
                total_loss (Tensor): Training objective: drift_loss + diffusion_scale * diffusion_loss. Shape: []
                drift_loss (Tensor): RMSE or NLL of drift estimation wrt. ground-truth. Shape: []
                diffusion_loss (Tensor): RMSE or NLL of diffusion estimation wrt. ground-truth. Shape: []
                + statistics about Nans and infinities during computations
        """
        assert target_concepts is not None, "Need ground-truth concepts at locations to compute train losses."

        if dimension_mask is None:
            dimension_mask = torch.ones_like(estimated_concepts.drift, dtype=bool)

        else:
            dimension_mask = dimension_mask.bool()

        if drift_log_loss_scale_per_location is None:
            drift_log_loss_scale_per_location = torch.zeros_like(dimension_mask[..., 0][..., None])  # [B, G, 1]

        if diffusion_log_loss_scale_per_location is None:
            diffusion_log_loss_scale_per_location = torch.zeros_like(dimension_mask[..., 0][..., None])  # [B, G, 1]

        assert dimension_mask.shape == estimated_concepts.drift.shape, (
            "Shapes of mask " + str(dimension_mask.shape) + " and concepts " + str(estimated_concepts.drift.shape) + " need to be equal."
        )

        # Ensure that estimation and target are on same normalization
        if self.config.train_with_normalized_head:
            estimated_concepts.normalize(self.states_norm, states_norm_stats, self.times_norm, times_norm_stats)
            target_concepts.normalize(self.states_norm, states_norm_stats, self.times_norm, times_norm_stats)
        else:
            estimated_concepts.renormalize(self.states_norm, states_norm_stats, self.times_norm, times_norm_stats)
            target_concepts.renormalize(self.states_norm, states_norm_stats, self.times_norm, times_norm_stats)

        # compute KL
        if self.data_delta_t is not None:
            data_delta_t = self.data_delta_t * torch.ones_like(estimated_concepts.drift[:, :, 0][:, :, None])
            data_delta_t = self.times_norm.normalization_map(data_delta_t, times_norm_stats)

        else:
            data_delta_t = torch.ones_like(estimated_concepts.drift[:, :, 0][:, :, None])

        kl_loss, _ = self.kl_loss(
            estimated_concepts.locations,
            estimated_concepts.drift,
            target_concepts.drift,
            estimated_concepts.diffusion,
            target_concepts.diffusion,
            dimension_mask,
            data_delta_t,
            drift_log_loss_scale_per_location,
        )

        short_time_trans_ll, _ = self.short_time_trans_ll_loss(
            estimated_concepts.locations,
            estimated_concepts.drift,
            target_concepts.drift,
            estimated_concepts.diffusion,
            target_concepts.diffusion,
            dimension_mask,
            data_delta_t,
            drift_log_loss_scale_per_location,
        )

        (
            drift_loss,
            drift_loss_above_threshold_or_nan_perc,
            drift_target_above_max_norm,
        ) = self.vector_field_loss(
            estimated_concepts.drift,
            estimated_concepts.log_var_drift,
            target_concepts.drift,
            dimension_mask,
            loss_threshold,
            vector_field_max_norm,
            target_diffusion=target_concepts.diffusion if self.divide_drift_loss_by_diffusion is True else None,
            log_loss_scale_per_location=drift_log_loss_scale_per_location,
        )
        (
            diffusion_loss,
            diffusion_loss_above_threshold_or_nan_perc,
            diffusion_target_above_max_norm,
        ) = self.vector_field_loss(
            estimated_concepts.diffusion,
            estimated_concepts.log_var_diffusion,
            target_concepts.diffusion,
            dimension_mask,
            loss_threshold,
            vector_field_max_norm,
            log_loss_scale_per_location=diffusion_log_loss_scale_per_location,
        )

        # balancing term for learned scaling
        assert drift_log_loss_scale_per_location.ndim == 3
        assert diffusion_log_loss_scale_per_location.ndim == 3

        learned_scale_add_loss_term_drift = drift_log_loss_scale_per_location.squeeze(-1).sum(dim=-1).mean()
        learned_scale_add_loss_term_diffusion = diffusion_log_loss_scale_per_location.squeeze(-1).sum(dim=-1).mean()

        if self.single_learnable_loss_scale_head is True:  # drift and diffusion add term are the same, so divide by 2
            learned_scale_add_loss_term_drift = learned_scale_add_loss_term_drift / 2
            learned_scale_add_loss_term_diffusion = learned_scale_add_loss_term_diffusion / 2

        # assemble losses
        total_loss = (
            drift_loss_scale * drift_loss
            + diffusion_loss_scale * diffusion_loss
            + kl_loss_scale * kl_loss
            + short_time_transition_log_likelihood_loss_scale * short_time_trans_ll
            + learned_scale_add_loss_term_drift
            + learned_scale_add_loss_term_diffusion
        )
        losses = {
            "loss": total_loss,
            "L1_drift_loss": drift_loss,
            "L1_diffusion_loss": diffusion_loss,
            "L2_KL_loss": kl_loss,
            "L3_short_time_trans_log_likelihood_loss": short_time_trans_ll,
            "drift_loss_above_threshold_or_nan_perc": drift_loss_above_threshold_or_nan_perc,
            "diffusion_loss_above_threshold_or_nan_perc": diffusion_loss_above_threshold_or_nan_perc,
            "loss_threshold": loss_threshold,
            "drift_target_norm_exceeds_threshold": drift_target_above_max_norm,
            "diffusion_target_norm_exceeds_threshold": diffusion_target_above_max_norm,
            "vector_field_max_norm": vector_field_max_norm,
            "drift_log_loss_scale_per_location": drift_log_loss_scale_per_location.mean(),
            "diffusion_log_loss_scale_per_location": diffusion_log_loss_scale_per_location.mean(),
        }

        return losses

    @torch.profiler.record_function("fimsde_finetune_loss")
    def finetune_loss(
        self,
        obs_times: Tensor,
        obs_values: Tensor,
        obs_mask: Tensor,
        paths_encoding: Tensor,
        one_step_ahead_loss_scale: float,
        whole_path_loss_scale: float,
        dimension_mask: Optional[Tensor] = None,
    ):
        """
        Compute finetuning losses (one-step ahead prediction and whole path prediction) at non-padded dimensions.

        Args:
            obs_times (Tensor): Time grid to sample paths on. Shape: [B, P, T, 1]
            obs_values (Tensor): Values to extract initial state(s) and compare sample paths to. Shape: [B, P, T, D]
            obs_mask (Tensor): True indicates values are observed. Shape: [B, P, T, 1]
            paths_encoding (Tensor): Encoding for sampling paths. Shape: [B, P, T-1, H]
            ...loss_scale (float): Factors to scale different loss terms.

        Returns:
            losses (dict):
                total_loss (Tensor): Training objective. one_step_ahead_scale * one_step_ahead_loss + whole_path_scale * whole_path_loss
                one_step_ahead_loss (Tensor): Simulate one step ahead and compute error.
                whole_path_loss (Tensor): Simulate whole path from same initial state and compute error.
        """
        obs_times = obs_times.to(torch.float32)  # Todo: should be handled better
        obs_values = obs_values.to(torch.float32)  # Todo: should be handled better
        dimension_mask = dimension_mask.to(torch.float32)  # Todo: should be handled better

        # one step ahead
        B, P, T, D = obs_values.shape
        assert paths_encoding.shape[:-1] == (B, P, T - 1)
        assert obs_times.shape == (B, P, T, 1)

        initial_states = obs_values[:, :, :-1, :].reshape(B, -1, D)
        target_states = obs_values[:, :, 1:, :].reshape(B, -1, D)
        delta_tau = (obs_times[:, :, 1:, :] - obs_times[:, :, :-1, :]).reshape(B, -1, 1)

        # multiple samples per state
        initial_states = torch.repeat_interleave(initial_states, self.finetune_samples_count, dim=1)
        target_states = torch.repeat_interleave(target_states, self.finetune_samples_count, dim=1)
        delta_tau = torch.repeat_interleave(delta_tau, self.finetune_samples_count, dim=1)

        if self.finetune_on_likelihood is False:
            predicted_states = self._euler_step(
                initial_states, delta_tau, self.finetune_em_steps, paths_encoding, obs_mask[:, :, :-1, :], dimension_mask
            )

            assert target_states.shape == predicted_states.shape

            # add mask (currently does not really work for irregular grids)
            one_step_ahead_mask = (obs_mask[:, :, 1:, :] * obs_mask[:, :, :-1, :] * dimension_mask[:, :, None, :]).reshape(B, -1, D)
            one_step_ahead_mask = torch.repeat_interleave(one_step_ahead_mask, self.finetune_samples_count, dim=1)

            # use MSE loss? can reuse location loss, as same shapes
            assert one_step_ahead_mask.shape == target_states.shape
            assert target_states.ndim == 3
            one_step_ahead_loss = mse_at_locations(predicted_states, target_states, one_step_ahead_mask)  # [B, -1]

        else:
            sde_concepts_at_initial_states, _ = self.get_estimated_sde_concepts(
                initial_states, paths_encoding=paths_encoding, obs_mask=obs_mask[:, :, :-1, :], dimension_mask=dimension_mask
            )

            drift, diffusion = sde_concepts_at_initial_states.drift, sde_concepts_at_initial_states.diffusion

            one_step_ahead_loss = ((target_states - initial_states - drift * delta_tau) ** 2) / (2 * diffusion**2 * delta_tau) + (
                2 * torch.log(diffusion)
            )

        # whole path
        if self.finetune_only_on_one_step_ahead is False:
            target_states = obs_values[:, :, 1:, :]  # [B, P, T-1, D]
            initial_states = obs_values[:, :, 0, :]  # [B, P, D]
            delta_tau = obs_times[:, :, 1:, :] - obs_times[:, :, :-1, :]  # [B, P, T-1, 1]

            # multiple samples per state
            initial_states = torch.repeat_interleave(initial_states, self.finetune_samples_count, dim=1)
            target_states = torch.repeat_interleave(target_states, self.finetune_samples_count, dim=1)
            delta_tau = torch.repeat_interleave(delta_tau, self.finetune_samples_count, dim=1)

            whole_path_mask = obs_mask[:, :, 1:, :] * obs_mask[:, :, :-1, :] * dimension_mask[:, :, None, :]
            whole_path_mask = torch.repeat_interleave(whole_path_mask, self.finetune_samples_count, dim=1)

            predicted_states = [initial_states]

            for t in range(T - 2):
                delta_tau_ = delta_tau[:, :, t]
                predicted_states.append(
                    self._euler_step(
                        predicted_states[-1], delta_tau_, self.finetune_em_steps, paths_encoding, obs_mask[:, :, :-1, :], dimension_mask
                    )
                )

            predicted_states = torch.stack(predicted_states, dim=-2)

            # use MSE loss? can reuse location loss, after reshaping
            assert target_states.shape == predicted_states.shape
            assert target_states.shape == whole_path_mask.shape
            assert target_states.ndim == 4

            target_states = target_states.reshape(B, -1, D)
            predicted_states = predicted_states.reshape(B, -1, D)
            whole_path_mask = whole_path_mask.reshape(B, -1, D)

            whole_path_loss = mse_at_locations(predicted_states, target_states, whole_path_mask)  # [B, -1]

        else:
            whole_path_loss = torch.zeros_like(one_step_ahead_loss)

        # gather losses
        one_step_ahead_loss = one_step_ahead_loss.mean()
        whole_path_loss = whole_path_loss.mean()
        total_loss = one_step_ahead_loss_scale * one_step_ahead_loss + whole_path_loss_scale * whole_path_loss

        losses = {
            "loss": total_loss,
            "one_step_ahead_loss": one_step_ahead_loss,
            "whole_path_loss": whole_path_loss,
        }

        return losses

    def _euler_step(
        self,
        current_states: Tensor,
        delta_tau: Tensor,
        solver_granularity: int,
        paths_encoding: Tensor,
        obs_mask: Tensor,
        dimension_mask: Tensor,
    ):
        assert current_states.shape[:-1] == delta_tau.shape[:-1]

        current_states = current_states.to(torch.float32)  # Todo: should be handled better

        for _ in range(solver_granularity):
            sde_concepts, _ = self.get_estimated_sde_concepts(
                current_states, paths_encoding=paths_encoding, obs_mask=obs_mask, dimension_mask=dimension_mask
            )

            drift_increment = sde_concepts.drift * (delta_tau / solver_granularity)  # [B, I, D]
            diffusion_increment = (
                sde_concepts.diffusion * torch.sqrt(delta_tau / solver_granularity) * torch.randn_like(current_states)
            )  # [B, I, D]

            if self.finetune_detach_diffusion:
                diffusion_increment = diffusion_increment.detach()

            current_states = current_states + drift_increment + diffusion_increment  # [B, I, D]

        assert current_states.shape[:-1] == delta_tau.shape[:-1]

        return current_states

    def metric(self, y: Any, y_target: Any) -> Dict:
        return super().metric(y, y_target)


ModelFactory.register(FIMSDEConfig.model_type, FIMSDE)
AutoConfig.register(FIMSDEConfig.model_type, FIMSDEConfig)
AutoModel.register(FIMSDEConfig, FIMSDE)
