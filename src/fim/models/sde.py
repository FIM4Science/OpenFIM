from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Optional

import optree
import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoConfig, AutoModel, PretrainedConfig

from fim.models.blocks import AModel, ModelFactory
from fim.models.blocks.neural_operators import AttentionOperator, ResidualEncoderLayer
from fim.utils.helper import create_class_instance


def mse_at_locations(estimated: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """
    Return MSE between target and estimated values per location. Mask indicates which values (in last dimension) to include.

    Args:
        estimated (Tensor): estimated of target values. Shape: [B, G, D]
        target (Tensor): Target values. Shape: [B, G, D]
        mask (Tensor): 0 at values to ignore in loss computations. Shape: [B, G, D]

    Return:
        mse (Tensor): MSE at locations. Shape: [B, G]
    """

    assert estimated.ndim == 3, "Got " + str(estimated.ndim)
    assert estimated.shape == target.shape, "Got " + str(estimated.shape) + " and " + str(target.shape)
    assert estimated.shape == mask.shape, "Got " + str(estimated.shape) + " and " + str(mask.shape)

    se = mask * ((estimated - target) ** 2)
    se = torch.sum(se, dim=-1)  # [B, G]

    non_masked_values_count = torch.sum(mask, dim=-1)
    mse = se / torch.clip(non_masked_values_count, min=1)  # [B, G]

    assert mse.ndim == 2, f"Got {mse.ndim}"

    return mse


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

        # Change in cumsum indicates new observation
        mask_cumsum = torch.cumsum(mask, dim=-2)

        # Extract index of change (* mask is important, s.t. index stays the same if masked out values are hit)
        indices_to_take = torch.cummax(mask_cumsum * mask, dim=-2)[1]  # [1] returns indices, [..., T, D]

        # Edge case: if first values are masked, we backward fill with first really observed value
        first_non_masked_index = torch.argmin(torch.where(mask_cumsum == 0, torch.inf, mask_cumsum), dim=-2, keepdim=True)
        indices_to_take = torch.where(mask_cumsum == 0, first_non_masked_index, indices_to_take)

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

        # Backward fill is just forward fill of flipped tensor
        mask = torch.flip(mask, dims=(-2,))
        x = torch.flip(x, dims=(-2,))

        x = forward_fill_masked_values(x, mask)

        return torch.flip(x, dims=(-2,))


@torch.profiler.record_function("fill_masekd_values")
def fill_masked_values(data: dict):
    """
    Backwardfill masked values in FIMSDE input data.
    Removes masked values entirely, never to leak any information.
    Fill makes it easier to compute features.

    Args:
        data (dict): From FIMSDE.forward arg.

    Returns:
        obs_times, obs_values, obs_mask (Tensor): Backward filled inputs.
    """

    if "obs_mask" in data.keys() and data["obs_mask"] is not None:
        obs_mask = data["obs_mask"].bool()

        # For sanity, removed masked out values
        obs_times = obs_mask * data["obs_times"]
        obs_values = obs_mask * data["obs_values"]

        # Backward fill masked values s.t. values differences and squares respect masks
        obs_times = backward_fill_masked_values(obs_times, obs_mask)
        obs_values = backward_fill_masked_values(obs_values, obs_mask)

    else:
        obs_mask = torch.ones_like(data["obs_times"]).bool()
        obs_times = data["obs_times"]
        obs_values = data["obs_values"]

    return obs_times, obs_values, obs_mask


class InstanceNormalization(ABC):
    """
    InstanceNormalization base class defines interface.
    Normalization and renormalization maps and their derivatives are purely based on pre-computed stats (the state).
    Utility function for handling instantiating varying shapes are provided.
    """

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
            get_norm_stats_kwargs (Optional[dict]): Passed to cls.get_norm_stats.
            kwargs (Optional[dict]): Passed to cls.__init__

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
            original_shape: Original shape of values for further use
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

        values, _ = self.squash_intermediate_dims(values)

        if mask is None:
            values_mean = torch.mean(values, dim=-2)
            values_std = torch.std(values, dim=-2)

        else:
            mask = mask.bool()

            mask, _ = self.squash_intermediate_dims(mask)
            mask = torch.broadcast_to(mask, values.shape)

            values_mean = torch.nanmean(torch.where(mask, values, torch.nan), dim=-2, keepdim=True)  # [B, 1, D]

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

        B, D = norm_stats[0].shape
        assert values.ndim >= 2, "Got values.ndim == " + str(values.ndim) + ", expected >=2."
        assert values.shape[0] == B, "Got batch size " + str(values.shape[0]) + ", expected " + str(B)
        assert values.shape[-1] == D, "Got dimension " + str(values.shape[-1]) + ", expected " + str(D)

        values, original_shape = self.squash_intermediate_dims(values)

        # Prepare shapes of norm stats
        expanded_norm_stats: tuple[Tensor] = self.expand_norm_stats(values.shape, norm_stats)
        ref_mean, ref_std = expanded_norm_stats

        # Apply transformation from unnormalized to normalized
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

        B, D = norm_stats[0].shape
        assert values.ndim >= 2, "Got values.ndim == " + str(values.ndim) + ", expected >=2."
        assert values.shape[0] == B, "Got batch size " + str(values.shape[0]) + ", expected " + str(B)
        assert values.shape[-1] == D, "Got dimension " + str(values.shape[-1]) + ", expected " + str(D)

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
        Assume masked values have been filled with 'fill_masked_values'.

        Args:
            values (Tensor): Shape: [B, ..., D]
            mask (Tensor): Shape: [B, ...., 1]

        Returns:
            ln_ref_mean (Tensor): Mean of ln(values) along dimensions 1 to -2. Shape: [B, D]
        """

        values, _ = self.squash_intermediate_dims(values)

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

        B, D = norm_stats[0].shape
        assert values.ndim >= 2, "Got values.ndim == " + str(values.ndim) + ", expected >=2."
        assert values.shape[0] == B, "Got batch size " + str(values.shape[0]) + ", expected " + str(B)
        assert values.shape[-1] == D, "Got dimension " + str(values.shape[-1]) + ", expected " + str(D)

        values, original_shape = self.squash_intermediate_dims(values)

        # Prepare shapes of norm stats
        expanded_norm_stats: tuple[Tensor] = self.expand_norm_stats(values.shape, norm_stats)
        ref_mean_ln, target_value = expanded_norm_stats

        # Apply transformation from unnormalized to normalized
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

        B, D = norm_stats[0].shape
        assert values.ndim >= 2, "Got values.ndim == " + str(values.ndim) + ", expected >=2."
        assert values.shape[0] == B, "Got batch size " + str(values.shape[0]) + ", expected " + str(B)
        assert values.shape[-1] == D, "Got dimension " + str(values.shape[-1]) + ", expected " + str(D)

        values, original_shape = self.squash_intermediate_dims(values)

        expanded_norm_stats: tuple[Tensor] = self.expand_norm_stats(values.shape, norm_stats)
        ref_mean_ln, target_value = expanded_norm_stats

        # Apply transformation from normalized to unnormalized
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
    A flag keeps track of the normalization status of these concepts.
    """

    # all attributes are of shape [B, ..., D]
    locations: Tensor
    drift: Tensor
    diffusion: Tensor
    normalized: bool = optree.dataclasses.field(default=False, pytree_node=False)

    def __eq__(self, other: object) -> bool:
        """
        Define equality by closeness of attributes.
        """

        rtol: float = 1e-5
        atol: float = 1e-6

        is_equal: bool = True

        is_equal = is_equal and torch.allclose(self.locations, other.locations, atol=atol, rtol=rtol)
        is_equal = is_equal and torch.allclose(self.drift, other.drift, atol=atol, rtol=rtol)
        is_equal = is_equal and torch.allclose(self.diffusion, other.diffusion, atol=atol, rtol=rtol)

        is_equal = is_equal and (self.normalized == other.normalized)

        return is_equal

    @classmethod
    def from_dict(cls, data: dict | None, normalized: Optional[bool] = False):
        """
        Construct SDEConcepts from data dict.

        Args:
            data (dict | None): Data to extract locations and concepts from its keys. Return None if not passed.
                Keys: "locations", "drift_at_locations", "diffusion_at_locations"
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
                    normalized=normalized,
                )

        else:
            return None

    def _assert_shape(self) -> None:
        """
        Assert that all attributes are of same shape.
        """

        assert self.locations.shape == self.drift.shape == self.diffusion.shape

    def _states_transformation(self, states_norm: InstanceNormalization, states_norm_stats: Any, normalize: bool) -> None:
        """
        Apply the transformation to concepts induced by the transformation of the states from the InstanceNormalization.

        Args:
            states_norm (InstanceNormalization): Underlying transformations of states.
            states_norm_stats (Any): Statistics used by states_norm.
            normalize (bool): If true, applies transformation induced by normalization, else by the inverse of normalization.
        """

        self._assert_shape()

        if normalize is True:
            grad = states_norm.normalization_map(self.locations, states_norm_stats, derivative_num=1)
            grad_grad = states_norm.normalization_map(self.locations, states_norm_stats, derivative_num=2)

        else:
            grad = states_norm.inverse_normalization_map(self.locations, states_norm_stats, derivative_num=1)
            grad_grad = states_norm.inverse_normalization_map(self.locations, states_norm_stats, derivative_num=2)

        # Transform equation by Ito's formula
        self.drift = self.drift * grad + 1 / 2 * self.diffusion**2 * grad_grad
        self.diffusion = self.diffusion * grad

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

        # Need gradient of reverse map for transformation
        # As concepts are purely state dependent, can pass in dummy value to time normalization
        dummy_times = torch.zeros_like(self.locations[..., 0].unsqueeze(-1))  # [..., 1]

        if normalize is True:
            inverse_grad = times_norm.inverse_normalization_map(dummy_times, times_norm_stats, derivative_num=1)

        else:
            inverse_grad = times_norm.normalization_map(dummy_times, times_norm_stats, derivative_num=1)

        # Transform equation by Oksendal, Theorem 8.5.7
        self.drift = self.drift * inverse_grad
        self.diffusion = self.diffusion * torch.sqrt(inverse_grad)

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
        max_dimension (int): Maximum input dimensions. Default is 3.
        model_embedding_size (int): Embedding size used throughout model. Default is 64.
        phi_0t (dict): Config for phi_0t. Default is {}.
        phi_0x (dict): Config for phi_0x. Default is {}.
        psi_1 (dict): Config for psi_1. Default is {}.
        phi_1x (dict): Config for phi_1x. Default is {}.
        operator (dict): Config for operator. Default is {}.
        states_norm (dict): Config for states instance normalization. Default is MinMaxNormalization.
        times_norm (dict): Config for times instance normalization. Default is MinMaxNormalization.
        learnable_loss_scales (dict | None): Config for AttentionOperator defining learnable loss scales per location.
        num_locations_on_path (int | None): If int is passed, compute vector field loss also on this many locations on paths.
        finetune (bool): Indicates fintuning on the observations. Default is False.
        finetune_on_sampling_mse (bool): Indicates finetuning by comparing samples from model to observations with MSE. Default is True.
        finetune_on_sampling_nll (bool): Indicates finetuning sampling from model, but using Gaussian NLL of the last EM step. Default is True.
        finetune_samples_count (int): Number of samples to generate for finetuning. Default is 1.
        finetune_samples_steps (int): Number of steps to generate from each intitial state for finetuning. Default is 1.
        finetune_em_steps (int): Number of EM steps between two observations for finetuning. Default is 1.
        finetune_detach_diffusion (bool): Detach diffusion head for finetuning. Default is False
    """

    model_type = "fimsde"

    def __init__(
        self,
        name: str = "FIMSDE",
        max_dimension: int = 3,
        model_embedding_size: int = 256,
        phi_0t: dict = {"name": "torch.nn.Linear"},
        phi_0x: dict = {"name": "torch.nn.Linear"},
        psi_1: dict = {"name": "CombinedPathTransformer", "num_layers": 2, "layer": {}},
        phi_1x: dict = {"torch.nn.Linear"},
        operator: dict = {},
        states_norm: dict = {"name": "fim.models.sde.MinMaxNormalization"},
        times_norm: dict = {"name": "fim.models.sde.MinMaxNormalization", "normalized_min": 0, "normalized_max": 1},
        learnable_loss_scales: Optional[dict] = None,
        num_locations_on_path: int | None = None,
        finetune: bool = False,
        finetune_samples_count: int = 1,
        finetune_samples_steps: int = 1,
        finetune_em_steps: int = 1,
        finetune_detach_diffusion: bool = False,
        finetune_on_sampling_mse: bool = False,
        finetune_on_sampling_nll: bool = False,
        finetune_num_points: int = -1,
        **kwargs,
    ):
        self.name = name
        self.max_dimension = max_dimension
        self.model_embedding_size = model_embedding_size
        self.phi_0t = phi_0t
        self.phi_0x = phi_0x
        self.psi_1 = psi_1
        self.phi_1x = phi_1x
        self.operator = operator

        # normalization
        self.states_norm = states_norm
        self.times_norm = times_norm

        # regularization
        self.learnable_loss_scales = learnable_loss_scales
        self.num_locations_on_path = num_locations_on_path

        # finetuning
        self.finetune = finetune
        self.finetune_samples_count = finetune_samples_count
        self.finetune_samples_steps = finetune_samples_steps
        self.finetune_em_steps = finetune_em_steps
        self.finetune_detach_diffusion = finetune_detach_diffusion
        self.finetune_on_sampling_mse = finetune_on_sampling_mse
        self.finetune_on_sampling_nll = finetune_on_sampling_nll
        self.finetune_num_points = finetune_num_points

        super().__init__(**kwargs)


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

        # For backward compatibility
        self.num_locations_on_path = config.num_locations_on_path if hasattr(config, "num_locations_on_path") else None

        self.finetune = config.finetune if hasattr(config, "finetune") else False
        self.finetune_samples_count = config.finetune_samples_count if hasattr(config, "finetune_samples_count") else 1
        self.finetune_samples_steps = config.finetune_samples_steps if hasattr(config, "finetune_samples_steps") else 1
        self.finetune_em_steps = config.finetune_em_steps if hasattr(config, "finetune_em_steps") else 1
        self.finetune_detach_diffusion = config.finetune_detach_diffusion if hasattr(config, "finetune_detach_diffusion") else False
        self.finetune_on_sampling_mse = config.finetune_on_sampling_mse if hasattr(config, "finetune_on_sampling_mse") else False
        self.finetune_on_sampling_nll = config.finetune_on_sampling_nll if hasattr(config, "finetune_on_sampling_nll") else False
        self.finetune_num_points = config.finetune_num_points if hasattr(config, "finetune_num_points") else -1

        # Set hyperparameters
        if isinstance(config, dict):
            self.config = FIMSDEConfig(**config)
        else:
            self.config = config

        self._create_modules()

        if device_map is not None:
            self.to(device_map)

    def _create_modules(self):
        config = deepcopy(self.config)  # Model loading won't work without it

        # States and times normalization
        states_norm_config = config.states_norm
        self.states_norm: InstanceNormalization = create_class_instance(states_norm_config.pop("name"), states_norm_config)

        times_norm_config = config.times_norm
        self.times_norm: InstanceNormalization = create_class_instance(times_norm_config.pop("name"), times_norm_config)

        # Observation times encoder
        phi_0t_out_features = config.model_embedding_size // 4

        config.phi_0t.update({"in_features": 1, "out_features": phi_0t_out_features})
        phi_0t_encoder = create_class_instance(config.phi_0t.pop("name"), config.phi_0t)

        self.phi_0t = phi_0t_encoder

        # Observation values encoder; encode X, del_X and (del_X)**2
        phi_0x_in_features = config.max_dimension
        phi_0x_out_features = config.model_embedding_size // 4
        config.phi_0x.update({"in_features": phi_0x_in_features, "out_features": phi_0x_out_features})

        phi_0x_x_encoder_config = deepcopy(config.phi_0x)
        phi_0x_dx_encoder_config = deepcopy(config.phi_0x)
        phi_0x_dx2_encoder_config = deepcopy(config.phi_0x)

        self.phi_0x_x = create_class_instance(phi_0x_x_encoder_config.pop("name"), phi_0x_x_encoder_config)
        self.phi_0x_dx = create_class_instance(phi_0x_dx_encoder_config.pop("name"), phi_0x_dx_encoder_config)
        self.phi_0x2_dx = create_class_instance(phi_0x_dx2_encoder_config.pop("name"), phi_0x_dx2_encoder_config)

        # Combine times and values embedding to config.model_embedding_size
        phi_0_projection_in_features = phi_0t_out_features + phi_0x_out_features

        self.phi_0 = nn.Linear(phi_0_projection_in_features, config.model_embedding_size)

        # Observations transformer encoder
        num_layers: int = config.psi_1.get("num_layers")
        layer_config = config.psi_1.get("layer")

        psi_1_transformer_layer = ResidualEncoderLayer(d_model=config.model_embedding_size, batch_first=True, **layer_config)
        self.psi_1 = nn.TransformerEncoder(psi_1_transformer_layer, num_layers=num_layers)

        # Locations encoder
        phi_1x_in_features = config.max_dimension
        phi_1x_out_features = config.model_embedding_size
        config.phi_1x.update({"in_features": phi_1x_in_features, "out_features": phi_1x_out_features})

        phi_1x_encoder = create_class_instance(config.phi_1x.pop("name"), config.phi_1x)

        self.phi_1x = phi_1x_encoder

        # Neural operators for drift, diffusion and the (log) loss scale
        self.operator_drift = AttentionOperator(
            embed_dim=config.model_embedding_size, out_features=config.max_dimension, **deepcopy(config.operator)
        )
        self.operator_diffusion = AttentionOperator(
            embed_dim=config.model_embedding_size, out_features=config.max_dimension, **deepcopy(config.operator)
        )

        if config.learnable_loss_scales is not None:
            self.operator_loss_scale = AttentionOperator(
                embed_dim=config.model_embedding_size, out_features=1, **deepcopy(config.learnable_loss_scales)
            )

        else:
            self.operator_loss_scale = None

    @torch.profiler.record_function("fimsde_forward")
    def forward(
        self,
        data: dict,
        locations: Optional[Tensor] = None,
        training: Optional[bool] = True,
        return_losses: Optional[bool] = False,
        schedulers: Optional[dict] = None,
        step: Optional[int] = 0,
    ) -> dict | tuple[SDEConcepts, dict]:
        """
        Args:
            data (dict):
                Required keys:
                    obs_values (Tensor): Observation values. optionally with noise. Shape: [B, P, T, D]
                    obs_times (Tensor): Observation times of obs_values. Shape: [B, P, T, 1]
                Optional keys:
                    obs_mask (Tensor): Mask for padded observations. == 1.0 if observed. Shape: [B, P, T, 1]
                    locations (Tensor): Points to evaluate the drift and diffusion function. Shape: [B, G, D]
                Optional keys for loss calculations:
                    drift/diffusion_at_locations (Tensor): Ground-truth concepts at locations. Shape: [B, G, D]
                    dimension_mask (Tensor): 0 at padded dimensions of ground-truth data at locations. Shape: [B, G, D]
                    obs_values_clean (Tensor): Observation values, without noise, for additional locations on path. Shape: [B, P, T, D]
                    drift/diffusion_at_obs_values (Tensor): Ground-truth concepts at clean obs values. Shape: [B, P, T, D]
                where B: batch size P: number of paths T: number of time steps G: location grid size D: dimensions

            locations (Optional[Tensor]): If passed, is prioritized over data.locations. Shape: [B, G, D]
            training (Optional[bool]): if True returns only dict with losses, including training objective.
            return_losses (Optional[bool]): If True computes and returns losses, even if training is False.
            step (Optional[int]): Optimization step for schedulers.

        Returns
            estimated_concepts (SDEConcepts): Estimated concepts at locations. Shape: [B, G, D]
            if training == True or return_losses == True return (additionally):
                losses (dict): training objective has key "loss", other keys are auxiliary for monitoring
        """

        # Optionally add locations on paths to locations during training
        if training is True:
            data = self.add_locations_on_paths(data)

        # Instance normalization and appyling mask to observations
        obs_times, obs_values, obs_mask, locations, states_norm_stats, times_norm_stats = self.preprocess_inputs(data, locations)

        # Apply neural operators
        estimated_concepts, paths_encoding = self.get_estimated_sde_concepts(
            locations, obs_times, obs_values, obs_mask, data.get("dimension_mask")
        )

        # Optionally weighting loss based on location
        if locations is not None:
            log_loss_scale_per_location = self.get_log_loss_scales(locations, paths_encoding, obs_mask)

        # Dimension masking for loss
        if data.get("dimension_mask") is not None:
            dimension_mask = data["dimension_mask"].bool()

        else:
            if estimated_concepts is not None:
                dimension_mask = torch.ones_like(estimated_concepts.drift, dtype=bool)

            else:
                dimension_mask = torch.ones_like(data.get("obs_values")[:, 0, 0, :][:, None, :])
                if dimension_mask.shape[-1] != self.config.max_dimension:
                    missing_dims = self.config.max_dimension - data.get("obs_values").shape[-1]

                    mask_pad = torch.broadcast_to(
                        torch.zeros_like(dimension_mask[..., 0][..., None]), dimension_mask.shape[:-1] + (missing_dims,)
                    )
                    dimension_mask = torch.concatenate([dimension_mask, mask_pad], dim=-1)

        # Targets for supervised loss
        target_concepts: SDEConcepts | None = SDEConcepts.from_dict(data)

        if schedulers is not None:
            if "sampling_mse_loss_scale" in schedulers.keys() and training is True:
                sampling_mse_loss_scale = schedulers.get("sampling_mse_loss_scale")(step)

            else:
                sampling_mse_loss_scale = 1.0

            if "sampling_nll_loss_scale" in schedulers.keys() and training is True:
                sampling_nll_loss_scale = schedulers.get("sampling_nll_loss_scale")(step)

            else:
                sampling_nll_loss_scale = 1.0
        else:
            sampling_mse_loss_scale = 1.0
            sampling_nll_loss_scale = 1.0

        if training is True:
            if self.finetune is False:
                losses: dict = self.loss(
                    estimated_concepts,
                    target_concepts,
                    states_norm_stats,
                    times_norm_stats,
                    obs_mask,
                    paths_encoding,
                    dimension_mask,
                    log_loss_scale_per_location,
                )

            else:
                losses: dict = self.finetune_loss(
                    obs_times,
                    obs_values,
                    paths_encoding,
                    obs_mask,
                    sampling_mse_loss_scale,
                    sampling_nll_loss_scale,
                    dimension_mask,
                )

            return {"losses": losses}

        else:
            if return_losses is True:
                if self.finetune is False:
                    losses: dict = self.loss(
                        estimated_concepts,
                        target_concepts,
                        states_norm_stats,
                        times_norm_stats,
                        obs_mask,
                        paths_encoding,
                        dimension_mask,
                        log_loss_scale_per_location,
                    )
                else:
                    losses: dict = self.finetune_loss(
                        obs_times,
                        obs_values,
                        paths_encoding,
                        obs_mask,
                        sampling_mse_loss_scale,
                        sampling_nll_loss_scale,
                        dimension_mask,
                    )

                estimated_concepts.renormalize(self.states_norm, states_norm_stats, self.times_norm, times_norm_stats)
                return estimated_concepts, {"losses": losses}

            else:
                estimated_concepts.renormalize(self.states_norm, states_norm_stats, self.times_norm, times_norm_stats)
                return estimated_concepts

    @torch.profiler.record_function("fimsde_add_locations_on_paths")
    def add_locations_on_paths(self, data: dict) -> dict:
        """
        Optionally add locations from (clean) observation values.
        Add equivalent (ground-truth) target concepts.
        Subsample self.num_locations_on_path locations randomly from observation values.

        Args:
            data (dict): From self.forward.

        Returns:
            data (dict): Enriched data with locations on path.
        """

        if self.num_locations_on_path is not None:
            obs_values_clean = data.get("obs_values_clean")  # [B, P, T, D]
            drift_at_obs_values = data.get("drift_at_obs_values")  # [B, P, T, D]
            diffusion_at_obs_values = data.get("diffusion_at_obs_values")  # [B, P, T, D]

            obs_values_clean = torch.flatten(obs_values_clean, start_dim=1, end_dim=2)  # [B, P * T, D]
            drift_at_obs_values = torch.flatten(drift_at_obs_values, start_dim=1, end_dim=2)  # [B, P * T, D]
            diffusion_at_obs_values = torch.flatten(diffusion_at_obs_values, start_dim=1, end_dim=2)  # [B, P * T, D]

            assert obs_values_clean.shape == drift_at_obs_values.shape == diffusion_at_obs_values.shape
            perm = torch.randperm(obs_values_clean.shape[1])[: self.num_locations_on_path]

            obs_values_clean = obs_values_clean[:, perm, :]
            drift_at_obs_values = drift_at_obs_values[:, perm, :]
            diffusion_at_obs_values = diffusion_at_obs_values[:, perm, :]

            data["locations"] = torch.concatenate([data["locations"], obs_values_clean], dim=1)
            data["drift_at_locations"] = torch.concatenate([data["drift_at_locations"], drift_at_obs_values], dim=1)
            data["diffusion_at_locations"] = torch.concatenate([data["diffusion_at_locations"], diffusion_at_obs_values], dim=1)

            add_dimension_mask = (
                torch.repeat_interleave(data["dimension_mask"][:, 0, :][:, None, :], repeats=self.num_locations_on_path, dim=1),
            )
            data["dimension_mask"] = torch.concatenate([data["dimension_mask"], add_dimension_mask], dim=1)

        return data

    @torch.profiler.record_function("fimsde_preprocess_inputs")
    def preprocess_inputs(self, data: dict, locations: Optional[Tensor] = None) -> tuple[Tensor, Any]:
        """
        Preprocessing of forward inputs. Includes:
            1. Backward fill on masked / padded observations.
            2. Extracting instance normalization statistics from data.
            3. Instance normalization of observations and locations.

        Args: See arguments of self.forward.

        Returns:
            Preprocessed inputs: obs_times, obs_values, obs_mask, locations
            Instance normalization statistics: states_norm_stats, times_norm_stats
        """

        assert data["obs_values"].shape[-1] <= self.config.max_dimension, (
            f"Can not process observations of dim >{self.config.max_dimension}. Got {data['obs_values'].shape[-1]}."
        )

        obs_times, obs_values, obs_mask = fill_masked_values(data)

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

        # Compute times norm based on delta taus
        # Transformation is still applied to obs_times, as delta_times will be recomputed later
        delta_times = obs_times[:, :, 1:, :] - obs_times[:, :, :-1, :]  # obs_times are backward filled
        delta_mask = obs_mask[:, :, :-1, :]
        times_norm_stats: InstanceNormalization = self.times_norm.get_norm_stats(delta_times, delta_mask)

        obs_times = self.times_norm.normalization_map(obs_times, times_norm_stats)

        return obs_times, obs_values, obs_mask, locations, states_norm_stats, times_norm_stats

    @torch.profiler.record_function("fimsde_get_estimated_sde_concepts")
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
        Applies neural operators to preprocessed model inputs to estimate SDEConcepts at locations.
        SDEConcepts are returned normalized. Padded dimensions of vector fields are set to 0.

        Args:
            locations (Tensor): Locations to extract SDEConcepts at. Shape: [B, G, D]
            obs_times, obs_values, obs_mask (Tensor): Pre-processed args of self.forward. Shape: [B, P, T, 1 or D]
            dimension_mask (Tensor): 0 at padded dimensions of ground-truth data at locations. Shape: [B, G, D]
            paths_encoding (Optional[Tensor]): Encoding of observed paths. If not passed, it is recalculated. Shape: [B, P, T, model_embedding_size]

        Returns:
            estimated_concepts (SDEConcepts): Estimated concepts at locations. Shape: [B, G, D]
            paths_encoding (Tensor): Encoding of observed paths. Shape: [B, P, T, model_embedding_size]
        """

        # Recompute paths_encoding if necessary
        if paths_encoding is None:
            assert obs_times is not None
            assert obs_values is not None

            paths_encoding, obs_mask = self.get_paths_encoding(obs_times, obs_values, obs_mask)  # [B, P, T - 1, embed_dim]

            B, P, T, _ = obs_values.shape
            assert paths_encoding.shape == (B, P, T - 1, self.config.model_embedding_size), (
                f"Expect {(B, P, T - 1, self.config.model_embedding_size)}. Got {paths_encoding.shape}"
            )

        # Encode locations
        if locations is not None:
            locations_encoding = self.phi_1x(locations)  # [B, G, embed_dim]

            B, G, _ = locations.shape
            assert locations_encoding.shape == (B, G, self.config.model_embedding_size), (
                f"Expect {(B, G, self.config.model_embedding_size)}. Got {locations_encoding.shape}"
            )

            # Projection to heads
            observations_padding_mask = torch.logical_not(obs_mask)  # revert convention for neural operator

            drift_estimator = self.operator_drift(locations_encoding, paths_encoding, observations_padding_mask=observations_padding_mask)

            diffusion_estimator = self.operator_diffusion(
                locations_encoding, paths_encoding, observations_padding_mask=observations_padding_mask
            )
            diffusion_estimator = torch.nn.functional.softplus(diffusion_estimator)

            # Set values at padded dimensions to 0 for convenience
            if dimension_mask is not None:
                zeros_ = torch.zeros_like(drift_estimator)

                dimension_mask = dimension_mask.bool()
                drift_estimator = torch.where(dimension_mask, drift_estimator, zeros_)
                diffusion_estimator = torch.where(dimension_mask, diffusion_estimator, zeros_)

            estimated_concepts = SDEConcepts(
                locations=locations,
                drift=drift_estimator,
                diffusion=diffusion_estimator,
                normalized=True,
            )

        else:
            estimated_concepts = None

        return estimated_concepts, paths_encoding

    @torch.profiler.record_function("fimsde_get_paths_encoding")
    def get_paths_encoding(self, obs_times: Tensor, obs_values: Tensor, obs_mask: Optional[Tensor] = None) -> Tensor:
        """
        Obtain encoding of all features in all paths.

        Args:
            obs_times (Tensor): Times of obs_values. Shape: [B, P, T, 1]
            obs_values (Tensor): Observation (noisy) values. Shape: [B, P, T, D]
            obs_mask (Tensor): Mask for padded observations. == 1.0 if observed. Shape: [B, P, T, 1]
            where B: batch size P: number of paths T: number of time steps dimensions

        Returns:
            paths_encoding (Tensor): Encoding of observations processed by transformer. Shape: [B, P, T-1, psi_1_tokes_dim]
        """

        # Somehow complains during inference without casting
        obs_times = obs_times.to(torch.float32)
        obs_values = obs_values.to(torch.float32)

        B, P, T, _ = obs_values.shape

        # Add features: difference and squared difference to next observation -> drop last observation
        delta_times = obs_times[:, :, 1:, :] - obs_times[:, :, :-1, :]
        X = obs_values[:, :, :-1, :]
        dX = obs_values[:, :, 1:, :] - obs_values[:, :, :-1, :]
        dX2 = dX**2

        delta_times = self.phi_0t(delta_times)  # [B, P, T-1, model_embedding_size]
        X = self.phi_0x_x(X)
        dX = self.phi_0x_dx(dX)
        dX2 = self.phi_0x2_dx(dX2)

        features = torch.concat([delta_times, X, dX, dX2], dim=-1)

        assert features.shape == (B, P, T - 1, self.config.model_embedding_size), (
            f"Expect {(B, P, T - 1, self.config.model_embedding_size)}. Got {features.shape}."
        )

        # Transformer processes features of all sequences without positional encoding
        if obs_mask is None:
            key_padding_mask = None

        else:
            # Drop last element because it is dropped for values
            obs_mask = obs_mask[:, :, :-1, :]  # [B, P, T-1, 1]

            # Revert mask as attention uses other convention
            key_padding_mask = torch.logical_not(obs_mask.bool())  # [B, P, T-1, 1]

        features = features.view(B, P * (T - 1), self.config.model_embedding_size)
        key_padding_mask = key_padding_mask.view(B, P * (T - 1), 1)

        paths_encoding = self.psi_1(features, src_key_padding_mask=key_padding_mask)  # [B, P * (T-1), H]

        return paths_encoding.view(B, P, T - 1, self.config.model_embedding_size), obs_mask

    @torch.profiler.record_function("fimsde_get_log_loss_scales")
    def get_log_loss_scales(self, locations: Tensor, paths_encoding: Tensor, obs_mask: Tensor) -> Tensor:
        """
        Optionally learn scales of loss at each location with another neural operator.

        Args:
            locations(Tensor): Locations of loss evaluations. Shape: [B, G, D]
            paths_encoding (Tensor): Encoding of observed paths. Shape: [B, P, T, model_embedding_size]
            obs_mask (Tensor): Pre-processed arg of self.forward. Shape: [B, P, T, 1]

        Returns:
            log_loss_scale_per_location(Tensor): Scale of loss at location. Shape: [B, G, 1]
        """

        if self.operator_loss_scale is not None:
            locations_encoding = self.phi_1x(locations)  # [B, G, embed_dim]

            locations_encoding = locations_encoding.detach()
            paths_encoding = paths_encoding.detach()
            observations_padding_mask = torch.logical_not(obs_mask[..., :-1, :].contiguous()).detach()

            log_loss_scale_per_location = self.operator_loss_scale(
                locations_encoding, paths_encoding, observations_padding_mask=observations_padding_mask
            )  # [B, G, 1]

        else:
            log_loss_scale_per_location = torch.zeros_like(locations[..., 0][..., None])

        return log_loss_scale_per_location

    @torch.profiler.record_function("fimsde_train_loss")
    def loss(
        self,
        estimated_concepts: SDEConcepts,
        target_concepts: SDEConcepts | None,
        states_norm_stats: Any,
        times_norm_stats: Any,
        obs_mask: Tensor,
        paths_encoding: Tensor,
        dimension_mask: Optional[Tensor] = None,
        log_loss_scale_per_location: Optional[Tensor] = None,
    ):
        """
        Compute supervised loss (MSE) of sde concepts at non-padded dimensions.

        Args:
            estimated_concepts (SDEConcepts): Learned SDEConcepts. Shape: [B, G, D]
            target_concepts (SDEConcepts ): Ground-truth, target SDEConcepts. Shape: [B, G, D]
            states_norm_stats (Any): Statistics used by self.states_norm for normalization.
            times_norm_stats (Any): Statistics used by self.times_norm for normalization.
            obs_mask (Tensor): True indicates values are observed, for one-step-ahead likelihood. Shape: [B, P, T, 1]
            paths_encoding (Tensor): Encoding of observed paths, or one-step-ahead loss.
            dimension_mask (Optional[Tensor]): Masks padded dimensions to ignore in loss computations. Shape: [B, G, D]
            log_loss_scale_per_location (Optional[Tensor]): Log of scale of loss per location. Shape: [B, G, 1]

        Returns:
            losses (dict):
                total_loss (Tensor): Training objective: drift_loss + diffusion_scale * diffusion_loss. Shape: []
                drift_loss (Tensor): MSE of drift estimation wrt. ground-truth. Shape: []
                diffusion_loss (Tensor): MSE of diffusion estimation wrt. ground-truth. Shape: []
                + statistics about Nans and infinities during computations
        """
        assert target_concepts is not None, "Need ground-truth concepts at locations to compute train losses."

        # Ensure dimensions are masked properly
        if dimension_mask is None:
            dimension_mask = torch.ones_like(estimated_concepts.drift, dtype=bool)

        else:
            dimension_mask = dimension_mask.bool()

        if log_loss_scale_per_location is None:
            log_loss_scale_per_location = torch.zeros_like(dimension_mask[..., 0][..., None])  # [B, G, 1]

        assert dimension_mask.shape == estimated_concepts.drift.shape, (
            "Shapes of mask " + str(dimension_mask.shape) + " and concepts " + str(estimated_concepts.drift.shape) + " need to be equal."
        )

        # Ensure that estimation and target are on same normalization
        estimated_concepts.normalize(self.states_norm, states_norm_stats, self.times_norm, times_norm_stats)
        target_concepts.normalize(self.states_norm, states_norm_stats, self.times_norm, times_norm_stats)

        # If additional locations on path were added, separate those and compute their loss per vector field
        if self.num_locations_on_path is not None:
            estimated_drift_on_path = estimated_concepts.drift[:, -self.num_locations_on_path :]
            estimated_diffusion_on_path = estimated_concepts.diffusion[:, -self.num_locations_on_path :]
            target_drift_on_path = target_concepts.drift[:, -self.num_locations_on_path :]
            target_diffusion_on_path = target_concepts.diffusion[:, -self.num_locations_on_path :]
            dimension_mask_on_path = dimension_mask[:, -self.num_locations_on_path :]
            log_loss_scale_per_path_location = log_loss_scale_per_location[:, -self.num_locations_on_path :]

            estimated_concepts.drift = estimated_concepts.drift[:, : -self.num_locations_on_path]
            estimated_concepts.diffusion = estimated_concepts.diffusion[:, : -self.num_locations_on_path]
            target_concepts.drift = target_concepts.drift[:, : -self.num_locations_on_path]
            target_concepts.diffusion = target_concepts.diffusion[:, : -self.num_locations_on_path]
            dimension_mask = dimension_mask[:, : -self.num_locations_on_path]
            log_loss_scale_per_location = log_loss_scale_per_location[:, : -self.num_locations_on_path]

            drift_loss_at_path, _ = self.vector_field_loss(
                estimated_drift_on_path,
                target_drift_on_path,
                dimension_mask_on_path,
                log_loss_scale_per_location=log_loss_scale_per_path_location,
            )

            diffusion_loss_at_path, _ = self.vector_field_loss(
                estimated_diffusion_on_path,
                target_diffusion_on_path,
                dimension_mask_on_path,
                log_loss_scale_per_location=log_loss_scale_per_path_location,
            )

        else:
            drift_loss_at_path = None
            diffusion_loss_at_path = None

        # Loss at locations per vector field
        (
            drift_loss_at_locations,
            drift_loss_nan_perc,
        ) = self.vector_field_loss(
            estimated_concepts.drift,
            target_concepts.drift,
            dimension_mask,
            log_loss_scale_per_location=log_loss_scale_per_location,
        )
        (
            diffusion_loss_at_locations,
            diffusion_loss_nan_perc,
        ) = self.vector_field_loss(
            estimated_concepts.diffusion,
            target_concepts.diffusion,
            dimension_mask,
            log_loss_scale_per_location=log_loss_scale_per_location,
        )

        # Balancing term for learned scaling
        assert log_loss_scale_per_location.ndim == 3

        learned_scale_add_loss_term_drift = log_loss_scale_per_location.squeeze(-1).sum(dim=-1).mean()

        if drift_loss_at_path is not None:
            drift_loss = drift_loss_at_path + drift_loss_at_locations
            diffusion_loss = diffusion_loss_at_path + diffusion_loss_at_locations

        else:
            drift_loss = drift_loss_at_locations
            diffusion_loss = diffusion_loss_at_locations

            drift_loss_at_path = 0
            diffusion_loss_at_path = 0

        # Assemble train objective
        total_loss = drift_loss + diffusion_loss + learned_scale_add_loss_term_drift

        losses = {
            "loss": total_loss,
            "drift_loss": drift_loss,
            "drift_loss_at_locations": drift_loss_at_locations,
            "drift_loss_at_path": drift_loss_at_path,
            "diffusion_loss": diffusion_loss,
            "diffusion_loss_at_locations": diffusion_loss_at_locations,
            "diffusion_loss_at_path": diffusion_loss_at_path,
            "drift_loss_nan_perc": drift_loss_nan_perc,
            "diffusion_loss_nan_perc": diffusion_loss_nan_perc,
            "log_loss_scale_per_location": log_loss_scale_per_location.mean(),
        }

        return losses

    @torch.profiler.record_function("fimsde_vector_field_loss")
    def vector_field_loss(
        self,
        estimated: Tensor,
        target: Tensor,
        mask: Tensor,
        log_loss_scale_per_location: Optional[Tensor] = None,
    ) -> tuple[Tensor]:
        """
        Compute (regularized) loss of vector field values at locations. Return statistics about regularization for monitoring.

        Regularizations:
            Remove Nans and infinite values in passed vector fields.
            Per location, remove Nans and infinite values from calculated loss.
            Per location, scale loss by a (learned) value.

        Args:
            vector field values (Tensor): Vector fields to compute loss with.  Shape: [B, G, D]
            mask (Tensor): 0 masks padded values to ignore in loss calculation. Shape: [B, G, D]
            log_loss_scale_per_location (Optional[Tensor]): Multiply the loss at each location by a learned scale. Shape: [B, G, 1]

        Returns
            loss (Tensor): Loss of vector field. Shape: []
            filtered_loss_locations_perc (Tensor): Percentage of locations where loss is Nan. Shape: []
        """

        assert estimated.ndim == 3
        assert estimated.shape == target.shape
        assert estimated.shape == mask.shape

        # Filter Nans and infinite values
        estimated = torch.nan_to_num(estimated)
        target = torch.nan_to_num(target)

        # Compute MSE per location
        loss_at_locations = mse_at_locations(estimated, target, mask)  # [B, G]

        # Weight per locations
        assert log_loss_scale_per_location.ndim == 3, f"Got {log_loss_scale_per_location.ndim}"

        loss_at_locations = loss_at_locations * torch.exp(-log_loss_scale_per_location[..., 0])

        assert loss_at_locations.ndim == 2, f"Got {loss_at_locations.ndim}"

        # Filter locations with Nans as loss
        loss_is_finite_mask = torch.isfinite(loss_at_locations)  # [B, G]
        loss_at_locations = torch.nan_to_num(loss_at_locations)

        assert loss_at_locations.ndim == 2
        assert loss_is_finite_mask.ndim == 2

        filtered_loss_locations_perc = torch.logical_not(loss_is_finite_mask).mean(dtype=torch.float32)  # []

        # Compute loss filtered locations
        loss_per_batch_element = torch.sum(loss_at_locations * loss_is_finite_mask, dim=-1)  # [B]
        assert loss_per_batch_element.ndim == 1

        loss = torch.mean(loss_per_batch_element)

        return loss, filtered_loss_locations_perc

    @torch.profiler.record_function("fimsde_finetune_loss")
    def finetune_loss(
        self,
        obs_times: Tensor,
        obs_values: Tensor,
        paths_encoding: Tensor,
        obs_mask: Tensor,
        sampling_mse_loss_scale: float,
        sampling_nll_loss_scale: float,
        dimension_mask: Optional[Tensor] = None,
    ):
        """
        Compute finetuning losses (NLL or MSE) at non-padded dimensions.

        Args:
            obs_times (Tensor): Time grid to sample paths on. Shape: [B, P, T, 1]
            obs_values (Tensor): Values to extract initial state(s) and compare sample paths to. Shape: [B, P, T, D]
            paths_encoding (Tensor): Encoding for sampling paths. Shape: [B, P, T-1, H]
            obs_mask (Tensor): True indicates values are observed. Used in conjunction with paths_encoding for sampling. Shape: [B, P, T, 1]
            ...loss_scale (float): Factors to scale different loss terms.
            dimension_mask (Tensor): 0 at padded dimensions of ground-truth data at locations. Shape: [B, G, D]

        Returns:
            losses (dict):
                total_loss (Tensor): Training objective. sampling_mse_loss_scale * sampling_mse + sampling_nll_loss_scale + sampling_nll
                sampling_mse_loss (Tensor): MSE of few steps ahead simulated paths.
                sampling_nll_loss (Tensor): NLL of few step ahead simulated paths.
        """

        # Somehow complains during inference without casting
        obs_times = obs_times.to(torch.float32)
        obs_values = obs_values.to(torch.float32)
        dimension_mask = dimension_mask.to(torch.float32)

        B, P, T, D = obs_values.shape
        assert paths_encoding.shape[:-1] == (B, P, T - 1)
        assert obs_times.shape == (B, P, T, 1)

        initial_states = obs_values[:, :, 0, :]  # [B, P, D]

        # Number of future observations to simulate and compute objective
        obs_patches = obs_values.unfold(dimension=-2, size=self.finetune_samples_steps + 1, step=1)  # [B, P, T-(steps+1), D, steps+1]
        obs_patches = torch.transpose(obs_patches, -1, -2)  # [B, P, T-(steps+1), steps+1, D]
        initial_states = obs_patches[:, :, :, 0, :]  # [B, P, T - (steps + 1), D]
        target_states = obs_patches[:, :, :, 1:, :]  # [B, P, T - (steps + 1), steps, D]

        obs_mask_valid_steps = obs_mask.unfold(
            dimension=-2, size=self.finetune_samples_steps + 1, step=1
        )  # [B, P, T-(steps+1), 1, steps+1]
        obs_mask_valid_steps = torch.transpose(obs_mask_valid_steps, -1, -2)  # [B, P, T - (steps+1), steps+1, 1]
        obs_mask_valid_steps = obs_mask_valid_steps[..., :-1, :] * obs_mask_valid_steps[..., 1:, :]  # [B, P, T - (steps + 1), steps, D]

        times_patches = obs_times.unfold(dimension=-2, size=self.finetune_samples_steps + 1, step=1)  # [B, P, T-(steps+1), 1, steps+1]
        times_patches = torch.transpose(times_patches, -1, -2)  # [B, P, T-(steps+1), steps+1, 1]
        delta_tau = times_patches[..., 1:, :] - times_patches[..., :-1, :]  # [B, P, T - (steps + 1), steps, 1]

        initial_states = torch.flatten(initial_states, start_dim=1, end_dim=2)  # [B, -1, D]
        target_states = torch.flatten(target_states, start_dim=1, end_dim=2)  # [B, -1, steps, D]
        obs_mask_valid_steps = torch.flatten(obs_mask_valid_steps, start_dim=1, end_dim=2)  # [B, -1, steps, D]
        delta_tau = torch.flatten(delta_tau, start_dim=1, end_dim=2)  # [B, -1, steps, D]

        # Optionally subsample points to generate trajectories from
        if self.finetune_num_points != -1:
            perm = torch.randperm(initial_states.shape[1])[: self.finetune_num_points]
            initial_states = initial_states[:, perm]
            target_states = target_states[:, perm]
            obs_mask_valid_steps = obs_mask_valid_steps[:, perm]
            delta_tau = delta_tau[:, perm]

        # Number of trajectories per initial state to simulate
        initial_states = torch.repeat_interleave(initial_states, self.finetune_samples_count, dim=1)
        target_states = torch.repeat_interleave(target_states, self.finetune_samples_count, dim=1)
        obs_mask_valid_steps = torch.repeat_interleave(obs_mask_valid_steps, self.finetune_samples_count, dim=1)
        delta_tau = torch.repeat_interleave(delta_tau, self.finetune_samples_count, dim=1)

        dimension_mask = torch.broadcast_to(dimension_mask[:, 0, :][:, None, None, :], target_states.shape)

        # Smulation
        predicted_states = []
        last_step_before_prediction = []
        current_states = initial_states

        for t in range(self.finetune_samples_steps):
            delta_tau_ = delta_tau[:, :, t]
            current_states, last_states = self._euler_step(
                current_states,
                delta_tau_,
                self.finetune_em_steps,
                paths_encoding,
                obs_mask[:, :, :-1, :],
                dimension_mask[:, :, 0, :],
            )
            if self.finetune_on_sampling_mse is True:
                predicted_states.append(current_states)

            if self.finetune_on_sampling_nll:
                last_step_before_prediction.append(last_states)

        if self.finetune_on_sampling_mse is True:
            predicted_states = torch.stack(predicted_states, dim=-2)
            assert target_states.shape == predicted_states.shape
            predicted_states = predicted_states.reshape(B, -1, D)

        if self.finetune_on_sampling_nll is True:
            last_step_before_prediction = torch.stack(last_step_before_prediction, dim=-2)
            assert target_states.shape == last_step_before_prediction.shape
            last_step_before_prediction = last_step_before_prediction.reshape(B, -1, D)

        assert target_states.shape[:-1] == dimension_mask.shape[:-1] == obs_mask_valid_steps.shape[:-1]
        assert target_states.ndim == 4
        target_states = target_states.reshape(B, -1, D)

        obs_mask_valid_steps = obs_mask_valid_steps.reshape(B, -1, 1)
        dimension_mask = dimension_mask.reshape(B, -1, D)

        location_mask = dimension_mask * obs_mask_valid_steps

        # Compute either or both objectives
        if self.finetune_on_sampling_mse is True:
            sampling_mse = mse_at_locations(predicted_states, target_states, location_mask)  # [B, -1]
            sampling_mse = sampling_mse.mean()

        else:
            sampling_mse = 0

        if self.finetune_on_sampling_nll is True:
            sde_concepts, _ = self.get_estimated_sde_concepts(
                last_step_before_prediction,
                paths_encoding=paths_encoding,
                obs_mask=obs_mask[:, :, :-1, :],
                dimension_mask=dimension_mask,
            )
            drift, diffusion = sde_concepts.drift, sde_concepts.diffusion

            if self.finetune_detach_diffusion is True:
                diffusion = diffusion.detach()

            em_delta_t = delta_tau / self.finetune_em_steps
            em_delta_t = em_delta_t.reshape(B, -1, 1)

            assert drift.shape == diffusion.shape == target_states.shape == last_step_before_prediction.shape == location_mask.shape

            # clip diffusion=0, em_delta_t=0, they are masked below
            diffusion = torch.clip(diffusion, min=1e-10)
            em_delta_t = torch.clip(em_delta_t, min=1e-10)

            sampling_nll = ((target_states - last_step_before_prediction - drift * em_delta_t) ** 2) / (
                2 * diffusion**2 * em_delta_t
            ) + torch.log(diffusion)
            assert sampling_nll.shape == drift.shape

            sampling_nll = sampling_nll * location_mask
            sampling_nll = sampling_nll.sum() / location_mask.sum()

        else:
            sampling_nll = 0

        total_loss = sampling_mse_loss_scale * sampling_mse + sampling_nll_loss_scale * sampling_nll

        losses = {
            "loss": total_loss,
            "sampling_mse": sampling_mse,
            "sampling_nll": sampling_nll,
        }

        return losses

    @torch.profiler.record_function("fimsde_euler_step")
    def _euler_step(
        self,
        current_states: Tensor,
        delta_tau: Tensor,
        solver_granularity: int,
        paths_encoding: Tensor,
        obs_mask: Tensor,
        dimension_mask: Tensor,
    ):
        """
        Simple EM scheme for multiple steps between two observations during finetuning.

        Args:
            current_states(Tensor): State of system.
            delta_tau(Tensor): Time interval to simulate.
            solver_granularity(int): Number of steps to take between observations.
            paths_encoding, obs_mask, dimension_mask (Tensor): Passed to self.get_estimated_sde_concepts.

        Returns:
            current_states(Tensor): Result of simulation.
            last_states(Tensor): Result one granular EM step before simulation result.
        """

        assert current_states.shape[:-1] == delta_tau.shape[:-1]

        last_states = current_states
        current_states = current_states.to(torch.float32)

        for _ in range(solver_granularity):
            last_states = current_states
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
        assert last_states.shape[:-1] == delta_tau.shape[:-1]

        return current_states, last_states

    def metric(self, y: Any, y_target: Any) -> dict:
        return super().metric(y, y_target)


ModelFactory.register(FIMSDEConfig.model_type, FIMSDE)
AutoConfig.register(FIMSDEConfig.model_type, FIMSDEConfig)
AutoModel.register(FIMSDEConfig, FIMSDE)
