from dataclasses import dataclass, field
from typing import Dict, Optional, Self, Tuple

import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch import Tensor

from fim.data.config_dataclasses import FIMDatasetConfig
from fim.data.data_generation.dynamical_systems_target import generate_all
from fim.data.datasets import FIMSDEDatabatchTuple
from fim.models.blocks import ModelFactory
from fim.models.blocks.base import MLP, TransformerModel
from fim.models.blocks.positional_encodings import SineTimeEncoding
from fim.models.config_dataclasses import FIMSDEConfig
from fim.pipelines.sde_pipelines import FIMSDEPipeline
from fim.utils.plots.sde_estimation_plots import images_log_1D, images_log_2D, images_log_3D


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


class NormalizationStats:
    """
    Stores statistics needed to map values into a particular interval.
    """

    def __init__(self, values: Tensor, normalized_min: float = -1, normalized_max: float = 1):
        # values and target interval boundaries
        self.normalized_min, self.normalized_max = normalized_min, normalized_max
        self.unnormalized_min, self.unnormalized_max = self.get_unnormalized_stats(values)

        # batch and observed dimension for reference
        self.batch_size = self.unnormalized_min.shape[0]
        self.dim = self.unnormalized_min.shape[-1]

        # apply transform map over three axes: batch, time, dimension
        transform_map_grad = torch.func.grad(self.transform_map)
        transform_map_grad_grad = torch.func.grad(transform_map_grad)

        self.batch_transform_map = torch.vmap(torch.vmap(torch.vmap(self.transform_map)))
        self.batch_transform_map_grad = torch.vmap(torch.vmap(torch.vmap(transform_map_grad)))
        self.batch_transform_map_grad_grad = torch.vmap(torch.vmap(torch.vmap(transform_map_grad_grad)))

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

    def get_unnormalized_stats(self, values: Tensor) -> tuple[Tensor]:
        """
        Return min and max of passed values along all dimensions 1 to -2.

        Args:
            values (Tensor): Shape: [B, ..., D]

        Returns:
            min, max (Tensor): Statistics of inputs along all dimensions 1 to -2. Shape: [B, D]
        """
        # Squash intermediate dimensions from values
        values, _ = self.squash_intermediate_dims(values)

        values_min = torch.amin(values, dim=-2)
        values_max = torch.amax(values, dim=-2)

        return values_min, values_max

    def get_intervals_boundaries(self, shape: tuple) -> tuple[Tensor]:
        """
        Return normalization statistics (attributes) as tensors in required shape.

        Args:
            shape (tuple): Expected shape. Must be of length 3, specifically (B, *, D), where self.unnormalized_...shape == [B, D].

        Returns:
            normalization_stats (tuple[Tensor]): tensors needed to describe normalization map

        """
        assert len(shape) == 3, "Expect 3 dimensions, got " + str(len(shape)) + ". Passed shape: " + str(shape)

        unnormalized_min = self.unnormalized_min.unsqueeze(-2).expand(shape)  # [B, *, D]
        unnormalized_max = self.unnormalized_max.unsqueeze(-2).expand(shape)  # [B, *, D]

        normalized_min = self.normalized_min * torch.ones_like(unnormalized_min)  # [B, *, D]
        normalized_max = self.normalized_max * torch.ones_like(unnormalized_max)  # [B, *, D]

        assert unnormalized_min.ndim == 3
        assert unnormalized_max.ndim == 3

        return unnormalized_min, unnormalized_max, normalized_min, normalized_max

    def normalization_map(self, values: Tensor, derivative_num: Optional[int] = 0) -> Tensor:
        """
        (Derivative of) normalization based on previously set statistics, i.e. evaluate the map
        x -> (x - unnormalized_min) / (unnormalized_max - unnormalized_min) * (normalized_max - normalized_min) + normalized_min
        at all values.

        Args:
            values (Tensor): Values to normalized based on previously set statistics. Shape: [B, ..., D]
            derivative_num (int): Derivative of normalization map to return.

        Returns:
            (derivative) of image of values under normalization_map: Normalized values. Shape: [B, ..., D]
        """
        assert values.ndim >= 2, "Got values.ndim == " + str(values.ndim) + ", expected >=2."
        assert values.shape[0] == self.batch_size, "Got batch size " + str(values.shape[0]) + ", expected " + str(self.batch_size)
        assert values.shape[-1] == self.dim, "Got dimension " + str(values.shape[-1]) + ", expected " + str(self.dim)

        # Squash intermediate dimensions from values
        values, original_shape = self.squash_intermediate_dims(values)

        unnormalized_min, unnormalized_max, normalized_min, normalized_max = self.get_intervals_boundaries(values.shape)

        # apply transformation from unnormalized to normalized
        if derivative_num == 0:
            out = self.batch_transform_map(values, unnormalized_min, unnormalized_max, normalized_min, normalized_max)

        elif derivative_num == 1:
            out = self.batch_transform_map_grad(values, unnormalized_min, unnormalized_max, normalized_min, normalized_max)

        elif derivative_num == 2:
            out = self.batch_transform_map_grad_grad(values, unnormalized_min, unnormalized_max, normalized_min, normalized_max)

        else:
            raise ValueError("Can only return up to second derivative. Got " + str(derivative_num))

        # Reintroduce intermediate dimensions from values
        out = out.reshape(original_shape)

        return out

    def inverse_normalization_map(self, values: Tensor, derivative_num: Optional[int] = 0) -> Tensor:
        """
        (Derivative of) inverse normalization of the passed values based on previously set statistics, i.e. evaluate the map
        x -> (x - normalized_min) / (normalized_max - normalized_min) * (unnormalized_max - unnormalized_min) + unnormalized_min
        at all values.

        Args:
            values (Tensor): Values to apply inverse normalization based on previously set statistics to. Shape: [B, ..., D]
            derivative_num (int): Derivative of inverse normalization map to return.

        Returns:
            renormalized_values: Reormalized values. Shape: [B, ..., D]
        """
        assert values.ndim >= 2, "Got values.ndim == " + str(values.ndim) + ", expected >=2."
        assert values.shape[0] == self.batch_size, "Got batch size " + str(values.shape[0]) + ", expected " + str(self.batch_size)
        assert values.shape[-1] == self.dim, "Got dimension " + str(values.shape[-1]) + ", expected " + str(self.dim)

        # Squash intermediate dimensions from values
        values, original_shape = self.squash_intermediate_dims(values)

        unnormalized_min, unnormalized_max, normalized_min, normalized_max = self.get_intervals_boundaries(values.shape)

        # apply transformation from normalized to unnormalized
        if derivative_num == 0:
            out = self.batch_transform_map(values, normalized_min, normalized_max, unnormalized_min, unnormalized_max)

        elif derivative_num == 1:
            out = self.batch_transform_map_grad(values, normalized_min, normalized_max, unnormalized_min, unnormalized_max)

        elif derivative_num == 2:
            out = self.batch_transform_map_grad_grad(values, normalized_min, normalized_max, unnormalized_min, unnormalized_max)

        else:
            raise ValueError("Can only return up to second derivative. Got " + str(derivative_num))

        # Reintroduce intermediate dimensions from values
        out = out.reshape(original_shape)

        return out


@dataclass(eq=False)
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
    normalized: bool = False

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
    def from_dbt(cls, databatch: FIMSDEDatabatchTuple | None, normalized: Optional[bool] = False) -> Self:
        """
        Construct SDEConcepts from FIMSDEDatabatchTuple.

        Args:
            databatch (FIMSDEDatabatchTuple | None): Data to extract locations and concepts from. Return None if not passed.
            normalized (bool): Flag if data in databatch is normalized. Default: False.

        Returns:
            sde_concepts (SDEConcepts): SDEConcepts with locations, drift and diffusion extracted from FIMSDEDatabatchTuple.
        """
        if databatch is not None:
            if (
                databatch.locations is not None
                and databatch.drift_at_locations is not None
                and databatch.diffusion_at_locations is not None
            ):
                return cls(
                    locations=databatch.locations,
                    drift=databatch.drift_at_locations,
                    diffusion=databatch.diffusion_at_locations,
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

    def _state_transformation(self, states_norm_stats: NormalizationStats, normalize: bool) -> None:
        """
        Apply the transformation to concepts induced by the transformation of the states from the NormalizationStats.

        Args:
            states_norm_stats (NormalizationStats): Underlying transformations of states.
            normalize (bool): If true, applies transformation induced by normalization, else by the inverse of normalization.
        """
        self._assert_shape()

        # evaluate gradient of the normalization map at the respective locations
        if normalize is True:
            grad = states_norm_stats.normalization_map(self.locations, derivative_num=1)
            grad_grad = states_norm_stats.normalization_map(self.locations, derivative_num=2)

        else:
            grad = states_norm_stats.inverse_normalization_map(self.locations, derivative_num=1)
            grad_grad = states_norm_stats.inverse_normalization_map(self.locations, derivative_num=2)

        log_grad = torch.log(grad)

        # transform equation by Ito's formula
        self.drift = self.drift * grad + 1 / 2 * self.diffusion**2 * grad_grad
        self.diffusion = self.diffusion * grad

        if self.log_var_drift is not None:
            self.log_var_drift = self.log_var_drift + 2 * log_grad

        if self.log_var_diffusion is not None:
            self.log_var_diffusion = self.log_var_diffusion + 2 * log_grad

        self._assert_shape()

    def _time_transformation(self, time_norm_stats: NormalizationStats, normalize: bool) -> None:
        """
        Apply the transformation to concepts induced by the transformation of time from the NormalizationStats.

        Args:
            time_norm_stats (NormalizationStats): Underlying transformations of time.
            normalize (bool): If true, applies transformation induced by normalization, else by the inverse of normalization.
        """
        self._assert_shape()

        # need gradient of reverse map for transformation
        # as concepts are purely state dependent, can pass in dummy value to time normalization
        dummy_times = torch.zeros_like(self.locations[..., 0].unsqueeze(-1))  # [..., 1]

        if normalize is True:
            inverse_grad = time_norm_stats.inverse_normalization_map(dummy_times, derivative_num=1)

        else:
            inverse_grad = time_norm_stats.normalization_map(dummy_times, derivative_num=1)

        log_inverse_grad = torch.log(inverse_grad)

        # transform equation by Oksendal, Theorem 8.5.7
        self.drift = self.drift * inverse_grad
        self.diffusion = self.diffusion * torch.sqrt(inverse_grad)

        if self.log_var_drift is not None:
            self.log_var_drift = self.log_var_drift + 2 * log_inverse_grad

        if self.log_var_diffusion is not None:
            self.log_var_diffusion = self.log_var_diffusion + log_inverse_grad

        self._assert_shape()

    def _locations_transformation(self, states_norm_stats: NormalizationStats, normalize: bool) -> None:
        """
        Apply transformation of states to the locations at which equation concepts are evaluated at.

        Args:
            states_norm_stats (NormalizationStats): Specifies transformations of states.
            normalize (bool): If true, applies transformation induced by normalization, else by the inverse of normalization.
        """
        self._assert_shape()

        if normalize is True:
            self.locations = states_norm_stats.normalization_map(self.locations)

        else:
            self.locations = states_norm_stats.inverse_normalization_map(self.locations)

        self._assert_shape()

    def normalize(self, states_norm_stats: NormalizationStats, time_norm_stats: NormalizationStats) -> None:
        """
        Normalize locations and concepts if not already normalized.

        Args:
            states_norm_stats, time_norm_stats (NormalizationStats): Specifies normalizations to apply.
        """
        if self.normalized is False:
            self._state_transformation(states_norm_stats, normalize=True)
            self._locations_transformation(states_norm_stats, normalize=True)
            self._time_transformation(time_norm_stats, normalize=True)

            self.normalized = True

    def renormalize(self, states_norm_stats: NormalizationStats, time_norm_stats: NormalizationStats) -> None:
        """
        Reormalize locations and concepts if not already renormalized.

        Args:
            states_norm_stats, time_norm_stats (NormalizationStats): Specifies renormalizations to apply.
        """
        if self.normalized is True:
            self._state_transformation(states_norm_stats, normalize=False)
            self._locations_transformation(states_norm_stats, normalize=False)
            self._time_transformation(time_norm_stats, normalize=False)

            self.normalized = False


@dataclass
class FIMSDEForward:
    """
    This class carries all objects required for the forward pass evaluation
    and subsequent loss evaluation, this includes input and target data
    as well as all the estimator heads.

    THE MAIN GOAL IS TO KEEP TRACK OF HOW NORMALIZATIONS ARE
    PERFORMED AND WHEN

    WE ASSUME THAT THE INITIAL VALUE OF THE TIME IS 0

    B: batch size
    P: number of paths
    T: number of time steps
    G: grid size
    D: dimensions
    """

    # Estimators (Learned Concepts)
    drift_estimator: Optional[Tensor] = None  # [B,P,G,D]
    log_var_drift_estimator: Optional[Tensor] = None  # [B,P,G,D]
    diffusion_estimator: Optional[Tensor] = None  # [B,P,G,D]
    log_var_diffusion_estimator: Optional[Tensor] = None  # [B,P,G,D]

    # Targets (Data Concepts)
    drift_target: Optional[Tensor] = None  # [B,P,G,D]
    diffusion_target: Optional[Tensor] = None  # [B,P,G,D]

    # Data
    locations: Optional[Tensor] = None  # [B,P,G,D]
    obs_times: Optional[Tensor] = None  # [B,P,T,1]
    obs_values: Optional[Tensor] = None  # [B,P,T,D]

    # Normalization stats
    max_obs_times: Optional[Tensor] = None  # [B,P,T,1]
    max_obs_values: Optional[Tensor] = None  # [B,1,D]
    min_obs_values: Optional[Tensor] = None  # [B,1,D]

    range_obs_vals: Optional[Tensor] = None  # [B,1,D]
    range_obs_times: Optional[Tensor] = None  # [B,1,D]

    # masks
    obs_mask: Optional[Tensor] = None  # [B,P,T,D]
    dimension_mask: Optional[Tensor] = None  # [B,P,T,D]

    # Basic stats and flags
    is_data_set: bool = False
    is_target_set: bool = False
    is_estimator_set: bool = False

    is_input_normalized: bool = False
    is_target_normalized: bool = False
    is_estimator_normalized: bool = False

    # loss
    losses: Dict[str, Tensor] = field(default_factory=lambda: {})

    min_border_factor: float = 2.0

    def set_input_data(self, obs_times: Tensor, obs_values: Tensor, obs_mask: Tensor, locations: Tensor, dimension_mask: Tensor):
        """Sets observation data and related variables."""
        self.obs_times = obs_times
        self.obs_values = obs_values
        self.obs_mask = obs_mask
        self.locations = locations
        self.dimension_mask = dimension_mask
        self.is_data_set = True

    def set_target_data(self, drift_data: Tensor, diffusion_data: Tensor):
        """Sets target data for drift and diffusion."""
        self.drift_target = drift_data
        self.diffusion_target = diffusion_data
        self.is_target_set = True

    def set_forward_estimators(
        self, drift_estimator: Tensor, diffusion_estimator: Tensor, var_drift_estimator: Tensor, var_diffusion_estimator: Tensor
    ):
        """
        Sets estimators for forward pass.

        IF INPUT DATA IS NORMALIZED WHEN ESTIMATOR IS SET
        ESTIMATOR IS ASSUMED NORMALIZED
        """
        self.drift_estimator = drift_estimator
        self.diffusion_estimator = diffusion_estimator
        self.log_var_drift_estimator = var_drift_estimator
        self.log_var_diffusion_estimator = var_diffusion_estimator

        if self.is_input_normalized:
            self.is_estimator_normalized = True
        else:
            self.is_estimator_normalized = False

    def set_losses(self, losses):
        """Set the losses"""
        self.losses = losses

    def normalize_input(self):
        """
        Normalizes observation data, locations, and time using shared min/max values.

        TAKEN FROM THE APPPENDIX B.1
        """
        # Flatten obs_values across P and T for computing min/max
        B, P, T, D = self.obs_values.shape
        obs_values_reshaped = self.obs_values.view(B, P * T, D)

        # Calculate min, max, and range for normalization (excluding time)
        self.min_obs_values = obs_values_reshaped.min(dim=1, keepdim=True).values.unsqueeze(1)
        self.max_obs_values = obs_values_reshaped.max(dim=1, keepdim=True).values.unsqueeze(1)
        self.range_obs_vals = (self.max_obs_values - self.min_border_factor * self.min_obs_values).unsqueeze(1)

        # Normalize obs_values
        self.obs_values = (self.obs_values - self.min_border_factor * self.min_obs_values) / self.range_obs_vals

        # Normalize locations using the same range
        if self.locations is not None:
            self.locations = (self.locations - self.min_border_factor * self.min_obs_values) / self.range_obs_vals

        # Normalize obs_times by dividing by max_obs_times
        obs_time_reshaped = self.obs_times.view(B, P * T, 1)
        self.max_obs_times = obs_time_reshaped.max(dim=1, keepdim=True).values.unsqueeze(1)  # [B,1,1,1]
        self.range_obs_times = self.max_obs_times  # times are assumed to start at zero

        self.obs_times = self.obs_times / self.range_obs_times
        self.is_input_normalized = True

    def normalize_concepts(self, drift_data, var_drift_data, diffusion_data, var_diffusion_data):
        # Normalize drift and diffusion targets
        if drift_data is not None:
            # scale from times normalisation
            drift_data = drift_data / self.range_obs_times
            drift_data = drift_data * self.range_obs_vals

        if diffusion_data is not None:
            # scale from times normalisation
            diffusion_data = diffusion_data / torch.sqrt(self.range_obs_vals)
            diffusion_data = diffusion_data * self.range_obs_times

        # var part
        if var_drift_data is not None:
            var_drift_data = var_drift_data + 2 * torch.log(self.range_obs_vals) - 2 * torch.log(self.range_obs_times)

        if var_diffusion_data is not None:
            var_diffusion_data = var_diffusion_data + 2.0 * torch.log(self.range_obs_vals) - torch.log(self.range_obs_times)

        return drift_data, diffusion_data

    def unnormalize_concepts(self, drift_data, var_drift_data, diffusion_data, var_diffusion_data):
        # Normalize drift and diffusion targets
        if drift_data is not None:
            # scale from times normalisation
            drift_data = drift_data * self.range_obs_times
            drift_data = drift_data / self.range_obs_vals

        if diffusion_data is not None:
            # scale from times normalisation
            diffusion_data = diffusion_data * torch.sqrt(self.range_obs_vals)
            diffusion_data = diffusion_data / self.range_obs_times

        # var part
        if var_drift_data is not None:
            var_drift_data = var_drift_data - torch.log(self.range_obs_vals) + torch.log(self.range_obs_times)

        if var_diffusion_data is not None:
            var_diffusion_data = var_diffusion_data - torch.log(self.range_obs_vals) + 0.5 * torch.log(self.range_obs_times)

        return drift_data, diffusion_data

    def normalize_targets(self):
        """Normalizes target data (drift_data and diffusion_data) using shared min/max values from data."""
        if not self.is_target_normalized:
            if self.is_target_set:
                self.drift_target, self.diffusion_target = self.normalize_concepts(self.drift_target, None, self.diffusion_target, None)
        self.is_target_normalized = True

    def normalize_estimators(self):
        """Normalizes estimator data (drift and diffusion estimators) using shared min/max values from data."""
        if not self.is_estimator_normalized:
            if self.is_estimator_set:
                self.drift_estimator, self.diffusion_estimator = self.normalize_concepts(
                    self.drift_estimator, self.log_var_drift_estimator, self.diffusion_estimator, self.log_var_diffusion_estimator
                )
        self.is_estimator_normalized = True

    def unnormalize_input(self):
        """Restores original scale for normalized fields."""
        if not self.is_input_normalized:
            return

        # Restore original scale for obs_values and locations
        self.obs_values = self.obs_values * self.range_obs_vals + self.min_obs_values
        if self.locations is not None:
            self.locations = self.locations * self.range_obs_vals + self.min_obs_values

        # Restore original scale for obs_times
        self.obs_times = self.obs_times * self.range_obs_times

        self.is_input_normalized = False

    def unnormalize_targets(self):
        if self.is_target_normalized:
            if self.is_target_set:
                self.drift_target, self.diffusion_target = self.unnormalize_concepts(self.drift_target, None, self.diffusion_target, None)
        self.is_target_normalized = True

    def unnormalize_estimators(self):
        if self.is_estimator_normalized:
            if self.is_estimator_set:
                self.drift_estimator, self.diffusion_estimator = self.unnormalize_concepts(
                    self.drift_estimator, self.log_var_drift_estimator, self.diffusion_estimator, self.log_var_diffusion_estimator
                )
        self.is_estimator_normalized = True

    def normalize_all(self):
        if not self.is_input_normalized:
            self.normalize_input()
        if not self.is_estimator_normalized:
            self.normalize_estimators()
        if not self.is_target_normalized:
            self.normalize_targets()

    def unnormalize_all(self):
        if self.is_input_normalized:
            self.unnormalize_input()
        if self.is_estimator_normalized:
            self.unnormalize_estimators()
        if self.is_target_normalized:
            self.unnormalize_targets()


# 3. Model Following FIM conventions
# class FIMSDE(AModel)
class FIMSDE(pl.LightningModule):
    """
    Stochastic Differential Equation Trainining
    """

    model_config: FIMSDEConfig
    data_config: FIMDatasetConfig

    def __init__(
        self,
        model_config: dict,
        data_config: dict,
        device_map: torch.device = None,
        **kwargs,
    ):
        super(FIMSDE, self).__init__()

        # Save hyperparameters
        self.save_hyperparameters()

        # Set hyperparameters
        if isinstance(model_config, dict):
            self.model_config = FIMSDEConfig(**model_config)
        else:
            self.model_config = model_config

        if isinstance(data_config, dict):
            self.data_config = FIMDatasetConfig(**data_config)
        else:
            self.data_config = data_config

        self._create_modules()

        # Set a dataset for fixed evaluation
        self.target_data = generate_all(self.model_config)

        if device_map is not None:
            self.to(device_map)

        self.DatabatchNameTuple = FIMSDEDatabatchTuple
        # Important: This property activates manual optimization (Lightning)
        self.automatic_optimization = False

    def _create_modules(
        self,
    ):
        # Define different versions
        x_dimension = self.data_config.max_dimension
        x_dimension_full = x_dimension * 3  # we encode the difference and its square
        spatial_plus_time_encoding = self.model_config.temporal_embedding_size + self.model_config.spatial_embedding_size
        self.psi_1_tokes_dim = self.model_config.sequence_encoding_tokenizer * self.model_config.sequence_encoding_transformer_heads

        # basic embedding
        self.phi_0t = SineTimeEncoding(self.model_config.temporal_embedding_size)
        self.phi_0x = MLP(
            in_features=x_dimension_full,
            out_features=self.model_config.spatial_embedding_size,
            hidden_layers=self.model_config.spatial_embedding_hidden_layers,
        )

        # trunk network
        self.trunk = MLP(
            in_features=x_dimension, out_features=self.psi_1_tokes_dim, hidden_layers=self.model_config.trunk_net_hidden_layers
        )

        # ensures that the embbeding that is sent to the transformer is a multiple of the number of heads
        self.phi_xt = nn.Linear(spatial_plus_time_encoding, self.psi_1_tokes_dim)

        # path transformer (causal encoding of paths)
        self.psi_1 = TransformerModel(
            input_dim=self.psi_1_tokes_dim,
            nhead=self.model_config.sequence_encoding_transformer_heads,
            hidden_dim=self.model_config.sequence_encoding_transformer_hidden_size,
            nlayers=self.model_config.sequence_encoding_transformer_layers,
        )

        # time attention
        self.omega_1 = nn.MultiheadAttention(
            self.psi_1_tokes_dim,
            self.model_config.combining_transformer_heads,
            batch_first=True,
        )

        # path attention
        self.path_queries = nn.Parameter(torch.randn(1, self.psi_1_tokes_dim))

        self.omega_2 = nn.MultiheadAttention(
            self.psi_1_tokes_dim,
            self.model_config.combining_transformer_heads,
            batch_first=True,
        )

        # drift head
        self.drift_head = nn.Linear(self.psi_1_tokes_dim, self.data_config.max_dimension)
        self.log_var_drift_head = nn.Linear(self.psi_1_tokes_dim, self.data_config.max_dimension)

        # diffusion head
        self.diffusion_head = nn.Linear(self.psi_1_tokes_dim, self.data_config.max_dimension)
        self.log_var_diffusion_head = nn.Linear(self.psi_1_tokes_dim, self.data_config.max_dimension)

    def path_encoding(
        self,
        databatch: FIMSDEDatabatchTuple,
        locations: Optional[Tensor] = None,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """
        This function obtains the paths encodings with functional
        attention, the intent is to provide a representation for
        the series of paths

        Args:

            databatch:FIMSDEpDatabatchTuple|FIMSDEpDatabatch
                keys,values:
                    locations (Tensor [B, G, D])
                        where to evaluate the drift and diffusion function
                    obs_values (Tensor [B, P, T+1, D])
                        observation values. optionally with noise.
                    obs_times (Tensor [B, P, T+1, 1])
                    observation_mask (dtype: bool)
                        (0: value is observed, 1: value is masked out)

            locations (tensor):
                where to evaluate the drift and diffusion function

            training (bool):
                flag indicating if model is in training mode. Has an impact on the output.

            with B: batch size, T: number of observation times, P: number of paths, D: dimensionsm, G: number of fine grid points (locations)

        Returns:
            b(x) (Tensor)  [B,G,psi_1_tokes_dim] representation for system at each grid point
            h(x) (Tensor)  [B,P,H,psi_1_tokes_dim] representation per path at each grid point
        """
        if locations is None:
            locations = databatch.locations

        B, P, T, D = databatch.obs_values.shape
        T = T - 1
        G = locations.size(1)

        # include the square of the difference
        X = databatch.obs_values[:, :, :-1, :]
        dX = databatch.obs_values[:, :, 1:, :] - databatch.obs_values[:, :, :-1, :]
        obs_times = databatch.obs_times[:, :, :-1, :]

        dX2 = dX**2
        x_full = torch.cat([X.unsqueeze(-1), dX.unsqueeze(-1), dX2.unsqueeze(-1)], dim=-1)
        x_full = x_full.view(x_full.shape[0], x_full.shape[1], x_full.shape[2], -1)

        spatial_encoding = self.phi_0x(x_full)  # [B,P,T,spatial_embedding_size]
        time_encoding = self.phi_0t(obs_times)  # [B,P,T,temporal_embedding_size]

        # trunk
        trunk_encoding = self.trunk(locations)  # [B,H,trunk_dim]
        trunk_encoding = trunk_encoding[:, None, :, :].repeat(1, P, 1, 1)  # [B,P,G,trunk_size]
        trunk_encoding = trunk_encoding.view(B * P, G, -1)

        # embbedded input
        U = torch.cat([spatial_encoding, time_encoding], dim=-1)  #  [B,P,T,spatial_plus_time_encoding]
        U = self.phi_xt(U)  #  [B,P,T,psi_1_tokes_dim]

        # TRANSFORMER THAT CREATES A REPRESENTATION FOR THE PATHS
        U = U.view(B * P, T, self.psi_1_tokes_dim)
        H = self.psi_1(torch.transpose(U, 0, 1))  # [T,B*P,psi_1_tokes_dim]
        H = torch.transpose(H, 0, 1)  # [B*P,T,psi_1_tokes_dim]

        # Attention on Time -> One representation per path
        hx, _ = self.omega_1(trunk_encoding, H, H)  # [B*P,G,psi_1_tokes_dim]
        hx = hx.view(B, P, G, -1)  # [B,P,G,psi_1_tokes_dim]

        # Attention on Paths -> One representation per expression
        hx_ = hx.transpose(1, 2).reshape(G * B, P, -1)  # [B*G,P,psi_1_tokes_dim]
        path_queries_ = self.path_queries[None, :, :].repeat(G * B, 1, 1)
        bx, _ = self.omega_2(path_queries_, hx_, hx_)
        bx = bx.view(B, G, -1)  # [B,G,psi_1_tokes_dim]

        return bx, hx

    def forward(
        self,
        databatch: FIMSDEDatabatchTuple,
        locations: Optional[Tensor] = None,
        training: bool = True,
        return_all: bool = False,
        return_heads: bool = False,
    ) -> Tuple[Tensor] | FIMSDEForward:
        """
        Args:
            databatch FIMPOODEDataBulk
            training (bool) if True returns Dict

            Returns
                if training true returns Dict of losses

                if return_all true returns FIMSDEForward with everything
        """
        # Dataclass to Handle Normalization
        forward_expressions = FIMSDEForward()

        if hasattr(databatch, "obs_mask"):
            obs_mask = databatch.obs_mask
        else:
            obs_mask = None

        forward_expressions.set_input_data(
            obs_times=databatch.obs_times,
            obs_values=databatch.obs_values,
            obs_mask=obs_mask,
            locations=databatch.locations,
            dimension_mask=databatch.dimension_mask,
        )
        forward_expressions.normalize_input()

        # Path Encoding
        bx, hx = self.path_encoding(databatch, locations)

        # Drift Heads
        drift_estimator = self.drift_head(bx)
        log_var_drift_estimator = self.log_var_drift_head(bx)
        diffusion_estimator = self.diffusion_head(bx)
        log_var_diffusion_estimator = self.log_var_diffusion_head(bx)

        forward_expressions.set_forward_estimators(
            drift_estimator=drift_estimator,
            diffusion_estimator=diffusion_estimator,
            var_drift_estimator=log_var_drift_estimator,
            var_diffusion_estimator=log_var_diffusion_estimator,
        )

        # Loss
        forward_expressions.set_target_data(drift_data=databatch.drift_at_locations, diffusion_data=databatch.diffusion_at_locations)

        # Returns
        if training:
            losses = self.loss(forward_expressions)
            return {"losses": losses}
        else:
            if return_all:
                losses = self.loss(forward_expressions)
                forward_expressions.set_losses(losses)
                forward_expressions.unnormalize_all()
                return forward_expressions

            if return_heads:
                forward_expressions.unnormalize_all()
                return (
                    forward_expressions.drift_estimator,
                    forward_expressions.log_var_drift_estimator,
                    forward_expressions.diffusion_estimator,
                    forward_expressions.log_var_diffusion_estimator,
                )

    def var_loss(self, estimator, target, log_var_estimator, dimension_mask):
        """
        loss with log var
        """
        loss_ = (estimator - target) ** 2.0
        var = torch.exp(log_var_estimator)
        loss_ = loss_ / (2.0 * var) + 0.5 * log_var_estimator

        # Apply the dimension mask and keep finite values
        loss_masked = torch.where((dimension_mask == 1) & torch.isfinite(loss_), loss_, torch.zeros_like(loss_))
        # Replace NaNs and Infs with zeros in the masked loss
        loss_ = torch.where(torch.isfinite(loss_masked), loss_masked, torch.zeros_like(loss_masked))

        # filter out
        loss_ = loss_.sum(-1)  # sum dimension
        loss_ = loss_.sum(-1)  # sum time
        loss_ = torch.sqrt(loss_.mean())

        return loss_

    def rmse_loss(self, estimator, target, dimension_mask):
        """
        root mean square loss applying the dimension masks
        """
        loss_ = (estimator - target) ** 2.0

        # Apply the dimension mask and keep finite values
        loss_masked = torch.where((dimension_mask == 1) & torch.isfinite(loss_), loss_, torch.zeros_like(loss_))
        # Replace NaNs and Infs with zeros in the masked loss
        loss_ = torch.where(torch.isfinite(loss_masked), loss_masked, torch.zeros_like(loss_masked))

        # Filter out NaNs in the masked tensor
        loss_ = loss_.sum(-1)
        loss_ = loss_.sum(-1)
        loss_ = torch.sqrt(loss_.mean())
        return loss_

    def loss(
        self,
        forward_expressions: FIMSDEForward,
    ):
        """
        forward_expressions

        Compute the loss of the FIMODE_mix model (in original space).
            Makes sure that the mask is included in the computation of the loss

        The loss consists of supervised losses
            - rmse of the vector field values at fine grid points

        Args:
            forward_expressions (FIMSDEForward):
        Returns:
            Tensor: {"total_loss":total_loss,"drift_loss":drift_loss,"diffusion_loss":diffusion_loss}
        """
        # ENSURES THAT ESTIMATOR AND TARGET LIE IN THE SAME UNITS
        if self.model_config.train_with_normalized_head:
            forward_expressions.normalize_all()
        else:
            forward_expressions.unnormalize_all()

        if self.model_config.loss_type == "rmse":
            drift_loss = self.rmse_loss(
                forward_expressions.drift_estimator, forward_expressions.drift_target, forward_expressions.dimension_mask
            )

            diffusion_loss = self.rmse_loss(
                forward_expressions.diffusion_estimator, forward_expressions.diffusion_target, forward_expressions.dimension_mask
            )

        elif self.model_config.loss_type == "var":
            drift_loss = self.var_loss(
                forward_expressions.drift_estimator,
                forward_expressions.drift_target,
                forward_expressions.log_var_drift_estimator,
                forward_expressions.dimension_mask,
            )

            diffusion_loss = self.var_loss(
                forward_expressions.diffusion_estimator,
                forward_expressions.diffusion_target,
                forward_expressions.log_var_diffusion_estimator,
                forward_expressions.dimension_mask,
            )

        total_loss = drift_loss + self.model_config.diffusion_loss_scale * diffusion_loss
        losses = {"loss": total_loss, "drift_loss": drift_loss, "diffusion_loss": diffusion_loss}
        return losses

    # ----------------------------- Lightning Functionality ---------------------------------------------
    def prepare_batch(self, batch) -> FIMSDEDatabatchTuple:
        """lightning will convert name tuple into a full tensor for training
        here we create the nametuple as requiered for the model
        """
        databatch = self.DatabatchNameTuple(*batch)
        return databatch

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        databatch: FIMSDEDatabatchTuple = self.prepare_batch(batch)
        losses = self.forward(databatch, training=True)

        total_loss = losses["losses"]["loss"]
        drift_loss = losses["losses"]["drift_loss"]
        diffusion_loss = losses["losses"]["diffusion_loss"]

        optimizer.zero_grad()
        self.manual_backward(total_loss)
        if self.model_config.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.model_config.clip_max_norm)
        optimizer.step()

        self.log("loss", total_loss, on_step=True, prog_bar=True, logger=True)
        self.log("drift_loss", drift_loss, on_step=True, prog_bar=True, logger=True)
        self.log("diffusion_loss", diffusion_loss, on_step=True, prog_bar=True, logger=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        databatch = self.prepare_batch(batch)
        forward_values = self.forward(databatch, training=False, return_all=True)

        total_loss = forward_values.losses["loss"]
        drift_loss = forward_values.losses["drift_loss"]
        diffusion_loss = forward_values.losses["diffusion_loss"]

        self.log("val_loss", total_loss, on_step=False, prog_bar=True, logger=True)
        self.log("drift_loss", drift_loss, on_step=False, prog_bar=True, logger=True)
        self.log("diffusion_loss", diffusion_loss, on_step=False, prog_bar=True, logger=True)

        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.model_config.learning_rate)

    def on_train_epoch_start(self):
        # Action to be executed at the start of each training epoch
        if (self.current_epoch + 1) % self.model_config.log_images_every_n_epochs == 0:
            pipeline = FIMSDEPipeline(self)
            pipeline_sample = pipeline(self.target_data)
            self.images_log(self.target_data, pipeline_sample)

    def images_log(self, databatch, pipeline_sample):
        fig = images_log_1D(databatch, pipeline_sample)
        if fig is not None:
            mlflow_client_ = self.logger.experiment
            mlflow_client_.log_figure(self.logger._run_id, fig, f"{self.current_epoch}_1D.png")

        fig = images_log_2D(databatch, pipeline_sample)
        if fig is not None:
            mlflow_client_ = self.logger.experiment
            mlflow_client_.log_figure(self.logger._run_id, fig, f"{self.current_epoch}_2D.png")

        fig = images_log_3D(databatch, pipeline_sample)
        if fig is not None:
            mlflow_client_ = self.logger.experiment
            mlflow_client_.log_figure(self.logger._run_id, fig, f"{self.current_epoch}_3D.png")


ModelFactory.register("FIMSDE", FIMSDE, with_data_params=True)
