from dataclasses import dataclass
from typing import Any, Dict, Optional, Self, Tuple

import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoConfig, AutoModel, PretrainedConfig

from fim import results_path
from fim.data.data_generation.dynamical_systems_target import generate_all
from fim.data.datasets import FIMSDEDatabatchTuple
from fim.models.blocks import AModel, ModelFactory
from fim.models.blocks.base import MLP, TransformerModel
from fim.models.blocks.positional_encodings import SineTimeEncoding
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
    def from_dbt(cls, databatch: FIMSDEDatabatchTuple | None, normalized: Optional[bool] = False) -> Self | None:
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
        temporal_embedding_size (int): Size of temporal embedding. Default is 19.
        spatial_embedding_size (int): Size of spatial embedding. Default is 19.
        spatial_embedding_hidden_layers (list[int]): Hidden layers for spatial embedding. Default is [25].
        sequence_encoding_tokenizer (int): Tokenizer size for sequence encoding. Default is 5.
        sequence_encoding_transformer_hidden_size (int): Hidden size for transformer in sequence encoding. Default is 50.
        sequence_encoding_transformer_heads (int): Number of heads in transformer for sequence encoding. Default is 1.
        sequence_encoding_transformer_layers (int): Number of layers in transformer for sequence encoding. Default is 1.
        combining_transformer_hidden_size (int): Hidden size for combining transformer. Default is 50.
        combining_transformer_heads (int): Number of heads in combining transformer. Default is 1.
        combining_transformer_layers (int): Number of layers in combining transformer. Default is 1.
        trunk_net_size (int): Size of trunk network. Default is 50.
        trunk_net_hidden_layers (list[int]): Hidden layers for trunk network. Default is [25].
        values_norm_min (float): lower normalized values range boundary. Default is -1.
        values_norm_max (float): upper normalized values range boundary. Default is 1.
        times_norm_min (float): lower normalized times range boundary. Default is 0.
        times_norm_max (float): upper normalized times range boundary. Default is 1.
        loss_filter_nans (bool): Default is True.
        num_epochs (int): Number of epochs. Default is 2.
        add_delta_x_to_value_encoder (bool): Whether to add delta x to value encoder. Default is True.
        learning_rate (float): Learning rate. Default is 1.0e-5.
        weight_decay (float): Weight decay. Default is 1.0e-4.
        dropout_rate (float): Dropout rate. Default is 0.1.
        diffusion_loss_scale (float): Scale for diffusion loss. Default is 1.0.
        loss_threshold (float): Threshold for loss. Default is 100.0.
        loss_type (str): Type of loss. Default is "rmse".
        log_images_every_n_epochs (int): Log images every n epochs. Default is 2.
        train_with_normalized_head (bool): Train with normalized head. Default is True.
        clip_grad (bool): Whether to clip gradients. Default is True.
        clip_max_norm (float): Maximum norm for gradient clipping. Default is 10.0.
        skip_nan_grads (bool): Skip optimizer update if (at least one) gradient is Nan. Default is True.
        dt_pipeline (float): Time step for pipeline. Default is 0.01.
        number_of_time_steps_pipeline (int): Number of time steps in the pipeline. Default is 128.
        evaluate_with_unnormalized_heads (bool): Evaluate with unnormalized heads. Default is True.
    """

    model_type = "fimsde"

    def __init__(
        self,
        name: str = "FIMSDE",
        experiment_name: str = "sde",
        experiment_dir: str = rf"{results_path}",
        max_dimension: int = 3,
        max_time_steps: int = 128,
        max_location_size: int = 1024,
        max_num_paths: int = 30,
        temporal_embedding_size: int = 19,
        spatial_embedding_size: int = 19,
        spatial_embedding_hidden_layers: list[int] = None,
        sequence_encoding_tokenizer: int = 5,
        sequence_encoding_transformer_hidden_size: int = 50,
        sequence_encoding_transformer_heads: int = 1,
        sequence_encoding_transformer_layers: int = 1,
        combining_transformer_hidden_size: int = 50,
        combining_transformer_heads: int = 1,
        combining_transformer_layers: int = 1,
        trunk_net_size: int = 50,
        trunk_net_hidden_layers: list[int] = None,
        values_norm_min: float = -1.0,
        values_norm_max: float = 1.0,
        times_norm_min: float = 0.0,
        times_norm_max: float = 1.0,
        loss_filter_nans: bool = True,
        lightning_training: bool = True,
        num_epochs: int = 2,  # training variables (MAYBE SEPARATED LATER)
        add_delta_x_to_value_encoder: bool = True,
        learning_rate: float = 1.0e-5,
        weight_decay: float = 1.0e-4,
        dropout_rate: float = 0.1,
        diffusion_loss_scale: float = 1.0,
        loss_threshold: float = 100.0,
        loss_type: str = "rmse",
        log_images_every_n_epochs: int = 2,
        train_with_normalized_head: bool = True,
        clip_grad: bool = True,
        clip_max_norm: float = 10.0,
        skip_nan_grads: bool = True,
        dt_pipeline: float = 0.01,
        number_of_time_steps_pipeline: int = 128,
        evaluate_with_unnormalized_heads: bool = True,
        **kwargs,
    ):
        self.name = name
        self.experiment_name = experiment_name
        self.experiment_dir = experiment_dir
        self.max_dimension = max_dimension
        self.max_time_steps = max_time_steps
        self.max_location_size = max_location_size
        self.max_num_paths = max_num_paths
        self.temporal_embedding_size = temporal_embedding_size
        self.spatial_embedding_size = spatial_embedding_size
        self.spatial_embedding_hidden_layers = spatial_embedding_hidden_layers or [25]
        self.sequence_encoding_tokenizer = sequence_encoding_tokenizer
        self.sequence_encoding_transformer_hidden_size = sequence_encoding_transformer_hidden_size
        self.sequence_encoding_transformer_heads = sequence_encoding_transformer_heads
        self.sequence_encoding_transformer_layers = sequence_encoding_transformer_layers
        self.combining_transformer_hidden_size = combining_transformer_hidden_size
        self.combining_transformer_heads = combining_transformer_heads
        self.combining_transformer_layers = combining_transformer_layers
        self.trunk_net_size = trunk_net_size
        self.trunk_net_hidden_layers = trunk_net_hidden_layers or [25]
        # normalization
        self.values_norm_min = values_norm_min
        self.values_norm_max = values_norm_max
        self.times_norm_min = times_norm_min
        self.times_norm_max = times_norm_max
        # regularization
        self.loss_filter_nans = loss_filter_nans
        # training variables
        self.num_epochs = num_epochs
        self.lightning_training = lightning_training
        self.add_delta_x_to_value_encoder = add_delta_x_to_value_encoder
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.diffusion_loss_scale = diffusion_loss_scale
        self.loss_threshold = loss_threshold
        self.loss_type = loss_type
        self.log_images_every_n_epochs = log_images_every_n_epochs
        self.train_with_normalized_head = train_with_normalized_head
        self.clip_grad = clip_grad
        self.clip_max_norm = clip_max_norm
        self.skip_nan_grads = skip_nan_grads
        self.dt_pipeline = dt_pipeline
        self.number_of_time_steps_pipeline = number_of_time_steps_pipeline
        self.evaluate_with_unnormalized_heads = evaluate_with_unnormalized_heads
        super().__init__(**kwargs)


# 3. Model Following FIM conventions
# class FIMSDE(AModel)
class FIMSDE(AModel, pl.LightningModule):
    """
    Stochastic Differential Equation Trainining
    """

    config_class = FIMSDEConfig

    def __init__(
        self,
        config: dict | FIMSDEConfig | FIMSDEConfig,
        device_map: torch.device = None,
        **kwargs,
    ):
        AModel.__init__(self, config, **kwargs)
        pl.LightningModule.__init__(self)

        # Save hyperparameters
        self.save_hyperparameters()

        # Set hyperparameters
        if isinstance(config, dict):
            self.config = FIMSDEConfig(**config)
        else:
            self.config = config

        self._create_modules()

        # Set a dataset for fixed evaluation
        self.target_data = generate_all(self.config.max_time_steps, self.config.max_num_paths)

        if device_map is not None:
            self.to(device_map)

        self.DatabatchNameTuple = FIMSDEDatabatchTuple
        # Important: This property activates manual optimization (Lightning)
        self.automatic_optimization = False

    def _create_modules(
        self,
    ):
        # Define different versions
        x_dimension = self.config.max_dimension
        x_dimension_full = x_dimension * 3  # we encode the difference and its square
        spatial_plus_time_encoding = self.config.temporal_embedding_size + self.config.spatial_embedding_size
        self.psi_1_tokes_dim = self.config.sequence_encoding_tokenizer * self.config.sequence_encoding_transformer_heads

        # basic embedding
        self.phi_0t = SineTimeEncoding(self.config.temporal_embedding_size)
        self.phi_0x = MLP(
            in_features=x_dimension_full,
            out_features=self.config.spatial_embedding_size,
            hidden_layers=self.config.spatial_embedding_hidden_layers,
        )

        # trunk network
        self.trunk = MLP(in_features=x_dimension, out_features=self.psi_1_tokes_dim, hidden_layers=self.config.trunk_net_hidden_layers)

        # ensures that the embbeding that is sent to the transformer is a multiple of the number of heads
        self.phi_xt = nn.Linear(spatial_plus_time_encoding, self.psi_1_tokes_dim)

        # path transformer (causal encoding of paths)
        self.psi_1 = TransformerModel(
            input_dim=self.psi_1_tokes_dim,
            nhead=self.config.sequence_encoding_transformer_heads,
            hidden_dim=self.config.sequence_encoding_transformer_hidden_size,
            nlayers=self.config.sequence_encoding_transformer_layers,
        )

        # time attention
        self.omega_1 = nn.MultiheadAttention(
            self.psi_1_tokes_dim,
            self.config.combining_transformer_heads,
            batch_first=True,
        )

        # path attention
        self.path_queries = nn.Parameter(torch.randn(1, self.psi_1_tokes_dim))

        self.omega_2 = nn.MultiheadAttention(
            self.psi_1_tokes_dim,
            self.config.combining_transformer_heads,
            batch_first=True,
        )

        # drift head
        self.drift_head = nn.Linear(self.psi_1_tokes_dim, self.config.max_dimension)
        self.log_var_drift_head = nn.Linear(self.psi_1_tokes_dim, self.config.max_dimension)

        # diffusion head
        self.diffusion_head = nn.Linear(self.psi_1_tokes_dim, self.config.max_dimension)
        self.log_var_diffusion_head = nn.Linear(self.psi_1_tokes_dim, self.config.max_dimension)

    def path_encoding(self, obs_times: Tensor, obs_values: Tensor, locations: Tensor) -> Tuple[torch.tensor, torch.tensor]:
        """
        This function obtains the paths encodings with functional
        attention, the intent is to provide a representation for
        the series of paths

        Args:
            obs_times (Tensor): observation times of obs_values. Shape: [B, P, T, 1]
            obs_values (Tensor): observation values. optionally with noise. Shape: [B, P, T, D]
            locations (Tensor): where to evaluate the drift and diffusion function. Shape: [B, G, D]
            where B: batch size P: number of paths T: number of time steps G: location grid size D: dimensions

        Returns:
            b(x) (Tensor)  [B,G,psi_1_tokes_dim] representation for system at each grid point
            h(x) (Tensor)  [B,P,G,psi_1_tokes_dim] representation per path at each grid point
        """

        B, P, T, D = obs_values.shape
        T = T - 1
        G = locations.size(1)

        # Trunk
        trunk_encoding = self.trunk(locations)  # [B,G,trunk_dim]
        trunk_encoding = trunk_encoding[:, None, :, :].repeat(1, P, 1, 1)  # [B,P,G,trunk_size]
        trunk_encoding = trunk_encoding.view(B * P, G, -1)

        # Embedded input; include difference and squared difference to next observation -> drop last observation
        X = obs_values[:, :, :-1, :]
        dX = obs_values[:, :, 1:, :] - obs_values[:, :, :-1, :]
        dX2 = dX**2

        x_full = torch.cat([X, dX, dX2], dim=-1)  # [B,P,T,3*D]
        spatial_encoding = self.phi_0x(x_full)  # [B,P,T,spatial_embedding_size]

        obs_times = obs_times[:, :, :-1, :]  # [B,P,T,1]
        time_encoding = self.phi_0t(obs_times)  # [B,P,T,temporal_embedding_size]

        U = torch.cat([spatial_encoding, time_encoding], dim=-1)  #  [B,P,T,spatial_plus_time_encoding]
        U = self.phi_xt(U)  #  [B,P,T,psi_1_tokes_dim]

        # Transformer that creates a representation for the paths
        U = U.view(B * P, T, self.psi_1_tokes_dim)
        H = self.psi_1(torch.transpose(U, 0, 1))  # [T,B*P,psi_1_tokes_dim]
        H = torch.transpose(H, 0, 1)  # [B*P,T,psi_1_tokes_dim]

        # Attention on Time -> One representation per path and location
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
        return_losses: bool = False,
        schedulers: dict = None,
        step: int = 0,
    ) -> dict | tuple[SDEConcepts, dict]:
        """
        Args:
            databatch (FIMSDEDatabatchTuple):
                obs_values (Tensor): observation values. optionally with noise. Shape: [B, P, T, D]
                obs_times (Tensor): observation times of obs_values. Shape: [B, P, T, 1]
                locations (Tensor): where to evaluate the drift and diffusion function. Shape: [B, G, D]
                drift/diffusion_at_locations (Tensor): ground-truth concepts at locations. Shape: [B, G, D]
                dimension_mask (Tensor): 0 at padded dimensions of ground-truth data at locations. Shape: [B, G, D]
                where B: batch size P: number of paths T: number of time steps G: location grid size D: dimensions

            training (bool): if True returns only dict with losses, including training objective
            return_losses (bool): is True computes and returns losses, even if training is False

            Returns
                estimated_concepts (SDEConcepts): Estimated concepts at locations. Shape: [B, G, D]
                if training == True or return_losses == True return (additionally):
                    losses (dict): training objective has key "loss", other keys are auxiliary for monitoring

        """
        assert not hasattr(
            databatch, "obs_mask"
        ), "Observation masking not implemented yet. Neither in model forward, nor in neural operator or loss."

        # Default to passed locations, otherwise use databatch.locations
        if locations is None:
            locations = databatch.locations

        # Instance normalization
        values_norm_stats = NormalizationStats(
            databatch.obs_values, normalized_min=self.config.values_norm_min, normalized_max=self.config.values_norm_max
        )
        obs_values = values_norm_stats.normalization_map(databatch.obs_values)
        locations = values_norm_stats.normalization_map(locations)

        times_norm_stats = NormalizationStats(
            databatch.obs_times, normalized_min=self.config.times_norm_min, normalized_max=self.config.times_norm_max
        )
        obs_times = times_norm_stats.normalization_map(databatch.obs_times)

        # Encoding paths and locations into encoding per location
        bx, _ = self.path_encoding(obs_times, obs_values, locations)  # [B, G, psi_1_tokes_dim]

        # Heads: [B, G, D]
        drift_estimator = self.drift_head(bx)
        log_var_drift_estimator = self.log_var_drift_head(bx)
        diffusion_estimator = self.diffusion_head(bx)
        log_var_diffusion_estimator = self.log_var_diffusion_head(bx)

        estimated_concepts = SDEConcepts(
            locations=locations,
            drift=drift_estimator,
            diffusion=diffusion_estimator,
            log_var_drift=log_var_drift_estimator,
            log_var_diffusion=log_var_diffusion_estimator,
            normalized=True,
        )

        # Losses
        target_concepts: SDEConcepts | None = SDEConcepts.from_dbt(databatch)

        if hasattr(databatch, "dimension_mask"):
            dimension_mask = databatch.dimension_mask

        else:
            dimension_mask = torch.ones(estimated_concepts.drift.shape, dtype=bool)

        # Returns
        if training is True:
            losses: dict = self.loss(estimated_concepts, target_concepts, values_norm_stats, times_norm_stats, dimension_mask)
            return {"losses": losses}

        else:
            estimated_concepts.renormalize(values_norm_stats, times_norm_stats)

            if return_losses is True:
                losses: dict = self.loss(estimated_concepts, target_concepts, values_norm_stats, times_norm_stats, dimension_mask)
                return estimated_concepts, {"losses": losses}

            else:
                return estimated_concepts

    @staticmethod
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

    @staticmethod
    def rmse_at_locations(estimated: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """
        Return RMSE between target and estimated values per location. Mask indicates which values (in last dimension) to include.

        Args:
            estimated (Tensor): estimated of target values. Shape: [B, G, D]
            target (Tensor): Target values. Shape: [B, G, D]
            mask (Tensor): 0 at values to ignore in loss computations. Shape: [B, G, D]

        Return:
            rmse (Tensor): RMSE at locations. Shape: [B, G]
        """
        assert estimated.ndim == 3, "Got " + str(estimated.ndim)
        assert estimated.shape == target.shape, "Got " + str(estimated.shape) + " and " + str(target.shape)
        assert estimated.shape == mask.shape, "Got " + str(estimated.shape) + " and " + str(mask.shape)

        # squared error at non-masked values
        se = mask * (estimated - target) ** 2
        se = torch.sum(se, dim=-1)  # [B, G]

        # mean over non-masked values
        non_masked_values_count = torch.sum(mask, dim=-1)
        mse = se / torch.clip(non_masked_values_count, min=1)  # [B, G]

        # take root per location
        rmse = torch.sqrt(mse)  # [B, G]

        assert rmse.ndim == 2, "Got " + str(rmse.ndim)

        return rmse

    @staticmethod
    def filter_nans_from_vector_fields(estimated: Tensor, log_var_estimated: Tensor | None, target: Tensor, mask: Tensor) -> tuple[Tensor]:
        """
        Filter locations where either estimate or target is Nan (or infinite). Record percentage of inifnite values (of non-masked values).

        Args:
            vector field values (Tensor): Vector fields to filter. Shape: [B, G, D]
            mask (Tensor): 0 masks padded values to ignore in percentage calculation. Shape: [B, G, D]

        Returns
            filtered vector field values (Tensor): Shape: [B, G, D]
            are_finite_mask (Tensor): marks finite values with 1. Shape: [B, G, D]
            estimated / target infinite perc (Tensor): Percentage of infinites in batch. Shape: []
        """
        # mask Nans per vector field
        estimated_is_finite_mask = torch.isfinite(estimated)
        target_is_finite_mask = torch.isfinite(target)

        if log_var_estimated is not None:
            estimated_is_finite_mask = estimated_is_finite_mask * torch.isfinite(torch.exp(log_var_estimated))

        # combine masks
        are_finite_mask = estimated_is_finite_mask * target_is_finite_mask

        # fill Nans with 0s
        estimated = torch.where(are_finite_mask, estimated, 0.0)
        target = torch.where(are_finite_mask, target, 0.0)

        if log_var_estimated is not None:
            log_var_estimated = torch.where(are_finite_mask, log_var_estimated, 0.0)

        # percentage of infinite values in batch, considering already masked values
        non_masked_values_count = torch.clip(torch.sum(mask), min=1)
        estimated_is_infinite_perc = (
            torch.sum(torch.logical_not(estimated_is_finite_mask) * mask, dtype=torch.float32) / non_masked_values_count
        )
        target_is_infinite_perc = torch.sum(torch.logical_not(target_is_finite_mask) * mask, dtype=torch.float32) / non_masked_values_count

        return estimated, log_var_estimated, target, are_finite_mask, estimated_is_infinite_perc, target_is_infinite_perc

    @staticmethod
    def filter_loss_at_locations(loss_at_locations: Tensor, threshold: Optional[float] = None) -> tuple[Tensor]:
        """
        Return mask that filters losses at locations if they are Nan or (optionally) above a threshold. Record statistics about the filtered locations.

        Args:
            loss_at_locations (Tensor): Single loss value per location. Shape: [B, G]
            threshold (Optional[float]): If passed, filter out locations with loss above threshold.

        Returns:
            filter_mask (Tensor): Masks Nans or above threshold values with 0. Shape: [B, G]
            filtered_loss_locations_perc (Tensor): Percentage of filtered locations in batch. Shape: []
        """
        # mask locations with non-Nan loss values
        loss_is_finite_mask = torch.isfinite(loss_at_locations)  # [B, G]

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

        return loss_at_locations_mask, filtered_loss_locations_perc

    def vector_field_loss(self, estimated: Tensor, log_var_estimated: Tensor | None, target: Tensor, mask: Tensor) -> tuple[Tensor]:
        """
        Compute (regularized) loss of vector field values at locations. Return statistics about regularization for monitoring.
        Regularizations:
            Remove Nans and infinite values in passed vector fields.
            Per location, remove Nans and infinite values from calculated loss.
            Per location, remove losses exceeding a threshold.

        Args:
            vector field values (Tensor): Vector fields to compute loss with.  Shape: [B, G, D]
            mask (Tensor): 0 masks padded values to ignore in loss calculation. Shape: [B, G, D]

        Returns
            loss (Tensor): Loss of vector field. Shape: []
            filtered_loss_locations_perc (Tensor): Percentage of locations where loss is above threshold or Nan. Shape: []
            estimated_is_infinite_perc (Tensor): Percentage of locations where estimated vector field is Nan. Shape: []
            target_is_infinite_perc (Tensor): Percentage of locations where target vector field is Nan. Shape: []
        """
        # comparing vector field should have 3 dimensions and equal shape
        assert estimated.ndim == 3
        assert estimated.shape == target.shape
        assert estimated.shape == mask.shape
        if log_var_estimated is not None:
            assert estimated.shape == log_var_estimated.shape

        # filter Nans and infinite values
        if self.config.loss_filter_nans:
            estimated, log_var_estimated, target, are_finite_mask, estimated_is_infinite_perc, target_is_infinite_perc = (
                self.filter_nans_from_vector_fields(estimated, log_var_estimated, target, mask)
            )
            mask = mask * are_finite_mask

        else:
            estimated_is_infinite_perc, target_is_infinite_perc = None, None

        # ensure: compute gradients at non-masked values
        estimated = estimated * mask
        target = target * mask
        if log_var_estimated is not None:
            log_var_estimated = log_var_estimated * mask

        # compute loss per location
        if self.config.loss_type == "rmse":
            loss_at_locations = self.rmse_at_locations(estimated, target, mask)  # [B, G]

        elif self.config.loss_type == "var":
            assert log_var_estimated is not None, "Must pass ´log_var_estimated` to compute nll loss."
            loss_at_locations = self.gaussian_nll_at_locations(estimated, log_var_estimated, target, mask)  # [B, G]

        else:
            raise ValueError("`loss_type` must be `rmse` or `var`, got " + self.config.loss_type)

        # filter out Nans or above threshold locations from loss
        loss_at_locations_mask, filtered_loss_locations_perc = self.filter_loss_at_locations(loss_at_locations, self.config.loss_threshold)

        # mean loss at non-masked locations
        non_masked_values_count = torch.sum(loss_at_locations_mask)
        loss = torch.sum(loss_at_locations_mask * loss_at_locations) / torch.clip(non_masked_values_count, min=1)  # []

        return loss, filtered_loss_locations_perc, estimated_is_infinite_perc, target_is_infinite_perc

    def loss(
        self,
        estimated_concepts: SDEConcepts,
        target_concepts: SDEConcepts | None,
        values_norm_stats: NormalizationStats,
        times_norm_stats: NormalizationStats,
        dimension_mask: Optional[Tensor] = None,
    ):
        """
        Compute supervised losses (RMSE or NLL) of sde concepts at non-padded dimensions.

        Args:
            estimated_concepts (SDEConcepts): Learned SDEConcepts. Shape: [B, G, D]
            target_concepts (SDEConcepts ): Ground-truth, target SDEConcepts. Shape: [B, G, D]
            values_norm_stats (NormalizationStats): Values instance normalization statistics.
            times_norm_stats (NormalizationStats): Times instance normalization statistics.
            dimension_mask (Optional[Tensor]): Masks padded dimensions to ignore in loss computations. Shape: [B, G, D]

        Returns:
            losses (dict):
                total_loss (Tensor): Training objective: drift_loss + diffusion_scale * diffusion_loss. Shape: []
                drift_loss (Tensor): RMSE or NLL of drift estimation wrt. ground-truth. Shape: []
                diffusion_loss (Tensor): RMSE or NLL of diffusion estimation wrt. ground-truth. Shape: []
                + statistics about Nans and infinities during computations
        """
        assert target_concepts is not None, "Need ground-truth concepts at locations to compute train losses."

        if dimension_mask is None:
            dimension_mask = torch.ones(estimated_concepts.drift.shape, dtype=bool)

        else:
            dimension_mask = dimension_mask.bool()

        assert dimension_mask.shape == estimated_concepts.drift.shape, (
            "Shapes of mask " + str(dimension_mask.shape) + " and concepts " + str(estimated_concepts.drift.shape) + " need to be equal."
        )

        # Ensure that estimation and target are on same normalization
        if self.config.train_with_normalized_head:
            estimated_concepts.normalize(values_norm_stats, times_norm_stats)
            target_concepts.normalize(values_norm_stats, times_norm_stats)
        else:
            estimated_concepts.renormalize(values_norm_stats, times_norm_stats)
            target_concepts.renormalize(values_norm_stats, times_norm_stats)

        # compute loss per vector field
        drift_loss, drift_loss_above_threshold_or_nan_perc, drift_estimated_is_infinite_perc, drift_target_is_infinite_perc = (
            self.vector_field_loss(estimated_concepts.drift, estimated_concepts.log_var_drift, target_concepts.drift, dimension_mask)
        )
        (
            diffusion_loss,
            diffusion_loss_above_threshold_or_nan_perc,
            diffusion_estimated_is_infinite_perc,
            diffusion_target_is_infinite_perc,
        ) = self.vector_field_loss(
            estimated_concepts.diffusion, estimated_concepts.log_var_diffusion, target_concepts.diffusion, dimension_mask
        )

        # assemble losses
        total_loss = drift_loss + self.config.diffusion_loss_scale * diffusion_loss
        losses = {
            "loss": total_loss,
            "drift_loss": drift_loss,
            "diffusion_loss": diffusion_loss,
            "drift_loss_above_threshold_or_nan_perc": drift_loss_above_threshold_or_nan_perc,
            "diffusion_loss_above_threshold_or_nan_perc": diffusion_loss_above_threshold_or_nan_perc,
            "drift_estimated_is_infinite_perc": drift_estimated_is_infinite_perc,
            "diffusion_estimated_is_infinite_perc": diffusion_estimated_is_infinite_perc,
            "drift_target_is_infinite_perc": drift_target_is_infinite_perc,
            "diffusion_target_is_infinite_perc": diffusion_target_is_infinite_perc,
        }

        return losses

    # ----------------------------- Lightning Functionality ---------------------------------------------
    def prepare_batch(self, batch) -> FIMSDEDatabatchTuple:
        """lightning will convert name tuple into a full tensor for training
        here we create the nametuple as required for the model
        """
        databatch = self.DatabatchNameTuple(*batch)
        return databatch

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        databatch: FIMSDEDatabatchTuple = self.prepare_batch(batch)
        losses = self.forward(databatch, training=True)

        total_loss = losses.get("loss")

        optimizer.zero_grad()
        self.manual_backward(total_loss)
        if self.config.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.clip_max_norm)

        if self.config.skip_nan_grads is True:  # skip updates if gradients contain Nans
            grad_is_finite = [torch.isfinite(p.grad).all() if p.grad is not None else True for p in self.parameters()]
            if all(grad_is_finite):
                optimizer.step()

        else:
            optimizer.step()

        prog_bar_labels = ["loss", "drift_loss", "diffusion_loss"]

        for label, loss in losses.items():
            prog_bar = label in prog_bar_labels
            self.log(label, loss, on_step=True, prog_bar=prog_bar, logger=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        databatch = self.prepare_batch(batch)
        _, losses = self.forward(databatch, training=False, return_losses=True)

        total_loss = losses.get("loss")

        losses["val_loss"] = losses.pop("loss")  # take loss as val_loss
        prog_bar_labels = ["val_loss", "drift_loss", "diffusion_loss"]

        for label, loss in losses.items():
            prog_bar = label in prog_bar_labels
            self.log(label, loss, on_step=False, prog_bar=prog_bar, logger=True)

        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)

    def on_train_epoch_start(self):
        # Action to be executed at the start of each training epoch
        if (self.current_epoch + 1) % self.config.log_images_every_n_epochs == 0:
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

    def metric(self, y: Any, y_target: Any) -> Dict:
        return super().metric(y, y_target)


ModelFactory.register(FIMSDEConfig.model_type, FIMSDE)
AutoConfig.register(FIMSDEConfig.model_type, FIMSDEConfig)
AutoModel.register(FIMSDEConfig, FIMSDE)
