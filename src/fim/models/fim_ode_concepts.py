"""
Data concepts, model configuration, trajectory encoders, and output types for FIMODE.

Encoders
--------
TrajectoryEncoder      — 4-feature standard encoder (x, Δx, Δx², Δt).
                         Compatible with all pre-trained checkpoints.
AxialTrajectoryEncoder  — 5-feature encoder (x, Δx, Δx_back, Δx², Δt).
                         Uses a backward-difference feature for improved
                         handling of irregular / subsampled trajectories.
                         Not yet trained; reserved for future model variants.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Final, Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from transformers import PretrainedConfig

from fim.models.sde import Standardization, DeltaLogCentering, InstanceNormalization
from fim.models.blocks.neural_operators import ResidualEncoderLayer, AttentionOperator

StateNorm = InstanceNormalization
TimesNorm = InstanceNormalization
StatesNormStat = Tuple[Tensor]
TimesNormStat = Tuple[Tensor]


# ---------- ODEConcepts ----------

class ODEConcepts:
    """
    Container for the model's vector field predictions at query locations.
    Holds locations and drift tensors, and handles normalization / renormalization.

    For an ODE  dx/dt = f(x), the chain-rule transform under state normalization
    ŷ = g(y) and time normalization t̂ = h(t) gives:

        dŷ/dt̂ = g'(y) · f(y) · (dt/dt̂)

    No Ito correction — diffusion is zero. Normalization and renormalization are
    therefore pure Jacobian scalings of the drift.
    """

    def __init__(
        self,
        locations: Tensor,
        drift: Tensor,
        normalized: bool,
        states_norm: StateNorm,
        states_norm_stats: StatesNormStat,
        times_norm: TimesNorm,
        times_norm_stats: TimesNormStat,
    ):
        self.locations = locations
        self.drift = drift
        self.normalized = normalized
        self.states_norm = states_norm
        self.states_norm_stats = states_norm_stats
        self.times_norm = times_norm
        self.times_norm_stats = times_norm_stats

    def normalize(self) -> None:
        """Transform drift from physical to normalized coordinates."""
        if self.normalized:
            return
        grad = self.states_norm.normalization_map(
            self.locations, self.states_norm_stats, derivative_num=1
        )
        self.drift = self.drift * grad
        dummy_t = torch.zeros_like(self.locations[..., 0:1])
        t_inv_grad = self.times_norm.inverse_normalization_map(
            dummy_t, self.times_norm_stats, derivative_num=1
        )
        self.drift = self.drift * t_inv_grad
        self.locations = self.states_norm.normalization_map(
            self.locations, self.states_norm_stats
        )
        self.normalized = True

    def renormalize(self) -> None:
        """Transform drift from normalized back to physical coordinates."""
        if not self.normalized:
            return
        inv_grad = self.states_norm.inverse_normalization_map(
            self.locations, self.states_norm_stats, derivative_num=1
        )
        self.drift = self.drift * inv_grad
        dummy_t = torch.zeros_like(self.locations[..., 0:1])
        t_grad = self.times_norm.normalization_map(
            dummy_t, self.times_norm_stats, derivative_num=1
        )
        self.drift = self.drift * t_grad
        self.locations = self.states_norm.inverse_normalization_map(
            self.locations, self.states_norm_stats
        )
        self.normalized = False

    @classmethod
    def builder(cls) -> "ODEConcepts.ODEConceptsBuilder":
        return ODEConcepts.ODEConceptsBuilder()

    class ODEConceptsBuilder:
        _locations: Tensor
        _drift: Tensor
        _normalized: bool
        _states_norm: StateNorm
        _states_norm_stats: StatesNormStat
        _times_norm: TimesNorm
        _times_norm_stats: TimesNormStat

        def locations(self, v: Tensor) -> "ODEConcepts.ODEConceptsBuilder":
            self._locations = v
            return self

        def drift(self, v: Tensor) -> "ODEConcepts.ODEConceptsBuilder":
            self._drift = v
            return self

        def normalized(self, v: bool) -> "ODEConcepts.ODEConceptsBuilder":
            self._normalized = v
            return self

        def states_norm(self, v: StateNorm) -> "ODEConcepts.ODEConceptsBuilder":
            self._states_norm = v
            return self

        def times_norm(self, v: TimesNorm) -> "ODEConcepts.ODEConceptsBuilder":
            self._times_norm = v
            return self

        def states_norm_stats(self, v: StatesNormStat) -> "ODEConcepts.ODEConceptsBuilder":
            self._states_norm_stats = v
            return self

        def times_norm_stats(self, v: TimesNormStat) -> "ODEConcepts.ODEConceptsBuilder":
            self._times_norm_stats = v
            return self

        def build(self) -> "ODEConcepts":
            assert self._locations.shape[0] == self._drift.shape[0], "Batch size mismatch"
            assert self._locations.shape[-1] == self._drift.shape[-1], "Dimension mismatch"
            assert self._states_norm is not None and self._times_norm is not None
            return ODEConcepts(
                locations=self._locations,
                drift=self._drift,
                normalized=self._normalized,
                states_norm=self._states_norm,
                times_norm=self._times_norm,
                states_norm_stats=self._states_norm_stats,
                times_norm_stats=self._times_norm_stats,
            )


# ---------- Trajectory features ----------

@dataclass
class TrajectoryFeatures:
    """
    4-feature representation of observed trajectories (standard encoder).

    Shapes: [B, T, N-1, *] where N-1 = time steps minus last.
    """
    x: Tensor               # state values
    delta_x: Tensor         # forward increment:  x(t+h) - x(t)
    delta_x_squared: Tensor # squared forward increment
    delta_t: Tensor         # time increments  [B, T, N-1, 1]
    feature_mask: Tensor    # valid-step mask   [B, T, N-1, 1]


@dataclass
class AxialTrajectoryFeatures:
    """
    5-feature representation for the Axial encoder (not yet trained).

    Adds delta_x_back (backward difference) to TrajectoryFeatures.
    Shapes: [B, T, N-2, *] — interior positions 1..N-2 only.
    """
    x: Tensor               # state values at interior positions
    delta_x: Tensor         # forward increment:  x(t+h) - x(t)
    delta_x_back: Tensor    # backward increment: x(t) - x(t-h)
    delta_x_squared: Tensor # squared forward increment
    delta_t: Tensor         # time increments  [B, T, N-2, 1]
    feature_mask: Tensor    # valid-step mask   [B, T, N-2, 1]


# ---------- Model configuration ----------

@dataclass(kw_only=True)
class FIMODEModelConfig(PretrainedConfig):
    """Hyperparameters for the FIMODE model architecture."""

    _ACTIVATION: Final[str] = "torch.nn.GELU"
    model_type: str = "FIMODEModel"

    dim_max_trajectory: int
    use_bias_for_projection: bool
    dim_embed: int

    num_context_encoder_layers: int
    attention_method: Literal["linear", "nn_multihead"]
    attention_map: Optional[Literal["softmax", "elu"]]
    use_bias_in_attention: Optional[bool]
    use_query_residual_in_attention: Optional[bool]
    num_heads: int
    dim_feedforward: int
    dropout: float

    num_res_layers_functional_decoder: int
    num_res_layer_u_model: int
    dim_hidden_u_model: int
    dim_ffn_u_model: int

    times_norm_on_deltas: Optional[bool] = True

    # "standard" → 4-feature TrajectoryEncoder (checkpoint-compatible)
    # "axial"    → 5-feature AxialTrajectoryEncoder (future use)
    encoder_type: str = "standard"

    def get_attention_layer_config(self) -> Dict:
        return {
            "nhead": self.num_heads,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "activation": self._ACTIVATION,
            "bias": self.use_bias_in_attention,
            "query_residual": self.use_query_residual_in_attention,
            "attn_method": self.attention_method,
            "lin_feature_map": self.attention_map,
            "lin_normalize": True,
        }

    def get_projection_config(self) -> Dict:
        return {
            "name": "fim.models.blocks.base.MLP",
            "hidden_layers": (self.dim_embed, self.dim_embed),
            "hidden_act": {"name": self._ACTIVATION},
            "dropout": self.dropout,
        }

    def get_functional_decoder_config(self) -> Dict:
        return {
            "paths_block_attention": False,
            "num_res_layers": self.num_res_layers_functional_decoder,
            "attention": self.get_attention_layer_config(),
            "projection": self.get_projection_config(),
        }

    def get_uncertainty_estimator_config(self) -> Dict:
        cfg = self.get_functional_decoder_config()
        cfg["num_res_layers"] = self.num_res_layer_u_model
        cfg["projection"]["hidden_layers"] = (self.dim_hidden_u_model, self.dim_hidden_u_model)
        cfg["attention"]["dim_feedforward"] = self.dim_ffn_u_model
        return cfg


# ---------- TrajectoryEncoder (4-feature, standard, checkpoint-compatible) ----------

class TrajectoryEncoder(nn.Module):
    """
    4-feature context encoder: x, Δx, Δx², Δt.

    Compatible with all pre-trained FIMODE checkpoints.
    Requires dim_embed divisible by 4.
    """

    _NUM_FEATURES: Final[int] = 4

    @dataclass
    class Output:
        D: Tensor

    def __init__(self, config: FIMODEModelConfig):
        super().__init__()
        self.config = config
        assert config.dim_embed % self._NUM_FEATURES == 0, (
            f"dim_embed must be divisible by {self._NUM_FEATURES} for TrajectoryEncoder"
        )
        dim_proj = config.dim_embed // self._NUM_FEATURES

        self.x_proj              = nn.Linear(config.dim_max_trajectory, dim_proj, bias=config.use_bias_for_projection)
        self.delta_x_proj        = nn.Linear(config.dim_max_trajectory, dim_proj, bias=config.use_bias_for_projection)
        self.delta_x_squared_proj = nn.Linear(config.dim_max_trajectory, dim_proj, bias=config.use_bias_for_projection)
        self.delta_t_proj        = nn.Linear(1,                          dim_proj, bias=config.use_bias_for_projection)

        layer = ResidualEncoderLayer(
            d_model=config.dim_embed,
            batch_first=True,
            **config.get_attention_layer_config(),
        )
        self.context_encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.num_context_encoder_layers,
            enable_nested_tensor=False,
        )

    def forward(self, features: TrajectoryFeatures) -> "TrajectoryEncoder.Output":
        x    = self.x_proj(features.x)
        dx   = self.delta_x_proj(features.delta_x)
        dx2  = self.delta_x_squared_proj(features.delta_x_squared)
        dt   = self.delta_t_proj(features.delta_t)

        feature_vector = torch.cat([dt, x, dx, dx2], dim=-1)
        if self.config.use_bias_for_projection:
            feature_vector = feature_vector * features.feature_mask

        b, t, n, d = feature_vector.shape
        feature_vector = feature_vector.view(b, t * n, d)
        src_key_padding_mask = ~features.feature_mask.view(b, t * n, 1)

        D = self.context_encoder(feature_vector, src_key_padding_mask=src_key_padding_mask)
        D = D.view(b, t, n, d) * features.feature_mask

        return self.Output(D=D)


# ---------- AxialTrajectoryEncoder (5-feature, future use) ----------

class AxialTrajectoryEncoder(nn.Module):
    """
    5-feature context encoder: x, Δx, Δx_back, Δx², Δt.

    Adds a backward-difference feature (Δx_back) for improved handling of
    irregular and subsampled trajectories. Not yet trained; use with new
    checkpoints only.  Requires dim_embed divisible by 5.
    """

    _NUM_FEATURES: Final[int] = 5

    @dataclass
    class Output:
        D: Tensor

    def __init__(self, config: FIMODEModelConfig):
        super().__init__()
        self.config = config
        assert config.dim_embed % self._NUM_FEATURES == 0, (
            f"dim_embed must be divisible by {self._NUM_FEATURES} for AxialTrajectoryEncoder"
        )
        dim_proj = config.dim_embed // self._NUM_FEATURES

        self.x_proj               = nn.Linear(config.dim_max_trajectory, dim_proj, bias=config.use_bias_for_projection)
        self.delta_x_proj         = nn.Linear(config.dim_max_trajectory, dim_proj, bias=config.use_bias_for_projection)
        self.delta_x_back_proj    = nn.Linear(config.dim_max_trajectory, dim_proj, bias=config.use_bias_for_projection)
        self.delta_x_squared_proj = nn.Linear(config.dim_max_trajectory, dim_proj, bias=config.use_bias_for_projection)
        self.delta_t_proj         = nn.Linear(1,                          dim_proj, bias=config.use_bias_for_projection)

        layer = ResidualEncoderLayer(
            d_model=config.dim_embed,
            batch_first=True,
            **config.get_attention_layer_config(),
        )
        self.context_encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.num_context_encoder_layers,
            enable_nested_tensor=False,
        )

    def forward(self, features: AxialTrajectoryFeatures) -> "AxialTrajectoryEncoder.Output":
        x    = self.x_proj(features.x)
        dx   = self.delta_x_proj(features.delta_x)
        dx_b = self.delta_x_back_proj(features.delta_x_back)
        dx2  = self.delta_x_squared_proj(features.delta_x_squared)
        dt   = self.delta_t_proj(features.delta_t)

        feature_vector = torch.cat([dt, x, dx, dx_b, dx2], dim=-1)
        if self.config.use_bias_for_projection:
            feature_vector = feature_vector * features.feature_mask

        b, t, n, d = feature_vector.shape
        feature_vector = feature_vector.view(b, t * n, d)
        src_key_padding_mask = ~features.feature_mask.view(b, t * n, 1)

        D = self.context_encoder(feature_vector, src_key_padding_mask=src_key_padding_mask)
        D = D.view(b, t, n, d) * features.feature_mask

        return self.Output(D=D)


# ---------- Output containers ----------

@dataclass
class EncoderOutput:
    """Wraps the trajectory embedding D returned by the encoder."""
    D: Tensor

    def detach(self) -> "EncoderOutput":
        return EncoderOutput(D=self.D.detach())


@dataclass
class FIMODEOutput:
    """Full model output: vector field predictions + intermediates needed for training."""
    predictions: ODEConcepts
    D: EncoderOutput
    encoded_locations: Tensor
    feature_mask: Tensor


# ---------- Uncertainty estimator ----------

class UncertaintyEstimator(nn.Module):
    """
    Predicts a per-location uncertainty scalar u from encoded locations and trajectory embedding.

    Loss weighting: loss_weighted = loss * exp(-u) + u
    """

    def __init__(self, config: FIMODEModelConfig):
        super().__init__()
        self.functional_encoder = AttentionOperator(
            embed_dim=config.dim_embed,
            out_features=1,
            **config.get_uncertainty_estimator_config(),
        )

    def forward(self, model_out: FIMODEOutput) -> Tensor:
        u = self.functional_encoder(
            model_out.encoded_locations,
            model_out.D.D,
            observations_padding_mask=~model_out.feature_mask,
        )
        return u.squeeze(dim=2)

    @staticmethod
    def apply_u_to_loss(loss: Tensor, u: Tensor) -> Tensor:
        return loss * torch.exp(-u) + u
