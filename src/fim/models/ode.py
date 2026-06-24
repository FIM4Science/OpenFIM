"""
FIMODE — Foundation Inference Model for Ordinary Differential Equations.

Architecture, trajectory encoders, model configuration, and preprocessing.

Two encoder paths are supported (selected via model_config.encoder_type):
  "standard"  — 4-feature TrajectoryEncoder (x, Δx, Δx², Δt).
                Compatible with all pre-trained checkpoints.
  "axial"      — 5-feature AxialTrajectoryEncoder (x, Δx, Δx_back, Δx², Δt).
                Requires dim_embed divisible by 5. Not yet trained.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Final, Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoConfig, AutoModel, PretrainedConfig

from fim.models.blocks import AModel, ModelFactory
from fim.models.blocks.neural_operators import ResidualEncoderLayer, AttentionOperator
from fim.models.sde import Standardization, DeltaLogCentering

from fim.models.ode_trainer import (
    ODEConcepts,
    EncoderOutput,
    FIMODEOutput,
    FIMODEConfig,
    FIMODETrainingConfig,
    TrainingData,
    DataPreparation,
    LossFactory,
    TrainIntegrator,
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


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing — standard 4-feature path (checkpoint-compatible)
#
# Feature mask covers positions 0..N-2 minus the last *observed* position.
# This is the exact preprocessing used to train all published checkpoints.
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _backward_fill(x: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
    """Backward-fill masked positions with next observed value. Returns (filled_x, last_obs_idx)."""
    mask = torch.broadcast_to(mask, x.shape)
    mask = torch.flip(mask, dims=(-2,))
    x    = torch.flip(x,    dims=(-2,))
    mask_cumsum = torch.cumsum(mask, dim=-2)
    indices = torch.cummax(mask_cumsum * mask, dim=-2)[1]
    first_obs = torch.argmin(
        torch.where(mask_cumsum == 0, torch.inf, mask_cumsum), dim=-2, keepdim=True
    )
    indices = torch.where(mask_cumsum == 0, first_obs, indices)
    x = torch.gather(x, dim=-2, index=indices)
    _, _, n, _ = x.shape
    return torch.flip(x, dims=(-2,)), (n - first_obs) - 1


@torch.no_grad()
def _backward_fill_and_feature_mask(
    trajectories: Tensor, times: Tensor, mask: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Backward-fill trajectories/times; zero out the last observed step in the mask.
    Feature mask covers positions 0..N-2 excluding the last observed position.
    """
    assert mask.dtype == torch.bool
    times        = mask * times
    trajectories = mask * trajectories
    times,        last_obs_idx = _backward_fill(times, mask)
    trajectories, _             = _backward_fill(trajectories, mask)
    mask = mask.scatter(
        dim=2,
        index=last_obs_idx,
        src=torch.zeros_like(last_obs_idx, dtype=torch.bool),
    )
    return trajectories, times, mask[:, :, :-1, :].contiguous()


@torch.no_grad()
def _extract_features(
    trajectories: Tensor, times: Tensor, mask: Tensor
) -> TrajectoryFeatures:
    """Compute x, Δx, Δx², Δt at positions 0..N-2 (standard 4-feature path)."""
    X   = trajectories[:, :, :-1, :] * mask
    dX  = (trajectories[:, :, 1:, :] - trajectories[:, :, :-1, :]) * mask
    dX2 = dX ** 2
    dt  = (times[:, :, 1:, :] - times[:, :, :-1, :]) * mask
    return TrajectoryFeatures(x=X, delta_x=dX, delta_x_squared=dX2, delta_t=dt, feature_mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing — Axial 5-feature path (for AxialTrajectoryEncoder)
#
# Feature mask covers interior positions 1..N-2.
# Uses both backward-fill (for forward diffs) and forward-fill (for backward diffs).
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _forward_fill(x: Tensor, mask: Tensor) -> Tensor:
    """Forward-fill masked positions with the previous observed value."""
    mask = torch.broadcast_to(mask, x.shape)
    cumsum = torch.cumsum(mask.float(), dim=-2)
    indices = torch.cummax(cumsum * mask, dim=-2)[1]
    first_obs = torch.argmin(
        torch.where(cumsum > 0, cumsum, torch.full_like(cumsum, float("inf"))),
        dim=-2, keepdim=True,
    )
    indices = torch.where(cumsum == 0, first_obs, indices)
    return torch.gather(x, dim=-2, index=indices)


@torch.no_grad()
def _backward_fill_and_axial_feature_mask(
    trajectories: Tensor, times: Tensor, mask: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    """Backward-fill; feature mask covers interior positions 1..N-2."""
    assert mask.dtype == torch.bool
    times        = mask * times
    trajectories = mask * trajectories
    times,        _ = _backward_fill(times, mask)
    trajectories, _ = _backward_fill(trajectories, mask)
    return trajectories, times, mask[:, :, 1:-1, :].contiguous()


@torch.no_grad()
def _extract_axial_features(
    traj_bwd: Tensor, traj_fwd: Tensor, times: Tensor, mask: Tensor
) -> AxialTrajectoryFeatures:
    """Compute 5 features at interior positions 1..N-2."""
    X    = traj_bwd[:, :, 1:-1, :] * mask
    dX   = (traj_bwd[:, :, 2:,  :] - traj_bwd[:, :, 1:-1, :]) * mask
    dX_b = (traj_fwd[:, :, 1:-1, :] - traj_fwd[:, :, :-2,  :]) * mask
    dX2  = dX ** 2
    dt   = (times[:, :, 2:, :] - times[:, :, 1:-1, :]) * mask
    return AxialTrajectoryFeatures(
        x=X, delta_x=dX, delta_x_back=dX_b, delta_x_squared=dX2, delta_t=dt, feature_mask=mask
    )


def _sanity_check(
    trajectories_shape: Tuple,
    times_shape: Tuple,
    locations_shape: Tuple,
    mask_shape: Tuple,
) -> None:
    b, t, n, d = trajectories_shape
    bt, tt, nt, _ = times_shape
    bc, _, dc = locations_shape
    bm, tm, nm, dm = mask_shape
    assert b > 0 and t > 0 and n > 0 and d > 0
    assert b == bt == bc == bm and t == tt == tm and n == nt == nm and d == dc
    assert dm == 1, "Mask last dim must be 1 for broadcasting"


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class FIMODE(AModel):
    """
    Foundation Inference Model for ODEs.

    In-context learner: given a few observed trajectories of an unknown ODE,
    predicts the vector field (drift) and its uncertainty at arbitrary query locations.

    Architecture:
        TrajectoryEncoder  — self-attention over all time×path tokens → embedding D
        AttentionOperator  — locations cross-attend D → drift prediction
        u_model            — separate cross-attention head → scalar uncertainty u per location

    Training:
        Vector field mode: minimise MSE between predicted drift and ground-truth drift,
        optionally weighted by learned uncertainty (loss * exp(-u) + u).

    Checkpoint compatibility:
        The default "standard" encoder path is byte-for-byte identical to the
        published pre-trained FimOdeonUnified checkpoints.  Module attribute
        names (u_model, trajectory_encoder, ...) are preserved exactly.
    """

    config_class = FIMODEConfig

    def __init__(
        self,
        config: FIMODEConfig,
        device_map: Optional[torch.device] = None,
        **kwargs: Any,
    ):
        AModel.__init__(self, config, **kwargs)

        model_config = FIMODEModelConfig(**config.model_config)
        train_config = FIMODETrainingConfig(**config.train_config)
        self.model_config = model_config
        self.train_config = train_config

        encoder_type = getattr(model_config, "encoder_type", "standard")

        if encoder_type == "axial":
            assert model_config.dim_embed % 5 == 0, (
                "dim_embed must be divisible by 5 for AxialTrajectoryEncoder"
            )
            self.trajectory_encoder = AxialTrajectoryEncoder(model_config)
        else:
            assert model_config.dim_embed % 4 == 0, (
                "dim_embed must be divisible by 4 for TrajectoryEncoder"
            )
            self.trajectory_encoder = TrajectoryEncoder(model_config)

        # Normalization
        self.spatial_norm  = Standardization()
        self.temporal_norm = DeltaLogCentering()

        # Decoder: locations + D -> drift
        self.location_proj = nn.Sequential(
            nn.Linear(model_config.dim_max_trajectory, model_config.dim_embed,
                      bias=model_config.use_bias_for_projection),
            nn.ReLU(),
            nn.Linear(model_config.dim_embed, model_config.dim_embed,
                      bias=model_config.use_bias_for_projection),
        )
        self.functional_decoder = AttentionOperator(
            embed_dim=model_config.dim_embed,
            out_features=model_config.dim_max_trajectory,
            **model_config.get_functional_decoder_config(),
        )

        # Uncertainty head — attribute named u_model for checkpoint compatibility
        self.u_model = UncertaintyEstimator(model_config)

        # Training utilities
        self.data_preparation = DataPreparation(train_config)
        self.integrator       = TrainIntegrator(train_config)
        self._criterion       = LossFactory.create(train_config.loss_type)

        self._relative_epoch    = -1
        self._is_training_manual = False

        if device_map is not None:
            self.to(device_map)

    # ---------- Preprocessing ----------

    @torch.no_grad()
    def pad_if_necessary(self, value: Tensor) -> Tensor:
        """Zero-pad last dimension to dim_max_trajectory if needed."""
        d_in = value.shape[-1]
        if d_in >= self.model_config.dim_max_trajectory:
            return value
        missing = self.model_config.dim_max_trajectory - d_in
        pad = torch.zeros_like(value[..., 0:1]).expand(value.shape[:-1] + (missing,))
        return torch.cat([value, pad], dim=-1)

    @torch.no_grad()
    def _normalize(
        self,
        trajectories: Tensor,
        times: Tensor,
        mask: Tensor,
        delta_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, ODEConcepts.ODEConceptsBuilder]:
        """Normalize trajectories and times; return updated tensors and a concept builder."""
        concept = ODEConcepts.builder()

        states_norm_stats = self.spatial_norm.get_norm_stats(trajectories, mask)
        concept.states_norm(self.spatial_norm).states_norm_stats(states_norm_stats)
        trajectories = self.spatial_norm.normalization_map(trajectories, states_norm_stats)

        if self.model_config.times_norm_on_deltas:
            delta_times = times[:, :, 1:, :] - times[:, :, :-1, :]
            dt_stats = self.temporal_norm.get_norm_stats(delta_times, delta_mask)
            concept.times_norm(self.temporal_norm).times_norm_stats(dt_stats)
            times = self.temporal_norm.normalization_map(times, dt_stats)
        else:
            t_stats = self.temporal_norm.get_norm_stats(times, mask)
            concept.times_norm(self.temporal_norm).times_norm_stats(t_stats)
            times = self.temporal_norm.normalization_map(times, t_stats)

        return trajectories, times, concept

    @torch.no_grad()
    def _prepare_input_features(
        self, trajectories: Tensor, times: Tensor, mask: Tensor
    ) -> Tuple[TrajectoryFeatures, ODEConcepts.ODEConceptsBuilder]:
        """Standard 4-feature preprocessing (checkpoint-compatible)."""
        trajectories = self.pad_if_necessary(trajectories)
        trajectories, times, feature_mask = _backward_fill_and_feature_mask(
            trajectories, times, copy.deepcopy(mask)
        )
        trajectories, times, concept = self._normalize(
            trajectories, times, mask, feature_mask
        )
        features = _extract_features(trajectories, times, feature_mask)
        return features, concept

    @torch.no_grad()
    def _prepare_axial_input_features(
        self, trajectories: Tensor, times: Tensor, mask: Tensor
    ) -> Tuple[AxialTrajectoryFeatures, ODEConcepts.ODEConceptsBuilder]:
        """Axial 5-feature preprocessing (for AxialTrajectoryEncoder)."""
        trajectories = self.pad_if_necessary(trajectories)
        traj_bwd, times, feature_mask = _backward_fill_and_axial_feature_mask(
            trajectories, times, copy.deepcopy(mask)
        )
        traj_fwd = _forward_fill(trajectories * mask, mask)
        delta_mask = mask[:, :, :-1, :]
        traj_bwd, times, concept = self._normalize(traj_bwd, times, mask, delta_mask)
        traj_fwd = self.spatial_norm.normalization_map(traj_fwd, concept._states_norm_stats)
        features = _extract_axial_features(traj_bwd, traj_fwd, times, feature_mask)
        return features, concept

    # ---------- Encoder / decoder ----------

    def trajectory_encoding(
        self,
        trajectories: Tensor,
        times: Tensor,
        mask: Tensor,
    ) -> Tuple[EncoderOutput, Tensor, ODEConcepts.ODEConceptsBuilder]:
        """Encode context trajectories -> (D, feature_mask, concept_builder)."""
        encoder_type = getattr(self.model_config, "encoder_type", "standard")
        if encoder_type == "axial":
            features, concept = self._prepare_axial_input_features(trajectories, times, mask)
        else:
            features, concept = self._prepare_input_features(trajectories, times, mask)
        enc = self.trajectory_encoder(features)
        return EncoderOutput(D=enc.D), features.feature_mask, concept

    def function_decoding(
        self,
        locations: Tensor,
        feature_mask: Tensor,
        wrapped_D: EncoderOutput,
        concept: ODEConcepts.ODEConceptsBuilder,
    ) -> FIMODEOutput:
        """Decode: locations cross-attend trajectory embedding -> drift predictions."""
        normalized_locations = locations
        locations_enc = self.location_proj(locations)
        prediction = self.functional_decoder(
            locations_encoding=locations_enc,
            observations_encoding=wrapped_D.D,
            observations_padding_mask=~feature_mask,
        )
        concept.locations(normalized_locations).drift(prediction).normalized(True)
        return FIMODEOutput(
            predictions=concept.build(),
            D=copy.deepcopy(wrapped_D.detach()),
            encoded_locations=copy.deepcopy(locations_enc.detach()),
            feature_mask=feature_mask.detach(),
        )

    def model_forward(
        self,
        trajectories: Tensor,
        times: Tensor,
        locations: Tensor,
        mask: Tensor,
    ) -> FIMODEOutput:
        """Inference entry point: predict vector field at locations given context trajectories."""
        _sanity_check(trajectories.shape, times.shape, locations.shape, mask.shape)
        wrapped_D, feature_mask, concept = self.trajectory_encoding(trajectories, times, mask)
        locations = self.pad_if_necessary(locations)
        locations = self.spatial_norm.normalization_map(locations, concept._states_norm_stats)
        return self.function_decoding(locations, feature_mask, wrapped_D, concept)

    # ---------- AModel interface ----------

    def forward(self, data: dict, is_validation_batch: bool = False, **kwargs: Any) -> Dict[str, Any]:
        """Training entry point: batch dict -> loss stats dict."""
        return self._training_forward(data, is_validation_batch=is_validation_batch)

    def summary(self, x: dict):
        num_params = sum(p.numel() for p in self.parameters())
        trainable  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"FIMODE — total params: {num_params:,}  trainable: {trainable:,}"

    def loss(self, *inputs: Any) -> Dict[str, Any]:
        raise NotImplementedError("Use forward(data) for training; loss is inside the returned dict.")

    def metric(self, y: Any, y_target: Any) -> Dict[str, Any]:
        return {}

    def get_prediction_for_eval(self, model_output: FIMODEOutput) -> Tensor:
        model_output.predictions.renormalize()
        return model_output.predictions.drift.detach()

    # ---------- Training ----------

    def _training_forward(
        self,
        data: dict,
        is_validation_batch: bool = False,
    ) -> Dict[str, Any]:
        trajectories   = data["obs_values"]
        times          = data["obs_times"]
        locations      = data["locations"]
        mask           = data["obs_mask"].bool()
        dimension_mask = data["dimension_mask"].bool()
        fx             = data.get("drift_at_locations")
        fx_at_traj     = data.get("drift_at_observations")

        b, t, n, d = trajectories.shape

        if (self.training or is_validation_batch) and self.train_config.train_type == "vector_field":
            num_obs = t * n
            num_loc = locations.shape[1]
            idx = torch.round(torch.linspace(0, num_obs - 1, 2 * num_loc)).long()
            locations = torch.cat(
                [locations,
                 trajectories.flatten(start_dim=1, end_dim=2)[:, idx, :]],
                dim=1,
            )
            fx = torch.cat(
                [fx,
                 fx_at_traj.flatten(start_dim=1, end_dim=2)[:, idx, :]],
                dim=1,
            )
            dimension_mask = dimension_mask[:, :1, :].expand(fx.shape)

        self._relative_epoch += int(self.training and not self._is_training_manual)
        self._is_training_manual = self.training

        training_data = TrainingData(
            locations=locations,
            fx=fx,
            mask=mask,
            dimension_mask=dimension_mask,
            truth=TrainingData.Trajectory(trajectories=trajectories, times=times),
        )
        training_data = self.data_preparation.prepare_data(
            do_corruption=(self.training or is_validation_batch),
            data=training_data,
        )

        if self.train_config.train_type == "vector_field":
            loss, stats = self._vector_field_step(training_data)
        elif self.train_config.train_type == "trajectory_reconstruction":
            loss, stats = self._trajectory_reconstruction_step(training_data, is_validation_batch)
        elif self.train_config.train_type == "vf_plus_traj":
            vf_loss, vf_stats = self._vector_field_step(training_data)
            tr_loss, tr_stats = self._trajectory_reconstruction_step(training_data, is_validation_batch)
            loss  = vf_loss + tr_loss
            stats = {**vf_stats, **tr_stats,
                     "vector_field_loss": vf_loss.detach(),
                     "trajectory_loss":   tr_loss.detach()}
        else:
            raise ValueError(f"Unknown train_type: {self.train_config.train_type!r}")

        return self._assemble_stats(loss, stats)

    def _vector_field_step(
        self, data: TrainingData
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        b, t, n, d = data.to_train.trajectories.shape
        vf_cfg = self.train_config.vector_field_training or {}
        last_idx = t // 2 if vf_cfg.get("with_reconstruction_trajectories", False) else t

        result = self.model_forward(
            data.to_train.trajectories[:, :last_idx],
            data.to_train.times[:, :last_idx],
            data.locations,
            data.mask[:, :last_idx],
        )
        u = self.u_model(result)

        target = ODEConcepts(
            locations=data.locations,
            drift=data.fx,
            normalized=False,
            states_norm=self.spatial_norm,
            states_norm_stats=result.predictions.states_norm_stats,
            times_norm=self.temporal_norm,
            times_norm_stats=result.predictions.times_norm_stats,
        )

        if self.train_config.train_with_normalized_head:
            result.predictions.normalize()
            target.normalize()
        else:
            result.predictions.renormalize()
            target.renormalize()

        if self.train_config.loss_filter_nans:
            est = torch.nan_to_num(result.predictions.drift)
            tgt = torch.nan_to_num(target.drift)
        else:
            est = result.predictions.drift
            tgt = target.drift

        loss_at_locations = self._criterion(est, tgt, data.dimension_mask.expand_as(est))

        if self.train_config.use_uncertainty_weighting:
            loss = UncertaintyEstimator.apply_u_to_loss(loss_at_locations, u).mean()
        else:
            loss = loss_at_locations.mean()

        stats = {
            self.train_config.loss_type: loss_at_locations.detach().mean(),
            "uncertainty_estimate":      u.detach().mean(),
        }
        return loss, stats

    def _trajectory_reconstruction_step(
        self, data: TrainingData, is_validation_batch: bool = False
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        result = self.integrator.solve_ivp_for_random_initial_conditions(
            model=self,
            data=data,
            relative_epoch=self._relative_epoch,
            is_validation_batch=is_validation_batch,
        )
        pred   = result.prediction
        target = result.truth

        if self.train_config.loss_filter_nans:
            pred   = torch.nan_to_num(pred)
            target = torch.nan_to_num(target)

        b, t, n, d = data.truth.trajectories.shape
        num_steps = self.train_config.traj_loss_steps
        num_ic    = self.train_config.num_ic
        pred      = pred.view(b, t * num_ic * num_steps, d)
        target    = target.view(b, t * num_ic * num_steps, d)
        dim_mask  = DataPreparation.get_dim_mask_for_traj_training(
            data.to_train.trajectories, data.dimension_mask, self.train_config
        )

        if self.train_config.only_final_points_for_loss:
            pred     = pred[:, num_steps - 1 :: num_steps, :]
            target   = target[:, num_steps - 1 :: num_steps, :]
            dim_mask = dim_mask[:, num_steps - 1 :: num_steps, :]

        loss = self._criterion(pred, target, dim_mask).mean()
        return loss, {}

    def _assemble_stats(
        self, loss: Tensor, stats: Dict[str, Tensor]
    ) -> Dict[str, Dict[str, Tensor]]:
        stats["weight_norm_model"] = torch.norm(
            torch.stack([p.detach().norm(2) for p in self.parameters()]), 2
        )
        if self.train_config.train_type != "trajectory_reconstruction":
            stats["weight_norm_u"] = torch.norm(
                torch.stack([p.detach().norm(2) for p in self.u_model.parameters()]), 2
            )
        stats["loss"] = loss
        return {"losses": stats}


# ---------- HuggingFace registration ----------

ModelFactory.register(FIMODEConfig.model_type, FIMODE)
AutoConfig.register(FIMODEConfig.model_type, FIMODEConfig)
AutoModel.register(FIMODEConfig, FIMODE)


# ---------- Model loading helpers ----------

_HF_REPO_ID   = "FIM4Science/fim-ode"
_HF_SUBFOLDER = "base_model/checkpoints/best-model"
# Repo root is 4 levels up from src/fim/models/ode.py
_REPO_ROOT    = Path(__file__).resolve().parents[3]
_DEFAULT_HF_CACHE = _REPO_ROOT / "results" / "ode" / "pretrained"


def _load_fimode_from_config_and_weights(config_path: Path, weights_path: Path, device: str) -> FIMODE:
    from safetensors.torch import load_file
    from fim.models.ode_trainer import FIMODEConfig as _Cfg

    with open(config_path) as f:
        config_dict = json.load(f)
    config = _Cfg()
    config.model_config = config_dict["model_config"]
    config.train_config = config_dict["train_config"]

    model = FIMODE(config)
    state_dict = load_file(str(weights_path), device=device)
    state_dict = {k.removeprefix("model."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model.to(device)


def load_fim_ode_hf(device: str = "cpu", cache_dir: Path = None) -> FIMODE:
    """Download and load FIMODE from HuggingFace Hub (FIM4Science/fim-ode).

    Files are cached in ``cache_dir`` (default: ``results/ode/pretrained/``
    inside the repo root).  Both this function and the notebook use the same
    default, so the model is only downloaded once.
    """
    from huggingface_hub import hf_hub_download

    local_dir = Path(cache_dir) if cache_dir is not None else _DEFAULT_HF_CACHE
    local_dir.mkdir(parents=True, exist_ok=True)

    config_path  = Path(hf_hub_download(_HF_REPO_ID, f"{_HF_SUBFOLDER}/config.json",
                                        local_dir=str(local_dir)))
    weights_path = Path(hf_hub_download(_HF_REPO_ID, f"{_HF_SUBFOLDER}/model.safetensors",
                                        local_dir=str(local_dir)))
    return _load_fimode_from_config_and_weights(config_path, weights_path, device)


def load_fim_ode_local(checkpoint_dir: Path, device: str = "cpu") -> FIMODE:
    """Load FIMODE from a local safetensors checkpoint directory."""
    checkpoint_dir = Path(checkpoint_dir)
    return _load_fimode_from_config_and_weights(
        checkpoint_dir / "config.json",
        checkpoint_dir / "model.safetensors",
        device,
    )
