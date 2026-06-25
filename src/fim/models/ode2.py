"""
FIMODE2 — Extended Foundation Inference Model for ODEs (paper 2).

This module contains the axial-attention trajectory encoder and a new model
class designed to work with the standard FIM Trainer (scripts/train_model.py),
as opposed to the custom ode_trainer.py.

"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Final, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoConfig, AutoModel, PretrainedConfig

from fim.models.blocks import AModel, ModelFactory
from fim.models.blocks.neural_operators import ResidualEncoderLayer, AttentionOperator
from fim.models.sde import Standardization, DeltaLogCentering
from fim.models.ode import (
    FIMODEModelConfig,
    UncertaintyEstimator,
    _backward_fill,
    _sanity_check,
)


# ─────────────────────────────────────────────────────────────────────────────
# 5-feature trajectory representation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AxialTrajectoryFeatures:
    """
    5-feature representation at interior positions 1..N-2.

    Backward difference (delta_x_back) encodes local curvature without
    explicit second-order time derivatives.
    """
    x: Tensor               # state values at interior positions  [B, T, N-2, D]
    delta_x: Tensor         # forward increment:  x(t+h) - x(t)
    delta_x_back: Tensor    # backward increment: x(t) - x(t-h)
    delta_x_squared: Tensor # squared forward increment
    delta_t: Tensor         # time increments                     [B, T, N-2, 1]
    feature_mask: Tensor    # valid-step mask                     [B, T, N-2, 1]


# ─────────────────────────────────────────────────────────────────────────────
# Axial trajectory encoder
# ─────────────────────────────────────────────────────────────────────────────

class AxialTrajectoryEncoder(nn.Module):
    """
    5-feature context encoder: x, Δx, Δx_back, Δx², Δt.

    Requires dim_embed divisible by 5.
    Not compatible with FIMODE (ode.py) checkpoints.
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


# ─────────────────────────────────────────────────────────────────────────────
# Axial preprocessing helpers
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


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

class FIMODE2Config(PretrainedConfig):
    """
    HuggingFace config for FIMODE2.

    Holds only the model architecture as a nested dict (model_config).
    Training hyperparameters (lr, loss type, etc.) are handled by the
    standard FIM Trainer via the YAML config, not by this class.
    """
    model_type = "FIMODE2"

    def __init__(self, model_config: Optional[dict] = None, **kwargs):
        super().__init__(**kwargs)
        self.model_config = model_config or {}


# ─────────────────────────────────────────────────────────────────────────────
# FIMODE2 — standard-trainer model with axial encoder
# ─────────────────────────────────────────────────────────────────────────────

class FIMODE2(AModel):
    """
    FIMODE with a 5-feature axial trajectory encoder, paper 2.

    Uses the standard FIM Trainer (scripts/train_model.py) rather than the
    custom Euler/Heun loop in scripts/ode/finetune.py.

    Architecture is identical to FIMODE except for the trajectory encoder:
      - AxialTrajectoryEncoder (5 features, interior positions 1..N-2)
      - dim_embed must be divisible by 5 (vs 4 for FIMODE)

    The training forward pass (forward method) is a placeholder until the
    paper-2 training objective is defined.
    """

    config_class = FIMODE2Config

    def __init__(
        self,
        config: FIMODE2Config,
        device_map: Optional[torch.device] = None,
        **kwargs: Any,
    ):
        AModel.__init__(self, config, **kwargs)

        model_config = FIMODEModelConfig(**config.model_config)
        self.model_config = model_config

        assert model_config.dim_embed % 5 == 0, (
            "dim_embed must be divisible by 5 for FIMODE2 / AxialTrajectoryEncoder"
        )
        self.trajectory_encoder = AxialTrajectoryEncoder(model_config)

        self.spatial_norm  = Standardization()
        self.temporal_norm = DeltaLogCentering()

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
        self.u_model = UncertaintyEstimator(model_config)

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
    def _prepare_axial_input_features(
        self, trajectories: Tensor, times: Tensor, mask: Tensor
    ) -> Tuple[AxialTrajectoryFeatures, Any, Any]:
        """
        Axial 5-feature preprocessing: backward+forward fill, normalize, extract features.
        Returns (features, traj_norm_stats, time_norm_stats).
        """
        trajectories = self.pad_if_necessary(trajectories)
        traj_bwd, times_filled, feature_mask = _backward_fill_and_axial_feature_mask(
            trajectories, times, copy.deepcopy(mask)
        )
        traj_fwd = _forward_fill(trajectories * mask, mask)

        traj_norm_stats = self.spatial_norm.get_norm_stats(trajectories, mask)
        traj_bwd = self.spatial_norm.normalization_map(traj_bwd, traj_norm_stats)
        traj_fwd = self.spatial_norm.normalization_map(traj_fwd, traj_norm_stats)

        delta_times     = times[:, :, 1:, :] - times[:, :, :-1, :]
        delta_mask      = mask[:, :, :-1, :]
        time_norm_stats = self.temporal_norm.get_norm_stats(delta_times, delta_mask)
        times_filled    = self.temporal_norm.normalization_map(times_filled, time_norm_stats)

        features = _extract_axial_features(traj_bwd, traj_fwd, times_filled, feature_mask)
        return features, traj_norm_stats, time_norm_stats

    # ---------- Inference ----------

    def model_forward(
        self,
        trajectories: Tensor,
        times: Tensor,
        locations: Tensor,
        mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Predict vector field at locations given context trajectories.
        Returns (prediction [B, L, D], feature_mask [B, T, N-2, 1]).
        """
        _sanity_check(trajectories.shape, times.shape, locations.shape, mask.shape)
        features, traj_norm_stats, _ = self._prepare_axial_input_features(
            trajectories, times, mask
        )
        enc = self.trajectory_encoder(features)

        locations     = self.pad_if_necessary(locations)
        locations     = self.spatial_norm.normalization_map(locations, traj_norm_stats)
        locations_enc = self.location_proj(locations)
        prediction    = self.functional_decoder(
            locations_encoding=locations_enc,
            observations_encoding=enc.D,
            observations_padding_mask=~features.feature_mask,
        )
        return prediction, features.feature_mask

    # ---------- AModel interface  ----------

    def forward(
        self,
        data: dict,
        schedulers: Optional[dict] = None,
        step: int = 0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Standard Trainer entry point: batch dict → {"losses": {"loss": tensor, ...}}.

        Not yet implemented — to be defined when the paper-2 training objective
        is finalised. The standard FIM Trainer (scripts/train_model.py) calls this.
        """
        raise NotImplementedError(
            "FIMODE2.forward() is not yet implemented. "
            "Define the paper-2 training objective here."
        )

    def loss(self, *inputs: Any) -> Dict[str, Any]:
        raise NotImplementedError

    def metric(self, y: Any, y_target: Any) -> Dict[str, Any]:
        return {}

    def summary(self, x: dict) -> str:
        n = sum(p.numel() for p in self.parameters())
        t = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"FIMODE2 — total params: {n:,}  trainable: {t:,}"


# ---------- HuggingFace registration ----------

ModelFactory.register(FIMODE2Config.model_type, FIMODE2)
AutoConfig.register(FIMODE2Config.model_type, FIMODE2Config)
AutoModel.register(FIMODE2Config, FIMODE2)
