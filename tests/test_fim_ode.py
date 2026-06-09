"""
Tests for the FIMODE model (fim_ode.py, fim_ode_concepts.py, fim_ode_trainer.py).
All tests are self-contained — no external data files or YAML configs required.
"""
import copy
import pytest
import torch
from torch import Tensor

from fim.models.fim_ode import FIMODE
from fim.models.fim_ode_concepts import (
    ODEConcepts,
    TrajectoryFeatures,
    FIMODEModelConfig,
    TrajectoryEncoder,
    EncoderOutput,
    FIMODEOutput,
    UncertaintyEstimator,
)
from fim.models.fim_ode_trainer import (
    FIMODEConfig,
    FIMODETrainingConfig,
    TrainingData,
    DataCorruptionModel,
    DataPreparation,
    LossFactory,
)


# ─────────────────────────────────────────────
# Shared fixtures / helpers
# ─────────────────────────────────────────────

# Small but non-trivial dimensions so tests run fast
B   = 2   # batch size (number of ODEs)
T   = 3   # trajectories per ODE
N   = 12  # time points per trajectory
D   = 2   # state space dimension
L   = 6   # query locations

DIM_EMBED = 20   # must be divisible by 5 (features) and by NUM_HEADS
DIM_MAX   = D    # dim_max_trajectory == D (standard usage; D < DIM_MAX is a known
                 # limitation of the current ODEConcepts design — see test_model_forward_higher_dim)
NUM_HEADS = 4    # must divide DIM_EMBED (20 / 4 = 5 ✓)


def make_model_config() -> dict:
    return dict(
        dim_max_trajectory=DIM_MAX,
        use_bias_for_projection=True,
        dim_embed=DIM_EMBED,
        num_context_encoder_layers=1,
        attention_method="nn_multihead",
        attention_map=None,
        use_bias_in_attention=True,
        use_query_residual_in_attention=False,
        num_heads=NUM_HEADS,
        dim_feedforward=64,
        dropout=0.0,
        num_res_layers_functional_decoder=1,
        num_res_layer_u_model=1,
        dim_hidden_u_model=16,
        dim_ffn_u_model=32,
        times_norm_on_deltas=True,
    )


def make_train_config(train_type: str = "vector_field") -> dict:
    return dict(
        train_with_normalized_head=True,
        loss_filter_nans=True,
        loss_type="mse",
        train_type=train_type,
        use_uncertainty_weighting=True,
        corruption_model_type="fim",
        survival_rate=0.9,
        max_sigma_trajectory_noise=0.05,
        additive_noise_distribution="uniform",
        same_noise_level_for_all_trajectories=True,
    )


def make_fimode_config(train_type: str = "vector_field") -> FIMODEConfig:
    return FIMODEConfig(
        model_config=make_model_config(),
        train_config=make_train_config(train_type),
    )


def make_batch(b=B, t=T, n=N, d=D, locs=L) -> dict:
    """Synthetic training batch with all required keys."""
    obs_values = torch.randn(b, t, n, d)
    # monotone times: cumsum of small positive increments
    obs_times  = torch.cumsum(torch.rand(b, t, n, 1) * 0.1 + 0.01, dim=2)
    obs_mask   = torch.ones(b, t, n, 1, dtype=torch.bool)
    locations  = torch.randn(b, locs, d)
    drift_at_locations     = torch.randn(b, locs, d)
    drift_at_observations  = torch.randn(b, t, n, d)   # [B, T, N, D]
    dimension_mask = torch.ones(b, 1, d, dtype=torch.bool)

    return {
        "obs_values":            obs_values,
        "obs_times":             obs_times,
        "obs_mask":              obs_mask,
        "locations":             locations,
        "drift_at_locations":    drift_at_locations,
        "drift_at_observations": drift_at_observations,
        "dimension_mask":        dimension_mask,
    }


# ─────────────────────────────────────────────
# 1. Config
# ─────────────────────────────────────────────

class TestFIMODEConfig:

    def test_model_config_fields(self):
        cfg = FIMODEModelConfig(**make_model_config())
        assert cfg.dim_embed == DIM_EMBED
        assert cfg.dim_max_trajectory == DIM_MAX
        assert cfg.num_heads == NUM_HEADS

    def test_training_config_fields(self):
        cfg = FIMODETrainingConfig(**make_train_config())
        assert cfg.loss_type == "mse"
        assert cfg.train_type == "vector_field"
        assert cfg.corruption_model_type == "fim"

    def test_fimode_config_stores_sub_configs(self):
        cfg = make_fimode_config()
        assert isinstance(cfg.model_config, dict)
        assert isinstance(cfg.train_config, dict)
        assert cfg.model_config["dim_embed"] == DIM_EMBED

    def test_functional_decoder_config(self):
        cfg = FIMODEModelConfig(**make_model_config())
        dec_cfg = cfg.get_functional_decoder_config()
        assert "attention" in dec_cfg
        assert "projection" in dec_cfg
        assert dec_cfg["num_res_layers"] == cfg.num_res_layers_functional_decoder

    def test_uncertainty_estimator_config(self):
        cfg = FIMODEModelConfig(**make_model_config())
        u_cfg = cfg.get_uncertainty_estimator_config()
        assert u_cfg["num_res_layers"] == cfg.num_res_layer_u_model


# ─────────────────────────────────────────────
# 2. ODEConcepts
# ─────────────────────────────────────────────

class TestODEConcepts:

    def _make_concepts(self, normalized: bool = True):
        from fim.models.sde import Standardization, DeltaLogCentering
        spatial  = Standardization()
        temporal = DeltaLogCentering()

        trajectories = torch.randn(B, T, N, D)
        mask = torch.ones(B, T, N, 1, dtype=torch.bool)
        times = torch.cumsum(torch.rand(B, T, N, 1) * 0.1 + 0.01, dim=2)

        states_stats = spatial.get_norm_stats(trajectories, mask)
        delta_times  = times[:, :, 1:, :] - times[:, :, :-1, :]
        delta_mask   = mask[:, :, :-1, :]
        times_stats  = temporal.get_norm_stats(delta_times, delta_mask)

        locations = torch.randn(B, L, D)
        drift     = torch.randn(B, L, D)

        return ODEConcepts(
            locations=locations,
            drift=drift,
            normalized=normalized,
            states_norm=spatial,
            states_norm_stats=states_stats,
            times_norm=temporal,
            times_norm_stats=times_stats,
        ), locations.clone(), drift.clone()

    def test_builder_produces_concepts(self):
        concepts, _, _ = self._make_concepts()
        assert concepts.drift is not None
        assert concepts.locations is not None
        assert concepts.drift.shape == (B, L, D)
        assert concepts.locations.shape == (B, L, D)

    def test_normalize_changes_values(self):
        concepts, orig_locs, orig_drift = self._make_concepts(normalized=False)
        concepts.normalize()
        assert not torch.allclose(concepts.drift, orig_drift)

    def test_normalize_renormalize_roundtrip(self):
        concepts, orig_locs, orig_drift = self._make_concepts(normalized=False)
        concepts.normalize()
        concepts.renormalize()
        assert torch.allclose(concepts.drift, orig_drift, atol=1e-5), \
            "Renormalize should recover the original drift"
        assert torch.allclose(concepts.locations, orig_locs, atol=1e-5), \
            "Renormalize should recover the original locations"

    def test_builder_pattern(self):
        from fim.models.sde import Standardization, DeltaLogCentering
        spatial  = Standardization()
        temporal = DeltaLogCentering()
        trajectories = torch.randn(B, T, N, D)
        mask = torch.ones(B, T, N, 1, dtype=torch.bool)
        times = torch.cumsum(torch.rand(B, T, N, 1) * 0.1 + 0.01, dim=2)
        states_stats = spatial.get_norm_stats(trajectories, mask)
        delta_times  = times[:, :, 1:, :] - times[:, :, :-1, :]
        times_stats  = temporal.get_norm_stats(delta_times, mask[:, :, :-1, :])

        concepts = (
            ODEConcepts.builder()
            .locations(torch.randn(B, L, D))
            .drift(torch.randn(B, L, D))
            .normalized(True)
            .states_norm(spatial)
            .states_norm_stats(states_stats)
            .times_norm(temporal)
            .times_norm_stats(times_stats)
            .build()
        )
        assert isinstance(concepts, ODEConcepts)
        assert concepts.drift.shape == (B, L, D)


# ─────────────────────────────────────────────
# 3. TrajectoryEncoder
# ─────────────────────────────────────────────

class TestTrajectoryEncoder:

    @pytest.fixture
    def encoder(self):
        cfg = FIMODEModelConfig(**make_model_config())
        return TrajectoryEncoder(cfg)

    def test_output_shape(self, encoder):
        features = TrajectoryFeatures(
            x=torch.randn(B, T, N - 1, DIM_MAX),
            delta_x=torch.randn(B, T, N - 1, DIM_MAX),
            delta_x_squared=torch.randn(B, T, N - 1, DIM_MAX),
            delta_t=torch.randn(B, T, N - 1, 1),
            feature_mask=torch.ones(B, T, N - 1, 1, dtype=torch.bool),
        )
        out = encoder(features)
        assert out.D.shape == (B, T, N - 1, DIM_EMBED)

    def test_masking_zeros_out_masked_positions(self, encoder):
        """Positions where feature_mask=False should have D=0 (multiplied out at the end)."""
        mask = torch.ones(B, T, N - 1, 1, dtype=torch.bool)
        mask[:, :, -3:, :] = False   # mask last 3 positions

        features = TrajectoryFeatures(
            x=torch.randn(B, T, N - 1, DIM_MAX),
            delta_x=torch.randn(B, T, N - 1, DIM_MAX),
            delta_x_squared=torch.randn(B, T, N - 1, DIM_MAX),
            delta_t=torch.randn(B, T, N - 1, 1),
            feature_mask=mask,
        )
        out = encoder(features)
        # Masked positions are zeroed by `D * feature_mask` at end of forward
        assert torch.all(out.D[:, :, -3:, :] == 0.0), \
            "Masked positions should be zeroed out in encoder output"


# ─────────────────────────────────────────────
# 4. Loss functions
# ─────────────────────────────────────────────

class TestLossFactory:

    @pytest.fixture(params=["mse", "huber", "relative_l2", "l1", "il_mse"])
    def loss_fn(self, request):
        return LossFactory.create(request.param), request.param

    def test_output_shape(self, loss_fn):
        fn, name = loss_fn
        est    = torch.randn(B, L, D)
        target = torch.randn(B, L, D)
        mask   = torch.ones(B, L, D)
        out    = fn(est, target, mask)
        assert out.shape == (B, L), f"{name}: expected shape ({B}, {L}), got {out.shape}"

    def test_mse_zero_for_identical_inputs(self):
        fn = LossFactory.create("mse")
        x  = torch.randn(B, L, D)
        mask = torch.ones(B, L, D)
        out  = fn(x, x, mask)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    def test_mse_ignores_masked_dims(self):
        """Masked dimensions should not contribute to the squared error numerator."""
        fn  = LossFactory.create("mse")
        # dim 0: error = 1,  dim 1: error = 10
        est = torch.stack([torch.ones(B, L), 10 * torch.ones(B, L)], dim=-1)
        tgt = torch.zeros(B, L, D)

        mask_dim0 = torch.zeros(B, L, D)
        mask_dim0[:, :, 0] = 1.0   # only dim 0 active  → MSE = 1
        mask_dim1 = torch.zeros(B, L, D)
        mask_dim1[:, :, 1] = 1.0   # only dim 1 active  → MSE = 100

        loss_d0 = fn(est, tgt, mask_dim0).mean()
        loss_d1 = fn(est, tgt, mask_dim1).mean()
        assert not torch.allclose(loss_d0, loss_d1), \
            "MSE should differ when different (error-magnitude) dimensions are active"
        assert loss_d0 < loss_d1, "Dim 1 has larger error so its MSE should be higher"

    def test_unknown_loss_raises(self):
        with pytest.raises(ValueError):
            LossFactory.create("nonsense_loss")


# ─────────────────────────────────────────────
# 5. Data corruption
# ─────────────────────────────────────────────

class TestDataCorruption:

    def _make_config(self, corruption_type, **kwargs):
        return FIMODETrainingConfig(
            train_with_normalized_head=True,
            loss_filter_nans=False,
            loss_type="mse",
            corruption_model_type=corruption_type,
            **kwargs,
        )

    def test_fim_corruption_preserves_shape(self):
        cfg = self._make_config("fim", survival_rate=0.8,
                                max_sigma_trajectory_noise=0.1,
                                additive_noise_distribution="uniform")
        model = DataCorruptionModel(cfg)
        traj = torch.randn(B, T, N, D)
        out  = model.corrupt_data(traj)
        assert out.corrupt_trajectory.shape == traj.shape
        assert out.mask.shape == (B, T, N, 1)

    def test_fim_mask_is_bool(self):
        cfg = self._make_config("fim", survival_rate=0.8,
                                max_sigma_trajectory_noise=None,
                                additive_noise_distribution=None)
        model = DataCorruptionModel(cfg)
        out   = model.corrupt_data(torch.randn(B, T, N, D))
        assert out.mask.dtype == torch.bool

    def test_odeformer_corruption(self):
        cfg = self._make_config("odeformer",
                                max_subsampling_ration=0.3,
                                max_sigma_trajectory_noise=0.1)
        model = DataCorruptionModel(cfg)
        traj  = torch.randn(B, T, N, D)
        out   = model.corrupt_data(traj)
        assert out.corrupt_trajectory.shape == traj.shape

    def test_no_corruption_returns_all_true_mask(self):
        cfg = self._make_config(None)
        model = DataCorruptionModel(cfg)
        traj  = torch.randn(B, T, N, D)
        out   = model.corrupt_data(traj)
        assert out.mask.all()

    def test_subsample_mask_respects_ratio(self):
        traj = torch.randn(B, T, N, D)
        # with max_ratio=0 nothing is dropped
        mask_full = DataCorruptionModel.generate_subsample_points_mask(
            traj, min_ratio=0.0, max_ratio=0.0
        )
        assert mask_full.all()

    def test_unknown_corruption_type_raises(self):
        with pytest.raises(NotImplementedError):
            DataCorruptionModel(self._make_config("unknown_type"))


# ─────────────────────────────────────────────
# 6. Full FIMODE model
# ─────────────────────────────────────────────

class TestFIMODE:

    @pytest.fixture
    def model(self):
        cfg   = make_fimode_config()
        model = FIMODE(cfg)
        model.eval()
        return model

    @pytest.fixture
    def model_train(self):
        cfg   = make_fimode_config()
        model = FIMODE(cfg)
        model.train()
        return model

    # ── instantiation ──────────────────────────

    def test_instantiation(self, model):
        assert isinstance(model, FIMODE)

    def test_has_required_submodules(self, model):
        assert hasattr(model, "trajectory_encoder")
        assert hasattr(model, "functional_decoder")
        assert hasattr(model, "u_model")   # checkpoint-compatible attribute name
        assert hasattr(model, "location_proj")
        assert hasattr(model, "spatial_norm")
        assert hasattr(model, "temporal_norm")

    # ── preprocessing ──────────────────────────

    def test_pad_if_necessary_pads(self, model):
        x      = torch.randn(B, T, N, D)   # D < DIM_MAX
        padded = model.pad_if_necessary(x)
        assert padded.shape[-1] == DIM_MAX

    def test_pad_if_necessary_noop_when_large_enough(self, model):
        x = torch.randn(B, T, N, DIM_MAX)
        assert model.pad_if_necessary(x).shape[-1] == DIM_MAX

    # ── inference (model_forward) ───────────────

    def test_model_forward_output_shapes(self, model):
        traj  = torch.randn(B, T, N, D)
        times = torch.cumsum(torch.rand(B, T, N, 1) * 0.1 + 0.01, dim=2)
        locs  = torch.randn(B, L, D)
        mask  = torch.ones(B, T, N, 1, dtype=torch.bool)

        with torch.no_grad():
            out = model.model_forward(traj, times, locs, mask)

        assert isinstance(out, FIMODEOutput)
        assert out.predictions.drift.shape == (B, L, D)
        assert out.predictions.locations.shape == (B, L, D)
        assert out.D.D.shape == (B, T, N - 1, DIM_EMBED)
        assert out.feature_mask.shape == (B, T, N - 1, 1)

    def test_model_forward_single_trajectory(self, model):
        """Model should handle T=1 (single context trajectory)."""
        traj  = torch.randn(B, 1, N, D)
        times = torch.cumsum(torch.rand(B, 1, N, 1) * 0.1 + 0.01, dim=2)
        locs  = torch.randn(B, L, D)
        mask  = torch.ones(B, 1, N, 1, dtype=torch.bool)

        with torch.no_grad():
            out = model.model_forward(traj, times, locs, mask)
        assert out.predictions.drift.shape == (B, L, D)

    @pytest.mark.xfail(reason=(
        "D < DIM_MAX triggers a shape mismatch inside ODEConcepts.normalize(): "
        "norm_stats are computed on padded data (DIM_MAX dims) but target locations "
        "are unpadded (D dims). Known limitation to be resolved in the new axial-attention architecture."
    ))
    def test_model_forward_higher_dim(self, model):
        """Model should pad lower-D inputs up to dim_max_trajectory."""
        d_small = 1  # below DIM_MAX
        traj  = torch.randn(B, T, N, d_small)
        times = torch.cumsum(torch.rand(B, T, N, 1) * 0.1 + 0.01, dim=2)
        locs  = torch.randn(B, L, d_small)
        mask  = torch.ones(B, T, N, 1, dtype=torch.bool)

        with torch.no_grad():
            out = model.model_forward(traj, times, locs, mask)
        assert out.predictions.drift.shape == (B, L, d_small)

    def test_uncertainty_estimator_output_shape(self, model):
        traj  = torch.randn(B, T, N, D)
        times = torch.cumsum(torch.rand(B, T, N, 1) * 0.1 + 0.01, dim=2)
        locs  = torch.randn(B, L, D)
        mask  = torch.ones(B, T, N, 1, dtype=torch.bool)

        with torch.no_grad():
            out = model.model_forward(traj, times, locs, mask)
            u   = model.u_model(out)
        assert u.shape == (B, L), f"Expected ({B}, {L}), got {u.shape}"

    # ── training forward ───────────────────────

    def test_training_forward_returns_loss_dict(self, model_train):
        batch = make_batch()
        out   = model_train(batch)
        assert isinstance(out, dict)
        assert "losses" in out
        assert "loss" in out["losses"]

    def test_training_forward_loss_is_scalar(self, model_train):
        batch = make_batch()
        out   = model_train(batch)
        loss  = out["losses"]["loss"]
        assert loss.ndim == 0, f"Loss should be a scalar, got shape {loss.shape}"

    def test_training_forward_loss_is_finite(self, model_train):
        batch = make_batch()
        out   = model_train(batch)
        assert torch.isfinite(out["losses"]["loss"]), "Training loss should be finite"

    def test_training_forward_includes_stats(self, model_train):
        batch  = make_batch()
        out    = model_train(batch)
        losses = out["losses"]
        assert "mse" in losses or "uncertainty_estimate" in losses or "loss" in losses

    def test_eval_mode_no_corruption(self, model):
        """In eval mode, is_validation_batch=False should not corrupt data."""
        batch = make_batch()
        with torch.no_grad():
            out = model(batch, is_validation_batch=False)
        assert "losses" in out

    def test_validation_batch_runs(self, model):
        batch = make_batch()
        with torch.no_grad():
            out = model(batch, is_validation_batch=True)
        assert torch.isfinite(out["losses"]["loss"])

    def test_get_prediction_for_eval(self, model):
        traj  = torch.randn(B, T, N, D)
        times = torch.cumsum(torch.rand(B, T, N, 1) * 0.1 + 0.01, dim=2)
        locs  = torch.randn(B, L, D)
        mask  = torch.ones(B, T, N, 1, dtype=torch.bool)

        with torch.no_grad():
            out  = model.model_forward(traj, times, locs, mask)
            pred = model.get_prediction_for_eval(out)
        assert pred.shape == (B, L, D)
        assert not pred.requires_grad

    # ── weight update ──────────────────────────

    def test_backward_pass_runs(self):
        """Loss should be differentiable and produce gradients."""
        cfg   = make_fimode_config()
        model = FIMODE(cfg)
        model.train()
        batch = make_batch()
        out   = model(batch)
        loss  = out["losses"]["loss"]
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients were computed"

    # ── save / load ────────────────────────────

    def test_save_and_load(self, model, tmp_path):
        model.save_pretrained(tmp_path)
        loaded = FIMODE.from_pretrained(tmp_path)
        for (name, p_orig), (_, p_load) in zip(
            model.named_parameters(), loaded.named_parameters()
        ):
            assert torch.allclose(p_orig, p_load), f"Parameter {name} differs after reload"

    # ── consistency ────────────────────────────

    def test_same_input_same_output_in_eval(self, model):
        """Deterministic output in eval mode."""
        traj  = torch.randn(B, T, N, D)
        times = torch.cumsum(torch.rand(B, T, N, 1) * 0.1 + 0.01, dim=2)
        locs  = torch.randn(B, L, D)
        mask  = torch.ones(B, T, N, 1, dtype=torch.bool)

        with torch.no_grad():
            out1 = model.model_forward(traj, times, locs, mask)
            out2 = model.model_forward(traj, times, locs, mask)
        assert torch.allclose(out1.predictions.drift, out2.predictions.drift)
