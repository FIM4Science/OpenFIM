"""
Output containers, configs, and training infrastructure for FIMODE.

ODEConcepts, EncoderOutput, and FIMODEOutput live here (rather than in ode.py)
so that TrainIntegrator can reference them without a circular import.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from transformers import PretrainedConfig

from fim.models.sde import InstanceNormalization


if TYPE_CHECKING:
    from fim.models.ode import FIMODE


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
        grad = self.states_norm.normalization_map(self.locations, self.states_norm_stats, derivative_num=1)
        self.drift = self.drift * grad
        dummy_t = torch.zeros_like(self.locations[..., 0:1])
        t_inv_grad = self.times_norm.inverse_normalization_map(dummy_t, self.times_norm_stats, derivative_num=1)
        self.drift = self.drift * t_inv_grad
        self.locations = self.states_norm.normalization_map(self.locations, self.states_norm_stats)
        self.normalized = True

    def renormalize(self) -> None:
        """Transform drift from normalized back to physical coordinates."""
        if not self.normalized:
            return
        inv_grad = self.states_norm.inverse_normalization_map(self.locations, self.states_norm_stats, derivative_num=1)
        self.drift = self.drift * inv_grad
        dummy_t = torch.zeros_like(self.locations[..., 0:1])
        t_grad = self.times_norm.normalization_map(dummy_t, self.times_norm_stats, derivative_num=1)
        self.drift = self.drift * t_grad
        self.locations = self.states_norm.inverse_normalization_map(self.locations, self.states_norm_stats)
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


# ---------- Configs ----------


@dataclass(kw_only=True)
class FIMODETrainingConfig:
    """All training hyperparameters for FIMODE."""

    train_with_normalized_head: bool
    loss_filter_nans: bool
    loss_type: Literal["mse", "huber", "relative_l2", "l1", "huber_mean_dim", "il_mse", "cos_sim"]
    train_type: Literal["vector_field", "trajectory_reconstruction", "vf_plus_traj"] = "vector_field"
    use_uncertainty_weighting: bool = True

    # ODE integration settings (for trajectory_reconstruction training):
    integrator_for_trajectory_training: Literal["euler", "improved_euler", "rk4"] = "improved_euler"
    relative_integration_horizon: Optional[float] = None
    traj_loss_steps: Optional[int] = None
    random_initial_conditions: bool = False
    num_ic: int = 1
    intermediate_steps_per_step: int = 1
    only_final_points_for_loss: bool = False
    use_h_max: bool = False
    h_max: float = 0.1

    # Stochastic regularization:
    step_noise_scale: float = 0.0
    ic_noise_scale: float = 0.0

    # Vector field training options:
    vector_field_training: Optional[dict] = None

    # Data corruption settings:
    corruption_model_type: Literal["flexible", "fim", "odeformer", "eval_odeformer", "fim_round_2"]
    survival_rate: Optional[float] = None
    max_subsampling_ration: Optional[float] = None
    same_noise_level_for_all_trajectories: bool = True
    additive_noise_distribution: Optional[Literal["uniform", "exponential"]] = None
    mean_sigma_trajectory_noise: Optional[float] = None
    max_sigma_trajectory_noise: Optional[float] = None
    num_trajs: Optional[dict] = None
    use_generalization_trajs: bool = False
    corruption_model: Optional[dict] = None
    num_nearest_points: Optional[int] = None


class FIMODEConfig(PretrainedConfig):
    """
    Top-level HuggingFace config for FIMODE.
    Holds model architecture and training hyperparameters as nested dicts.
    """

    model_type = "FIMODE"

    def __init__(self, model_config: Optional[dict] = None, train_config: Optional[dict] = None, **kwargs):
        super().__init__(**kwargs)
        self.model_config = model_config
        self.train_config = train_config


# ---------- Training data ----------


@dataclass(kw_only=True)
class TrainingData:
    """
    Everything needed for a single training step.

    Shapes:
        b = batch size (number of ODEs)
        t = number of trajectories per ODE
        n = number of time points per trajectory
        d = state dimension
    """

    @dataclass(kw_only=True)
    class Trajectory:
        trajectories: Tensor  # [B, T, N, D]
        times: Tensor  # [B, T, N, 1]

    locations: Tensor  # [B, L, D]
    fx: Tensor  # [B, L, D]
    mask: Tensor  # [B, T, N, 1]
    dimension_mask: Tensor  # [B, L, D]

    truth: Trajectory
    to_train: Optional[Trajectory] = None


# ---------- Data corruption ----------


class DataCorruptionModel:
    """
    Applies noise and subsampling to trajectories before training.
    Corruption is applied BEFORE normalization.
    """

    def __init__(self, config: FIMODETrainingConfig):
        self.config = config
        _dispatch = {
            "odeformer": self._odeformer,
            "eval_odeformer": self._eval_odeformer,
            "fim": self._fim,
            "flexible": self._flexible,
            None: self._no_corruption,
        }
        fn = _dispatch.get(config.corruption_model_type)
        if fn is None:
            raise NotImplementedError(f"Unknown corruption_model_type: {config.corruption_model_type!r}")
        self.corrupt_data = fn

    @dataclass
    class CorruptData:
        corrupt_trajectory: Tensor
        mask: Tensor

    def _no_corruption(self, trajectory: Tensor) -> "DataCorruptionModel.CorruptData":
        mask = torch.ones((*trajectory.shape[:-1], 1), dtype=torch.bool, device=trajectory.device)
        return DataCorruptionModel.CorruptData(corrupt_trajectory=trajectory, mask=mask)

    def _flexible(self, trajectory: Tensor) -> "DataCorruptionModel.CorruptData":
        subsampling = self.config.corruption_model["subsampling"]
        additive_noise = self.config.corruption_model["additive_noise"]
        multiplicative_noise = self.config.corruption_model["multiplicative_noise"]

        if subsampling["procedure"] == "regular_and_uniformly_random_without_replacement":
            cfg = subsampling["regular_and_uniformly_random_without_replacement"]
            mask = DataCorruptionModel.generate_subsample_points_mask(
                trajectory,
                p_random=cfg["p_random"],
                min_ratio=cfg["min_rho"],
                max_ratio=cfg["max_rho"],
            )
        if multiplicative_noise is not None:
            trajectory = DataCorruptionModel.add_multiplicative_noise(trajectory, **multiplicative_noise)
        if additive_noise is not None:
            trajectory = DataCorruptionModel.add_relative_gaussian_noise(trajectory, **additive_noise)
        return DataCorruptionModel.CorruptData(corrupt_trajectory=trajectory, mask=mask)

    def _odeformer(self, trajectory: Tensor) -> "DataCorruptionModel.CorruptData":
        if self.config.max_sigma_trajectory_noise is not None:
            trajectory = DataCorruptionModel.add_multiplicative_noise(
                trajectory,
                sigma_distribution="uniform",
                min_sigma=0.0,
                max_sigma=self.config.max_sigma_trajectory_noise,
            )
        if self.config.max_subsampling_ration is not None:
            mask = DataCorruptionModel.generate_subsample_points_mask(
                trajectory, min_ratio=0.0, max_ratio=self.config.max_subsampling_ration
            )
        else:
            mask = torch.ones((*trajectory.shape[:-1], 1), dtype=torch.bool)
        return DataCorruptionModel.CorruptData(corrupt_trajectory=trajectory, mask=mask)

    def _eval_odeformer(self, trajectory: Tensor) -> "DataCorruptionModel.CorruptData":
        if self.config.max_sigma_trajectory_noise is not None:
            s = self.config.max_sigma_trajectory_noise
            trajectory = DataCorruptionModel.add_multiplicative_noise(trajectory, sigma_distribution="uniform", min_sigma=s, max_sigma=s)
        if self.config.max_subsampling_ration is not None:
            r = self.config.max_subsampling_ration
            mask = DataCorruptionModel.generate_subsample_points_mask(trajectory, min_ratio=r, max_ratio=r)
        else:
            mask = torch.ones((*trajectory.shape[:-1], 1), dtype=torch.bool)
        return DataCorruptionModel.CorruptData(corrupt_trajectory=trajectory, mask=mask)

    def _fim(self, trajectory: Tensor) -> "DataCorruptionModel.CorruptData":
        mask_shape = (*trajectory.shape[:-1], 1)
        if self.config.survival_rate is not None:
            mask = torch.rand(mask_shape, device=trajectory.device) < self.config.survival_rate
        else:
            mask = torch.ones(mask_shape, dtype=torch.bool)
        if self.config.max_sigma_trajectory_noise is not None:
            trajectory = DataCorruptionModel.add_relative_gaussian_noise(
                trajectory,
                sigma_distribution=self.config.additive_noise_distribution,
                max_sigma=self.config.max_sigma_trajectory_noise,
            )
        return DataCorruptionModel.CorruptData(corrupt_trajectory=trajectory, mask=mask)

    @staticmethod
    def generate_subsample_points_mask(
        tensor: Tensor,
        p_random: float = 1.0,
        min_ratio: float = 0.0,
        max_ratio: float = 0.5,
    ) -> Tensor:
        """
        Boolean mask subsampling trajectory points.
        Same subsampling ratio for all trajectories within an ODE (batch element).
        With probability p_random uses random subsampling; otherwise uses a fixed grid.
        """
        b, t, n, d = tensor.shape
        device = tensor.device

        sampling_ratios = torch.empty(b, 1, 1, 1, device=device).uniform_(min_ratio, max_ratio).expand(-1, t, -1, -1)
        num_keep = torch.floor((1.0 - sampling_ratios) * n).long().clamp(min=1, max=n)

        use_random = (torch.rand(b, 1, 1, 1, device=device) < p_random).expand(b, t, n, 1)

        # Random mode
        random_scores = torch.rand(b, t, n, 1, device=device)
        thresholds = torch.gather(
            random_scores.sort(dim=2)[0],
            dim=2,
            index=(n - num_keep).expand(b, t, 1, 1),
        )
        mask_random = random_scores >= thresholds

        # Fixed-step grid mode
        mask_grid = torch.zeros(b, t, n, 1, dtype=torch.bool, device=device)
        for bi in range(b):
            k = int(num_keep[bi, 0, 0, 0].item())
            if k >= n:
                idx = torch.arange(n, device=device)
            elif k <= 1:
                idx = torch.tensor([0], device=device)
            else:
                idx = torch.unique(torch.linspace(0, n - 1, steps=k, device=device).round().long())
            mask_grid[bi, :, idx, 0] = True

        return (use_random & mask_random) | (~use_random & mask_grid)

    @staticmethod
    def add_multiplicative_noise(data: Tensor, **kwargs) -> Tensor:
        b = data.shape[0]
        sigma_distribution = kwargs.get("sigma_distribution")
        if sigma_distribution == "uniform":
            min_sigma = kwargs.get("min_sigma", 0.0)
            max_sigma = kwargs.get("max_sigma", 0.0)
            sigma = torch.empty((b, 1, 1, 1), device=data.device, dtype=data.dtype).uniform_(min_sigma, max_sigma)
        else:
            raise ValueError(f"Unknown sigma distribution: {sigma_distribution}")
        return (1 + sigma * torch.randn_like(data)) * data

    @staticmethod
    def add_relative_gaussian_noise(
        x: Tensor,
        sigma_distribution: Literal["uniform", "exponential"],
        mean_sigma: float = 0.1,
        max_sigma: float = 0.1,
        same_noise_level_for_all_trajectories: bool = True,
    ) -> Tensor:
        b, t, n, d = x.shape
        device = x.device

        x_flat = x.view(b, -1, d)
        range_val = 0.5 * (x_flat.max(dim=1)[0] - x_flat.min(dim=1)[0])

        if sigma_distribution == "uniform":
            if same_noise_level_for_all_trajectories:
                sigma = torch.rand(b, 1, 1, 1, device=device) * max_sigma
                sigma = sigma.expand(b, t, 1, 1)
            else:
                sigma = torch.rand(b, t, 1, 1, device=device) * max_sigma
        elif sigma_distribution == "exponential":
            dist = torch.distributions.Exponential(rate=1.0 / mean_sigma)
            if same_noise_level_for_all_trajectories:
                sigma = dist.sample((b, 1, 1, 1)).to(device).expand(b, t, 1, 1)
            else:
                sigma = dist.sample((b, t, 1, 1)).to(device)
        else:
            raise ValueError(f"Unknown sigma distribution: {sigma_distribution}")

        std_dev = sigma * range_val.unsqueeze(1).unsqueeze(1)
        return x + torch.randn_like(x) * std_dev


# ---------- Data preparation ----------


class DataPreparation:
    """Applies DataCorruptionModel to produce corrupted training trajectories."""

    def __init__(self, config: FIMODETrainingConfig):
        self.config = config
        self.data_corruption = DataCorruptionModel(config)

    def prepare_data(self, do_corruption: bool, data: TrainingData) -> TrainingData:
        trajectories, times = data.truth.trajectories, data.truth.times

        if do_corruption:
            if self.config.num_nearest_points is not None:
                nnp = self.config.num_nearest_points
                batch_indices, nearest_indices = self.get_nearest_neigh_indices(trajectories, data.locations, nnp)
                data.locations = data.locations[batch_indices, nearest_indices]
                data.fx = data.fx[batch_indices, nearest_indices]
                data.dimension_mask = data.dimension_mask[batch_indices, nearest_indices]

            corrupt = self.data_corruption.corrupt_data(trajectories)
            data.to_train = TrainingData.Trajectory(trajectories=corrupt.corrupt_trajectory, times=times)
            data.mask = corrupt.mask
        else:
            data.to_train = TrainingData.Trajectory(trajectories=trajectories, times=times)

        return data

    @staticmethod
    def get_dim_mask_for_traj_training(trajectories: Tensor, dimension_mask: Tensor, config: FIMODETrainingConfig) -> Tensor:
        t = trajectories.size(1)
        b, _, d = dimension_mask.shape
        num_points = config.num_ic * config.traj_loss_steps * t
        return dimension_mask[:, :1, :].expand(b, num_points, d)

    @staticmethod
    def get_nearest_neigh_indices(trajectories: Tensor, locations: Tensor, num: int) -> Tuple[Tensor, Tensor]:
        b, t, n, d = trajectories.shape
        targets = trajectories.view(b, t * n, d)
        batch, num_locations, _ = locations.shape

        distances = ((targets.unsqueeze(2) - locations.unsqueeze(1)) ** 2).sum(dim=-1)
        min_distances = distances.min(dim=1).values
        _, sorted_indices = min_distances.sort(dim=-1)
        nearest_indices = sorted_indices[:, :num]
        batch_indices = torch.arange(batch).view(batch, 1).expand(batch, num)
        return batch_indices, nearest_indices


# ---------- Loss functions ----------


class LossFactory:
    """Factory returning a masked loss function (estimated, target, dimension_mask) -> Tensor."""

    @staticmethod
    def create(loss_type: str) -> Callable[[Tensor, Tensor, Tensor], Tensor]:
        if loss_type == "mse":
            return LossFactory.mse_at_locations
        elif loss_type == "huber":
            hub = nn.HuberLoss(reduction="none")

            def loss(e, t, m):
                return (hub(e * m, t) * m).sum(dim=-1)

            return loss
        elif loss_type == "relative_l2":
            return LossFactory.relative_l2_loss_masked
        elif loss_type == "l1":
            l1 = nn.L1Loss(reduction="none")

            def loss(e, t, m):
                lv = l1(e * m, t).sum(dim=-1)
                return lv / torch.sum(m, dim=-1).clamp(min=1)

            return loss
        elif loss_type == "huber_mean_dim":
            huber = nn.HuberLoss(reduction="none")

            def loss(e, t, m):
                lv = huber(e * m, t).sum(dim=-1)
                return lv / torch.sum(m, dim=-1).clamp(min=1)

            return loss
        elif loss_type == "il_mse":
            return LossFactory.il_mse_at_locations
        elif loss_type == "cos_sim":

            def loss(e, t, m):
                return 1 - torch.nn.functional.cosine_similarity(e * m, t * m, dim=-1)

            return loss
        else:
            raise ValueError(f"Unknown loss_type: {loss_type!r}")

    @staticmethod
    def mse_at_locations(estimated: Tensor, target: Tensor, dimension_mask: Tensor) -> Tensor:
        assert estimated.ndim == 3
        assert estimated.shape == target.shape == dimension_mask.shape
        se = dimension_mask * (estimated - target) ** 2
        se = se.sum(dim=-1)
        count = dimension_mask.sum(dim=-1).clamp(min=1)
        return se / count

    @staticmethod
    def relative_l2_loss_masked(estimated: Tensor, target: Tensor, dimension_mask: Tensor, eps: float = 1e-8) -> Tensor:
        err = (estimated - target) * dimension_mask
        return err.norm(2, dim=-1) / (target.norm(2, dim=-1) + eps)

    @staticmethod
    def il_mse_at_locations(estimated: Tensor, target: Tensor, dimension_mask: Tensor) -> Tensor:
        """Scale-invariant MSE (https://arxiv.org/pdf/2406.09130)."""
        count = dimension_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        mean_pred = (estimated * dimension_mask).sum(dim=-1, keepdim=True) / count
        mean_truth = target.sum(dim=-1, keepdim=True) / count
        N = (count - 1).clamp(min=1)
        var_pred = ((estimated - mean_pred) ** 2).sum(dim=-1, keepdim=True) / N
        var_truth = ((target - mean_truth) ** 2).sum(dim=-1, keepdim=True) / N
        y_pred = (estimated - mean_pred) / (var_pred.sqrt() + 1e-8)
        y_truth = (target - mean_truth) / (var_truth.sqrt() + 1e-8)
        return ((y_pred - y_truth) ** 2).sum(dim=-1) / count.squeeze(-1)


# ---------- ODE integrator ----------


class TrainIntegrator:
    """
    Integrates the learned vector field during trajectory-reconstruction training.
    Supports Euler, Improved Euler (Heun), and RK4.
    """

    def __init__(self, config: FIMODETrainingConfig):
        self.config = config
        assert config.intermediate_steps_per_step > 0

        integrators = {
            "euler": self.euler,
            "improved_euler": self.improved_euler,
            "rk4": self.runge_kutta_4,
        }
        self.integrate = integrators[config.integrator_for_trajectory_training]

    @dataclass
    class DriftAt:
        drift: Tensor

    @dataclass
    class SolverResult:
        prediction: Tensor
        truth: Tensor
        encoded_locations: Optional[Tensor] = None
        D: Optional[EncoderOutput] = None
        feature_mask: Optional[Tensor] = None

    @staticmethod
    def euler(f: Callable, h: Tensor, y: Tensor) -> Tensor:
        return y + h * f(y).drift

    @staticmethod
    def improved_euler(f: Callable, h: Tensor, y: Tensor) -> Tensor:
        k1 = f(y)
        k2 = f(y + (h / 2) * k1.drift)
        return y + h * k2.drift

    @staticmethod
    def runge_kutta_4(f: Callable, h: Tensor, y: Tensor) -> Tensor:
        k1 = f(y)
        k2 = f(y + (h / 2) * k1.drift)
        k3 = f(y + (h / 2) * k2.drift)
        k4 = f(y + h * k3.drift)
        return y + (h / 6) * (k1.drift + 2 * k2.drift + 2 * k3.drift + k4.drift)

    def build_f(
        self,
        model: "FIMODE",
        D: EncoderOutput,
        feature_mask: Tensor,
        concept: ODEConcepts.ODEConceptsBuilder,
    ) -> Callable[[Tensor], "TrainIntegrator.DriftAt"]:
        def f(y: Tensor) -> "TrainIntegrator.DriftAt":
            result = model.function_decoding(y, feature_mask, D, copy.deepcopy(concept))
            return self.DriftAt(drift=result.predictions.drift)

        return f

    def solve_ivp_for_random_initial_conditions(
        self,
        model: "FIMODE",
        data: TrainingData,
        relative_epoch: int,
        is_validation_batch: bool = False,
    ) -> "TrainIntegrator.SolverResult":
        """
        Integrates the ODE from multiple initial conditions during training.

        Context trajectories are encoded once; the vector field is then integrated
        from randomly (or evenly) spaced initial conditions along the ground-truth trajectory.
        Predictions are compared to the corresponding ground-truth points for the loss.
        All solving is done in normalized space/time.
        """
        device = data.truth.trajectories.device
        b, t, n, d = data.truth.trajectories.shape

        last_idx = t // 2 if self.config.use_generalization_trajs else t
        train_traj = data.to_train.trajectories[:, :last_idx]
        train_times = data.to_train.times[:, :last_idx]
        train_mask = data.mask[:, :last_idx]
        truth_traj = data.truth.trajectories
        truth_times = data.truth.times

        D, feature_mask, concept = model.trajectory_encoding(train_traj, train_times, train_mask)
        f = self.build_f(model, D, feature_mask, concept)

        num_steps = (
            max(1, round(self.config.relative_integration_horizon * n))
            if self.config.relative_integration_horizon is not None
            else self.config.traj_loss_steps
        )
        num_ic = self.config.num_ic
        intermediate_steps = self.config.intermediate_steps_per_step
        step_noise_scale = self.config.step_noise_scale
        ic_noise_scale = self.config.ic_noise_scale

        if self.config.random_initial_conditions:
            rand_vals = torch.rand(b, t, n - num_steps, 1, device=device)
            ic_indices = torch.topk(rand_vals, k=num_ic, dim=2).indices.expand(b, t, num_ic, d)
        else:
            max_start = n - num_steps - 1
            if num_ic == 1:
                pts = torch.tensor([max_start // 2], device=device)
            else:
                pts = torch.linspace(0, max_start, num_ic, device=device).long()
            ic_indices = pts.view(1, 1, num_ic, 1).expand(b, t, num_ic, d).long()

        y0 = torch.gather(truth_traj, dim=2, index=ic_indices)
        y0 = model.pad_if_necessary(y0)
        y0 = model.spatial_norm.normalization_map(y0, concept._states_norm_stats)
        if ic_noise_scale > 0.0 and not is_validation_batch:
            y0 = y0 + ic_noise_scale * torch.randn_like(y0)

        target_indices = ic_indices.repeat(1, 1, num_steps, 1)
        shift = torch.arange(1, num_steps + 1, device=device).repeat_interleave(num_ic).view(1, 1, num_steps * num_ic, 1)
        target_indices = target_indices + shift

        times = model.temporal_norm.normalization_map(truth_times, concept._times_norm_stats)
        delta_times = times[:, :, 1:, :] - times[:, :, :-1, :]
        time_indices = target_indices.view(b, t, num_steps, num_ic, d) - 1

        y = y0.view(b, t * num_ic, d).clone()
        ys = []

        if self.config.use_h_max:
            h_max = torch.tensor(self.config.h_max, device=device, dtype=train_traj.dtype)
            for step in range(num_steps):
                h = torch.gather(delta_times, dim=2, index=time_indices[:, :, step, :, :1]).view(b, t * num_ic, 1)
                n_inter = torch.ceil(h / h_max).int()
                max_n_inter = n_inter.max().item()
                h_inter = h / n_inter
                for i in range(max_n_inter):
                    active = (i < n_inter).float()
                    y = self.integrate(f, h_inter * active, y)
                    if step_noise_scale > 0.0 and not is_validation_batch:
                        y = y + step_noise_scale * torch.sqrt(h_inter * active) * torch.randn_like(y) * active
                ys.append(y.view(b, t, num_ic, d))
        else:
            for step in range(num_steps):
                h = torch.gather(delta_times, dim=2, index=time_indices[:, :, step, :, :1]).view(b, t * num_ic, 1)
                h_inter = h / intermediate_steps
                for _ in range(intermediate_steps):
                    y = self.integrate(f, h_inter, y)
                    if step_noise_scale > 0.0 and not is_validation_batch:
                        y = y + step_noise_scale * torch.sqrt(h_inter) * torch.randn_like(y)
                ys.append(y.view(b, t, num_ic, d))

        ys = torch.cat(ys, dim=2)

        trajs = model.spatial_norm.normalization_map(truth_traj, concept._states_norm_stats)
        if not self.config.train_with_normalized_head:
            concept = concept.locations(ys.view(b, t * num_ic * num_steps, d)).drift(torch.empty((b, 1, d))).normalized(True).build()
            concept.renormalize()
            ys = concept.locations.view(b, t, num_ic * num_steps, d)
            trajs = truth_traj

        corresponding_trajs = torch.gather(trajs, dim=2, index=target_indices)
        return self.SolverResult(prediction=ys, truth=corresponding_trajs)
