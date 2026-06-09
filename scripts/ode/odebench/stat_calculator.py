import pickle
from dataclasses import dataclass, is_dataclass, fields
from operator import truth
from pathlib import Path
from typing import Final, List, Optional, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor
from tqdm import tqdm

from fim.data_generation.sde.lipschitz_systems import solve_ivp_one_step_method, \
    solve_ivp_one_step_method_with_delta_times
from utils.eval_models import OdeonEval, PredictionModel, OdeFormerEval
from odebench.plotting_utils import save_all_figures_to_pdf
from odebench.dataProvider import FimDataloader, SpecificDimFimDataset


@dataclass(kw_only=True)
class StatAtom:
    name: Final[str]

    def join(self, other: 'StatAtom') -> 'StatAtom':
        pass


    def visualize(self, title_extension: Optional[str] = ""):
        pass


class VFStatCalculator:
    def calc_stat(self, location: Tensor, estimated_drift: Tensor, truth_drift: Tensor, dim_mask: Tensor) -> Tensor:
        raise NotImplementedError("This method should be implemented in subclasses.")


class TrajStatCalculator:
    def calc_stat(self, estimated_trajectory: Tensor, truth_trajectory: Tensor, dim: int) -> Tensor:
        raise NotImplementedError("This method should be implemented in subclasses.")


class RMSEStatCalculator(VFStatCalculator):
    @dataclass(kw_only=True)
    class RMSEStat(StatAtom):
        name: Final[str] = "RMSE"
        mean_over_all_locations: Tensor

        def join(self, other: 'RMSEStat') -> 'RMSEStat':
            if not isinstance(other, RMSEStatCalculator.RMSEStat):
                raise TypeError(f"Cannot join with {type(other)}")

            val = torch.cat([self.mean_over_all_locations, other.mean_over_all_locations], dim=0)
            return RMSEStatCalculator.RMSEStat(mean_over_all_locations=val)

        def visualize(self, title_extension: Optional[str] = ""):
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))

            counts, bins, patches = ax.hist(self.mean_over_all_locations.cpu().numpy(), edgecolor='black')
            ax.bar_label(patches, labels=[f'{int(c)}' if c > 0 else '' for c in counts],
                         padding=3, fontsize=10)
            ax.set_title(f"RMSE Mean over all locations for all batches\n{title_extension}")
            ax.set_xlabel('RMSE')
            ax.set_ylabel('Frequency')

            plt.tight_layout()

    def calc_stat(self, location: Tensor, estimated_drift: Tensor, truth_drift: Tensor, dim_mask: Tensor) -> RMSEStat:
        se = (estimated_drift - truth_drift) ** 2
        se_masked = se * dim_mask

        dim_count = dim_mask.sum(dim=-1)
        assert (dim_count > 0).all(), "All dimensions should have at least one active element in the mask."

        mse = se_masked.sum(dim=-1) / dim_count

        rmse = torch.sqrt(mse)

        rmse_for_batch = rmse.mean(dim=1)

        return self.RMSEStat(mean_over_all_locations=rmse_for_batch)


class MAEStatCalculator(VFStatCalculator):
    @dataclass(kw_only=True)
    class MAEStat(StatAtom):
        name: Final[str] = "MAE"
        mean_over_all_locations: Tensor

        def join(self, other: 'MAEStat') -> 'MAEStat':
            if not isinstance(other, MAEStatCalculator.MAEStat):
                raise TypeError(f"Cannot join with {type(other)}")

            val = torch.cat([self.mean_over_all_locations, other.mean_over_all_locations], dim=0)
            return MAEStatCalculator.MAEStat(mean_over_all_locations=val)

    def calc_stat(self, location: Tensor, estimated_drift: Tensor, truth_drift: Tensor, dim_mask: Tensor) -> MAEStat:
        dim_count = dim_mask.sum(dim=-1, keepdim=True)
        assert (dim_count > 0).all(), "All dimensions should have at least one non-zero element in the mask."

        mae = torch.nn.functional.l1_loss(estimated_drift, truth_drift, reduction="none")
        mae = mae / dim_count
        mae = mae * dim_mask
        mae = torch.sum(mae, dim=-1)

        mae_for_batch = mae.mean(dim=1)

        return self.MAEStat(mean_over_all_locations=mae_for_batch)


class RelativeL2StatCalculator(VFStatCalculator):
    @dataclass(kw_only=True)
    class RelativeL2Stat(StatAtom):
        name: Final[str] = "Relative L2"
        mean_over_all_locations: Tensor

        def join(self, other: 'RelativeL2Stat') -> 'RelativeL2Stat':
            if not isinstance(other, RelativeL2StatCalculator.RelativeL2Stat):
                raise TypeError(f"Cannot join with {type(other)}")

            val = torch.cat([self.mean_over_all_locations, other.mean_over_all_locations], dim=0)
            return RelativeL2StatCalculator.RelativeL2Stat(mean_over_all_locations=val)

        def visualize(self, title_extension: Optional[str] = ""):
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))

            counts, bins, patches = ax.hist(self.mean_over_all_locations.cpu().numpy(), edgecolor='black')
            ax.bar_label(patches, labels=[f'{int(c)}' if c > 0 else '' for c in counts],
                         padding=3, fontsize=10)
            ax.set_title(f"Relative L2 Mean over all locations for all batches\n{title_extension}")
            ax.set_xlabel('Relative L2')
            ax.set_ylabel('Frequency')

            plt.tight_layout()

    def calc_stat(self, location: Tensor, estimated_drift: Tensor, truth_drift: Tensor,
                  dim_mask: Tensor) -> RelativeL2Stat:
        estimated_drift = estimated_drift * dim_mask
        truth_drift = truth_drift * dim_mask

        gt_norm = torch.linalg.norm(truth_drift, dim=-1)

        error = estimated_drift - truth_drift
        error_norm = torch.linalg.norm(error, dim=-1)
        relative_l2 = error_norm / gt_norm

        relative_l2_for_batch = relative_l2.mean(dim=1)

        return self.RelativeL2Stat(mean_over_all_locations=relative_l2_for_batch)


class PercentStatCalculator(VFStatCalculator):
    threshold: Final[float] = 0.1

    @dataclass(kw_only=True)
    class PercentStat(StatAtom):
        name: Final[str] = "Percent"
        percent_over_all_locations: Tensor

        def join(self, other: 'PercentStat') -> 'PercentStat':
            if not isinstance(other, PercentStatCalculator.PercentStat):
                raise TypeError(f"Cannot join with {type(other)}")

            val = torch.cat([self.percent_over_all_locations, other.percent_over_all_locations], dim=0)
            return PercentStatCalculator.PercentStat(percent_over_all_locations=val)

        def visualize(self, title_extension: Optional[str] = ""):
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))

            counts, bins, patches = ax.hist(self.percent_over_all_locations.cpu().numpy(), edgecolor='black')
            ax.bar_label(patches, labels=[f'{int(c)}' if c > 0 else '' for c in counts],
                         padding=3, fontsize=10)
            ax.set_title(f"Percent of locations with l1_norm > truth_magnitude\n{title_extension}")
            ax.set_xlabel('Percent')
            ax.set_ylabel('Frequency')

            plt.tight_layout()

    def calc_stat(self, location: Tensor, estimated_drift: Tensor, truth_drift: Tensor,
                  dim_mask: Tensor) -> PercentStat:
        mae = torch.nn.functional.l1_loss(estimated_drift, truth_drift, reduction="none")
        mae = mae * dim_mask

        mae_norm = torch.linalg.norm(mae, dim=-1)
        truth_drift = truth_drift * dim_mask
        truth_norm = torch.linalg.norm(truth_drift, dim=-1)

        num_bigger_threshold = (mae_norm > truth_norm * self.threshold).sum(dim=1)

        percent = num_bigger_threshold / truth_drift.size(1)

        return self.PercentStat(percent_over_all_locations=percent)


class CosineSimStatCalculator(VFStatCalculator):
    @dataclass(kw_only=True)
    class CosineSimilarityStat(StatAtom):
        name: Final[str] = "Cosine Similarity"
        mean_over_all_locations: Tensor

        def join(self, other: 'CosineSimilarityStat') -> 'CosineSimilarityStat':
            if not isinstance(other, CosineSimStatCalculator.CosineSimilarityStat):
                raise TypeError(f"Cannot join with {type(other)}")

            val = torch.cat([self.mean_over_all_locations, other.mean_over_all_locations], dim=0)
            return CosineSimStatCalculator.CosineSimilarityStat(mean_over_all_locations=val)

        def visualize(self, title_extension: Optional[str] = ""):
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))

            counts, bins, patches = ax.hist(self.mean_over_all_locations.cpu().numpy(), edgecolor='black')
            ax.bar_label(patches, labels=[f'{int(c)}' if c > 0 else '' for c in counts],
                         padding=3, fontsize=10)
            ax.set_title(f"Cosine Similarity Mean over all location for all batches\n{title_extension}")
            ax.set_xlabel('Cosine Similarity')
            ax.set_ylabel('Frequency')

            plt.tight_layout()

    def calc_stat(self, location: Tensor, estimated_drift: Tensor, truth_drift: Tensor,
                  dim_mask: Tensor) -> CosineSimilarityStat:
        estimated_drift = estimated_drift * dim_mask
        truth_drift = truth_drift * dim_mask

        cos_sim = torch.nn.functional.cosine_similarity(estimated_drift, truth_drift, dim=-1)

        cos_sim = cos_sim.mean(dim=-1)

        return self.CosineSimilarityStat(mean_over_all_locations=cos_sim)


class MagnitudeErrorStatCalculator(VFStatCalculator):
    @dataclass(kw_only=True)
    class MagnitudeErrorStat(StatAtom):
        name: Final[str] = "Magnitude Error"
        mean_over_all_locations: Tensor

        def join(self, other: 'MagnitudeErrorStat') -> 'MagnitudeErrorStat':
            if not isinstance(other, MagnitudeErrorStatCalculator.MagnitudeErrorStat):
                raise TypeError(f"Cannot join with {type(other)}")

            val = torch.cat([self.mean_over_all_locations, other.mean_over_all_locations], dim=0)
            return MagnitudeErrorStatCalculator.MagnitudeErrorStat(mean_over_all_locations=val)

    def calc_stat(self, location: Tensor, estimated_drift: Tensor, truth_drift: Tensor,
                  dim_mask: Tensor) -> MagnitudeErrorStat:
        estimated_drift = estimated_drift * dim_mask
        truth_drift = truth_drift * dim_mask

        pred_mag = torch.linalg.norm(estimated_drift, dim=-1)
        truth_mag = torch.linalg.norm(truth_drift, dim=-1)

        mag_err = torch.abs(pred_mag - truth_mag)

        mag_err = mag_err.mean(dim=-1)

        return self.MagnitudeErrorStat(mean_over_all_locations=mag_err)


class R2VarianceWeighterStatCalculator(TrajStatCalculator):
    thresholds: Final[list[float]] = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]

    @dataclass(kw_only=True)
    class R2VarianceWeightedStat(StatAtom):
        name: Final[str] = "R2 Variance Weighted"
        number_of_r2_above_threshold: Dict[float, float]
        num_trajectories: int

        def join(self, other: 'R2VarianceWeightedStat') -> 'R2VarianceWeightedStat':
            if not isinstance(other, R2VarianceWeighterStatCalculator.R2VarianceWeightedStat):
                raise TypeError(f"Cannot join with {type(other)}")

            combined_thresholds = set(self.number_of_r2_above_threshold.keys()).union(
                other.number_of_r2_above_threshold.keys())
            combined_results = {
                threshold: self.number_of_r2_above_threshold.get(threshold, 0) + other.number_of_r2_above_threshold.get(
                    threshold, 0)
                for threshold in combined_thresholds}

            return R2VarianceWeighterStatCalculator.R2VarianceWeightedStat(
                number_of_r2_above_threshold=combined_results,
                num_trajectories=self.num_trajectories + other.num_trajectories
            )

        def visualize(self, title_extension: Optional[str] = ""):
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))

            labels = list(self.number_of_r2_above_threshold.keys())
            values = np.array(list(self.number_of_r2_above_threshold.values())) / self.num_trajectories
            values = np.round(values, 2)

            bars = ax.bar(labels, values, width=0.025, edgecolor='black')
            ax.bar_label(bars)

            ax.set_title(f"Percent of trajectories with R2 > threshold\n{title_extension}")
            ax.set_xlabel('Threshold')
            ax.set_ylabel('Frequency')

            plt.tight_layout()

    def calc_stat(self, estimated_trajectory: Tensor, truth_trajectory: Tensor,
                  dim: int) -> R2VarianceWeightedStat:
        estimated_trajectory = estimated_trajectory[..., :dim]
        truth_trajectory = truth_trajectory[..., :dim]

        b, t, n, d = estimated_trajectory.shape

        estimated_trajectory = estimated_trajectory.reshape(b * t, n, d)
        truth_trajectory = truth_trajectory.reshape(b * t, n, d)

        r2_score = self.r2_score(truth_trajectory, estimated_trajectory, variance_weighted=True)

        # r2_score = r2_score.view(b, t)

        results = {}
        for threshold in self.thresholds:
            count = (r2_score > threshold).sum().item()
            results[threshold] = count

        return self.R2VarianceWeightedStat(number_of_r2_above_threshold=results, num_trajectories=b * t)

    @staticmethod
    def r2_score(y_true: torch.Tensor, y_pred: torch.Tensor, variance_weighted: bool = True) -> torch.Tensor:
        """
        https://github.com/scikit-learn/scikit-learn/blob/da08f3d99194565caaa2b6757a3816eef258cd70/sklearn/metrics/_regression.py

        Calculate variance-weighted R² score for shape (b, n, d).

        Args:
            y_true: Ground truth values, shape (b, n, d)
            y_pred: Predicted values, shape (b, n, d)
            variance_weighted: If True, returns variance-weighted R² score across dimensions d.

        Returns:
            R² scores, shape (b,) - variance-weighted across d dimension

        Formula:
            R² = 1 - (SS_res / SS_tot) for each dimension d
            Final R² = weighted average across d, weighted by variance of each dimension
        """
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape")

        # Calculate mean along n dimension for each batch and dimension
        y_mean = y_true.mean(dim=1, keepdim=True)  # shape: (b, 1, d)

        # Calculate SS_res and SS_tot for each dimension
        ss_res = ((y_true - y_pred) ** 2).sum(dim=1)  # shape: (b, d)
        ss_tot = ((y_true - y_mean) ** 2).sum(dim=1)  # shape: (b, d)

        # Calculate R² for each dimension
        r2_per_dim = 1 - (ss_res / (ss_tot + 1e-15))  # shape: (b, d)

        if not variance_weighted:
            return r2_per_dim

        # Calculate variance weights for each dimension
        variance_weights = ss_tot  # Use ss_tot as variance weights

        # Check if any batch has non-zero denominators
        nonzero_denominator = variance_weights > 1e-15  # shape: (b, d)
        batch_has_nonzero = nonzero_denominator.any(dim=1)  # shape: (b,)

        # For batches where all denominators are zero, fall back to uniform weights
        uniform_weights = torch.ones_like(variance_weights) / variance_weights.shape[1]  # shape: (b, d)

        # Use variance weights where possible, uniform weights otherwise
        final_weights = torch.where(
            batch_has_nonzero.unsqueeze(1),  # Broadcast to (b, 1)
            variance_weights / (variance_weights.sum(dim=1, keepdim=True) + 1e-15),  # Normalized variance weights
            uniform_weights  # Uniform weights fallback
        )

        # Variance-weighted average across dimensions
        r2_weighted = (r2_per_dim * final_weights).sum(dim=1)  # shape: (b,)

        return r2_weighted


class TrajL1Calculator(TrajStatCalculator):
    @dataclass(kw_only=True)
    class TrajL1Stat(StatAtom):
        name: Final[str] = "Trajectory L1"
        mean_over_all_trajectories_per_batch: Tensor

        def join(self, other: 'TrajL1Stat') -> 'TrajL1Stat':
            if not isinstance(other, TrajL1Calculator.TrajL1Stat):
                raise TypeError(f"Cannot join with {type(other)}")

            val = torch.cat([self.mean_over_all_trajectories_per_batch, other.mean_over_all_trajectories_per_batch],
                            dim=0)
            return TrajL1Calculator.TrajL1Stat(mean_over_all_trajectories_per_batch=val)

        def visualize(self, title_extension: Optional[str] = ""):
            fig, ax = plt.subplots(1, 1, figsize=(15, 10))

            counts, bins, patches = ax.hist(self.mean_over_all_trajectories_per_batch, edgecolor='black')
            ax.bar_label(patches, labels=[f'{int(c)}' if c > 0 else '' for c in counts],
                            padding=3, fontsize=10)
            ax.set_title(f"L1 Distribution over all trajectories in batch\n{title_extension}")
            ax.set_xlabel('L1')
            ax.set_ylabel('Frequency')

            plt.tight_layout()


    def calc_stat(self, estimated_trajectory: Tensor, truth_trajectory: Tensor, dim: int) -> TrajL1Stat:
        estimated_trajectory = estimated_trajectory[..., :dim]
        truth_trajectory = truth_trajectory[..., :dim]

        b, t, n, d = estimated_trajectory.shape

        l1_error = torch.nn.functional.l1_loss(estimated_trajectory, truth_trajectory, reduction="none")
        l1_error = l1_error.view(b, t * n, d).sum(dim=-1) / dim

        l1_error = l1_error.mean(dim=1)

        return self.TrajL1Stat(mean_over_all_trajectories_per_batch=l1_error)


class msMAPEStatCalculator(TrajStatCalculator):
    # https://arxiv.org/pdf/2310.10688 (https://arxiv.org/pdf/2105.06643)
    eps: float = 0.1

    @dataclass(kw_only=True)
    class msMAPEStat(StatAtom):
        name: Final[str] = "msMAPE"
        mean_dim_msmape_per_batch: Tensor
        median_dim_msmape_per_batch: Tensor

        def join(self, other: 'msMAPEStat') -> 'msMAPEStat':
            if not isinstance(other, msMAPEStatCalculator.msMAPEStat):
                raise TypeError(f"Cannot join with {type(other)}")

            mean_val = torch.cat([self.mean_dim_msmape_per_batch, other.mean_dim_msmape_per_batch], dim=0)
            median_val = torch.cat([self.median_dim_msmape_per_batch, other.median_dim_msmape_per_batch], dim=0)
            return msMAPEStatCalculator.msMAPEStat(mean_dim_msmape_per_batch=mean_val,
                                                   median_dim_msmape_per_batch=median_val)

        def visualize(self, title_extension: Optional[str] = ""):
            fig, ax = plt.subplots(2, 1, figsize=(15, 20))

            counts, bins, patches = ax[0].hist(self.mean_dim_msmape_per_batch, edgecolor='black')
            ax[0].bar_label(patches, labels=[f'{int(c)}' if c > 0 else '' for c in counts],
                         padding=3, fontsize=10)
            ax[0].set_title(f"Mean msMAPE over dim per function (over all trajectories in batch)\n{title_extension}")
            ax[0].set_xlabel('Mean msMAPE')
            ax[0].set_ylabel('Frequency')

            counts, bins, patches = ax[1].hist(self.median_dim_msmape_per_batch, edgecolor='black')
            ax[1].bar_label(patches, labels=[f'{int(c)}' if c > 0 else '' for c in counts],
                            padding=3, fontsize=10)
            ax[1].set_title(f"Median msMAPE over dim per function (over all trajectories in batch)\n{title_extension}")
            ax[1].set_xlabel('Meadian msMAPE')
            ax[1].set_ylabel('Frequency')

            plt.tight_layout()

    def calc_stat(self, estimated_trajectory: Tensor, truth_trajectory: Tensor,
                  dim: int) -> msMAPEStat:
        est_traj = estimated_trajectory[..., :dim]
        truth_traj = truth_trajectory[..., :dim]

        numerator = 2 * torch.abs(truth_traj - est_traj)

        denominator_left = torch.abs(truth_traj) + torch.abs(est_traj) + self.eps
        denominator_right = torch.tensor(0.5 + self.eps, device=est_traj.device, dtype=est_traj.dtype)
        denominator = torch.maximum(denominator_left, denominator_right)

        denominator[denominator == 0] = 1e-15

        msmape_value = torch.mean(numerator / denominator, dim=2)

        msmape_value = msmape_value.mean(dim=1)

        msmape_mean = msmape_value.mean(dim=-1)
        msmape_median = msmape_value.median(dim=-1)

        return self.msMAPEStat(mean_dim_msmape_per_batch=msmape_mean, median_dim_msmape_per_batch=msmape_median.values)


class StatCalculationRunner:

    @dataclass
    class ResultContainer:
        stats: List[StatAtom]
        num_predictions_total: int
        num_invalid_predictions: int

    def __init__(self, model: PredictionModel, batch_size: int, num_workers: int, prefetch_factor: int, steps_per_dt: int = 2):
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.steps_per_dt = steps_per_dt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.vf_calculators = [
            RMSEStatCalculator(),
            MAEStatCalculator(),
            # RelativeL2StatCalculator(),
            # PercentStatCalculator(),
            CosineSimStatCalculator(),
            MagnitudeErrorStatCalculator()
        ]

        self.traj_calculators = [
            R2VarianceWeighterStatCalculator(),
            TrajL1Calculator(),
            msMAPEStatCalculator()
        ]

    def _get_dataloader(self, data_path: Path, dim: int) -> torch.utils.data.DataLoader:
        train_dataset = SpecificDimFimDataset(data_path=data_path, expected_dim=dim)

        data = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=True
        )

        return tqdm(data, desc=f"Loading data for dim {dim}", unit="batch")

    @torch.no_grad()
    def run_vf_stats(self, data_path: Path, dim: int, max_num_traj: Optional[int] = None, num_equispaced_points: Optional[int] = None) -> List[StatAtom]:
        acc_results = []

        for i, (fx, coord, traj, time) in enumerate(self._get_dataloader(data_path, dim)):
            fx, coord, traj, time = fx.to(self.device), coord.to(self.device), traj.to(self.device), time.to(self.device)
            if max_num_traj is not None:
                traj = traj[:, :max_num_traj, :, :]
                time = time[:, :max_num_traj, :, :]
            if num_equispaced_points is not None:
                p_idx = torch.linspace(0, traj.size(2) - 1, num_equispaced_points).long().to(self.device)
                traj = traj[:, :, p_idx, :]
                time = time[:, :, p_idx, :]

            self.model.fit(traj, time)

            pred_fx = self.model.system(coord)

            dim_mask = torch.ones_like(fx, dtype=torch.bool, device=fx.device)
            vf_stats = [calculator.calc_stat(coord, pred_fx, fx, dim_mask) for calculator in self.vf_calculators]

            if i == 0:
                acc_results = vf_stats
            else:
                acc_results = [acc.join(stat) for acc, stat in zip(acc_results, vf_stats)]

        return acc_results

    @torch.no_grad()
    def run_reconstruction_traj_stats(self, data_path: Path, dim: int, max_num_traj: Optional[int] = None, num_equispaced_points: Optional[int] = None) -> ResultContainer:
        acc_results = []

        num_preds_total = 0
        num_invalid_preds = 0

        for i, (_, _, traj, time) in enumerate(self._get_dataloader(data_path, dim)):
            traj, time = traj.to(self.device), time.to(self.device)
            if max_num_traj is not None:
                traj = traj[:, :max_num_traj, :, :]
                time = time[:, :max_num_traj, :, :]

            fit_traj = traj
            fit_time = time
            if num_equispaced_points is not None:
                p_idx = torch.linspace(0, traj.size(2) - 1, num_equispaced_points).long().to(self.device)
                fit_traj = fit_traj[:, :, p_idx, :]
                fit_time = fit_time[:, :, p_idx, :]

            valid_predictions = self.model.fit(fit_traj, fit_time)

            num_preds_total += traj.size(0)
            num_invalid_preds += (traj.size(0) - len(valid_predictions))
            traj = traj[valid_predictions, ...]
            time = time[valid_predictions, ...]

            delta_times = torch.diff(time, dim=2)
            y0 = traj[:, :, 0, :]
            t0 = time[:, :, 0, :]
            ys, _ = solve_ivp_one_step_method_with_delta_times(y0, t0, delta_times, self.steps_per_dt, self.model.system)

            ys = torch.stack(ys, dim=2)

            traj_stats = [calculator.calc_stat(ys, traj, dim) for calculator in self.traj_calculators]

            if i == 0:
                acc_results = traj_stats
            else:
                acc_results = [acc.join(stat) for acc, stat in zip(acc_results, traj_stats)]

        return self.ResultContainer(stats=acc_results, num_predictions_total=num_preds_total, num_invalid_predictions=num_invalid_preds)


    @torch.no_grad()
    def run_generalization_traj_stats(self, data_path: Path, dim: int, num_model_input_traj: int, max_num_traj: Optional[int] = None, num_equispaced_points: Optional[int] = None) -> ResultContainer:
        acc_results = []

        num_preds_total = 0
        num_invalid_preds = 0

        for i, (_, _, traj, time) in enumerate(self._get_dataloader(data_path, dim)):
            traj, time = traj.to(self.device), time.to(self.device)
            if max_num_traj is not None:
                traj = traj[:, :max_num_traj, :, :]
                time = time[:, :max_num_traj, :, :]

            if num_equispaced_points is not None:
                p_idx = torch.linspace(0, traj.size(2) - 1, num_equispaced_points).long().to(self.device)
                traj = traj[:, :, p_idx, :]
                time = time[:, :, p_idx, :]

            fit_traj = traj[:, :num_model_input_traj, :, :]
            fit_time = time[:, :num_model_input_traj, :, :]

            valid_predictions = self.model.fit(fit_traj, fit_time)

            num_preds_total += traj.size(0)
            num_invalid_preds += (traj.size(0) - len(valid_predictions))
            traj = traj[valid_predictions, ...]
            time = time[valid_predictions, ...]

            delta_times = torch.diff(time, dim=2)
            y0 = traj[:, :, 0, :]
            t0 = time[:, :, 0, :]
            ys, _ = solve_ivp_one_step_method_with_delta_times(y0, t0, delta_times, self.steps_per_dt, self.model.system)

            ys = torch.stack(ys, dim=2)

            ys = ys[:, num_model_input_traj:, :, :]
            traj = traj[:, num_model_input_traj:, :, :]

            traj_stats = [calculator.calc_stat(ys, traj, dim) for calculator in self.traj_calculators]

            if i == 0:
                acc_results = traj_stats
            else:
                acc_results = [acc.join(stat) for acc, stat in zip(acc_results, traj_stats)]

        return self.ResultContainer(stats=acc_results, num_predictions_total=num_preds_total, num_invalid_predictions=num_invalid_preds)

    @staticmethod
    def move_to_cpu(obj):
        if torch.is_tensor(obj):
            return obj.cpu()
        elif isinstance(obj, list):
            return [StatCalculationRunner.move_to_cpu(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: StatCalculationRunner.move_to_cpu(v) for k, v in obj.items()}
        elif is_dataclass(obj):
            for f in fields(obj):
                setattr(obj, f.name, StatCalculationRunner.move_to_cpu(getattr(obj, f.name)))
        return obj
    
    def save_results_to_file(self, results: List[StatAtom], file_path: Path):
        with open(file_path.with_suffix('.pkl'), 'wb') as f:
            res = self.move_to_cpu(results)
            pickle.dump(res, f)

        for res in results:
            res.visualize(title_extension=f"{file_path.name}")

        save_all_figures_to_pdf(str(file_path.with_suffix('.pdf')))


    def save_result_container_to_file(self, results: ResultContainer, file_path: Path):
        with open(file_path.with_suffix('.pkl'), 'wb') as f:
            results.stats = self.move_to_cpu(results.stats)
            pickle.dump(results, f)



if __name__ == '__main__':
    # with open("/home/teddev/Downloads/res/plots/generalization_stats_FIM-37-normal_model_9path.pkl", "rb") as f:
    #     res = pickle.load(f)
    #
    #     for r in res:
    #         r.visualize("Dim 1")
    #
    #     plt.show()
    #
    #     print(res)
    
    
    
    
    
    batch_size = 2
    num_workers = 1
    prefetch_factor = 2

    dim = 2
    path = Path(
        f"/home/teddev/PycharmProjects/pytorch_stuff/foundation_models_dynamical_systems/data_local/fim-data-more-path/0/data/processed/train/30k_drift_deg_3_ablation_studies/degree_and_monomial_survival_uniform/train/train_deg_{dim}")

    # model = OdeonEval(
    #     Path("/home/teddev/PycharmProjects/FIM/scripts/results/new_data_hub_5kpoints_05-18-1922/checkpoints"))
    model = OdeFormerEval()

    with torch.no_grad():
        scr = StatCalculationRunner(model, batch_size, num_workers, prefetch_factor)
        res = scr.run_reconstruction_traj_stats(path, dim, max_num_traj=1, num_equispaced_points=50)
        # res = scr.run_generalization_traj_stats(path, dim, 1, 2)
        # res = scr.run_vf_stats(path, dim, 1,  num_equispaced_points=100)
        print(res)
        scr.save_results_to_file(res, Path("results.pkl"))


# b, l, d = 4, 11, 2
# estimated_drift = torch.randn((b, l, d))
# truth_drift = torch.randn((b, l, d))
# dim_mask = torch.ones((b, l, d), dtype=torch.bool)
#
# calculator = MagnitudeErrorStatCalculator()
# stat = calculator.calc_stat(None, estimated_drift, truth_drift, dim_mask)
#
# print(stat.name)
# print(stat)

# t, n = 10, 100
# estimated_traj = torch.randn((b, t, n, d))
# truth_traj = torch.randn((b, t, n, d))
#
# calculator = msMAPEStatCalculator()
# stat = calculator.calc_stat(estimated_traj, truth_traj, dim=d)
#
# print(stat.name)
# print(stat)
