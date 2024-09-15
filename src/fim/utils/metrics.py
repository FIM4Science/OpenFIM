"""metrics used throughout the project."""

from typing import Optional

import torch


def r2_score(prediction: torch.Tensor, ground_truth: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute the R2 score of a prediction. The score is normalized with respect to the mean of the ground truth.

    Args:
        prediction: the predicted values. Shape [B, T, D]
        ground_truth: the true values. Shape [B, T, D]
        mask: a mask indicating if there are padding values. 1 if value is padding, 0 else. Shape [B, T]

    Returns:
        percentage of R2 scores above 0.9. Shape [1]
        the R2 score, averaged over the dimensions. Shape [1]
        the standard deviation of the R2 score. Shape [1]
    """
    if mask is None:
        mask = torch.zeros((prediction.size(0), prediction.size(1)), dtype=torch.bool)
    expanded_mask = mask.unsqueeze(-1).expand(-1, -1, prediction.size(-1))

    ground_truth_mean = torch.nanmean(
        torch.where(~expanded_mask, ground_truth, torch.tensor(float("nan"))), axis=1
    )  #  [B, D]

    squared_diff_res = (ground_truth - prediction) ** 2
    masked_squared_diff_res = torch.where(~expanded_mask, squared_diff_res, torch.tensor(float("nan")))
    ss_res = torch.nansum(masked_squared_diff_res, axis=1)  # [B, D]

    squared_diff_tot = (ground_truth - ground_truth_mean.unsqueeze(1)) ** 2
    squared_diff_tot_masked = torch.where(~expanded_mask, squared_diff_tot, torch.tensor(float("nan")))
    ss_tot = torch.nansum(squared_diff_tot_masked, axis=1)  # [B, D]

    r2 = 1 - ss_res / ss_tot  # [B, D]

    # compute average across dimensions
    r2_mean_per_sample = torch.mean(r2, axis=1)  # [B]

    r2_above09 = torch.sum(r2_mean_per_sample > 0.9) / r2_mean_per_sample.size(0)  # [1]
    r2_mean = torch.mean(r2_mean_per_sample)  # [1]
    r2_std = torch.std(r2_mean_per_sample)  # [1]

    return r2_above09, r2_mean, r2_std


def compute_metrics(predictions: torch.Tensor, ground_truth: torch.Tensor, mask: Optional[torch.Tensor] = None) -> dict:
    """
    Compute the metrics of a prediction given the ground truth.

    Metrics: (with x_t^i the ground truth, \\hat{x}_t^i the predicted value, D the number of dimensions, T the horizon length, \bar{x}^i the mean of the ground truth)
        MAE = 1/T \\sum_{t=1}^T 1/D \\sum_{i=1}^D | x_t^i - \\hat{x}_t^i |
        MSE = 1/T \\sum_{t=1}^T 1/D \\sum_{i=1}^D ( x_t^i - \\hat{x}_t^i)^2
        RMSE = \\sqrt{MSE}
        R2 = 1/D \\sum_{i=1}^D [ 1 - (\\sum_{t=1}^T ( x_t^i - \\hat{x}_t^i )^2)/(\\sum_{t=1}^{T} ( x_t^i - \bar{x}^i )^2) ]

    Args:
        predictions: values predicted by the model. shape: (B, horizon_length, n_dims)
        ground_truth: the true values. shape: (B, horizon_length, n_dims)
        mask: a mask indicating if there are padding values. 1 if value is padding, 0 else. shape: (B, horizon_length)

    Returns:
        a dictionary containing the metrics, averaged over all samples and the standard deviation.
    """
    if predictions.shape != ground_truth.shape:
        raise ValueError(
            f"prediction and ground_truth must have the same shape, got {predictions.shape} and {ground_truth.shape}"
        )
    if mask is None:
        mask = torch.zeros((predictions.size(0), predictions.size(1)), dtype=torch.bool)
    expanded_mask = mask.unsqueeze(-1).expand(-1, -1, predictions.size(-1))

    r2_above09, r2_mean, r2_std = r2_score(predictions, ground_truth, mask)

    abs_diff = torch.abs(ground_truth - predictions)
    abs_diff_masked = torch.where(~expanded_mask, abs_diff, torch.tensor(float("nan")))
    mae_per_sample = torch.mean(torch.nanmean(abs_diff_masked, axis=1), axis=-1)

    squared_diff = (ground_truth - predictions) ** 2
    squared_diff_masked = torch.where(~expanded_mask, squared_diff, torch.tensor(float("nan")))
    mse_per_sample = torch.mean(torch.nanmean(squared_diff_masked, axis=1), axis=-1)

    return {
        "r2_score_mean": r2_mean.item(),
        "r2_score_std": r2_std.item(),
        "r2_score_above0.9": r2_above09.item(),
        "mae_mean": torch.mean(mae_per_sample).item(),
        "mae_std": torch.std(mae_per_sample).item(),
        "mse_mean": torch.mean(mse_per_sample).item(),
        "mse_std": torch.std(mse_per_sample).item(),
        "rmse_mean": torch.mean(torch.sqrt(mse_per_sample)).item(),
        "rmse_std": torch.std(torch.sqrt(mse_per_sample)).item(),
    }
