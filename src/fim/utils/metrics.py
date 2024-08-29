"""metrics used throughout the project."""

import torch


def r2_score(prediction: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    """
    Compute the R2 score of a prediction. The score is normalized with respect to the mean of the ground truth.

    Args:
        prediction: the predicted values. Shape [B, T, D]
        ground_truth: the true values. Shape [B, T, D]

    Returns:
        percentage of R2 scores above 0.9. Shape [1]
        the R2 score, averaged over the dimensions. Shape [1]
        the standard deviation of the R2 score. Shape [1]
    """
    ground_truth_mean = torch.mean(ground_truth, axis=1)  #  [B, D]
    ss_res = torch.sum((ground_truth - prediction) ** 2, axis=1)  # [B, D]
    ss_tot = torch.sum((ground_truth - ground_truth_mean.unsqueeze(1)) ** 2, axis=1)  # [B, D]
    r2 = 1 - ss_res / ss_tot  # [B, D]

    # compute average across dimensions
    r2_mean_per_sample = torch.mean(r2, axis=1)  # [B]

    r2_above09 = torch.sum(r2_mean_per_sample > 0.9) / r2_mean_per_sample.size(0)  # [1]
    r2_mean = torch.mean(r2_mean_per_sample)  # [1]
    r2_std = torch.std(r2_mean_per_sample)  # [1]

    return r2_above09, r2_mean, r2_std


def compute_metrics(predictions: torch.Tensor, ground_truth: torch.Tensor) -> dict:
    """
    Compute the metrics of a prediction given the ground truth.

    Metrics: (with x_t^i the ground truth, \\hat{x}_t^i the predicted value, D the number of dimensions, T the horizon length, \bar{x}^i the mean of the ground truth)
        MAE = 1/T \\sum_{t=1}^T 1/D \\sum_{i=1}^D | x_t^i - \\hat{x}_t^i |
        MSE = 1/T \\sum_{t=1}^T 1/D \\sum_{i=1}^D ( x_t^i - \\hat{x}_t^i)^2
        RMSE = \\sqrt{MSE}
        R2 = 1/D \\sum_{i=1}^D [ 1 - (\\sum_{t=1}^T ( x_t^i - \\hat{x}_t^i )^2)/(\\sum_{t=1}^{T} ( x_t^i - \bar{x}^i )^2) ]

    Args:
        predictions: values predicted by the model. shape: (horizon_length, n_dims)
        ground_truth: the true values. shape: (horizon_length, n_dims)

    Returns:
        a dictionary containing the metrics, averaged over all samples and the standard deviation.
    """
    if predictions.shape != ground_truth.shape:
        raise ValueError(
            f"prediction and ground_truth must have the same shape, got {predictions.shape} and {ground_truth.shape}"
        )
    r2_above09, r2_mean, r2_std = r2_score(predictions, ground_truth)

    mae_across_dims = torch.mean(torch.abs(ground_truth - predictions), axis=1)
    mse_across_dims = torch.mean((ground_truth - predictions) ** 2, axis=1)

    return {
        "r2_score_mean": r2_mean.item(),
        "r2_score_std": r2_std.item(),
        "r2_score_above0.9": r2_above09.item(),
        "mae_mean": torch.mean(mae_across_dims).item(),
        "mae_std": torch.std(mae_across_dims).item(),
        "mse_mean": torch.mean(mse_across_dims).item(),
        "mse_std": torch.std(mse_across_dims).item(),
        "rmse_mean": torch.mean(torch.sqrt(mse_across_dims)).item(),
        "rmse_std": torch.std(torch.sqrt(mse_across_dims)).item(),
    }