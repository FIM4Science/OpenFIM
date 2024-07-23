"""metrics used throughout the project."""

import torch


def r2_score_mean(prediction: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    """
    Compute the R2 score of a prediction. The score is normalized with respect to the mean of the ground truth.

    Note: only for a single prediction window.

    Args:
        prediction: the predicted values. Shape [B, T, D]
        ground_truth: the true values. Shape [B, T, D]

    Returns:
        the R2 score, averaged over the dimensions. Shape [B]
    """
    ground_truth_mean = torch.mean(ground_truth, axis=1)  #  [B, D]
    ss_res = torch.sum((ground_truth - prediction) ** 2, axis=1)  # [B, D]
    ss_tot = torch.sum((ground_truth - ground_truth_mean.unsqueeze(1)) ** 2, axis=1)  # [B, D]
    r2 = 1 - ss_res / ss_tot  # [B, D]

    return torch.mean(r2, axis=1)  # [B]


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
        a dictionary containing the metrics, each averaged over the dimensions
    """
    if predictions.shape != ground_truth.shape:
        raise ValueError(
            f"prediction and ground_truth must have the same shape, got {predictions.shape} and {ground_truth.shape}"
        )

    metrics = {
        "r2_score": r2_score_mean(predictions, ground_truth),
        "mae": torch.mean(torch.abs(ground_truth - predictions), axis=1).flatten(),
        "mse": torch.mean((ground_truth - predictions) ** 2, axis=1).flatten(),
    }
    metrics["rmse"] = torch.sqrt(metrics["mse"])

    return metrics
