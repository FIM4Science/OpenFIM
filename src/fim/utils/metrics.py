"""metrics used throughout the project."""

import numpy as np


def r2_score_mean(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Compute the R2 score of a prediction. The score is normalized with respect to the mean of the ground truth.

    Note: only for  a single prediction window.

    Args:
        prediction: the predicted values
        ground_truth: the true values

    Returns:
        the R2 score, averaged over the dimensions
    """

    ground_truth_mean = np.mean(ground_truth, axis=0)
    ss_res = np.sum((ground_truth - prediction) ** 2, axis=0)
    ss_tot = np.sum((ground_truth - ground_truth_mean) ** 2, axis=0)
    r2 = 1 - ss_res / ss_tot

    return np.mean(r2)


def compute_metrics(predictions: np.ndarray, ground_truth: np.ndarray) -> dict:
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
        "mae": np.mean(np.mean(np.abs(ground_truth - predictions), axis=1)),
        "mse": np.mean(np.mean((ground_truth - predictions) ** 2, axis=1)),
    }
    metrics["rmse"] = np.sqrt(metrics["mse"])

    return metrics
