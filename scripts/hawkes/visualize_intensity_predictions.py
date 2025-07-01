#!/usr/bin/env python
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from fim.models.hawkes import FIMHawkes


### Run via:
# CUDA_VISIBILE_DEVICES="" python scripts/hawkes/visualize_intensity_predictions.py --checkpoint "results/FIM_Hawkes_1-3st_optimized_mixed_rmse_norm_2000_paths_mixed_250_events_mixed-experiment-seed-10-dataset-dataset_kwargs-field_name_for_dimension_grouping-base_intensity_functions_06-30-1916/checkpoints/best-model"  --dataset "data/synthetic_data/hawkes/1k_2D_1k_paths_diag_only_old_params/test" --sample_idx 4 --path_idx 0


def load_data_from_dir(dir_path: Path) -> dict:
    """
    Load all .pt files in a directory and return as a dict of tensors.
    """
    tensors = {}
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dir_path}")
    for file in dir_path.iterdir():
        if file.suffix == ".pt":
            try:
                # Use map_location to ensure tensors are loaded to CPU
                tensors[file.stem] = torch.load(file, map_location="cpu")
            except Exception as e:
                print(f"Could not load {file}: {e}")
    return tensors


def create_evaluation_times(inference_event_times, inference_seq_lengths, num_eval_points=200):
    """
    Create evaluation times using only a dense grid for smooth plotting.

    IMPORTANT: This function intentionally does NOT include actual event times in the
    evaluation grid. Including event times would cause spurious correlations between
    different marks because:
    1. All marks would be evaluated exactly at event times
    2. The model's attention mechanism would cause cross-contamination
    3. All marks would show spikes at event times regardless of the actual event type

    Instead, we use a regular grid like the dataloader does during training, which
    allows each mark's intensity to be evaluated independently.

    Args:
        inference_event_times: Event times tensor [B, P_inference, L, 1]
        inference_seq_lengths: Sequence lengths [B, P_inference]
        num_eval_points: Number of evaluation points in the dense grid

    Returns:
        evaluation_times_batch: Regular grid of evaluation times [B, P_inference, num_eval_points]
    """
    B, P_inference, L, _ = inference_event_times.shape
    device = inference_event_times.device

    evaluation_times_batch = torch.zeros(B, P_inference, num_eval_points, device=device)

    for b in range(B):
        for p in range(P_inference):
            seq_len = inference_seq_lengths[b, p].item()
            if seq_len == 0:
                continue

            max_time = inference_event_times[b, p, seq_len - 1, 0].item()
            if max_time <= 0:
                continue

            # Create only a dense grid, do NOT include actual event times
            # This prevents cross-contamination between marks
            dense_grid = torch.linspace(0.0, max_time, num_eval_points, device=device)
            evaluation_times_batch[b, p, :] = dense_grid

    return evaluation_times_batch


def prepare_batch_for_model(data_sample, inference_path_idx=0, num_eval_points=200):
    """
    Prepare a single data sample for the model by splitting into context and inference paths.
    Uses all paths except the specified inference path for context.
    """
    # FIX: Ensure all tensors have a batch dimension of 1 before processing.
    # This resolves the IndexError for 1D tensors like seq_lengths.
    for key, value in data_sample.items():
        if torch.is_tensor(value):
            data_sample[key] = value.unsqueeze(0)

    # Get total number of paths
    total_paths = data_sample["event_times"].shape[1]

    if total_paths < 2:
        raise ValueError(f"Need at least 2 paths in sample ({total_paths}) to have both context and inference paths.")

    # Validate inference path index
    if inference_path_idx >= total_paths:
        print(f"Warning: inference_path_idx {inference_path_idx} >= total paths {total_paths}. Using path 0.")
        inference_path_idx = 0

    print(f"Total paths: {total_paths}, using path {inference_path_idx} for inference, remaining {total_paths - 1} for context")

    # Create masks for context and inference paths
    all_path_indices = torch.arange(total_paths)
    context_mask = all_path_indices != inference_path_idx
    inference_mask = all_path_indices == inference_path_idx

    context_indices = all_path_indices[context_mask]
    inference_indices = all_path_indices[inference_mask]

    # Split into context and inference paths
    event_times = data_sample["event_times"]
    event_types = data_sample["event_types"]
    seq_lengths = data_sample.get("seq_lengths")
    if seq_lengths is None:
        B, P, L, _ = event_times.shape
        seq_lengths = torch.full((B, P), L, device=event_times.device)

    model_data = {
        "context_event_times": event_times[:, context_indices],
        "context_event_types": event_types[:, context_indices],
        "context_seq_lengths": seq_lengths[:, context_indices],
        "inference_event_times": event_times[:, inference_indices],
        "inference_event_types": event_types[:, inference_indices],
        "inference_seq_lengths": seq_lengths[:, inference_indices],
    }

    model_data["intensity_evaluation_times"] = create_evaluation_times(
        model_data["inference_event_times"], model_data["inference_seq_lengths"], num_eval_points=num_eval_points
    )

    for key in ["kernel_functions", "base_intensity_functions"]:
        if key in data_sample:
            model_data[key] = data_sample[key]

    if "base_intensity_functions" in model_data:
        model_data["num_marks"] = model_data["base_intensity_functions"].shape[1]
    else:
        model_data["num_marks"] = 1

    return model_data


def plot_intensity_comparison(model_output, model_data, save_path="intensity_comparison.png", path_idx=0):
    """
    Create vertically stacked plots comparing predicted and ground truth intensities.
    """
    predicted_intensities = model_output["predicted_intensity_values"].detach().cpu().numpy()

    if "target_intensity_values" in model_output:
        target_intensities = model_output["target_intensity_values"].detach().cpu().numpy()
    else:
        target_intensities = None

    evaluation_times = model_data["intensity_evaluation_times"].detach().cpu().numpy()
    inference_event_times = model_data["inference_event_times"].detach().cpu().numpy()
    inference_event_types = model_data["inference_event_types"].detach().cpu().numpy()
    inference_seq_lengths = model_data["inference_seq_lengths"].detach().cpu().numpy()

    B, M, P_inference, _ = predicted_intensities.shape

    b = 0  # Always use first batch
    p = path_idx  # Use specified path index

    # Validate path index
    if p >= P_inference:
        print(f"Warning: path_idx {p} >= number of inference paths {P_inference}. Using path 0.")
        p = 0

    fig, axes = plt.subplots(M, 1, figsize=(15, 5 * M), sharex=True)
    if M == 1:
        axes = [axes]  # Ensure axes is always a list/array

    # Get all event times for the inference path to be plotted
    seq_len = inference_seq_lengths[b, p]
    all_event_times = inference_event_times[b, p, :seq_len, 0]
    all_event_types = inference_event_types[b, p, :seq_len, 0]

    for m in range(M):
        ax = axes[m]

        eval_times_p = evaluation_times[b, p]
        pred_intensity_p_m = predicted_intensities[b, m, p]

        # Filter out padded zeros and ensure proper alignment
        valid_mask = eval_times_p > 0
        sort_indices = None

        if valid_mask.any():
            eval_times_p = eval_times_p[valid_mask]
            pred_intensity_p_m = pred_intensity_p_m[valid_mask]

            # Ensure times are sorted and sort intensities accordingly
            sort_indices = np.argsort(eval_times_p)
            eval_times_p = eval_times_p[sort_indices]
            pred_intensity_p_m = pred_intensity_p_m[sort_indices]
        else:
            # If no valid times, create minimal plot
            eval_times_p = np.array([0.0])
            pred_intensity_p_m = np.array([0.0])

        ax.plot(eval_times_p, pred_intensity_p_m, "b-", linewidth=1.5, label="Predicted Intensity", alpha=0.9)

        if target_intensities is not None:
            target_intensity_p_m = target_intensities[b, m, p]

            # Apply the same filtering and sorting to target intensities
            if valid_mask.any() and sort_indices is not None:
                target_intensity_p_m = target_intensity_p_m[valid_mask]
                target_intensity_p_m = target_intensity_p_m[sort_indices]
            else:
                target_intensity_p_m = np.array([0.0])

            ax.plot(eval_times_p, target_intensity_p_m, "r--", linewidth=1.5, label="Ground Truth Intensity", alpha=0.9)

        # Plot event markers on the x-axis
        y_min, y_max = ax.get_ylim()
        marker_y_pos = y_min - 0.05 * (y_max - y_min)  # Place markers slightly below the axis

        # Events of the current mark
        events_this_mark = all_event_times[all_event_types == m]
        if len(events_this_mark) > 0:
            ax.plot(
                events_this_mark,
                np.full_like(events_this_mark, marker_y_pos),
                "o",
                color="green",
                markersize=6,
                label=f"Events (Mark {m})",
                clip_on=False,
                zorder=10,
            )

        # Events of other marks
        events_other_marks = all_event_times[all_event_types != m]
        if len(events_other_marks) > 0:
            ax.plot(
                events_other_marks,
                np.full_like(events_other_marks, marker_y_pos),
                "x",
                color="gray",
                markersize=5,
                alpha=0.7,
                label="Events (Other Marks)",
                clip_on=False,
                zorder=5,
            )

        ax.set_ylabel("Intensity")
        ax.set_title(f"Intensity Function for Mark {m}")

        # Create a clean legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        ax.set_ylim(bottom=0)

    axes[-1].set_xlabel("Time")
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])  # Adjust for suptitle

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Intensity comparison plot saved to {save_path}")


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    device = torch.device("cpu")

    model_checkpoint = Path(args.checkpoint)
    dataset_dir = Path(args.dataset)

    print(f"Loading model from: {model_checkpoint}")
    print(f"Loading dataset from: {dataset_dir}")

    model = FIMHawkes.from_pretrained(model_checkpoint)
    model.eval()
    model.to(device)

    data = load_data_from_dir(dataset_dir)
    if not data:
        print("No data loaded. Exiting.")
        return

    print(f"Loaded data with keys: {list(data.keys())}")

    # Use the specified sample index
    sample_idx = args.sample_idx

    # Validate sample index
    sample_count = None
    for key, value in data.items():
        if torch.is_tensor(value):
            sample_count = value.shape[0]
            break

    if sample_count is not None and sample_idx >= sample_count:
        print(f"Warning: sample_idx {sample_idx} >= number of samples {sample_count}. Using sample 0.")
        sample_idx = 0

    print(f"Using sample index: {sample_idx}")

    single_sample_data = {}
    for key, value in data.items():
        if torch.is_tensor(value):
            single_sample_data[key] = value[sample_idx]
        else:
            # Handle non-tensor data if necessary, e.g., lists of tensors
            single_sample_data[key] = value[sample_idx]

    try:
        model_data = prepare_batch_for_model(single_sample_data, args.path_idx)
    except ValueError as e:
        print(f"Error preparing batch: {e}")
        return

    print("Model input shapes:")
    for key, value in model_data.items():
        if torch.is_tensor(value):
            print(f"  {key}: {value.shape}")

    print(f"Using path index: {args.path_idx}")

    with torch.no_grad():
        model_output = model(model_data)

    print(f"Model output keys: {list(model_output.keys())}")

    save_path = "intensity_comparison.png"
    plot_intensity_comparison(model_output, model_data, save_path, path_idx=args.path_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Hawkes Process Intensity Functions.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/FIM_Hawkes_1-3st_optimized_mixed_rmse_norm_2000_paths_mixed_250_events_mixed-experiment-seed-10-dataset-dataset_kwargs-field_name_for_dimension_grouping-base_intensity_functions_06-30-1453/checkpoints/best-model",
        help="Path to the model checkpoint directory.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/synthetic_data/hawkes/1k_2D_1k_paths_diag_only_old_params/test",
        help="Path to the validation/test dataset directory.",
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Index of the sample to visualize from the dataset (default: 0).",
    )
    parser.add_argument(
        "--path_idx",
        type=int,
        default=0,
        help="Index of the inference path to visualize (default: 0).",
    )
    args = parser.parse_args()
    main(args)
