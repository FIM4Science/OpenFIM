"""
Next Event Prediction Script for FIM-Hawkes Models

Usage:
- Set USE_EASYTPP=True for HuggingFace datasets
- Set USE_EASYTPP=False and LOCAL_DATASET_PATH for local datasets
- Adjust CONTEXT_SIZE and INFERENCE_SIZE based on your memory constraints
"""

import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from safetensors import safe_open
from tqdm import tqdm


# ===================================================================
# SCRIPT CONFIGURATION
# ===================================================================
# Set the path to your trained FIM-Hawkes model checkpoint directory.
MODEL_CHECKPOINT_PATH = "/cephfs/users/berghaus/FoundationModels/FIM/results/FIM_Hawkes_1-3st_optimized_mixed_rmse_norm_2000_paths_mixed_250_events_mixed-experiment-seed-10-dataset-dataset_kwargs-field_name_for_dimension_grouping-base_intensity_functions_07-20-1027/checkpoints/epoch-89"

# Flag to control dataset source
# If True: Load from EasyTPP HuggingFace repository
# If False: Load from local path
USE_EASYTPP = False

# Sample index to use when loading local datasets (used only if USE_EASYTPP=False)
# Local datasets have shape [N_samples, P_processes, K_events, 1]
# This variable selects which of the N_samples to use (0-indexed)
SAMPLE_INDEX = 0

# Set the Hugging Face dataset identifier (used only if USE_EASYTPP=True).
DATASET_IDENTIFIER = "easytpp/retweet"

# Set the local dataset path (used only if USE_EASYTPP=False).
LOCAL_DATASET_PATH = "/cephfs/users/berghaus/FoundationModels/FIM/data/synthetic_data/hawkes/1k_3D_1k_paths_diag_only_old_params"

# Set the number of event types for the chosen dataset.
NUM_EVENT_TYPES = 3

# Maximum number of sequences from the training set to use as context.
CONTEXT_SIZE = 1000

# Number of sequences from the test set to use for inference.
INFERENCE_SIZE = 3
# ===================================================================


# --- FIM Hawkes Model Definition ---
try:
    from fim.models.hawkes import FIMHawkes, FIMHawkesConfig
except ImportError:
    print("Error: Could not import FIMHawkes. Please ensure the 'fim' library is in your PYTHONPATH.")
    exit(1)

# This part is crucial for loading the model correctly
FIMHawkesConfig.register_for_auto_class()
FIMHawkes.register_for_auto_class("AutoModel")


def load_local_dataset(dataset_path: str, split: str):
    """
    Load local dataset from PyTorch tensor files.

    Args:
        dataset_path: Path to the dataset directory
        split: Dataset split ("train", "val", or "test")

    Returns:
        Dictionary containing dataset in a format compatible with HuggingFace datasets
    """
    split_path = Path(dataset_path) / split
    if not split_path.exists():
        raise FileNotFoundError(f"Split directory not found: {split_path}")

    # Load tensor files
    event_times = torch.load(split_path / "event_times.pt")  # [N_samples, P_processes, K_events, 1]
    event_types = torch.load(split_path / "event_types.pt")  # [N_samples, P_processes, K_events, 1]
    seq_lengths = torch.load(split_path / "seq_lengths.pt")  # [N_samples, P_processes]

    # Load ground truth functions if they exist
    kernel_functions = None
    base_intensity_functions = None
    kernel_functions_path = split_path / "kernel_functions.pt"
    base_intensity_functions_path = split_path / "base_intensity_functions.pt"

    if kernel_functions_path.exists() and base_intensity_functions_path.exists():
        kernel_functions = torch.load(kernel_functions_path)  # [N_samples, M, M]
        base_intensity_functions = torch.load(base_intensity_functions_path)  # [N_samples, M]
        print(f"✅ Loaded ground truth functions for {split} split")
    else:
        print(f"⚠️  Ground truth functions not found for {split} split")

    # Select only the specified sample index instead of flattening all samples
    N, P, K = event_times.shape[:3]
    if SAMPLE_INDEX >= N:
        raise ValueError(f"SAMPLE_INDEX ({SAMPLE_INDEX}) is out of bounds. Dataset has {N} samples (0-indexed).")

    # Extract the selected sample
    event_times_sample = event_times[SAMPLE_INDEX]  # [P_processes, K_events, 1]
    event_types_sample = event_types[SAMPLE_INDEX]  # [P_processes, K_events, 1]
    seq_lengths_sample = seq_lengths[SAMPLE_INDEX]  # [P_processes]

    # Now work with processes dimension only (P instead of N*P)
    total_sequences = P

    # Squeeze the last dimension if it exists
    if event_times_sample.dim() == 3:
        event_times_flat = event_times_sample.squeeze(-1)  # [P_processes, K_events]
        event_types_flat = event_types_sample.squeeze(-1)  # [P_processes, K_events]
    else:
        event_times_flat = event_times_sample  # [P_processes, K_events]
        event_types_flat = event_types_sample  # [P_processes, K_events]

    seq_lengths_flat = seq_lengths_sample  # [P_processes]

    # Convert to list format compatible with HuggingFace datasets
    time_since_start = []
    time_since_last_event = []
    type_event = []
    seq_len = []

    for i in range(total_sequences):
        actual_len = seq_lengths_flat[i].item()

        # Extract actual sequence (up to actual_len)
        times = event_times_flat[i, :actual_len].tolist()
        types = event_types_flat[i, :actual_len].tolist()

        # Calculate time deltas
        deltas = [0.0] + [times[j] - times[j - 1] for j in range(1, len(times))]

        time_since_start.append(times)
        time_since_last_event.append(deltas)
        type_event.append(types)
        seq_len.append(actual_len)

    result = {
        "time_since_start": time_since_start,
        "time_since_last_event": time_since_last_event,
        "type_event": type_event,
        "seq_len": seq_len,
    }

    # Add ground truth functions if available
    if kernel_functions is not None and base_intensity_functions is not None:
        # Extract functions for the selected sample
        result["kernel_functions"] = kernel_functions[SAMPLE_INDEX : SAMPLE_INDEX + 1]  # Keep batch dimension [1, M, M]
        result["base_intensity_functions"] = base_intensity_functions[SAMPLE_INDEX : SAMPLE_INDEX + 1]  # Keep batch dimension [1, M]

    return result


def load_local_dataset_subset(dataset_dict, size):
    """
    Extract a subset from the loaded local dataset.

    Args:
        dataset_dict: Dictionary containing the full dataset
        size: Number of sequences to extract

    Returns:
        Dictionary containing the subset
    """
    if size >= len(dataset_dict["seq_len"]):
        return dataset_dict

    return {key: values[:size] for key, values in dataset_dict.items()}


def load_fimhawkes_with_proper_weights(checkpoint_path: str) -> FIMHawkes:
    """
    Load FIMHawkes model with all weights properly loaded.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

    config_path = checkpoint_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in checkpoint directory: {config_path}")
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    if "model_type" not in config_dict:
        config_dict["model_type"] = "fimhawkes"

    config = FIMHawkesConfig.from_dict(config_dict)
    model = FIMHawkes(config)

    safetensors_path = checkpoint_path / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"model.safetensors not found in checkpoint directory: {safetensors_path}")

    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        state_dict = {key: f.get_tensor(key) for key in f.keys()}

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    print("✅ Model loaded with custom weight loading.")
    if missing_keys:
        print(f"⚠️  Missing keys in state_dict: {len(missing_keys)}")
    if unexpected_keys:
        print(f"⚠️  Unexpected keys in state_dict: {len(unexpected_keys)}")

    return model


def predict_next_event_for_sequence(model, inference_sequence, context_batch, device):
    """
    Perform next-event prediction for a single inference sequence using a fixed context batch.
    """
    L = inference_sequence["time_seqs"].shape[1]
    seq_len = inference_sequence["seq_len"].item()

    all_paths_dtime_preds = []
    all_paths_type_preds = []

    current_path_dtime_preds = []
    current_path_type_preds = []

    if seq_len <= 1:
        pass
    else:
        for prefix_len in range(1, seq_len):
            context_times = context_batch["time_seqs"].unsqueeze(0).unsqueeze(-1).to(device)
            context_types = context_batch["type_seqs"].unsqueeze(0).unsqueeze(-1).to(device)
            context_lengths = context_batch["seq_non_pad_mask"].sum(dim=1).unsqueeze(0).to(device)

            inf_times_padded = torch.zeros(1, 1, L, 1, device=device)
            inf_types_padded = torch.zeros(1, 1, L, 1, device=device)

            prefix_times = inference_sequence["time_seqs"][0, :prefix_len]
            prefix_types = inference_sequence["type_seqs"][0, :prefix_len]

            inf_times_padded[0, 0, :prefix_len, 0] = prefix_times
            inf_types_padded[0, 0, :prefix_len, 0] = prefix_types

            # Pad *after* the prefix with +inf so that these indices are guaranteed
            # to be treated as "future" events by the intensity implementation and
            # never influence the sampler.
            inf_times_padded[0, 0, prefix_len:, 0] = float("inf")

            inf_lengths = torch.tensor([[prefix_len]], device=device)

            x = {
                "context_event_times": context_times,
                "context_event_types": context_types,
                "context_seq_lengths": context_lengths,
                "inference_event_times": inf_times_padded,
                "inference_event_types": inf_types_padded,
                "inference_seq_lengths": inf_lengths,
                "intensity_evaluation_times": torch.zeros(1, 1, 1, device=device),
            }

            model_out = model.forward(x)
            intensity_obj = model_out["intensity_function"]

            # Extract only the *real* history up to `prefix_len` so the sampler and
            # subsequent calculations ignore the +inf padding.
            hist_times_full = x["inference_event_times"].squeeze(0).squeeze(-1)
            hist_times = hist_times_full[:, :prefix_len]

            hist_dtimes = torch.zeros_like(hist_times)
            hist_dtimes[:, 1:] = hist_times[:, 1:] - hist_times[:, :-1]

            hist_types = x["inference_event_types"].squeeze(0).squeeze(-1)[:, :prefix_len]

            def intensity_fn_for_sampler(query_times, hist_ignored):
                intensity_per_mark = intensity_obj.evaluate(query_times)
                # The model's intensity function returns [B, M, P, T], but the sampler expects
                # [B, P, T, M]. We permute the dimensions to match.
                return intensity_per_mark.permute(0, 2, 3, 1)

            accepted_dtimes, weights = model.event_sampler.draw_next_time_one_step(
                time_seq=hist_times,
                time_delta_seq=hist_dtimes,
                event_seq=hist_types,
                intensity_fn=intensity_fn_for_sampler,
                compute_last_step_only=True,
            )

            # `accepted_dtimes` returned by the sampler are ABSOLUTE timestamps.  To obtain
            # the inter-event time we need to subtract the timestamp of the last observed
            # event (t_last).
            t_last_tensor = hist_times[:, -1:].unsqueeze(-1)  # Shape compatible with accepted_dtimes
            delta_samples = accepted_dtimes - t_last_tensor

            # Robustness: fall back to `dtime_max` if, due to numerical issues, any delta becomes negative.
            delta_samples = torch.clamp(delta_samples, min=0.0)

            # Expected next-event inter-arrival time
            dtime_pred = torch.sum(delta_samples * weights, dim=-1).squeeze()

            intensities_at_samples = intensity_obj.evaluate(accepted_dtimes)
            total_intensity_at_samples = intensities_at_samples.sum(dim=1, keepdim=True)
            probabilities_at_samples = intensities_at_samples / (total_intensity_at_samples + 1e-9)
            expected_probabilities = torch.sum(probabilities_at_samples * weights.unsqueeze(1), dim=-1)
            type_pred = torch.argmax(expected_probabilities.squeeze())

            current_path_dtime_preds.append(dtime_pred.item())
            current_path_type_preds.append(type_pred.item())

    num_preds = len(current_path_dtime_preds)
    padding_needed = (L - 1) - num_preds

    padded_dtimes = current_path_dtime_preds + [0.0] * padding_needed
    padded_types = current_path_type_preds + [0] * padding_needed

    all_paths_dtime_preds.append(padded_dtimes)
    all_paths_type_preds.append(padded_types)

    return {
        "predicted_event_dtimes": torch.tensor(all_paths_dtime_preds, dtype=torch.float32),
        "predicted_event_types": torch.tensor(all_paths_type_preds, dtype=torch.long),
    }


def predict_next_event_for_sequence_ground_truth(model, inference_sequence, context_batch, ground_truth_functions, device):
    """
    Perform next-event prediction for a single inference sequence using ground truth intensity functions.
    """
    L = inference_sequence["time_seqs"].shape[1]
    seq_len = inference_sequence["seq_len"].item()

    all_paths_dtime_preds = []
    all_paths_type_preds = []

    current_path_dtime_preds = []
    current_path_type_preds = []

    if seq_len <= 1:
        pass
    else:
        for prefix_len in range(1, seq_len):
            context_times = context_batch["time_seqs"].unsqueeze(0).unsqueeze(-1).to(device)
            context_lengths = context_batch["seq_non_pad_mask"].sum(dim=1).unsqueeze(0).to(device)

            inf_times_padded = torch.zeros(1, 1, L, 1, device=device)
            # Ensure integer type for event types right from creation
            inf_types_padded = torch.zeros(1, 1, L, 1, device=device, dtype=torch.long)

            prefix_times = inference_sequence["time_seqs"][0, :prefix_len]
            prefix_types = inference_sequence["type_seqs"][0, :prefix_len]

            inf_times_padded[0, 0, :prefix_len, 0] = prefix_times
            inf_types_padded[0, 0, :prefix_len, 0] = prefix_types

            # Pad *after* the prefix with +inf so that these indices are guaranteed
            # to be treated as "future" events by the intensity implementation and
            # never influence the sampler.
            inf_times_padded[0, 0, prefix_len:, 0] = float("inf")

            # The `x` dictionary is now just a convenient container for ground truth data.
            x = {
                "inference_event_times": inf_times_padded,
                "inference_event_types": inf_types_padded,
                "kernel_functions": ground_truth_functions["kernel_functions"].to(device),
                "base_intensity_functions": ground_truth_functions["base_intensity_functions"].to(device),
            }

            # Get ground truth intensity function from the computed target values
            kernel_functions_list, base_intensity_functions_list = model._decode_functions(
                x["kernel_functions"], x["base_intensity_functions"]
            )

            # Compute normalization constants (same as model does)
            norm_constants = torch.ones(1, device=device)
            if model.normalize_times:
                # Use same normalization as the model
                if model.normalize_by_max_time:
                    if context_times.numel() > 0 and context_lengths.numel() > 0:
                        batch_indices = torch.arange(1, device=device).view(-1, 1).expand(-1, context_times.size(1))
                        path_indices = torch.arange(context_times.size(1), device=device).view(1, -1).expand(1, -1)
                        # Ensure context_lengths is not empty and has the right dimensions
                        valid_lengths = context_lengths[context_lengths > 0]
                        if valid_lengths.numel() > 0:
                            max_times = context_times[batch_indices, path_indices, valid_lengths - 1]
                            if max_times.numel() > 0:
                                norm_constants = max_times.amax(dim=[1, 2])
                else:
                    if context_times.shape[2] > 1:
                        masked_delta_times = torch.diff(context_times, dim=2)
                        masked_delta_times = torch.cat([torch.zeros_like(masked_delta_times[:, :, :1]), masked_delta_times], dim=2)
                        norm_constants = masked_delta_times.amax(dim=[1, 2, 3])

            # Extract only the *real* history up to `prefix_len`
            hist_times_full = x["inference_event_times"].squeeze(0).squeeze(-1)
            hist_times = hist_times_full[:, :prefix_len]

            hist_dtimes = torch.zeros_like(hist_times)
            hist_dtimes[:, 1:] = hist_times[:, 1:] - hist_times[:, :-1]

            hist_types = x["inference_event_types"].squeeze(0).squeeze(-1)[:, :prefix_len]

            # Create ground truth intensity function for sampling
            def ground_truth_intensity_fn_for_sampler(query_times, hist_ignored):
                B_query, P_query, T_query = query_times.shape
                # Compute target intensity using ground truth functions
                target_intensity = model.compute_target_intensity_values(
                    kernel_functions_list,
                    base_intensity_functions_list,
                    query_times,  # [B, P, T_query]
                    x["inference_event_times"][:, :, :prefix_len, :],  # [B, P, prefix_len, 1]
                    x["inference_event_types"][:, :, :prefix_len, :],  # [B, P, prefix_len, 1]
                    torch.tensor([[prefix_len]], device=device),  # [B, P]
                    norm_constants,
                    num_marks=NUM_EVENT_TYPES,
                )
                # target_intensity shape: [B, M, P, T_query]
                # Sampler expects [B, P, T_query, M]
                return target_intensity.permute(0, 2, 3, 1)

            accepted_dtimes, weights = model.event_sampler.draw_next_time_one_step(
                time_seq=hist_times,
                time_delta_seq=hist_dtimes,
                event_seq=hist_types,
                intensity_fn=ground_truth_intensity_fn_for_sampler,
                compute_last_step_only=True,
            )

            # `accepted_dtimes` returned by the sampler are ABSOLUTE timestamps.  To obtain
            # the inter-event time we need to subtract the timestamp of the last observed
            # event (t_last).
            t_last_tensor = hist_times[:, -1:].unsqueeze(-1)  # Shape compatible with accepted_dtimes
            delta_samples = accepted_dtimes - t_last_tensor

            # Robustness: fall back to `dtime_max` if, due to numerical issues, any delta becomes negative.
            delta_samples = torch.clamp(delta_samples, min=0.0)

            # Expected next-event inter-arrival time
            dtime_pred = torch.sum(delta_samples * weights, dim=-1).squeeze()

            gt_intensities_at_samples = model.compute_target_intensity_values(
                kernel_functions_list,
                base_intensity_functions_list,
                accepted_dtimes,
                x["inference_event_times"][:, :, :prefix_len, :],
                x["inference_event_types"][:, :, :prefix_len, :],
                torch.tensor([[prefix_len]], device=device),
                norm_constants,
                num_marks=NUM_EVENT_TYPES,
            )
            total_gt_intensity = gt_intensities_at_samples.sum(dim=1, keepdim=True)
            gt_probabilities_at_samples = gt_intensities_at_samples / (total_gt_intensity + 1e-9)
            expected_gt_probabilities = torch.sum(gt_probabilities_at_samples * weights.unsqueeze(1), dim=-1)
            type_pred = torch.argmax(expected_gt_probabilities.squeeze())

            current_path_dtime_preds.append(dtime_pred.item())
            current_path_type_preds.append(type_pred.item())

    num_preds = len(current_path_dtime_preds)
    padding_needed = (L - 1) - num_preds

    padded_dtimes = current_path_dtime_preds + [0.0] * padding_needed
    padded_types = current_path_type_preds + [0] * padding_needed

    all_paths_dtime_preds.append(padded_dtimes)
    all_paths_type_preds.append(padded_types)

    return {
        "predicted_event_dtimes": torch.tensor(all_paths_dtime_preds, dtype=torch.float32),
        "predicted_event_types": torch.tensor(all_paths_type_preds, dtype=torch.long),
    }


def main():
    """
    Main function to load the model and dataset, and run the evaluation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the pre-trained FIM-Hawkes model
    print(f"Loading model from: {MODEL_CHECKPOINT_PATH}")
    model = load_fimhawkes_with_proper_weights(MODEL_CHECKPOINT_PATH)
    model.eval()
    model.to(device)
    print("Model loaded successfully.")

    # The sampler's estimate of the intensity's upper bound is critical.
    # A low sampling resolution (`num_samples_boundary`) can lead to a poor
    # estimate, causing high rejection rates. We increase it here for a more
    # robust bound.
    model.event_sampler.num_samples_boundary = 50
    print(f"Updated sampler num_samples_boundary to {model.event_sampler.num_samples_boundary}")

    # 2. Load context and inference data
    if USE_EASYTPP:
        print(f"Loading and preprocessing dataset from Hugging Face Hub: {DATASET_IDENTIFIER}...")
        train_dataset = load_dataset(DATASET_IDENTIFIER, split="train")
        test_dataset = load_dataset(DATASET_IDENTIFIER, split="test")

        # Select a subset for context and inference based on configuration
        context_data_raw = train_dataset[:CONTEXT_SIZE]
        inference_data_raw = test_dataset[:INFERENCE_SIZE]

        # Ground truth functions not available for HuggingFace datasets
        ground_truth_available = False
        ground_truth_functions = None
    else:
        print(f"Loading and preprocessing local dataset from: {LOCAL_DATASET_PATH}...")
        print(f"Using sample index: {SAMPLE_INDEX}")
        train_dataset_dict = load_local_dataset(LOCAL_DATASET_PATH, "train")
        test_dataset_dict = load_local_dataset(LOCAL_DATASET_PATH, "test")

        # Select a subset for context and inference based on configuration
        context_data_raw = load_local_dataset_subset(train_dataset_dict, CONTEXT_SIZE)
        inference_data_raw = load_local_dataset_subset(test_dataset_dict, INFERENCE_SIZE)

        # Check if ground truth functions are available
        ground_truth_available = "kernel_functions" in test_dataset_dict and "base_intensity_functions" in test_dataset_dict
        if ground_truth_available:
            ground_truth_functions = {
                "kernel_functions": test_dataset_dict["kernel_functions"],
                "base_intensity_functions": test_dataset_dict["base_intensity_functions"],
            }
            print("✅ Ground truth functions available - will compute baseline predictions")
        else:
            ground_truth_functions = None
            print("⚠️  Ground truth functions not available - model predictions only")

    # 3. Prepare the fixed context batch (on-the-fly)
    # This batch will be cached and reused for all inference sequences.
    print("Preparing fixed context data...")

    # ------------------------------------------------------------------
    # Adjust the sampler's dtime_max to match the time scale of the dataset and avoid artificial truncation.
    # ------------------------------------------------------------------

    if USE_EASYTPP:
        max_dtime_train = max(max(seq) for seq in train_dataset["time_since_last_event"]) if CONTEXT_SIZE > 0 else 1.0
    else:
        max_dtime_train = max(max(seq) for seq in train_dataset_dict["time_since_last_event"]) if CONTEXT_SIZE > 0 else 1.0

    sampler_cap = float(max_dtime_train) * 1.2  # 20 % safety margin
    model.event_sampler.dtime_max = sampler_cap
    print(f"Updated sampler dtime_max to {sampler_cap:.2f}")

    # Lower the over-sampling factor to avoid overly conservative
    # Oversampling factor influences the intensity upper bound.  A larger value makes
    # rejection sampling more likely to ACCEPT a draw (at the expense of a few more
    # evaluations). We therefore revert to a safer default of 5.0.
    model.event_sampler.over_sample_rate = 5.0
    print(f"Set sampler over_sample_rate to {model.event_sampler.over_sample_rate:.1f}")

    max_context_len = max(len(seq) for seq in context_data_raw["time_since_start"]) if CONTEXT_SIZE > 0 else 0

    context_batch = {
        "time_seqs": [],
        "type_seqs": [],
        "seq_len": [],
    }
    for i in range(len(context_data_raw["time_since_start"])):
        pad_len = max_context_len - len(context_data_raw["time_since_start"][i])
        context_batch["time_seqs"].append(context_data_raw["time_since_start"][i] + [0] * pad_len)
        context_batch["type_seqs"].append(context_data_raw["type_event"][i] + [0] * pad_len)
        context_batch["seq_len"].append(context_data_raw["seq_len"][i])

    # Convert context data to tensors and move to the correct device
    context_batch["time_seqs"] = torch.tensor(context_batch["time_seqs"], device=device)
    context_batch["type_seqs"] = torch.tensor(context_batch["type_seqs"], device=device)
    context_batch["seq_len"] = torch.tensor(context_batch["seq_len"], device=device)
    context_batch["seq_non_pad_mask"] = torch.arange(max_context_len, device=device).expand(CONTEXT_SIZE, max_context_len) < context_batch[
        "seq_len"
    ].unsqueeze(1)

    # Initialize metrics for model predictions
    total_mae, total_sq_err, total_acc, total_events = 0, 0, 0, 0

    # Initialize metrics for ground truth predictions if available
    if ground_truth_available:
        gt_total_mae, gt_total_sq_err, gt_total_acc, gt_total_events = 0, 0, 0, 0

    # 4. Loop through inference sequences and perform next-event prediction
    with torch.no_grad():
        for i in tqdm(range(INFERENCE_SIZE), desc="Evaluating Inference Sequences"):
            # a. Prepare a single inference sequence
            inference_item = {
                "time_seqs": torch.tensor([inference_data_raw["time_since_start"][i]], device=device),
                "time_delta_seqs": torch.tensor([inference_data_raw["time_since_last_event"][i]], device=device),
                "type_seqs": torch.tensor([inference_data_raw["type_event"][i]], device=device),
                "seq_len": torch.tensor([inference_data_raw["seq_len"][i]], device=device),
            }
            inf_max_len = inference_item["time_seqs"].shape[1]
            inference_item["seq_non_pad_mask"] = torch.arange(inf_max_len, device=device).expand(1, inf_max_len) < inference_item[
                "seq_len"
            ].unsqueeze(1)

            # b. Get model predictions
            predictions = predict_next_event_for_sequence(model, inference_item, context_batch, device)
            pred_dtimes = predictions["predicted_event_dtimes"].cpu()
            pred_types = predictions["predicted_event_types"].cpu()

            # c. Get ground truth predictions if available
            if ground_truth_available:
                gt_predictions = predict_next_event_for_sequence_ground_truth(
                    model, inference_item, context_batch, ground_truth_functions, device
                )
                gt_pred_dtimes = gt_predictions["predicted_event_dtimes"].cpu()
                gt_pred_types = gt_predictions["predicted_event_types"].cpu()

            # d. Get ground truth for comparison
            true_dtimes = inference_item["time_delta_seqs"].cpu()[:, 1:]
            true_types = inference_item["type_seqs"].cpu()[:, 1:]

            # e. Calculate metrics for the current sequence
            mask = torch.zeros_like(true_types, dtype=torch.bool)
            original_length = inference_item["seq_len"].item()
            if original_length > 1:
                mask[0, : original_length - 1] = True

            num_events_in_batch = mask.sum().item()
            if num_events_in_batch > 0:
                # Model metrics
                mae = torch.abs(pred_dtimes[mask] - true_dtimes[mask]).sum().item()
                sq_err = ((pred_dtimes[mask] - true_dtimes[mask]) ** 2).sum().item()
                acc = (pred_types[mask] == true_types[mask]).sum().item()

                total_mae += mae
                total_sq_err += sq_err
                total_acc += acc
                total_events += num_events_in_batch

                # Ground truth metrics
                if ground_truth_available:
                    gt_mae = torch.abs(gt_pred_dtimes[mask] - true_dtimes[mask]).sum().item()
                    gt_sq_err = ((gt_pred_dtimes[mask] - true_dtimes[mask]) ** 2).sum().item()
                    gt_acc = (gt_pred_types[mask] == true_types[mask]).sum().item()

                    gt_total_mae += gt_mae
                    gt_total_sq_err += gt_sq_err
                    gt_total_acc += gt_acc
                    gt_total_events += num_events_in_batch

    # 5. Report final results
    final_mae = total_mae / total_events if total_events > 0 else 0
    final_rmse = np.sqrt(total_sq_err / total_events) if total_events > 0 else 0
    final_acc = total_acc / total_events if total_events > 0 else 0

    print("\n" + "=" * 60)
    print("--- EVALUATION RESULTS ---")
    print(f"Dataset: {DATASET_IDENTIFIER if USE_EASYTPP else LOCAL_DATASET_PATH}")
    print(f"Total Events Evaluated: {total_events}")
    print()
    print("MODEL PREDICTIONS:")
    print(f"  Time Prediction MAE:       {final_mae:.4f}")
    print(f"  Time Prediction RMSE:      {final_rmse:.4f}")
    print(f"  Type Prediction Accuracy:  {final_acc:.4f}")

    if ground_truth_available:
        gt_final_mae = gt_total_mae / gt_total_events if gt_total_events > 0 else 0
        gt_final_rmse = np.sqrt(gt_total_sq_err / gt_total_events) if gt_total_events > 0 else 0
        gt_final_acc = gt_total_acc / gt_total_events if gt_total_events > 0 else 0

        print()
        print("GROUND TRUTH BASELINE:")
        print(f"  Time Prediction MAE:       {gt_final_mae:.4f}")
        print(f"  Time Prediction RMSE:      {gt_final_rmse:.4f}")
        print(f"  Type Prediction Accuracy:  {gt_final_acc:.4f}")

        print()
        print("COMPARISON (Model vs Ground Truth):")
        mae_improvement = ((gt_final_mae - final_mae) / gt_final_mae * 100) if gt_final_mae > 0 else 0
        rmse_improvement = ((gt_final_rmse - final_rmse) / gt_final_rmse * 100) if gt_final_rmse > 0 else 0
        acc_improvement = ((final_acc - gt_final_acc) / gt_final_acc * 100) if gt_final_acc > 0 else 0

        print(f"  MAE improvement:           {mae_improvement:+.1f}%")
        print(f"  RMSE improvement:          {rmse_improvement:+.1f}%")
        print(f"  Accuracy improvement:      {acc_improvement:+.1f}%")

    print("=" * 60)


if __name__ == "__main__":
    if "path/to/your" in MODEL_CHECKPOINT_PATH:
        print("=" * 60)
        print("!!! WARNING: Please update the MODEL_CHECKPOINT_PATH variable !!!")
        print(f"Current path is: {MODEL_CHECKPOINT_PATH}")
        print("=" * 60)
    else:
        main()
