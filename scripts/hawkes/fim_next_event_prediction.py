"""
Next Event Prediction Script for FIM-Hawkes Models

Usage:
- Set USE_EASYTPP=True for HuggingFace datasets
- Set USE_EASYTPP=False and LOCAL_DATASET_PATH for local datasets
- Adjust CONTEXT_SIZE and INFERENCE_SIZE based on your memory constraints
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm


# ===================================================================
# SCRIPT CONFIGURATION
# ===================================================================
# Set the path to your trained FIM-Hawkes model checkpoint directory.
MODEL_CHECKPOINT_PATH = "results/FIM_Hawkes_10-22st_2000_paths_mixed_100_events_mixed-experiment-seed-10-dataset-dataset_kwargs-field_name_for_dimension_grouping-base_intensity_functions_08-19-1612/checkpoints/epoch-25"

# Flag to control dataset source
# If True: Load from EasyTPP HuggingFace repository
# If False: Load from local path
USE_EASYTPP = True

# Set the Hugging Face dataset identifier (used only if USE_EASYTPP=True).
DATASET_IDENTIFIER = "easytpp/taxi"

# Sample index to use when loading local datasets (used only if USE_EASYTPP=False)
# Local datasets have shape [N_samples, P_processes, K_events, 1]
# This variable selects which of the N_samples to use (0-indexed)
SAMPLE_INDEX = 0

# Set the local dataset path (used only if USE_EASYTPP=False).
LOCAL_DATASET_PATH = "data/synthetic_data/hawkes/EVAL_10D_2k_context_paths_100_inference_paths_const_base_exp_kernel_no_interactions"

# Maximum number of sequences from the training set to use as context.
CONTEXT_SIZE = None

# Number of sequences from the test set to use for inference.
INFERENCE_SIZE = None

# Number of points to use for log-likelihood evaluation
NUM_INTEGRATION_POINTS = 5000

# Only consider paths up to this length
MAX_NUM_EVENTS = 100

PLOT_INTENSITY_PREDICTIONS = True

PLOT_EVENT_PREDICTIONS = True
# Index of the inference trajectory/path to visualize (used by both event and intensity plots)
PLOT_PATH_INDEX = 0

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
        split: Dataset split ("context", "val", or "test")

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

    # Load ground truth functions if they exist, and mandatory time offsets
    kernel_functions = None
    base_intensity_functions = None
    kernel_functions_path = split_path / "kernel_functions.pt"
    base_intensity_functions_path = split_path / "base_intensity_functions.pt"
    time_offsets_path = split_path / "time_offsets.pt"
    time_offsets = torch.load(time_offsets_path)  # [N_samples, P]

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
    time_offsets_sample = time_offsets[SAMPLE_INDEX]  # [P_processes]

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

    # Add time offsets (per-path tensor [P])
    result["time_offsets"] = time_offsets_sample

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


def detect_num_event_types_from_data(context_data_raw: dict, inference_data_raw: dict) -> int:
    """
    Detect number of unique event types by scanning the loaded data's type sequences.

    Works for both locally loaded datasets (converted to lists) and HF slices
    (dict of lists).
    """
    unique_types = set()
    for seq in context_data_raw.get("type_event", []):
        unique_types.update(seq)
    for seq in inference_data_raw.get("type_event", []):
        unique_types.update(seq)
    if not unique_types:
        # Fallback to 1 to avoid zero marks; caller may override if model config is known
        return 1
    return int(len(unique_types))


def load_fimhawkes_with_proper_weights(checkpoint_path: str) -> FIMHawkes:
    """
    Load FIMHawkes using the generic AModel loader, which reads config.json
    and model-checkpoint.pth from the checkpoint directory.
    """
    return FIMHawkes.load_model(Path(checkpoint_path))


def predict_next_event_for_sequence(
    model, inference_sequence, context_batch, device, precomputed_enhanced_context=None, num_marks: Optional[int] = None
):
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

            prefix_times = inference_sequence["time_seqs"][0, :prefix_len]
            prefix_types = inference_sequence["type_seqs"][0, :prefix_len]

            # Build exact-length 4D tensors (no padding)
            inf_times = prefix_times.view(1, 1, prefix_len, 1).to(device)
            inf_types = prefix_types.to(dtype=torch.long).view(1, 1, prefix_len, 1).to(device)
            inf_lengths = torch.tensor([[prefix_len]], device=device)

            x = {
                "context_event_times": context_times,
                "context_event_types": context_types,
                "context_seq_lengths": context_lengths,
                "inference_event_times": inf_times,
                "inference_event_types": inf_types,
                "inference_seq_lengths": inf_lengths,
                "intensity_evaluation_times": torch.zeros(1, 1, 1, device=device),
            }
            if precomputed_enhanced_context is not None:
                x["precomputed_enhanced_context"] = precomputed_enhanced_context
            if num_marks is not None:
                x["num_marks"] = torch.tensor([num_marks], device=device)

            # Some model.forward implementations compute NLL by default and expect
            # fully padded shapes. For pure inference, temporarily disable NLL.
            _orig_nll = getattr(model, "_nll_loss", None)
            try:
                if _orig_nll is not None:

                    def _zero_nll(**kwargs):
                        return torch.tensor(0.0, device=device)

                    model._nll_loss = lambda *args, **kwargs: _zero_nll(**kwargs)
                model_out = model.forward(x)
            finally:
                if _orig_nll is not None:
                    model._nll_loss = _orig_nll
            intensity_obj = model_out["intensity_function"]

            # Extract only the *real* history up to `prefix_len` so the sampler and
            # subsequent calculations ignore the +inf padding.
            hist_times = x["inference_event_times"].squeeze(0).squeeze(-1)

            hist_dtimes = torch.zeros_like(hist_times)
            hist_dtimes[:, 1:] = hist_times[:, 1:] - hist_times[:, :-1]

            hist_types = x["inference_event_types"].squeeze(0).squeeze(-1)

            def intensity_fn_for_sampler(query_times, hist_ignored):
                intensity_per_mark = intensity_obj.evaluate(query_times)
                # The model's intensity function returns [B, M, P, T], but the sampler expects
                # [B, P, T, M]. We permute the dimensions to match.
                return intensity_per_mark.permute(0, 2, 3, 1)

            if getattr(model.event_sampler, "sampling_method", "thinning") == "inverse_transform":
                accepted_dtimes, weights = model.event_sampler.draw_next_time_one_step_inverse_transform(
                    intensity_obj, compute_last_step_only=True
                )
            else:
                accepted_dtimes, weights = model.event_sampler.draw_next_time_one_step(
                    time_seq=hist_times,
                    time_delta_seq=hist_dtimes,
                    event_seq=hist_types,
                    intensity_fn=intensity_fn_for_sampler,
                    compute_last_step_only=True,
                )

            # `accepted_dtimes` returned by the sampler are ABSOLUTE timestamps. To obtain
            # the inter-event time we need to subtract the timestamp of the last observed
            # event (t_last).
            t_last_tensor = hist_times[:, -1:].unsqueeze(-1)
            raw_delta_samples = accepted_dtimes - t_last_tensor
            # Warn if sampler fell back to its max dtime
            dtime_max = model.event_sampler.dtime_max
            if (raw_delta_samples == dtime_max).any():
                print(f"⚠️  Fallback to max dtime (dtime_max={dtime_max:.2f}) at prefix length {prefix_len}")
            # Robustness: fall back to `dtime_max` if, due to numerical issues, any delta becomes negative.
            delta_samples = torch.clamp(raw_delta_samples, min=0.0)

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


def predict_next_event_for_sequence_ground_truth(model, inference_sequence, context_batch, ground_truth_functions, device, num_marks: int):
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
                    num_marks=num_marks,
                    inference_time_offsets=inference_sequence["time_offset_tensor"],
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
            raw_delta_samples = accepted_dtimes - t_last_tensor
            # Warn if sampler fell back to its max dtime
            dtime_max = model.event_sampler.dtime_max
            if (raw_delta_samples == dtime_max).any():
                print(f"⚠️  Fallback to max dtime (dtime_max={dtime_max:.2f}) at prefix length {prefix_len}")
            # Robustness: fall back to `dtime_max` if, due to numerical issues, any delta becomes negative.
            delta_samples = torch.clamp(raw_delta_samples, min=0.0)

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
                num_marks=num_marks,
                inference_time_offsets=(
                    inference_sequence.get("time_offset_tensor") if inference_sequence.get("time_offset_tensor") is not None else None
                ),
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


class GroundTruthIntensity:
    def __init__(self, model, inference_sequence, context_batch, ground_truth_functions, num_marks, device):
        self.model = model
        self.device = device
        self.num_marks = num_marks

        # Combine context and inference data for normalization calculation
        context_times = context_batch["time_seqs"].unsqueeze(0).unsqueeze(-1).to(self.device)
        context_lengths = context_batch["seq_non_pad_mask"].sum(dim=1).unsqueeze(0).to(self.device)

        self.x = {
            "inference_event_times": inference_sequence["time_seqs"].unsqueeze(0).unsqueeze(-1).to(self.device),
            "inference_event_types": inference_sequence["type_seqs"].unsqueeze(0).unsqueeze(-1).to(self.device),
            "inference_seq_lengths": inference_sequence["seq_len"].unsqueeze(0).to(self.device),
            "kernel_functions": ground_truth_functions["kernel_functions"].to(self.device),
            "base_intensity_functions": ground_truth_functions["base_intensity_functions"].to(self.device),
        }
        # Optional per-path time offset, shape expected by model: [B, P]
        if "time_offset_tensor" in inference_sequence:
            self.x["inference_time_offsets"] = inference_sequence["time_offset_tensor"].to(self.device)

        self.kernel_functions_list, self.base_intensity_functions_list = model._decode_functions(
            self.x["kernel_functions"], self.x["base_intensity_functions"]
        )

        # Compute normalization constants consistently with the model's forward pass
        self.norm_constants = torch.ones(1, device=self.device)
        if self.model.normalize_times:
            if self.model.normalize_by_max_time:
                if context_times.numel() > 0 and context_lengths.numel() > 0:
                    batch_indices = torch.arange(1, device=self.device).view(-1, 1).expand(-1, context_times.size(1))
                    path_indices = torch.arange(context_times.size(1), device=self.device).view(1, -1).expand(1, -1)
                    # Ensure context_lengths is not empty and has the right dimensions
                    valid_lengths = context_lengths[context_lengths > 0]
                    if valid_lengths.numel() > 0:
                        max_times = context_times[batch_indices, path_indices, valid_lengths - 1]
                        if max_times.numel() > 0:
                            self.norm_constants = max_times.amax(dim=[1, 2])
            else:
                if context_times.shape[2] > 1:
                    masked_delta_times = torch.diff(context_times, dim=2)
                    masked_delta_times = torch.cat([torch.zeros_like(masked_delta_times[:, :, :1]), masked_delta_times], dim=2)
                    self.norm_constants = masked_delta_times.amax(dim=[1, 2, 3])

        # Normalize inference times for GT calculation if model does
        if self.model.normalize_times:
            self.x["inference_event_times_norm"] = self.x["inference_event_times"] / self.norm_constants.view(-1, 1, 1, 1)
        else:
            self.x["inference_event_times_norm"] = self.x["inference_event_times"]

    def evaluate(self, query_times, normalized_times=True):
        return self.model.compute_target_intensity_values(
            self.kernel_functions_list,
            self.base_intensity_functions_list,
            query_times,
            self.x["inference_event_times_norm"],
            self.x["inference_event_types"],
            self.x["inference_seq_lengths"],
            self.norm_constants,
            self.num_marks,
            self.x.get("inference_time_offsets", None),
        )

    def integral(self, t_end, t_start=None, num_samples: int = 100, normalized_times: bool = False):
        # Reshape t_end from [B, P] to [B, P, 1] for compatibility
        t_end_reshaped = t_end.unsqueeze(-1)

        integral_at_tend = self.model.compute_target_integrated_intensity(
            self.kernel_functions_list,
            self.base_intensity_functions_list,
            t_end_reshaped,
            self.x["inference_event_times_norm"],
            self.x["inference_event_types"],
            self.x["inference_seq_lengths"],
            self.norm_constants,
            self.num_marks,
            num_samples=num_samples,
            inference_time_offsets=self.x.get("inference_time_offsets", None),
        )
        # This function computes integral from 0 to t_end. _nll_loss provides t_start=0.
        # The result will be [B, M, P, 1], we need to squeeze it for the loss function.
        return integral_at_tend.squeeze(-1)


def compute_nll(
    model,
    inference_sequence,
    context_batch,
    device,
    ground_truth_functions=None,
    precomputed_enhanced_context=None,
    num_marks: Optional[int] = None,
):
    """
    Compute the Negative Log-Likelihood for a given sequence for the model and ground truth.
    """
    results = {}

    # --- Model NLL ---
    context_times = context_batch["time_seqs"].unsqueeze(0).unsqueeze(-1).to(device)
    context_types = context_batch["type_seqs"].unsqueeze(0).unsqueeze(-1).to(device)
    context_lengths = context_batch["seq_non_pad_mask"].sum(dim=1).unsqueeze(0).to(device)

    inf_times = inference_sequence["time_seqs"].unsqueeze(0).unsqueeze(-1).to(device)
    inf_types = inference_sequence["type_seqs"].unsqueeze(0).unsqueeze(-1).to(device)
    inf_lengths = inference_sequence["seq_len"].unsqueeze(0).to(device)

    x = {
        "context_event_times": context_times,
        "context_event_types": context_types,
        "context_seq_lengths": context_lengths,
        "inference_event_times": inf_times,
        "inference_event_types": inf_types,
        "inference_seq_lengths": inf_lengths,
        "intensity_evaluation_times": torch.zeros(1, 1, 1, device=device),  # Dummy
    }
    if precomputed_enhanced_context is not None:
        x["precomputed_enhanced_context"] = precomputed_enhanced_context
    if num_marks is not None:
        x["num_marks"] = torch.tensor([num_marks], device=device)
    # Provide offsets to model.forward for target intensity computation only if present (local datasets)
    if "time_offset_tensor" in inference_sequence:
        x["inference_time_offsets"] = inference_sequence["time_offset_tensor"].to(device)

    with torch.no_grad():
        model_out = model.forward(x)

    intensity_obj = model_out["intensity_function"]

    event_times_for_nll = x["inference_event_times_norm"] if model.normalize_times else x["inference_event_times"]

    model_nll = model._nll_loss(
        intensity_fn=intensity_obj,
        event_times=event_times_for_nll.squeeze(-1),
        event_types=inf_types.squeeze(-1),
        seq_lengths=inf_lengths,
    ).item()
    results["model_nll"] = model_nll

    # --- Ground Truth NLL ---
    if ground_truth_functions:
        gt_intensity_obj = GroundTruthIntensity(
            model, inference_sequence, context_batch, ground_truth_functions, model.config.max_num_marks, device
        )

        # Use normalized times for GT NLL calculation if model normalizes
        event_times_for_gt_nll = gt_intensity_obj.x["inference_event_times_norm"]

        gt_nll = model._nll_loss(
            intensity_fn=gt_intensity_obj,
            event_times=event_times_for_gt_nll.squeeze(-1),
            event_types=inf_types.squeeze(-1),
            seq_lengths=inf_lengths,
        ).item()
        results["gt_nll"] = gt_nll

    return results


def compute_baseline_predictions(context_data_raw, inference_data_raw):
    """
    Compute simple baseline predictions using majority mark and mean time.

    Args:
        context_data_raw: Raw context data dictionary
        inference_data_raw: Raw inference data dictionary

    Returns:
        Dictionary containing baseline predictions for all inference sequences
    """
    # Compute majority mark from context data
    all_types = []
    all_dtimes = []

    for i in range(len(context_data_raw["time_since_start"])):
        seq_len = context_data_raw["seq_len"][i]
        # Skip first event (no previous event to predict time from)
        if seq_len > 1:
            all_types.extend(context_data_raw["type_event"][i][1:seq_len])
            all_dtimes.extend(context_data_raw["time_since_last_event"][i][1:seq_len])

    if not all_types:
        # Fallback if no data
        majority_type = 0
        mean_dtime = 1.0
    else:
        # Find majority type
        type_counts = {}
        for t in all_types:
            type_counts[t] = type_counts.get(t, 0) + 1
        majority_type = max(type_counts, key=type_counts.get)

        # Compute mean inter-event time
        mean_dtime = sum(all_dtimes) / len(all_dtimes)

    print("Baseline statistics from context data:")
    print(f"  Majority event type: {majority_type}")
    print(f"  Mean inter-event time: {mean_dtime:.4f}")

    # Generate baseline predictions for all inference sequences
    baseline_dtime_preds = []
    baseline_type_preds = []

    # Determine global maximum number of predictions (skip first event) across sequences
    global_max_pred_len = max(len(seq) for seq in inference_data_raw["time_since_start"]) - 1

    for i in range(len(inference_data_raw["time_since_start"])):
        seq_len = inference_data_raw["seq_len"][i]

        # Create predictions for this sequence (skip first event)
        seq_dtime_preds = [mean_dtime] * max(0, seq_len - 1)
        seq_type_preds = [majority_type] * max(0, seq_len - 1)

        # Pad to match global expected prediction length
        if len(seq_dtime_preds) < global_max_pred_len:
            pad_len = global_max_pred_len - len(seq_dtime_preds)
            seq_dtime_preds.extend([0.0] * pad_len)
            seq_type_preds.extend([0] * pad_len)

        baseline_dtime_preds.append(seq_dtime_preds)
        baseline_type_preds.append(seq_type_preds)

    return {
        "predicted_event_dtimes": torch.tensor(baseline_dtime_preds, dtype=torch.float32),
        "predicted_event_types": torch.tensor(baseline_type_preds, dtype=torch.long),
        "majority_type": majority_type,
        "mean_dtime": mean_dtime,
    }


def _bootstrap_ci(samples: np.ndarray, stat_fn, num_samples: int = 1000, alpha: float = 0.05):
    """
    Compute percentile bootstrap confidence interval for a statistic.

    Args:
        samples: 1D array-like of observations
        stat_fn: Callable that maps a 1D array to a scalar statistic
        num_samples: Number of bootstrap resamples
        alpha: 1 - confidence level (0.05 => 95% CI)

    Returns:
        (lower, upper) tuple or (None, None) if samples are empty
    """
    arr = np.asarray(samples)
    n = arr.shape[0]
    if n == 0:
        return None, None
    rng = np.random.default_rng()
    stats = np.empty(num_samples, dtype=float)
    for i in range(num_samples):
        idx = rng.integers(0, n, size=n)
        stats[i] = float(stat_fn(arr[idx]))
    lower = float(np.quantile(stats, alpha / 2))
    upper = float(np.quantile(stats, 1 - alpha / 2))
    return lower, upper


def run_next_event_evaluation(
    model_checkpoint_path: str,
    dataset: str,
    context_size: Optional[int] = None,
    inference_size: Optional[int] = None,
    max_num_events: Optional[int] = 100,
    sample_index: int = 0,
    num_integration_points: int = 5000,
    plot_intensity_predictions: bool = False,
    num_bootstrap_samples: int = 1000,
    sampling_method: Optional[str] = None,
    nll_method: Optional[str] = None,
):
    """
    Programmatic entry point for evaluating next-event prediction on a dataset.

    Returns a dictionary with aggregate metrics for the model, naive baseline,
    and ground truth (if available), plus bookkeeping fields.
    """
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_fimhawkes_with_proper_weights(model_checkpoint_path)
    # Optional: override NLL method from caller
    if nll_method in ("closed_form", "monte_carlo"):
        try:
            nll_cfg = getattr(model.config, "nll", None)
            if isinstance(nll_cfg, dict):
                nll_cfg["method"] = nll_method
            elif nll_cfg is None:
                model.config.nll = {"method": nll_method}
            else:
                setattr(nll_cfg, "method", nll_method)
        except Exception:
            try:
                model.config.nll = {"method": nll_method}
            except Exception:
                pass
    model.eval()
    model.to(device)

    # Make sampler more robust (match defaults used in main)
    model.event_sampler.num_samples_boundary = 50
    if sampling_method in ("thinning", "inverse_transform"):
        model.event_sampler.sampling_method = sampling_method

    # Always use Hugging Face EasyTPP datasets for next-event prediction.
    # Coerce short names like "amazon" to "easytpp/amazon".
    dataset_str = str(dataset)
    dataset_id = dataset_str if dataset_str.startswith("easytpp/") else f"easytpp/{dataset_str}"
    use_easytpp = True

    # Set globals used in some internals
    global SAMPLE_INDEX, NUM_INTEGRATION_POINTS
    SAMPLE_INDEX = sample_index
    NUM_INTEGRATION_POINTS = num_integration_points

    if use_easytpp:
        train_dataset = load_dataset(dataset_id, split="train")
        test_dataset = load_dataset(dataset_id, split="test")

        effective_context_size = len(train_dataset) if context_size is None else context_size
        effective_inference_size = len(test_dataset) if inference_size is None else inference_size

        context_data_raw = train_dataset[:effective_context_size]
        inference_data_raw = test_dataset[:effective_inference_size]

        ground_truth_available = False
        ground_truth_functions = None
    else:
        # This branch is intentionally unreachable (always using HF datasets)
        # Kept for compatibility; not used.
        train_dataset_dict = load_local_dataset(dataset_id, "context")
        test_dataset_dict = load_local_dataset(dataset_id, "test")

        effective_context_size = len(train_dataset_dict["seq_len"]) if context_size is None else context_size
        effective_inference_size = len(test_dataset_dict["seq_len"]) if inference_size is None else inference_size

        context_data_raw = load_local_dataset_subset(train_dataset_dict, effective_context_size)
        inference_data_raw = load_local_dataset_subset(test_dataset_dict, effective_inference_size)

        ground_truth_available = "kernel_functions" in test_dataset_dict and "base_intensity_functions" in test_dataset_dict
        if ground_truth_available:
            ground_truth_functions = {
                "kernel_functions": test_dataset_dict["kernel_functions"],
                "base_intensity_functions": test_dataset_dict["base_intensity_functions"],
            }
        else:
            ground_truth_functions = None

        # Thread mandatory time offsets through the raw dicts (shape [P])
        context_data_raw["time_offsets"] = train_dataset_dict["time_offsets"]
        inference_data_raw["time_offsets"] = test_dataset_dict["time_offsets"]

    # Truncate sequences
    def _truncate_batch(data_dict):
        truncated = {"time_since_start": [], "time_since_last_event": [], "type_event": [], "seq_len": []}
        for times, deltas, types, length in zip(
            data_dict["time_since_start"],
            data_dict["time_since_last_event"],
            data_dict["type_event"],
            data_dict["seq_len"],
        ):
            trunc = length if max_num_events is None else min(length, max_num_events)
            truncated["time_since_start"].append(times[:trunc])
            truncated["time_since_last_event"].append(deltas[:trunc])
            truncated["type_event"].append(types[:trunc])
            truncated["seq_len"].append(trunc)
        for key in ("kernel_functions", "base_intensity_functions", "time_offsets"):
            if key in data_dict:
                truncated[key] = data_dict[key]
        return truncated

    context_data_raw = _truncate_batch(context_data_raw)
    inference_data_raw = _truncate_batch(inference_data_raw)

    detected_num_marks = detect_num_event_types_from_data(context_data_raw, inference_data_raw)

    # Sampler calibration using context deltas
    num_context_sequences = len(context_data_raw["seq_len"]) if "seq_len" in context_data_raw else 0
    if num_context_sequences > 0:
        max_dtime_train = (
            max(max(seq) for seq in context_data_raw["time_since_last_event"]) if context_data_raw["time_since_last_event"] else 1.0
        )
    else:
        max_dtime_train = 1.0
    sampler_cap = float(max_dtime_train) * 1.2
    model.event_sampler.dtime_max = sampler_cap
    model.event_sampler.over_sample_rate = 5.0

    # Build context batch
    max_context_len = max(len(seq) for seq in context_data_raw["time_since_start"]) if num_context_sequences > 0 else 0
    context_batch = {"time_seqs": [], "type_seqs": [], "seq_len": []}
    for i in range(len(context_data_raw["time_since_start"])):
        pad_len = max_context_len - len(context_data_raw["time_since_start"][i])
        context_batch["time_seqs"].append(context_data_raw["time_since_start"][i] + [0] * pad_len)
        context_batch["type_seqs"].append(context_data_raw["type_event"][i] + [0] * pad_len)
        context_batch["seq_len"].append(context_data_raw["seq_len"][i])
    context_batch["time_seqs"] = torch.tensor(context_batch["time_seqs"], device=device)
    context_batch["type_seqs"] = torch.tensor(context_batch["type_seqs"], device=device)
    context_batch["seq_len"] = torch.tensor(context_batch["seq_len"], device=device)
    context_batch["seq_non_pad_mask"] = torch.arange(max_context_len, device=device).expand(
        num_context_sequences, max_context_len
    ) < context_batch["seq_len"].unsqueeze(1)

    # Metrics accumulators
    total_mae, total_sq_err, total_acc, total_events = 0, 0, 0, 0
    total_nll_model = 0
    if ground_truth_available:
        gt_total_mae, gt_total_sq_err, gt_total_acc, gt_total_events = 0, 0, 0, 0
        total_nll_gt = 0

    baseline_total_mae, baseline_total_sq_err, baseline_total_acc, baseline_total_events = 0, 0, 0, 0

    # For bootstrap CIs, collect per-event and per-sequence stats
    model_abs_errors: list = []
    model_sq_errors: list = []
    model_correct_flags: list = []
    model_seq_loglikes: list = []  # per-sequence average loglike

    baseline_abs_errors: list = []
    baseline_sq_errors: list = []
    baseline_correct_flags: list = []
    # Error flags (1 for incorrect type prediction)
    model_error_flags: list = []
    baseline_error_flags: list = []

    if ground_truth_available:
        gt_abs_errors: list = []
        gt_sq_errors: list = []
        gt_correct_flags: list = []
        gt_error_flags: list = []
        gt_seq_loglikes: list = []

    # Baseline predictions
    baseline_predictions = compute_baseline_predictions(context_data_raw, inference_data_raw)
    baseline_pred_dtimes = baseline_predictions["predicted_event_dtimes"]
    baseline_pred_types = baseline_predictions["predicted_event_types"]

    # Precompute enhanced context
    with torch.no_grad():
        precomp_ctx = {
            "context_event_times": context_batch["time_seqs"].unsqueeze(0).unsqueeze(-1).to(device),
            "context_event_types": context_batch["type_seqs"].unsqueeze(0).unsqueeze(-1).to(device),
            "context_seq_lengths": context_batch["seq_non_pad_mask"].sum(dim=1).unsqueeze(0).to(device),
        }
        precomputed_enhanced_context = model.encode_context(precomp_ctx)

    # Inference loop
    num_inference_sequences = len(inference_data_raw["seq_len"]) if "seq_len" in inference_data_raw else 0
    with torch.no_grad():
        for i in range(num_inference_sequences):
            inference_item = {
                "time_seqs": torch.tensor([inference_data_raw["time_since_start"][i]], device=device),
                "time_delta_seqs": torch.tensor([inference_data_raw["time_since_last_event"][i]], device=device),
                "type_seqs": torch.tensor([inference_data_raw["type_event"][i]], device=device),
                "seq_len": torch.tensor([inference_data_raw["seq_len"][i]], device=device),
            }
            if (not use_easytpp) and ("time_offsets" in inference_data_raw):
                toff_val = inference_data_raw["time_offsets"][i]
                scalar_off = float(toff_val.item()) if torch.is_tensor(toff_val) else float(toff_val)
                inference_item["time_offset_tensor"] = torch.tensor([[scalar_off]], device=device, dtype=torch.float32)
            inf_max_len = inference_item["time_seqs"].shape[1]
            inference_item["seq_non_pad_mask"] = torch.arange(inf_max_len, device=device).expand(1, inf_max_len) < inference_item[
                "seq_len"
            ].unsqueeze(1)

            predictions = predict_next_event_for_sequence(
                model,
                inference_item,
                context_batch,
                device,
                precomputed_enhanced_context=precomputed_enhanced_context,
                num_marks=detected_num_marks,
            )
            pred_dtimes = predictions["predicted_event_dtimes"].cpu()
            pred_types = predictions["predicted_event_types"].cpu()

            if ground_truth_available:
                gt_predictions = predict_next_event_for_sequence_ground_truth(
                    model, inference_item, context_batch, ground_truth_functions, device, detected_num_marks
                )
                gt_pred_dtimes = gt_predictions["predicted_event_dtimes"].cpu()
                gt_pred_types = gt_predictions["predicted_event_types"].cpu()

            baseline_seq_dtimes = baseline_pred_dtimes[i : i + 1]
            baseline_seq_types = baseline_pred_types[i : i + 1]

            true_dtimes = inference_item["time_delta_seqs"].cpu()[:, 1:]
            true_types = inference_item["type_seqs"].cpu()[:, 1:]

            mask = torch.zeros_like(true_types, dtype=torch.bool)
            original_length = inference_item["seq_len"].item()
            if original_length > 1:
                mask[0, : original_length - 1] = True

            num_events_in_batch = mask.sum().item()
            if num_events_in_batch > 0:
                mae = torch.abs(pred_dtimes[mask] - true_dtimes[mask]).sum().item()
                sq_err = ((pred_dtimes[mask] - true_dtimes[mask]) ** 2).sum().item()
                acc = (pred_types[mask] == true_types[mask]).sum().item()

                total_mae += mae
                total_sq_err += sq_err
                total_acc += acc
                total_events += num_events_in_batch

                # Collect per-event stats for bootstrap
                model_abs_errors.extend(torch.abs(pred_dtimes[mask] - true_dtimes[mask]).view(-1).tolist())
                model_sq_errors.extend(((pred_dtimes[mask] - true_dtimes[mask]) ** 2).view(-1).tolist())
                correct_flags_tensor = (pred_types[mask] == true_types[mask]).to(torch.float32).view(-1)
                model_correct_flags.extend(correct_flags_tensor.tolist())
                model_error_flags.extend((1.0 - correct_flags_tensor).tolist())

                seq_pred_dtimes = baseline_seq_dtimes[:, : original_length - 1]
                seq_pred_types = baseline_seq_types[:, : original_length - 1]
                baseline_mae = torch.abs(seq_pred_dtimes[mask] - true_dtimes[mask]).sum().item()
                baseline_sq_err = ((seq_pred_dtimes[mask] - true_dtimes[mask]) ** 2).sum().item()
                baseline_acc = (seq_pred_types[mask] == true_types[mask]).sum().item()

                baseline_total_mae += baseline_mae
                baseline_total_sq_err += baseline_sq_err
                baseline_total_acc += baseline_acc
                baseline_total_events += num_events_in_batch

                # Collect baseline per-event stats for bootstrap
                baseline_abs_errors.extend(torch.abs(seq_pred_dtimes[mask] - true_dtimes[mask]).view(-1).tolist())
                baseline_sq_errors.extend(((seq_pred_dtimes[mask] - true_dtimes[mask]) ** 2).view(-1).tolist())
                baseline_correct_tensor = (seq_pred_types[mask] == true_types[mask]).to(torch.float32).view(-1)
                baseline_correct_flags.extend(baseline_correct_tensor.tolist())
                baseline_error_flags.extend((1.0 - baseline_correct_tensor).tolist())

                if ground_truth_available:
                    gt_mae = torch.abs(gt_pred_dtimes[mask] - true_dtimes[mask]).sum().item()
                    gt_sq_err = ((gt_pred_dtimes[mask] - true_dtimes[mask]) ** 2).sum().item()
                    gt_acc = (gt_pred_types[mask] == true_types[mask]).sum().item()

                    gt_total_mae += gt_mae
                    gt_total_sq_err += gt_sq_err
                    gt_total_acc += gt_acc
                    gt_total_events += num_events_in_batch

                    # Collect GT per-event stats for bootstrap
                    gt_abs_errors.extend(torch.abs(gt_pred_dtimes[mask] - true_dtimes[mask]).view(-1).tolist())
                    gt_sq_errors.extend(((gt_pred_dtimes[mask] - true_dtimes[mask]) ** 2).view(-1).tolist())
                    gt_correct_tensor = (gt_pred_types[mask] == true_types[mask]).to(torch.float32).view(-1)
                    gt_correct_flags.extend(gt_correct_tensor.tolist())
                    gt_error_flags.extend((1.0 - gt_correct_tensor).tolist())

            nll_results = compute_nll(
                model,
                inference_item,
                context_batch,
                device,
                ground_truth_functions if ground_truth_available else None,
                precomputed_enhanced_context=precomputed_enhanced_context,
                num_marks=detected_num_marks,
            )
            total_nll_model += nll_results.get("model_nll", 0)
            # Per-sequence loglike for bootstrap (mean per sequence)
            if "model_nll" in nll_results:
                model_seq_loglikes.append(-float(nll_results["model_nll"]))
            if ground_truth_available:
                total_nll_gt += nll_results.get("gt_nll", 0)
                if "gt_nll" in nll_results:
                    gt_seq_loglikes.append(-float(nll_results["gt_nll"]))

    # Aggregate
    final_mae = total_mae / total_events if total_events > 0 else 0.0
    final_rmse = float(np.sqrt(total_sq_err / total_events)) if total_events > 0 else 0.0
    final_acc = total_acc / total_events if total_events > 0 else 0.0
    final_ll_model = -total_nll_model / num_inference_sequences if num_inference_sequences > 0 else 0.0

    baseline_final_mae = baseline_total_mae / baseline_total_events if baseline_total_events > 0 else 0.0
    baseline_final_rmse = float(np.sqrt(baseline_total_sq_err / baseline_total_events)) if baseline_total_events > 0 else 0.0
    baseline_final_acc = baseline_total_acc / baseline_total_events if baseline_total_events > 0 else 0.0

    # Compute bootstrap CIs (95%)
    mae_ci_lower, mae_ci_upper = (
        _bootstrap_ci(np.array(model_abs_errors), np.mean, num_bootstrap_samples) if total_events > 0 else (None, None)
    )
    rmse_ci_lower, rmse_ci_upper = (
        _bootstrap_ci(np.array(model_sq_errors), lambda x: np.sqrt(np.mean(x)), num_bootstrap_samples) if total_events > 0 else (None, None)
    )
    # Compute CI directly on error rate (not as 1 - accuracy)
    type_error_ci_lower, type_error_ci_upper = (
        _bootstrap_ci(np.array(model_error_flags), np.mean, num_bootstrap_samples) if total_events > 0 else (None, None)
    )
    loglike_ci_lower, loglike_ci_upper = (
        _bootstrap_ci(np.array(model_seq_loglikes), np.mean, num_bootstrap_samples) if len(model_seq_loglikes) > 0 else (None, None)
    )

    baseline_mae_ci_lower, baseline_mae_ci_upper = (
        _bootstrap_ci(np.array(baseline_abs_errors), np.mean, num_bootstrap_samples) if baseline_total_events > 0 else (None, None)
    )
    baseline_rmse_ci_lower, baseline_rmse_ci_upper = (
        _bootstrap_ci(np.array(baseline_sq_errors), lambda x: np.sqrt(np.mean(x)), num_bootstrap_samples)
        if baseline_total_events > 0
        else (None, None)
    )
    baseline_type_error_ci_lower, baseline_type_error_ci_upper = (
        _bootstrap_ci(np.array(baseline_error_flags), np.mean, num_bootstrap_samples) if baseline_total_events > 0 else (None, None)
    )

    if ground_truth_available:
        gt_mae_ci_lower, gt_mae_ci_upper = (
            _bootstrap_ci(np.array(gt_abs_errors), np.mean, num_bootstrap_samples) if gt_total_events > 0 else (None, None)
        )
        gt_rmse_ci_lower, gt_rmse_ci_upper = (
            _bootstrap_ci(np.array(gt_sq_errors), lambda x: np.sqrt(np.mean(x)), num_bootstrap_samples)
            if gt_total_events > 0
            else (None, None)
        )
        gt_type_error_ci_lower, gt_type_error_ci_upper = (
            _bootstrap_ci(np.array(gt_error_flags), np.mean, num_bootstrap_samples) if gt_total_events > 0 else (None, None)
        )
        gt_loglike_ci_lower, gt_loglike_ci_upper = (
            _bootstrap_ci(np.array(gt_seq_loglikes), np.mean, num_bootstrap_samples) if len(gt_seq_loglikes) > 0 else (None, None)
        )
    else:
        gt_mae_ci_lower = gt_mae_ci_upper = gt_rmse_ci_lower = gt_rmse_ci_upper = None
        gt_type_error_ci_lower = gt_type_error_ci_upper = gt_loglike_ci_lower = gt_loglike_ci_upper = None

    result = {
        "dataset": dataset_id if use_easytpp else dataset,
        "model_checkpoint": model_checkpoint_path,
        "sampling_method": model.event_sampler.sampling_method,
        "ground_truth_available": bool(ground_truth_available),
        "num_events": int(total_events),
        "num_inference_sequences": int(num_inference_sequences),
        "duration_seconds": float(time.time() - start_time),
        "metrics": {
            "model": {
                "mae": float(final_mae),
                "rmse": float(final_rmse),
                "type_error": float(100.0 * (1.0 - final_acc)),
                "loglike": float(final_ll_model),
                "mae_ci_error": (None if mae_ci_lower is None or mae_ci_upper is None else float(0.5 * (mae_ci_upper - mae_ci_lower))),
                "rmse_ci_error": (None if rmse_ci_lower is None or rmse_ci_upper is None else float(0.5 * (rmse_ci_upper - rmse_ci_lower))),
                "type_error_ci_error": (
                    None
                    if type_error_ci_lower is None or type_error_ci_upper is None
                    else float(100.0 * 0.5 * (type_error_ci_upper - type_error_ci_lower))
                ),
                "loglike_ci_error": (
                    None if loglike_ci_lower is None or loglike_ci_upper is None else float(0.5 * (loglike_ci_upper - loglike_ci_lower))
                ),
            },
            "baseline": {
                "mae": float(baseline_final_mae),
                "rmse": float(baseline_final_rmse),
                "type_error": float(100.0 * (1.0 - baseline_final_acc)),
                "loglike": None,
                "mae_ci_error": (
                    None
                    if baseline_mae_ci_lower is None or baseline_mae_ci_upper is None
                    else float(0.5 * (baseline_mae_ci_upper - baseline_mae_ci_lower))
                ),
                "rmse_ci_error": (
                    None
                    if baseline_rmse_ci_lower is None or baseline_rmse_ci_upper is None
                    else float(0.5 * (baseline_rmse_ci_upper - baseline_rmse_ci_lower))
                ),
                "type_error_ci_error": (
                    None
                    if baseline_type_error_ci_lower is None or baseline_type_error_ci_upper is None
                    else float(100.0 * 0.5 * (baseline_type_error_ci_upper - baseline_type_error_ci_lower))
                ),
                "loglike_ci_error": None,
            },
            "ground_truth": None,
        },
    }

    if ground_truth_available:
        gt_final_mae = gt_total_mae / gt_total_events if gt_total_events > 0 else 0.0
        gt_final_rmse = float(np.sqrt(gt_total_sq_err / gt_total_events)) if gt_total_events > 0 else 0.0
        gt_final_acc = gt_total_acc / gt_total_events if gt_total_events > 0 else 0.0
        final_ll_gt = -total_nll_gt / num_inference_sequences if num_inference_sequences > 0 else 0.0
        result["metrics"]["ground_truth"] = {
            "mae": float(gt_final_mae),
            "rmse": float(gt_final_rmse),
            "type_error": float(100.0 * (1.0 - gt_final_acc)),
            "loglike": float(final_ll_gt),
            "mae_ci_error": (
                None if gt_mae_ci_lower is None or gt_mae_ci_upper is None else float(0.5 * (gt_mae_ci_upper - gt_mae_ci_lower))
            ),
            "rmse_ci_error": (
                None if gt_rmse_ci_lower is None or gt_rmse_ci_upper is None else float(0.5 * (gt_rmse_ci_upper - gt_rmse_ci_lower))
            ),
            "type_error_ci_error": (
                None
                if gt_type_error_ci_lower is None or gt_type_error_ci_upper is None
                else float(100.0 * 0.5 * (gt_type_error_ci_upper - gt_type_error_ci_lower))
            ),
            "loglike_ci_error": (
                None
                if gt_loglike_ci_lower is None or gt_loglike_ci_upper is None
                else float(0.5 * (gt_loglike_ci_upper - gt_loglike_ci_lower))
            ),
        }

    # Optional: plotting is supported only when run as a standalone script; skip inside the function
    if plot_intensity_predictions and False:
        pass

    return result


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
        effective_context_size = len(train_dataset) if CONTEXT_SIZE is None else CONTEXT_SIZE
        effective_inference_size = len(test_dataset) if INFERENCE_SIZE is None else INFERENCE_SIZE

        context_data_raw = train_dataset[:effective_context_size]
        inference_data_raw = test_dataset[:effective_inference_size]

        # Ground truth functions not available for HuggingFace datasets
        ground_truth_available = False
        ground_truth_functions = None
    else:
        print(f"Loading and preprocessing local dataset from: {LOCAL_DATASET_PATH}...")
        print(f"Using sample index: {SAMPLE_INDEX}")
        train_dataset_dict = load_local_dataset(LOCAL_DATASET_PATH, "context")
        test_dataset_dict = load_local_dataset(LOCAL_DATASET_PATH, "test")

        # Select a subset for context and inference based on configuration
        effective_context_size = len(train_dataset_dict["seq_len"]) if CONTEXT_SIZE is None else CONTEXT_SIZE
        effective_inference_size = len(test_dataset_dict["seq_len"]) if INFERENCE_SIZE is None else INFERENCE_SIZE

        context_data_raw = load_local_dataset_subset(train_dataset_dict, effective_context_size)
        inference_data_raw = load_local_dataset_subset(test_dataset_dict, effective_inference_size)

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

        # Thread mandatory time offsets through the raw dicts (shape [P])
        context_data_raw["time_offsets"] = train_dataset_dict["time_offsets"]
        inference_data_raw["time_offsets"] = test_dataset_dict["time_offsets"]

    # Truncate sequences longer than MAX_NUM_EVENTS
    limit_str = "no limit" if MAX_NUM_EVENTS is None else str(MAX_NUM_EVENTS)
    print(f"Truncating all sequences to at most {limit_str} events...")

    def _truncate_example(example):
        length = example.get("seq_len", len(example.get("time_since_start", [])))
        trunc = length if MAX_NUM_EVENTS is None else min(length, MAX_NUM_EVENTS)
        example["time_since_start"] = example["time_since_start"][:trunc]
        example["time_since_last_event"] = example["time_since_last_event"][:trunc]
        example["type_event"] = example["type_event"][:trunc]
        example["seq_len"] = trunc
        return example

    def _truncate_batch(data_dict):
        truncated = {"time_since_start": [], "time_since_last_event": [], "type_event": [], "seq_len": []}
        for times, deltas, types, length in zip(
            data_dict["time_since_start"],
            data_dict["time_since_last_event"],
            data_dict["type_event"],
            data_dict["seq_len"],
        ):
            trunc = length if MAX_NUM_EVENTS is None else min(length, MAX_NUM_EVENTS)
            truncated["time_since_start"].append(times[:trunc])
            truncated["time_since_last_event"].append(deltas[:trunc])
            truncated["type_event"].append(types[:trunc])
            truncated["seq_len"].append(trunc)
        # Preserve any ground truth functions and time offsets
        for key in ("kernel_functions", "base_intensity_functions", "time_offsets"):
            if key in data_dict:
                truncated[key] = data_dict[key]
        return truncated

    # Apply truncation to the dict-of-lists data
    context_data_raw = _truncate_batch(context_data_raw)
    inference_data_raw = _truncate_batch(inference_data_raw)

    # Detect number of unique event types from the data
    detected_num_marks = detect_num_event_types_from_data(context_data_raw, inference_data_raw)
    print(f"Detected number of event types (marks): {detected_num_marks}")

    # 3. Prepare the fixed context batch (on-the-fly)
    # This batch will be cached and reused for all inference sequences.
    print("Preparing fixed context data...")

    # ------------------------------------------------------------------
    # Adjust the sampler's dtime_max to match the time scale of the dataset and avoid artificial truncation.
    # ------------------------------------------------------------------

    num_context_sequences = len(context_data_raw["seq_len"]) if "seq_len" in context_data_raw else 0
    num_inference_sequences = len(inference_data_raw["seq_len"]) if "seq_len" in inference_data_raw else 0

    if num_context_sequences > 0:
        max_dtime_train = (
            max(max(seq) for seq in context_data_raw["time_since_last_event"]) if context_data_raw["time_since_last_event"] else 1.0
        )
    else:
        max_dtime_train = 1.0

    sampler_cap = float(max_dtime_train) * 1.2  # 20 % safety margin
    model.event_sampler.dtime_max = sampler_cap
    print(f"Updated sampler dtime_max to {sampler_cap:.2f}")

    # Lower the over-sampling factor to avoid overly conservative
    # Oversampling factor influences the intensity upper bound.  A larger value makes
    # rejection sampling more likely to ACCEPT a draw (at the expense of a few more
    # evaluations). We therefore revert to a safer default of 5.0.
    model.event_sampler.over_sample_rate = 5.0
    print(f"Set sampler over_sample_rate to {model.event_sampler.over_sample_rate:.1f}")

    max_context_len = max(len(seq) for seq in context_data_raw["time_since_start"]) if num_context_sequences > 0 else 0

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
    context_batch["seq_non_pad_mask"] = torch.arange(max_context_len, device=device).expand(
        num_context_sequences, max_context_len
    ) < context_batch["seq_len"].unsqueeze(1)

    # Initialize metrics for model predictions
    total_mae, total_sq_err, total_acc, total_events = 0, 0, 0, 0
    total_nll_model = 0

    # Initialize metrics for ground truth predictions if available
    if ground_truth_available:
        gt_total_mae, gt_total_sq_err, gt_total_acc, gt_total_events = 0, 0, 0, 0
        total_nll_gt = 0

    # Initialize metrics for baseline predictions
    baseline_total_mae, baseline_total_sq_err, baseline_total_acc, baseline_total_events = 0, 0, 0, 0

    # Compute baseline predictions once for all sequences
    print("\nComputing baseline predictions...")
    baseline_predictions = compute_baseline_predictions(context_data_raw, inference_data_raw)
    baseline_pred_dtimes = baseline_predictions["predicted_event_dtimes"]
    baseline_pred_types = baseline_predictions["predicted_event_types"]

    # Precompute enhanced context embeddings once, since context paths remain fixed across inference
    with torch.no_grad():
        precomp_ctx = {
            "context_event_times": context_batch["time_seqs"].unsqueeze(0).unsqueeze(-1).to(device),
            "context_event_types": context_batch["type_seqs"].unsqueeze(0).unsqueeze(-1).to(device),
            "context_seq_lengths": context_batch["seq_non_pad_mask"].sum(dim=1).unsqueeze(0).to(device),
        }
        precomputed_enhanced_context = model.encode_context(precomp_ctx)

    # 4. Loop through inference sequences and perform next-event prediction
    with torch.no_grad():
        for i in tqdm(range(num_inference_sequences), desc="Evaluating Inference Sequences"):
            # a. Prepare a single inference sequence
            inference_item = {
                "time_seqs": torch.tensor([inference_data_raw["time_since_start"][i]], device=device),
                "time_delta_seqs": torch.tensor([inference_data_raw["time_since_last_event"][i]], device=device),
                "type_seqs": torch.tensor([inference_data_raw["type_event"][i]], device=device),
                "seq_len": torch.tensor([inference_data_raw["seq_len"][i]], device=device),
            }
            # Attach per-path time offset only for local datasets (if available): model expects shape [B=1, P=1]
            if (not USE_EASYTPP) and ("time_offsets" in inference_data_raw):
                toff_val = inference_data_raw["time_offsets"][i]
                scalar_off = float(toff_val.item()) if torch.is_tensor(toff_val) else float(toff_val)
                inference_item["time_offset_tensor"] = torch.tensor([[scalar_off]], device=device, dtype=torch.float32)
            inf_max_len = inference_item["time_seqs"].shape[1]
            inference_item["seq_non_pad_mask"] = torch.arange(inf_max_len, device=device).expand(1, inf_max_len) < inference_item[
                "seq_len"
            ].unsqueeze(1)

            # b. Get model predictions
            predictions = predict_next_event_for_sequence(
                model,
                inference_item,
                context_batch,
                device,
                precomputed_enhanced_context=precomputed_enhanced_context,
                num_marks=detected_num_marks,
            )
            pred_dtimes = predictions["predicted_event_dtimes"].cpu()
            pred_types = predictions["predicted_event_types"].cpu()

            # c. Get ground truth predictions if available
            if ground_truth_available:
                gt_predictions = predict_next_event_for_sequence_ground_truth(
                    model, inference_item, context_batch, ground_truth_functions, device, detected_num_marks
                )
                gt_pred_dtimes = gt_predictions["predicted_event_dtimes"].cpu()
                gt_pred_types = gt_predictions["predicted_event_types"].cpu()

            # d. Get baseline predictions for this sequence
            baseline_seq_dtimes = baseline_pred_dtimes[i : i + 1]  # Keep batch dimension
            baseline_seq_types = baseline_pred_types[i : i + 1]  # Keep batch dimension

            # e. Get ground truth for comparison
            true_dtimes = inference_item["time_delta_seqs"].cpu()[:, 1:]
            true_types = inference_item["type_seqs"].cpu()[:, 1:]

            # f. Calculate metrics for the current sequence
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

                # Baseline metrics
                # Baseline metrics (slice predictions to match true sequence length)
                seq_pred_dtimes = baseline_seq_dtimes[:, : original_length - 1]
                seq_pred_types = baseline_seq_types[:, : original_length - 1]
                baseline_mae = torch.abs(seq_pred_dtimes[mask] - true_dtimes[mask]).sum().item()
                baseline_sq_err = ((seq_pred_dtimes[mask] - true_dtimes[mask]) ** 2).sum().item()
                baseline_acc = (seq_pred_types[mask] == true_types[mask]).sum().item()

                baseline_total_mae += baseline_mae
                baseline_total_sq_err += baseline_sq_err
                baseline_total_acc += baseline_acc
                baseline_total_events += num_events_in_batch

                # Ground truth metrics
                if ground_truth_available:
                    gt_mae = torch.abs(gt_pred_dtimes[mask] - true_dtimes[mask]).sum().item()
                    gt_sq_err = ((gt_pred_dtimes[mask] - true_dtimes[mask]) ** 2).sum().item()
                    gt_acc = (gt_pred_types[mask] == true_types[mask]).sum().item()

                    gt_total_mae += gt_mae
                    gt_total_sq_err += gt_sq_err
                    gt_total_acc += gt_acc
                    gt_total_events += num_events_in_batch

            # g. Calculate NLL for the current sequence
            nll_results = compute_nll(
                model,
                inference_item,
                context_batch,
                device,
                ground_truth_functions if ground_truth_available else None,
                precomputed_enhanced_context=precomputed_enhanced_context,
                num_marks=detected_num_marks,
            )
            total_nll_model += nll_results.get("model_nll", 0)
            if ground_truth_available:
                total_nll_gt += nll_results.get("gt_nll", 0)

    # 5. Report final results
    final_mae = total_mae / total_events if total_events > 0 else 0
    final_rmse = np.sqrt(total_sq_err / total_events) if total_events > 0 else 0
    final_acc = total_acc / total_events if total_events > 0 else 0
    final_ll_model = -total_nll_model / num_inference_sequences if num_inference_sequences > 0 else 0

    baseline_final_mae = baseline_total_mae / baseline_total_events if baseline_total_events > 0 else 0
    baseline_final_rmse = np.sqrt(baseline_total_sq_err / baseline_total_events) if baseline_total_events > 0 else 0
    baseline_final_acc = baseline_total_acc / baseline_total_events if baseline_total_events > 0 else 0

    print("\n" + "=" * 60)
    print("--- EVALUATION RESULTS ---")
    print(f"Dataset: {DATASET_IDENTIFIER if USE_EASYTPP else LOCAL_DATASET_PATH}")
    print(f"Total Events Evaluated: {total_events}")
    print()
    print("SIMPLE BASELINE (Majority Type + Mean Time):")
    print(f"  Time Prediction MAE:       {baseline_final_mae:.4f}")
    print(f"  Time Prediction RMSE:      {baseline_final_rmse:.4f}")
    print(f"  Type Prediction Error (%): {100.0 * (1.0 - baseline_final_acc):.4f}")
    print()
    print("MODEL PREDICTIONS:")
    print(f"  Time Prediction MAE:       {final_mae:.4f}")
    print(f"  Time Prediction RMSE:      {final_rmse:.4f}")
    print(f"  Type Prediction Error (%): {100.0 * (1.0 - final_acc):.4f}")
    print(f"  Log-Likelihood:          {final_ll_model:.4f}")

    if ground_truth_available:
        gt_final_mae = gt_total_mae / gt_total_events if gt_total_events > 0 else 0
        gt_final_rmse = np.sqrt(gt_total_sq_err / gt_total_events) if gt_total_events > 0 else 0
        gt_final_acc = gt_total_acc / gt_total_events if gt_total_events > 0 else 0
        final_ll_gt = -total_nll_gt / num_inference_sequences if num_inference_sequences > 0 else 0

        print()
        print("GROUND TRUTH BASELINE:")
        print(f"  Time Prediction MAE:       {gt_final_mae:.4f}")
        print(f"  Time Prediction RMSE:      {gt_final_rmse:.4f}")
        print(f"  Type Prediction Error (%): {100.0 * (1.0 - gt_final_acc):.4f}")
        print(f"  Log-Likelihood:          {final_ll_gt:.4f}")

        print()
        print("COMPARISON (Model vs Ground Truth):")
        mae_improvement = ((gt_final_mae - final_mae) / gt_final_mae * 100) if gt_final_mae > 0 else 0
        rmse_improvement = ((gt_final_rmse - final_rmse) / gt_final_rmse * 100) if gt_final_rmse > 0 else 0
        # Use error (1 - accuracy) for improvement
        gt_error = 1.0 - gt_final_acc
        model_error = 1.0 - final_acc
        err_improvement = ((gt_error - model_error) / gt_error * 100) if gt_error > 0 else 0

        print(f"  MAE improvement:           {mae_improvement:+.1f}%")
        print(f"  RMSE improvement:          {rmse_improvement:+.1f}%")
        print(f"  Error improvement:         {err_improvement:+.1f}%")

    print()
    print("COMPARISON (Model vs Simple Baseline):")
    baseline_mae_improvement = ((baseline_final_mae - final_mae) / baseline_final_mae * 100) if baseline_final_mae > 0 else 0
    baseline_rmse_improvement = ((baseline_final_rmse - final_rmse) / baseline_final_rmse * 100) if baseline_final_rmse > 0 else 0
    # Use error (1 - accuracy) for improvement
    baseline_error = 1.0 - baseline_final_acc
    model_error = 1.0 - final_acc
    baseline_err_improvement = ((baseline_error - model_error) / baseline_error * 100) if baseline_error > 0 else 0

    print(f"  MAE improvement:           {baseline_mae_improvement:+.1f}%")
    print(f"  RMSE improvement:          {baseline_rmse_improvement:+.1f}%")
    print(f"  Error improvement:         {baseline_err_improvement:+.1f}%")

    print("=" * 60)

    # Plot intensity predictions if requested
    if PLOT_INTENSITY_PREDICTIONS:
        print("\nPlotting intensity predictions...")
        try:
            from visualize_intensity_predictions import (
                load_data_from_dir,
                plot_intensity_comparison,
                prepare_batch_for_model,
            )
        except ImportError:
            print("⚠️  Visualization utilities not available. Skipping intensity plots.")
        else:
            if USE_EASYTPP:
                # Plot intensity predictions using CONTEXT_SIZE context paths and INFERENCE_SIZE inference paths
                print("Plotting intensity predictions for EasyTPP dataset sample...")
                # Combine context and inference sequences into one multi-path sample
                all_times = context_data_raw["time_since_start"] + inference_data_raw["time_since_start"]
                all_types = context_data_raw["type_event"] + inference_data_raw["type_event"]
                all_lens = context_data_raw["seq_len"] + inference_data_raw["seq_len"]
                # Pad sequences to equal length and build raw (unbatched) tensors
                max_L = max(all_lens)
                times_padded, types_padded = [], []
                for t_seq, c_seq, L_seq in zip(all_times, all_types, all_lens):
                    pad = max_L - L_seq
                    times_padded.append(t_seq + [0.0] * pad)
                    types_padded.append(c_seq + [0] * pad)
                # Build sample tensors without leading batch dim; helper will add it
                event_times = torch.tensor(times_padded, dtype=torch.float32).unsqueeze(-1)  # [P_paths, L, 1]
                event_types = torch.tensor(types_padded, dtype=torch.long).unsqueeze(-1)  # [P_paths, L, 1]
                seq_lengths = torch.tensor(all_lens, dtype=torch.long)  # [P_paths]
                single_sample = {
                    "event_times": event_times,
                    "event_types": event_types,
                    "seq_lengths": seq_lengths,
                }
                try:
                    # Use configurable path index within the inference split
                    _sel_inf_idx = 0
                    try:
                        _sel_inf_idx = int(PLOT_PATH_INDEX)
                    except Exception:
                        _sel_inf_idx = 0
                    # Constrain by actual number of inference sequences
                    _sel_inf_idx = max(0, min(_sel_inf_idx, int(num_inference_sequences) - 1))
                    model_data_vis = prepare_batch_for_model(
                        single_sample,
                        inference_path_idx=int(num_context_sequences) + _sel_inf_idx,
                        num_points_between_events=10,
                    )
                    # Ensure we plot all marks for EasyTPP data
                    model_data_vis["num_marks"] = detected_num_marks
                except ValueError as e:
                    print(f"⚠️  Could not prepare EasyTPP context/inference sample: {e}")
                else:
                    # Move visualization batch to device
                    for key, val in model_data_vis.items():
                        if torch.is_tensor(val):
                            model_data_vis[key] = val.to(device)
                    try:
                        with torch.no_grad():
                            model_output_vis = model(model_data_vis)
                        save_path = f"intensity_comparison_sample_{SAMPLE_INDEX}_path_{PLOT_PATH_INDEX}.png"
                        plot_intensity_comparison(
                            model_output_vis,
                            model_data_vis,
                            save_path=save_path,
                            path_idx=0,
                        )
                    except Exception as e:
                        print(f"⚠️  Intensity visualization failed: {e}")
            else:
                dataset_dir = Path(LOCAL_DATASET_PATH) / "test"
                data = load_data_from_dir(dataset_dir)
                # Select the sample based on SAMPLE_INDEX
                single_sample = {
                    key: (value[SAMPLE_INDEX] if torch.is_tensor(value) else value[SAMPLE_INDEX]) for key, value in data.items()
                }
                try:
                    # Use configurable path index within the sample
                    try:
                        _p = int(single_sample["event_times"].shape[0])
                    except Exception:
                        _p = 1
                    _sel_path_idx = 0
                    try:
                        _sel_path_idx = int(PLOT_PATH_INDEX)
                    except Exception:
                        _sel_path_idx = 0
                    _sel_path_idx = max(0, min(_sel_path_idx, _p - 1))
                    model_data_vis = prepare_batch_for_model(single_sample, inference_path_idx=_sel_path_idx, num_points_between_events=10)
                except ValueError as e:
                    print(f"⚠️  Could not prepare batch for intensity visualization: {e}")
                else:
                    # Move visualization batch to the same device as the model
                    for key, val in model_data_vis.items():
                        if torch.is_tensor(val):
                            model_data_vis[key] = val.to(device)
                    try:
                        with torch.no_grad():
                            model_output_vis = model(model_data_vis)
                        save_path = f"intensity_comparison_sample_{SAMPLE_INDEX}_path_{PLOT_PATH_INDEX}.png"
                        plot_intensity_comparison(model_output_vis, model_data_vis, save_path=save_path, path_idx=0)
                    except Exception as e:
                        print(f"⚠️  Intensity visualization failed: {e}")

    # Plot event predictions for a single trajectory if requested
    if PLOT_EVENT_PREDICTIONS:
        print("\nPlotting event predictions for a single trajectory...")
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️  Matplotlib not available. Skipping event prediction plot.")
        else:
            if num_inference_sequences == 0:
                print("⚠️  No inference sequences available to plot.")
            else:
                i = min(max(0, PLOT_PATH_INDEX), num_inference_sequences - 1)
                try:
                    # Prepare the selected inference sequence
                    inference_item = {
                        "time_seqs": torch.tensor([inference_data_raw["time_since_start"][i]], device=device),
                        "time_delta_seqs": torch.tensor([inference_data_raw["time_since_last_event"][i]], device=device),
                        "type_seqs": torch.tensor([inference_data_raw["type_event"][i]], device=device),
                        "seq_len": torch.tensor([inference_data_raw["seq_len"][i]], device=device),
                    }
                    if (not USE_EASYTPP) and ("time_offsets" in inference_data_raw):
                        toff_val = inference_data_raw["time_offsets"][i]
                        scalar_off = float(toff_val.item()) if torch.is_tensor(toff_val) else float(toff_val)
                        inference_item["time_offset_tensor"] = torch.tensor([[scalar_off]], device=device, dtype=torch.float32)
                    inf_max_len = inference_item["time_seqs"].shape[1]
                    inference_item["seq_non_pad_mask"] = torch.arange(inf_max_len, device=device).expand(1, inf_max_len) < inference_item[
                        "seq_len"
                    ].unsqueeze(1)

                    # Get model predictions for the chosen trajectory
                    predictions = predict_next_event_for_sequence(
                        model,
                        inference_item,
                        context_batch,
                        device,
                        precomputed_enhanced_context=precomputed_enhanced_context,
                        num_marks=detected_num_marks,
                    )
                    pred_dtimes = predictions["predicted_event_dtimes"].cpu()[0]  # [L-1]
                    pred_types = predictions["predicted_event_types"].cpu()[0]  # [L-1]

                    # Ground truth (skip first event since it's the conditioning event)
                    time_seq = inference_item["time_seqs"].cpu()[0]
                    type_seq = inference_item["type_seqs"].cpu()[0]
                    original_length = inference_item["seq_len"].item()

                    if original_length <= 1:
                        print("⚠️  Selected sequence has length <= 1. Nothing to plot.")
                    else:
                        # True next-event absolute times and types
                        true_next_times = time_seq[1:original_length].numpy()
                        true_next_types = type_seq[1:original_length].numpy()

                        # Predicted next-event absolute times: t_k + predicted_dtime_k
                        prev_times = time_seq[: original_length - 1].numpy()
                        pred_next_times = prev_times + pred_dtimes[: original_length - 1].numpy()
                        pred_next_types = pred_types[: original_length - 1].numpy()

                        # Row-per-step timeline: x = event time, y = step index (k->k+1)
                        from matplotlib.lines import Line2D

                        n_steps = len(true_next_times)
                        height = max(4.0, min(14.0, 0.35 * n_steps + 2.0))
                        fig, ax = plt.subplots(1, 1, figsize=(12, height))

                        # Build color palette for types
                        types_present = sorted(set(map(int, true_next_types.tolist())) | set(map(int, pred_next_types.tolist())))
                        cmap = plt.get_cmap("tab20")
                        colors = {t: cmap(i % 20) for i, t in enumerate(types_present)}

                        # Rows correspond to step indices k (predicting event k+1)
                        y_rows = np.arange(n_steps)

                        # Draw faint guide lines for rows
                        for y in y_rows:
                            ax.axhline(y, color="#dddddd", linewidth=0.6, zorder=0)

                        # Scatter true and predicted markers per step, colored by type
                        for idx in range(n_steps):
                            t_true = float(true_next_times[idx])
                            t_pred = float(pred_next_times[idx])
                            type_true = int(true_next_types[idx])
                            type_pred = int(pred_next_types[idx])
                            y = y_rows[idx]

                            # True marker (circle)
                            ax.scatter(
                                t_true,
                                y,
                                color=colors.get(type_true, "#1f77b4"),
                                marker="o",
                                edgecolors="k",
                                linewidths=0.5,
                                s=40,
                                label=None,
                                zorder=3,
                            )
                            # Predicted marker (x)
                            ax.scatter(
                                t_pred,
                                y,
                                color=colors.get(type_pred, "#ff7f0e"),
                                marker="x",
                                linewidths=1.0,
                                s=50,
                                label=None,
                                zorder=3,
                            )

                            # Optional small annotations with type ids
                            ax.text(t_true, y + 0.15, f"T{type_true}", fontsize=7, color="#333333", ha="center", va="bottom")
                            ax.text(t_pred, y - 0.15, f"P{type_pred}", fontsize=7, color="#333333", ha="center", va="top")

                        # Y ticks as step labels
                        ax.set_yticks(y_rows)
                        ax.set_yticklabels([f"{k + 1}→{k + 2}" for k in range(n_steps)])
                        ax.invert_yaxis()  # earliest step at top

                        ax.set_xlabel("Event time")
                        ax.set_ylabel("Step (k→k+1)")
                        ax.set_title("Next-step predictions: rows = steps, circles = True, crosses = Predicted; color = Type")

                        # Legends: marker semantics and type colors
                        marker_handles = [
                            Line2D([0], [0], marker="o", color="k", markerfacecolor="w", markersize=7, linestyle="None", label="True"),
                            Line2D([0], [0], marker="x", color="k", markersize=7, linestyle="None", label="Predicted"),
                        ]
                        type_handles = [
                            Line2D(
                                [0],
                                [0],
                                marker="o",
                                color="none",
                                markeredgecolor="k",
                                markerfacecolor=colors[t],
                                markersize=7,
                                linestyle="None",
                                label=f"Type {t}",
                            )
                            for t in types_present
                        ]
                        legend1 = ax.legend(handles=marker_handles, loc="upper right", title="Kind")
                        ax.add_artist(legend1)
                        ax.legend(handles=type_handles, loc="lower right", title="Types", ncol=min(3, len(type_handles)))

                        fig.tight_layout()
                        save_path = f"event_predictions_sample_{SAMPLE_INDEX}_seq_{i}.png"
                        plt.savefig(save_path, dpi=200, bbox_inches="tight")
                        plt.close(fig)
                        print(f"Saved event prediction plot to: {save_path}")
                except Exception as e:
                    print(f"⚠️  Event prediction plotting failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run a single FIM-Hawkes next-event evaluation")
    parser.add_argument("--checkpoint", type=str, required=False, help="Model checkpoint directory")
    parser.add_argument("--dataset", type=str, required=False, help="HF id (easytpp/...) or local dataset path")
    parser.add_argument("--run-dir", type=str, required=False, help="Directory to write metrics.json and artifacts")
    parser.add_argument("--context-size", type=int, default=None, help="Number of sequences from train split to use as context")
    parser.add_argument("--inference-size", type=int, default=None, help="Number of sequences from test split to evaluate")
    parser.add_argument("--max-num-events", type=int, default=100, help="Truncate sequences to this many events; -1 means no truncation")
    parser.add_argument("--sample-idx", type=int, default=0, help="Sample index (for local datasets)")
    parser.add_argument("--num-integration-points", type=int, default=5000, help="NLL integration points")
    parser.add_argument("--nll-method", type=str, choices=["closed_form", "monte_carlo"], default=None, help="NLL method override")
    parser.add_argument("--num-bootstrap-samples", type=int, default=1000, help="Number of bootstrap samples for 95% CI")
    parser.add_argument(
        "--sampling-method",
        type=str,
        choices=["thinning", "inverse_transform"],
        default=None,
        help="Sampling method for next-event time generation",
    )

    args = parser.parse_args()

    # If required CLI args are not provided, fall back to the legacy main() behavior
    if not args.checkpoint or not args.dataset or not args.run_dir:
        main()
    else:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        max_num_events = None if (args.max_num_events is not None and args.max_num_events < 0) else args.max_num_events

        try:
            # Optional: configure sampling method on the model's sampler
            if args.sampling_method:
                try:
                    # model not yet loaded here; set via environment in child run_next_event_evaluation call instead
                    pass
                except Exception:
                    pass
            result = run_next_event_evaluation(
                model_checkpoint_path=args.checkpoint,
                dataset=args.dataset,
                context_size=args.context_size,
                inference_size=args.inference_size,
                max_num_events=max_num_events,
                sample_index=args.sample_idx,
                num_integration_points=args.num_integration_points,
                plot_intensity_predictions=False,
                num_bootstrap_samples=args.num_bootstrap_samples,
                sampling_method=args.sampling_method,
                nll_method=args.nll_method,
            )
            status = "OK"
        except Exception as e:
            import traceback

            result = {
                "dataset": args.dataset,
                "model_checkpoint": str(args.checkpoint),
                "status": "FAIL",
                "error": f"{e}\n{traceback.format_exc()}",
            }
            status = "FAIL"

        result.setdefault("status", status)
        (run_dir / "metrics.json").write_text(json.dumps(result, indent=2))
