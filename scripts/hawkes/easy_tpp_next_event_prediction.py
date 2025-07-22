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

# Set the Hugging Face dataset identifier.
DATASET_IDENTIFIER = "easytpp/retweet"

# Set the number of event types for the chosen dataset.
NUM_EVENT_TYPES = 3

# --- New Configuration Options ---
# Maximum number of sequences from the training set to use as context.
CONTEXT_SIZE = 1000

# Number of sequences from the test set to use for inference.
INFERENCE_SIZE = 5
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

            hist_times = x["inference_event_times"].squeeze(0).squeeze(-1)
            hist_dtimes = torch.zeros_like(hist_times)
            hist_dtimes[:, 1:] = hist_times[:, 1:] - hist_times[:, :-1]
            hist_types = x["inference_event_types"].squeeze(0).squeeze(-1)

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

            dtime_pred = torch.sum(accepted_dtimes * weights, dim=-1).squeeze()

            t_last = hist_times[0, prefix_len - 1]
            predicted_time = t_last + dtime_pred

            intensities_at_pred_time = intensity_obj.evaluate(predicted_time.view(1, 1, 1))
            type_pred = torch.argmax(intensities_at_pred_time.squeeze())

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

    # 2. Load context and inference data from Hugging Face
    print(f"Loading and preprocessing dataset from Hugging Face Hub: {DATASET_IDENTIFIER}...")
    train_dataset = load_dataset(DATASET_IDENTIFIER, split="train")
    test_dataset = load_dataset(DATASET_IDENTIFIER, split="test")

    # Select a subset for context and inference based on configuration
    context_data_raw = train_dataset[:CONTEXT_SIZE]
    inference_data_raw = test_dataset[:INFERENCE_SIZE]

    # 3. Prepare the fixed context batch (on-the-fly)
    # This batch will be cached and reused for all inference sequences.
    print("Preparing fixed context data...")
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

    # Initialize metrics
    total_mae, total_sq_err, total_acc, total_events = 0, 0, 0, 0

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
            # Note: `predict_next_event_for_sequence` uses `time_seqs` for model input.
            # `time_delta_seqs` is NOT used by the model; it's the ground truth for evaluation.
            predictions = predict_next_event_for_sequence(model, inference_item, context_batch, device)

            pred_dtimes = predictions["predicted_event_dtimes"].cpu()
            pred_types = predictions["predicted_event_types"].cpu()

            # c. Get ground truth for comparison
            true_dtimes = inference_item["time_delta_seqs"].cpu()[:, 1:]
            true_types = inference_item["type_seqs"].cpu()[:, 1:]

            # d. Calculate metrics for the current sequence
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

    # 5. Report final results
    final_mae = total_mae / total_events if total_events > 0 else 0
    final_rmse = np.sqrt(total_sq_err / total_events) if total_events > 0 else 0
    final_acc = total_acc / total_events if total_events > 0 else 0

    print("\n" + "=" * 30)
    print("--- Evaluation Results ---")
    print(f"Dataset: {DATASET_IDENTIFIER}")
    print(f"Total Events Evaluated: {total_events}")
    print(f"Time Prediction MAE:    {final_mae:.4f}")
    print(f"Time Prediction RMSE:   {final_rmse:.4f}")
    print(f"Type Prediction Accuracy: {final_acc:.4f}")
    print("=" * 30)


if __name__ == "__main__":
    if "path/to/your" in MODEL_CHECKPOINT_PATH:
        print("=" * 60)
        print("!!! WARNING: Please update the MODEL_CHECKPOINT_PATH variable !!!")
        print(f"Current path is: {MODEL_CHECKPOINT_PATH}")
        print("=" * 60)
    else:
        main()
