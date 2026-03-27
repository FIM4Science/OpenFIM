from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def discover_tensor_split(root: str | Path) -> Path:
    root = Path(root)
    candidates = [root, root / "test", root / "validation", root / "val", root / "train"]
    required = ("event_times.pt", "event_types.pt")
    for candidate in candidates:
        if all((candidate / name).exists() for name in required):
            return candidate
    raise FileNotFoundError(f"Could not find a tensor split under {root}.")


def load_hawkes_tensors(root: str | Path) -> dict[str, torch.Tensor]:
    split_dir = discover_tensor_split(root)
    tensors = {}
    for tensor_path in sorted(split_dir.glob("*.pt")):
        tensors[tensor_path.stem] = torch.load(tensor_path, map_location="cpu")
    return tensors


def create_evaluation_times(
    inference_event_times: torch.Tensor,
    inference_seq_lengths: torch.Tensor,
    num_points_between_events: int = 10,
) -> torch.Tensor:
    batch_size, num_paths, max_len, _ = inference_event_times.shape
    device = inference_event_times.device
    max_points = max_len + max_len * max(1, num_points_between_events)
    evaluation_times = torch.zeros(batch_size, num_paths, max_points, device=device, dtype=inference_event_times.dtype)

    for batch_idx in range(batch_size):
        for path_idx in range(num_paths):
            seq_len = int(inference_seq_lengths[batch_idx, path_idx].item())
            if seq_len <= 0:
                continue

            path_times = inference_event_times[batch_idx, path_idx, :seq_len, 0]
            pieces = [torch.tensor([0.0], device=device, dtype=path_times.dtype)]
            previous = 0.0

            for time_value in path_times:
                time_float = float(time_value.item())
                if time_float > previous:
                    interval = torch.linspace(
                        previous,
                        time_float,
                        num_points_between_events + 2,
                        device=device,
                        dtype=path_times.dtype,
                    )[1:-1]
                    if interval.numel():
                        pieces.append(interval)
                pieces.append(time_value.unsqueeze(0))
                previous = time_float

            tail_end = previous * 1.05 if previous > 0 else 1.0
            tail = torch.linspace(
                previous,
                tail_end,
                num_points_between_events + 2,
                device=device,
                dtype=path_times.dtype,
            )[1:]
            if tail.numel():
                pieces.append(tail)

            combined = torch.unique(torch.cat(pieces), sorted=True)
            evaluation_times[batch_idx, path_idx, : combined.numel()] = combined[:max_points]

    return evaluation_times


def prepare_hawkes_batch(
    tensors: dict[str, torch.Tensor],
    sample_idx: int = 0,
    inference_path_idx: int = 0,
    num_points_between_events: int = 10,
) -> dict[str, torch.Tensor]:
    sample = {}
    for key, value in tensors.items():
        if torch.is_tensor(value):
            sample[key] = value[sample_idx : sample_idx + 1]

    event_times = sample["event_times"]
    event_types = sample["event_types"]
    seq_lengths = sample.get("seq_lengths")
    if seq_lengths is None:
        seq_lengths = torch.full(event_times.shape[:2], event_times.shape[2], dtype=torch.long)

    total_paths = event_times.shape[1]
    if total_paths < 2:
        raise ValueError("Point-process tutorials require at least two paths so one can be held out for inference.")
    if inference_path_idx >= total_paths:
        raise IndexError(f"inference_path_idx {inference_path_idx} is out of range for {total_paths} paths.")

    all_indices = torch.arange(total_paths)
    context_mask = all_indices != inference_path_idx
    inference_mask = all_indices == inference_path_idx

    batch = {
        "context_event_times": event_times[:, context_mask],
        "context_event_types": event_types[:, context_mask],
        "context_seq_lengths": seq_lengths[:, context_mask],
        "inference_event_times": event_times[:, inference_mask],
        "inference_event_types": event_types[:, inference_mask],
        "inference_seq_lengths": seq_lengths[:, inference_mask],
    }

    batch["intensity_evaluation_times"] = create_evaluation_times(
        batch["inference_event_times"],
        batch["inference_seq_lengths"],
        num_points_between_events=num_points_between_events,
    )

    for key in ("base_intensity_functions", "kernel_functions", "kernel_grids", "time_offsets"):
        if key in sample:
            batch[key] = sample[key]

    if "base_intensity_functions" in sample:
        batch["num_marks"] = int(sample["base_intensity_functions"].shape[1])
    else:
        batch["num_marks"] = int(torch.max(event_types).item()) + 1

    return batch


def plot_intensity_comparison(model_output: dict, batch: dict, path_idx: int = 0):
    predicted = model_output["predicted_intensity_values"].detach().cpu()
    target = model_output.get("target_intensity_values")
    if target is not None:
        target = target.detach().cpu()

    evaluation_times = batch["intensity_evaluation_times"].detach().cpu()
    inference_event_times = batch["inference_event_times"].detach().cpu()
    inference_event_types = batch["inference_event_types"].detach().cpu()
    inference_seq_lengths = batch["inference_seq_lengths"].detach().cpu()

    num_marks = predicted.shape[1]
    fig, axes = plt.subplots(num_marks, 1, figsize=(12, max(3.0, 2.2 * num_marks)), sharex=True)
    if num_marks == 1:
        axes = [axes]

    seq_len = int(inference_seq_lengths[0, path_idx].item())
    event_times = inference_event_times[0, path_idx, :seq_len, 0]
    event_types = inference_event_types[0, path_idx, :seq_len, 0]
    valid_eval = evaluation_times[0, path_idx]
    valid_eval = valid_eval[valid_eval > 0]

    for mark_idx, axis in enumerate(axes):
        pred_curve = predicted[0, mark_idx, path_idx, : valid_eval.numel()]
        axis.plot(valid_eval.numpy(), pred_curve.numpy(), color="#0072B2", linewidth=2.0, label="FIM-PP")

        if target is not None:
            target_curve = target[0, mark_idx, path_idx, : valid_eval.numel()]
            axis.plot(valid_eval.numpy(), target_curve.numpy(), color="#111111", linestyle="--", linewidth=1.6, label="Ground truth")

        mark_event_times = event_times[event_types == mark_idx]
        if mark_event_times.numel():
            axis.scatter(mark_event_times.numpy(), torch.zeros_like(mark_event_times).numpy(), color="#D55E00", s=20, label="Events")

        axis.set_ylabel(f"mark {mark_idx}")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("time")
    fig.tight_layout()
    return fig
