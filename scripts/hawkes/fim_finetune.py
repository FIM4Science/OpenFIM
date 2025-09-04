#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune a trained FIM Hawkes model on CDiff-style datasets using NLL-only.

Supported data inputs:
- CDiff format: a directory containing `train.pkl` with a list of sequences.
  Each sequence is a list of dicts with keys like `time_since_start` and `type_event`.
  We shift times to start at zero (like other scripts) and build tensors.
- Optional PT format: if `event_times.pt` and `event_types.pt` exist in the directory
  they will be used instead. These should have shapes [P, L] or [P, L, 1], and we add [B=1,...].
 - EasyTPP on Hugging Face Hub: pass a dataset id like "easytpp/retweet" (or just
   "retweet" which will be coerced to "easytpp/retweet"). We load the train split
   as multi-path sequences. If a validation split is available ("validation" or "test"),
   it's used for validation NLL.

Usage example:
  python3 scripts/hawkes/fim_finetune.py \
    --config configs/train/hawkes/david_small.yaml \
    --dataset easytpp/taobao \
    --epochs 500 \
    --lr 5e-5 \
    --save_dir results/finetuned \
    --max_paths 2000 \
    --max_events 100 \
    --resume_model results/FIM_Hawkes_10-22st_2000_paths_mixed_100_events_mixed-experiment-seed-10-dataset-dataset_kwargs-field_name_for_dimension_grouping-base_intensity_functions_08-24-1124/checkpoints/best-model
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import torch
from safetensors.torch import save_file as save_safetensors
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from fim.models.blocks import ModelFactory
from fim.utils.helper import GenericConfig, expand_params, load_yaml
from fim.utils.logging import RankLoggerAdapter, setup_logging


logger = RankLoggerAdapter(logging.getLogger(__name__))


def _ensure_4d_times(t: torch.Tensor) -> torch.Tensor:
    """Ensure event_times have shape [B, P, L, 1]. Input may be [P, L] or [P, L, 1]."""
    if t.dim() == 2:
        # [P, L] -> [1, P, L, 1]
        return t.unsqueeze(0).unsqueeze(-1)
    elif t.dim() == 3:
        # [P, L, 1] -> [1, P, L, 1]
        return t.unsqueeze(0)
    elif t.dim() == 4:
        return t
    else:
        raise ValueError(f"Unsupported event_times tensor shape {tuple(t.shape)}")


def _ensure_4d_types(t: torch.Tensor) -> torch.Tensor:
    """Ensure event_types have shape [B, P, L, 1]. Input may be [P, L] or [P, L, 1]."""
    if t.dim() == 2:
        return t.unsqueeze(0).unsqueeze(-1).long()
    elif t.dim() == 3:
        return t.unsqueeze(0).long()
    elif t.dim() == 4:
        return t.long()
    else:
        raise ValueError(f"Unsupported event_types tensor shape {tuple(t.shape)}")


def _infer_seq_lengths(event_times_bpl1: torch.Tensor) -> torch.Tensor:
    """Infer sequence lengths per path from times [B, P, L, 1] -> [B, P].

    Heuristic: count strictly positive entries per sequence along L. If none are positive,
    fall back to full length.
    """
    B, P, L = event_times_bpl1.shape[:3]
    times = event_times_bpl1.squeeze(-1)
    positive_mask = times > 0.0
    lengths = positive_mask.sum(dim=2)
    # If a path has zero positives but contains non-decreasing times, fall back to L
    zero_len_mask = lengths == 0
    if zero_len_mask.any():
        lengths = torch.where(zero_len_mask, torch.full_like(lengths, L), lengths)
    return lengths


def _build_batch(
    event_times_1pl1: torch.Tensor,
    event_types_1pl1: torch.Tensor,
    seq_lengths_1p: Optional[torch.Tensor],
    target_path_idx: int,
    device: torch.device,
) -> Tuple[dict, torch.Tensor, torch.Tensor]:
    """Construct a model input dict x for NLL-only training with one target path."""
    # Input shapes: [1, P, L, 1]
    _, P, L, _ = event_times_1pl1.shape

    # Split into context (all except target) and inference (target)
    ctx_indices = [i for i in range(P) if i != target_path_idx]
    if len(ctx_indices) == 0:
        raise ValueError("CDiff dataset must contain at least 2 paths (P>=2) for context/inference split.")

    context_times = event_times_1pl1[:, ctx_indices, :, :].to(device)
    context_types = event_types_1pl1[:, ctx_indices, :, :].to(device)
    inference_times = event_times_1pl1[:, target_path_idx : target_path_idx + 1, :, :].to(device)
    inference_types = event_types_1pl1[:, target_path_idx : target_path_idx + 1, :, :].to(device)

    if seq_lengths_1p is None:
        all_lengths = _infer_seq_lengths(event_times_1pl1)
    else:
        all_lengths = seq_lengths_1p

    context_lengths = all_lengths[:, ctx_indices].to(device)
    inference_lengths = all_lengths[:, target_path_idx : target_path_idx + 1].to(device)

    # Dummy evaluation times, forward() will concatenate history event times for evaluation
    intensity_eval = torch.zeros(1, 1, 1, device=device)

    x = {
        "context_event_times": context_times,
        "context_event_types": context_types,
        "context_seq_lengths": context_lengths,
        "inference_event_times": inference_times,
        "inference_event_types": inference_types,
        "inference_seq_lengths": inference_lengths,
        "intensity_evaluation_times": intensity_eval,
    }
    return x, context_lengths, inference_lengths


def _load_cdiff_train(dataset_dir: Path) -> Dict[str, List]:
    """Load CDiff train split from `<dataset_dir>/train.pkl` and return lists.

    Returns a dict with keys: time_since_start, type_event, seq_len.
    Times are shifted so the first event starts at 0.
    """
    import pickle

    pkl_path = dataset_dir / "train.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"CDiff train split not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    seqs_raw = d.get("train")
    if not isinstance(seqs_raw, list):
        raise ValueError(f"Unexpected 'train' content in {pkl_path}: {type(seqs_raw)}")

    time_since_start: List[List[float]] = []
    type_event: List[List[int]] = []
    seq_len: List[int] = []

    for seq in seqs_raw:
        if not isinstance(seq, list) or len(seq) == 0:
            time_since_start.append([])
            type_event.append([])
            seq_len.append(0)
            continue
        first_time = float(seq[0].get("time_since_start", 0.0))
        times = [float(ev.get("time_since_start", 0.0)) - first_time for ev in seq]
        types = [int(ev.get("type_event", 0)) for ev in seq]
        time_since_start.append(times)
        type_event.append(types)
        seq_len.append(len(times))

    return {"time_since_start": time_since_start, "type_event": type_event, "seq_len": seq_len}


def _build_tensors_from_cdiff(train_data: Dict[str, List], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build [1, P, L, 1] event_times/types and [1, P] seq_lengths from CDiff lists."""
    times_list = train_data["time_since_start"]
    types_list = train_data["type_event"]
    lens_list = train_data["seq_len"]

    P = len(times_list)
    if P < 2:
        raise ValueError("CDiff train split must contain at least 2 sequences")
    max_L = max(lens_list) if lens_list else 0
    if max_L == 0:
        raise ValueError("CDiff train split appears empty (all sequences have length 0)")

    times_tensor = torch.zeros(1, P, max_L, 1, dtype=torch.float32, device=device)
    types_tensor = torch.zeros(1, P, max_L, 1, dtype=torch.long, device=device)
    lengths_tensor = torch.zeros(1, P, dtype=torch.long, device=device)

    for p in range(P):
        L = int(lens_list[p])
        lengths_tensor[0, p] = L
        if L > 0:
            times_tensor[0, p, :L, 0] = torch.tensor(times_list[p][:L], dtype=torch.float32, device=device)
            types_tensor[0, p, :L, 0] = torch.tensor([int(t) for t in types_list[p][:L]], dtype=torch.long, device=device)

    return times_tensor, types_tensor, lengths_tensor


def _load_cdiff_split(dataset_dir: Path, split: str) -> Dict[str, List]:
    """Generic CDiff split loader (expects `<split>.pkl` with list under key `<split>`)."""
    import pickle

    pkl_path = dataset_dir / f"{split}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"CDiff split not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    seqs_raw = d.get(split)
    if not isinstance(seqs_raw, list):
        raise ValueError(f"Unexpected '{split}' content in {pkl_path}: {type(seqs_raw)}")

    time_since_start: List[List[float]] = []
    type_event: List[List[int]] = []
    seq_len: List[int] = []

    for seq in seqs_raw:
        if not isinstance(seq, list) or len(seq) == 0:
            time_since_start.append([])
            type_event.append([])
            seq_len.append(0)
            continue
        first_time = float(seq[0].get("time_since_start", 0.0))
        times = [float(ev.get("time_since_start", 0.0)) - first_time for ev in seq]
        types = [int(ev.get("type_event", 0)) for ev in seq]
        time_since_start.append(times)
        type_event.append(types)
        seq_len.append(len(times))

    return {"time_since_start": time_since_start, "type_event": type_event, "seq_len": seq_len}


def _apply_limits_to_cdiff_lists(data: Dict[str, List], max_paths: Optional[int], max_events: Optional[int]) -> Dict[str, List]:
    out = {k: v[:] for k, v in data.items()}
    if max_paths is not None and max_paths > 0:
        for k in ["time_since_start", "type_event", "seq_len"]:
            out[k] = out[k][:max_paths]
    if max_events is not None and max_events > 0:
        truncated_times = []
        truncated_types = []
        truncated_lens = []
        for times, types, length in zip(out["time_since_start"], out["type_event"], out["seq_len"]):
            L_new = min(int(length), int(max_events))
            truncated_times.append(times[:L_new])
            truncated_types.append(types[:L_new])
            truncated_lens.append(L_new)
        out["time_since_start"] = truncated_times
        out["type_event"] = truncated_types
        out["seq_len"] = truncated_lens
    return out


def _load_easytpp_split(dataset: str, split: str) -> Dict[str, List]:
    """Load an EasyTPP dataset split from Hugging Face as CDiff-style lists.

    If the provided dataset id does not start with "easytpp/", it will be prefixed.
    """
    try:
        from datasets import load_dataset  # Lazy import to avoid hard dependency for local-only runs
    except Exception as e:
        raise ImportError("The 'datasets' package is required for Hugging Face loading. Install with 'pip install datasets'.") from e

    dataset_id = dataset if str(dataset).startswith("easytpp/") else f"easytpp/{dataset}"
    ds = load_dataset(dataset_id, split=split)
    # Convert to dict-of-lists; keys expected: time_since_start, type_event, seq_len
    data_slice = ds[: len(ds)] if hasattr(ds, "__len__") else ds[:]
    # Some HF datasets may name length differently; prefer provided seq_len if present, else derive
    time_since_start = data_slice.get("time_since_start") or []
    type_event = data_slice.get("type_event") or []
    seq_len = data_slice.get("seq_len") or [len(t) for t in time_since_start]
    return {"time_since_start": time_since_start, "type_event": type_event, "seq_len": seq_len}


@click.command()
@click.option("--config", "cfg_path", required=False, type=click.Path(exists=True), help="Path to hawkes YAML config (optional)")
@click.option(
    "--dataset",
    required=True,
    type=str,
    help=(
        "Dataset identifier or path. Use a local folder with CDiff/PT format or an EasyTPP Hugging Face id "
        "like 'easytpp/retweet' (or just 'retweet')."
    ),
)
@click.option("--resume_model", default=None, type=click.Path(exists=True), help="Path to model-checkpoint.pth to load")
@click.option("--epochs", default=1000, type=int, help="Fine-tuning epochs")
@click.option("--lr", default=5e-5, type=float, help="Learning rate")
@click.option("--weight_decay", default=1e-4, type=float, help="Weight decay")
@click.option("--save_dir", default=None, type=click.Path(), help="Directory to save fine-tuned checkpoints")
@click.option("--target_idx", default=None, type=int, help="Fixed target path index (overrides random choice)")
@click.option("--max_paths", default=None, type=int, help="Limit number of paths (P) used for fine-tuning")
@click.option("--max_events", default=None, type=int, help="Limit number of events per path (truncate L)")
@click.option("--val_integration_points", default=5000, type=int, help="Monte Carlo samples for validation NLL integral")
@click.option("--deterministic_val", is_flag=True, default=True, help="Use fixed RNG seed for validation NLL")
@click.option("--grad-accum-steps", default=1, type=int, help="Gradient accumulation steps per epoch")
def main(
    cfg_path: Optional[str],
    dataset: str,
    resume_model: Optional[str],
    epochs: int,
    lr: float,
    weight_decay: float,
    save_dir: Optional[str],
    target_idx: Optional[int],
    max_paths: Optional[int],
    max_events: Optional[int],
    val_integration_points: int,
    deterministic_val: bool,
    grad_accum_steps: int,
):
    setup_logging()

    # Load config and instantiate model
    config: Optional[GenericConfig] = None
    if cfg_path is not None:
        cfg_raw = load_yaml(Path(cfg_path))
        configs = expand_params(cfg_raw)
        if len(configs) == 0:
            raise ValueError("No configs produced by expand_params. Check your YAML.")
        if len(configs) > 1:
            logger.warning("Multiple configs detected (%d). Fine-tune will use the first one.", len(configs))
        config = configs[0]

    # Strictly load model from checkpoint config when resume_model is provided,
    # but fall back to YAML config when the checkpoint config is missing fields
    # like 'model_type'. This improves robustness for older checkpoints.
    if resume_model is not None:
        resume_path = Path(resume_model)
        ckpt_dir = resume_path if resume_path.is_dir() else resume_path.parent
        ckpt_config_path = ckpt_dir / "config.json"
        if not ckpt_config_path.exists():
            raise FileNotFoundError(f"Checkpoint config.json not found: {ckpt_config_path}")
        with open(ckpt_config_path, "r", encoding="utf-8") as f:
            ckpt_cfg_dict = json.load(f)
        mt = ckpt_cfg_dict.get("model_type")
        if not isinstance(mt, str) or not mt:
            # Prefer loading directly from checkpoint by inferring model type.
            # Hawkes finetune supports only FIMHawkes, so default to that.
            logger.warning("Checkpoint config missing 'model_type'. Assuming 'fimhawkes' and proceeding with checkpoint config.")
            ckpt_cfg_dict["model_type"] = "fimhawkes"
        logger.info("Creating model from checkpoint config (%s)", ckpt_cfg_dict.get("model_type"))
        model = ModelFactory.create(ckpt_cfg_dict)
    else:
        if config is None:
            raise ValueError("No --resume_model provided and no --config YAML; cannot instantiate model.")
        logger.info("Creating model from YAML config (%s)", config.model.model_type)
        model = ModelFactory.create(config.model.to_dict())

    # Load weights if provided (supports file or directory path)
    # Track which weights file was actually loaded (for metadata)
    loaded_weights_path: Optional[Path] = None
    if resume_model is not None:
        resume_path = Path(resume_model)
        logger.info("Loading model weights from %s", resume_path)
        state = None
        if resume_path.is_dir():
            candidates = [
                resume_path / "model-checkpoint.pth",
                resume_path / "model.safetensors",
                resume_path / "pytorch_model.bin",
            ]
            found = None
            for c in candidates:
                if c.exists():
                    found = c
                    break
            if found is None:
                raise FileNotFoundError(f"No model file found in {resume_path}; expected one of {', '.join(str(c) for c in candidates)}")
            loaded_weights_path = found
            if found.suffix == ".safetensors":
                from safetensors.torch import load_file as _load_safetensors

                state = _load_safetensors(found)
            else:
                state = torch.load(found, map_location="cpu", weights_only=False)
        else:
            loaded_weights_path = resume_path
            state = torch.load(resume_path, map_location="cpu", weights_only=False)

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logger.warning("Missing keys when loading state_dict: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys when loading state_dict: %s", unexpected)

    # Device
    if config is not None:
        device = torch.device("cuda" if torch.cuda.is_available() and getattr(config.experiment, "device_map", "cuda") != "cpu" else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    # Enable bf16 autocast on CUDA to reduce memory (matches trainer bf16_mixed)
    use_bf16_autocast = device.type == "cuda"

    def _save_checkpoint(save_dir: Path):
        save_dir.mkdir(parents=True, exist_ok=True)
        # Save HF-style config for downstream loaders
        model.config.to_json_file(str(save_dir / "config.json"))
        # Save safetensors (CPU tensors)
        state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        save_safetensors(state_cpu, str(save_dir / "model.safetensors"))
        # Also save a conventional PyTorch checkpoint for convenience
        torch.save(state_cpu, save_dir / "model-checkpoint.pth")

    # Load dataset (supports local CDiff/PT folders or EasyTPP HF datasets)
    dataset_arg = str(dataset)
    dpath = Path(dataset_arg)
    if dpath.exists():
        # Local directory modes
        dataset_name = dpath.name
        if (dpath / "train.pkl").exists():
            train_lists = _apply_limits_to_cdiff_lists(_load_cdiff_train(dpath), max_paths, max_events)
            event_times, event_types, seq_lengths = _build_tensors_from_cdiff(train_lists, device)
        else:
            times_path = dpath / "event_times.pt"
            types_path = dpath / "event_types.pt"
            lengths_path = dpath / "seq_lengths.pt"
            if not times_path.exists() or not types_path.exists():
                raise FileNotFoundError(f"Expected either train.pkl or (event_times.pt & event_types.pt) in {dpath}")
            event_times = _ensure_4d_times(torch.load(times_path, map_location="cpu").float()).to(device)
            event_types = _ensure_4d_types(torch.load(types_path, map_location="cpu")).to(device)
            seq_lengths = torch.load(lengths_path, map_location="cpu") if lengths_path.exists() else None
            if seq_lengths is not None:
                if seq_lengths.dim() == 1:  # [P] -> [1, P]
                    seq_lengths = seq_lengths.unsqueeze(0)
                elif seq_lengths.dim() != 2:
                    raise ValueError(f"Unsupported seq_lengths shape {tuple(seq_lengths.shape)}; expected [P] or [B=1,P]")
                seq_lengths = seq_lengths.to(device)

            # Apply limits if provided
            _, P_full, L_full, _ = event_times.shape
            if max_paths is not None and max_paths > 0 and max_paths < P_full:
                event_times = event_times[:, :max_paths, :, :]
                event_types = event_types[:, :max_paths, :, :]
                if seq_lengths is not None:
                    seq_lengths = seq_lengths[:, :max_paths]
            if max_events is not None and max_events > 0 and max_events < L_full:
                event_times = event_times[:, :, :max_events, :]
                event_types = event_types[:, :, :max_events, :]
                if seq_lengths is not None:
                    seq_lengths = torch.minimum(seq_lengths, torch.tensor(max_events, device=seq_lengths.device))
        # Validation for local: optional dev.pkl
        has_val = (dpath / "dev.pkl").exists()
        if has_val:
            dev_lists = _apply_limits_to_cdiff_lists(_load_cdiff_split(dpath, "dev"), max_paths, max_events)
            val_event_times, val_event_types, val_seq_lengths = _build_tensors_from_cdiff(dev_lists, device)
            P_val = int(val_event_times.shape[1])
    else:
        # Hugging Face EasyTPP dataset
        dataset_id = dataset_arg if dataset_arg.startswith("easytpp/") else f"easytpp/{dataset_arg}"
        dataset_name = dataset_id.split("/")[-1]
        train_lists = _apply_limits_to_cdiff_lists(_load_easytpp_split(dataset_id, "train"), max_paths, max_events)
        event_times, event_types, seq_lengths = _build_tensors_from_cdiff(train_lists, device)
        # Try validation: prefer 'validation', then 'test'
        has_val = False
        val_event_times = val_event_types = val_seq_lengths = None  # type: ignore
        try:
            dev_lists = _apply_limits_to_cdiff_lists(_load_easytpp_split(dataset_id, "validation"), max_paths, max_events)
            val_event_times, val_event_types, val_seq_lengths = _build_tensors_from_cdiff(dev_lists, device)
            has_val = True
        except Exception:
            try:
                dev_lists = _apply_limits_to_cdiff_lists(_load_easytpp_split(dataset_id, "test"), max_paths, max_events)
                val_event_times, val_event_types, val_seq_lengths = _build_tensors_from_cdiff(dev_lists, device)
                has_val = True
            except Exception:
                has_val = False
        if has_val:
            P_val = int(val_event_times.shape[1])

    # Determine initial target path index (may be overridden per-epoch below)
    P = int(event_times.shape[1])
    if target_idx is None:
        target_idx = torch.randint(low=0, high=P, size=(1,)).item()
        logger.info("Initial training target path index (will resample each epoch): %d", target_idx)
    else:
        if not (0 <= target_idx < P):
            raise ValueError(f"target_idx {target_idx} out of range [0, {P})")
        logger.info("Using provided initial training target path index (will resample each epoch): %d", target_idx)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Simple training loop
    best_loss: float = float("inf")
    # Create timestamped run directory under the chosen save root
    save_root_base = (Path(save_dir) if save_dir is not None else (Path("results") / "finetuned_cdiff")) / dataset_name
    timestamp = datetime.now().strftime("%y%m%d-%H%M")
    save_root = save_root_base / timestamp
    save_root.mkdir(parents=True, exist_ok=True)
    logger.info("Saving fine-tune artifacts under: %s", str(save_root))
    # Persist metadata about this fine-tuning run (including base model used)
    try:
        dataset_meta_key = "dataset_dir" if Path(str(dataset)).exists() else "dataset_id"
        dataset_meta_val = str(Path(str(dataset)).resolve()) if dataset_meta_key == "dataset_dir" else str(dataset)
        meta = {
            dataset_meta_key: dataset_meta_val,
            "dataset_name": dataset_name,
            "base_model_path": str(loaded_weights_path.resolve()) if loaded_weights_path is not None else None,
            "resume_model_arg": str(resume_model) if resume_model is not None else None,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "epochs": epochs,
        }
        with open(save_root / "finetune_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        logger.warning("Failed to write finetune_meta.json due to %s", e)
    writer = SummaryWriter(log_dir=str(save_root / "tensorboard"))

    # Prepare optional validation set already handled above for both local/HF

    for epoch in range(1, epochs + 1):
        # Gradient accumulation to simulate larger batch size without increasing memory
        optimizer.zero_grad(set_to_none=True)
        epoch_loss_sum: float = 0.0
        nll_train = float("nan")
        steps_this_epoch = max(1, int(grad_accum_steps))
        for _ in range(steps_this_epoch):
            # 1) Training micro-step: resample target path index each iteration
            P = int(event_times.shape[1])
            target_idx_epoch = torch.randint(low=0, high=P, size=(1,)).item()
            x, _, _ = _build_batch(event_times, event_types, seq_lengths, target_path_idx=target_idx_epoch, device=device)

            # Forward with bf16 autocast to reduce activation memory
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16_autocast):
                out = model(x)
            losses = out["losses"]
            # Scale the loss to average across micro-steps
            loss = losses["loss"] / steps_this_epoch
            epoch_loss_sum += float(loss.detach().cpu())
            nll_train = float(losses.get("nll_loss", float("nan")))
            loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # Validation NLL
        if has_val:
            model.eval()
            with torch.no_grad():
                # 2) Validation step: evaluate NLL across all possible target paths and average
                nll_vals: List[float] = []
                for p_idx in range(P_val):
                    x_val, _, _ = _build_batch(val_event_times, val_event_types, val_seq_lengths, p_idx, device)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16_autocast):
                        out_val = model(x_val)
                    try:
                        if deterministic_val:
                            with torch.random.fork_rng(devices=[device], enabled=True):
                                torch.manual_seed(12345)
                                event_times_for_nll = (
                                    x_val.get("inference_event_times_norm", x_val["inference_event_times"])
                                    if model.normalize_times
                                    else x_val["inference_event_times"]
                                )
                                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16_autocast):
                                    nll_one = (
                                        model._nll_loss(
                                            intensity_fn=out_val["intensity_function"],
                                            event_times=event_times_for_nll,
                                            event_types=x_val["inference_event_types"].squeeze(-1),
                                            seq_lengths=x_val["inference_seq_lengths"],
                                            num_integration_points=val_integration_points,
                                        )
                                        .detach()
                                        .item()
                                    )
                        else:
                            event_times_for_nll = (
                                x_val.get("inference_event_times_norm", x_val["inference_event_times"])
                                if model.normalize_times
                                else x_val["inference_event_times"]
                            )
                            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16_autocast):
                                nll_one = (
                                    model._nll_loss(
                                        intensity_fn=out_val["intensity_function"],
                                        event_times=event_times_for_nll,
                                        event_types=x_val["inference_event_types"].squeeze(-1),
                                        seq_lengths=x_val["inference_seq_lengths"],
                                        num_integration_points=val_integration_points,
                                    )
                                    .detach()
                                    .item()
                                )
                    except Exception:
                        nll_one = out_val["losses"].get("nll_loss", float("nan"))
                    if nll_one == nll_one:
                        nll_vals.append(float(nll_one))
                if len(nll_vals) > 0:
                    nll_val = float(sum(nll_vals) / len(nll_vals))
                else:
                    nll_val = float("nan")
            model.train()
        else:
            nll_val = float("nan")

        logger.info(
            "Epoch %d/%d - loss: %.6f, nll_train: %.6f, nll_val: %.6f",
            epoch,
            epochs,
            float(epoch_loss_sum),
            float(nll_train),
            float(nll_val),
        )
        # TensorBoard scalars
        writer.add_scalar("train/loss", float(epoch_loss_sum), epoch)
        if nll_train == nll_train:  # not NaN
            writer.add_scalar("train/nll", float(nll_train), epoch)
        if has_val and nll_val == nll_val:
            writer.add_scalar("val/nll", float(nll_val), epoch)
        # log current LR from optimizer
        try:
            current_lr = next(iter(optimizer.param_groups)).get("lr", lr)
            writer.add_scalar("train/lr", current_lr, epoch)
        except Exception:
            pass

        # Save best and periodic
        is_best = float(epoch_loss_sum) < best_loss
        if is_best:
            best_loss = float(epoch_loss_sum)
            best_dir = save_root / "best-model"
            _save_checkpoint(best_dir)

        if epoch % 50 == 0 or epoch == epochs:
            epoch_dir = save_root / f"epoch-{epoch}"
            _save_checkpoint(epoch_dir)

    writer.flush()
    writer.close()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
