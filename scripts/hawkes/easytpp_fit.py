"""
Run examples:
  Local Hawkes dataset:
    python scripts/hawkes/easytpp_fit.py \
      data/synthetic_data/hawkes/EVAL_10_3D_1k_paths_diag_only_large_scale \
      --sample-idx 0 --model NHP --epochs 100 --batch-size 256 --max-num-events 100
  HuggingFace EasyTPP dataset:
    python scripts/hawkes/easytpp_fit.py easytpp/retweet \
      --model NHP --epochs 100 --batch-size 256 --max-num-events 100

Script to fit an EasyTPP model on either
1. a *local* Hawkes dataset that is stored the way this repository produces it, i.e. a
   folder that contains the files

       event_times.pt  –  shape [B, P, L, 1]
       event_types.pt  –  shape [B, P, L, 1]
       seq_lengths.pt  –  shape [B, P]

   where B denotes the number of *samples* (also called *bulks* in other parts of the
   code base).  Each sample in turn contains ``P`` independent realisations of a multi
   dimensional Hawkes process.

   For *local* datasets only **one** sample (the first axis) is selected via the
   ``--sample-idx`` argument and converted to the JSON‐Lines format expected by
   EasyTPP.  Every path (dimension ``P``) becomes one sequence in the resulting file.

2. a dataset that is hosted on the HuggingFace hub and is already provided in the
   EasyTPP format (e.g. *"FIM4Science/easytpp-synthetic-1d"*).  These datasets can be
   passed through untouched.

After the conversion a temporary YAML configuration file is generated and the EasyTPP
CLI is invoked to actually run the training.  The script itself stays **agnostic** of
the concrete model – you may specify any EasyTPP model id via ``--model``.

The script intentionally keeps dependencies minimal.  Only *torch*, *PyYAML* and the

    datasets  (🤗)
    easytpp

packages must be importable in the chosen environment.  The latter two are available
in the *model_training* conda env as stated in the project documentation.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Union


def parse_args() -> argparse.Namespace:  # noqa: D401
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser("Train an EasyTPP model on a selected dataset")

    parser.add_argument(
        "dataset",
        type=str,
        help=(
            "Either a local path that contains a Hawkes dataset in .pt format or the "
            "name of a HuggingFace dataset (e.g. 'FIM4Science/easytpp-synthetic-1d')."
        ),
    )
    parser.add_argument(
        "--sample-idx",
        type=int,
        default=0,
        help="Index along the *batch* axis (B) that selects the sample of interest in local datasets.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="NHP",
        help="EasyTPP model id to use (e.g. NHP, SAHP, THP, …).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs (kept small as a sensible default).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size passed to EasyTPP.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory where EasyTPP stores its checkpoints.",
    )
    parser.add_argument(
        "--max-num-events",
        type=int,
        default=100,
        help="Maximum number of events per sequence; longer sequences will be truncated.",
    )

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Conversion helpers for local datasets
# -----------------------------------------------------------------------------


def _load_local_sample(sample_dir: Path, sample_idx: int) -> Dict[str, List[List]]:
    """Load a *single* sample from a local Hawkes dataset and return it as seq dict.

    The returned dictionary has the keys ``time_seqs`` and ``type_seqs`` where each
    value is a list with length *P* – one entry per path.  The inner lists contain the
    event times / event types for the respective path.
    """

    import torch  # local import to avoid unnecessary global dependency

    # ------------------------------------------------------------
    # Required tensor files
    # ------------------------------------------------------------
    event_times_t = torch.load(sample_dir / "event_times.pt")  # [B, P, L, 1]
    event_types_t = torch.load(sample_dir / "event_types.pt")  # [B, P, L, 1]
    seq_lengths_t = torch.load(sample_dir / "seq_lengths.pt")  # [B, P]

    if sample_idx >= event_times_t.shape[0]:
        raise IndexError(f"sample_idx {sample_idx} exceeds available samples ({event_times_t.shape[0]}).")

    # Select the wanted sample along the *B* axis.
    times_sample = event_times_t[sample_idx]  # shape [P, L, 1]
    types_sample = event_types_t[sample_idx]  # shape [P, L, 1]
    lengths_sample = seq_lengths_t[sample_idx]  # shape [P]

    time_since_start: List[List[float]] = []
    type_event: List[List[int]] = []
    time_since_last_event: List[List[float]] = []

    for path_idx in range(times_sample.shape[0]):
        seq_len = int(lengths_sample[path_idx].item())
        if seq_len == 0:
            # Skip empty sequences although this should not happen.
            continue
        times = times_sample[path_idx, :seq_len, 0].tolist()
        types = [int(t) for t in types_sample[path_idx, :seq_len, 0].tolist()]

        time_since_start.append(times)
        type_event.append(types)

        # compute delta times, first event delta 0
        deltas = [0.0]
        deltas.extend([t2 - t1 for t1, t2 in zip(times[:-1], times[1:])])
        time_since_last_event.append(deltas)

    # How many *unique* event types does this dataset contain?
    num_event_types = int(event_types_t.unique().numel())

    return {
        "time_since_start": time_since_start,
        "type_event": type_event,
        "time_since_last_event": time_since_last_event,
        "num_event_types": num_event_types,
    }


def convert_local_dataset_to_json(
    base_dir: Path,
    split: str,
    sample_idx: int,
    tmp_dir: Path,
    max_num_events: int | None = None,
) -> Path:
    """Convert the given *split* of a local dataset to JSON-Lines format.

    Returns
    -------
    Path
        Path to the freshly written ``*.json`` file that stores the converted data.
    """

    sample_dir = base_dir / split
    if not sample_dir.exists():
        raise FileNotFoundError(f"Expected split directory '{sample_dir}' not found.")

    sample_dict = _load_local_sample(sample_dir, sample_idx)

    # Truncate sequences longer than max_num_events if specified
    if max_num_events is not None:
        sample_dict["time_since_start"] = [seq[:max_num_events] for seq in sample_dict["time_since_start"]]
        sample_dict["type_event"] = [seq[:max_num_events] for seq in sample_dict["type_event"]]
        sample_dict["time_since_last_event"] = [seq[:max_num_events] for seq in sample_dict["time_since_last_event"]]

    json_path = tmp_dir / f"{split}.json"
    with json_path.open("w") as fh:
        for idx in range(len(sample_dict["time_since_start"])):
            record = {
                "dim_process": sample_dict["num_event_types"],
                "time_since_start": sample_dict["time_since_start"][idx],
                "type_event": sample_dict["type_event"][idx],
                "time_since_last_event": sample_dict["time_since_last_event"][idx],
            }
            fh.write(json.dumps(record))
            fh.write("\n")

    # Also return the number of unique event types so that the caller can populate
    # the EasyTPP config accordingly.
    return json_path, sample_dict["num_event_types"]


def write_training_config(
    *,
    data_id: str,
    train_json: Union[str, Path],
    val_json: Union[str, Path],
    test_json: Union[str, Path],
    num_event_types: int,
    model_id: str,
    epochs: int,
    batch_size: int,
    checkpoint_dir: Path,
    sampler_dtime_max: float,  # <-- NEW ARGUMENT
    sampler_num_samples_boundary: int,  # <-- NEW ARGUMENT
    sampler_over_sample_rate: float,  # <-- NEW ARGUMENT
    gpu: int = -1,
) -> Path:
    """Render the YAML config file that EasyTPP expects."""

    # The template is now defined inside this function for clarity
    EASYTPP_TEMPLATE = """
    pipeline_config_id: runner_config

    data:
      {data_id}:
        data_format: json
        train_dir: {train_json}
        valid_dir: {val_json}
        test_dir: {test_json}
        data_specs:
          num_event_types: {num_event_types}
          pad_token_id: {pad_token_id}
          padding_side: right

    {model_id}_train:
      base_config:
        stage: train
        backend: torch
        dataset_id: {data_id}
        runner_id: std_tpp
        model_id: {model_id}
        base_dir: {checkpoint_dir}
      trainer_config:
        batch_size: {batch_size}
        max_epoch: {max_epoch}
        shuffle: False
        optimizer: adam
        learning_rate: 1.e-3
        valid_freq: 1
        use_tfb: True
        metrics: ['acc', 'rmse']
        seed: 42
        gpu: {gpu}
      model_config:
        hidden_size: 32
        loss_integral_num_sample_per_step: 20
        thinning:
          num_seq: 10
          num_sample: 1
          num_exp: 500
          look_ahead_time: 10
          patience_counter: 5
          over_sample_rate: {sampler_over_sample_rate}
          num_samples_boundary: {sampler_num_samples_boundary}
          dtime_max: {sampler_dtime_max}
          num_step_gen: 1
    """

    # +1 is a safety pad token id.
    pad_token_id = num_event_types

    rendered = EASYTPP_TEMPLATE.format(
        data_id=data_id,
        train_json=str(train_json),
        val_json=str(val_json),
        test_json=str(test_json),
        num_event_types=num_event_types,
        pad_token_id=pad_token_id,
        model_id=model_id,
        max_epoch=epochs,
        batch_size=batch_size,
        checkpoint_dir=checkpoint_dir,
        gpu=gpu,
        sampler_dtime_max=sampler_dtime_max,
        sampler_num_samples_boundary=sampler_num_samples_boundary,
        sampler_over_sample_rate=sampler_over_sample_rate,
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = checkpoint_dir / f"easytpp_{model_id.lower()}_train.yaml"
    cfg_path.write_text(rendered)
    return cfg_path


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------


def main() -> None:  # noqa: D401
    args = parse_args()

    # Ensure torch creates FloatTensors (float32) by default to avoid dtype mismatch
    try:
        import torch  # noqa: WPS433 (optional import)

        torch.set_default_dtype(torch.float64)
        torch.set_default_tensor_type(torch.DoubleTensor)
    except ImportError:
        pass

    dataset_arg = args.dataset
    dataset_path = Path(dataset_arg)

    # Prepare a tmp directory that will hold the converted dataset and config.
    tmp_dir = Path(tempfile.mkdtemp(prefix="easytpp_data_"))

    # For local datasets always use the "context" split; HF datasets use the built-in "train" split.
    local_train_split = "context"

    try:
        # ==================================================================
        # Calculate dynamic sampler params after args are parsed for fair comparison
        # ==================================================================
        print("[INFO] Calculating dynamic sampler parameters for fair comparison...")
        if dataset_path.exists():
            # Logic to load local train split (renamed to context) and find max dtime
            train_sample_dict = _load_local_sample(dataset_path / local_train_split, args.sample_idx)
            # Add 'if seq' to handle potentially empty sequences gracefully
            max_dtime_train = max(max(seq) for seq in train_sample_dict["time_since_last_event"] if seq)
        else:
            # Logic to load HF train split and find max dtime
            from datasets import load_dataset

            train_ds = load_dataset(dataset_arg, split="train")
            max_dtime_train = max(max(seq) for seq in train_ds["time_since_last_event"] if seq)

        # Harmonize sampler settings to match the FIM-Hawkes script
        sampler_dtime_max = float(max_dtime_train) * 1.2  # 20% safety margin
        sampler_num_samples_boundary = 50
        sampler_over_sample_rate = 5.0
        print(f"  - Set sampler dtime_max to {sampler_dtime_max:.4f}")
        print(f"  - Set sampler num_samples_boundary to {sampler_num_samples_boundary}")
        print(f"  - Set sampler over_sample_rate to {sampler_over_sample_rate:.1f}")
        # ==================================================================

        if dataset_path.exists():
            # ------------------------------------------------------------
            # LOCAL DATASET  –  needs conversion (train split renamed to context).
            # ------------------------------------------------------------
            train_json, num_event_types = convert_local_dataset_to_json(
                dataset_path,
                local_train_split,
                args.sample_idx,
                tmp_dir,
                args.max_num_events,
            )
            val_json, _ = convert_local_dataset_to_json(
                dataset_path,
                "val",
                args.sample_idx,
                tmp_dir,
                args.max_num_events,
            )
            test_json, _ = convert_local_dataset_to_json(
                dataset_path,
                "test",
                args.sample_idx,
                tmp_dir,
                args.max_num_events,
            )

        else:
            # ------------------------------------------------------------
            # HUGGINGFACE DATASET  –  we assume it is already compatible.
            # ------------------------------------------------------------
            # We do *not* download the dataset here – EasyTPP can deal with a HF id
            # directly.  Just reference the id in the config.
            try:
                from datasets import load_dataset  # noqa: WPS433 (third-party import)

                # Trigger a tiny, light-weight loading to find out how many event types
                # are present.  The first sample of the train split should be enough.
                ds = load_dataset(dataset_arg, split="train", streaming=False)
                first_item = ds[0]
                # Determine number of event types from dataset fields
                if "dim_process" in first_item:
                    num_event_types = int(first_item["dim_process"])
                elif "type_event" in first_item:
                    num_event_types = int(max(first_item["type_event"]) + 1)
                elif "type_seqs" in first_item:
                    num_event_types = int(max(first_item["type_seqs"]) + 1)
                else:
                    # Fallback: guess 2 event types
                    num_event_types = 2
            except Exception:  # pragma: no cover – robust against HF outages etc.
                # Fallback:  guess 2 event types which is the most common case in our
                # synthetic data.
                num_event_types = 2

            # Use the same HuggingFace dataset id for all splits; the runner API
            # will call load_dataset with the appropriate split.
            train_json = dataset_arg
            val_json = dataset_arg
            test_json = dataset_arg

        # ------------------------------------------------------------
        # Write YAML config and run EasyTPP.
        # ------------------------------------------------------------

        # Detect available GPU.  EasyTPP expects the index of the CUDA device or
        # -1 to enforce CPU execution.  We try to import torch here to check if
        # CUDA is available.  If the import fails (or CUDA is not available), we
        # gracefully fall back to CPU which equals the previous behaviour.

        try:
            import torch  # noqa: WPS433 (optional import for CUDA check)

            gpu_index = 0 if torch.cuda.is_available() else -1
        except Exception:  # pragma: no cover – torch might be absent
            gpu_index = -1

        # Choose data_id for logging clarity (local vs HF)
        data_id = "hawkes_local" if dataset_path.exists() else "easytpp"
        cfg_path = write_training_config(
            data_id=data_id,
            train_json=train_json,
            val_json=val_json,
            test_json=test_json,
            num_event_types=num_event_types,
            model_id=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            checkpoint_dir=args.output_dir,
            gpu=gpu_index,
            sampler_dtime_max=sampler_dtime_max,
            sampler_num_samples_boundary=sampler_num_samples_boundary,
            sampler_over_sample_rate=sampler_over_sample_rate,
        )

        print(f"[INFO] EasyTPP config written to: {cfg_path}")

        # ------------------------------------------------------------
        # Kick off training directly through the easy_tpp Python API.  This avoids
        # problems with module names (`easy_tpp` vs. `easytpp`) and eliminates the
        # need for spawning a subprocess.
        # ------------------------------------------------------------

        if os.getenv("TRAIN_EASYTPP", "1") != "0":
            try:
                from easy_tpp.config_factory import Config  # type: ignore
                from easy_tpp.runner import Runner  # type: ignore

                config = Config.build_from_yaml_file(str(cfg_path), experiment_id=f"{args.model}_train")
                runner = Runner.build_from_config(config)

                print("[INFO] Starting EasyTPP training via Python API …")
                runner.run()

                # ----------------------------------------------------
                # Optional: Evaluate the trained model (next-event
                # prediction) on the *test* split.  We reuse the *same*
                # configuration but run in VALIDATE phase to load the
                # best checkpoint and compute the default metrics
                # (RMSE, accuracy, …).
                # ----------------------------------------------------

                try:
                    from easy_tpp.runner.base_runner import RunnerPhase

                    print("[INFO] Starting EasyTPP evaluation (next-event prediction) …")

                    test_loader = runner._data_loader.test_loader()
                    # Use VALIDATE phase here to ensure proper masking of padded events when computing metrics
                    metrics = runner.run_one_epoch(test_loader, RunnerPhase.VALIDATE)

                    # Display standard metrics if present.
                    print("[INFO] Evaluation metrics:")
                    for key in ["rmse", "acc", "loglike", "num_events"]:
                        if key in metrics:
                            print(f"    {key}: {metrics[key]}")

                except Exception as eval_err:  # pragma: no cover – evaluation is optional
                    print("[WARNING] Automatic evaluation failed. Reason: " + str(eval_err))

            except ImportError as err:
                raise SystemExit(
                    "[ERROR] Could not import easy_tpp – please ensure that you are running inside the 'model_training' conda environment."
                ) from err
        else:
            print("[INFO] TRAIN_EASYTPP=0 → Skipping the actual training run.")

    finally:
        # Clean up temporary directory.  Comment-in the following line to *persist*
        # the converted data for debugging purposes.
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
