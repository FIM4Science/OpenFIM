#!/usr/bin/env python3
"""
Shift CDiff datasets so that each sequence's event times start at 0.

This script scans a root folder containing CDiff datasets (each a directory with
train.pkl/dev.pkl/val.pkl/test.pkl). For each split file, it loads the pickle,
expects a dictionary with a list under the key named like the split (e.g.,
"train" -> list[ list[ dict{"time_since_start", "type_event", ...} ] ]), and
writes back an updated pickle where every sequence's time_since_start is shifted
by subtracting the first event's time in that sequence.

We preserve any other top-level metadata (e.g., dim_process) and any additional
fields per-event. Only the numeric value stored under "time_since_start" is
shifted. Empty sequences are left as-is.

Usage examples:
  Dry-run (no writes):
    python scripts/helpers/cdiff_shift_times_to_zero.py \
      --root /cephfs/users/berghaus/FoundationModels/FIM/data/external/CDiff_dataset

  Apply changes, all datasets:
    python scripts/helpers/cdiff_shift_times_to_zero.py \
      --root /cephfs/users/berghaus/FoundationModels/FIM/data/external/CDiff_dataset --apply

  Apply changes, specific datasets only:
    python scripts/helpers/cdiff_shift_times_to_zero.py \
      --root /cephfs/users/berghaus/FoundationModels/FIM/data/external/CDiff_dataset \
      --apply --datasets amazon retweet
"""

from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, List


def shift_split_file(split_path: Path, split_key: str, apply: bool) -> Dict[str, Any]:
    """Shift times in one CDiff split file.

    Returns a summary dict with counts and a preview of before/after for the first
    non-empty sequence when available.
    """
    with open(split_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Unexpected pickle structure in {split_path}: {type(data)}")

    seqs = data.get(split_key)
    if not isinstance(seqs, list):
        raise ValueError(f"Expected list under key '{split_key}' in {split_path}, got {type(seqs)}")

    num_sequences = len(seqs)
    num_shifted = 0
    first_preview_before: List[float] | None = None
    first_preview_after: List[float] | None = None

    for idx, seq in enumerate(seqs):
        if not isinstance(seq, list) or len(seq) == 0:
            continue
        # Extract first time; support missing key with 0.0 default
        first_time_raw = seq[0].get("time_since_start", 0.0)
        try:
            first_time = float(first_time_raw)
        except Exception:
            # If unparsable, skip this sequence
            continue

        # Build shifted times (preview and optionally write)
        times_before: List[float] = []
        times_after: List[float] = []
        for ev in seq:
            t_raw = ev.get("time_since_start", 0.0)
            try:
                t_val = float(t_raw)
            except Exception:
                t_val = 0.0
            times_before.append(t_val)
            times_after.append(t_val - first_time)
            if apply:
                ev["time_since_start"] = t_val - first_time

        num_shifted += 1
        if first_preview_before is None and len(times_before) > 0:
            first_preview_before = times_before[: min(10, len(times_before))]
            first_preview_after = times_after[: min(10, len(times_after))]

    # If apply, write back with backup
    if apply and num_shifted > 0:
        backup_path = split_path.with_suffix(split_path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(split_path, backup_path)
        with open(split_path, "wb") as f:
            pickle.dump(data, f)

    return {
        "path": str(split_path),
        "split": split_key,
        "num_sequences": num_sequences,
        "num_nonempty_shifted": num_shifted,
        "preview_before": first_preview_before,
        "preview_after": first_preview_after,
        "applied": bool(apply and num_shifted > 0),
    }


def is_cdiff_dir(d: Path) -> bool:
    if not d.exists() or not d.is_dir():
        return False
    expected_any = [d / s for s in ("train.pkl", "val.pkl", "dev.pkl", "test.pkl")]
    return any(p.exists() for p in expected_any)


def main():
    ap = argparse.ArgumentParser(description="Shift CDiff event times to start at zero.")
    ap.add_argument("--root", type=str, required=True, help="Root directory containing CDiff dataset subfolders")
    ap.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of dataset subfolder names to process (e.g., amazon retweet). If omitted, process all.",
    )
    ap.add_argument("--apply", action="store_true", help="Apply changes. If omitted, run dry-run only.")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Root directory not found: {root}")

    # Collect candidate dataset dirs
    dataset_dirs: List[Path] = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and is_cdiff_dir(child):
            if args.datasets is None or child.name in set(args.datasets):
                dataset_dirs.append(child)

    if len(dataset_dirs) == 0:
        print("No CDiff datasets found to process.")
        return

    print(f"Found {len(dataset_dirs)} CDiff dataset(s) under {root}.")
    print(f"Mode: {'APPLY' if args.apply else 'DRY-RUN'}")

    splits = ["train", "val", "dev", "test"]
    summaries: List[Dict[str, Any]] = []

    for ddir in dataset_dirs:
        print(f"\n=== Dataset: {ddir.name} ===")
        for split in splits:
            split_file = ddir / f"{split}.pkl"
            if not split_file.exists():
                continue
            try:
                summary = shift_split_file(split_file, split, apply=args.apply)
                summaries.append(summary)
                prev_b = summary["preview_before"]
                prev_a = summary["preview_after"]
                preview_str = ""
                if prev_b is not None and prev_a is not None:
                    preview_str = f" preview(before->after first seq head): {prev_b} -> {prev_a}"
                print(
                    f"  {split}.pkl: sequences={summary['num_sequences']} nonempty_shifted={summary['num_nonempty_shifted']}"
                    f" applied={summary['applied']}{preview_str}"
                )
            except Exception as e:
                print(f"  {split}.pkl: ERROR {e}")

    # Final aggregate
    total_files = len(summaries)
    total_sequences = sum(int(s.get("num_sequences", 0)) for s in summaries)
    total_nonempty = sum(int(s.get("num_nonempty_shifted", 0)) for s in summaries)
    total_applied = sum(1 for s in summaries if bool(s.get("applied", False)))
    print(
        f"\nDone. Files processed: {total_files}, total sequences: {total_sequences}, "
        f"nonempty shifted: {total_nonempty}, files written: {total_applied}"
    )


if __name__ == "__main__":
    main()
