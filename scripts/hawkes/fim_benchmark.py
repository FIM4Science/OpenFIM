"""
Run a grid of next-event evaluations (inference only) and aggregate results.

Usage:
python3 scripts/hawkes/fim_benchmark.py \
  --config scripts/hawkes/fim_benchmark.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Run a grid of FIM-Hawkes next-event evaluations and aggregate results")
    p.add_argument("--config", type=Path, required=True, help="YAML config file")
    return p.parse_args()


def run_single(cfg: Dict) -> Tuple[str, str, Path, int]:
    """Run one evaluation and return a tuple identifying the result.

    Returns (dataset, checkpoint_name, result_dir, return_code)
    """
    dataset = cfg["dataset"]
    checkpoint: Path = Path(cfg["checkpoint"]) if cfg.get("checkpoint") else Path("")
    checkpoint_name = checkpoint.name or str(cfg.get("checkpoint"))
    results_root: Path = cfg["results_root"]

    base_name = (
        f"{checkpoint_name}__{Path(dataset).name if Path(dataset).exists() else str(dataset).replace('/', '-')}__"
        f"ctx{cfg.get('context_size', 'all')}__inf{cfg.get('inference_size', 'all')}"
    )
    result_dir = results_root / base_name
    result_dir.mkdir(parents=True, exist_ok=True)
    run_log = result_dir / "run.log"

    print(
        f"[EVAL START] ckpt={checkpoint_name} dataset={dataset} ctx={cfg.get('context_size')} inf={cfg.get('inference_size')} → {result_dir}",
        flush=True,
    )

    cmd = [
        sys.executable,
        str(Path(__file__).with_name("fim_next_event_prediction.py")),
        "--checkpoint",
        str(cfg["checkpoint"]),
        "--dataset",
        str(dataset),
        "--run-dir",
        str(result_dir),
    ]
    if cfg.get("context_size") is not None:
        cmd.extend(["--context-size", str(cfg.get("context_size"))])
    if cfg.get("inference_size") is not None:
        cmd.extend(["--inference-size", str(cfg.get("inference_size"))])
    # max_num_events: None => no truncation; pass -1 as sentinel to the runner
    if cfg.get("max_num_events") is None:
        cmd.extend(["--max-num-events", "-1"])
    else:
        cmd.extend(["--max-num-events", str(cfg.get("max_num_events"))])
    if cfg.get("sample_idx") is not None:
        cmd.extend(["--sample-idx", str(cfg.get("sample_idx"))])
    if cfg.get("num_integration_points") is not None:
        cmd.extend(["--num-integration-points", str(cfg.get("num_integration_points"))])
    if cfg.get("num_bootstrap_samples") is not None:
        cmd.extend(["--num-bootstrap-samples", str(cfg.get("num_bootstrap_samples"))])

    with run_log.open("w") as log_f:
        proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT)

    rc = proc.returncode
    status = "OK" if rc == 0 else f"FAIL(rc={rc})"
    note = "metrics.json" if (result_dir / "metrics.json").exists() else "no metrics.json"
    print(f"[EVAL END]   ckpt={checkpoint_name} dataset={dataset} → {status} ({note}) log={run_log}", flush=True)

    if rc != 0:
        try:
            lines = run_log.read_text().rstrip().splitlines()
            tail = "\n".join(lines[-20:])
            print(f"[EVAL LOG TAIL] {checkpoint_name} on {dataset}:\n{tail}\n---", flush=True)
        except Exception:
            pass
    return str(dataset), checkpoint_name, result_dir, rc


def collect_rows(results_root: Path) -> List[Dict]:
    rows: List[Dict] = []
    for child in results_root.glob("*__*__*"):
        metrics_path = child / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            payload = json.loads(metrics_path.read_text())
        except Exception:
            continue

        dataset = payload.get("dataset")
        checkpoint = payload.get("model_checkpoint")
        status = payload.get("status", "unknown")
        duration_seconds = payload.get("duration_seconds")
        num_events = payload.get("num_events")

        metrics = payload.get("metrics", {})
        model_m = (metrics or {}).get("model") or {}
        baseline_m = (metrics or {}).get("baseline") or {}
        gt_m = (metrics or {}).get("ground_truth") or {}

        def add_row(src: str, m: Dict):
            if not m:
                return
            rows.append(
                {
                    "dataset": dataset,
                    "checkpoint": checkpoint,
                    "source": src,
                    "mae": m.get("mae"),
                    "mae_ci_lower": m.get("mae_ci_lower"),
                    "mae_ci_upper": m.get("mae_ci_upper"),
                    "mae_ci_error": m.get("mae_ci_error"),
                    "rmse": m.get("rmse"),
                    "rmse_ci_lower": m.get("rmse_ci_lower"),
                    "rmse_ci_upper": m.get("rmse_ci_upper"),
                    "rmse_ci_error": m.get("rmse_ci_error"),
                    "acc": m.get("acc"),
                    "acc_ci_lower": m.get("acc_ci_lower"),
                    "acc_ci_upper": m.get("acc_ci_upper"),
                    "acc_ci_error": m.get("acc_ci_error"),
                    "loglike": m.get("loglike"),
                    "loglike_ci_lower": m.get("loglike_ci_lower"),
                    "loglike_ci_upper": m.get("loglike_ci_upper"),
                    "loglike_ci_error": m.get("loglike_ci_error"),
                    "num_events": num_events,
                    "duration_seconds": duration_seconds,
                    "run_dir": str(child),
                    "status": status,
                }
            )

        add_row("model", model_m)
        add_row("baseline", baseline_m)
        if gt_m:
            add_row("ground_truth", gt_m)

    return rows


def write_summary(results_root: Path, rows: List[Dict]) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    csv_path = results_root / "summary.csv"
    fieldnames = [
        "dataset",
        "checkpoint",
        "source",
        "status",
        "mae",
        "mae_ci_error",
        "rmse",
        "rmse_ci_error",
        "acc",
        "acc_ci_error",
        "loglike",
        "loglike_ci_error",
        "num_events",
        "duration_seconds",
        "run_dir",
    ]

    def pretty_source(src: str) -> str:
        return {
            "baseline": "Mean-Baseline",
            "ground_truth": "Ground Truth",
            "model": "FIM",
        }.get(src, src)

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            row_out = dict(r)
            row_out["source"] = pretty_source(r.get("source", ""))
            writer.writerow(row_out)


def write_matrices(results_root: Path, rows: List[Dict]) -> None:
    # axes: dataset x (checkpoint,source) → metric
    datasets = sorted({r.get("dataset", "") for r in rows})
    models = sorted({(r.get("checkpoint", ""), r.get("source", "")) for r in rows})

    def pretty_source(src: str) -> str:
        return {
            "baseline": "Mean-Baseline",
            "ground_truth": "Ground Truth",
            "model": "FIM",
        }.get(src, src)

    def pretty_dataset(ds: str) -> str:
        ds_str = str(ds)
        return ds_str if ds_str.startswith("easytpp/") else Path(ds_str).name

    def build_matrix(metric: str) -> List[List[str]]:
        unique_ckpts = sorted({ckpt for ckpt, _ in models})
        multiple_ckpts = len(unique_ckpts) > 1
        if multiple_ckpts:
            header = ["dataset"] + [f"{Path(ckpt).name}:{pretty_source(src)}" for ckpt, src in models]
        else:
            header = ["dataset"] + [pretty_source(src) for _, src in models]
        matrix: List[List[str]] = [header]
        idx = {(r.get("dataset"), r.get("checkpoint"), r.get("source")): r for r in rows}
        for ds in datasets:
            row_vals: List[str] = [pretty_dataset(ds)]
            for ckpt, src in models:
                val = idx.get((ds, ckpt, src), {}).get(metric)
                row_vals.append("" if val is None else str(val))
            matrix.append(row_vals)
        return matrix

    def save_matrix(metric: str) -> None:
        matrix = build_matrix(metric)
        out_path = results_root / f"matrix_{metric}.csv"
        with out_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            for row in matrix:
                writer.writerow(row)

    for metric in [
        "mae",
        "mae_ci_error",
        "rmse",
        "rmse_ci_error",
        "acc",
        "acc_ci_error",
        "loglike",
        "loglike_ci_error",
        "num_events",
    ]:
        save_matrix(metric)


def main() -> None:
    args = parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    datasets: List[str] = cfg.get("datasets", [])
    checkpoints: List[str] = cfg.get("checkpoints", [])
    if not datasets or not checkpoints:
        print("[ERROR] Config must define 'datasets' and 'checkpoints'.")
        sys.exit(2)

    # Create a timestamped results_root so runs don't overwrite each other
    ts_root = datetime.now().strftime("%y%m%d-%H%M")

    base = {
        "context_size": cfg.get("context_size"),
        "inference_size": cfg.get("inference_size"),
        "max_num_events": cfg.get("max_num_events"),  # allow null for no truncation
        "sample_idx": int(cfg.get("sample_idx", 0)),
        "num_integration_points": int(cfg.get("num_integration_points", 5000)),
        "num_bootstrap_samples": int(cfg.get("num_bootstrap_samples", 1000)),
        "results_root": Path(cfg.get("results_root", "results/next_event_eval")) / ts_root,
    }

    parallel = int(cfg.get("parallel", 1))
    base["results_root"].mkdir(parents=True, exist_ok=True)

    # Build run configs
    run_confs: List[Dict] = []
    for d in datasets:
        for ck in checkpoints:
            rc = {**base, "dataset": d, "checkpoint": ck}
            run_confs.append(rc)

    total = len(run_confs)
    print(f"[GRID] Total evaluations: {total} | parallel={parallel}")

    # Execute
    if parallel <= 1:
        results = []
        for idx, rc in enumerate(run_confs, start=1):
            print(f"[GRID] ({idx}/{total}) launching …", flush=True)
            results.append(run_single(rc))
    else:
        results = []
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            fut_map = {ex.submit(run_single, rc): (rc["dataset"], rc["checkpoint"]) for rc in run_confs}
            done = 0
            for fut in as_completed(fut_map):
                results.append(fut.result())
                done += 1
                print(f"[GRID] progress: {done}/{total} finished", flush=True)

    # Aggregate
    rows = collect_rows(base["results_root"])
    write_summary(base["results_root"], rows)
    write_matrices(base["results_root"], rows)

    # Print a compact overview
    ok = sum(1 for _, _, _, rc in results if rc == 0)
    print(
        "Completed {}/{} evaluations. Summary: {}. Matrices: {}".format(
            ok,
            total,
            base["results_root"] / "summary.csv",
            ", ".join(str(base["results_root"] / f"matrix_{m}.csv") for m in ["mae", "rmse", "acc", "loglike", "num_events"]),
        )
    )


if __name__ == "__main__":
    main()
