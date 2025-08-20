"""
run via:

python3 scripts/hawkes/easytpp_benchmark.py
 --config scripts/hawkes/easytpp_benchmark.yaml
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
from typing import Dict, Iterable, List, Tuple

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run a grid of EasyTPP trainings and aggregate results")
    parser.add_argument("--config", type=Path, required=True, help="YAML config file")
    return parser.parse_args()


def run_single(cfg: Dict) -> Tuple[str, str, Path, int]:
    """Run one training and return a tuple identifying the result.

    Returns (dataset, model, result_dir, return_code)
    """
    dataset = cfg["dataset"]
    model = cfg["model"]
    results_root: Path = cfg["results_root"]
    result_dir = results_root / (
        f"{model}__{Path(dataset).name if Path(dataset).exists() else str(dataset).replace('/', '-')}__"
        f"s{cfg['sample_idx']}__e{cfg['epochs']}__bs{cfg['batch_size']}"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    run_log = result_dir / "run.log"

    # Announce run start
    print(
        f"[RUN START] model={model} dataset={dataset} epochs={cfg['epochs']} bs={cfg['batch_size']} → {result_dir}",
        flush=True,
    )

    cmd = [
        sys.executable,
        str(Path(__file__).with_name("easytpp_fit.py")),
        dataset,
        "--model",
        model,
        "--epochs",
        str(cfg["epochs"]),
        "--batch-size",
        str(cfg["batch_size"]),
        "--sample-idx",
        str(cfg["sample_idx"]),
        "--output-dir",
        str(cfg["output_dir"]),
        "--results-dir",
        str(cfg["results_root"]),
    ]
    # max_num_events: None => no truncation; pass -1 sentinel for fit.py to interpret as None
    if cfg.get("max_num_events") is None:
        cmd.extend(["--max-num-events", "-1"])
    else:
        cmd.extend(["--max-num-events", str(cfg["max_num_events"])])
    if cfg.get("num_train_paths") is not None:
        cmd.extend(["--num-train-paths", str(cfg["num_train_paths"])])
    if cfg.get("num_eval_paths") is not None:
        cmd.extend(["--num-eval-paths", str(cfg["num_eval_paths"])])
    if cfg.get("early_stop_patience") is not None:
        cmd.extend(["--early-stop-patience", str(cfg["early_stop_patience"])])
    if cfg.get("early_stop_metric"):
        cmd.extend(["--early-stop-metric", str(cfg["early_stop_metric"])])
    if cfg.get("early_stop_mode"):
        cmd.extend(["--early-stop-mode", str(cfg["early_stop_mode"])])

    with run_log.open("w") as log_f:
        proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT)

    rc = proc.returncode
    status = "OK" if rc == 0 else f"FAIL(rc={rc})"
    note = "metrics.json" if (result_dir / "metrics.json").exists() else "no metrics.json"
    print(f"[RUN END]   model={model} dataset={dataset} → {status} ({note}) log={run_log}", flush=True)

    # If failed, show a short tail of the log for quick triage
    if rc != 0:
        try:
            lines = run_log.read_text().rstrip().splitlines()
            tail = "\n".join(lines[-20:])
            print(f"[RUN LOG TAIL] {model} on {dataset}:\n{tail}\n---", flush=True)
        except Exception:
            pass
    return dataset, model, result_dir, rc


def collect_metrics(results_root: Path, grid: Iterable[Tuple[str, str]]) -> List[Dict]:
    rows: List[Dict] = []
    # Collect from all subdirectories under results_root
    for child in results_root.glob("*__*__*"):
        metrics_path = child / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            payload = json.loads(metrics_path.read_text())
        except Exception:
            continue
        row = {
            "dataset": payload.get("dataset"),
            "model": payload.get("model"),
            "status": payload.get("status", "unknown"),
            "rmse": payload.get("metrics", {}).get("rmse"),
            "acc": payload.get("metrics", {}).get("acc"),
            "loglike": payload.get("metrics", {}).get("loglike"),
            "num_events": payload.get("metrics", {}).get("num_events"),
            "duration_seconds": payload.get("duration_seconds"),
            "run_dir": str(child),
        }
        rows.append(row)
    return rows


def write_summary(results_root: Path, rows: List[Dict]) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    csv_path = results_root / "summary.csv"
    fieldnames = [
        "dataset",
        "model",
        "status",
        "rmse",
        "acc",
        "loglike",
        "num_events",
        "duration_seconds",
        "run_dir",
    ]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_matrices(results_root: Path, rows: List[Dict]) -> None:
    # Gather unique axes
    datasets = sorted({r.get("dataset", "") for r in rows})
    models = sorted({r.get("model", "") for r in rows})

    # Helper to build matrix for a metric
    def build_matrix(metric: str) -> List[List[str]]:
        header = ["dataset"] + models
        matrix: List[List[str]] = [header]
        # index rows for quick lookup
        index: Dict[Tuple[str, str], Dict] = {(r.get("dataset"), r.get("model")): r for r in rows}
        for ds in datasets:
            row_vals: List[str] = [str(ds)]
            for m in models:
                val = index.get((ds, m), {}).get(metric)
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

    for metric in ["rmse", "acc", "loglike", "num_events"]:
        save_matrix(metric)


def main() -> None:
    args = parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    datasets: List[str] = cfg.get("datasets", [])
    models: List[str] = cfg.get("models", [])
    if not datasets or not models:
        print("[ERROR] Config must define 'datasets' and 'models'.")
        sys.exit(2)

    # Create a timestamped results_root so runs don't overwrite each other
    ts_root = datetime.now().strftime("%y%m%d-%H%M")

    base = {
        "epochs": int(cfg.get("epochs", 50)),
        "batch_size": int(cfg.get("batch_size", 256)),
        # allow null to indicate no truncation; keep None and translate to sentinel later
        "max_num_events": cfg.get("max_num_events", 100),
        "num_train_paths": cfg.get("num_train_paths"),
        "num_eval_paths": cfg.get("num_eval_paths"),
        "sample_idx": int(cfg.get("sample_idx", 0)),
        "output_dir": Path(cfg.get("output_dir", "checkpoints")),
        "results_root": Path(cfg.get("results_root", "results/easytpp_grid")) / ts_root,
    }

    parallel = int(cfg.get("parallel", 1))

    base["results_root"].mkdir(parents=True, exist_ok=True)

    # Build run configs
    run_confs: List[Dict] = []
    for d in datasets:
        for m in models:
            rc = {**base, "dataset": d, "model": m}
            run_confs.append(rc)

    total = len(run_confs)
    print(f"[GRID] Total runs: {total} | parallel={parallel}")

    # Execute
    if parallel <= 1:
        results = []
        for idx, rc in enumerate(run_confs, start=1):
            print(f"[GRID] ({idx}/{total}) launching …", flush=True)
            results.append(run_single(rc))
    else:
        results = []
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            fut_map = {ex.submit(run_single, rc): (rc["dataset"], rc["model"]) for rc in run_confs}
            done = 0
            for fut in as_completed(fut_map):
                results.append(fut.result())
                done += 1
                print(f"[GRID] progress: {done}/{total} finished", flush=True)

    # Aggregate
    rows = collect_metrics(base["results_root"], [(rc["dataset"], rc["model"]) for rc in run_confs])
    write_summary(base["results_root"], rows)
    write_matrices(base["results_root"], rows)

    # Print a compact overview
    ok = sum(1 for _, _, _, rc in results if rc == 0)
    print(
        "Completed {}/{} runs. Summary: {}. Matrices: {}".format(
            ok,
            total,
            base["results_root"] / "summary.csv",
            ", ".join(str(base["results_root"] / f"matrix_{m}.csv") for m in ["rmse", "acc", "loglike", "num_events"]),
        )
    )


if __name__ == "__main__":
    main()
