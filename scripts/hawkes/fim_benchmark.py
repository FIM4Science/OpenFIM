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
    p.add_argument(
        "--task",
        type=str,
        choices=["next_event", "long_horizon", "both"],
        default=None,
        help="Evaluation task: next_event, long_horizon, or both (sequential)",
    )
    return p.parse_args()


def run_single(cfg: Dict) -> Tuple[str, str, Path, int]:
    """Run one evaluation and return a tuple identifying the result.

    Returns (dataset, checkpoint_name, result_dir, return_code)
    """
    dataset = cfg["dataset"]
    checkpoint: Path = Path(cfg["checkpoint"]) if cfg.get("checkpoint") else Path("")
    checkpoint_name = checkpoint.name or str(cfg.get("checkpoint"))
    results_root: Path = cfg["results_root"]
    task = str(cfg.get("task", "next_event"))

    def base_name_for(task_label: str) -> str:
        hz = ""
        if task_label == "long_horizon":
            hz = f"__N{cfg.get('forecast_horizon_size', 'NA')}__S{cfg.get('num_ensemble_trajectories', '5')}"
        return (
            f"{checkpoint_name}__{Path(dataset).name if Path(dataset).exists() else str(dataset).replace('/', '-')}__"
            f"{task_label}{hz}__ctx{cfg.get('context_size', 'all')}__inf{cfg.get('inference_size', 'all')}"
        )

    if task == "both":
        parent_dir = results_root / base_name_for("both")
        parent_dir.mkdir(parents=True, exist_ok=True)
        run_log = parent_dir / "run.log"
        next_dir = parent_dir / "next_event"
        long_dir = parent_dir / "long_horizon"
        next_dir.mkdir(parents=True, exist_ok=True)
        long_dir.mkdir(parents=True, exist_ok=True)
        start_dir, return_dir = parent_dir, parent_dir
    else:
        result_dir = results_root / base_name_for(task)
        result_dir.mkdir(parents=True, exist_ok=True)
        run_log = result_dir / "run.log"
        start_dir, return_dir = result_dir, result_dir

    print(
        f"[EVAL START] task={task} ckpt={checkpoint_name} dataset={dataset} → {start_dir}",
        flush=True,
    )

    commands: List[List[str]] = []

    def build_common_args(run_dir: Path) -> List[str]:
        args = ["--checkpoint", str(cfg["checkpoint"]), "--dataset", str(dataset), "--run-dir", str(run_dir)]
        if cfg.get("context_size") is not None:
            args.extend(["--context-size", str(cfg.get("context_size"))])
        if cfg.get("inference_size") is not None:
            args.extend(["--inference-size", str(cfg.get("inference_size"))])
        if cfg.get("max_num_events") is None:
            args.extend(["--max-num-events", "-1"])
        else:
            args.extend(["--max-num-events", str(cfg.get("max_num_events"))])
        if cfg.get("sample_idx") is not None:
            args.extend(["--sample-idx", str(cfg.get("sample_idx"))])
        if cfg.get("num_integration_points") is not None:
            args.extend(["--num-integration-points", str(cfg.get("num_integration_points"))])
        return args

    if task in ("next_event", "both"):
        sub_run_dir = next_dir if task == "both" else result_dir
        cmd = [sys.executable, str(Path(__file__).with_name("fim_next_event_prediction.py"))] + build_common_args(sub_run_dir)
        if cfg.get("num_bootstrap_samples") is not None:
            cmd.extend(["--num-bootstrap-samples", str(cfg.get("num_bootstrap_samples"))])
        commands.append(cmd)

    if task in ("long_horizon", "both"):
        sub_run_dir = long_dir if task == "both" else result_dir
        cmd = [sys.executable, str(Path(__file__).with_name("fim_long_horizon_prediction.py"))] + build_common_args(sub_run_dir)
        if cfg.get("forecast_horizon_size") is None:
            raise ValueError("long_horizon task requires 'forecast_horizon_size'")
        cmd.extend(["--forecast-horizon-size", str(cfg.get("forecast_horizon_size"))])
        if cfg.get("num_ensemble_trajectories") is not None:
            cmd.extend(["--num-ensemble-trajectories", str(cfg.get("num_ensemble_trajectories"))])
        commands.append(cmd)

    rc = 0
    with run_log.open("w") as log_f:
        for idx, cmd in enumerate(commands, start=1):
            log_f.write(f"\n===== [RUNNING COMMAND {idx}/{len(commands)}] =====\n{' '.join(cmd)}\n\n")
            log_f.flush()
            proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT)
            rc = proc.returncode
            if rc != 0:
                break

    status = "OK" if rc == 0 else f"FAIL(rc={rc})"
    print(f"[EVAL END]   ckpt={checkpoint_name} dataset={dataset} → {status} log={run_log}", flush=True)

    if rc != 0:
        try:
            lines = run_log.read_text().rstrip().splitlines()
            print(f"[EVAL LOG TAIL] {checkpoint_name} on {dataset}:\n{chr(10).join(lines[-20:])}\n---", flush=True)
        except Exception:
            pass
    return str(dataset), checkpoint_name, return_dir, rc


def collect_rows(results_root: Path) -> List[Dict]:
    rows: List[Dict] = []
    for metrics_path in results_root.rglob("metrics.json"):
        try:
            payload = json.loads(metrics_path.read_text())
            if payload.get("status") != "OK":
                continue

            run_dir_str = str(metrics_path.parent)
            if "next_event" in run_dir_str:
                task = "next_event"
            elif "long_horizon" in run_dir_str:
                task = "long_horizon"
            # Heuristic fallback for combined 'both' directories
            elif (payload.get("metrics", {}).get("model") or {}).get("rmsex_plus") is not None:
                task = "long_horizon"
            else:
                task = "next_event"

            def add_row(src: str, m: Dict):
                if not m:
                    return
                rows.append(
                    {
                        "dataset": payload.get("dataset"),
                        "checkpoint": payload.get("model_checkpoint"),
                        "source": src,
                        "task": task,
                        "mae": m.get("mae"),
                        "mae_ci_error": m.get("mae_ci_error"),
                        "rmse": m.get("rmse"),
                        "rmse_ci_error": m.get("rmse_ci_error"),
                        "type_error": m.get("type_error"),
                        "type_error_ci_error": m.get("type_error_ci_error"),
                        "loglike": m.get("loglike"),
                        "loglike_ci_error": m.get("loglike_ci_error"),
                        "rmsex_plus": m.get("rmsex_plus"),
                        "smape": m.get("smape"),
                        "rmse_e": m.get("rmse_e"),
                        "otd": m.get("otd"),
                        "num_events": payload.get("num_events", payload.get("num_eval_sequences")),
                        "duration_seconds": payload.get("duration_seconds"),
                        "run_dir": run_dir_str,
                        "status": payload.get("status"),
                    }
                )

            metrics = payload.get("metrics", {})
            for src in ["model", "baseline", "ground_truth"]:
                if (metrics or {}).get(src):
                    add_row(src, metrics[src])
        except Exception:
            continue
    return rows


def write_summary(results_root: Path, rows: List[Dict]) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    csv_path = results_root / "summary.csv"
    fieldnames = [
        "dataset",
        "checkpoint",
        "source",
        "status",
        "task",
        "mae",
        "mae_ci_error",
        "rmse",
        "rmse_ci_error",
        "type_error",
        "type_error_ci_error",
        "loglike",
        "loglike_ci_error",
        "rmsex_plus",
        "smape",
        "rmse_e",
        "otd",
        "num_events",
        "duration_seconds",
        "run_dir",
    ]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_matrices(results_root: Path, rows: List[Dict]) -> None:
    def save_matrix(metric: str, task: str) -> None:
        datasets = sorted({r["dataset"] for r in rows if r.get("task") == task})
        if not datasets:
            return

        models = sorted({r["checkpoint"] for r in rows if r.get("task") == task})
        sources = sorted({r["source"] for r in rows if r.get("task") == task})
        header = ["dataset"] + [f"{Path(m).name}:{s}" for m in models for s in sources]
        matrix = [header]
        idx = {(r["dataset"], r["checkpoint"], r["source"]): r for r in rows if r.get("task") == task}

        for ds in datasets:
            row_vals = [Path(ds).name if not str(ds).startswith("easytpp/") else str(ds)]
            for m in models:
                for s in sources:
                    val = idx.get((ds, m, s), {}).get(metric)
                    row_vals.append(f"{val:.4f}" if isinstance(val, float) else str(val) if val is not None else "")
            matrix.append(row_vals)

        prefix = "long_horizon" if task == "long_horizon" else "next_event"
        with (results_root / f"{prefix}_{metric}.csv").open("w", newline="") as fh:
            csv.writer(fh).writerows(matrix)

    metrics_to_write = {"next_event": ["mae", "rmse", "type_error", "loglike"], "long_horizon": ["rmsex_plus", "smape", "rmse_e", "otd"]}
    for task, metrics in metrics_to_write.items():
        for metric in metrics:
            save_matrix(metric, task)


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if not cfg.get("datasets") or not cfg.get("checkpoints"):
        sys.exit("[ERROR] Config must define 'datasets' and 'checkpoints'.")

    ts_root = datetime.now().strftime("%y%m%d-%H%M")
    task = args.task or str(cfg.get("task", "next_event"))
    base = {
        **{k: v for k, v in cfg.items() if k not in ["datasets", "checkpoints"]},
        "task": task,
        "results_root": Path(cfg.get("results_root", "results/fim_benchmark")) / ts_root,
    }

    run_confs = [{**base, "dataset": d, "checkpoint": ck} for d in cfg.get("datasets", []) for ck in cfg.get("checkpoints", [])]
    total = len(run_confs)
    parallel = int(cfg.get("parallel", 1))
    print(f"[GRID] Total evaluations: {total} | parallel={parallel}")

    base["results_root"].mkdir(parents=True, exist_ok=True)
    if parallel <= 1:
        results = [run_single(rc) for rc in run_confs]
    else:
        with ThreadPoolExecutor(max_workers=parallel) as ex:
            futures = {ex.submit(run_single, rc): rc for rc in run_confs}
            results = [fut.result() for fut in as_completed(futures)]

    rows = collect_rows(base["results_root"])
    if rows:
        write_summary(base["results_root"], rows)
        write_matrices(base["results_root"], rows)

    ok = sum(1 for _, _, _, rc in results if rc == 0)
    print(f"\nCompleted {ok}/{total} evaluations. Results are in {base['results_root']}")


if __name__ == "__main__":
    main()
