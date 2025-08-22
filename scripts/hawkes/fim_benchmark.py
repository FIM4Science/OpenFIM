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
        # Parent directory for the combined run
        parent_dir = results_root / base_name_for("both")
        parent_dir.mkdir(parents=True, exist_ok=True)
        run_log = parent_dir / "run.log"
        # Child dirs per task
        next_dir = results_root / base_name_for("next_event")
        long_dir = results_root / base_name_for("long_horizon")
        next_dir.mkdir(parents=True, exist_ok=True)
        long_dir.mkdir(parents=True, exist_ok=True)
        start_dir = parent_dir
        return_dir = parent_dir
    else:
        result_dir = results_root / base_name_for(task)
        result_dir.mkdir(parents=True, exist_ok=True)
        run_log = result_dir / "run.log"
        start_dir = result_dir
        return_dir = result_dir

    print(
        f"[EVAL START] ckpt={checkpoint_name} dataset={dataset} ctx={cfg.get('context_size')} inf={cfg.get('inference_size')} → {start_dir}",
        flush=True,
    )

    # Build command(s) based on task
    commands: List[List[str]] = []

    def build_common_args(run_dir: Path) -> List[str]:
        args = [
            "--checkpoint",
            str(cfg["checkpoint"]),
            "--dataset",
            str(dataset),
            "--run-dir",
            str(run_dir),
        ]
        if cfg.get("context_size") is not None:
            args.extend(["--context-size", str(cfg.get("context_size"))])
        if cfg.get("inference_size") is not None:
            args.extend(["--inference-size", str(cfg.get("inference_size"))])
        # max_num_events: None => no truncation; pass -1 as sentinel
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
        script_path = Path(__file__).with_name("fim_next_event_prediction.py")
        sub_run_dir = next_dir if task == "both" else result_dir
        cmd = [sys.executable, str(script_path)] + build_common_args(sub_run_dir)
        if cfg.get("num_bootstrap_samples") is not None:
            cmd.extend(["--num-bootstrap-samples", str(cfg.get("num_bootstrap_samples"))])
        commands.append(cmd)

    if task in ("long_horizon", "both"):
        script_path = Path(__file__).with_name("fim_long_horizon_prediction.py")
        sub_run_dir = long_dir if task == "both" else result_dir
        cmd = [sys.executable, str(script_path)] + build_common_args(sub_run_dir)
        if cfg.get("forecast_horizon_size") is None:
            raise ValueError("long_horizon task requires 'forecast_horizon_size' in config")
        cmd.extend(["--forecast-horizon-size", str(cfg.get("forecast_horizon_size"))])
        if cfg.get("num_ensemble_trajectories") is not None:
            cmd.extend(["--num-ensemble-trajectories", str(cfg.get("num_ensemble_trajectories"))])
        commands.append(cmd)
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
    if cfg.get("num_bootstrap_samples") is not None and task == "next_event":
        # Only used by next-event evaluation
        cmd.extend(["--num-bootstrap-samples", str(cfg.get("num_bootstrap_samples"))])

    # Long-horizon specific args
    if task == "long_horizon":
        if cfg.get("forecast_horizon_size") is None:
            raise ValueError("long_horizon task requires 'forecast_horizon_size' in config")
        cmd.extend(["--forecast-horizon-size", str(cfg.get("forecast_horizon_size"))])
        if cfg.get("num_ensemble_trajectories") is not None:
            cmd.extend(["--num-ensemble-trajectories", str(cfg.get("num_ensemble_trajectories"))])

    # Execute sequentially if multiple commands
    rc = 0
    with run_log.open("w") as log_f:
        for idx, cmd in enumerate(commands, start=1):
            seg = f"[{idx}/{len(commands)}]"
            log_f.write(f"[RUN] {seg} {' '.join(cmd)}\n")
            log_f.flush()
            proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT)
            rc = proc.returncode
            if rc != 0:
                break

    status = "OK" if rc == 0 else f"FAIL(rc={rc})"
    # Determine a note summarizing presence of metrics
    note = []
    if task == "both":
        if (next_dir / "metrics.json").exists():
            note.append("next_event:metrics.json")
        else:
            note.append("next_event:no metrics.json")
        if (long_dir / "metrics.json").exists():
            note.append("long_horizon:metrics.json")
        else:
            note.append("long_horizon:no metrics.json")
        note_str = ", ".join(note)
        print(f"[EVAL END]   ckpt={checkpoint_name} dataset={dataset} → {status} ({note_str}) log={run_log}", flush=True)
    else:
        note_str = "metrics.json" if (result_dir / "metrics.json").exists() else "no metrics.json"
        print(f"[EVAL END]   ckpt={checkpoint_name} dataset={dataset} → {status} ({note_str}) log={run_log}", flush=True)

    if rc != 0:
        try:
            lines = run_log.read_text().rstrip().splitlines()
            tail = "\n".join(lines[-20:])
            print(f"[EVAL LOG TAIL] {checkpoint_name} on {dataset}:\n{tail}\n---", flush=True)
        except Exception:
            pass
    return str(dataset), checkpoint_name, return_dir, rc


def collect_rows(results_root: Path) -> List[Dict]:
    rows: List[Dict] = []
    # Search recursively for metrics.json in any sub-run directory
    for metrics_path in results_root.rglob("metrics.json"):
        child = metrics_path.parent
        try:
            payload = json.loads(metrics_path.read_text())
        except Exception:
            continue

        dataset = payload.get("dataset")
        checkpoint = payload.get("model_checkpoint")
        status = payload.get("status", "unknown")
        duration_seconds = payload.get("duration_seconds")
        # next_event writes num_events; long_horizon writes num_eval_sequences
        num_events = payload.get("num_events", payload.get("num_eval_sequences"))

        metrics = payload.get("metrics", {})
        model_m = (metrics or {}).get("model") or {}
        baseline_m = (metrics or {}).get("baseline") or {}
        gt_m = (metrics or {}).get("ground_truth") or {}

        # Detect task from run_dir name or metrics content
        run_dir_str = str(child)
        if "__long_horizon__" in run_dir_str or "/long_horizon" in run_dir_str:
            task = "long_horizon"
        elif "__next_event__" in run_dir_str or "/next_event" in run_dir_str:
            task = "next_event"
        else:
            # Heuristic based on available metrics
            if ((metrics or {}).get("model") or {}).get("rmsex_plus") is not None:
                task = "long_horizon"
            else:
                task = "next_event"

        def add_row(src: str, m: Dict):
            if not m:
                return
            rows.append(
                {
                    "dataset": dataset,
                    "checkpoint": checkpoint,
                    "source": src,
                    "task": task,
                    # next-event metrics
                    "mae": m.get("mae"),
                    "mae_ci_lower": m.get("mae_ci_lower"),
                    "mae_ci_upper": m.get("mae_ci_upper"),
                    "mae_ci_error": m.get("mae_ci_error"),
                    "rmse": m.get("rmse"),
                    "rmse_ci_lower": m.get("rmse_ci_lower"),
                    "rmse_ci_upper": m.get("rmse_ci_upper"),
                    "rmse_ci_error": m.get("rmse_ci_error"),
                    "type_error": m.get("type_error"),
                    "type_error_ci_error": m.get("type_error_ci_error"),
                    "loglike": m.get("loglike"),
                    "loglike_ci_lower": m.get("loglike_ci_lower"),
                    "loglike_ci_upper": m.get("loglike_ci_upper"),
                    "loglike_ci_error": m.get("loglike_ci_error"),
                    # long-horizon metrics
                    "rmsex_plus": m.get("rmsex_plus"),
                    "smape": m.get("smape"),
                    "rmse_e": m.get("rmse_e"),
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
        # next-event
        "mae",
        "mae_ci_error",
        "rmse",
        "rmse_ci_error",
        "type_error",
        "type_error_ci_error",
        "loglike",
        "loglike_ci_error",
        # long-horizon
        "rmsex_plus",
        "smape",
        "rmse_e",
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
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            row_out = dict(r)
            row_out["source"] = pretty_source(r.get("source", ""))
            writer.writerow(row_out)


def write_matrices(results_root: Path, rows: List[Dict]) -> None:
    # axes: dataset x (checkpoint,task,source) → metric; we will write separate matrices per task
    datasets = sorted({r.get("dataset", "") for r in rows})
    models_by_task = {
        task: sorted({(r.get("checkpoint", ""), r.get("source", "")) for r in rows if r.get("task") == task})
        for task in ("next_event", "long_horizon")
    }

    def pretty_source(src: str) -> str:
        return {
            "baseline": "Mean-Baseline",
            "ground_truth": "Ground Truth",
            "model": "FIM",
        }.get(src, src)

    def pretty_dataset(ds: str) -> str:
        ds_str = str(ds)
        return ds_str if ds_str.startswith("easytpp/") else Path(ds_str).name

    def build_matrix(metric: str, task: str) -> List[List[str]]:
        models = models_by_task.get(task, [])
        unique_ckpts = sorted({ckpt for ckpt, _ in models})
        multiple_ckpts = len(unique_ckpts) > 1
        if multiple_ckpts:
            header = ["dataset"] + [f"{Path(ckpt).name}:{pretty_source(src)}" for ckpt, src in models]
        else:
            header = ["dataset"] + [pretty_source(src) for _, src in models]
        matrix: List[List[str]] = [header]
        idx = {(r.get("dataset"), r.get("checkpoint"), r.get("source")): r for r in rows if r.get("task") == task}
        for ds in datasets:
            row_vals: List[str] = [pretty_dataset(ds)]
            for ckpt, src in models:
                val = idx.get((ds, ckpt, src), {}).get(metric)
                row_vals.append("" if val is None else str(val))
            matrix.append(row_vals)
        return matrix

    def save_matrix(metric: str, task: str) -> None:
        matrix = build_matrix(metric, task)
        prefix = "next_event" if task == "next_event" else "long_horizon"
        out_path = results_root / f"{prefix}_{metric}.csv"
        with out_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            for row in matrix:
                writer.writerow(row)

    # Write next-event matrices
    for metric in [
        "mae",
        "mae_ci_error",
        "rmse",
        "rmse_ci_error",
        "type_error",
        "type_error_ci_error",
        "loglike",
        "loglike_ci_error",
        "num_events",
    ]:
        save_matrix(metric, task="next_event")
    # Write long-horizon matrices
    for metric in ["rmsex_plus", "smape", "rmse_e", "num_events"]:
        save_matrix(metric, task="long_horizon")


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

    # Task selection: CLI overrides YAML; default to next_event
    task = args.task or str(cfg.get("task", "next_event"))

    base = {
        "context_size": cfg.get("context_size"),
        "inference_size": cfg.get("inference_size"),
        "max_num_events": cfg.get("max_num_events"),  # allow null for no truncation
        "sample_idx": int(cfg.get("sample_idx", 0)),
        "num_integration_points": int(cfg.get("num_integration_points", 5000)),
        "num_bootstrap_samples": int(cfg.get("num_bootstrap_samples", 1000)),
        "task": task,
        # long-horizon specific
        "forecast_horizon_size": cfg.get("forecast_horizon_size"),
        "num_ensemble_trajectories": int(cfg.get("num_ensemble_trajectories", 5))
        if cfg.get("num_ensemble_trajectories") is not None
        else None,
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
            ", ".join(
                [str(base["results_root"] / f"next_event_{m}.csv") for m in ["mae", "rmse", "type_error", "loglike", "num_events"]]
                + [str(base["results_root"] / f"long_horizon_{m}.csv") for m in ["rmsex_plus", "smape", "rmse_e", "num_events"]]
            ),
        )
    )


if __name__ == "__main__":
    main()
