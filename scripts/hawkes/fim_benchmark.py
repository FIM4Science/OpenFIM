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
import os
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
    # Human-friendly label when no checkpoint is provided
    checkpoint_name = checkpoint.name or (str(cfg.get("checkpoint")) if cfg.get("checkpoint") else "scratch")
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

    # Persist a snapshot of the effective configuration for this run
    try:

        def _stringify(o):
            from pathlib import Path as _P

            if isinstance(o, _P):
                return str(o)
            return o

        cfg_snapshot = {k: _stringify(v) for k, v in cfg.items()}
        (start_dir / "config_used.json").write_text(json.dumps(cfg_snapshot, indent=2))
    except Exception:
        pass

    commands: List[List[str]] = []
    command_labels: List[str] = []  # "next_event" or "long_horizon"
    command_run_dirs: List[Path] = []

    # ============================
    # Optional: Fine-tuning support
    # ============================
    def _resolve_cdiff_dataset_path(ds: str) -> Path:
        base = Path(ds)
        if base.exists():
            return base
        candidate = Path(__file__).resolve().parents[2] / "data" / "external" / "CDiff_dataset" / ds
        return candidate

    def _latest_best_model_dir(save_root: Path, dataset_name: str) -> Path:
        base_dir = save_root / dataset_name
        if not base_dir.exists():
            return Path("")
        candidates = []
        try:
            for ts_dir in base_dir.iterdir():
                if ts_dir.is_dir():
                    bm = ts_dir / "best-model"
                    if bm.exists():
                        candidates.append((bm.stat().st_mtime, bm))
        except Exception:
            candidates = []
        if not candidates:
            return Path("")
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def _is_valid_checkpoint_dir(p: Path) -> bool:
        try:
            return p.is_dir() and (
                (p / "config.json").exists()
                and (
                    ((p / "model-checkpoint.pth").exists()) or ((p / "model.safetensors").exists()) or ((p / "pytorch_model.bin").exists())
                )
            )
        except Exception:
            return False

    def _extract_num_trials_from_ft_yaml(ft_yaml_path: Path) -> int | None:
        try:
            if not ft_yaml_path or not ft_yaml_path.exists():
                return None
            raw = yaml.safe_load(ft_yaml_path.read_text())
            if not isinstance(raw, dict):
                return None
            # Prefer top-level 'num_trials'
            if isinstance(raw.get("num_trials"), int):
                return int(raw["num_trials"])
            # Or nested under 'evaluation' key
            eval_blk = raw.get("evaluation")
            if isinstance(eval_blk, dict) and isinstance(eval_blk.get("num_trials"), int):
                return int(eval_blk["num_trials"])
        except Exception:
            return None
        return None

    # Shared cache to reuse the same fine-tuned checkpoint across tasks in a single run
    shared_finetuned_ckpt: Path | None = None

    def _maybe_finetune(for_task: str) -> Path:
        """Run finetuning for the specific sub-task and return best checkpoint dir.

        For next_event: use HF EasyTPP dataset id (e.g., easytpp/amazon).
        For long_horizon: use local CDiff dataset folder path.
        """
        nonlocal shared_finetuned_ckpt

        # If we've already fine-tuned in this run, reuse the same checkpoint for both tasks
        if shared_finetuned_ckpt is not None and _is_valid_checkpoint_dir(shared_finetuned_ckpt):
            return shared_finetuned_ckpt
        if not bool(cfg.get("fine_tune", False)):
            # If no checkpoint was provided and fine-tuning is disabled, we cannot proceed.
            # Downstream loaders require a valid checkpoint directory.
            if not (str(checkpoint) and Path(str(checkpoint)).exists()):
                raise ValueError("No checkpoint provided and fine_tune is False; cannot initialize model from scratch for evaluation.")
            return checkpoint

        finetune_config = cfg.get("finetune_config")
        finetune_epochs = int(cfg.get("fine_tune_epochs", 100))
        finetune_lr = cfg.get("fine_tune_lr", 5e-5)
        # Default finetune save root co-located under current results root for easy discovery
        ft_save_root = Path(cfg.get("finetune_save_root", results_root / "_finetune"))
        ft_save_root.mkdir(parents=True, exist_ok=True)

        # Build dataset argument for finetune
        if for_task == "next_event":
            ds_arg = dataset if str(dataset).startswith("easytpp/") else f"easytpp/{dataset}"
            dataset_name_for_ft = ds_arg.split("/")[-1]
        else:
            cdiff_path = _resolve_cdiff_dataset_path(str(dataset))
            if not cdiff_path.exists():
                raise FileNotFoundError(f"CDiff dataset path not found for long_horizon finetune: {cdiff_path}")
            ds_arg = str(cdiff_path)
            dataset_name_for_ft = Path(ds_arg).name

        # Always resume from the original base checkpoint when provided explicitly in config
        base_ckpt_str = cfg.get("checkpoint") if cfg.get("checkpoint") else None
        base_ckpt_path = Path(base_ckpt_str) if base_ckpt_str else None

        ft_cmd = [
            sys.executable,
            str(Path(__file__).with_name("fim_finetune.py")),
        ]
        if finetune_config:
            ft_cmd.extend(["--config", str(finetune_config)])
        ft_cmd.extend(
            [
                "--dataset",
                str(ds_arg),
                "--epochs",
                str(finetune_epochs),
                "--lr",
                str(finetune_lr),
                "--save_dir",
                str(ft_save_root),
                "--grad-accum-steps",
                str(cfg.get("finetune_grad_accum_steps", 1)),
                "--val-every",
                str(cfg.get("finetune_val_every", 100)),
            ]
        )
        # Only append resume_model if a valid checkpoint directory exists (not current '.')
        if base_ckpt_path is not None and _is_valid_checkpoint_dir(base_ckpt_path):
            ft_cmd.extend(["--resume_model", str(base_ckpt_path)])
        else:
            # Starting from scratch: require a finetune config to construct the model
            if not finetune_config:
                raise ValueError("Starting from scratch requires 'finetune_config' in the benchmark YAML so the model can be constructed.")
        # Constrain memory during fine-tune regardless of eval settings
        ft_max_paths = int(cfg.get("finetune_max_paths", 2000))
        ft_max_events = int(cfg.get("finetune_max_events", cfg.get("max_num_events") or 100))
        ft_cmd.extend(["--max_paths", str(ft_max_paths), "--max_events", str(ft_max_events)])

        # Ensure child processes can import 'fim' by adding repo src to PYTHONPATH
        env = os.environ.copy()
        repo_src = str(Path(__file__).resolve().parents[2] / "src")
        if env.get("PYTHONPATH"):
            if repo_src not in env["PYTHONPATH"].split(":"):
                env["PYTHONPATH"] = repo_src + ":" + env["PYTHONPATH"]
        else:
            env["PYTHONPATH"] = repo_src

        with run_log.open("a") as log_f:
            log_f.write(f"\n===== [FINETUNE {for_task}] =====\n{' '.join(ft_cmd)}\n\n")
            log_f.flush()
            proc = subprocess.run(ft_cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
            if proc.returncode != 0:
                print(
                    f"[FINETUNE FAIL] task={for_task} dataset={dataset} rc={proc.returncode}",
                    flush=True,
                )
                raise RuntimeError(f"Fine-tune failed for task={for_task} dataset={dataset} (rc={proc.returncode}). See log: {run_log}")

        # Locate newest best-model for this dataset under save root
        best_dir = _latest_best_model_dir(Path(ft_save_root), dataset_name_for_ft)
        if not best_dir or not best_dir.exists():
            print(
                f"[FINETUNE WARN] Could not locate best-model under {ft_save_root}/{dataset_name_for_ft}; using base checkpoint",
                flush=True,
            )
            return checkpoint

        print(f"[FINETUNE OK] task={for_task} dataset={dataset} → {best_dir}", flush=True)
        shared_finetuned_ckpt = best_dir
        return shared_finetuned_ckpt

    def build_common_args_for_dataset(run_dir: Path, ds: str, ckpt_dir: Path) -> List[str]:
        args = ["--checkpoint", str(ckpt_dir), "--dataset", str(ds), "--run-dir", str(run_dir)]
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
        dataset_ne = dataset if str(dataset).startswith("easytpp/") else f"easytpp/{dataset}"
        eff_ckpt_ne = _maybe_finetune("next_event")
        cmd = [
            sys.executable,
            str(Path(__file__).with_name("fim_next_event_prediction.py")),
            *build_common_args_for_dataset(sub_run_dir, dataset_ne, eff_ckpt_ne),
        ]
        if cfg.get("sampling_method") is not None:
            cmd.extend(["--sampling-method", str(cfg.get("sampling_method"))])
        if cfg.get("nll_method") is not None:
            cmd.extend(["--nll-method", str(cfg.get("nll_method"))])
        # Validate checkpoint directory
        if not _is_valid_checkpoint_dir(Path(str(eff_ckpt_ne))):
            raise FileNotFoundError(f"Next-event checkpoint directory invalid: {eff_ckpt_ne}")
        if cfg.get("num_bootstrap_samples") is not None:
            cmd.extend(["--num-bootstrap-samples", str(cfg.get("num_bootstrap_samples"))])
        commands.append(cmd)
        command_labels.append("next_event")
        command_run_dirs.append(sub_run_dir)

    if task in ("long_horizon", "both"):
        sub_run_dir = long_dir if task == "both" else result_dir
        # For long-horizon, load CDiff local data (script resolves short names to repo path)
        eff_ckpt_lh = _maybe_finetune("long_horizon")
        cmd = [
            sys.executable,
            str(Path(__file__).with_name("fim_long_horizon_prediction.py")),
            *build_common_args_for_dataset(sub_run_dir, str(dataset), eff_ckpt_lh),
        ]
        if cfg.get("sampling_method") is not None:
            cmd.extend(["--sampling-method", str(cfg.get("sampling_method"))])
        if cfg.get("nll_method") is not None:
            cmd.extend(["--nll-method", str(cfg.get("nll_method"))])
        if not _is_valid_checkpoint_dir(Path(str(eff_ckpt_lh))):
            raise FileNotFoundError(f"Long-horizon checkpoint directory invalid: {eff_ckpt_lh}")
        if cfg.get("forecast_horizon_size") is None:
            raise ValueError("long_horizon task requires 'forecast_horizon_size'")
        cmd.extend(["--forecast-horizon-size", str(cfg.get("forecast_horizon_size"))])
        if cfg.get("num_ensemble_trajectories") is not None:
            cmd.extend(["--num-ensemble-trajectories", str(cfg.get("num_ensemble_trajectories"))])
        # Trials configuration for long-horizon (prefer benchmark YAML, else finetune YAML, else default 10)
        num_trials_val = cfg.get("num_trials")
        if num_trials_val is None and cfg.get("finetune_config") is not None:
            ft_yaml_path = Path(str(cfg.get("finetune_config")))
            num_trials_val = _extract_num_trials_from_ft_yaml(ft_yaml_path)
        if num_trials_val is None:
            num_trials_val = 10
        cmd.extend(["--num-trials", str(int(num_trials_val))])
        if cfg.get("base_seed") is not None:
            cmd.extend(["--base-seed", str(cfg.get("base_seed"))])
        commands.append(cmd)
        command_labels.append("long_horizon")
        command_run_dirs.append(sub_run_dir)

    rc = 0
    # Ensure child processes can import 'fim' in eval as well
    env = os.environ.copy()
    repo_src = str(Path(__file__).resolve().parents[2] / "src")
    if env.get("PYTHONPATH"):
        if repo_src not in env["PYTHONPATH"].split(":"):
            env["PYTHONPATH"] = repo_src + ":" + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = repo_src

    with run_log.open("w") as log_f:
        for idx, cmd in enumerate(commands, start=1):
            log_f.write(f"\n===== [RUNNING COMMAND {idx}/{len(commands)}] =====\n{' '.join(cmd)}\n\n")
            log_f.flush()
            proc = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)
            rc = proc.returncode
            # If the just-finished command is the next-event eval and succeeded, persist its loglike as a simple file
            try:
                if rc == 0 and command_labels[idx - 1] == "next_event":
                    ne_dir = command_run_dirs[idx - 1]
                    metrics_path = ne_dir / "metrics.json"
                    if metrics_path.exists():
                        payload = json.loads(metrics_path.read_text())
                        model_metrics = (payload.get("metrics", {}) or {}).get("model", {})
                        loglike_val = model_metrics.get("loglike")
                        if isinstance(loglike_val, (int, float)):
                            (ne_dir / "loglike.txt").write_text(f"{loglike_val:.6f}\n")
            except Exception:
                # Ignore any issues writing the auxiliary file
                pass
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
                        "rmsex_plus_std": m.get("rmsex_plus_std"),
                        "smape": m.get("smape"),
                        "smape_std": m.get("smape_std"),
                        "rmse_e": m.get("rmse_e"),
                        "rmse_e_std": m.get("rmse_e_std"),
                        "otd": m.get("otd"),
                        "otd_std": m.get("otd_std"),
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
        "rmsex_plus_std",
        "smape",
        "smape_std",
        "rmse_e",
        "rmse_e_std",
        "otd",
        "otd_std",
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

    metrics_to_write = {
        "next_event": ["mae", "rmse", "type_error", "loglike"],
        "long_horizon": [
            "rmsex_plus",
            "rmsex_plus_std",
            "smape",
            "smape_std",
            "rmse_e",
            "rmse_e_std",
            "otd",
            "otd_std",
        ],
    }
    for task, metrics in metrics_to_write.items():
        for metric in metrics:
            save_matrix(metric, task)


def write_latex_row_next_event_fim(results_root: Path, rows: List[Dict]) -> Path:
    """Aggregate next-event 'model' results across datasets into a single LaTeX row.

    The order is fixed to: AMAZON, TAXI, TAOBAO, STACKOVERFLOW-V1
    (matching datasets easytpp/amazon, easytpp/taxi, easytpp/taobao, easytpp/stackoverflow).

    Missing datasets produce empty fields.

    Returns the path to the written .tex file.
    """
    # Build quick index for model rows of next_event task, keyed by dataset id
    model_rows = {(r.get("dataset") or ""): r for r in rows if r.get("task") == "next_event" and r.get("source") == "model"}

    # Dataset ids to look for in fixed table order
    ds_ids = [
        "easytpp/amazon",
        "easytpp/taxi",
        "easytpp/taobao",
        "easytpp/stackoverflow",
    ]

    # Determine if fine-tuned checkpoints are used for next_event (any row contains '_finetune')
    is_finetuned = any(
        (r.get("task") == "next_event" and r.get("source") == "model" and "_finetune" in str(r.get("checkpoint", ""))) for r in rows
    )

    cells: List[str] = []
    for ds in ds_ids:
        r = model_rows.get(ds)
        if r is None:
            # Two empty fields: RMSE and ERROR
            cells.extend(["", ""])
        else:
            rmse = r.get("rmse")
            rmse_ci = r.get("rmse_ci_error")
            terr = r.get("type_error")
            terr_ci = r.get("type_error_ci_error")

            def fmt(val):
                return f"{val:.2f}" if isinstance(val, (int, float)) else ""

            cells.append(f"{fmt(rmse)} $\\pm$ {fmt(rmse_ci)}")
            cells.append(f"{fmt(terr)} $\\pm$ {fmt(terr_ci)}")

    label = "\\textbf{FIM (fine-tuned)}" if is_finetuned else "\\textbf{FIM$^{\\dagger}$}"
    row = " ".join([label, "&", " & ".join(cells), "\\\\"])

    out_path = results_root / "next_event_fim_row.tex"
    out_path.write_text(row)
    return out_path


def write_latex_rows_long_horizon_fim(results_root: Path, rows: List[Dict]) -> Path:
    """Aggregate long-horizon 'model' results into LaTeX rows for the table layout.

    Top row order: Taxi, Taobao
    Middle row order: StackOverflow, Amazon
    Bottom row: Retweet (left 4 cells), remaining 4 cells empty

    Metrics per dataset (order): OTD, RMSE_e, RMSE_{x+}, sMAPE
    """
    # Index long_horizon model rows by dataset id
    model_rows = {(r.get("dataset") or ""): r for r in rows if r.get("task") == "long_horizon" and r.get("source") == "model"}

    # Helper to retrieve a row by either full id (e.g., easytpp/taxi) or short name (e.g., taxi)
    def get_row_for_dataset_short(short_name: str) -> Dict:
        # Direct match on keys
        if short_name in model_rows:
            return model_rows[short_name]
        # Try prefixed EasyTPP id
        pref = f"easytpp/{short_name}"
        if pref in model_rows:
            return model_rows[pref]
        # Try matching by basename of the path/id
        for k, v in model_rows.items():
            base = Path(str(k)).name
            if base == short_name:
                return v
            # Also handle ids like easytpp/amazon -> amazon
            if "/" in str(k) and str(k).split("/")[-1] == short_name:
                return v
        return {}

    def fmt_cell_mean_std(mean_val, std_val) -> str:
        if isinstance(mean_val, (int, float)) and isinstance(std_val, (int, float)):
            return f"$\\mathbf{{{mean_val:.3f}}}$ \\tinymath{{\\pm {std_val:.3f}}}"
        if isinstance(mean_val, (int, float)):
            return f"$\\mathbf{{{mean_val:.3f}}}$"
        return ""

    def cells_for(ds_shorts: List[str]) -> List[str]:
        cells: List[str] = []
        for ds_short in ds_shorts:
            r = get_row_for_dataset_short(ds_short)
            if r is None:
                cells.extend(["", "", "", ""])  # OTD, RMSE_e, RMSE_{x+}, sMAPE
            else:
                otd = r.get("otd")
                otd_std = r.get("otd_std")
                rmse_e = r.get("rmse_e")
                rmse_e_std = r.get("rmse_e_std")
                rmsex_plus = r.get("rmsex_plus")
                rmsex_plus_std = r.get("rmsex_plus_std")
                smape = r.get("smape")
                smape_std = r.get("smape_std")
                cells.extend(
                    [
                        fmt_cell_mean_std(otd, otd_std),
                        fmt_cell_mean_std(rmse_e, rmse_e_std),
                        fmt_cell_mean_std(rmsex_plus, rmsex_plus_std),
                        fmt_cell_mean_std(smape, smape_std),
                    ]
                )
        return cells

    # Use short names; resolver maps from either HF ids or local names
    top_cells = cells_for(["taxi", "taobao"])
    middle_cells = cells_for(["stackoverflow", "amazon"])
    # Retweet occupies the left block; right block intentionally left empty
    retweet_row_cells_left = cells_for(["retweet"])
    retweet_row_cells_right = ["", "", "", ""]

    # Determine if fine-tuned checkpoints are used for long_horizon (any row contains '_finetune')
    is_finetuned = any(
        (r.get("task") == "long_horizon" and r.get("source") == "model" and "_finetune" in str(r.get("checkpoint", ""))) for r in rows
    )
    label = "\\textbf{FIM (fine-tuned)}" if is_finetuned else "\\textbf{FIM}"
    row_top = " ".join([label, "&", " & ".join(top_cells), "\\\\"])  # Taxi | Taobao
    row_middle = " ".join([label, "&", " & ".join(middle_cells), "\\\\"])  # StackOverflow | Amazon
    row_bottom = " ".join([label, "&", " & ".join(retweet_row_cells_left + retweet_row_cells_right), "\\\\"])  # Retweet | (empty)

    out_path = results_root / "long_horizon_fim_rows.tex"
    out_path.write_text(f"{row_top}\n{row_middle}\n{row_bottom}\n")
    return out_path


def main() -> None:
    args = parse_args()

    # Launch the GPU occupier script in the background
    occupier_proc = None
    try:
        # Assume keep_gpu_busy.py is in the same directory as this script
        keep_gpu_script_path = Path(__file__).with_name("keep_gpu_busy.py")
        if keep_gpu_script_path.exists():
            print(f"[INFO] Starting GPU occupier process from: {keep_gpu_script_path}", flush=True)
            # Use Popen for a non-blocking call to run it in the background
            # The environment is inherited, including CUDA_VISIBLE_DEVICES
            occupier_proc = subprocess.Popen([sys.executable, str(keep_gpu_script_path)])
        else:
            print("[WARN] keep_gpu_busy.py not found. GPU might be deallocated by Slurm.", file=sys.stderr, flush=True)

        cfg = yaml.safe_load(Path(args.config).read_text())
        if not cfg.get("datasets"):
            sys.exit("[ERROR] Config must define 'datasets'.")

        ts_root = datetime.now().strftime("%y%m%d-%H%M")
        task = args.task or str(cfg.get("task", "next_event"))
        base = {
            **{k: v for k, v in cfg.items() if k not in ["datasets", "checkpoints"]},
            "task": task,
            "results_root": Path(cfg.get("results_root", "results/fim_benchmark")) / ts_root,
        }

        # Determine checkpoints list. If empty or missing, interpret as "from scratch" and
        # require fine-tuning to be enabled so a model can be trained per dataset.
        checkpoints_list = cfg.get("checkpoints")
        if not checkpoints_list:
            if not bool(cfg.get("fine_tune", False)):
                sys.exit(
                    "[ERROR] 'checkpoints' is empty and fine_tune is False. Enable fine_tune and provide 'finetune_config' to start from scratch."
                )
            # Use a single placeholder per dataset; run_single will fine-tune from scratch
            checkpoints_list = [None]

        run_confs = [{**base, "dataset": d, "checkpoint": ck} for d in cfg.get("datasets", []) for ck in checkpoints_list]
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
            # Also emit a LaTeX row for FIM (next-event) for direct inclusion in papers
            tex_path = write_latex_row_next_event_fim(base["results_root"], rows)
            print(f"[LATEX] Wrote FIM next-event row → {tex_path}")
            # And two LaTeX rows for FIM (long-horizon) matching the table layout
            tex_long = write_latex_rows_long_horizon_fim(base["results_root"], rows)
            print(f"[LATEX] Wrote FIM long-horizon rows → {tex_long}")

        ok = sum(1 for _, _, _, rc in results if rc == 0)
        print(f"\nCompleted {ok}/{total} evaluations. Results are in {base['results_root']}")

    finally:
        # This block ensures that the occupier process is killed when the
        # main script exits, whether it succeeds, fails, or is interrupted.
        if occupier_proc:
            print("[INFO] Terminating GPU occupier process.", flush=True)
            occupier_proc.terminate()
            try:
                occupier_proc.wait(timeout=5)  # Wait briefly for it to exit cleanly
            except subprocess.TimeoutExpired:
                print("[WARN] GPU occupier process did not terminate gracefully, killing.", file=sys.stderr)
                occupier_proc.kill()


if __name__ == "__main__":
    main()
