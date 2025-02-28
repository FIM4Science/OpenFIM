import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from compute_mmd import get_mmd
from tabulate import tabulate


tau = 0.002  # [0.002, 0.01, 0.02]
max_num_paths = 100


class ModelResults:
    def __init__(self, name: str, path: Path):
        self.model_name = name
        self.path = path

        if str(path).endswith(".json"):
            with open(path, "r") as f:
                data = json.load(f)
        self.data = data

    def __call__(self, experiment_id: int, dataset_name: str):
        if self.model_name == "GroundTruth":
            dataset_idx = [d["name"] for d in self.data].index(dataset_name)
            return torch.tensor(self.data[dataset_idx]["real_paths"][experiment_id])[
                :max_num_paths
            ]  # [num_paths, num_time_points, num_dims]
        else:
            dataset_idx = None
            for i, d in enumerate(self.data):
                if d["name"] == dataset_name and d["tau"] == tau:
                    dataset_idx = i
                    break
            if dataset_idx is None:
                raise ValueError(f"Dataset {dataset_name} with tau {tau} not found in {self.path}")
            return torch.tensor(self.data[dataset_idx]["synthetic_paths"][experiment_id])[:max_num_paths]


@dataclass
class Benchmark:
    GroundTruth: ModelResults
    # FIM: ModelResults
    BISDE: ModelResults
    # SparseGP: ModelResults


ground_truth_cache = {}  # K_xx for the ground truth data is the same for all models, so we cache it


def run_benchmark(benchmark: Benchmark, evaluation_dir):
    ground_truth = benchmark.GroundTruth
    for experiment_id in range(5):
        with open(evaluation_dir / f"results_{experiment_id}.txt", "w") as f:
            # for dataset_name in ['Damped Linear', 'Damped Cubic', 'Duffing', 'Glycosis', 'Hopf', 'Double Well', 'Wang']:
            for dataset_name in ["Damped Cubic", "Double Well"]:
                table = []
                # for model_name in ['FIM', 'BISDE', 'SparseGP']:
                # for model_name in ['FIM', 'SparseGP']:
                for model_name in ["BISDE"]:
                    model = getattr(benchmark, model_name)
                    ground_truth_paths = ground_truth(experiment_id, dataset_name)
                    model_paths = model(experiment_id, dataset_name)
                    if not torch.isnan(model_paths).any():
                        mmd = get_mmd(ground_truth_paths, model_paths, kernel_cache=ground_truth_cache)
                        table.append([model_name, mmd])

                print(f"Results for {dataset_name}")
                print(tabulate(table, headers=["Model", "MMD"]))
                print("\n")

                f.write(f"Results for {dataset_name}\n")
                f.write(tabulate(table, headers=["Model", "MMD"]))
                f.write("\n\n")


def rmse(m_pred, y_true):
    return (m_pred - y_true).pow(2).mean().sqrt()


def nrmse(m_pred, y_true):
    return rmse(m_pred, y_true) / rmse(torch.zeros_like(y_true), y_true)


if __name__ == "__main__":
    # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
    evaluation_path = "evaluations"
    time: str = str(datetime.now().strftime("%m%d%H%M"))
    evaluation_dir: Path = Path(f"evaluations/ksig/{time}/{tau}")
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    benchmarks = [
        Benchmark(
            GroundTruth=ModelResults(
                "GroundTruth",
                Path(
                    "/cephfs_projects/foundation_models/data/SDE/evaluation/20250129_coarse_synthetic_systems_5000_points_data/ksig_reference_paths.json"
                ),
            ),
            # FIM = ModelResults("FIM", Path("/cephfs_projects/foundation_models/data/SDE/evaluation/20250129_coarse_synthetic_systems_5000_points_data/20M_trained_even_longer_synthetic_paths.json")),
            BISDE=ModelResults("BISDE", Path("data/processed/test/bisde_experiments_friday_full.json")),
            # SparseGP = ModelResults("SparseGP", Path("data/processed/test/sparse_gp_sparse_observations_many_paths.json")),
        ),
    ]

    for benchmark in benchmarks:
        run_benchmark(benchmark, evaluation_dir)
