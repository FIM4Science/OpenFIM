from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import pickle
from tabulate import tabulate
import torch
import json

from fim.data.utils import load_h5s_in_folder
from fim.utils.plots.fim_sde_quiver_plots import create_2D_quiver_plot

from compute_mmd import get_mmd

class ModelResults:
    def __init__(self, name: str, path: Path, **plot_kwargs):
        self.name = name
        self.path = path
        
        if str(path).endswith(".json"):
            with open(path, "r") as f:
                data = json.load(f)
        self.data = data
    


@dataclass
class Benchmark:
    FIM_11M: ModelResults
    FIM_20M: ModelResults
    BISDE: ModelResults
    SparseGP: ModelResults
    

def get_dataset_idx(data, dataset_name):
    """
    Fuck me that I have to write this function...
    """
    data_names = [d["name"] for d in data]
    if 'Glycosis' in data_names:
        data_names[data_names.index('Glycosis')] = "Glycolysis"
    if 'Syn Drift' in data_names:
        data_names[data_names.index('Syn Drift')] = "SynDrift"
    if 'Double Well Max Diffusion' in data_names:
        data_names[data_names.index('Double Well Max Diffusion')] = "DoubleWell"
    if 'Damped Linear' in data_names:
        data_names[data_names.index('Damped Linear')] = "DampedLinear"
    if 'Damped Cubic' in data_names:
        data_names[data_names.index('Damped Cubic')] = "DampedCubic"
    if dataset_name not in data_names:
        raise ValueError(f"Dataset {dataset_name} not found in {data_names}")
    return data_names.index(dataset_name)


def throw_away_nan_paths(estimated_paths, ground_truth_paths, dataset_name="", model_name=""):
    original_num_paths = estimated_paths.shape[0]
    # Detect which rows contain NaNs in estimated_paths
    valid_idx = torch.tensor([i for i in range(original_num_paths) if not torch.isnan(estimated_paths[i]).any()], dtype=torch.long)
    estimated_paths = estimated_paths[valid_idx]
    ground_truth_paths = ground_truth_paths[valid_idx]
    if len(valid_idx) < original_num_paths:
        print(f"Warning: {len(estimated_paths) - len(valid_idx)} NaN paths in {dataset_name} for {model_name}")
    return estimated_paths, ground_truth_paths    


def run_benchmark(benchmark: Benchmark, evaluation_dir):
    with open(evaluation_dir / "results.txt","w") as f: 
        for dataset_name in ['Duffing', 'Glycolysis', 'SynDrift', 'Wang', 'DoubleWell', 'DampedLinear', 'DampedCubic']:
            table = []
            for model_name in ['FIM_11M', 'FIM_20M', 'BISDE', 'SparseGP']:
                model = getattr(benchmark, model_name)
                dataset_idx = get_dataset_idx(model.data, dataset_name)
                data = model.data[dataset_idx]
                # drift_rmse = rmse(data["estimated_drift_at_locations"], data["real_drift_at_locations"])
                # diffusion_rmse = rmse(data["estimated_diffusion_at_locations"], data["real_diffusion_at_locations"])
                ground_truth_paths = torch.tensor(data["real_paths"])
                estimated_paths = torch.tensor(data["synthetic_paths"])
                if not torch.isnan(estimated_paths).any():
                    # mean_path_rmse = rmse(estimated_paths.mean(dim=0), ground_truth_paths.mean(dim=0))
                    mmd = get_mmd(estimated_paths, ground_truth_paths)
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
    evaluation_path =  "evaluations"
    time: str = str(datetime.now().strftime("%m%d%H%M"))
    evaluation_dir: Path = Path(f"evaluations/ksig/{time}")
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    
    benchmarks = [
        Benchmark(
            FIM_11M = ModelResults("FIM 11M", Path("data/processed/test/paths/fim_11M_params_stride_1_observations_many_paths.json")),
            FIM_20M = ModelResults("FIM 20M", Path("data/processed/test/paths/fim_20M_params_trained_even_longer_stride_10_many_paths.json")),
            BISDE = ModelResults("BISDE", Path("data/processed/test/paths/bisde_dense_observations_paths.json")),
            SparseGP = ModelResults("SparseGP", Path("data/processed/test/paths/sparse_gp_dense_observations_many_paths.json")),
        ),
    ]
    
    for benchmark in benchmarks:
        run_benchmark(benchmark, evaluation_dir)
