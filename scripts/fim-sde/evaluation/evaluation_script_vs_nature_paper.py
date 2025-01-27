from dataclasses import dataclass
from pathlib import Path
import pickle
from tabulate import tabulate
import torch

from fim.data.utils import load_h5s_in_folder
from fim.utils.plots.fim_sde_quiver_plots import create_2D_quiver_plot



class Dataset:
    def __init__(self, name: str, path: Path, **plot_kwargs):
        self.name = name
        self.path = path
        
        if str(path).endswith(".pickle"):
            with open(path, "rb") as f:
                data = pickle.load(f).results
            self.mean_path = data["sample_paths"].mean(dim=1)[...,:2]
            self.grid = data["sample_paths_grid"][:,:1,...]
        else:
            data = load_h5s_in_folder(path)
            self.mean_path = data["obs_values"][0] # actually thats the ode solution
            self.grid = data["obs_times"][0]
            
        # Make sure that data is on cpu
        self.mean_path = self.mean_path.cpu()
        self.grid = self.grid.cpu()
    


@dataclass
class Benchmark:
    name: str
    GroundTruth: Dataset
    FIM: Dataset
    ComparisonModel: Dataset = None
    

def run_benchmark_and_plot(benchmark: Benchmark):
    assert torch.allclose(benchmark.GroundTruth.grid, benchmark.FIM.grid), "Grids do not match"
    ground_truth_mean = benchmark.GroundTruth.mean_path
    model_mean = benchmark.FIM.mean_path
    
    table = [["FIM",nrmse(model_mean, ground_truth_mean)]]
    
    print(f"Results for {benchmark.name}")
    print(tabulate(table, headers=["Model", "NRMSE Mean Path"]))
    print("\n")
    
def rmse(m_pred, y_true):
   return (m_pred - y_true).pow(2).mean().sqrt()

def nrmse(m_pred, y_true):
    return rmse(m_pred, y_true) / rmse(torch.zeros_like(y_true), y_true)

if __name__ == "__main__":
    benchmarks = [
        Benchmark(
            name="Damped Cubic Oscillator",
            GroundTruth = Dataset("Ground Truth", Path("/cephfs_projects/foundation_models/data/SDE/test/20250124_svise/ode/damped_cubic_oscillator")),
            FIM = Dataset("FIM", Path("evaluations/nature_datasets/01271243_develop/model_evaluations/11M_params/svise_damped_cubic/default11M_params_svise_damped_cubic.pickle")),
        ),
        Benchmark(
            name="Damped Linear Oscillator",
            GroundTruth = Dataset("Ground Truth", Path("/cephfs_projects/foundation_models/data/SDE/test/20250124_svise/ode/damped_linear_oscillator")),
            FIM = Dataset("FIM", Path("evaluations/nature_datasets/01271243_develop/model_evaluations/11M_params/svise_damped_linear/default11M_params_svise_damped_linear.pickle")),
        ),
        Benchmark(
            name="Duffing Oscillator",
            GroundTruth = Dataset("Ground Truth", Path("/cephfs_projects/foundation_models/data/SDE/test/20250124_svise/ode/duffing_oscillator")),
            FIM = Dataset("FIM", Path("evaluations/nature_datasets/01271243_develop/model_evaluations/11M_params/svise_duffing/default11M_params_svise_duffing.pickle")),
        ),
        Benchmark(
            name="Hopf Bifurcation",
            GroundTruth = Dataset("Ground Truth", Path("/cephfs_projects/foundation_models/data/SDE/test/20250124_svise/ode/hopf_bifurcation")),
            FIM = Dataset("FIM", Path("evaluations/nature_datasets/01271243_develop/model_evaluations/11M_params/svise_hopf_bifurcation/default11M_params_svise_hopf_bifurcation.pickle")),
        ),
        Benchmark(
            name="Selkov Glycolysis",
            GroundTruth = Dataset("Ground Truth", Path("/cephfs_projects/foundation_models/data/SDE/test/20250124_svise/ode/selkov_glycolysis")),
            FIM = Dataset("FIM", Path("evaluations/nature_datasets/01271243_develop/model_evaluations/11M_params/svise_selkov_glycolysis/default11M_params_svise_selkov_glycolysis.pickle")),
        ),
    ]
    
    for benchmark in benchmarks:
        run_benchmark_and_plot(benchmark)
    

# SVISE Log10 RMSE Medians    
# -1.5177725118483412
# -1.5225118483412323
# -1.3412322274881516
# -1.5367298578199051
# -1.9466824644549763
# -1.3554502369668247

# SVISE Log10 RMSE Lower bound    
# -1.6171293161814488
# -1.7095463777928233
# -1.4524373730534867
# -1.7853757616790793
# -2.0815842924847665
# -1.6467501692620177

# SVISE Log10 RMSE Upper bound
# -1.390825998645904
# -1.310257278266757
# -1.2664184157075153
# -1.1846648612051456
# -1.846987136086662
# -1.2154705484089372

# PF Log10 RMSE Medians
# -1.4595463777928233
# -1.3209207853757616
# -1.5045700744752877
# -1.4927217332430602
# -1.772342586323629
# -1.3422477995937712

# PF Log10 RMSE Lower bound
# -1.5815842924847665
# -1.4631008801624916
# -1.6135748138117807
# -1.6538591740013542
# -1.8825321597833446
# -1.4571767095463777

# PF Log10 RMSE Upper bound
# -1.367298578199052
# -1.2097156398104265
# -1.4277251184834123
# -1.1623222748815167
# -1.6670616113744077
# -1.1812796208530805