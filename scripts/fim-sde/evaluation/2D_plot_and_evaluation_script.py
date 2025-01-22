from dataclasses import dataclass
from pathlib import Path
import pickle

import torch

from fim.data.utils import load_h5s_in_folder
from fim.utils.plots.fim_sde_quiver_plots import create_2D_quiver_plot



class Dataset:
    def __init__(self, name: str, path: Path, **plot_kwargs):
        self.name = name
        self.path = path
        
        if str(path).endswith(".pickle"):
            with open(path, "rb") as f:
                data = pickle.load(f).results["estimated_concepts"]
            self.locations, self.drift, self.diffusion = data.locations[:,:,:2], data.drift[:,:,:2], data.diffusion[:,:,:2]
        else:
           data = load_h5s_in_folder(path)
           self.locations, self.drift, self.diffusion = data["locations"], data["drift_at_locations"], data["diffusion_at_locations"]
    


@dataclass
class Benchmark:
    name: str
    GroundTruth: Dataset
    FIM: Dataset
    ComparisonModel: Dataset = None
    
    # Some plot kwargs
    stride: int = 1
    zoom_area_drift: tuple = None
    zoom_area_diffusion: tuple = None
    zoom_position_drift: str = 'lower left'
    zoom_position_diffusion: str = 'lower left'
    inset_scale_drift: float = 0.2
    inset_scale_diffusion: float = 0.5
    

def run_benchmark_and_plot(benchmark: Benchmark):
    ground_truth_locations, ground_truth_drift, ground_truth_diffusion = benchmark.GroundTruth.locations, benchmark.GroundTruth.drift, benchmark.GroundTruth.diffusion
    model_locations, model_drift, model_diffusion = benchmark.FIM.locations, benchmark.FIM.drift, benchmark.FIM.diffusion
    if benchmark.ComparisonModel is not None:
        comparison_model_locations, comparison_model_drift, comparison_model_diffusion = benchmark.ComparisonModel.locations, benchmark.ComparisonModel.drift, benchmark.ComparisonModel.diffusion
    else:
        comparison_model_locations, comparison_model_drift, comparison_model_diffusion = None, None, None
    assert model_locations.shape == ground_truth_locations.shape, f"Locations have different shapes between model and ground truth data: {model_locations.shape} vs {ground_truth_locations.shape}"
    if comparison_model_locations is not None:
        assert model_locations.shape == comparison_model_locations.shape, f"Locations have different shapes between model and comparison model data: {model_locations.shape} vs {comparison_model_locations.shape}"
    assert model_drift.shape == ground_truth_drift.shape, f"Drifts have different shapes between model and ground truth data: {model_drift.shape} vs {ground_truth_drift.shape}"
    if comparison_model_drift is not None:
        assert model_drift.shape == comparison_model_drift.shape, f"Drifts have different shapes between model and comparison model data: {model_drift.shape} vs {comparison_model_drift.shape}"
    assert model_diffusion.shape == ground_truth_diffusion.shape, f"Diffusions have different shapes between model and ground truth data: {model_diffusion.shape} vs {ground_truth_diffusion.shape}"
    if comparison_model_diffusion is not None:
        assert model_diffusion.shape == comparison_model_diffusion.shape, f"Diffusions have different shapes between model and comparison model data: {model_diffusion.shape} vs {comparison_model_diffusion.shape}"
    assert torch.allclose(model_locations, ground_truth_locations), "Locations are not the same between model and ground truth data"
    if comparison_model_locations is not None:
        assert torch.allclose(model_locations, comparison_model_locations), "Locations are not the same between model and comparison model data"
    
    stride = benchmark.stride    
    model_locations_plot, model_drift_plot, model_diffusion_plot = model_locations[:,::stride], model_drift[:,::stride], model_diffusion[:,::stride]
    ground_truth_drift_plot, ground_truth_diffusion_plot = ground_truth_drift[:,::stride], ground_truth_diffusion[:,::stride]
    if comparison_model_locations is not None:
        comparison_model_locations_plot, comparison_model_drift_plot, comparison_model_diffusion_plot = comparison_model_locations[:,::stride], comparison_model_drift[:,::stride], comparison_model_diffusion[:,::stride]
    else:
        comparison_model_locations_plot, comparison_model_drift_plot, comparison_model_diffusion_plot = None, None, None
    
    ## Some code to restrict the plot to a certain region
    # too_large_mask = (torch.abs(model_locations) > 1)
    # # Compute the logical or between the last two dimensions
    # too_large_mask = too_large_mask.any(dim=-1)[:,:,None]
    # too_large_mask = too_large_mask.repeat(1, 1, 2)

    # model_locations[too_large_mask] = 0
    # model_drift[too_large_mask] = 0
    # model_diffusion[too_large_mask] = 0
    # ground_truth_drift[too_large_mask] = 0
    # ground_truth_diffusion[too_large_mask] = 0
    

    create_2D_quiver_plot(
        model_locations_plot, 
        ground_truth_drift_plot, 
        ground_truth_diffusion_plot, 
        model_drift_plot, 
        model_diffusion_plot, 
        comparison_model_drift_plot, 
        comparison_model_diffusion_plot, 
        zoom_area_drift=benchmark.zoom_area_drift, 
        zoom_area_diffusion=benchmark.zoom_area_diffusion, 
        zoom_position_drift=benchmark.zoom_position_drift, 
        zoom_position_diffusion=benchmark.zoom_position_diffusion, 
        inset_scale_drift=benchmark.inset_scale_drift, 
        inset_scale_diffusion=benchmark.inset_scale_diffusion, 
        title=benchmark.name
    )
    


if __name__ == "__main__":
    benchmarks = [
        Benchmark(
            name="Wang 2D",
            GroundTruth = Dataset("Ground Truth", Path("/cephfs_projects/foundation_models/data/SDE/test/20241223_opper_and_wang_cut_to_128_lenght_paths/two_d_wang_80000_points")),
            FIM = Dataset("FIM", Path("evaluations/synthetic_datasets/01221033_30k_deg_3_ablation_studies_only_learn_scale/model_evaluations/11M_params/wang_two_d_80000_points/default11M_params_wang_two_d_80000_points.pickle")),
            ComparisonModel = Dataset("BISDE", Path("data/processed/test/20250117_wang_estimated_equations/bisde_est_2D_synth_80000_points_split_128_length")),
            stride = 10,
            zoom_area_drift = dict(xlim=(2, 2.8), ylim=(-1.1, -0.9)),
            zoom_area_diffusion = dict(xlim=(1.2, 2.2), ylim=(-2, -1)),
            inset_scale_drift = 0.2,
            inset_scale_diffusion = 0.5,
        ),
        Benchmark(
            name="Damped Cubic Oscillator",
            GroundTruth = Dataset("Ground Truth", Path("/cephfs_projects/foundation_models/data/SDE/test/20241222_svise_1_perc_diffusion_no_additive_noise/damped_cubic_oscillator")),
            FIM = Dataset("FIM", Path("evaluations/synthetic_datasets/01221033_30k_deg_3_ablation_studies_only_learn_scale/model_evaluations/11M_params/svise_damped_cubic/default11M_params_svise_damped_cubic.pickle")),
            stride = 10,
        ),
        Benchmark(
            name="Damped Linear Oscillator",
            GroundTruth = Dataset("Ground Truth", Path("/cephfs_projects/foundation_models/data/SDE/test/20241222_svise_1_perc_diffusion_no_additive_noise/damped_linear_oscillator")),
            FIM = Dataset("FIM", Path("evaluations/synthetic_datasets/01221033_30k_deg_3_ablation_studies_only_learn_scale/model_evaluations/11M_params/svise_damped_linear/default11M_params_svise_damped_linear.pickle")),
            stride = 10,
        ),
        Benchmark(
            name="Duffing Oscillator",
            GroundTruth = Dataset("Ground Truth", Path("/cephfs_projects/foundation_models/data/SDE/test/20241222_svise_1_perc_diffusion_no_additive_noise/duffing_oscillator")),
            FIM = Dataset("FIM", Path("evaluations/synthetic_datasets/01221033_30k_deg_3_ablation_studies_only_learn_scale/model_evaluations/11M_params/svise_duffing/default11M_params_svise_duffing.pickle")),
            stride = 10,
        ),
        Benchmark(
            name="Hopf Bifurcation",
            GroundTruth = Dataset("Ground Truth", Path("/cephfs_projects/foundation_models/data/SDE/test/20241222_svise_1_perc_diffusion_no_additive_noise/hopf_bifurcation")),
            FIM = Dataset("FIM", Path("evaluations/synthetic_datasets/01221033_30k_deg_3_ablation_studies_only_learn_scale/model_evaluations/11M_params/svise_hopf_bifurcation/default11M_params_svise_hopf_bifurcation.pickle")),
            stride = 10,
        ),
        Benchmark(
            name="Selkov Glycolysis",
            GroundTruth = Dataset("Ground Truth", Path("/cephfs_projects/foundation_models/data/SDE/test/20241222_svise_1_perc_diffusion_no_additive_noise/selkov_glycolysis")),
            FIM = Dataset("FIM", Path("evaluations/synthetic_datasets/01221033_30k_deg_3_ablation_studies_only_learn_scale/model_evaluations/11M_params/svise_selkov_glycolysis/default11M_params_svise_selkov_glycolysis.pickle")),
            stride = 10,
        ),
    ]
    
    for benchmark in benchmarks:
        run_benchmark_and_plot(benchmark)
    
    