from dataclasses import asdict
from fim.data.config_dataclasses import FIMDatasetConfig
from fim.data.dataloaders import FIMSDEDataloader
from fim.data.datasets import FIMSDEDatabatchTuple

from fim.models.sde import FIMSDE
from fim.models.config_dataclasses import FIMSDEConfig
from fim.pipelines.sde_pipelines import FIMSDEPipeline
from fim.data.data_generation.dynamical_systems_target import generate_all
from fim.utils.helper import select_dimension_for_plot
from fim.utils.plots.sde_estimation_plots import (
    plot_one_dimension,
    plot_drift_diffussion,
    plot_3d_drift_and_diffusion
)

def test_pipeline():
    model_config = FIMSDEConfig()
    data_config = FIMDatasetConfig()
    
    model = FIMSDE(model_config,data_config)
    databatch_target = model.target_data

    pipeline = FIMSDEPipeline(model)
    pipeline_output = pipeline(databatch_target)
    
def test_target_data():
    data_config = FIMDatasetConfig()
    data_tuple = generate_all(data_config)
    print(data_tuple.drift_at_locations.shape)

if __name__=="__main__":
    test_pipeline()
