from dataclasses import asdict
from fim.models.sde import FIMSDE
from fim.data.config_dataclasses import FIMDatasetConfig
from fim.models.config_dataclasses import FIMSDEConfig
from fim.utils.experiment_files import ExperimentsFiles
from fim.data.dataloaders import FIMSDEDataloader
from fim.pipelines.sde_pipelines import FIMSDEPipeline

def test_load_model():
    experiment_dir = r"C:\Users\cesar\Desktop\Projects\FoundationModels\FIM\results\1732042941"
    experiment_files = ExperimentsFiles(experiment_dir=experiment_dir)
    model_config = FIMSDEConfig.from_yaml(experiment_files.model_config_yaml)
    data_config = FIMDatasetConfig.from_yaml(experiment_files.data_config_yaml)

    checkpoint_path = experiment_files.get_lightning_checkpoint_path("best")
    model = FIMSDE.load_from_checkpoint(checkpoint_path,model_config=model_config,data_config=data_config,map_location="cuda")
    dataloader = FIMSDEDataloader(**asdict(data_config))
    pipeline = FIMSDEPipeline(model)
    test_output = pipeline(model.target_data)
    print(test_output.drift_at_locations_estimator)


if __name__=="__main__":
    test_load_model()
