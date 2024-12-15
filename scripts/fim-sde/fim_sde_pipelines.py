from fim.data.config_dataclasses import FIMDatasetConfig
from fim.data.data_generation.dynamical_systems_target import generate_all
from fim.models.config_dataclasses import FIMSDEConfig
from fim.models.sde import FIMSDE
from fim.pipelines.sde_pipelines import FIMSDEPipeline


def test_pipeline():
    model_config = FIMSDEConfig()
    data_config = FIMDatasetConfig()

    model = FIMSDE(model_config, data_config)
    databatch_target = model.target_data

    pipeline = FIMSDEPipeline(model)
    pipeline(databatch_target)


def test_target_data():
    data_config = FIMDatasetConfig()
    data_tuple = generate_all(data_config)
    print(data_tuple.drift_at_locations.shape)


if __name__ == "__main__":
    test_pipeline()
