import os

from torch.utils.data import DataLoader

from fim.data.config_dataclasses import FIMDatasetConfig
from fim.data.data_generation.dynamical_systems_sample import define_dynamicals_models_from_yaml
from fim.data.dataloaders import DataLoaderFactory
from fim.data.datasets import FIMSDEDatabatchTuple, FIMSDEDataset
from fim.utils.helper import load_yaml


def test_datasets():
    from fim import project_path

    parameters_yaml = os.path.join(project_path, "configs", "train", "fim-sde", "fim-train-patrick.yaml")
    config = load_yaml(parameters_yaml, return_object=True)
    config_data: FIMDatasetConfig = FIMDatasetConfig(**config.dataset.to_dict())
    train_dataset = FIMSDEDataset(config_data, config_data.dataset_path_collections.train)
    dataloader = DataLoader(train_dataset, batch_size=5)
    databatch: FIMSDEDatabatchTuple = next(dataloader.__iter__())
    print(databatch.diffusion_at_locations.shape)


def test_dataloaders():
    from fim import project_path

    parameters_yaml = os.path.join(project_path, "configs", "train", "fim-sde", "fim-train-patrick.yaml")
    config = load_yaml(parameters_yaml, return_object=True)
    dataloader = DataLoaderFactory.create(**config.dataset.to_dict())
    databatch: FIMSDEDatabatchTuple = next(dataloader.train_it.__iter__())
    print(databatch.diffusion_at_locations.shape)


def test_synthetic_dataloaders():
    from fim import project_path

    parameters_yaml = os.path.join(project_path, "configs", "train", "fim-sde", "fim-train-dynamical-systems.yaml")

    config = load_yaml(parameters_yaml, return_object=True)
    dataloader = DataLoaderFactory.create(**config.dataset.to_dict())
    databatch: FIMSDEDatabatchTuple = next(dataloader.train_it.__iter__())
    print(databatch.diffusion_at_locations.shape)


def test_models_from_yaml():
    from fim import project_path

    yaml_file = os.path.join(project_path, "configs", "train", "fim-sde", "sde-systems-hyperparameters.yaml")
    dataset_type, experiment_name, train_studies, test_studies, validation_studies = define_dynamicals_models_from_yaml(
        yaml_file, return_data=True
    )
    print(train_studies[0])


if __name__ == "__main__":
    test_models_from_yaml()
