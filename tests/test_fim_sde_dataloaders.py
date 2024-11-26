from torch.utils.data import DataLoader
from fim.data.dataloaders import DataLoaderFactory
from fim.utils.helper import load_yaml

from fim.data.config_dataclasses import FIMDatasetConfig
from fim.data.datasets import FIMSDEDataset, FIMSDEDatabatchTuple
from fim.data.data_generation.dynamical_systems_sample import define_dynamicals_models_from_yaml


def test_datasets():
    parameters_yaml = r"C:\Users\cesar\Desktop\Projects\FoundationModels\FIM\configs\train\fim-sde\fim-train-patrick.yaml"
    config = load_yaml(parameters_yaml, return_object=True)
    config_data: FIMDatasetConfig = FIMDatasetConfig(**config.dataset.to_dict())
    train_dataset = FIMSDEDataset(config_data, config_data.dataset_path_collections.train)
    dataloader = DataLoader(train_dataset, batch_size=5)
    databatch: FIMSDEDatabatchTuple = next(dataloader.__iter__())
    print(databatch.diffusion_at_locations.shape)


def test_dataloaders():
    parameters_yaml = r"C:\Users\cesar\Desktop\Projects\FoundationModels\FIM\configs\train\fim-sde\fim-train-patrick.yaml"
    config = load_yaml(parameters_yaml, return_object=True)
    dataloader = DataLoaderFactory.create(**config.dataset.to_dict())
    databatch: FIMSDEDatabatchTuple = next(dataloader.train_it.__iter__())
    print(databatch.diffusion_at_locations.shape)


def test_synthetic_dataloaders():
    parameters_yaml = r"C:\Users\cesar\Desktop\Projects\FoundationModels\FIM\configs\train\fim-sde\fim-train-dynamical-systems.yaml"
    config = load_yaml(parameters_yaml, return_object=True)
    dataloader = DataLoaderFactory.create(**config.dataset.to_dict())
    databatch: FIMSDEDatabatchTuple = next(dataloader.train_it.__iter__())
    print(databatch.diffusion_at_locations.shape)


def test_models_from_yaml():
    from fim import project_path

    yaml_file = rf"{project_path}\configs\train\fim-sde\sde-systems-hyperparameters.yaml"
    dataset_type, experiment_name, train_studies, test_studies, validation_studies = define_dynamicals_models_from_yaml(
        yaml_file, return_data=True
    )
    print(train_studies[0])


if __name__ == "__main__":
    test_dataloaders()
