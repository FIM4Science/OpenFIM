import os
import fim
from torch.utils.data import DataLoader
from fim.data.dataloaders import (
    DataLoaderFactory
)
from fim.utils.helper import (
    GenericConfig, 
    expand_params, 
    load_yaml
)

from fim.data.datasets import FIMSDEDataset,FIMSDEDatabatchTuple
from fim.data.config_dataclasses import FIMDatasetConfig

def test_datasets():
    parameters_yaml = r"C:\Users\cesar\Desktop\Projects\FoundationModels\FIM\configs\train\fim-sde\fim-train-patrick.yaml"
    config = load_yaml(parameters_yaml,return_object=True)
    config_data:FIMDatasetConfig = FIMDatasetConfig(**config.dataset.to_dict())
    train_dataset = FIMSDEDataset(config_data,config_data.dataset_path_collections.train)
    dataloader = DataLoader(train_dataset,batch_size=5)
    databatch:FIMSDEDatabatchTuple = next(dataloader.__iter__())
    print(databatch.diffusion_at_locations.shape)

def test_dataloaders():
    parameters_yaml = r"C:\Users\cesar\Desktop\Projects\FoundationModels\FIM\configs\train\fim-sde\fim-train-patrick.yaml"
    config = load_yaml(parameters_yaml,return_object=True)
    dataloader = DataLoaderFactory.create(**config.dataset.to_dict())
    databatch:FIMSDEDatabatchTuple = next(dataloader.train_it.__iter__())
    print(databatch.diffusion_at_locations.shape)

if __name__=="__main__":
    test_dataloaders()
