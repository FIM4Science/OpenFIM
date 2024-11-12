import os
import fim
import torch
import numpy as np

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
from fim.models.blocks import ModelFactory
from fim.utils.helper import nametuple_to_device

def test_model():
    parameters_yaml = r"C:\Users\cesar\Desktop\Projects\FoundationModels\FIM\configs\train\fim-sde\fim-train-patrick.yaml"

    config = load_yaml(parameters_yaml,return_object=True)
    torch.manual_seed(int(config.experiment.seed))
    torch.cuda.manual_seed(int(config.experiment.seed))
    np.random.seed(int(config.experiment.seed))
    torch.cuda.empty_cache()
    device_map = config.experiment.device_map

    config = config.to_dict()
    dataloader = DataLoaderFactory.create(**config["dataset"])
    if hasattr(dataloader,"update_kwargs"):
        # fim model requieres that config is updated after loading the data
        dataloader.update_kwargs(config)
    model = ModelFactory.create(config,device_map=device_map,resume=False)
    databatch:FIMSDEDatabatchTuple = next(dataloader.train_it.__iter__())
    databatch = nametuple_to_device(databatch,device_map)

    forward_pass = model(databatch,training=True)
    print(forward_pass)


if __name__=="__main__":
    test_model()