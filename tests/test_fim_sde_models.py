import os

import numpy as np
import torch

from fim.data.dataloaders import DataLoaderFactory
from fim.data.datasets import FIMSDEDatabatchTuple
from fim.models.blocks import ModelFactory
from fim.utils.helper import load_yaml, nametuple_to_device


def test_model_kosta_init():
    from fim import project_path

    parameters_yaml = os.path.join(project_path, "configs", "train", "fim-sde", "fim-train-patrick.yaml")
    config = load_yaml(parameters_yaml, return_object=True)

    torch.manual_seed(int(config.experiment.seed))
    torch.cuda.manual_seed(int(config.experiment.seed))
    np.random.seed(int(config.experiment.seed))
    torch.cuda.empty_cache()
    device_map = config.experiment.device_map

    config = config.to_dict()
    dataloader = DataLoaderFactory.create(**config["dataset"])
    if hasattr(dataloader, "update_kwargs"):
        # fim model requieres that config is updated after loading the data
        config = dataloader.update_kwargs(config)

    model = ModelFactory.create_deprecated(config, device_map=device_map, resume=False)
    databatch: FIMSDEDatabatchTuple = next(dataloader.train_it.__iter__())
    databatch = nametuple_to_device(databatch, device_map)
    forward_pass = model(databatch, training=True)
    print(forward_pass)


def test_model_():
    pass


if __name__ == "__main__":
    test_model_kosta_init()
