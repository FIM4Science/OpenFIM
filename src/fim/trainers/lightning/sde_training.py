import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import os
import time

import torch
import shutil
import numpy as np
import torch.nn as nn

from pathlib import Path
from dataclasses import dataclass
from dataclasses import dataclass,asdict, field
from typing import Any, Dict, Optional, Union, List,Tuple

from lightning.pytorch import Trainer
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from dataclasses import asdict
from fim.data.config_dataclasses import FIMDatasetConfig
from fim.data.dataloaders import FIMSDEDataloader
from fim.data.datasets import FIMSDEDatabatchTuple

from fim.models.sde import FIMSDE
from fim.models.config_dataclasses import FIMSDEConfig
from fim.pipelines.sde_pipelines import FIMSDEPipeline
from fim.utils.experiment_files import ExperimentsFiles

from fim.data.config_dataclasses import FIMDatasetConfig
from fim.data.dataloaders import FIMSDEDataloader
from fim.data.datasets import FIMSDEDatabatchTuple
from fim.utils.helper import save_hyperparameters_to_yaml

def train_fim_sde(params:FIMSDEConfig):
    """
    This function creates an MLFlow Logger and a Lightning Trainer to train FIMSDE_p
    """
    # Experiment Files
    experiment_files = ExperimentsFiles(experiment_indentifier=None,
                                        delete=True)
    # Set up TensorBoard logger
    # logger = MLFlowLogger(experiment_name="time_series_transformer",tracking_uri='http://localhost:5000')
    logger = TensorBoardLogger(experiment_files.tensorboard_dir, 
                               name=experiment_files.experiment_indentifier)

    # Set up Model Checkpointing
    checkpoint_callback_best = ModelCheckpoint(dirpath=experiment_files.checkpoints_dir,
                                               save_top_k=1, 
                                               monitor="val_loss",
                                               filename="best-{epoch:02d}")
    
    checkpoint_callback_last = ModelCheckpoint(dirpath=experiment_files.checkpoints_dir,
                                               save_top_k=1,
                                               monitor=None,
                                               filename="last-{epoch:02d}")
    
    model_config = FIMSDEConfig()
    data_config = FIMDatasetConfig()

    dataloaders = FIMSDEDataloader(**asdict(data_config))
    model = FIMSDE(model_config,data_config)

    # Set up Model
    save_hyperparameters_to_yaml(model_config,experiment_files.params_yaml)
    #save_hyperparameters_to_yaml(data_config,experiment_files.params_yaml)

    #Set up trainers
    trainer = Trainer(
        default_root_dir=experiment_files.experiment_dir,
        logger=logger,
        log_every_n_steps=1,
        max_epochs=model_config.num_epochs,
        callbacks=[checkpoint_callback_best,
                   checkpoint_callback_last]
    )
    trainer.fit(model, 
                dataloaders.train_it,
                dataloaders.validation_it)

if __name__=="__main__":
    params = FIMSDEConfig(num_epochs=10,
                          temporal_embedding_size=20)
    train_fim_sde(params)

