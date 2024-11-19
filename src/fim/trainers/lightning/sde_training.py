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
from fim.pipelines.sde_pipelines import(
    FIMSDEPipeline,
    sample_and_save_from_test
)

from fim.utils.experiment_files import ExperimentsFiles
from pathlib import Path
from fim.data.config_dataclasses import FIMDatasetConfig
from fim.data.dataloaders import FIMSDEDataloader
from fim.data.datasets import FIMSDEDatabatchTuple
from fim.utils.helper import save_hyperparameters_to_yaml
from fim import results_path

def train_fim_sde(model_config:FIMSDEConfig,data_config:FIMDatasetConfig):
    """
    This function creates an MLFlow Logger and a Lightning Trainer to train FIMSDE_p
    """
    ml_flow_folder = Path(results_path) / "mlruns"
    experiment_name="sde"

    # Experiment Files
    experiment_files = ExperimentsFiles(experiment_indentifier=None,delete=True)
    model_config.experiment_dir = experiment_files.experiment_dir
    model_config.experiment_name = experiment_name

    # Set up TensorBoard logger
    logger = MLFlowLogger(experiment_name=experiment_name,
                          tracking_uri=f"file:{ml_flow_folder}")
    
    # Set up Model Checkpointing
    checkpoint_callback_best = ModelCheckpoint(dirpath=experiment_files.checkpoints_dir,
                                               save_top_k=1, 
                                               monitor="val_loss",
                                               filename="best-{epoch:02d}")
    checkpoint_callback_last = ModelCheckpoint(dirpath=experiment_files.checkpoints_dir,
                                               save_top_k=1,
                                               monitor=None,
                                               filename="last-{epoch:02d}")

    data_config = asdict(data_config)
    dataloaders = FIMSDEDataloader(**data_config)
    data_config = FIMDatasetConfig(**data_config)
    model = FIMSDE(model_config,data_config)
    # Set up Model
    save_hyperparameters_to_yaml(model_config,experiment_files.model_config_yaml)
    save_hyperparameters_to_yaml(data_config,experiment_files.data_config_yaml)

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
    
    # Save test samples from best model
    checkpoint_path = experiment_files.get_lightning_checkpoint_path("best")
    model = FIMSDE.load_from_checkpoint(checkpoint_path,model_config=model_config,data_config=data_config,map_location="cuda")
    sample_and_save_from_test(model,dataloaders,experiment_files)

if __name__=="__main__":
    model_config = FIMSDEConfig(num_epochs=4,
                                log_images_every_n_epochs=2)
    data_config = FIMDatasetConfig()
    train_fim_sde(model_config,data_config)

