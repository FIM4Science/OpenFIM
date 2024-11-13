import os
import time
import yaml

import torch
import shutil
import numpy as np
import torch.nn as nn
from torch import Tensor

from pathlib import Path
import lightning.pytorch as pl
from dataclasses import dataclass
from lightning.pytorch import Trainer
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from fim.models.config_dataclasses import FIMSDEConfig

from fim.data.datasets import (
    FIMSDEDataset,
    FIMSDEDatabatch,
    FIMSDEDatabatchTuple
)

from fim.data.dataloaders import (
    FIMSDEDataloader
)

from dataclasses import dataclass,asdict, field
from typing import Any, Dict, Optional, Union, List,Tuple
from fim.models.blocks.base import Mlp,TimeEncoding,TransformerModel
from fim.utils.experiment_files import ExperimentsFiles
from fim.models.blocks import AModel, ModelFactory
from fim.data.datasets import FIMSDEDataset,FIMSDEDatabatchTuple
from fim.pipelines.sde_pipelines import (
    FIMSDEPipeline,
    FIMSDEPipelineOutput
)

from fim.utils.helper import select_dimension_for_plot
from fim.utils.plots.sde_estimation_plots import (
    plot_one_dimension,
    plot_drift_diffussion,
    plot_3d_drift_and_diffusion
)

class FIMSDEL(pl.LightningModule):
    """
    """
    def __init__():
        pass

    # ----------------------------- Lightning functionality ---------------------------------------------^
    def images_log_1D(self,pipeline_sample:FIMSDEPipelineOutput,datatuple:FIMSDEDatabatchTuple,batch_idx:int):
        """ """
        selected_data = select_dimension_for_plot(1,
                                                  datatuple.dimension_mask,
                                                  datatuple.locations,
                                                  pipeline_sample.drift_at_locations_estimator,
                                                  pipeline_sample.diffusion_at_locations_estimator,
                                                  datatuple.drift_at_locations,
                                                  datatuple.diffusion_at_locations,
                                                  index_to_select=0)
        if selected_data is not None:
            print(f"Validation batch index: {batch_idx}")
            locations,drift_at_locations_real,diffusion_at_locations_real,drift_at_locations_estimation,diffusion_at_locations_estimation = selected_data
            fig = plot_one_dimension(locations,
                                        drift_at_locations_real,
                                        drift_at_locations_estimation,
                                        diffusion_at_locations_real,
                                        diffusion_at_locations_estimation,
                                        show=False)
            # Log the figure using the logger (assuming TensorBoard)
            self.logger.experiment.add_figure(f"1D_{batch_idx}", fig, global_step=self.current_epoch)
            self.log_1D_images = True # log images once per epoch

    def images_log_2D(self,pipeline_sample:FIMSDEPipelineOutput,datatuple:FIMSDEDatabatchTuple,batch_idx:int):    
        """ """
        selected_data = select_dimension_for_plot(2,
                                                    datatuple.dimension_mask,
                                                    datatuple.locations,
                                                    pipeline_sample.drift_at_locations_estimator,
                                                    pipeline_sample.diffusion_at_locations_estimator,
                                                    datatuple.drift_at_locations,
                                                    datatuple.diffusion_at_locations,
                                                    index_to_select=0)
        if selected_data is not None:
            print(f"Validation batch index: {batch_idx}")
            locations,drift_at_locations_real,diffusion_at_locations_real,drift_at_locations_estimation,diffusion_at_locations_estimation = selected_data
            fig = plot_drift_diffussion(locations,
                                drift_at_locations_real,
                                diffusion_at_locations_real, 
                                drift_at_locations_estimation, 
                                diffusion_at_locations_estimation, 
                                show=False)
            # Log the figure using the logger (assuming TensorBoard)
            self.logger.experiment.add_figure(f"2D_{batch_idx}", fig, global_step=self.current_epoch)
            self.log_2D_images = True # log images once per epoch

    def images_log_3D(self,pipeline_sample:FIMSDEPipelineOutput,datatuple:FIMSDEDatabatchTuple,batch_idx:int):
        """ """
        selected_data = select_dimension_for_plot(3,
                                                    datatuple.dimension_mask,
                                                    datatuple.locations,
                                                    pipeline_sample.drift_at_locations_estimator,
                                                    pipeline_sample.diffusion_at_locations_estimator,
                                                    datatuple.drift_at_locations,
                                                    datatuple.diffusion_at_locations,
                                                    index_to_select=0)
        if selected_data is not None:
            print(f"Validation batch index: {batch_idx}")
            locations,drift_at_locations_real,diffusion_at_locations_real,drift_at_locations_estimation,diffusion_at_locations_estimation = selected_data
            fig = plot_3d_drift_and_diffusion(
                    locations,
                    drift_at_locations_real,
                    diffusion_at_locations_real, 
                    drift_at_locations_estimation, 
                    diffusion_at_locations_estimation,
                    your_fixed_x_value=-1.,
                    show=False
            )
            # Log the figure using the logger (assuming TensorBoard)
            self.logger.experiment.add_figure(f"3D_{batch_idx}", fig, global_step=self.current_epoch)
            self.log_3D_images = True # log images once per epoch

    def images_log(self,datatuple:FIMSDEDatabatchTuple,batch_idx):
        """ """
        # sample plots
        pipeline = FIMSDEPipeline(self)
        pipeline_sample = pipeline(datatuple,evaluate_paths=False)
        # 2D Images
        if not self.log_1D_images:
            self.images_log_1D(pipeline_sample,datatuple,batch_idx)

        if not self.log_2D_images:
            self.images_log_2D(pipeline_sample,datatuple,batch_idx)

        #if not self.log_3D_images:
        #    self.images_log_3D(pipeline_sample,datatuple,batch_idx)

    # ----------------------------- Lightning Functions ---------------------------------------------
    def prepare_batch(self,batch)->FIMSDEDatabatchTuple:
        """lightning will convert name tuple into a full tensor for training 
        here we create the nametuple as requiered for the model
        """
        databatch = self.DatabatchNameTuple(*batch)
        return databatch
    
    def training_step(
            self, 
            batch, 
            batch_idx
    ):
        optimizer = self.optimizers()
        databatch:FIMSDEDatabatchTuple = self.prepare_batch(batch)
        losses = self.forward(databatch,training=True)

        total_loss = losses["losses"]["total_loss"]
        drift_loss = losses["losses"]["drift_loss"]
        diffusion_loss = losses["losses"]["diffusion_loss"]

        optimizer.zero_grad()
        self.manual_backward(total_loss)
        if self.config.clip_grad:
           torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.clip_max_norm)

        optimizer.step()

        self.log('training_loss', total_loss, on_step=True, prog_bar=True, logger=True)
        self.log('drift_loss', drift_loss, on_step=True, prog_bar=True, logger=True)
        self.log('diffusion_loss', diffusion_loss, on_step=True, prog_bar=True, logger=True)

        return total_loss
    
    def validation_step(
        self, 
        batch, 
        batch_idx
    ):
        databatch = self.prepare_batch(batch)        
        forward_values = self.forward(databatch,training=False,return_all=True)

        total_loss = forward_values.losses["total_loss"]
        drift_loss = forward_values.losses["drift_loss"]
        diffusion_loss = forward_values.losses["diffusion_loss"]

        self.log('val_loss', total_loss, on_step=False, prog_bar=True, logger=True)
        self.log('drift_loss', drift_loss, on_step=False, prog_bar=True, logger=True)
        self.log('diffusion_loss', diffusion_loss, on_step=False, prog_bar=True, logger=True)

        self.images_log(self.target_datatuple,batch_idx)

        return total_loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)

    def on_train_epoch_start(self):
        # Action to be executed at the start of each training epoch
        self.log_1D_images = False
        self.log_2D_images = False
        self.log_3D_images = False

    def on_train_epoch_end(self):
        # Only run every `interval_epochs`
        if (self.current_epoch + 1) % self.interval_epochs == 0:
            self.run_periodic_function()
