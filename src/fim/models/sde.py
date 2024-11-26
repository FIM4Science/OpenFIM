
import torch
import torch.nn as nn
from torch import Tensor

import lightning.pytorch as pl
from dataclasses import dataclass

from fim.models.config_dataclasses import FIMSDEConfig
from fim.pipelines.sde_pipelines import FIMSDEPipeline

from fim.data.datasets import (
    FIMSDEDatabatchTuple
)

from dataclasses import field
from typing import Dict, Optional, Tuple
from fim.models.blocks.base import MLP,TransformerModel
from fim.models.blocks import ModelFactory
from fim.data.data_generation.dynamical_systems_target import generate_all
from fim.data.config_dataclasses import FIMDatasetConfig

from fim.utils.plots.sde_estimation_plots import (
    images_log_1D,
    images_log_2D,
    images_log_3D
)

# 1. Define your query generation model (a simple linear layer can work)
class QueryGenerator(nn.Module):
    def __init__(self, input_dim, query_dim):
        super(QueryGenerator, self).__init__()
        self.linear = nn.Linear(input_dim, query_dim)

    def forward(self, x):
        return self.linear(x)

# 2. Define a static query matrix as a learnable parameter
class StaticQuery(nn.Module):
    def __init__(self, num_steps, query_dim):
        super(StaticQuery, self).__init__()
        self.queries = nn.Parameter(torch.randn(num_steps, query_dim))  # Learnable queries

    def forward(self):
        return self.queries

@dataclass
class FIMSDEForward:
    """
    This class carries all objects required for the forward pass evaluation 
    and subsequent loss evaluation, this includes input and target data 
    as well as all the estimator heads. 

    THE MAIN GOAL IS TO KEEP TRACK OF HOW NORMALIZATIONS ARE 
    PERFORMED AND WHEN
    
    WE ASSUME THAT THE INITIAL VALUE OF THE TIME IS 0

    B: batch size
    P: number of paths
    T: number of time steps
    G: grid size
    D: dimensions
    """
    # Estimators (Learned Concepts)
    drift_estimator: Optional[Tensor] = None  # [B,P,G,D]
    log_var_drift_estimator: Optional[Tensor] = None  # [B,P,G,D]
    diffusion_estimator: Optional[Tensor] = None  # [B,P,G,D]
    log_var_diffusion_estimator: Optional[Tensor] = None  # [B,P,G,D]

    # Targets (Data Concepts)
    drift_target: Optional[Tensor] = None  # [B,P,G,D]
    diffusion_target: Optional[Tensor] = None  # [B,P,G,D]

    # Data
    locations: Optional[Tensor] = None  # [B,P,G,D]
    obs_times: Optional[Tensor] = None  # [B,P,T,1]
    obs_values: Optional[Tensor] = None  # [B,P,T,D]

    # Normalization stats
    max_obs_times: Optional[Tensor] = None  # [B,P,T,1]
    max_obs_values: Optional[Tensor] = None  # [B,1,D]
    min_obs_values: Optional[Tensor] = None  # [B,1,D]

    range_obs_vals: Optional[Tensor] = None  # [B,1,D]
    range_obs_times: Optional[Tensor] = None  # [B,1,D]

    # masks
    obs_mask: Optional[Tensor] = None  # [B,P,T,D]
    dimension_mask: Optional[Tensor] = None  # [B,P,T,D]

    # Basic stats and flags
    is_data_set: bool = False
    is_target_set: bool = False
    is_estimator_set: bool = False

    is_input_normalized: bool = False
    is_target_normalized: bool = False
    is_estimator_normalized: bool = False

    #loss
    losses:Dict[str,Tensor] = field(default_factory=lambda:{})

    min_border_factor:float = 2.

    def set_input_data(self, obs_times: Tensor, obs_values: Tensor, obs_mask: Tensor, locations: Tensor,dimension_mask:Tensor):
        """Sets observation data and related variables."""
        self.obs_times = obs_times
        self.obs_values = obs_values
        self.obs_mask = obs_mask
        self.locations = locations
        self.dimension_mask = dimension_mask
        self.is_data_set = True

    def set_target_data(self, drift_data: Tensor, diffusion_data: Tensor):
        """Sets target data for drift and diffusion."""
        self.drift_target = drift_data
        self.diffusion_target = diffusion_data
        self.is_target_set = True

    def set_forward_estimators(self, drift_estimator: Tensor, diffusion_estimator: Tensor,
                               var_drift_estimator: Tensor, var_diffusion_estimator: Tensor):
        """
        Sets estimators for forward pass.

        IF INPUT DATA IS NORMALIZED WHEN ESTIMATOR IS SET
        ESTIMATOR IS ASSUMED NORMALIZED
        """
        self.drift_estimator = drift_estimator
        self.diffusion_estimator = diffusion_estimator
        self.log_var_drift_estimator = var_drift_estimator
        self.log_var_diffusion_estimator = var_diffusion_estimator

        if self.is_input_normalized:
            self.is_estimator_normalized = True
        else:
            self.is_estimator_normalized = False

    def set_losses(self,losses):
        """ Set the losses"""
        self.losses = losses

    def normalize_input(self):
        """ 
        Normalizes observation data, locations, and time using shared min/max values.
        
        TAKEN FROM THE APPPENDIX B.1
        """
        # Flatten obs_values across P and T for computing min/max
        B, P, T, D = self.obs_values.shape
        obs_values_reshaped = self.obs_values.view(B, P * T, D)

        # Calculate min, max, and range for normalization (excluding time)
        self.min_obs_values = obs_values_reshaped.min(dim=1, keepdim=True).values.unsqueeze(1)
        self.max_obs_values = obs_values_reshaped.max(dim=1, keepdim=True).values.unsqueeze(1)
        self.range_obs_vals = (self.max_obs_values - self.min_border_factor*self.min_obs_values).unsqueeze(1)

        # Normalize obs_values
        self.obs_values = (self.obs_values - self.min_border_factor*self.min_obs_values) / self.range_obs_vals

        # Normalize locations using the same range
        if self.locations is not None:
            self.locations = (self.locations - self.min_border_factor*self.min_obs_values) / self.range_obs_vals

        # Normalize obs_times by dividing by max_obs_times
        obs_time_reshaped = self.obs_times.view(B, P * T, 1)
        self.max_obs_times = obs_time_reshaped.max(dim=1, keepdim=True).values.unsqueeze(1)  # [B,1,1,1]
        self.range_obs_times = self.max_obs_times # times are assumed to start at zero

        self.obs_times = self.obs_times/self.range_obs_times
        self.is_input_normalized = True

    def normalize_concepts(self,drift_data,var_drift_data,diffusion_data,var_diffusion_data):
        # Normalize drift and diffusion targets
        if drift_data is not None:
            # scale from times normalisation
            drift_data = drift_data / self.range_obs_times
            drift_data = drift_data*self.range_obs_vals

        if diffusion_data is not None:
            # scale from times normalisation
            diffusion_data = diffusion_data/torch.sqrt(self.range_obs_vals)
            diffusion_data = diffusion_data*self.range_obs_times

        # var part
        if var_drift_data is not None:
            var_drift_data = var_drift_data + 2*torch.log(self.range_obs_vals) - 2*torch.log(self.range_obs_times)

        if var_diffusion_data is not None:
            var_diffusion_data = var_diffusion_data + 2.*torch.log(self.range_obs_vals) - torch.log(self.range_obs_times)

        return drift_data,diffusion_data

    def unnormalize_concepts(self,drift_data,var_drift_data,diffusion_data,var_diffusion_data):
        # Normalize drift and diffusion targets
        if drift_data is not None:
            # scale from times normalisation
            drift_data = drift_data*self.range_obs_times
            drift_data = drift_data/self.range_obs_vals

        if diffusion_data is not None:
            # scale from times normalisation
            diffusion_data = diffusion_data*torch.sqrt(self.range_obs_vals)
            diffusion_data = diffusion_data/self.range_obs_times

        # var part
        if var_drift_data is not None:
            var_drift_data = var_drift_data - torch.log(self.range_obs_vals) + torch.log(self.range_obs_times)

        if var_diffusion_data is not None:
            var_diffusion_data = var_diffusion_data - torch.log(self.range_obs_vals)+ .5*torch.log(self.range_obs_times)


        return drift_data,diffusion_data

    def normalize_targets(self):
        """Normalizes target data (drift_data and diffusion_data) using shared min/max values from data."""
        if not self.is_target_normalized:
            if self.is_target_set:
                self.drift_target,self.diffusion_target = self.normalize_concepts(self.drift_target,
                                                                                  None,
                                                                                  self.diffusion_target,
                                                                                  None)
        self.is_target_normalized = True

    def normalize_estimators(self):
        """Normalizes estimator data (drift and diffusion estimators) using shared min/max values from data."""
        if not self.is_estimator_normalized:
            if self.is_estimator_set:
                self.drift_estimator,self.diffusion_estimator = self.normalize_concepts(self.drift_estimator,
                                                                                        self.log_var_drift_estimator,
                                                                                        self.diffusion_estimator,
                                                                                        self.log_var_diffusion_estimator)
        self.is_estimator_normalized = True

    def unnormalize_input(self):
        """Restores original scale for normalized fields."""
        if not self.is_input_normalized:
            return

        # Restore original scale for obs_values and locations
        self.obs_values = self.obs_values * self.range_obs_vals + self.min_obs_values
        if self.locations is not None:
            self.locations = self.locations * self.range_obs_vals + self.min_obs_values

        # Restore original scale for obs_times
        self.obs_times = self.obs_times * self.range_obs_times

        self.is_input_normalized = False

    def unnormalize_targets(self):
        if self.is_target_normalized:
            if self.is_target_set:
                self.drift_target,self.diffusion_target = self.unnormalize_concepts(self.drift_target,
                                                                                    None,
                                                                                    self.diffusion_target,
                                                                                    None)
        self.is_target_normalized = True

    def unnormalize_estimators(self):
        if self.is_estimator_normalized:
            if self.is_estimator_set:
                self.drift_estimator,self.diffusion_estimator = self.unnormalize_concepts(self.drift_estimator,
                                                                                          self.log_var_drift_estimator,
                                                                                          self.diffusion_estimator,
                                                                                          self.log_var_diffusion_estimator)
        self.is_estimator_normalized = True

    def normalize_all(self):
        if not self.is_input_normalized:
            self.normalize_input()
        if not self.is_estimator_normalized:
            self.normalize_estimators()
        if not self.is_target_normalized:
            self.normalize_targets()

    def unnormalize_all(self):
        if self.is_input_normalized:
            self.unnormalize_input()
        if self.is_estimator_normalized:
            self.unnormalize_estimators()
        if self.is_target_normalized:
            self.unnormalize_targets()

# 3. Model Following FIM conventions
#class FIMSDE(AModel)
class FIMSDE(pl.LightningModule):
    """
    Stochastic Differential Equation Trainining
    """
    model_config:FIMSDEConfig
    data_config:FIMDatasetConfig

    def __init__(
            self,
            model_config:dict,
            data_config:dict,
            device_map:torch.device = None,
            **kwargs,
        ):
        super(FIMSDE, self).__init__()

        # Save hyperparameters
        self.save_hyperparameters()

        # Set hyperparameters
        if isinstance(model_config,dict):
            self.model_config = FIMSDEConfig(**model_config)
        else:
            self.model_config = model_config

        if isinstance(data_config,dict):
            self.data_config = FIMDatasetConfig(**data_config)
        else:
            self.data_config = data_config

        self._create_modules()

        # Set a dataset for fixed evaluation
        self.target_data = generate_all(self.model_config)

        if device_map is not None:
            self.to(device_map)

        self.DatabatchNameTuple = FIMSDEDatabatchTuple
        # Important: This property activates manual optimization (Lightning)
        self.automatic_optimization = False

    def _create_modules(
        self,
    ):
        # Define different versions
        x_dimension = self.data_config.max_dimension
        x_dimension_full = x_dimension*3 # we encode the difference and its square
        spatial_plus_time_encoding = self.model_config.temporal_embedding_size + self.model_config.spatial_embedding_size
        self.psi_1_tokes_dim = self.model_config.sequence_encoding_tokenizer*self.model_config.sequence_encoding_transformer_heads

        # basic embedding
        self.phi_0t = SineTimeEncoding(self.model_config.temporal_embedding_size)
        self.phi_0x = MLP(
            in_features=x_dimension_full,
            out_features=self.model_config.spatial_embedding_size,
            hidden_layers=self.model_config.spatial_embedding_hidden_layers
        )

        # trunk network
        self.trunk = MLP(
            in_features=x_dimension,
            out_features=self.psi_1_tokes_dim,
            hidden_layers=self.model_config.trunk_net_hidden_layers
        )

        #ensures that the embbeding that is sent to the transformer is a multiple of the number of heads
        self.phi_xt = nn.Linear(spatial_plus_time_encoding,
                        self.psi_1_tokes_dim)

        # path transformer (causal encoding of paths)
        self.psi_1 = TransformerModel(input_dim=self.psi_1_tokes_dim,
                                nhead=self.model_config.sequence_encoding_transformer_heads,
                                hidden_dim=self.model_config.sequence_encoding_transformer_hidden_size,
                                nlayers=self.model_config.sequence_encoding_transformer_layers)

        # time attention
        self.omega_1 = nn.MultiheadAttention(
            self.psi_1_tokes_dim,
            self.model_config.combining_transformer_heads,
            batch_first=True,
        )

        # path attention
        self.path_queries = nn.Parameter(torch.randn(1, self.psi_1_tokes_dim))

        self.omega_2 = nn.MultiheadAttention(
            self.psi_1_tokes_dim,
            self.model_config.combining_transformer_heads,
            batch_first=True,
        )

        # drift head
        self.drift_head = nn.Linear(self.psi_1_tokes_dim,self.data_config.max_dimension)
        self.log_var_drift_head = nn.Linear(self.psi_1_tokes_dim,self.data_config.max_dimension)

        #diffusion head
        self.diffusion_head = nn.Linear(self.psi_1_tokes_dim,self.data_config.max_dimension)
        self.log_var_diffusion_head = nn.Linear(self.psi_1_tokes_dim,self.data_config.max_dimension)

    def path_encoding(
            self,
            databatch:FIMSDEDatabatchTuple,
            locations:Optional[Tensor]=None,
        ) -> Tuple[torch.tensor,torch.tensor]:
        """
        This function obtains the paths encodings with functional
        attention, the intent is to provide a representation for
        the series of paths

        Args:

            databatch:FIMSDEpDatabatchTuple|FIMSDEpDatabatch
                keys,values:
                    locations (Tensor [B, G, D]) 
                        where to evaluate the drift and diffusion function 
                    obs_values (Tensor [B, P, T+1, D])
                        observation values. optionally with noise. 
                    obs_times (Tensor [B, P, T+1, 1])
                    observation_mask (dtype: bool) 
                        (0: value is observed, 1: value is masked out)
            
            locations (tensor):
                where to evaluate the drift and diffusion function 

            training (bool): 
                flag indicating if model is in training mode. Has an impact on the output.
            
            with B: batch size, T: number of observation times, P: number of paths, D: dimensionsm, G: number of fine grid points (locations)

        Returns:
            b(x) (Tensor)  [B,G,psi_1_tokes_dim] representation for system at each grid point  
            h(x) (Tensor)  [B,P,H,psi_1_tokes_dim] representation per path at each grid point
        """
        if locations is None:
            locations = databatch.locations

        B,P,T,D = databatch.obs_values.shape
        T = T-1
        G = locations.size(1)

        # include the square of the difference
        X = databatch.obs_values[:,:,:-1,:]
        dX = databatch.obs_values[:,:,1:,:] - databatch.obs_values[:,:,:-1,:]
        obs_times = databatch.obs_times[:,:,:-1,:]

        dX2 = dX**2
        x_full = torch.cat([X.unsqueeze(-1),dX.unsqueeze(-1),dX2.unsqueeze(-1)],dim=-1)
        x_full = x_full.view(x_full.shape[0], x_full.shape[1], x_full.shape[2], -1)

        spatial_encoding = self.phi_0x(x_full) # [B,P,T,spatial_embedding_size]
        time_encoding = self.phi_0t(obs_times) # [B,P,T,temporal_embedding_size]

        # trunk
        trunk_encoding = self.trunk(locations) #[B,H,trunk_dim]
        trunk_encoding = trunk_encoding[:,None,:,:].repeat(1,P,1,1)  # [B,P,G,trunk_size]
        trunk_encoding = trunk_encoding.view(B*P,G,-1)

        # embbedded input
        U =  torch.cat([spatial_encoding,time_encoding],dim=-1) #  [B,P,T,spatial_plus_time_encoding]
        U = self.phi_xt(U) #  [B,P,T,psi_1_tokes_dim]

        # TRANSFORMER THAT CREATES A REPRESENTATION FOR THE PATHS
        U = U.view(B*P,T,self.psi_1_tokes_dim)
        H = self.psi_1(torch.transpose(U,0,1))  # [T,B*P,psi_1_tokes_dim]
        H = torch.transpose(H,0,1) # [B*P,T,psi_1_tokes_dim]

        # Attention on Time -> One representation per path
        hx,_ = self.omega_1(trunk_encoding,H,H) # [B*P,G,psi_1_tokes_dim]
        hx = hx.view(B,P,G,-1) # [B,P,G,psi_1_tokes_dim]

        # Attention on Paths -> One representation per expression
        hx_ = hx.transpose(1,2).reshape(G*B,P,-1) # [B*G,P,psi_1_tokes_dim]
        path_queries_ = self.path_queries[None,:,:].repeat(G*B,1,1)
        bx,_ = self.omega_2(path_queries_,hx_,hx_)
        bx = bx.view(B,G,-1) # [B,G,psi_1_tokes_dim]

        return bx,hx

    def forward(
            self,
            databatch:FIMSDEDatabatchTuple,
            locations:Optional[Tensor]=None,
            training:bool=True,
            return_all:bool=False,
            return_heads:bool=False,
        ) -> Tuple[Tensor]|FIMSDEForward:
        """
        Args:
            databatch FIMPOODEDataBulk
            training (bool) if True returns Dict
        
            Returns
                if training true returns Dict of losses

                if return_all true returns FIMSDEForward with everything
        """
        # Dataclass to Handle Normalization
        forward_expressions = FIMSDEForward()

        if hasattr(databatch,"obs_mask"):
            obs_mask = databatch.obs_mask
        else:
            obs_mask = None

        forward_expressions.set_input_data(obs_times=databatch.obs_times,
                                           obs_values=databatch.obs_values,
                                           obs_mask=obs_mask,
                                           locations=databatch.locations,
                                           dimension_mask=databatch.dimension_mask)
        forward_expressions.normalize_input()

        # Path Encoding
        bx,hx = self.path_encoding(databatch,locations)

        # Drift Heads
        drift_estimator = self.drift_head(bx)
        log_var_drift_estimator = self.log_var_drift_head(bx)
        diffusion_estimator = self.diffusion_head(bx)
        log_var_diffusion_estimator = self.log_var_diffusion_head(bx)

        forward_expressions.set_forward_estimators(drift_estimator=drift_estimator,
                                                   diffusion_estimator=diffusion_estimator,
                                                   var_drift_estimator=log_var_drift_estimator,
                                                   var_diffusion_estimator=log_var_diffusion_estimator)

        # Loss
        forward_expressions.set_target_data(drift_data=databatch.drift_at_locations,
                                            diffusion_data=databatch.diffusion_at_locations)

        # Returns
        if training:
            losses = self.loss(forward_expressions)
            return {"losses":losses}
        else:
            if return_all:
                losses = self.loss(forward_expressions)
                forward_expressions.set_losses(losses)
                forward_expressions.unnormalize_all()
                return forward_expressions

            if return_heads:
                forward_expressions.unnormalize_all()
                return (forward_expressions.drift_estimator,
                        forward_expressions.log_var_drift_estimator,
                        forward_expressions.diffusion_estimator,
                        forward_expressions.log_var_diffusion_estimator)

    def var_loss(self,estimator,target,log_var_estimator,dimension_mask):
        """
        loss with log var
        """
        loss_ = (estimator - target)**2.
        var = torch.exp(log_var_estimator)
        loss_ = loss_/(2.*var) + .5*log_var_estimator

        # Apply the dimension mask and keep finite values
        loss_masked = torch.where((dimension_mask == 1) & torch.isfinite(loss_), loss_, torch.zeros_like(loss_))
        # Replace NaNs and Infs with zeros in the masked loss
        loss_ = torch.where(torch.isfinite(loss_masked), loss_masked, torch.zeros_like(loss_masked))

        # filter out
        loss_ = loss_.sum(-1) # sum dimension
        loss_ = loss_.sum(-1) # sum time
        loss_ = torch.sqrt(loss_.mean())

        return loss_

    def rmse_loss(self,estimator,target,dimension_mask):
        """
        root mean square loss applying the dimension masks
        """
        loss_ = (estimator - target)**2.

        # Apply the dimension mask and keep finite values
        loss_masked = torch.where((dimension_mask == 1) & torch.isfinite(loss_), loss_, torch.zeros_like(loss_))
        # Replace NaNs and Infs with zeros in the masked loss
        loss_ = torch.where(torch.isfinite(loss_masked), loss_masked, torch.zeros_like(loss_masked))

        # Filter out NaNs in the masked tensor
        loss_ = loss_.sum(-1)
        loss_ = loss_.sum(-1)
        loss_ = torch.sqrt(loss_.mean())
        return loss_

    def loss(
            self,
            forward_expressions:FIMSDEForward,
        ):
        """
        forward_expressions

        Compute the loss of the FIMODE_mix model (in original space).
            Makes sure that the mask is included in the computation of the loss

        The loss consists of supervised losses
            - rmse of the vector field values at fine grid points
            
        Args:
            forward_expressions (FIMSDEForward): 
        Returns:
            Tensor: {"total_loss":total_loss,"drift_loss":drift_loss,"diffusion_loss":diffusion_loss}
        """
        # ENSURES THAT ESTIMATOR AND TARGET LIE IN THE SAME UNITS
        if self.model_config.train_with_normalized_head:
            forward_expressions.normalize_all()
        else:
            forward_expressions.unnormalize_all()

        if self.model_config.loss_type == "rmse":
            drift_loss = self.rmse_loss(forward_expressions.drift_estimator,
                                        forward_expressions.drift_target,
                                        forward_expressions.dimension_mask)

            diffusion_loss = self.rmse_loss(forward_expressions.diffusion_estimator,
                                        forward_expressions.diffusion_target,
                                        forward_expressions.dimension_mask)

        elif self.model_config.loss_type == "var":
            drift_loss = self.var_loss(forward_expressions.drift_estimator,
                                       forward_expressions.drift_target,
                                       forward_expressions.log_var_drift_estimator,
                                       forward_expressions.dimension_mask)

            diffusion_loss = self.var_loss(forward_expressions.diffusion_estimator,
                                           forward_expressions.diffusion_target,
                                           forward_expressions.log_var_diffusion_estimator,
                                           forward_expressions.dimension_mask)

        total_loss = drift_loss + self.model_config.diffusion_loss_scale*diffusion_loss
        losses = {"loss":total_loss,"drift_loss":drift_loss,"diffusion_loss":diffusion_loss}
        return losses

    # ----------------------------- Lightning Functionality ---------------------------------------------
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

        total_loss = losses["losses"]["loss"]
        drift_loss = losses["losses"]["drift_loss"]
        diffusion_loss = losses["losses"]["diffusion_loss"]

        optimizer.zero_grad()
        self.manual_backward(total_loss)
        if self.model_config.clip_grad:
           torch.nn.utils.clip_grad_norm_(self.parameters(), self.model_config.clip_max_norm)
        optimizer.step()

        self.log('loss', total_loss, on_step=True, prog_bar=True, logger=True)
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

        total_loss = forward_values.losses["loss"]
        drift_loss = forward_values.losses["drift_loss"]
        diffusion_loss = forward_values.losses["diffusion_loss"]

        self.log('val_loss', total_loss, on_step=False, prog_bar=True, logger=True)
        self.log('drift_loss', drift_loss, on_step=False, prog_bar=True, logger=True)
        self.log('diffusion_loss', diffusion_loss, on_step=False, prog_bar=True, logger=True)

        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.model_config.learning_rate)

    def on_train_epoch_start(self):
        # Action to be executed at the start of each training epoch
        if (self.current_epoch + 1) % self.model_config.log_images_every_n_epochs == 0:
            pipeline = FIMSDEPipeline(self)
            pipeline_sample = pipeline(self.target_data)
            self.images_log(self.target_data,pipeline_sample)

    def images_log(self,databatch,pipeline_sample):
        fig = images_log_1D(databatch,pipeline_sample)
        if fig is not None:
            mlflow_client_ = self.logger.experiment
            mlflow_client_.log_figure(self.logger._run_id, fig, f"{self.current_epoch}_1D.png")

        fig = images_log_2D(databatch,pipeline_sample)
        if fig is not None:
            mlflow_client_ = self.logger.experiment
            mlflow_client_.log_figure(self.logger._run_id, fig, f"{self.current_epoch}_2D.png")

        fig = images_log_3D(databatch,pipeline_sample)
        if fig is not None:
            mlflow_client_ = self.logger.experiment
            mlflow_client_.log_figure(self.logger._run_id, fig, f"{self.current_epoch}_3D.png")

ModelFactory.register("FIMSDE",FIMSDE,with_data_params=True)
