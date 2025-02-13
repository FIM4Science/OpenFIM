import copy
import logging
from typing import Any, Dict

import torch
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel, PretrainedConfig

from fim.models.blocks import MultiHeadLearnableQueryAttention

from ..utils.helper import create_class_instance
from ..utils.logging import RankLoggerAdapter
from .blocks import AModel, ModelFactory, RNNEncoder
from .blocks.neural_operators import AttentionOperator


class FIMHawkesConfig(PretrainedConfig):
    model_type = "fimhawkes"

    def __init__(
        self,
        mark_encoder: dict = None,
        time_encoder: dict = None,
        delta_time_encoder: dict = None,
        kernel_time_encoder: dict = None,
        evaluation_mark_encoder: dict = None,     
        ts_encoder: dict = None,
        time_dependent_functional_attention: dict = None,
        static_functional_attention: dict = None,
        static_functional_attention_learnable_query: dict = None,
        kernel_value_decoder: dict = None,
        base_intensity_decoder: dict = None,
        hidden_dim: int = None,
        max_num_marks: int = 1,
        **kwargs,
    ):
        self.max_num_marks = max_num_marks
        self.mark_encoder = mark_encoder
        self.time_encoder = time_encoder
        self.delta_time_encoder = delta_time_encoder
        self.kernel_time_encoder = kernel_time_encoder
        self.evaluation_mark_encoder = evaluation_mark_encoder
        self.ts_encoder = ts_encoder
        self.time_dependent_functional_attention = time_dependent_functional_attention
        self.static_functional_attention = static_functional_attention
        self.static_functional_attention_learnable_query = static_functional_attention_learnable_query
        self.kernel_value_decoder = kernel_value_decoder
        self.base_intensity_decoder = base_intensity_decoder
        self.hidden_dim = hidden_dim

        super().__init__(**kwargs)


class FIMHawkes(AModel):
    """
    FIMHawkes: A Neural Recognition Model for Zero-Shot Inference of Hawkes Processes

    Attributes:
        max_num_marks (int): Maximum number of marks in the Hawkes process.
        mark_encoder (nn.Module): The mark encoder for the observed data.
        time_encoder (nn.Module): The time encoder for the observed data.
        delta_time_encoder (nn.Module): The delta time encoder.
        kernel_time_encoder (nn.Module): The kernel time encoder.
        evaluation_mark_encoder (nn.Module): The mark encoder for the selected mark during evaluation.
        ts_encoder (nn.Module): The time series encoder.
        time_dependent_functional_attention (nn.Module): The time dependent functional attention.
        static_functional_attention (nn.Module): The static functional attention.
        static_functional_attention_learnable_query (nn.Module): The learnable query for static functional attention.
        kernel_value_decoder (nn.Module): The kernel value decoder.
        base_intensity_decoder (nn.Module): The base intensity decoder.
        loss: TBD

    """
    

    config_class = FIMHawkesConfig

    def __init__(self, config: FIMHawkesConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.max_num_marks = config.max_num_marks
        self.logger = RankLoggerAdapter(logging.getLogger(self.__class__.__name__))
        self.__create_modules()

        # self.gaussian_nll = nn.GaussianNLLLoss(full=True, reduction="none")
        # self.init_cross_entropy = nn.CrossEntropyLoss(reduction="none")
        

    def __create_modules(self) -> None:
        mark_encoder = copy.deepcopy(self.config.mark_encoder)
        time_encoder = copy.deepcopy(self.config.time_encoder)
        delta_time_encoder = copy.deepcopy(self.config.delta_time_encoder)
        kernel_time_encoder = copy.deepcopy(self.config.kernel_time_encoder)
        evaluation_mark_encoder = copy.deepcopy(self.config.evaluation_mark_encoder)
        ts_encoder = copy.deepcopy(self.config.ts_encoder)
        time_dependent_functional_attention = copy.deepcopy(self.config.time_dependent_functional_attention)
        static_functional_attention = copy.deepcopy(self.config.static_functional_attention)
        static_functional_attention_learnable_query = copy.deepcopy(self.config.static_functional_attention_learnable_query)
        kernel_value_decoder = copy.deepcopy(self.config.kernel_value_decoder)
        base_intensity_decoder = copy.deepcopy(self.config.base_intensity_decoder)
        self.hidden_dim = self.config.hidden_dim
           

        mark_encoder["in_features"] = self.max_num_marks
        self.mark_encoder = create_class_instance(mark_encoder.pop("name"), mark_encoder)
        time_encoder["in_features"] = 1
        self.time_encoder = create_class_instance(time_encoder.pop("name"), time_encoder)
        delta_time_encoder["in_features"] = 1
        self.delta_time_encoder = create_class_instance(delta_time_encoder.pop("name"), delta_time_encoder)
        kernel_time_encoder["in_features"] = 1
        kernel_time_encoder["out_features"] = self.hidden_dim
        self.kernel_time_encoder = create_class_instance(kernel_time_encoder.pop("name"), kernel_time_encoder)
        evaluation_mark_encoder["in_features"] = self.max_num_marks
        evaluation_mark_encoder["out_features"] = self.hidden_dim
        self.evaluation_mark_encoder = create_class_instance(evaluation_mark_encoder.pop("name"), evaluation_mark_encoder)
        
        ts_encoder["encoder_layer"]["d_model"] = self.hidden_dim
        self.ts_encoder = create_class_instance(ts_encoder.pop("name"), ts_encoder)
        
        
        self.time_dependent_functional_attention = AttentionOperator(
            embed_dim=self.hidden_dim, out_features=self.hidden_dim, **time_dependent_functional_attention
        )
        
        self.static_functional_attention = AttentionOperator(
            embed_dim=self.hidden_dim, out_features=self.hidden_dim, **static_functional_attention
        )
        
        static_functional_attention_learnable_query["in_features"] = self.max_num_marks
        self.static_functional_attention_learnable_query = create_class_instance(static_functional_attention_learnable_query.pop("name"), static_functional_attention_learnable_query)

        kernel_value_decoder["in_features"] = self.static_functional_attention_learnable_query.out_features
        kernel_value_decoder["out_features"] = 1
        self.kernel_value_decoder = create_class_instance(kernel_value_decoder.pop("name"), kernel_value_decoder)

        base_intensity_decoder["in_features"] = self.static_functional_attention_learnable_query.out_features
        base_intensity_decoder["out_features"] = 1
        self.base_intensity_decoder = create_class_instance(base_intensity_decoder.pop("name"), base_intensity_decoder)
        

    def forward(self, x: dict[str, Tensor], schedulers: dict = None, step: int = None) -> dict:
        """
        Forward pass for the model.

        Args:
            x (dict[str, Tensor]): A dictionary containing the input tensors:
                - "event_times": Tensor representing the event times. [B, P, L, 1]
                - "event_types": Tensor representing the event types. [B, P, L, 1]
                - "kernel_grids": Tensor representing the times at which to evaluate the kernel. [B, M, L_kernel]
                - Optional keys:
                    - "base_intensities": Tensor representing the ground truth base intensity. [B, M]
                    - "kernel_evaluations": Tensor representing the ground truth kernel evaluation values. [B, M, L_kernel]
                    - "seq_lengths": Tensor representing the sequence lengths (which we use for masking). [B, P]
            schedulers (dict, optional): A dictionary of schedulers for the training process. Default is None.
            step (int, optional): The current step in the training process. Default is None.
        Returns:
            dict: A dictionary containing the following keys:
                - "kernel_eval_values": Tensor representing the predicted kernel evaluation values. [B, M, L_kernel]
                - "baseline_intensity": Tensor representing the predicted baseline intensity. [B, M]
                - "losses" (optional): Tensor representing the calculated losses, if the required keys are present in `x`.
        """
        obs_grid = x["event_times"]
        B, P, L = obs_grid.shape[:3]

        x["delta_times"] = obs_grid[:, :, 1:] - obs_grid[:, :, :-1]
        # Add a delta time of 0 for the first event
        x["delta_times"] = torch.cat([torch.zeros_like(x["delta_times"][:, :, :1]), x["delta_times"]], dim=2)
        
        # FIXME: Do this inside the dataloader
        x["seq_lengths"] = torch.tensor([L] * B * P, device=self.device)
        x["seq_lengths"] = x["seq_lengths"].view(B, P)
        
        observations_padding_mask = self._generate_padding_mask(x["seq_lengths"], L).unsqueeze(-1) # [B, P, L, 1]
        
        if "time_normalization_factors" not in x:
            norm_constants, obs_grid = self.__normalize_obs_grid(obs_grid)
            x["time_normalization_factors"] = norm_constants
            x["observation_grid_normalized"] = obs_grid
        else:
            norm_constants = x["time_normalization_factors"]
            x["observation_grid_normalized"] = obs_grid

        # FIXME: REMOVE THIS!
        x["kernel_grids"] = x["kernel_grids"][:, :, ::10]
        x["kernel_evaluations"] = x["kernel_evaluations"][:, :, ::10]
        # self.logger.warning("Kernel grids and evaluations are truncated to 10!")

        sequence_encodings = self._encode_observations(x)  # [B, P, L, D]

        time_dependent_encodings = self._time_dependent_encoder(x, sequence_encodings, observations_padding_mask=observations_padding_mask)  # [B, M, L_kernel, D]
        
        static_encodings = self._static_encoder(x, sequence_encodings, observations_padding_mask=observations_padding_mask)  # [B, M, D]
    
        predicted_kernel_values = self._kernel_value_decoder(time_dependent_encodings)  # [B, M, L_kernel]

        predicted_base_intensity = torch.exp(self._base_intensity_decoder(static_encodings))  # [B, M]
        
        norm_constants = norm_constants.view(B, 1, 1)
        predicted_kernel_values = predicted_kernel_values / norm_constants
        predicted_base_intensity = predicted_base_intensity / norm_constants.squeeze(-1)

        out = {
            "predicted_kernel_values": predicted_kernel_values,
            "predicted_base_intensity": predicted_base_intensity,
        }
        if "base_intensities" in x and "kernel_evaluations" in x:
            out["losses"] = self.loss(
                x["kernel_grids"],
                predicted_kernel_values,
                predicted_base_intensity,
                x["kernel_evaluations"],
                x["base_intensities"],
                schedulers,
                step,
            )

        return out

    def _encode_observations(self, x: dict) -> Tensor:
        obs_grid_normalized = x["observation_grid_normalized"]
        
        encodings_per_event_mark = self.mark_encoder(torch.nn.functional.one_hot(torch.arange(self.max_num_marks, device=self.device), num_classes=self.max_num_marks).float())
        B, P, L = obs_grid_normalized.shape[:3]

        time_enc = self.time_encoder(obs_grid_normalized)
        delta_time_enc = self.delta_time_encoder(x["delta_times"])
        # Select encoding from encodings_per_event_mark from event_types
        state_enc = encodings_per_event_mark[x["event_types"].view(-1).int()].view(B, P, L, -1)
        path = time_enc + delta_time_enc + state_enc
        causal_mask = torch.triu(torch.ones(L, L),diagonal=1).bool().to(self.device)
        causal_mask = causal_mask.repeat(B, P, 1, 1)
                
        positions = torch.arange(L, device=self.device).unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1, 1, L, 1)

        # Expand seq_lengths to (B, P, 1, 1) and compare with positions to create padding mask
        padding_mask = positions >= x["seq_lengths"].unsqueeze(-1).unsqueeze(-1)  # (B, P, L, 1)
        padding_mask = padding_mask.expand(-1, -1, -1, L)  # (B, P, L, L)

        # Combine causal mask with padding mask
        mask = causal_mask | padding_mask
                
        h = self.ts_encoder(path.view(B * P, L, -1), mask=mask.view(B*P, L, L), is_causal=True)

        return h.view(B, P, L, -1)

    def _trunk_net_encoder(self, x: dict) -> Tensor:
        kernel_grids = x["kernel_grids"]  # TODO: Dont work with the full grid
        (B, M, L_kernel) = kernel_grids.shape
        time_encodings = self.kernel_time_encoder(kernel_grids.reshape(B * M * L_kernel, -1))
        encodings_per_event_mark = self.evaluation_mark_encoder(torch.nn.functional.one_hot(torch.arange(self.max_num_marks, device=self.device), num_classes=self.max_num_marks).float())
        marks = torch.arange(M, device=self.device).repeat_interleave(L_kernel).repeat(B)
        mark_encodings = encodings_per_event_mark[marks]
        return (time_encodings + mark_encodings).view(B, M, L_kernel, -1)

    def _time_dependent_encoder(self, x: dict, sequence_encodings: Tensor, observations_padding_mask=None) -> Tensor:
        """
        Apply functional attention to obtain a time dependent summary of the paths.
        """
        trunk_net_encodings = self._trunk_net_encoder(x)  # [B, M, L_kernel, D]
        B, M, L_kernel, D = trunk_net_encodings.shape
        return self.time_dependent_functional_attention(trunk_net_encodings.view(B, M*L_kernel, -1), sequence_encodings, observations_padding_mask=observations_padding_mask).view(B, M, L_kernel, -1) # [B, M, L_kernel, D]

    def _static_encoder(self, x: dict, sequence_encodings: Tensor, observations_padding_mask=None) -> Tensor:
        """
        Apply functional attention to obtain a static summary of the paths.
        """
        (B, M, _) = x["kernel_grids"].shape
        learnable_queries = self.static_functional_attention_learnable_query(torch.nn.functional.one_hot(torch.arange(M, device=self.device), num_classes=self.max_num_marks).float()) # [M, D]
        # Stack B learnable_queries together to reshape to [B, M, D]
        learnable_queries = learnable_queries.repeat(B, 1).view(B, M, -1)
        return self.static_functional_attention(learnable_queries, sequence_encodings, observations_padding_mask=observations_padding_mask)  # [B, M, D]        

    def _kernel_value_decoder(self, time_dependent_path_summary: Tensor) -> Tensor:
        B, M, L_kernel, D_3 = time_dependent_path_summary.shape
        time_dependent_path_summary = time_dependent_path_summary.view(B * M * L_kernel, D_3)
        h = self.kernel_value_decoder(time_dependent_path_summary)
        return h.view(B, M, L_kernel)

    def _base_intensity_decoder(self, static_path_summary: Tensor) -> Tensor:
        B, M, D_2 = static_path_summary.shape
        h = self.base_intensity_decoder(static_path_summary)
        return h.view(-1, M)
    
    def _decay_parameter_decoder(self, static_path_summary: Tensor) -> Tensor:
        B, M, D_2 = static_path_summary.shape
        h = self.decay_parameter_decoder(static_path_summary)
        return h.view(-1, M)

    def __normalize_obs_grid(self, obs_grid: Tensor) -> tuple[Tensor, Tensor]:
        norm_constants = obs_grid.amax(dim=[-3, -2, -1])
        obs_grid_normalized = obs_grid / norm_constants.view(-1, 1, 1, 1)
        return norm_constants, obs_grid_normalized

    def loss(
        self,
        kernel_grid: Tensor,
        predicted_kernel_function: Tensor,
        predicted_base_intensity: Tensor,
        target_kernel_values: Tensor,
        target_base_intensity: Tensor,
        schedulers: dict = None,
        step: int = None,
    ) -> dict:
        B, M, L_kernel = predicted_kernel_function.shape
        assert target_kernel_values.shape == predicted_kernel_function.shape
        assert target_base_intensity.shape == predicted_base_intensity.shape

        kernel_rmse = torch.sqrt(torch.mean((predicted_kernel_function - target_kernel_values) ** 2))
        base_intensity_rmse = torch.sqrt(torch.mean((predicted_base_intensity - target_base_intensity) ** 2))

        loss = kernel_rmse + base_intensity_rmse

        return {
            "loss": loss,
            "kernel_rmse": kernel_rmse,
            "base_intensity_rmse": base_intensity_rmse,
        }

    def metric(self, y: Any, y_target: Any) -> Dict:
        return super().metric(y, y_target)
    
    def _generate_padding_mask(self, sequence_lengths, L):
        B, P = sequence_lengths.shape
        mask = torch.arange(L).expand(B, P, L).to(self.device) >= sequence_lengths.unsqueeze(-1)
        return mask


ModelFactory.register(FIMHawkesConfig.model_type, FIMHawkes)
AutoConfig.register(FIMHawkesConfig.model_type, FIMHawkesConfig)
AutoModel.register(FIMHawkesConfig, FIMHawkes)