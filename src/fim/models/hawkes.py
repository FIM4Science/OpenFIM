import copy
from typing import Any, Dict

import torch
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel, PretrainedConfig

from ..utils.helper import create_class_instance
from .blocks import AModel, ModelFactory, RNNEncoder, TransformerEncoder
from .blocks.positional_encodings import SineTimeEncoding
from .utils import create_matrix_from_off_diagonal, create_padding_mask, get_off_diagonal_elements
from fim.models.blocks import MultiHeadLearnableQueryAttention


class FIMHawkesConfig(PretrainedConfig):
    model_type = "fimhawkes"

    def __init__(
        self,
        ts_encoder: dict = None,
        trunk_net: dict = None,
        Omega_1_encoder: dict = None,
        Omega_2_encoder: dict = None,
        Omega_3_encoder: dict = None,
        Omega_4_encoder: dict = None,
        kernel_value_decoder: dict = None,
        kernel_parameter_decoder: dict = None,
        num_marks: int = 1,
        time_encodings: dict = None,
        event_type_embedding: dict = None,
        **kwargs,
    ):
        self.num_marks = num_marks
        self.ts_encoder = ts_encoder
        self.time_encodings = time_encodings
        self.event_type_embedding = event_type_embedding
        self.trunk_net = trunk_net
        self.Omega_1_encoder = Omega_1_encoder
        self.Omega_2_encoder = Omega_2_encoder
        self.Omega_3_encoder = Omega_3_encoder
        self.Omega_4_encoder = Omega_4_encoder
        self.kernel_value_decoder = kernel_value_decoder
        self.kernel_parameter_decoder = kernel_parameter_decoder

        super().__init__(**kwargs)

class FIMHawkes(AModel):
    """
    FIMHawkes: A Neural Recognition Model for Zero-Shot Inference of Hawkes Processes

    Attributes:        
        num_marks (int): Number of marks in the Hawkes process.        
        ts_encoder (dict | TransformerEncoder): Time series encoder.
        time_encodings (dict | SineTimeEncoding): Time encodings.
        event_type_embedding (dict | nn.Module): Event type embedding.
        trunk_net (dict | nn.Module): Trunk network.
        Omega_1_encoder (dict | nn.Module): Encoder for the time-dependent path embeddings.
        Omega_2_encoder (dict | nn.Module): Encoder for the static path embeddings.
        Omega_3_encoder (dict | nn.Module): Encoder for the time-dependent path summary.
        Omega_4_encoder (dict | nn.Module): Encoder for the static path summary.
        kernel_value_decoder (dict | nn.Module): Decoder for the kernel value.
        kernel_parameter_decoder (dict | nn.Module): Decoder for the kernel parameters.
        loss: TBD

    Methods:
        forward(x: dict[str, Tensor], schedulers: dict = None, step: int = None) -> dict:
            Forward pass of the model.
        __decode(tuple[t: Float, h: Tensor]) -> Float:
            Decode the hidden representation to obtain the kernel evaluation.
        __encode(x: dict) -> Tensor:
            Encode the input observations to obtain the hidden representation. x denotes the mini batch.
        __normalize_obs_grid(obs_grid: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
            Normalize the observation grid.
        loss(pred_im: Tensor, pred_logvar_im: Tensor, pred_init_cond: Tensor, target_im: Tensor, target_init_cond: Tensor, adjaceny_matrix: Tensor, normalization_constants: Tensor, schedulers: dict = None, step: int = None) -> dict:
            Compute the loss for the model.
        new_stats() -> dict:
            Initialize new statistics.
        metric(y: Any, y_target: Any) -> Dict:
            Compute the metric for the model.
    """
    config_class = FIMHawkesConfig
    
    def __init__(self, config: FIMHawkesConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.num_marks = config.num_marks

        self.__create_modules()

        # self.gaussian_nll = nn.GaussianNLLLoss(full=True, reduction="none")
        # self.init_cross_entropy = nn.CrossEntropyLoss(reduction="none")
        
        assert isinstance(self.Omega_1_encoder, nn.MultiheadAttention), "Omega_1_encoder must be an instance of nn.MultiheadAttention"
        assert isinstance(self.Omega_2_encoder, MultiHeadLearnableQueryAttention), "Omega_2_encoder must be an instance of MultiHeadLearnableQueryAttention"
        assert isinstance(self.Omega_3_encoder, MultiHeadLearnableQueryAttention), "Omega_3_encoder must be an instance of MultiHeadLearnableQueryAttention"
        assert isinstance(self.Omega_4_encoder, MultiHeadLearnableQueryAttention), "Omega_4_encoder must be an instance of MultiHeadLearnableQueryAttention"

    def __create_modules(self) -> None:
        ts_encoder = copy.deepcopy(self.config.ts_encoder)
        time_encodings = copy.deepcopy(self.config.time_encodings)
        event_type_embedding = copy.deepcopy(self.config.event_type_embedding)
        trunk_net = copy.deepcopy(self.config.trunk_net)
        Omega_1_encoder = copy.deepcopy(self.config.Omega_1_encoder)
        Omega_2_encoder = copy.deepcopy(self.config.Omega_2_encoder)
        Omega_3_encoder = copy.deepcopy(self.config.Omega_3_encoder)
        Omega_4_encoder = copy.deepcopy(self.config.Omega_4_encoder)
        kernel_value_decoder = copy.deepcopy(self.config.kernel_value_decoder)
        kernel_parameter_decoder = copy.deepcopy(self.config.kernel_parameter_decoder)
        
        self.time_encodings = create_class_instance(time_encodings.pop("name"), time_encodings)
        
        self.event_type_embedding = create_class_instance(event_type_embedding.pop("name"), event_type_embedding)
        
        ts_encoder["in_features"] = self.time_encodings.out_features + self.event_type_embedding.out_features
        self.ts_encoder = create_class_instance(ts_encoder.pop("name"), ts_encoder)
        
        trunk_net["in_features"] = self.time_encodings.out_features
        self.trunk_net = create_class_instance(trunk_net.pop("name"), trunk_net)
        
        self.Omega_1_encoder = create_class_instance(Omega_1_encoder.pop("name"), Omega_1_encoder)
        
        self.Omega_2_encoder = create_class_instance(Omega_2_encoder.pop("name"), Omega_2_encoder)
        
        self.Omega_3_encoder = create_class_instance(Omega_3_encoder.pop("name"), Omega_3_encoder)
        
        self.Omega_4_encoder = create_class_instance(Omega_4_encoder.pop("name"), Omega_4_encoder)
        
        kernel_value_decoder["out_features"] = self.num_marks
        self.kernel_value_decoder = create_class_instance(kernel_value_decoder.pop("name"), kernel_value_decoder)
        
        kernel_parameter_decoder["out_features"] = 2*self.num_marks
        self.kernel_parameter_decoder = create_class_instance(kernel_parameter_decoder.pop("name"), kernel_parameter_decoder)
        

    def forward(self, x: dict[str, Tensor], schedulers: dict = None, step: int = None) -> dict:
        """
        Forward pass for the model.

        Args:
            x (dict[str, Tensor]): A dictionary containing the input tensors:
                - "event_times": Tensor representing the event times.
                - "event_marks": Tensor representing the event marks.
                - "kernel_grids": Tensor representing the times at which to evaluate the kernel.
                - Optional keys for loss calculation:
                    - "base_intensities": Tensor representing the ground truth base intensity.
                    - "kernel_evaluations": Tensor representing the ground truth kernel evaluation values.
                    - "mask_seq_lengths": Tensor representing the sequence lengths (which we use for masking).
            schedulers (dict, optional): A dictionary of schedulers for the training process. Default is None.
            step (int, optional): The current step in the training process. Default is None.
        Returns:
            dict: A dictionary containing the following keys:
                - "kernel_eval_values": Tensor representing the predicted kernel evaluation values.
                - "baseline_intensity": Tensor representing the predicted baseline intensity.
                - "losses" (optional): Tensor representing the calculated losses, if the required keys are present in `x`.
        """
        x["observation_values_one_hot"] = torch.nn.functional.one_hot(x["event_types"].long().squeeze(-1), num_classes=self.num_marks)
        
        obs_grid = x["event_times"]
        if "time_normalization_factors" not in x:
            norm_constants, obs_grid = self.__normalize_obs_grid(obs_grid)
            x["time_normalization_factors"] = norm_constants
            x["observation_grid_normalized"] = obs_grid
        else:
            norm_constants = x["time_normalization_factors"]
            x["observation_grid_normalized"] = obs_grid

        #FIXME: REMOVE THIS!
        x["kernel_grids"] = x["kernel_grids"][:, :, ::10]
        x["kernel_evaluations"] = x["kernel_evaluations"][:,:,::10]
        print("WARNING: Kernel grids and evaluations are truncated to 10")

        sequence_encodings = self.__encode_observations(x) # [B, P, L, D]
            
        trunk_net_encodings = self.__trunk_net_encoder(x) # [B, M, L_kernel, D]
        
        time_dependent_path_embeddings = self.__Omega_1_encoder(x, trunk_net_encodings, sequence_encodings) # [B, M, L_kernel, P, D_1]
        
        static_path_embeddings = self.__Omega_2_encoder(sequence_encodings) # [B, P, D_2]
        
        time_dependent_path_summary = self.__Omega_3_encoder(time_dependent_path_embeddings) # [B, M, L_kernel, D_3]
        
        static_path_summary = self.__Omega_4_encoder(static_path_embeddings) # [B, D_4]
        
        predicted_kernel_values = self.__kernel_value_decoder(time_dependent_path_summary).squeeze() # [B, M, L_kernel]
        
        predicted_kernel_decay_and_base_intensity = torch.exp(self.__kernel_parameter_decoder(static_path_summary)) # [B, M, 2]
        predicted_base_intensity = predicted_kernel_decay_and_base_intensity[:,:,0]
        predicted_kernel_decay = predicted_kernel_decay_and_base_intensity[:,:,1]
        
        out = {
            "predicted_kernel_values": predicted_kernel_values,
            "predicted_base_intensity": predicted_base_intensity,
            "predicted_kernel_decay": predicted_kernel_decay,
        }
        if "base_intensities" in x and "kernel_evaluations" in x:
            out["losses"] = self.loss(
                predicted_kernel_values,
                predicted_base_intensity,
                predicted_kernel_decay,
                x["kernel_evaluations"],
                x["base_intensities"],
                schedulers,
                step,
            )
        
        return out
        

    def __encode_observations(self, x: dict) -> Tensor:
        obs_grid_normalized = x["observation_grid_normalized"]
        obs_values_one_hot = x["observation_values_one_hot"]
        B, P, L = obs_grid_normalized.shape[:3]
        
        #FIXME: Do this inside the dataloader
        x["seq_lengths"] = torch.tensor([L] * B * P, device=self.device)
        x["seq_lengths"] = x["seq_lengths"].view(B, P)
        
        time_enc = self.time_encodings(obs_grid_normalized)
        state_enc = self.event_type_embedding(obs_values_one_hot)
        path = torch.cat([time_enc, state_enc], dim=-1)
        assert isinstance(self.ts_encoder, RNNEncoder)
        h = self.ts_encoder(path.view(B * P, L, -1), x["seq_lengths"].view(B * P))
        # last_observation = x["seq_lengths"].view(B * P) - 1
        # h = h[torch.arange(B * P), last_observation].view(B, P, -1)

        return h.view(B, P, L, -1)
    
    def __trunk_net_encoder(self, x: dict) -> Tensor:
        kernel_grids = x["kernel_grids"] #TODO: Dont work with the full grid
        (B, M, L_kernel) = kernel_grids.shape
        time_encodings = self.time_encodings(kernel_grids.reshape(B*M*L_kernel, -1))
        return self.trunk_net(time_encodings).view(B, M, L_kernel, -1)
    
    def __Omega_1_encoder(self, x: dict, trunk_net_encoding: Tensor, observation_encoding: Tensor) -> Tensor:
        """
        The time-dependent path embeddings with variable sequence lengths.
        """
        ### This function should be a batched implementation of this logic:
        # for i in range(B):
        #     for j in range(M):
        #         for k in range(L_kernel):
        #             for l in range(P):
        #                 seq_len = x["seq_lengths"][i, l]
        #                 # Create seq_len copies of trunk_net_encoding[i,j,k]
        #                 query = trunk_net_encoding[i,j,k].repeat(seq_len, 1)
        #                 self.Omega_1_encoder(query, observation_encoding[i,l,:seq_len], observation_encoding[i,l,:seq_len])
        # TODO: If there are any bugs, its likely due to this function because its a bit complex
        # TODO: I am also not sure if this function is optimized for memory usage
        assert trunk_net_encoding.shape[0] == observation_encoding.shape[0]
        assert trunk_net_encoding.shape[-1] == observation_encoding.shape[-1]
        B, M, L_kernel, D = trunk_net_encoding.shape
        B, P, L, D = observation_encoding.shape

        seq_lengths = x["seq_lengths"]  # Shape: (B, P)

        # Repeat seq_lengths for M and L_kernel
        seq_lengths = seq_lengths.unsqueeze(1).unsqueeze(2)  # Shape: (B, 1, 1, P)
        seq_lengths = seq_lengths.repeat(1, M, L_kernel, 1).view(B * M * L_kernel * P)  # Shape: (B*M*L_kernel*P)

        # Prepare queries
        queries = trunk_net_encoding.view(B * M * L_kernel, 1, D).expand(-1, P, -1)
        queries = queries.contiguous().view(B * M * L_kernel * P, 1, D)
        queries = queries.expand(-1, L, -1)  # Shape: (B*M*L_kernel*P, L, D)

        # Prepare keys and values
        keys = observation_encoding.unsqueeze(1).unsqueeze(2)  # Shape: (B,1,1,P,L,D)
        keys = keys.expand(B, M, L_kernel, P, L, D).contiguous().view(B * M * L_kernel * P, L, D)
        values = keys

        # Create key_padding_mask based on seq_lengths
        # key_padding_mask: (batch_size, seq_length)
        # Here, batch_size = B * M * L_kernel * P
        mask = torch.arange(L, device=seq_lengths.device).unsqueeze(0).expand(B * M * L_kernel * P, L) >= seq_lengths.unsqueeze(1)
        
        # Apply encoder with key_padding_mask
        encoder_output, _ = self.Omega_1_encoder(queries, keys, values, key_padding_mask=mask)
        # Select the last valid output based on seq_lengths
        # To avoid indexing errors, clamp seq_lengths to at least 1
        last_indices = torch.clamp(seq_lengths - 1, min=0)
        h = encoder_output[torch.arange(encoder_output.size(0)), last_indices]

        return h.view(B, M, L_kernel, P, -1)
    
    def __Omega_2_encoder(self, sequence_encodings: Tensor) -> Tensor:
        """
        The static path embeddings.
        """
        B, P, L, D = sequence_encodings.shape
        sequence_encodings = sequence_encodings.view(B*P, L, D)
        h = self.Omega_2_encoder(None, sequence_encodings, sequence_encodings)
        
        return h.view(B, P, -1)
    
    def __Omega_3_encoder(self, time_dependent_path_embeddings: Tensor) -> Tensor:
        """
        The time dependent path summary.
        """
        B, M, L_kernel, P, D_1 = time_dependent_path_embeddings.shape
        time_dependent_path_embeddings = time_dependent_path_embeddings.view(B*M*L_kernel, P, D_1)
        h = self.Omega_3_encoder(None, time_dependent_path_embeddings, time_dependent_path_embeddings)
        
        return h.view(B, M, L_kernel, -1)
    
    def __Omega_4_encoder(self, static_path_embeddings: Tensor) -> Tensor:
        """
        The static path summary.
        """
        return self.Omega_4_encoder(None, static_path_embeddings, static_path_embeddings)
    
    def __kernel_value_decoder(self, time_dependent_path_summary: Tensor) -> Tensor:
        B, M, L_kernel, D_3 = time_dependent_path_summary.shape
        time_dependent_path_summary = time_dependent_path_summary.view(B*M*L_kernel, D_3)
        h = self.kernel_value_decoder(time_dependent_path_summary)
        
        return h.view(B, M, L_kernel, -1)
    
    def __kernel_parameter_decoder(self, static_path_summary: Tensor) -> Tensor:
        h = self.kernel_parameter_decoder(static_path_summary)
        return h.view(-1, self.num_marks, 2)

    def __normalize_obs_grid(self, obs_grid: Tensor) -> tuple[Tensor, Tensor]:
        norm_constants = obs_grid.amax(dim=[-3, -2, -1])
        obs_grid_normalized = obs_grid / norm_constants.view(-1, 1, 1, 1)
        return norm_constants, obs_grid_normalized

    def loss(
        self,
        predicted_kernel_values: Tensor,
        predicted_base_intensity: Tensor,
        predicted_kernel_decay: Tensor,
        target_kernel_values: Tensor,
        target_base_intensity: Tensor,
        schedulers: dict = None,
        step: int = None,
    ) -> dict:
        predicted_kernel_function = predicted_kernel_values * torch.exp(-predicted_kernel_decay.unsqueeze(-1))
        
        kernel_rmse = torch.sqrt(torch.mean((predicted_kernel_function - target_kernel_values) ** 2))
        base_intensity_rmse = torch.sqrt(torch.mean((predicted_base_intensity - target_base_intensity) ** 2))
        
        # print("Prediction", predicted_kernel_function)  
        # print("Target", target_kernel_values)      
        
        loss_1 = kernel_rmse + base_intensity_rmse
        
        loss = loss_1

        return {
            "loss": loss,
            "loss_1": loss_1,
            "kernel_rmse": kernel_rmse,
            "base_intensity_rmse": base_intensity_rmse,
        }

    def metric(self, y: Any, y_target: Any) -> Dict:
        return super().metric(y, y_target)


ModelFactory.register(FIMHawkesConfig.model_type, FIMHawkes)
AutoConfig.register(FIMHawkesConfig.model_type, FIMHawkesConfig)
AutoModel.register(FIMHawkesConfig, FIMHawkes)
