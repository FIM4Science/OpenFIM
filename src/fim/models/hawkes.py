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
        
        trunk_net["in_features"] = self.ts_encoder.out_features
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
                - "kernel_eval_times": Tensor representing the times at which to evaluate the kernel.
                - Optional keys for loss calculation:
                    - "ground_truth_baseline_intensity": Tensor representing the ground truth baseline intensity.
                    - "ground_truth_kernel_eval_times": Tensor representing the ground truth kernel evaluation times.
                    - "ground_truth_kernel_eval_values": Tensor representing the ground truth kernel evaluation values.
                    - "mask_seq_lengths": Tensor representing the sequence lengths (which we use for masking).
            schedulers (dict, optional): A dictionary of schedulers for the training process. Default is None.
            step (int, optional): The current step in the training process. Default is None.
        Returns:
            dict: A dictionary containing the following keys:
                - "kernel_eval_values": Tensor representing the predicted kernel evaluation values.
                - "baseline_intensity": Tensor representing the predicted baseline intensity.
                - "losses" (optional): Tensor representing the calculated losses, if the required keys are present in `x`.
        """
        x["observation_values_one_hot"] = torch.nn.functional.one_hot(x["event_type_data"].long().squeeze(-1), num_classes=self.num_marks)
        
        breakpoint()
        if "time_normalization_factors" not in x:
            norm_constants, obs_grid = self.__normalize_obs_grid(obs_grid)
            x["time_normalization_factors"] = norm_constants
            x["observation_grid_normalized"] = obs_grid
        else:
            norm_constants = x["time_normalization_factors"]
            x["observation_grid_normalized"] = obs_grid

        sequence_encodings = self.__encode_observations(x)



        ##############

        h = self.__encode(x, obs_grid, obs_values_one_hot)

        pred_offdiag_im_mean_logvar, init_cond = self.__decode(h)

        pred_offdiag_im_mean, pred_offdiag_im_logvar = self.__denormalize_offdiag_mean_logstd(norm_constants, pred_offdiag_im_mean_logvar)

        out = {
            "im": create_matrix_from_off_diagonal(pred_offdiag_im_mean, self.n_states),
            "log_var_im": create_matrix_from_off_diagonal(pred_offdiag_im_logvar, self.n_states),
            "init_cond": init_cond,
        }
        if "intensity_matrices" in x and "initial_distributions" in x:
            out["losses"] = self.loss(
                pred_offdiag_im_mean, pred_offdiag_im_logvar, init_cond, x, norm_constants.view(-1, 1), schedulers, step
            )

        return out

    def __encode_observations(self, x: dict) -> Tensor:
        obs_grid_normalized = x["observation_grid_normalized"]
        obs_values_one_hot = x["observation_values_one_hot"]
        B, P, L = obs_grid_normalized.shape[:3]
        pos_enc = self.pos_encodings(obs_grid_normalized)
        path = torch.cat([pos_enc, obs_values_one_hot], dim=-1)
        if isinstance(self.ts_encoder, TransformerEncoder):
            padding_mask = create_padding_mask(x["seq_lengths"].view(B * P), L)
            padding_mask[:, 0] = True
            h = self.ts_encoder(path.view(B * P, L, -1), padding_mask)[:, 1, :].view(B, P, -1)
        elif isinstance(self.ts_encoder, RNNEncoder):
            h = self.ts_encoder(path.view(B * P, L, -1), x["seq_lengths"].view(B * P))

        return h
    
    def __trunk_net_encoder(self, location_times: Tensor) -> Tensor:
        if isinstance(self.ts_encoder, TransformerEncoder):
            location_times = self.time_encodings(location_times)
        return self.trunk_net(location_times)
    
    def __Omega_1_encoder(self, trunk_net_encoding: Tensor, observation_encoding: Tensor) -> Tensor:
        """
        The time-dependent path embeddings.
        """
        if isinstance(self.Omega_1_encoder, nn.MultiheadAttention):
            h = self.Omega_1_encoder(trunk_net_encoding, observation_encoding, observation_encoding)[0][:, -1]
        else:
            h = self.Omega_1_encoder(trunk_net_encoding, observation_encoding, observation_encoding)        
        return h
    
    def __Omega_2_encoder(self, observation_encoding: Tensor) -> Tensor:
        """
        The static path embeddings.
        """
        return self.Omega_2_encoder(None, observation_encoding, observation_encoding)
    
    def __Omega_3_encoder(self, time_dependent_path_embeddings: Tensor) -> Tensor:
        """
        The time dependent path summary.
        """
        return self.Omega_3_encoder(None, time_dependent_path_embeddings, time_dependent_path_embeddings)
    
    def __Omega_4_encoder(self, static_path_embeddings: Tensor) -> Tensor:
        """
        The static path summary.
        """
        return self.Omega_4_encoder(None, static_path_embeddings, static_path_embeddings)
    
    def __kernel_value_decoder(self, time_dependent_path_summary: Tensor) -> Tensor:
        return self.kernel_value_decoder(time_dependent_path_summary)
    
    def __kernel_parameter_decoder(self, static_path_summary: Tensor) -> Tensor:
        return self.kernel_parameter_decoder(static_path_summary)

    def __normalize_obs_grid(self, obs_grid: Tensor) -> tuple[Tensor, Tensor]:
        norm_constants = obs_grid.amax(dim=[-3, -2, -1])
        obs_grid_normalized = obs_grid / norm_constants.view(-1, 1, 1, 1)
        return norm_constants, obs_grid_normalized

    def loss(
        self,
        pred_im: Tensor,
        pred_logstd_im: Tensor,
        pred_init_cond: Tensor,
        target: dict,
        normalization_constants: Tensor,
        schedulers: dict = None,
        step: int = None,
    ) -> dict:
        target_im = target["intensity_matrices"]
        target_init_cond = target["initial_distributions"]
        adjaceny_matrix = target["adjacency_matrices"]
        target_mean = get_off_diagonal_elements(target_im)
        P = target["observation_grid"].shape[1]
        adjaceny_matrix = get_off_diagonal_elements(adjaceny_matrix)
        target_init_cond = torch.argmax(target_init_cond, dim=-1).long()
        pred_im_std = torch.exp(pred_logstd_im)
        loss_gauss = adjaceny_matrix * self.gaussian_nll(pred_im, target_mean, torch.pow(pred_im_std, 2))
        loss_gauss = loss_gauss.sum() / (adjaceny_matrix.sum() + 1e-8)
        loss_initial = self.init_cross_entropy(pred_init_cond, target_init_cond).mean()
        zero_entries = 1.0 - adjaceny_matrix
        loss_missing_link = normalization_constants * zero_entries * (torch.pow(pred_im, 2) + torch.pow(pred_im_std, 2))
        loss_missing_link = loss_missing_link.sum() / (zero_entries.sum() + 1e-8)
        rmse_loss = torch.sqrt(torch.mean((target_mean - pred_im) ** 2))

        gaus_cons = schedulers.get("gauss_nll")(step) if schedulers else torch.tensor(1.0)
        init_cons = schedulers.get("init_cross_entropy")(step) if schedulers else torch.tensor(1.0)
        missing_link_cons = schedulers.get("missing_link")(step) if schedulers else torch.tensor(1.0)
        gaus_cons = gaus_cons.to(self.device)
        init_cons = init_cons.to(self.device)
        missing_link_cons = missing_link_cons.to(self.device)

        loss = gaus_cons * loss_gauss + init_cons * loss_initial + missing_link_cons * loss_missing_link
        # loss = rmse_loss
        return {
            "loss": loss,
            "loss_gauss": loss_gauss,
            "loss_initial": loss_initial,
            "loss_missing_link": loss_missing_link,
            "rmse_loss": rmse_loss,
            "beta_gauss_nll": gaus_cons,
            "beta_init_cross_entropy": init_cons,
            "beta_missing_link": missing_link_cons,
            "number_of_paths": torch.tensor(P, device=self.device),
        }

    def metric(self, y: Any, y_target: Any) -> Dict:
        return super().metric(y, y_target)


ModelFactory.register(FIMHawkesConfig.model_type, FIMHawkes)
AutoConfig.register(FIMHawkesConfig.model_type, FIMHawkesConfig)
AutoModel.register(FIMHawkesConfig, FIMHawkes)
