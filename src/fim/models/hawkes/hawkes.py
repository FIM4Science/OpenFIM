import copy
import logging
from functools import partial
from typing import Any, Callable, Dict

import torch
from torch import Tensor
from transformers import AutoConfig, AutoModel, PretrainedConfig

from ...utils.helper import create_class_instance
from ...utils.interpolator import KernelInterpolator
from ...utils.logging import RankLoggerAdapter
from ..blocks import AModel, ModelFactory
from ..blocks.neural_operators import AttentionOperator
from .thinning import EventSampler


class FIMHawkesConfig(PretrainedConfig):
    model_type = "fimhawkes"

    def __init__(
        self,
        mark_encoder: dict = None,
        time_encoder: dict = None,
        delta_time_encoder: dict = None,
        kernel_time_encoder: dict = None,
        evaluation_mark_encoder: dict = None,
        context_ts_encoder: dict = None,
        inference_ts_encoder: dict = None,
        functional_attention: dict = None,
        intensity_decoder: dict = None,
        hidden_dim: int = None,
        max_num_marks: int = 1,
        is_bulk_model: bool = False,
        normalize_by_max_time: bool = True,
        thinning: dict = None,
        **kwargs,
    ):
        self.max_num_marks = max_num_marks
        self.is_bulk_model = is_bulk_model
        self.normalize_by_max_time = normalize_by_max_time
        self.mark_encoder = mark_encoder
        self.time_encoder = time_encoder
        self.delta_time_encoder = delta_time_encoder
        self.kernel_time_encoder = kernel_time_encoder
        self.evaluation_mark_encoder = evaluation_mark_encoder
        self.context_ts_encoder = context_ts_encoder
        self.inference_ts_encoder = inference_ts_encoder
        self.functional_attention = functional_attention
        self.intensity_decoder = intensity_decoder
        self.hidden_dim = hidden_dim
        self.thinning = thinning
        if "model_type" in kwargs:
            del kwargs["model_type"]
        super().__init__(model_type=self.model_type, **kwargs)


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
        kernel_value_var_decoder (nn.Module): The kernel value variance decoder.
        base_intensity_decoder (nn.Module): The base intensity decoder.
        base_intensity_var_decoder (nn.Module): The base intensity variance decoder.
        loss: TBD

    """

    config_class = FIMHawkesConfig

    def __init__(self, config: FIMHawkesConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.max_num_marks = config.max_num_marks
        self.is_bulk_model = config.is_bulk_model
        self.normalize_by_max_time = config.normalize_by_max_time
        if self.is_bulk_model and self.max_num_marks != 2:
            raise NotImplementedError("Bulk model only supports 2 marks.")
        self.logger = RankLoggerAdapter(logging.getLogger(self.__class__.__name__))
        self.__create_modules()
        self._register_cached_tensors()

        # self.gaussian_nll = nn.GaussianNLLLoss(full=True, reduction="none")
        # self.init_cross_entropy = nn.CrossEntropyLoss(reduction="none")

    def _register_cached_tensors(self):
        """Register pre-computed tensors as buffers for efficiency"""
        # Only cache the one-hot encodings (no gradients involved)
        mark_one_hot = torch.eye(self.max_num_marks)
        self.register_buffer("mark_one_hot", mark_one_hot)

    def __create_modules(self) -> None:
        mark_encoder = copy.deepcopy(self.config.mark_encoder)
        time_encoder = copy.deepcopy(self.config.time_encoder)
        delta_time_encoder = copy.deepcopy(self.config.delta_time_encoder)
        intensity_evaluation_time_encoder = copy.deepcopy(self.config.intensity_evaluation_time_encoder)
        evaluation_mark_encoder = copy.deepcopy(self.config.evaluation_mark_encoder)
        context_ts_encoder = copy.deepcopy(self.config.context_ts_encoder)
        inference_ts_encoder = copy.deepcopy(self.config.inference_ts_encoder)
        functional_attention = copy.deepcopy(self.config.functional_attention)
        intensity_decoder = copy.deepcopy(self.config.intensity_decoder)
        self.hidden_dim = self.config.hidden_dim
        self.normalize_times = self.config.normalize_times

        mark_encoder["in_features"] = self.max_num_marks
        self.mark_encoder = create_class_instance(mark_encoder.pop("name"), mark_encoder)
        time_encoder["in_features"] = 1
        self.time_encoder = create_class_instance(time_encoder.pop("name"), time_encoder)
        delta_time_encoder["in_features"] = 1
        self.delta_time_encoder = create_class_instance(delta_time_encoder.pop("name"), delta_time_encoder)
        intensity_evaluation_time_encoder["in_features"] = 1
        intensity_evaluation_time_encoder["out_features"] = self.hidden_dim
        self.intensity_evaluation_time_encoder = create_class_instance(
            intensity_evaluation_time_encoder.pop("name"), intensity_evaluation_time_encoder
        )
        evaluation_mark_encoder["in_features"] = self.max_num_marks
        evaluation_mark_encoder["out_features"] = self.hidden_dim
        self.evaluation_mark_encoder = create_class_instance(evaluation_mark_encoder.pop("name"), evaluation_mark_encoder)

        context_ts_encoder["encoder_layer"]["d_model"] = self.hidden_dim
        self.context_ts_encoder = create_class_instance(context_ts_encoder.pop("name"), context_ts_encoder)
        inference_ts_encoder["encoder_layer"]["d_model"] = self.hidden_dim
        self.inference_ts_encoder = create_class_instance(inference_ts_encoder.pop("name"), inference_ts_encoder)

        self.functional_attention = AttentionOperator(embed_dim=self.hidden_dim, out_features=self.hidden_dim, **functional_attention)

        intensity_decoder["in_features"] = self.hidden_dim
        intensity_decoder["out_features"] = 1
        self.intensity_decoder = create_class_instance(intensity_decoder.pop("name"), intensity_decoder)

        if self.config.thinning is not None:
            self.event_sampler = EventSampler(**self.config.thinning)
        else:
            self.event_sampler = EventSampler(num_sample=1, num_exp=500, over_sample_rate=5, num_samples_boundary=5, dtime_max=5)

    def forward(self, x: dict[str, Tensor], schedulers: dict = None, step: int = None) -> dict:
        """
        Forward pass for the model.

        Args:
            x (dict[str, Tensor]): A dictionary containing the input tensors:
                - "context_event_times": Tensor representing the event times. [B, P_context, L, 1]
                - "context_event_types": Tensor representing the event types. [B, P_context, L, 1]
                - "inference_event_times": Tensor representing the event times. [B, P_inference, L, 1]
                - "inference_event_types": Tensor representing the event types. [B, P_inference, L, 1]
                - "intensity_evaluation_times": Tensor representing the times at which to evaluate the intensity. [B, P_inference, L_inference]
                - Optional keys:
                    - "kernel_functions": Tensor representing the kernel functions using byte-encoded strings. [B, M, M]
                    - "base_intensity_functions": Tensor representing the base intensity functions using byte-encoded strings. [B, M]
                    - "context_seq_lengths": Tensor representing the sequence lengths (which we use for masking). [B, P_context]
                    - "inference_seq_lengths": Tensor representing the sequence lengths (which we use for masking). [B, P_inference]
            schedulers (dict, optional): A dictionary of schedulers for the training process. Default is None.
            step (int, optional): The current step in the training process. Default is None.
        Returns:
            dict: A dictionary containing the following keys:
                - "intensity": Tensor representing the predicted intensity. [B, P_inference, L_inference, M]
                - "losses" (optional): Tensor representing the calculated losses, if the required keys are present in `x`.
        """
        B, P_context, L = x["context_event_times"].shape[:3]
        P_inference = x["inference_event_times"].shape[1]

        if "context_seq_lengths" not in x:
            x["context_seq_lengths"] = torch.full((B, P_context), L, device=self.device)
        if "inference_seq_lengths" not in x:
            x["inference_seq_lengths"] = torch.full((B, P_inference), L, device=self.device)

        # Compute delta times
        self._compute_delta_times_inplace(x, "context")
        self._compute_delta_times_inplace(x, "inference")

        if self.normalize_times:
            norm_constants = self._normalize_input_times(x)

        sequence_encodings_context = self._encode_observations_optimized(x, "context")  # [B, P, L, D]
        sequence_encodings_inference = self._encode_observations_optimized(x, "inference")  # [B, P, L, D]

        # Concatenate sequence encodings REMOVE THIS
        sequence_encodings = torch.cat(
            [sequence_encodings_context, sequence_encodings_inference], dim=1
        )  # [B, P_context + P_inference, L, D]
        observations_padding_mask = None  # TODO: Create a proper mask here

        time_dependent_encodings = self._time_dependent_encoder_optimized(
            x, sequence_encodings, observations_padding_mask=observations_padding_mask
        )  # [B, M, P_inference, L_inference, D]

        predicted_intensity_values = self._intensity_decoder(time_dependent_encodings)  # [B, M, P_inference, L_inference]

        out = {
            "predicted_intensity_values": predicted_intensity_values,
        }

        if "intensity_evaluation_times" in x:
            out["losses"] = {}
            out["losses"]["loss"] = torch.mean(predicted_intensity_values**2)
            # out["losses"] = self.loss(
            #     out["predicted_intensity_values"],
            #     x["intensity_evaluation_times"],
            #     schedulers,
            #     step,
            # )

        if self.normalize_times:
            self._denormalize_output(x, out, norm_constants)

        return out

    def _encode_observations(self, x: dict) -> Tensor:
        obs_grid_normalized = x["event_times"]

        encodings_per_event_mark = self.mark_encoder(
            torch.nn.functional.one_hot(torch.arange(self.max_num_marks, device=self.device), num_classes=self.max_num_marks).float()
        )
        B, P, L = obs_grid_normalized.shape[:3]

        time_enc = self.time_encoder(obs_grid_normalized)
        delta_time_enc = self.delta_time_encoder(x["delta_times"])
        # Select encoding from encodings_per_event_mark from event_types
        state_enc = encodings_per_event_mark[x["event_types"].reshape(-1).int()].reshape(B, P, L, -1)
        path = time_enc + delta_time_enc + state_enc
        mask = torch.triu(torch.ones(L, L), diagonal=1).bool().to(self.device)
        mask = mask.repeat(B, P, 1, 1)

        positions = torch.arange(L, device=self.device).unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1, 1, L, 1)

        # if self.training:
        # Expand seq_lengths to (B, P, 1, 1) and compare with positions to create padding mask
        padding_mask = positions >= x["seq_lengths"].unsqueeze(-1).unsqueeze(-1)  # (B, P, L, 1)
        padding_mask = padding_mask.expand(-1, -1, -1, L)  # (B, P, L, L)

        # Combine causal mask with padding mask
        mask = mask | padding_mask
        padding_mask = None

        h = self.ts_encoder(path.view(B * P, L, -1), mask=mask.view(B * P, L, L), is_causal=True)

        return h.view(B, P, L, -1)

    def _trunk_net_encoder(self, x: dict) -> Tensor:
        kernel_grids = x["kernel_grids"]  # TODO: Dont work with the full grid
        (B, M, L_kernel) = kernel_grids.shape
        time_encodings = self.kernel_time_encoder(kernel_grids.reshape(B * M * L_kernel, -1))
        encodings_per_event_mark = self.evaluation_mark_encoder(
            torch.nn.functional.one_hot(torch.arange(self.max_num_marks, device=self.device), num_classes=self.max_num_marks).float()
        )
        marks = torch.arange(M, device=self.device).repeat_interleave(L_kernel).repeat(B)
        mark_encodings = encodings_per_event_mark[marks]
        return (time_encodings + mark_encodings).view(B, M, L_kernel, -1)

    def _time_dependent_encoder(self, x: dict, sequence_encodings: Tensor, observations_padding_mask=None) -> Tensor:
        """
        Apply functional attention to obtain a time dependent summary of the paths.
        """
        trunk_net_encodings = self._trunk_net_encoder(x)  # [B, M, L_kernel, D]
        B, M, L_kernel, D = trunk_net_encodings.shape
        return self.time_dependent_functional_attention(
            trunk_net_encodings.view(B, M * L_kernel, -1), sequence_encodings, observations_padding_mask=observations_padding_mask
        ).view(B, M, L_kernel, -1)  # [B, M, L_kernel, D]

    def _static_encoder(self, x: dict, sequence_encodings: Tensor, observations_padding_mask=None) -> Tensor:
        """
        Apply functional attention to obtain a static summary of the paths.
        """
        (B, M, _) = x["kernel_grids"].shape
        learnable_queries = self.static_functional_attention_learnable_query(
            torch.nn.functional.one_hot(torch.arange(M, device=self.device), num_classes=self.max_num_marks).float()
        )  # [M, D]
        # Stack B learnable_queries together to reshape to [B, M, D]
        learnable_queries = learnable_queries.repeat(B, 1).view(B, M, -1)
        return self.static_functional_attention(
            learnable_queries, sequence_encodings, observations_padding_mask=observations_padding_mask
        )  # [B, M, D]

    def _intensity_decoder(self, time_dependent_path_summary: Tensor) -> Tensor:
        B, M, P_inference, L_inference, D = time_dependent_path_summary.shape
        time_dependent_path_summary = time_dependent_path_summary.view(B * M * P_inference * L_inference, D)
        h = self.intensity_decoder(time_dependent_path_summary)
        return h.view(B, M, P_inference, L_inference)

    def _normalize_input_times(self, x: dict) -> dict:
        """
        Normalize the input times either by the maximum time in the context sequences or by the maximum time in the delta times.
        """
        if self.normalize_by_max_time:
            batch_indices = (
                torch.arange(x["context_event_times"].size(0), device=x["context_event_times"].device)
                .view(-1, 1)
                .expand(-1, x["context_event_times"].size(1))
            )
            path_indices = (
                torch.arange(x["context_event_times"].size(1), device=x["context_event_times"].device)
                .view(1, -1)
                .expand(x["context_event_times"].size(0), -1)
            )
            max_times = x["context_event_times"][batch_indices, path_indices, x["context_seq_lengths"] - 1]
            norm_constants = max_times.amax(dim=[1, 2])
        else:
            masked_delta_times = x["context_delta_times"].clone()
            B, P, L, _ = masked_delta_times.shape

            # Remove last dimension if it's size 1
            masked_delta_times = masked_delta_times.squeeze(-1)  # Now shape is (B, P, L)

            # Create positions tensor
            positions = torch.arange(L).view(1, 1, L).to(masked_delta_times.device)  # Shape: (1, 1, L)

            # Expand seq_lengths to match dimensions
            seq_lengths_expanded = x["context_seq_lengths"].unsqueeze(2)  # Shape: (B, P, 1)

            # Create mask for invalid positions
            mask = positions >= seq_lengths_expanded  # Shape: (B, P, L)

            # Apply mask to set invalid positions to -inf
            masked_delta_times[mask] = float("-inf")

            # Compute the maximum over the sequence length dimension
            norm_constants = masked_delta_times.amax(dim=[1, 2])
        x["context_event_times"] = x["context_event_times"] / norm_constants.view(-1, 1, 1, 1)
        x["context_delta_times"] = x["context_delta_times"] / norm_constants.view(-1, 1, 1, 1)
        x["inference_event_times"] = x["inference_event_times"] / norm_constants.view(-1, 1, 1, 1)
        x["inference_delta_times"] = x["inference_delta_times"] / norm_constants.view(-1, 1, 1, 1)
        x["intensity_evaluation_times"] = x["intensity_evaluation_times"] / norm_constants.view(-1, 1, 1)
        if "kernel_functions" in x:
            x["kernel_functions"] = x["kernel_functions"] * norm_constants.view(-1, 1)
        if "base_intensity_functions" in x:
            x["base_intensity_functions"] = x["base_intensity_functions"] * norm_constants.view(-1, 1, 1)

        return norm_constants

    def _compute_delta_times_inplace(self, x: dict, type="context"):
        """Compute delta times more efficiently"""
        B, P, L = x[f"{type}_event_times"].shape[:3]
        # Pre-allocate tensor with zeros for the first event
        delta_times = torch.zeros(B, P, L, 1, device=x[f"{type}_event_times"].device, dtype=x[f"{type}_event_times"].dtype)
        delta_times[:, :, 1:] = x[f"{type}_event_times"][:, :, 1:] - x[f"{type}_event_times"][:, :, :-1]
        x[f"{type}_delta_times"] = delta_times

    def _encode_observations_optimized(self, x: dict, type="context") -> Tensor:
        """Optimized observation encoding using cached one-hot encodings"""
        if type == "context":
            obs_grid_normalized = x[f"{type}_event_times"]
        elif type == "inference":
            obs_grid_normalized = x[f"{type}_event_times"]
        else:
            raise ValueError(f"Invalid type: {type}")
        B, P, L = obs_grid_normalized.shape[:3]

        time_enc = self.time_encoder(obs_grid_normalized)
        delta_time_enc = self.delta_time_encoder(x[f"{type}_delta_times"])

        # More efficient mark encoding using cached one-hot matrix
        event_types_flat = x[f"{type}_event_types"].reshape(-1).long()
        # Use cached one-hot matrix instead of computing it each time
        one_hot_marks = self.mark_one_hot[event_types_flat]  # [B*P*L, max_num_marks]
        mark_encodings = self.mark_encoder(one_hot_marks)  # [B*P*L, hidden_dim]
        state_enc = mark_encodings.reshape(B, P, L, -1)

        # Fused addition
        path = time_enc + delta_time_enc + state_enc

        # Use original mask creation (exactly as in original method)
        mask = torch.triu(torch.ones(L, L), diagonal=1).bool().to(self.device)
        mask = mask.repeat(B, P, 1, 1)

        positions = torch.arange(L, device=self.device).unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1, 1, L, 1)
        padding_mask = positions >= x[f"{type}_seq_lengths"].unsqueeze(-1).unsqueeze(-1)  # (B, P, L, 1)
        padding_mask = padding_mask.expand(-1, -1, -1, L)  # (B, P, L, L)
        mask = mask | padding_mask
        padding_mask = None

        if type == "context":
            h = self.context_ts_encoder(path.view(B * P, L, -1), mask=mask.view(B * P, L, L), is_causal=True)
        elif type == "inference":
            h = self.inference_ts_encoder(path.view(B * P, L, -1), mask=mask.view(B * P, L, L), is_causal=True)
        else:
            raise ValueError(f"Invalid type: {type}")

        return h.view(B, P, L, -1)

    def _trunk_net_encoder_optimized(self, x: dict) -> Tensor:
        """Optimized trunk net encoding using cached one-hot encodings"""
        intensity_evaluation_times = x["intensity_evaluation_times"]
        B, P_inference, L_inference = intensity_evaluation_times.shape

        # Expand to include marks dimension: [B, M, P_inference, L_inference]
        M = self.max_num_marks
        intensity_evaluation_times_expanded = intensity_evaluation_times.unsqueeze(1).expand(B, M, P_inference, L_inference)

        # More efficient reshaping and encoding
        time_encodings = self.intensity_evaluation_time_encoder(intensity_evaluation_times_expanded.reshape(-1, 1)).reshape(
            B, M, P_inference, L_inference, -1
        )

        # Use cached one-hot encodings for evaluation marks
        marks = torch.arange(M, device=intensity_evaluation_times.device)
        one_hot_eval_marks = self.mark_one_hot[marks]  # [M, max_num_marks]
        eval_mark_encodings = self.evaluation_mark_encoder(one_hot_eval_marks)  # [M, hidden_dim]
        mark_encodings = (
            eval_mark_encodings.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(B, -1, P_inference, L_inference, -1)
        )  # [B, M, P_inference, L_inference, D]

        return time_encodings + mark_encodings

    def _time_dependent_encoder_optimized(self, x: dict, sequence_encodings: Tensor, observations_padding_mask=None) -> Tensor:
        """Optimized time dependent encoder"""
        trunk_net_encodings = self._trunk_net_encoder_optimized(x)  # [B, M, P_inference, L_inference, D]
        B, M, P_inference, L_inference, D = trunk_net_encodings.shape

        # Reshape for attention: [B, M * P_inference * L_inference, D]
        trunk_reshaped = trunk_net_encodings.reshape(B, M * P_inference * L_inference, D)

        # Apply functional attention
        result = self.functional_attention(trunk_reshaped, sequence_encodings, observations_padding_mask=observations_padding_mask)

        # Reshape back to [B, M, P_inference, L_inference, D]
        return result.reshape(B, M, P_inference, L_inference, D)

    def _denormalize_output(self, x: dict, out: dict, norm_constants: Tensor) -> None:
        out["predicted_intensity_values"] = out["predicted_intensity_values"] / norm_constants.view(-1, 1, 1, 1)
        if "log_predicted_intensity_values_var" in out:
            out["log_predicted_intensity_values_var"] = out["log_predicted_intensity_values_var"] - torch.log(norm_constants).view(
                -1, 1, 1, 1
            )
        x["context_event_times"] = x["context_event_times"] * norm_constants.view(-1, 1, 1, 1)
        x["context_delta_times"] = x["context_delta_times"] * norm_constants.view(-1, 1, 1, 1)
        x["inference_event_times"] = x["inference_event_times"] * norm_constants.view(-1, 1, 1, 1)
        x["inference_delta_times"] = x["inference_delta_times"] * norm_constants.view(-1, 1, 1, 1)
        x["intensity_evaluation_times"] = x["intensity_evaluation_times"] * norm_constants.view(-1, 1, 1)
        if "kernel_functions" in x:
            x["kernel_functions"] = x["kernel_functions"] * norm_constants.view(-1, 1)
        if "base_intensity_functions" in x:
            x["base_intensity_functions"] = x["base_intensity_functions"] * norm_constants.view(-1, 1, 1)

    def loss(
        self,
        predicted_kernel_function: Tensor,
        log_predicted_kernel_var: Tensor,
        predicted_base_intensity: Tensor,
        target_kernel_values: Tensor,
        target_base_intensity: Tensor,
        kernel_grids: Tensor,
        schedulers: dict = None,
        step: int = None,
    ) -> dict:
        B, M, L_kernel = predicted_kernel_function.shape
        assert target_kernel_values.shape == predicted_kernel_function.shape
        assert target_base_intensity.shape == predicted_base_intensity.shape
        U = log_predicted_kernel_var

        # First perform the RMSE per mark
        kernel_rmse = torch.sqrt(torch.mean((predicted_kernel_function - target_kernel_values) ** 2, dim=-1))
        base_intensity_rmse = torch.sqrt(torch.mean((predicted_base_intensity - target_base_intensity) ** 2, dim=-1))

        # Then compute the mean over all marks
        kernel_rmse = torch.mean(kernel_rmse)
        base_intensity_rmse = torch.mean(base_intensity_rmse)

        kernel_loss = (torch.exp(-U) * (predicted_kernel_function - target_kernel_values) ** 2 + U).mean() / 2
        base_intensity_loss = base_intensity_rmse**2

        # loss = kernel_loss + base_intensity_loss
        loss = kernel_rmse + base_intensity_rmse

        # # With a 1% probability plot the kernel function and the ground truth and add the RMSE as a title to the plot
        # if torch.rand(1) < 0.01:
        #     import matplotlib.pyplot as plt

        #     B, M, T = predicted_kernel_function.shape
        #     predicted_kernel_function_np = predicted_kernel_function.clone().detach().cpu().float().numpy()
        #     ground_truth_kernel_function_np = target_kernel_values.clone().detach().cpu().float().numpy()
        #     # Define scaling factors
        #     width_per_subplot = 3
        #     height_per_subplot = 3
        #     figsize = (width_per_subplot * M, height_per_subplot * B)
        #     fig, axs = plt.subplots(B, M, figsize=figsize, squeeze=False)
        #     kernel_rmse_np = torch.sqrt(
        #         torch.mean(torch.tensor((predicted_kernel_function_np - ground_truth_kernel_function_np)) ** 2, dim=-1)
        #     )
        #     kernel_rmse_np = torch.mean(kernel_rmse_np)
        #     for b in range(B):
        #         for m in range(M):
        #             axs[b, m].scatter(
        #                 kernel_grids[b, m].clone().detach().cpu().float().numpy(), predicted_kernel_function_np[b, m], label="Model"
        #             )
        #             axs[b, m].scatter(
        #                 kernel_grids[b, m].clone().detach().cpu().float().numpy(),
        #                 ground_truth_kernel_function_np[b, m],
        #                 label="Ground Truth",
        #             )
        #             axs[b, m].legend()
        #             axs[b, m].tick_params(axis="both", which="major", labelsize=8)
        #     plt.tight_layout()
        #     plt.savefig("foo.png", dpi=300)
        #     plt.close()

        return {
            "loss": loss,
            "kernel_loss": kernel_loss,
            "base_intensity_loss": base_intensity_loss,
            "kernel_rmse": kernel_rmse,
            "base_intensity_rmse": base_intensity_rmse,
        }

    def metric(self, y: Any, y_target: Any) -> Dict:
        return super().metric(y, y_target)

    def _generate_padding_mask(self, sequence_lengths, L):
        B, P = sequence_lengths.shape
        mask = torch.arange(L).expand(B, P, L).to(self.device) >= sequence_lengths.unsqueeze(-1)
        return mask

    def predict_one_step_at_every_event(self, batch: dict, intensity_fn: Callable):
        """One-step prediction for every event in the sequence.

        Code taken from https://github.com/ant-research/EasyTemporalPointProcess/blob/main/easy_tpp/model/torch_model/torch_basemodel.py
        Args:
            batch (dict): A dictionary containing the following keys:
                - "event_times": Tensor representing the event times. [B, L]
                - "delta_times": Tensor representing the delta times. [B, L]
                - "event_types": Tensor representing the event types. [B, L]
                - "seq_lengths": Tensor representing the sequence lengths (which we use for masking). [B]

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """
        time_seq = batch["event_times"]
        time_delta_seq = batch["delta_times"]
        event_seq = batch["event_types"]

        # remove the last event, as the prediction based on the last event has no label
        # note: the first dts is 0
        # [batch_size, seq_len]
        time_seq, time_delta_seq, event_seq = time_seq[:, :-1], time_delta_seq[:, :-1], event_seq[:, :-1]

        # [batch_size, seq_len]
        # dtime_boundary = torch.max(time_delta_seq * self.event_sampler.dtime_max, time_delta_seq + self.event_sampler.dtime_max)

        # [batch_size, seq_len, num_sample]
        accepted_dtimes, weights = self.event_sampler.draw_next_time_one_step(
            time_seq, time_delta_seq, event_seq, intensity_fn, compute_last_step_only=False
        )

        # We should condition on each accepted time to sample event mark, but not conditioned on the expected event time.
        # 1. Use all accepted_dtimes to get intensity.
        # [batch_size, seq_len, num_sample, num_marks]
        intensities_at_times = intensity_fn(accepted_dtimes, time_seq)

        # 2. Normalize the intensity over last dim and then compute the weighted sum over the `num_sample` dimension.
        # Each of the last dimension is a categorical distribution over all marks.
        # [batch_size, seq_len, num_sample, num_marks]
        intensities_normalized = intensities_at_times / intensities_at_times.sum(dim=-1, keepdim=True)

        # 3. Compute weighted sum of distributions and then take argmax.
        # [batch_size, seq_len, num_marks]
        intensities_weighted = torch.einsum("...s,...sm->...m", weights, intensities_normalized)

        # [batch_size, seq_len, num_marks]
        types_pred = torch.argmax(intensities_weighted, dim=-1)

        # [batch_size, seq_len]
        dtimes_pred = torch.sum(accepted_dtimes * weights, dim=-1)  # compute the expected next event time
        return {
            "predicted_event_times": dtimes_pred,
            "predicted_event_types": types_pred,
            "predicted_intensities": intensities_at_times,
            "predicted_dtimes": accepted_dtimes,
        }

    @staticmethod
    def intentsity(
        t: Tensor,
        time_seqs: Tensor,
        kernel: KernelInterpolator,
        base_intensity: Tensor,
    ):
        """
        Calculate the intensity function at time t given the history of events.

        Args:
            t (Tensor): The time at which to compute the intensity. Shape: [B, L, N].
            time_seqs (Tensor): The sequence of event times. Shape: [B, L, 1].
            kernel (KernelInterpolator): The kernel function to use for computing the intensity.
            base_intensity (Tensor): The base intensity of the process. Shape: [M].

        Note:
            B is the number of processes, L is the number of events, N is the number of samples, M is the number of differentmarks.

        Returns:
            Tensor: The intensity at time t. Shape: [B, L, N, M].
        """
        dtime_sample = t.unsqueeze(-2) - time_seqs.unsqueeze(1).unsqueeze(-1)
        mask = (dtime_sample >= 0).float()
        dtime_sample = dtime_sample * mask
        B, S, L, N = dtime_sample.shape
        kernel_evaluations = kernel(dtime_sample.reshape(B, S, L * N))
        kernel_evaluations = kernel_evaluations.reshape(B, -1, S, L, N) * mask.reshape(B, -1, S, L, N)
        # [batch_size, num_marks, seq_len, num_samples]
        intensities = base_intensity + kernel_evaluations.sum(dim=-2)
        # [batch_size, seq_len, num_samples, num_marks]
        return torch.nn.functional.relu(intensities).permute(0, 2, 3, 1)

    def compute_intensities_at_sample_times(
        self, sample_dtimes, time_seqs, kernel_interpolator: KernelInterpolator, base_intensity: Tensor
    ):
        """Compute the intensity at sampled times, not only event times.

        Args:
            time_seq (tensor): [B, P, L], times seqs.
            time_delta_seq (tensor): [B, P, L], time delta seqs.
            event_seq (tensor): [B, P, L], event type seqs.
            sample_dtimes (tensor): [B, P, N], sampled inter-event timestamps.

        Returns:
            tensor: [batch_size, num_times, num_mc_sample, num_event_types],
                    intensity at each timestamp for each event type.
        """
        intensity_fn = partial(FIMHawkes.intentsity, kernel=kernel_interpolator, base_intensity=base_intensity)
        return intensity_fn(sample_dtimes, time_seqs)


ModelFactory.register(FIMHawkesConfig.model_type, FIMHawkes)
AutoConfig.register(FIMHawkesConfig.model_type, FIMHawkesConfig)
AutoModel.register(FIMHawkesConfig, FIMHawkes)
