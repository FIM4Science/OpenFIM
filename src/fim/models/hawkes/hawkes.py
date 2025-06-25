import copy
import logging
from typing import Any, Dict

import torch
from torch import Tensor
from transformers import AutoConfig, AutoModel, PretrainedConfig

from ...utils.helper import create_class_instance, decode_byte_string
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
            # out["losses"] = {}
            # out["losses"]["loss"] = torch.mean(predicted_intensity_values**2)
            out["losses"] = self.loss(
                out["predicted_intensity_values"],
                x["kernel_functions"],
                x["base_intensity_functions"],
                x["intensity_evaluation_times"],
                x["inference_event_times"],
                x["inference_event_types"],
                x["inference_seq_lengths"],
                norm_constants,
                schedulers,
                step,
            )

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

        past_event_shifted_intensity_evaluation_times = self._get_past_event_shifted_intensity_evaluation_times(
            intensity_evaluation_times, x["inference_event_times"]
        )

        # More efficient reshaping and encoding
        time_encodings = self.intensity_evaluation_time_encoder(past_event_shifted_intensity_evaluation_times.reshape(-1, 1)).reshape(
            B, self.max_num_marks, P_inference, L_inference, -1
        )

        # Use cached one-hot encodings for evaluation marks
        marks = torch.arange(self.max_num_marks, device=intensity_evaluation_times.device)
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

    def loss(
        self,
        predicted_intensity_values: Tensor,
        kernel_functions: Tensor,
        base_intensity_functions: Tensor,
        intensity_evaluation_times: Tensor,
        inference_event_times: Tensor,
        inference_event_types: Tensor,
        inference_seq_lengths: Tensor,
        norm_constants: Tensor,
        schedulers: dict = None,
        step: int = None,
    ) -> dict:
        kernel_functions_list, base_intensity_functions_list = self._decode_functions(kernel_functions, base_intensity_functions)

        target_intensity_values = self.compute_target_intensity_values(
            kernel_functions_list,
            base_intensity_functions_list,
            intensity_evaluation_times,
            inference_event_times,
            inference_event_types,
            inference_seq_lengths,
            norm_constants,
        )
        assert target_intensity_values.shape == predicted_intensity_values.shape

        pass

    def compute_target_intensity_values(
        self,
        kernel_functions_list: list,
        base_intensity_functions_list: list,
        intensity_evaluation_times: Tensor,
        inference_event_times: Tensor,
        inference_event_types: Tensor,
        inference_seq_lengths: Tensor,
        norm_constants: Tensor,
    ):
        """
        Compute the target intensity values based on the formula:
        λ_i(t) = max(0, μ_i(t) + Σ_{j=1..D} Σ_{k: t_jk < t} φ_ij(t - t_jk))

        This function calculates the ground-truth intensity for each mark `i` at each
        time `t` specified in `intensity_evaluation_times`, based on the historical
        events provided in `inference_event_times`.

        Args:
            kernel_functions_list (list): A nested list of shape [B, D, D] containing
                the callable kernel functions φ_ij.
            base_intensity_functions_list (list): A list of shape [B, D] containing
                the callable base intensity functions μ_i.
            intensity_evaluation_times (Tensor): The times `t` to evaluate the intensity at.
                Shape: [B, P, L_eval].
            inference_event_times (Tensor): The historical event timestamps t_jk.
                Shape: [B, P, L_hist, 1].
            inference_event_types (Tensor): The historical event types `j`.
                Shape: [B, P, L_hist, 1].
            inference_seq_lengths (Tensor): The actual length of each historical sequence.
                Shape: [B, P].
            norm_constants (Tensor): The normalization constants.
                Shape: [B, 1].

        Returns:
            Tensor: The computed target intensity values. Shape: [B, D, P, L_eval],
                    where D is the number of marks.
        """
        device = intensity_evaluation_times.device
        B, P, L_eval = intensity_evaluation_times.shape
        _, _, L_hist, _ = inference_event_times.shape
        D = self.max_num_marks

        # Ensure tensors are 3D for easier processing by removing the trailing dimension
        event_times = inference_event_times.squeeze(-1)  # Shape: [B, P, L_hist]
        event_types = inference_event_types.squeeze(-1)  # Shape: [B, P, L_hist]

        # The final output tensor, matching the model's prediction shape [B, M, P, L_eval]
        # where M (self.max_num_marks) is D.
        total_intensity = torch.zeros(B, D, P, L_eval, device=device)

        # Loop over batch items because kernel/base functions are heterogeneous Python objects.
        # The operations inside this loop are vectorized over paths (P) and time dimensions.
        for b in range(B):
            # Extract data for the current batch item for clarity
            eval_times_b = intensity_evaluation_times[b]  # [P, L_eval]
            hist_times_b = event_times[b]  # [P, L_hist]
            hist_types_b = event_types[b]  # [P, L_hist]
            seq_lengths_b = inference_seq_lengths[b]  # [P]

            # --- Calculate Summed Kernel Term ---
            # Shape: [P, L_eval, L_hist]. Element (p, l, k) is eval_times[p,l] - hist_times[p,k]
            delta_t_b = eval_times_b.unsqueeze(-1) - hist_times_b.unsqueeze(-2)

            # Mask for causality: t_jk < t  (historical event time < evaluation time).
            # Use a small epsilon for strict inequality to handle floating point issues.
            causality_mask_b = delta_t_b > 1e-9  # [P, L_eval, L_hist]

            # Mask for padding: only consider historical events within the actual sequence length.
            hist_indices = torch.arange(L_hist, device=device).view(1, L_hist)
            padding_mask_b = hist_indices < seq_lengths_b.unsqueeze(-1)  # [P, L_hist]

            # Combine causality and padding masks. Broadcast padding_mask from [P,1,L_hist] to [P,L_eval,L_hist].
            valid_history_mask_b = causality_mask_b & padding_mask_b.unsqueeze(1)

            # Initialize the intensity with the baseline values.
            for i in range(D):
                mu_func_b_i = base_intensity_functions_list[b][i]
                # Assuming mu_func can take a [P, L_eval] tensor and return a tensor of the same shape.
                total_intensity[b, i, :, :] = mu_func_b_i(eval_times_b)

            # Accumulate kernel influences by iterating over target (i) and source (j) marks.
            for i in range(D):
                for j in range(D):
                    phi_func_b_ij = kernel_functions_list[b][i][j]

                    # Mask for events of source type j. Broadcast from [P,1,L_hist] to [P,L_eval,L_hist].
                    source_mark_mask_b = (hist_types_b == j).unsqueeze(1)

                    # Final mask for this (i, j) pair's contribution.
                    final_mask_b = valid_history_mask_b & source_mark_mask_b

                    if not torch.any(final_mask_b):
                        continue

                    # Evaluate the kernel function only on the valid (t - t_jk) values.
                    delta_t_to_eval = delta_t_b[final_mask_b]
                    kernel_values = phi_func_b_ij(delta_t_to_eval)

                    # Place the computed kernel values back into a structured tensor and sum over the history axis.
                    kernel_contribution = torch.zeros_like(delta_t_b)
                    kernel_contribution[final_mask_b] = kernel_values

                    # Sum over the history dimension (k) to get the total influence on each evaluation time.
                    summed_kernel_values = kernel_contribution.sum(dim=-1)  # Shape: [P, L_eval]

                    # Add the summed influence to the total intensity for the target mark i.
                    total_intensity[b, i, :, :] += summed_kernel_values

        # Apply the non-negativity constraint from the max(0, ...) in the formula.
        target_intensity_values = torch.relu(total_intensity)

        return target_intensity_values * norm_constants.view(-1, 1, 1, 1)

    def _get_past_event_shifted_intensity_evaluation_times(self, intensity_evaluation_times: Tensor, inference_event_times: Tensor):
        """
        Computes the time elapsed since the last event for each evaluation time.

        This function implements the calculation of `t - T_k*(t)` from the paper, where `t` is an
        intensity evaluation time and `T_k*(t) = max_{t_k,j < t} t_k,j` is the time of the
        most recent event before `t`.

        - For each time `t` in `intensity_evaluation_times` and each sequence in the batch,
          it finds the time `t_last` of the latest event in `inference_event_times` such that
          `t_last < t`.
        - If no such event exists (i.e., `t` is before the first event), `t_last` is considered to be 0.
        - It then computes the difference `delta_t = t - t_last`.
        - The result is expanded to match the number of marks, as this time difference is a
          component of the query embedding for every possible mark.

        Args:
            intensity_evaluation_times (Tensor): Times to evaluate the intensity at.
                                                 Shape: [B, P_inference, L_inference].
            inference_event_times (Tensor): Historical event times for the inference sequences.
                                            Shape: [B, P_inference, L_hist, 1].

        Returns:
            Tensor: The time difference `t - t_last` for each evaluation time, expanded
                    for all mark types. Shape: [B, M, P_inference, L_inference], where M is
                    the maximum number of marks.
        """
        # Get dimensions. This function is a method of FIMHawkes, so it has access to self.
        B, P, L_eval = intensity_evaluation_times.shape
        M = self.max_num_marks

        # Squeeze the last dimension of event times for easier broadcasting.
        # Shape: [B, P, L_hist]
        hist_times = inference_event_times.squeeze(-1)

        # Expand dimensions for broadcasting comparison.
        # eval_times_expanded -> [B, P, L_eval, 1]
        # hist_times_expanded -> [B, P, 1, L_hist]
        eval_times_expanded = intensity_evaluation_times.unsqueeze(3)
        hist_times_expanded = hist_times.unsqueeze(2)

        # Find all historical events that occurred strictly before each evaluation time.
        # This corresponds to the definition T_k*(t) = max_{t_k,j < t} t_k,j.
        # Shape: [B, P, L_eval, L_hist]
        is_past_event = hist_times_expanded < eval_times_expanded

        # Mask the historical times, keeping only past events.
        # Non-past events are set to a very small number (-inf) so they are ignored by the max operation.
        masked_hist_times = torch.where(is_past_event, hist_times_expanded, -float("inf"))

        # Find the maximum time among past events for each evaluation time.
        # This computes T_k*(t) for each t.
        # Shape: [B, P, L_eval]
        last_event_time, _ = torch.max(masked_hist_times, dim=3)

        # Handle the case where no past events exist for an evaluation time `t`.
        # In this scenario, `torch.max` returns -inf. We replace it with 0,
        # effectively treating the process as starting at time 0.
        last_event_time = torch.nan_to_num(last_event_time, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute the delta time: t - T_k*(t)
        # Shape: [B, P, L_eval]
        delta_t = intensity_evaluation_times - last_event_time

        # The time delta is the same for all target marks `m`.
        # Expand the result to include the mark dimension `M` as expected by the calling function.
        # Shape: [B, M, P, L_eval]
        delta_t_expanded = delta_t.unsqueeze(1).expand(-1, M, -1, -1)

        return delta_t_expanded

    def _decode_functions(self, kernel_functions: Tensor, base_intensity_functions: Tensor):
        """
        Decode the kernel and base intensity functions into lists of functions using eval.

        Input:
            "kernel_functions": Tensor representing the kernel functions using byte-encoded strings. [B, M, M]
            "base_intensity_functions": Tensor representing the base intensity functions using byte-encoded strings. [B, M]
        Output:
            "kernel_functions": List of kernel functions. [B, M, M]
            "base_intensity_functions": List of base intensity functions. [B, M]
        """
        B, M, _, _ = kernel_functions.shape
        kernel_functions_list = []
        base_intensity_functions_list = []
        for b in range(B):
            kernel_functions_list.append([])
            base_intensity_functions_list.append([])
            for m in range(M):
                kernel_functions_list[b].append([])
                for m_prime in range(M):
                    kernel_functions_list[b][m].append(eval(decode_byte_string(kernel_functions[b, m, m_prime])))
                base_intensity_functions_list[b].append(eval(decode_byte_string(base_intensity_functions[b, m])))
        return kernel_functions_list, base_intensity_functions_list

    def metric(self, y: Any, y_target: Any) -> Dict:
        return super().metric(y, y_target)

    # def predict_one_step_at_every_event(self, batch: dict, intensity_fn: Callable):
    #     """One-step prediction for every event in the sequence.

    #     Code taken from https://github.com/ant-research/EasyTemporalPointProcess/blob/main/easy_tpp/model/torch_model/torch_basemodel.py
    #     Args:
    #         batch (dict): A dictionary containing the following keys:
    #             - "event_times": Tensor representing the event times. [B, L]
    #             - "delta_times": Tensor representing the delta times. [B, L]
    #             - "event_types": Tensor representing the event types. [B, L]
    #             - "seq_lengths": Tensor representing the sequence lengths (which we use for masking). [B]

    #     Returns:
    #         tuple: tensors of dtime and type prediction, [batch_size, seq_len].
    #     """
    #     time_seq = batch["event_times"]
    #     time_delta_seq = batch["delta_times"]
    #     event_seq = batch["event_types"]

    #     # remove the last event, as the prediction based on the last event has no label
    #     # note: the first dts is 0
    #     # [batch_size, seq_len]
    #     time_seq, time_delta_seq, event_seq = time_seq[:, :-1], time_delta_seq[:, :-1], event_seq[:, :-1]

    #     # [batch_size, seq_len]
    #     # dtime_boundary = torch.max(time_delta_seq * self.event_sampler.dtime_max, time_delta_seq + self.event_sampler.dtime_max)

    #     # [batch_size, seq_len, num_sample]
    #     accepted_dtimes, weights = self.event_sampler.draw_next_time_one_step(
    #         time_seq, time_delta_seq, event_seq, intensity_fn, compute_last_step_only=False
    #     )

    #     # We should condition on each accepted time to sample event mark, but not conditioned on the expected event time.
    #     # 1. Use all accepted_dtimes to get intensity.
    #     # [batch_size, seq_len, num_sample, num_marks]
    #     intensities_at_times = intensity_fn(accepted_dtimes, time_seq)

    #     # 2. Normalize the intensity over last dim and then compute the weighted sum over the `num_sample` dimension.
    #     # Each of the last dimension is a categorical distribution over all marks.
    #     # [batch_size, seq_len, num_sample, num_marks]
    #     intensities_normalized = intensities_at_times / intensities_at_times.sum(dim=-1, keepdim=True)

    #     # 3. Compute weighted sum of distributions and then take argmax.
    #     # [batch_size, seq_len, num_marks]
    #     intensities_weighted = torch.einsum("...s,...sm->...m", weights, intensities_normalized)

    #     # [batch_size, seq_len, num_marks]
    #     types_pred = torch.argmax(intensities_weighted, dim=-1)

    #     # [batch_size, seq_len]
    #     dtimes_pred = torch.sum(accepted_dtimes * weights, dim=-1)  # compute the expected next event time
    #     return {
    #         "predicted_event_times": dtimes_pred,
    #         "predicted_event_types": types_pred,
    #         "predicted_intensities": intensities_at_times,
    #         "predicted_dtimes": accepted_dtimes,
    #     }


ModelFactory.register(FIMHawkesConfig.model_type, FIMHawkes)
AutoConfig.register(FIMHawkesConfig.model_type, FIMHawkesConfig)
AutoModel.register(FIMHawkesConfig, FIMHawkes)
