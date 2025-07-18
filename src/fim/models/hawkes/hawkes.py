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
from .piecewise_intensity import PiecewiseHawkesIntensity
from .thinning import EventSampler


class FIMHawkesConfig(PretrainedConfig):
    model_type = "fimhawkes"

    def __init__(
        self,
        mark_encoder: dict = None,
        time_encoder: dict = None,
        delta_time_encoder: dict = None,
        intensity_evaluation_time_encoder: dict = None,
        evaluation_mark_encoder: dict = None,
        context_ts_encoder: dict = None,
        inference_ts_encoder: dict = None,
        functional_attention: dict = None,
        context_self_attention: dict = None,
        mu_decoder: dict = None,
        alpha_decoder: dict = None,
        beta_decoder: dict = None,
        intensity_decoder: dict = None,
        hidden_dim: int = None,
        max_num_marks: int = 1,
        normalize_times: bool = True,
        normalize_by_max_time: bool = True,
        thinning: dict = None,
        loss_weights: dict = None,
        uncertainty_decoder: dict = None,
        **kwargs,
    ):
        self.max_num_marks = max_num_marks
        self.normalize_times = normalize_times
        self.normalize_by_max_time = normalize_by_max_time
        self.mark_encoder = mark_encoder
        self.time_encoder = time_encoder
        self.delta_time_encoder = delta_time_encoder
        self.intensity_evaluation_time_encoder = intensity_evaluation_time_encoder
        self.evaluation_mark_encoder = evaluation_mark_encoder
        self.context_ts_encoder = context_ts_encoder
        self.inference_ts_encoder = inference_ts_encoder
        self.functional_attention = functional_attention
        self.context_self_attention = context_self_attention
        self.mu_decoder = mu_decoder
        self.alpha_decoder = alpha_decoder
        self.beta_decoder = beta_decoder
        self.intensity_decoder = intensity_decoder
        self.hidden_dim = hidden_dim
        self.thinning = thinning
        self.loss_weights = loss_weights
        self.uncertainty_decoder = uncertainty_decoder
        if "model_type" in kwargs:
            del kwargs["model_type"]
        super().__init__(model_type=self.model_type, **kwargs)


class FIMHawkes(AModel):
    """
    FIMHawkes: A Neural Recognition Model for Zero-Shot Inference of Hawkes Processes

    Attributes:
        max_num_marks (int): Maximum number of marks in the Hawkes process.
        normalize_by_max_time (bool): Whether to normalize the input times by the maximum time in the context sequences.
        thinning (dict): The thinning parameters.
        mark_encoder (dict): The mark encoder configuration.
        time_encoder (dict): The time encoder configuration.
        delta_time_encoder (dict): The delta time encoder configuration.
        evaluation_mark_encoder (dict): The mark encoder for the selected mark during evaluation.
        context_ts_encoder (dict): The time series encoder for the context sequences.
        inference_ts_encoder (dict): The time series encoder for the inference sequences.
        functional_attention (dict): The functional attention configuration.
        intensity_decoder (dict): The intensity decoder configuration.
        loss: TBD

    """

    config_class = FIMHawkesConfig

    def __init__(self, config: FIMHawkesConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.max_num_marks = config.max_num_marks
        self.normalize_by_max_time = config.normalize_by_max_time
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
        context_self_attention = copy.deepcopy(self.config.context_self_attention)
        mu_decoder = copy.deepcopy(self.config.mu_decoder)
        alpha_decoder = copy.deepcopy(self.config.alpha_decoder)
        beta_decoder = copy.deepcopy(self.config.beta_decoder)
        # Optional uncertainty decoder – if none is provided we fall back to a simple linear layer.
        uncertainty_decoder_cfg = (
            copy.deepcopy(self.config.uncertainty_decoder) if getattr(self.config, "uncertainty_decoder", None) is not None else None
        )
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

        self.input_layernorm = torch.nn.LayerNorm(self.hidden_dim)

        self.functional_attention = AttentionOperator(embed_dim=self.hidden_dim, out_features=self.hidden_dim, **functional_attention)

        # Learnable mark-specific query embeddings q(m)
        self.mark_queries = torch.nn.Parameter(torch.randn(self.max_num_marks, self.hidden_dim))

        # Separate decoders for the three Hawkes parameters
        # Each maps the final event representation h_i^m to a single parameter
        mu_decoder["in_features"] = self.hidden_dim
        mu_decoder["out_features"] = 1
        self.mu_decoder = create_class_instance(mu_decoder.pop("name"), mu_decoder)

        alpha_decoder["in_features"] = self.hidden_dim
        alpha_decoder["out_features"] = 1
        self.alpha_decoder = create_class_instance(alpha_decoder.pop("name"), alpha_decoder)

        beta_decoder["in_features"] = self.hidden_dim
        beta_decoder["out_features"] = 1
        self.beta_decoder = create_class_instance(beta_decoder.pop("name"), beta_decoder)

        # ------------------------------------------------------------------
        # φ_u  –  per-sample log-variance head
        # ------------------------------------------------------------------
        if uncertainty_decoder_cfg is not None:
            uncertainty_decoder_cfg["in_features"] = self.hidden_dim
            uncertainty_decoder_cfg["out_features"] = 1
            self.uncertainty_decoder = create_class_instance(uncertainty_decoder_cfg.pop("name"), uncertainty_decoder_cfg)
        else:
            # Default: a single linear projection to one log-variance value
            self.uncertainty_decoder = torch.nn.Linear(self.hidden_dim, 1)

        # ------------------------------------------------------------------
        # Global task-uncertainty scalars
        # ------------------------------------------------------------------
        self.omega_lambda_balanced = torch.nn.Parameter(torch.zeros(1))
        self.omega_nll = torch.nn.Parameter(torch.zeros(1))

        # Single learnable query for path summaries
        self.path_summary_query = torch.nn.Parameter(torch.randn(1, self.hidden_dim))

        # Self-attention layer to enhance context summaries
        self.context_self_attn = torch.nn.MultiheadAttention(embed_dim=self.hidden_dim, **context_self_attention)

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
        # Combine random evaluation times with event times
        x["intensity_evaluation_times"] = torch.cat([x["intensity_evaluation_times"], x["inference_event_times"][:, :, :, 0]], dim=2)

        B, P_context, L = x["context_event_times"].shape[:3]
        P_inference = x["inference_event_times"].shape[1]

        num_marks = x.get("num_marks", self.max_num_marks)
        # Handle potential tensor input for `num_marks`
        if isinstance(num_marks, torch.Tensor):
            num_marks = int(num_marks.flatten()[0].item())

        # Sanity-check for mark vocabulary size
        if num_marks > self.max_num_marks:
            raise ValueError(
                f"Batch contains {num_marks} marks, but the model was initialised with max_num_marks="
                f"{self.max_num_marks}. Increase `max_num_marks` in the config or filter the dataset."
            )

        # Provide default sequence lengths if missing
        if "context_seq_lengths" not in x:
            x["context_seq_lengths"] = torch.full((B, P_context), L, device=self.device)
        if "inference_seq_lengths" not in x:
            x["inference_seq_lengths"] = torch.full((B, P_inference), L, device=self.device)

        # Compute delta times
        self._compute_delta_times_inplace(x, "context")
        self._compute_delta_times_inplace(x, "inference")

        # Normalise input times if requested
        norm_constants = self._normalize_input_times(x) if self.normalize_times else torch.ones(B, device=self.device)

        # ------------------------------------------------------------------
        # Encoding of observations (context & target/inference)
        # ------------------------------------------------------------------
        sequence_encodings_context = self._encode_observations_optimized(x, "context")  # [B, P_context, L, D]
        sequence_encodings_inference = self._encode_observations_optimized(x, "inference")  # [B,P,L,D]

        # ------------------------------------------------------------------
        # (10) Path Summary: obtain h_k^{context} via functional attention
        # ------------------------------------------------------------------
        # For each context path k, we compute h_k^{context} = ψ_attn(q, H_k^{context}, H_k^{context})
        # where H_k^{context} = {h_{k,1}^{context}, ..., h_{k,I_k}^{context}} are the event encodings

        B, P_context, L, D = sequence_encodings_context.shape

        # Reshape for batch processing: [B*P_context, L, D]
        context_flat = sequence_encodings_context.view(B * P_context, L, D)

        # Expand the learnable query q for each path: [B*P_context, 1, D]
        q_expanded = self.path_summary_query.expand(B * P_context, -1, -1)

        # Create attention mask for padded positions
        context_seq_lengths_flat = x["context_seq_lengths"].view(-1)  # [B*P_context]
        positions = torch.arange(L, device=self.device).unsqueeze(0)  # [1, L]
        key_padding_mask = positions >= context_seq_lengths_flat.unsqueeze(1)  # [B*P_context, L]

        # Apply functional attention: q attends to H_k^{context}
        # This implements Eq. (4): h_k^{context} = ψ_attn(q, H_k^{context}, H_k^{context})
        h_k_context_flat = self.functional_attention(
            q_expanded,  # locations_encoding: [B*P_context, 1, D]
            context_flat.unsqueeze(2),  # observations_encoding: [B*P_context, L, 1, D]
            observations_padding_mask=key_padding_mask.unsqueeze(-1),  # [B*P_context, L, 1]
        )  # Result shape: [B*P_context, 1, D]

        # Reshape back to [B, P_context, D] (squeeze the singleton query dimension)
        h_k_context = h_k_context_flat.squeeze(1).view(B, P_context, D)

        # ------------------------------------------------------------------
        # (5) Concatenate all context path summaries: H_context = {h_1^{context}, ..., h_K^{context}}
        # ------------------------------------------------------------------
        H_context = h_k_context  # [B, P_context, D] - this is already the concatenation

        # ------------------------------------------------------------------
        # (6) Enhance context embeddings via self-attention across paths
        # ------------------------------------------------------------------
        enhanced_context = H_context + self.context_self_attn(H_context, H_context, H_context)[0]  # [B, P_context, D]

        # ------------------------------------------------------------------
        # Convert event representations to intensity parameters
        # ------------------------------------------------------------------
        # (11) Cross-Attention: q(m) = φ_meval(m) - learnable mark queries
        mark_queries = self.mark_queries[:num_marks]  # [M, D] - these are q(m)

        # (8) Cross-attention: h_final^{i,m} = ψ_attn(q(m) + h_i^{target}, H_combined^i, H_combined^i)
        # where H_combined^i = {H_context, h_1^{target}, ..., h_i^{target}}

        # Vectorized implementation to avoid nested loops
        B, P_inference, L, D = sequence_encodings_inference.shape
        P_context = enhanced_context.shape[1]

        # Prepare queries: q(m) + h_i^{target} for all combinations
        # Shape: [B, M, P_inference, L, D]
        target_expanded = sequence_encodings_inference.unsqueeze(1).expand(-1, num_marks, -1, -1, -1)  # [B, M, P_inference, L, D]
        mark_queries_expanded = mark_queries.view(1, num_marks, 1, 1, D).expand(B, -1, P_inference, L, -1)  # [B, M, P_inference, L, D]
        queries = target_expanded + mark_queries_expanded  # [B, M, P_inference, L, D]

        # Prepare keys/values: H_combined^i = {H_context, h_1^{target}, ..., h_i^{target}}
        # For each event i, we need context + target history up to i
        # We'll create a large tensor with proper causal masking

        # Context part: same for all events (fully visible)
        # Shape: [B, P_context, D] -> [B, P_inference, L, P_context, D]
        context_expanded = enhanced_context.unsqueeze(1).unsqueeze(2).expand(-1, P_inference, L, -1, -1)

        # Target part: causal history for each event
        # Shape: [B, P_inference, L, L, D] where the last L dimension is causal
        target_causal = sequence_encodings_inference.unsqueeze(2).expand(-1, -1, L, -1, -1)  # [B, P_inference, L, L, D]

        # Create causal mask for target part: event i can see events 0 to i
        causal_mask = torch.triu(torch.ones(L, L, device=self.device), diagonal=1).bool()  # [L, L]
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(B, P_inference, -1, -1)  # [B, P_inference, L, L]

        # Apply causal mask to target part
        target_causal = target_causal.masked_fill(causal_mask.unsqueeze(-1), 0.0)

        # Combine context and target parts
        # Shape: [B, P_inference, L, P_context + L, D]
        combined_keys_values = torch.cat([context_expanded, target_causal], dim=3)

        # Reshape for batch attention
        # Queries: [B, M, P_inference, L, D] -> [B * M * P_inference * L, 1, D]
        queries_flat = queries.reshape(B * num_marks * P_inference * L, 1, D)

        # Keys/Values: [B, P_inference, L, P_context + L, D] -> [B * P_inference * L, P_context + L, D]
        # We need to expand this for each mark
        keys_values_expanded = combined_keys_values.unsqueeze(1).expand(
            -1, num_marks, -1, -1, -1, -1
        )  # [B, M, P_inference, L, P_context + L, D]
        keys_values_flat = keys_values_expanded.reshape(B * num_marks * P_inference * L, P_context + L, D)

        # Create attention mask for padded positions
        # Context part: use context sequence lengths
        context_seq_lengths_expanded = x["context_seq_lengths"].unsqueeze(1).unsqueeze(2)  # [B, 1, 1, P_context]
        context_positions = torch.arange(P_context, device=self.device).view(1, 1, 1, P_context)  # [1, 1, 1, P_context]
        context_padding_mask = context_positions >= context_seq_lengths_expanded  # [B, 1, 1, P_context]
        context_padding_mask = context_padding_mask.expand(-1, P_inference, L, -1)  # [B, P_inference, L, P_context]

        # Target part: use inference sequence lengths and causal mask
        inference_seq_lengths_expanded = x["inference_seq_lengths"].unsqueeze(2)  # [B, P_inference, 1]
        target_positions = torch.arange(L, device=self.device).view(1, 1, L)  # [1, 1, L]
        target_seq_padding_mask = target_positions >= inference_seq_lengths_expanded  # [B, P_inference, L]
        target_seq_padding_mask = target_seq_padding_mask.unsqueeze(2).expand(-1, -1, L, -1)  # [B, P_inference, L, L]

        # Combine sequence padding mask with causal mask for target part
        target_final_mask = target_seq_padding_mask | causal_mask  # [B, P_inference, L, L]

        # Combine context and target masks
        combined_padding_mask = torch.cat(
            [
                context_padding_mask,  # [B, P_inference, L, P_context]
                target_final_mask,  # [B, P_inference, L, L]
            ],
            dim=3,
        )  # [B, P_inference, L, P_context + L]

        # Expand for marks and flatten
        combined_padding_mask_expanded = combined_padding_mask.unsqueeze(1).expand(
            -1, num_marks, -1, -1, -1
        )  # [B, M, P_inference, L, P_context + L]
        combined_padding_mask_flat = combined_padding_mask_expanded.reshape(B * num_marks * P_inference * L, P_context + L)

        # Apply cross-attention
        h_final_flat = self.functional_attention(
            queries_flat,  # locations_encoding: [B * M * P_inference * L, 1, D]
            keys_values_flat.unsqueeze(2),  # observations_encoding: [B * M * P_inference * L, P_context + L, 1, D]
            observations_padding_mask=combined_padding_mask_flat.unsqueeze(-1),  # [B * M * P_inference * L, P_context + L, 1]
        )  # Result shape: [B * M * P_inference * L, 1, D]

        # Reshape back to original structure
        # [B * M * P_inference * L, 1, D] -> [B, M, P_inference, L, D]
        combined_enc = h_final_flat.squeeze(1).reshape(B, num_marks, P_inference, L, D)

        # Decode raw parameters and apply Softplus to ensure positivity.
        raw_params = self.mu_decoder(combined_enc.reshape(-1, self.hidden_dim))  # [...,1]
        raw_params = raw_params.view(B, num_marks, P_inference, L, 1)

        mu = torch.nn.functional.softplus(raw_params[..., 0])

        raw_params = self.alpha_decoder(combined_enc.reshape(-1, self.hidden_dim))  # [...,1]
        raw_params = raw_params.view(B, num_marks, P_inference, L, 1)

        alpha = torch.nn.functional.softplus(raw_params[..., 0])

        raw_params = self.beta_decoder(combined_enc.reshape(-1, self.hidden_dim))  # [...,1]
        raw_params = raw_params.view(B, num_marks, P_inference, L, 1)

        beta = torch.nn.functional.softplus(raw_params[..., 0])

        # ------------------------------------------------------------------
        # Predict per-sample log-variance u_{λ,τ,m}
        # We detach h_final to avoid back-propagating through the shared encoder,
        # ensuring gradients only flow into the uncertainty head.
        # ------------------------------------------------------------------
        combined_enc_detached = combined_enc.detach()  # Detach to avoid gradients to shared encoder

        u_event_flat = self.uncertainty_decoder(combined_enc_detached.reshape(-1, self.hidden_dim))  # [...,1]
        u_event = u_event_flat.view(B, num_marks, P_inference, L)  # [B,M,P,L]

        # Map event-level uncertainties to evaluation times τ by selecting the last event
        # preceding each τ for the same path (implementation of T_k*(τ)).
        eval_times = x["intensity_evaluation_times"]  # [B,P,L_eval]
        event_times_path = x["inference_event_times"].squeeze(-1)  # [B,P,L]

        idx_range = torch.arange(L, device=self.device)

        # Compute mask identifying events occurring before each evaluation time
        before_mask = event_times_path.unsqueeze(2) < eval_times.unsqueeze(-1)  # [B,P,L_eval,L]
        idx_range_broadcast = idx_range.view(1, 1, 1, L)
        idx_masked = torch.where(before_mask, idx_range_broadcast, torch.full_like(idx_range_broadcast, -1))
        last_event_idx = idx_masked.max(dim=-1).values  # [B,P,L_eval]
        last_event_idx = last_event_idx.clamp(min=0)  # Replace -1 (no past event) by 0

        # Gather u values corresponding to last event before τ
        last_event_idx_expanded = last_event_idx.unsqueeze(1).expand(-1, num_marks, -1, -1)  # [B,M,P,L_eval]
        u_eval = torch.gather(u_event, dim=3, index=last_event_idx_expanded)  # [B,M,P,L_eval]

        # ------------------------------------------------------------------
        # Build piece-wise intensity object and evaluate at requested times
        # ------------------------------------------------------------------
        event_times = x["inference_event_times"].squeeze(-1)  # [B,P,L]
        intensity_fn = PiecewiseHawkesIntensity(event_times, mu, alpha, beta)

        predicted_intensity_values = intensity_fn.evaluate(x["intensity_evaluation_times"])  # [B,M,P,L_eval]

        out = {
            "predicted_intensity_values": predicted_intensity_values,
            "intensity_function": intensity_fn,
            "log_predicted_intensity_values_var": u_eval,
        }

        if "kernel_functions" in x:
            # Compute target intensities for plotting and loss computation
            kernel_functions_list, base_intensity_functions_list = self._decode_functions(
                x["kernel_functions"], x["base_intensity_functions"]
            )
            target_intensity_values = self.compute_target_intensity_values(
                kernel_functions_list,
                base_intensity_functions_list,
                x["intensity_evaluation_times"],
                x["inference_event_times"],
                x["inference_event_types"],
                x["inference_seq_lengths"],
                norm_constants,
                num_marks=num_marks,
            )
            out["target_intensity_values"] = target_intensity_values

            # # Compute target integrated intensities using high-precision Monte Carlo
            # target_integrated_intensity = self.compute_target_integrated_intensity(
            #     kernel_functions_list,
            #     base_intensity_functions_list,
            #     x["intensity_evaluation_times"],
            #     x["inference_event_times"],
            #     x["inference_event_types"],
            #     x["inference_seq_lengths"],
            #     norm_constants,
            #     num_marks=num_marks,
            # )
            # out["target_integrated_intensity"] = target_integrated_intensity
            out["target_integrated_intensity"] = torch.zeros_like(out["target_intensity_values"])

            out["losses"] = self.loss(
                intensity_fn=intensity_fn,
                target_intensity_values=out["target_intensity_values"],
                target_integrated_intensity=out["target_integrated_intensity"],
                event_times=x["inference_event_times"].squeeze(-1),
                event_types=x["inference_event_types"].squeeze(-1),
                seq_lengths=x["inference_seq_lengths"],
                eval_times=x["intensity_evaluation_times"],
                uncertainty_values=u_eval,
                schedulers=schedulers,
                step=step,
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
        # Approach: Use the original mask logic but make PyTorch handle it properly
        # The key insight is that the original combined both causal and padding masks
        # We need to create the exact same combined mask but handle the head dimension properly

        # 1. Create base causal mask [L, L]
        causal_mask = torch.triu(torch.ones(L, L), diagonal=1).bool().to(self.device)

        # 2. Create padding mask [B*P, L] for keys
        positions = torch.arange(L, device=self.device).unsqueeze(0)  # (1, L)
        seq_lengths_flat = x["seq_lengths"].view(B * P)  # (B*P,)
        key_padding_mask = positions >= seq_lengths_flat.unsqueeze(1)  # (B*P, L)

        # 3. For now, let's use the simpler approach that should be functionally equivalent
        # The causal mask handles temporal dependencies, key_padding_mask handles sequence lengths
        h = self.ts_encoder(
            path.view(B * P, L, -1),
            mask=causal_mask,  # 2D causal mask - PyTorch will broadcast
            src_key_padding_mask=key_padding_mask,  # 2D key padding mask
        )

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

        path = self.input_layernorm(path)

        # Approach: Use the original mask logic but make PyTorch handle it properly
        # The key insight is that the original combined both causal and padding masks
        # We need to create the exact same combined mask but handle the head dimension properly

        # 1. Create base causal mask [L, L]
        causal_mask = torch.triu(torch.ones(L, L), diagonal=1).bool().to(self.device)

        # 2. Create padding mask [B*P, L] for keys
        positions = torch.arange(L, device=self.device).unsqueeze(0)  # (1, L)
        seq_lengths_flat = x[f"{type}_seq_lengths"].view(B * P)  # (B*P,)
        key_padding_mask = positions >= seq_lengths_flat.unsqueeze(1)  # (B*P, L)

        # This prevents them from contributing to the LayerNorm statistics inside the encoder,
        # which is the source of the unstable gradients in the backward pass.
        path = path.view(B * P, L, -1)
        path.masked_fill_(key_padding_mask.unsqueeze(-1), 0.0)
        path = path.view(B, P, L, -1)

        # 3. For now, let's use the simpler approach that should be functionally equivalent
        # The causal mask handles temporal dependencies, key_padding_mask handles sequence lengths
        if type == "context":
            h = self.context_ts_encoder(
                path.view(B * P, L, -1),
                mask=causal_mask,  # 2D causal mask - PyTorch will broadcast
                src_key_padding_mask=key_padding_mask,  # 2D key padding mask
            )
        elif type == "inference":
            h = self.inference_ts_encoder(
                path.view(B * P, L, -1),
                mask=causal_mask,  # 2D causal mask - PyTorch will broadcast
                src_key_padding_mask=key_padding_mask,  # 2D key padding mask
            )
        else:
            raise ValueError(f"Invalid type: {type}")

        return h.view(B, P, L, -1)

    def _trunk_net_encoder_optimized(self, x: dict, num_marks: int) -> Tensor:
        """Optimized trunk net encoding using cached one-hot encodings"""
        intensity_evaluation_times = x["intensity_evaluation_times"]
        B, P_inference, L_inference = intensity_evaluation_times.shape

        past_event_shifted_intensity_evaluation_times = self._get_past_event_shifted_intensity_evaluation_times(
            intensity_evaluation_times, x["inference_event_times"], num_marks
        )

        # More efficient reshaping and encoding
        time_encodings = self.intensity_evaluation_time_encoder(past_event_shifted_intensity_evaluation_times.reshape(-1, 1)).reshape(
            B, num_marks, P_inference, L_inference, -1
        )

        # Use cached one-hot encodings for evaluation marks
        marks = torch.arange(num_marks, device=intensity_evaluation_times.device)
        one_hot_eval_marks = self.mark_one_hot[marks]  # [M, max_num_marks]
        eval_mark_encodings = self.evaluation_mark_encoder(one_hot_eval_marks)  # [M, hidden_dim]
        mark_encodings = (
            eval_mark_encodings.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(B, -1, P_inference, L_inference, -1)
        )  # [B, M, P_inference, L_inference, D]

        return time_encodings + mark_encodings

    def _functional_attention_encoder_optimized(self, queries: Tensor, keys_values: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Performs functional attention for a single inference target.
        This version formats its output to be compatible with the unmodified AttentionOperator.

        Args:
            queries (Tensor): Query embeddings from the trunk net.
                            Shape: [B, M, L_inference, D]
            keys_values (Tensor): Key/Value embeddings from context + one inference path.
                                Shape: [B, P_context + 1, L, D]
            attention_mask (Tensor): The corresponding dynamic attention mask.
                                    Shape: [B, L_inference, P_context + 1, L]

        Returns:
            Tensor: The final time-dependent encodings after attention.
                    Shape: [B, M, L_inference, D]
        """
        B, M, L_inference, D = queries.shape
        _, P_relevant, L, _ = keys_values.shape

        # Reshape for batched attention. Each (L_inference) query gets its own mask.
        # We treat the L_inference dimension as part of the batch.

        # Reshape queries: [B, M, L_inf, D] -> [B * L_inf, M, D]
        queries_reshaped = queries.permute(0, 2, 1, 3).reshape(B * L_inference, M, D)

        # Expand and reshape keys/values: [B, P_rel, L, D] -> [B, 1, P_rel*L, D] -> [B*L_inf, P_rel*L, D]
        keys_values_expanded = keys_values.unsqueeze(1).expand(-1, L_inference, -1, -1, -1)
        keys_values_reshaped = keys_values_expanded.reshape(B * L_inference, P_relevant * L, D)

        # Reshape mask: [B, L_inf, P_rel, L] -> [B * L_inf, P_rel * L]
        mask_reshaped = attention_mask.reshape(B * L_inference, P_relevant * L)

        # Shape: [B*L_inf, P_rel*L, D] -> [B*L_inf, P_rel*L, 1, D]
        keys_values_4d = keys_values_reshaped.unsqueeze(2)

        # The padding mask is already prepared in the correct 3D shape for the operator's internal layers.
        # Shape: [B*L_inf, P_rel*L, 1]
        padding_mask_3d = mask_reshaped.unsqueeze(-1)

        # Apply functional attention with the correctly shaped tensors
        result = self.functional_attention(
            queries_reshaped,
            keys_values_4d,  # Pass the 4D tensor here
            observations_padding_mask=padding_mask_3d,
        )  # Result shape: [B * L_inf, M, D]

        # Reshape back to original format
        # [B * L_inf, M, D] -> [B, L_inf, M, D] -> [B, M, L_inf, D]
        return result.reshape(B, L_inference, M, D).permute(0, 2, 1, 3)

    def _denormalize_output(self, x: dict, out: dict, norm_constants: Tensor) -> None:
        out["predicted_intensity_values"] = out["predicted_intensity_values"] / norm_constants.view(-1, 1, 1, 1)
        if "target_intensity_values" in out:
            out["target_intensity_values"] = out["target_intensity_values"] / norm_constants.view(-1, 1, 1, 1)
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
        intensity_fn: "PiecewiseHawkesIntensity",
        target_intensity_values: Tensor,
        target_integrated_intensity: Tensor,
        event_times: Tensor,
        event_types: Tensor,
        seq_lengths: Tensor,
        eval_times: Tensor,
        uncertainty_values: Tensor,
        schedulers: dict = None,
        step: int = None,
    ) -> dict:
        """Hybrid uncertainty-weighted loss .

        L_total = L_NLL + exp(-ω_λ) * L_λ,balanced + ω_λ
        where L_λ,balanced incorporates heteroscedastic uncertainty on a per-sample basis.
        """
        # --- 1. Predicted intensity ---
        predicted_intensity_values = intensity_fn.evaluate(eval_times)
        squared_error = (predicted_intensity_values - target_intensity_values) ** 2

        # --- 2. Balanced Intensity Regression Loss  ---
        # L_λ,balanced = E[ exp(-u) * (error^2) + u ]
        lambda_balanced_loss = torch.mean(torch.exp(-uncertainty_values) * squared_error + uncertainty_values)

        # For logging – plain MSE & MAE.
        mse_loss = torch.mean(squared_error)
        mae_loss = torch.mean(torch.abs(predicted_intensity_values - target_intensity_values))

        # --- 3. Negative Log-Likelihood Loss (normalised per path, mark, and event) ---
        B, P, L = event_times.shape

        num_marks = intensity_fn.mu.shape[1]

        # a) Log-intensity at actual event times and marks
        log_intensity_at_events = torch.log(intensity_fn.evaluate(event_times))  # [B,M,P,L]
        type_idx = event_types.unsqueeze(1).expand(-1, 1, -1, -1)  # [B,1,P,L]
        log_lambda_at_event_m = torch.gather(log_intensity_at_events, 1, type_idx).squeeze(1)  # [B,P,L]

        # Mask out padded events and sum per path
        positions = torch.arange(L, device=self.device).view(1, 1, L)
        valid_mask = positions < seq_lengths.unsqueeze(2)  # [B,P,L]
        log_ll_per_path = (log_lambda_at_event_m * valid_mask).sum(dim=2)  # [B,P]

        # b) Integrated intensity per path & mark, then sum over marks
        t_start = torch.zeros(B, P, device=self.device)
        t_end = event_times.max(dim=2).values  # [B,P]
        integral_per_mark_path = intensity_fn.integral(t_start=t_start, t_end=t_end)  # [B,M,P]
        integral_sum_per_path = integral_per_mark_path.sum(dim=1)  # [B,P]

        # c) Per-path NLL and normalisation
        nll_per_path = integral_sum_per_path - log_ll_per_path  # [B,P]
        # Divide by (#events in path * #marks) for per-event & per-mark normalisation
        nll_per_path_normalised = nll_per_path / num_marks

        # d) Average across paths and batch
        nll_loss = nll_per_path_normalised.mean()

        # --- 4. Hybrid Uncertainty Weighting ---
        total_loss = (
            torch.exp(-self.omega_nll) * nll_loss
            + self.omega_nll
            + torch.exp(-self.omega_lambda_balanced) * lambda_balanced_loss
            + self.omega_lambda_balanced
        )

        # Prepare a logging-friendly dictionary: tensors -> Python floats
        losses_out = {
            "loss": total_loss,  # keep tensor for downstream back-prop accounting
            "nll_loss": nll_loss.detach().item(),
            "lambda_balanced_loss": lambda_balanced_loss.detach().item(),
            "mse_loss": mse_loss.detach().item(),
            "mae_loss": mae_loss.detach().item(),
            "omega_lambda_balanced": self.omega_lambda_balanced.detach().item(),
            "omega_nll": self.omega_nll.detach().item(),
        }

        return losses_out

    def compute_target_integrated_intensity(
        self,
        kernel_functions_list: list,
        base_intensity_functions_list: list,
        intensity_evaluation_times: Tensor,
        inference_event_times: Tensor,
        inference_event_types: Tensor,
        inference_seq_lengths: Tensor,
        norm_constants: Tensor,
        num_marks: int,
        num_samples: int = 100,
    ):
        """
        Computes the ground-truth integrated intensity Λ(t) = ∫λ(s)ds from 0 to t
        for each t in `intensity_evaluation_times` using high-precision Monte Carlo integration.
        """
        B, P, L_eval = intensity_evaluation_times.shape
        device = intensity_evaluation_times.device

        # 1. Generate random time samples for integration for each interval [0, t]
        # Shape: [B, P, L_eval, num_samples]
        rand_for_sampling = torch.rand(B, P, L_eval, num_samples, device=device)

        # Shape of t_end: [B, P, L_eval] -> [B, P, L_eval, 1] for broadcasting
        t_end = intensity_evaluation_times.unsqueeze(-1)

        # Create samples in [0, t_end] for each t_end
        # Shape: [B, P, L_eval, num_samples]
        time_samples = rand_for_sampling * t_end

        # 2. Evaluate the ground-truth intensity λ(s) at these sample points.
        # The `compute_target_intensity_values` function expects time inputs of shape [B, P, L_points].
        # We flatten the L_eval and num_samples dimensions to comply.
        time_samples_flat = time_samples.reshape(B, P, L_eval * num_samples)

        intensity_at_samples_flat = self.compute_target_intensity_values(
            kernel_functions_list,
            base_intensity_functions_list,
            time_samples_flat,  # Pass the dense samples here
            inference_event_times,
            inference_event_types,
            inference_seq_lengths,
            norm_constants,
            num_marks,
        )
        # Result shape: [B, M, P, L_eval * num_samples]

        # 3. Reshape back and perform Monte Carlo estimation
        # Reshape to: [B, M, P, L_eval, num_samples]
        intensity_at_samples = intensity_at_samples_flat.reshape(B, num_marks, P, L_eval, num_samples)

        # Average the intensity values for each interval
        # Shape: [B, M, P, L_eval]
        mean_intensity = intensity_at_samples.mean(dim=-1)

        # Integral ≈ (t_end - t_start) * mean_intensity. Here t_start is 0.
        # Shape of t_end: [B, P, L_eval] -> [B, 1, P, L_eval] for broadcasting
        integral_estimate = mean_intensity * intensity_evaluation_times.unsqueeze(1)

        return integral_estimate

    def compute_target_intensity_values(
        self,
        kernel_functions_list: list,
        base_intensity_functions_list: list,
        intensity_evaluation_times: Tensor,
        inference_event_times: Tensor,
        inference_event_types: Tensor,
        inference_seq_lengths: Tensor,
        norm_constants: Tensor,
        num_marks: int,
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
                Shape: [B, P, L_eval]. These are NORMALIZED.
            inference_event_times (Tensor): The historical event timestamps t_jk.
                Shape: [B, P, L_hist, 1]. These are NORMALIZED.
            inference_event_types (Tensor): The historical event types `j`.
                Shape: [B, P, L_hist, 1].
            inference_seq_lengths (Tensor): The actual length of each historical sequence.
                Shape: [B, P].
            norm_constants (Tensor): The normalization constants.
                Shape: [B].
            num_marks (int): The number of marks for the batch.

        Returns:
            Tensor: The computed target intensity values, scaled for the normalized time domain.
                    Shape: [B, D, P, L_eval], where D is the number of marks.
        """
        device = intensity_evaluation_times.device
        B, P, L_eval = intensity_evaluation_times.shape
        _, _, L_hist, _ = inference_event_times.shape
        D = num_marks

        # --- START: MODIFICATION ---
        # The input times are normalized. We must denormalize them to use with the
        # original kernel and base intensity functions.
        norm_constants_eval = norm_constants.view(B, 1, 1)
        norm_constants_hist = norm_constants.view(B, 1, 1, 1)

        intensity_evaluation_times_orig = intensity_evaluation_times * norm_constants_eval
        inference_event_times_orig = inference_event_times * norm_constants_hist
        # --- END: MODIFICATION ---

        # Ensure tensors are 3D for easier processing by removing the trailing dimension
        event_times = inference_event_times_orig.squeeze(-1)  # Use original times
        event_types = inference_event_types.squeeze(-1)

        # The final output tensor, matching the model's prediction shape [B, M, P, L_eval]
        total_intensity = torch.zeros(B, D, P, L_eval, device=device)

        # Loop over batch items because kernel/base functions are heterogeneous Python objects.
        for b in range(B):
            # The number of marks can vary per sample in the ground truth data.
            # Use the length of the provided function list for this sample.
            actual_marks_b = len(base_intensity_functions_list[b])

            # Extract original-scale data for the current batch item
            eval_times_b = intensity_evaluation_times_orig[b]  # Use original times
            hist_times_b = event_times[b]  # Use original times
            hist_types_b = event_types[b]
            seq_lengths_b = inference_seq_lengths[b]

            # --- Calculate Summed Kernel Term (in original time scale) ---
            delta_t_b = eval_times_b.unsqueeze(-1) - hist_times_b.unsqueeze(-2)

            causality_mask_b = delta_t_b > 1e-9
            hist_indices = torch.arange(L_hist, device=device).view(1, L_hist)
            padding_mask_b = hist_indices < seq_lengths_b.unsqueeze(-1)
            valid_history_mask_b = causality_mask_b & padding_mask_b.unsqueeze(1)

            # Initialize with baseline values
            for i in range(actual_marks_b):
                mu_func_b_i = base_intensity_functions_list[b][i]
                total_intensity[b, i, :, :] = mu_func_b_i(eval_times_b)  # Pass original times

            # Accumulate kernel influences
            for i in range(actual_marks_b):
                for j in range(actual_marks_b):
                    phi_func_b_ij = kernel_functions_list[b][i][j]
                    source_mark_mask_b = (hist_types_b == j).unsqueeze(1)
                    final_mask_b = valid_history_mask_b & source_mark_mask_b

                    if not torch.any(final_mask_b):
                        continue

                    delta_t_to_eval = delta_t_b[final_mask_b]
                    kernel_values = phi_func_b_ij(delta_t_to_eval)  # Pass original time deltas

                    kernel_contribution = torch.zeros_like(delta_t_b)
                    kernel_contribution[final_mask_b] = kernel_values
                    summed_kernel_values = kernel_contribution.sum(dim=-1)
                    total_intensity[b, i, :, :] += summed_kernel_values

        # Apply non-negativity constraint
        target_intensity_values_orig = torch.relu(total_intensity)

        # Re-normalize the final intensity to match the model's output space (λ'(t') = c * λ(t))
        norm_constants_final = norm_constants.view(B, 1, 1, 1)
        target_intensity_values_normalized = target_intensity_values_orig * norm_constants_final

        return target_intensity_values_normalized

    def _get_past_event_shifted_intensity_evaluation_times(
        self, intensity_evaluation_times: Tensor, inference_event_times: Tensor, num_marks: int
    ):
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
        M = num_marks

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

    def _create_single_inference_mask(self, x: dict, target_inference_idx: int) -> Tensor:
        """
        Creates a dynamic visibility mask for a single inference path.

        The mask makes all context paths fully visible (respecting padding) and makes
        the single target inference path causally visible based on evaluation times.

        Args:
            x (dict): The input data dictionary.
            target_inference_idx (int): The index of the current target inference path.

        Returns:
            Tensor: A boolean mask where `True` indicates a masked position.
                    Shape: [B, L_inference, P_context + 1, L]
        """
        B, P_context, L = x["context_event_times"].shape[:3]
        L_inference = x["intensity_evaluation_times"].shape[2]
        device = x["context_event_times"].device

        # --- 1. Static Padding Mask ---
        # Combine seq lengths for context paths and the single target inference path.
        context_lengths = x["context_seq_lengths"]  # [B, P_context]
        target_inference_length = x["inference_seq_lengths"][:, target_inference_idx : target_inference_idx + 1]  # [B, 1]

        # Shape: [B, P_context + 1]
        relevant_seq_lengths = torch.cat([context_lengths, target_inference_length], dim=1)

        positions = torch.arange(L, device=device).view(1, 1, L)

        # static_mask is True for padded positions. Shape: [B, P_context + 1, L]
        static_mask = positions >= relevant_seq_lengths.unsqueeze(2)
        # Expand for broadcasting with the dynamic mask. Shape: [B, 1, P_context + 1, L]
        static_mask = static_mask.unsqueeze(1)

        # --- 2. Dynamic Causality Mask for the Target Path ---
        # Get evaluation times and event times for only the target path.
        eval_times = x["intensity_evaluation_times"][:, target_inference_idx, :]  # [B, L_inference]
        inference_times = x["inference_event_times"][:, target_inference_idx, :, :].squeeze(-1)  # [B, L]

        # dynamic_mask is True if event_time >= eval_time. Shape: [B, L_inference, L]
        dynamic_mask = inference_times.unsqueeze(1) >= eval_times.unsqueeze(2)

        # --- 3. Combine Masks ---
        # The dynamic mask only applies to the inference path, which is the last one
        # in our concatenated sequence. We create a mask of Falses for the context paths.

        # Shape: [B, L_inference, P_context, L]
        context_dynamic_mask = torch.zeros(B, L_inference, P_context, L, dtype=torch.bool, device=device)

        # Add a dimension to the inference dynamic mask and concatenate.
        # Shape: [B, L_inference, P_context + 1, L]
        full_dynamic_mask = torch.cat([context_dynamic_mask, dynamic_mask.unsqueeze(2)], dim=2)

        # Final mask is the union of static padding and dynamic causality.
        final_mask = static_mask | full_dynamic_mask

        return final_mask

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
