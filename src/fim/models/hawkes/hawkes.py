import copy
import logging
from typing import Any, Dict, Tuple

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
        context_summary_pooling: dict = None,
        evaluation_mark_encoder: dict = None,
        context_ts_encoder: dict = None,
        decoder_ts: dict = None,
        mark_fusion_attention: dict = None,
        context_summary_encoder: dict = None,
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
        **kwargs,
    ):
        self.max_num_marks = max_num_marks
        self.normalize_times = normalize_times
        self.normalize_by_max_time = normalize_by_max_time
        self.mark_encoder = mark_encoder
        self.time_encoder = time_encoder
        self.delta_time_encoder = delta_time_encoder
        self.context_summary_pooling = context_summary_pooling
        self.evaluation_mark_encoder = evaluation_mark_encoder
        self.context_ts_encoder = context_ts_encoder
        self.context_summary_encoder = context_summary_encoder
        self.context_self_attention = context_self_attention
        self.decoder_ts = decoder_ts
        self.mark_fusion_attention = mark_fusion_attention
        self.mu_decoder = mu_decoder
        self.alpha_decoder = alpha_decoder
        self.beta_decoder = beta_decoder
        self.intensity_decoder = intensity_decoder
        self.hidden_dim = hidden_dim
        self.thinning = thinning
        self.loss_weights = loss_weights
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
        loss_weights (dict): The loss weights.
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
        evaluation_mark_encoder = copy.deepcopy(self.config.evaluation_mark_encoder)
        context_summary_pooling = copy.deepcopy(self.config.context_summary_pooling)
        context_ts_encoder = copy.deepcopy(self.config.context_ts_encoder)
        context_summary_encoder = copy.deepcopy(self.config.context_summary_encoder)
        context_self_attention = copy.deepcopy(self.config.context_self_attention)
        decoder_ts = copy.deepcopy(self.config.decoder_ts)
        mark_fusion_attention = copy.deepcopy(getattr(self.config, "mark_fusion_attention", None))
        mu_decoder = copy.deepcopy(self.config.mu_decoder)
        alpha_decoder = copy.deepcopy(self.config.alpha_decoder)
        beta_decoder = copy.deepcopy(self.config.beta_decoder)
        self.hidden_dim = self.config.hidden_dim
        self.normalize_times = self.config.normalize_times

        mark_encoder["in_features"] = self.max_num_marks
        self.mark_encoder = create_class_instance(mark_encoder.pop("name"), mark_encoder)
        # Support both Linear and custom time encoders (e.g., SineTimeEncoding)
        _te_name = time_encoder.get("name", "")
        if _te_name.endswith("SineTimeEncoding"):
            # SineTimeEncoding expects only out_features
            time_encoder["out_features"] = self.hidden_dim
            self.time_encoder = create_class_instance(time_encoder.pop("name"), time_encoder)
        else:
            time_encoder["in_features"] = 1
            self.time_encoder = create_class_instance(time_encoder.pop("name"), time_encoder)
        _dte_name = delta_time_encoder.get("name", "")
        if _dte_name.endswith("SineTimeEncoding"):
            delta_time_encoder["out_features"] = self.hidden_dim
            self.delta_time_encoder = create_class_instance(delta_time_encoder.pop("name"), delta_time_encoder)
        else:
            delta_time_encoder["in_features"] = 1
            self.delta_time_encoder = create_class_instance(delta_time_encoder.pop("name"), delta_time_encoder)
        evaluation_mark_encoder["in_features"] = self.max_num_marks
        evaluation_mark_encoder["out_features"] = self.hidden_dim
        self.evaluation_mark_encoder = create_class_instance(evaluation_mark_encoder.pop("name"), evaluation_mark_encoder)

        # Context summarization via AttentionOperator with a static learnable query
        csd_cfg = copy.deepcopy(context_summary_pooling)
        csd_name = csd_cfg.pop("name")
        if csd_name.endswith("neural_operators.AttentionOperator"):
            csd_cfg["embed_dim"] = self.hidden_dim
            csd_cfg["out_features"] = self.hidden_dim
            self.context_summary_pooling = AttentionOperator(**csd_cfg)
        else:
            csd_cfg["embed_dim"] = self.hidden_dim
            csd_cfg["out_features"] = self.hidden_dim
            self.context_summary_pooling = create_class_instance(csd_name, csd_cfg)

        context_ts_encoder["encoder_layer"]["d_model"] = self.hidden_dim
        self.context_ts_encoder = create_class_instance(context_ts_encoder.pop("name"), context_ts_encoder)
        # Optional TransformerEncoder to enhance per-path context summaries with residuals and MLPs
        if context_summary_encoder is not None:
            context_summary_encoder["encoder_layer"]["d_model"] = self.hidden_dim
            self.context_summary_encoder = create_class_instance(context_summary_encoder.pop("name"), context_summary_encoder)
        else:
            self.context_summary_encoder = None

        decoder_ts["decoder_layer"]["d_model"] = self.hidden_dim
        self.ts_decoder = create_class_instance(decoder_ts.pop("name"), decoder_ts)

        self.input_layernorm = torch.nn.LayerNorm(self.hidden_dim)

        # Mark fusion attention (ResidualAttentionLayer) to combine decoded tokens with per-mark embeddings
        if mark_fusion_attention is not None:
            mark_fusion_attention["d_model"] = self.hidden_dim
            mark_fusion_attention.setdefault("batch_first", True)
            self.mark_fusion_attn = create_class_instance(mark_fusion_attention.pop("name"), mark_fusion_attention)
        else:
            self.mark_fusion_attn = None

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

        # Single learnable query for path summaries
        self.path_summary_query = torch.nn.Parameter(torch.randn(1, self.hidden_dim))

        self.context_self_attn = torch.nn.MultiheadAttention(embed_dim=self.hidden_dim, **context_self_attention)

        if self.config.thinning is not None:
            self.event_sampler = EventSampler(**self.config.thinning)
        else:
            self.event_sampler = EventSampler(num_sample=1, num_exp=500, over_sample_rate=5, num_samples_boundary=5, dtime_max=5)

        self.loss_weights = self.config.loss_weights

    def encode_context(self, x: Dict[str, Tensor]) -> Tensor:
        """
        Compute and return the enhanced context embeddings given only context tensors.

        Expected keys in `x`:
            - "context_event_times": [B, P_context, L, 1]
            - "context_event_types": [B, P_context, L, 1]
            - "context_seq_lengths": [B, P_context]

        Returns:
            Tensor of shape [B, P_context, D] with enhanced context embeddings.
        """
        # Make a shallow copy so we can insert temporary fields
        ctx: Dict[str, Tensor] = {
            "context_event_times": x["context_event_times"],
            "context_event_types": x["context_event_types"],
            "context_seq_lengths": x["context_seq_lengths"],
        }

        # Compute delta times and normalization (consistent with forward)
        self._compute_delta_times_inplace(ctx, "context")
        if self.normalize_times:
            _, x_norm = self._normalize_input_times(
                {
                    # Minimal inputs required by _normalize_input_times
                    "context_event_times": ctx["context_event_times"],
                    "context_seq_lengths": ctx["context_seq_lengths"],
                    # Provide placeholders for inference keys (not used here)
                    "inference_event_times": ctx["context_event_times"],
                    "intensity_evaluation_times": ctx["context_event_times"].squeeze(-1),
                    "context_delta_times": ctx["context_delta_times"],
                    "inference_delta_times": ctx["context_event_times"],
                }
            )
            ctx.update(x_norm)

        # Encode observations for context and build enhanced context
        sequence_encodings_context = self._encode_observations_optimized(ctx, "context")  # [B, P_context, L, D]

        B, P_context, L, D = sequence_encodings_context.shape
        context_flat = sequence_encodings_context.view(B * P_context, L, D)
        q_expanded = self.path_summary_query.expand(B * P_context, -1, -1)
        context_seq_lengths_flat = ctx["context_seq_lengths"].view(-1)
        positions = torch.arange(L, device=self.device).unsqueeze(0)
        key_padding_mask = positions >= context_seq_lengths_flat.unsqueeze(1)
        # Use AttentionOperator: query attends over path tokens
        path_summary = self.context_summary_pooling(
            q_expanded,
            context_flat.unsqueeze(2),
            observations_padding_mask=key_padding_mask.unsqueeze(-1),
        )  # [B*P, 1, D]
        h_k_context = path_summary.squeeze(1).view(B, P_context, D)
        enhanced_context = self.context_summary_encoder(h_k_context)
        return enhanced_context

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

        # Normalise input times if requested (produce normalized copies, keep originals intact)
        if self.normalize_times:
            norm_constants, x_norm = self._normalize_input_times(x)
            x.update(x_norm)
        else:
            norm_constants = torch.ones(B, device=self.device)

        # ------------------------------------------------------------------
        # Encoding of observations (context & target/inference)
        # ------------------------------------------------------------------
        # Build target/inference token embeddings (decoder inputs)
        sequence_encodings_inference = self._encode_observations_optimized(x, "inference")  # [B, P_inference, L, D]

        # ------------------------------------------------------------------
        # Convert event representations to intensity parameters using TransformerDecoder
        # ------------------------------------------------------------------
        # Build memory from context via attention pooling with ResidualAttentionLayer
        # This yields one summary token per context path
        enhanced_context = self.encode_context(
            {
                "context_event_times": x["context_event_times"],
                "context_event_types": x["context_event_types"],
                "context_seq_lengths": x["context_seq_lengths"],
            }
        )  # [B, P_context, D]
        B, P_context, D = enhanced_context.shape
        memory = enhanced_context  # [B, P_context, D]

        # Prepare target from inference sequences
        B, P_inference, L, D = sequence_encodings_inference.shape
        tgt = sequence_encodings_inference  # [B, P_inference, L, D]
        # Causal mask for tgt (size L x L) and pad mask up to seq_len
        tgt_causal_mask = torch.triu(torch.ones(L, L, device=self.device), diagonal=1).bool()
        positions_inf = torch.arange(L, device=self.device).view(1, 1, L)
        tgt_key_padding_mask = (positions_inf >= x["inference_seq_lengths"].unsqueeze(2)).reshape(B * P_inference, L)

        # Repeat memory for each inference path
        mem_repeated = memory.unsqueeze(1).expand(-1, P_inference, -1, -1).reshape(B * P_inference, P_context, D)
        # Path-level memory tokens: no padding mask required (already pooled over length)

        # Reshape tgt to [B*P_inference, L, D]
        tgt_reshaped = tgt.reshape(B * P_inference, L, D)

        # Decode with TransformerDecoder
        decoded = self.ts_decoder(
            tgt=tgt_reshaped,
            memory=mem_repeated,
            tgt_mask=tgt_causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )  # [B*P_inference, L, D]

        decoded = decoded.reshape(B, P_inference, L, D)

        # Fuse decoded tokens with evaluation mark embeddings using ResidualAttentionLayer over full causal sequence
        mark_basis = self.evaluation_mark_encoder(self.mark_one_hot[:num_marks])  # [M, D]
        # Prepare keys/values as the full causal sequence per (B,P)
        decoded_bp = decoded.view(B * P_inference, L, D)  # [B*P, L, D]
        # Build per-time-step queries by repeating mark queries at each position t
        mark_queries = mark_basis.view(1, 1, num_marks, D).expand(B * P_inference, L, -1, -1)  # [B*P, L, M, D]
        # Flatten time into batch for attention calls
        queries = mark_queries.reshape(B * P_inference * L, num_marks, D)  # [B*P*L, M, D]
        keys_values = decoded_bp.unsqueeze(1).expand(-1, L, -1, -1).reshape(B * P_inference * L, L, D)  # [B*P*L, L, D]
        # Create a strictly causal key padding mask per step t: mask future keys j>t
        pos = torch.arange(L, device=self.device)
        causal_k_mask = pos.view(1, L) > pos.view(L, 1)  # [L, L] True where j>t
        causal_k_mask = causal_k_mask.unsqueeze(0).expand(B * P_inference, -1, -1)  # [B*P, L, L]
        causal_k_mask = causal_k_mask.reshape(B * P_inference * L, L)  # [B*P*L, L]
        # Also mask padded target positions
        tgt_kpm = (positions_inf >= x["inference_seq_lengths"].unsqueeze(2)).reshape(B * P_inference, L)  # [B*P, L]
        tgt_kpm_expanded = tgt_kpm.unsqueeze(1).expand(-1, L, -1)  # [B*P, L, L]
        tgt_kpm_expanded = tgt_kpm_expanded.reshape(B * P_inference * L, L)
        kpm = causal_k_mask | tgt_kpm_expanded  # [B*P*L, L]
        fused = self.mark_fusion_attn(queries, keys_values, keys_values, key_padding_mask=kpm.unsqueeze(-1))  # [B*P*L, M, D]
        combined_enc = fused.view(B, P_inference, L, num_marks, D).permute(0, 3, 1, 2, 4)  # [B, M, P_inference, L, D]

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
        # Build piece-wise intensity object and evaluate at requested times
        # ------------------------------------------------------------------
        # Select normalized or original inference event times for intensity function
        if self.normalize_times:
            event_times = x["inference_event_times_norm"].squeeze(-1)
            eval_times = x["intensity_evaluation_times_norm"]
        else:
            event_times = x["inference_event_times"].squeeze(-1)
            eval_times = x["intensity_evaluation_times"]
        # If model used normalization, provide constants for downstream denormalization
        norm_consts_for_intensity = norm_constants if self.normalize_times else None
        intensity_fn = PiecewiseHawkesIntensity(event_times, mu, alpha, beta, norm_consts_for_intensity)

        # Evaluate using normalized times
        predicted_intensity_values = intensity_fn.evaluate(eval_times, normalized_times=True)

        out = {
            "predicted_intensity_values": predicted_intensity_values,
            "intensity_function": intensity_fn,
        }

        if "kernel_functions" in x:
            # Compute target intensities for plotting and loss computation
            kernel_functions_list, base_intensity_functions_list = self._decode_functions(
                x["kernel_functions"], x["base_intensity_functions"]
            )
            # Compute target intensities using same normalized times
            target_intensity_values = self.compute_target_intensity_values(
                kernel_functions_list,
                base_intensity_functions_list,
                x["intensity_evaluation_times_norm"] if self.normalize_times else x["intensity_evaluation_times"],
                x["inference_event_times_norm"] if self.normalize_times else x["inference_event_times"],
                x["inference_event_types"],
                x["inference_seq_lengths"],
                norm_constants,
                num_marks=num_marks,
                inference_time_offsets=x.get("inference_time_offsets", None),
            )
            out["target_intensity_values"] = target_intensity_values

            # Prepare decomposed supervision targets at event times
            mu_star_at_events, _, lambda_post_at_events = self.compute_decomposed_targets_at_events(
                kernel_functions_list=kernel_functions_list,
                base_intensity_functions_list=base_intensity_functions_list,
                inference_event_times=x["inference_event_times_norm"] if self.normalize_times else x["inference_event_times"],
                inference_event_types=x["inference_event_types"],
                inference_seq_lengths=x["inference_seq_lengths"],
                norm_constants=norm_constants,
                num_marks=num_marks,
                inference_time_offsets=x.get("inference_time_offsets", None),
            )

            # Valid mask for (B, P, L) positions excluding padding
            B, P, L = event_times.shape
            positions = torch.arange(L, device=self.device).view(1, 1, L)
            valid_event_mask = positions < x["inference_seq_lengths"].unsqueeze(2)

            out["losses"] = self.loss(
                intensity_fn=intensity_fn,
                predicted_intensity_values=out["predicted_intensity_values"],
                target_intensity_values=out["target_intensity_values"],
                event_times=event_times,
                event_types=x["inference_event_types"].squeeze(-1),
                seq_lengths=x["inference_seq_lengths"],
                schedulers=schedulers,
                step=step,
                mu_targets_at_events=mu_star_at_events,
                lambda_post_targets_at_events=lambda_post_at_events,
                valid_event_mask=valid_event_mask,
            )
        else:
            # No ground-truth functions available: fall back to NLL-only fine-tuning
            # Compute NLL on normalized time domain (internally denormalized if needed)
            nll_only = self._nll_loss(
                intensity_fn=intensity_fn,
                event_times=event_times,
                event_types=x["inference_event_types"].squeeze(-1),
                seq_lengths=x["inference_seq_lengths"],
            )

            # Weight with configured loss weight for compatibility
            relative_spike_loss = self._relative_spike(intensity_fn.mu, intensity_fn.alpha)
            total_loss = self.loss_weights.get("nll", 1.0) * nll_only + self.loss_weights.get("relative_spike", 0.1) * relative_spike_loss

            out["losses"] = {
                "loss": total_loss,
                "nll_loss": nll_only.detach().item(),
                "relative_spike_loss": relative_spike_loss.detach().item(),
                # Placeholders for logging consistency
                "smape_loss": 0.0,
                "mae_loss": 0.0,
            }

        if self.normalize_times:
            self._denormalize_output(out, norm_constants)

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

    def _intensity_decoder(self, time_dependent_path_summary: Tensor) -> Tensor:
        B, M, P_inference, L_inference, D = time_dependent_path_summary.shape
        time_dependent_path_summary = time_dependent_path_summary.view(B * M * P_inference * L_inference, D)
        h = self.intensity_decoder(time_dependent_path_summary)
        return h.view(B, M, P_inference, L_inference)

    def _normalize_input_times(self, x: dict) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute normalization constants and normalized time tensors without mutating inputs.
        Returns (norm_constants, normalized_times) where normalized_times is a dict with keys:
            'context_event_times_norm', 'context_delta_times_norm',
            'inference_event_times_norm', 'inference_delta_times_norm',
            'intensity_evaluation_times_norm'.
        """
        # Compute normalization constants as before
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

            masked_delta_times = masked_delta_times.squeeze(-1)  # (B, P, L)
            positions = torch.arange(L, device=masked_delta_times.device).view(1, 1, L)
            seq_lengths_expanded = x["context_seq_lengths"].unsqueeze(2)
            masked_delta_times[positions >= seq_lengths_expanded] = float("-inf")
            norm_constants = masked_delta_times.amax(dim=[1, 2])

        # Build normalized time tensors without altering original x
        nc_view4 = norm_constants.view(-1, 1, 1, 1)
        nc_view3 = norm_constants.view(-1, 1, 1)
        normalized = {
            "context_event_times_norm": x["context_event_times"] / nc_view4,
            "context_delta_times_norm": x["context_delta_times"] / nc_view4,
            "inference_event_times_norm": x["inference_event_times"] / nc_view4,
            "inference_delta_times_norm": x["inference_delta_times"] / nc_view4,
            "intensity_evaluation_times_norm": x["intensity_evaluation_times"] / nc_view3,
        }
        return norm_constants, normalized

    def _compute_delta_times_inplace(self, x: dict, type="context"):
        """Compute delta times more efficiently"""
        B, P, L = x[f"{type}_event_times"].shape[:3]
        # Pre-allocate tensor with zeros for the first event
        delta_times = torch.zeros(B, P, L, 1, device=x[f"{type}_event_times"].device, dtype=x[f"{type}_event_times"].dtype)
        delta_times[:, :, 1:] = x[f"{type}_event_times"][:, :, 1:] - x[f"{type}_event_times"][:, :, :-1]
        x[f"{type}_delta_times"] = delta_times

    def _encode_observations_optimized(self, x: dict, type="context") -> Tensor:
        """Optimized observation encoding using cached one-hot encodings"""
        # Use normalized event times if available
        et_key = f"{type}_event_times_norm" if (self.normalize_times and f"{type}_event_times_norm" in x) else f"{type}_event_times"
        obs_grid_normalized = x[et_key]
        B, P, L = obs_grid_normalized.shape[:3]

        time_enc = self.time_encoder(obs_grid_normalized)
        # Use normalized delta times if available
        dt_key = f"{type}_delta_times_norm" if (self.normalize_times and f"{type}_delta_times_norm" in x) else f"{type}_delta_times"
        delta_time_enc = self.delta_time_encoder(x[dt_key])

        # Create padding mask before encoding marks
        positions = torch.arange(L, device=self.device).unsqueeze(0)
        seq_lengths_flat = x[f"{type}_seq_lengths"].view(B * P)
        key_padding_mask = positions >= seq_lengths_flat.unsqueeze(1)
        key_padding_mask_flat = key_padding_mask.view(-1)

        # More efficient mark encoding using cached one-hot matrix
        event_types_flat = x[f"{type}_event_types"].reshape(-1).long()
        # Clamp event types to avoid out-of-bounds access with padding tokens.
        # The padded values will be zeroed out later using the mask.
        event_types_flat[key_padding_mask_flat] = 0

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
                # No causal mask for context sequences; keep padding mask only
                src_key_padding_mask=key_padding_mask,  # 2D key padding mask
            )
        elif type == "inference":
            # Bypass any TransformerEncoder: use token embeddings directly for the decoder
            # We still zero out padded tokens and rely on the decoder's causal + padding masks
            h = path.view(B * P, L, -1)
        else:
            raise ValueError(f"Invalid type: {type}")

        return h.view(B, P, L, -1)

    # def _functional_attention_encoder_optimized(self, queries: Tensor, keys_values: Tensor, attention_mask: Tensor) -> Tensor:
    #     """
    #     Performs functional attention for a single inference target.
    #     This version formats its output to be compatible with the unmodified AttentionOperator.

    #     Args:
    #         queries (Tensor): Query embeddings from the trunk net.
    #                         Shape: [B, M, L_inference, D]
    #         keys_values (Tensor): Key/Value embeddings from context + one inference path.
    #                             Shape: [B, P_context + 1, L, D]
    #         attention_mask (Tensor): The corresponding dynamic attention mask.
    #                                 Shape: [B, L_inference, P_context + 1, L]

    #     Returns:
    #         Tensor: The final time-dependent encodings after attention.
    #                 Shape: [B, M, L_inference, D]
    #     """
    #     B, M, L_inference, D = queries.shape
    #     _, P_relevant, L, _ = keys_values.shape

    #     # Reshape for batched attention. Each (L_inference) query gets its own mask.
    #     # We treat the L_inference dimension as part of the batch.

    #     # Reshape queries: [B, M, L_inf, D] -> [B * L_inf, M, D]
    #     queries_reshaped = queries.permute(0, 2, 1, 3).reshape(B * L_inference, M, D)

    #     # Expand and reshape keys/values: [B, P_rel, L, D] -> [B, 1, P_rel*L, D] -> [B*L_inf, P_rel*L, D]
    #     keys_values_expanded = keys_values.unsqueeze(1).expand(-1, L_inference, -1, -1, -1)
    #     keys_values_reshaped = keys_values_expanded.reshape(B * L_inference, P_relevant * L, D)

    #     # Reshape mask: [B, L_inf, P_rel, L] -> [B * L_inf, P_rel * L]
    #     mask_reshaped = attention_mask.reshape(B * L_inference, P_relevant * L)

    #     # Shape: [B*L_inf, P_rel*L, D] -> [B*L_inf, P_rel*L, 1, D]
    #     keys_values_4d = keys_values_reshaped.unsqueeze(2)

    #     # The padding mask is already prepared in the correct 3D shape for the operator's internal layers.
    #     # Shape: [B*L_inf, P_rel*L, 1]
    #     padding_mask_3d = mask_reshaped.unsqueeze(-1)

    #     # Apply functional attention with the correctly shaped tensors
    #     result = self.functional_attention(
    #         queries_reshaped,
    #         keys_values_4d,  # Pass the 4D tensor here
    #         observations_padding_mask=padding_mask_3d,
    #     )  # Result shape: [B * L_inf, M, D]

    #     # Reshape back to original format
    #     # [B * L_inf, M, D] -> [B, L_inf, M, D] -> [B, M, L_inf, D]
    #     return result.reshape(B, L_inference, M, D).permute(0, 2, 1, 3)

    def _denormalize_output(self, out: dict, norm_constants: Tensor) -> None:
        """
        Adjust output intensity values back to original time scale using normalization constants.
        """
        out["predicted_intensity_values"] = out["predicted_intensity_values"] / norm_constants.view(-1, 1, 1, 1)
        if "target_intensity_values" in out:
            out["target_intensity_values"] = out["target_intensity_values"] / norm_constants.view(-1, 1, 1, 1)
        if "log_predicted_intensity_values_var" in out:
            out["log_predicted_intensity_values_var"] = out["log_predicted_intensity_values_var"] - torch.log(norm_constants).view(
                -1, 1, 1, 1
            )

    def _smape(self, predicted_intensity_values: Tensor, target_intensity_values: Tensor) -> Tensor:
        """Symmetric Mean Absolute Percentage Error.

        Any NaN/inf contributions (e.g., from NaN targets) are replaced with 2.0
        before averaging so the loss never returns NaN.
        """
        numerator = 2.0 * torch.abs(predicted_intensity_values - target_intensity_values)
        denominator = torch.abs(predicted_intensity_values) + torch.abs(target_intensity_values) + 1e-8
        smape_values = numerator / denominator
        smape_values = torch.nan_to_num(smape_values, nan=2.0, posinf=2.0, neginf=2.0)
        return torch.mean(smape_values)

    def _smape_masked(self, y_pred: Tensor, y_true: Tensor, valid_mask: Tensor) -> Tensor:
        """Masked sMAPE over [B, M, P, L].

        Args:
            y_pred: Tensor [B, M, P, L]
            y_true: Tensor [B, M, P, L]
            valid_mask: Bool tensor [B, P, L] with True for valid events.
        """
        # Broadcast mask across mark dim
        mask = valid_mask.unsqueeze(1).expand_as(y_pred)
        numerator = 2.0 * torch.abs(y_pred - y_true)
        denominator = torch.abs(y_pred) + torch.abs(y_true) + 1e-8
        smape_values = numerator / denominator
        smape_values = torch.nan_to_num(smape_values, nan=2.0, posinf=2.0, neginf=2.0)
        smape_values = smape_values * mask
        denom = mask.sum().clamp(min=1)
        return smape_values.sum() / denom

    def _relative_spike(self, mu: Tensor, alpha: Tensor, epsilon: float = 1e-8) -> Tensor:
        """Relative spike regularization term averaged over all elements.

        L_relative_spike = mean( 2 * |alpha - mu| / (|alpha| + |mu| + eps) )
        """
        diff = torch.abs(alpha - mu)
        denom = torch.abs(alpha) + torch.abs(mu) + epsilon
        rel = 2.0 * diff / denom
        return rel.mean()

    def _nll_loss(
        self,
        intensity_fn: "PiecewiseHawkesIntensity",
        event_times: Tensor,
        event_types: Tensor,
        seq_lengths: Tensor,
        apply_log_c_correction: bool = False,
    ) -> Tensor:
        """
        Negative log-likelihood loss normalized by number of events.
        """
        B, P, L = event_times.shape

        # Closed-form integral of intensity over [0, T]
        # Build inter-event gaps Δ_i with Δ_0 = t_1 - 0 and Δ_i = t_{i+1} - t_i for i ≥ 1.
        # We integrate up to the last valid event time per path, so the terminal gap Δ_N is 0.
        t = event_times.clone().squeeze(-1)  # [B, P, L]
        deltas = torch.zeros_like(t)
        deltas[:, :, 0] = t[:, :, 0]
        if t.size(2) > 1:
            deltas[:, :, 1:] = t[:, :, 1:] - t[:, :, :-1]

        # Mask out padded intervals (i >= seq_length)
        positions = torch.arange(L, device=self.device).view(1, 1, L)
        valid_interval_mask = positions < seq_lengths.unsqueeze(2)  # [B, P, L]
        deltas = deltas * valid_interval_mask

        # Broadcast Δ over marks and apply the closed-form integral per interval and mark
        delta_expanded = deltas.unsqueeze(1)  # [B, 1, P, L]
        mu = intensity_fn.mu  # [B, M, P, L]
        alpha = intensity_fn.alpha  # [B, M, P, L]
        beta = intensity_fn.beta  # [B, M, P, L]
        eps = 1e-8
        integral_terms = mu * delta_expanded + (alpha - mu) / (beta + eps) * (1.0 - torch.exp(-beta * delta_expanded))  # [B, M, P, L]

        # Sum over intervals to obtain ∫ λ dt for each mark and path
        integral_per_mark_path = integral_terms.sum(dim=3)  # [B, M, P]
        integral_sum_per_path = integral_per_mark_path.sum(dim=1)  # [B, P]

        # To align with EasyTPP, we exclude the first event from the log-likelihood's summation term.
        # The model is evaluated on its ability to predict events t_1, ..., t_N given t_0.
        event_times_for_ll = event_times[:, :, 1:]
        event_types_for_ll = event_types[:, :, 1:]

        # Adjust sequence lengths for the sliced view. A sequence of length L has L-1 events to evaluate.
        seq_lengths_for_ll = (seq_lengths - 1).clamp(min=0)
        L_eval = L - 1

        # --- 1. Calculate NLL' Summation Term in the Normalized Time Domain ---

        # Intensity and its log at event times (from the second event onwards)
        intensity_at_events = intensity_fn.evaluate(event_times_for_ll, normalized_times=True)
        log_intensity_at_events = torch.log(intensity_at_events + 1e-9)  # Add epsilon for stability

        # Gather the log-intensity for the specific event type (mark) that occurred
        # Clamp event types to avoid out-of-bounds access with padding tokens.
        num_marks = log_intensity_at_events.shape[1]
        event_types_for_ll_clamped = torch.clamp(event_types_for_ll, 0, num_marks - 1)
        type_idx = event_types_for_ll_clamped.unsqueeze(1).expand(-1, 1, -1, -1)
        log_lambda_at_event_m = torch.gather(log_intensity_at_events, 1, type_idx).squeeze(1)

        # Create a mask for all valid (non-padded) events in the SLICED sequences
        positions = torch.arange(L_eval, device=self.device).view(1, 1, L_eval)
        valid_mask = positions < seq_lengths_for_ll.unsqueeze(2)

        # Sum of log-intensities for all valid events (Σ log(λ'(t'_i))) from i=1 to N
        log_ll_per_path = (log_lambda_at_event_m * valid_mask).sum(dim=2)

        # NLL' = ∫ λ'(t') dt' - Σ log(λ'(t'_i))
        nll_prime_per_path = integral_sum_per_path - log_ll_per_path
        total_nll_prime = nll_prime_per_path.sum()

        total_nll = total_nll_prime  # Initialize with the normalized-scale NLL

        # --- 2. Apply Correction for Time Scaling (if requested) ---
        # The true NLL is related to the normalized-scale NLL (NLL') by:
        # NLL = NLL' + N * log(c), where N is the event count and c is the normalization constant.
        # This correction is crucial for evaluation but should be omitted during training
        # to keep the loss scale consistent with the normalized model parameters.
        if self.normalize_times and apply_log_c_correction and intensity_fn.norm_constants is not None:
            # Number of events per batch item (N_b), using the corrected event count (excluding the first event)
            events_per_batch_item = valid_mask.sum(dim=[1, 2])  # Shape: [B]

            # Normalization constants per batch item (c_b)
            norm_constants = intensity_fn.norm_constants  # Shape: [B]

            # Correction term: sum over batch { N_b * log(c_b) }
            nll_correction = (events_per_batch_item * torch.log(norm_constants + 1e-9)).sum()

            total_nll = total_nll_prime + nll_correction

        # --- 3. Normalize by Event Count ---
        # The total number of events is now based on the sliced sequences
        total_events = valid_mask.sum()
        return total_nll / (total_events + 1e-8)

    def loss(
        self,
        intensity_fn: "PiecewiseHawkesIntensity",
        predicted_intensity_values: Tensor,
        target_intensity_values: Tensor,
        event_times: Tensor,
        event_types: Tensor,
        seq_lengths: Tensor,
        schedulers: dict = None,
        step: int = None,
        mu_targets_at_events: Tensor | None = None,
        lambda_post_targets_at_events: Tensor | None = None,
        valid_event_mask: Tensor | None = None,
    ) -> dict:
        """Hybrid loss combining negative log-likelihood and symmetric mean absolute percentage error.

        L_total = w_nll * L_NLL + w_smape * L_sMAPE + w_mu * L_mu + w_alpha * L_alpha
        """
        # --- 1. Symmetric Mean Absolute Percentage Error ---
        smape_loss = self._smape(predicted_intensity_values, target_intensity_values)

        # --- 2. Negative Log-Likelihood Loss ---
        nll_loss = self._nll_loss(intensity_fn, event_times, event_types, seq_lengths)

        # --- 2.25 Decomposed supervision on μ and α at event times ---
        mu_loss = torch.tensor(0.0, device=predicted_intensity_values.device)
        alpha_loss = torch.tensor(0.0, device=predicted_intensity_values.device)
        if (mu_targets_at_events is not None) and (lambda_post_targets_at_events is not None) and (valid_event_mask is not None):
            mu_loss = self._smape_masked(intensity_fn.mu, mu_targets_at_events, valid_event_mask)
            alpha_loss = self._smape_masked(intensity_fn.alpha, lambda_post_targets_at_events, valid_event_mask)

        # --- 2.5 Relative spike regularization ---
        relative_spike_loss = self._relative_spike(intensity_fn.mu, intensity_fn.alpha)

        # --- 3. Hybrid weighting of sMAPE and NLL ---
        total_loss = (
            self.loss_weights["nll"] * nll_loss
            + self.loss_weights["smape"] * smape_loss
            + self.loss_weights.get("mu", 0.0) * mu_loss
            + self.loss_weights.get("alpha", 0.0) * alpha_loss
            + self.loss_weights.get("relative_spike", 0.1) * relative_spike_loss
        )

        mae_loss = torch.mean(torch.abs(predicted_intensity_values - target_intensity_values))

        # Prepare a logging-friendly dictionary: tensors -> Python floats
        losses_out = {
            "loss": total_loss,  # keep tensor for downstream back-prop accounting
            "nll_loss": nll_loss.detach().item(),
            "smape_loss": smape_loss.detach().item(),
            "mu_smape_loss": mu_loss.detach().item() if isinstance(mu_loss, Tensor) else float(mu_loss),
            "alpha_smape_loss": alpha_loss.detach().item() if isinstance(alpha_loss, Tensor) else float(alpha_loss),
            "relative_spike_loss": relative_spike_loss.detach().item(),
            "mae_loss": mae_loss.detach().item(),
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
        inference_time_offsets: Tensor | None = None,
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
            inference_time_offsets=inference_time_offsets,
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
        inference_time_offsets: Tensor | None = None,
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
            inference_time_offsets (Tensor): The time offsets for the inference paths (since we shifted events to zero).
                Shape: [B, P].

        Returns:
            Tensor: The computed target intensity values, scaled for the normalized time domain.
                    Shape: [B, D, P, L_eval], where D is the number of marks.
        """
        device = intensity_evaluation_times.device
        B, P, L_eval = intensity_evaluation_times.shape
        _, _, L_hist, _ = inference_event_times.shape
        D = num_marks

        # The input times are normalized. We must denormalize them to use with the
        # original kernel and base intensity functions.
        norm_constants_eval = norm_constants.view(B, 1, 1)
        norm_constants_hist = norm_constants.view(B, 1, 1, 1)

        intensity_evaluation_times_orig = intensity_evaluation_times * norm_constants_eval
        inference_event_times_orig = inference_event_times * norm_constants_hist

        # If offsets are provided (events originally shifted to start at 0), add them back
        if inference_time_offsets is not None:
            # Add to evaluation times [B, P, L_eval]
            intensity_evaluation_times_orig = intensity_evaluation_times_orig + inference_time_offsets.unsqueeze(-1)
            # Add to historical event times [B, P, L_hist, 1]
            inference_event_times_orig = inference_event_times_orig + inference_time_offsets.unsqueeze(-1).unsqueeze(-1)

        # Ensure tensors are 3D for easier processing by removing the trailing dimension
        event_times = inference_event_times_orig.squeeze(-1)  # Use original times
        event_types = inference_event_types.squeeze(-1)

        # The final output tensor, matching the model's prediction shape [B, M, P, L_eval]
        total_intensity = torch.zeros(B, D, P, L_eval, device=device)

        # Loop over batch items because kernel/base functions are heterogeneous Python objects.
        for b in range(B):
            actual_marks_b = len(base_intensity_functions_list[b])

            eval_times_b = intensity_evaluation_times_orig[b]  # [P, L_eval] (original times)
            hist_times_b = event_times[b]  # [P, L_hist] (original times)
            hist_types_b = event_types[b]  # [P, L_hist]
            seq_lengths_b = inference_seq_lengths[b]  # [P]

            # Vectorized implementation
            # Baseline: μ_i(t) per mark i at all evaluation times
            for i in range(actual_marks_b):
                mu_func_b_i = base_intensity_functions_list[b][i]
                total_intensity[b, i, :, :] = mu_func_b_i(eval_times_b)

            # Precompute deltas and masks once per batch element
            delta_t_b = eval_times_b.unsqueeze(-1) - hist_times_b.unsqueeze(-2)  # [P, L_eval, L_hist]
            hist_indices = torch.arange(L_hist, device=device).view(1, L_hist)
            padding_mask_b = hist_indices < seq_lengths_b.unsqueeze(-1)  # [P, L_hist]
            valid_history_mask_b = (delta_t_b > 1e-9) & padding_mask_b.unsqueeze(1)  # [P, L_eval, L_hist]

            # Accumulate kernel influences vectorized over P and L_eval
            for j in range(actual_marks_b):
                source_mark_mask_b = hist_types_b == j  # [P, L_hist]
                final_mask_b = valid_history_mask_b & source_mark_mask_b.unsqueeze(1)

                if not torch.any(final_mask_b):
                    continue

                deltas = delta_t_b  # [P, L_eval, L_hist]
                for i in range(actual_marks_b):
                    phi_func_b_ij = kernel_functions_list[b][i][j]
                    kij = phi_func_b_ij(deltas)
                    kij = torch.where(final_mask_b, kij, torch.zeros_like(kij))
                    summed = kij.sum(dim=-1)  # [P, L_eval]
                    total_intensity[b, i, :, :] += summed

        # Apply non-negativity constraint
        target_intensity_values_orig = torch.relu(total_intensity)

        # Re-normalize the final intensity to match the model's output space (λ'(t') = c * λ(t))
        norm_constants_final = norm_constants.view(B, 1, 1, 1)
        target_intensity_values_normalized = target_intensity_values_orig * norm_constants_final

        return target_intensity_values_normalized

    def compute_decomposed_targets_at_events(
        self,
        kernel_functions_list: list,
        base_intensity_functions_list: list,
        inference_event_times: Tensor,
        inference_event_types: Tensor,
        inference_seq_lengths: Tensor,
        norm_constants: Tensor,
        num_marks: int,
        inference_time_offsets: Tensor | None = None,
        post_epsilon: float = 1e-6,
        pre_epsilon: float = 1e-6,
    ):
        """Compute μ*(t_i), λ*(t_i^-) and λ*(t_i^+) at event times in normalized intensity scale.

        `inference_event_times` are expected in the normalized time domain with shape [B, P, L, 1].
        """
        device = inference_event_times.device
        B, P, L, _ = inference_event_times.shape
        D = num_marks

        # Denormalize to query python callables
        times_orig = inference_event_times * norm_constants.view(B, 1, 1, 1)
        if inference_time_offsets is not None:
            times_orig = times_orig + inference_time_offsets.view(B, P, 1, 1)

        # μ*(t_i)
        mu_star = torch.zeros(B, D, P, L, device=device)
        for b in range(B):
            actual_marks_b = len(base_intensity_functions_list[b])
            t_b = times_orig[b, :, :, 0]  # [P, L]
            for i in range(actual_marks_b):
                mu_func = base_intensity_functions_list[b][i]
                mu_star[b, i, :, :] = mu_func(t_b)
        # Scale to normalized intensity
        mu_star = mu_star * norm_constants.view(B, 1, 1, 1)

        # λ*(t_i^-) evaluated safely at t_i - ε (clamped to 0)
        eps_pre = torch.full((B, P, L), pre_epsilon, device=device)
        lambda_pre = self.compute_target_intensity_values(
            kernel_functions_list,
            base_intensity_functions_list,
            torch.clamp(inference_event_times.squeeze(-1) - eps_pre, min=0.0),
            inference_event_times,
            inference_event_types,
            inference_seq_lengths,
            norm_constants,
            D,
            inference_time_offsets=inference_time_offsets,
        )

        # λ*(t_i^+) approximated with t_i + ε
        eps = torch.full((B, P, L), post_epsilon, device=device)
        lambda_post = self.compute_target_intensity_values(
            kernel_functions_list,
            base_intensity_functions_list,
            (inference_event_times.squeeze(-1) + eps),
            inference_event_times,
            inference_event_types,
            inference_seq_lengths,
            norm_constants,
            D,
            inference_time_offsets=inference_time_offsets,
        )

        return mu_star, lambda_pre, lambda_post

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

    def predict_next_event(self, batch: dict) -> Dict[str, Tensor]:
        """
        Perform next-event prediction for a batch of sequences from the test set.

        This is an OPTIMIZED version that avoids Python loops by batching all
        prefixes of all paths into a single forward pass.

        Args:
            batch (dict): A dictionary containing the input tensors from the EasyTPP dataloader.
                          Expected keys:
                          - "event_times": Tensor of shape [B, P, L, 1]
                          - "event_types": Tensor of shape [B, P, L, 1]
                          - "seq_lengths": Tensor of shape [B, P]
                          Where B is batch size (usually 1 for this task), P is the number of paths,
                          and L is the sequence length.

        Returns:
            dict: A dictionary containing the aggregated predictions for all paths.
                  - "predicted_event_dtimes": Predicted inter-event times [P, L-1].
                  - "predicted_event_types": Predicted event types [P, L-1].
        """
        B, P, L, _ = batch["event_times"].shape
        device = self.device

        if B > 1:
            self.logger.warning(f"Prediction is designed for B=1, but got B={B}. Processing B=0 only.")
            # Slicing to handle B>1 case, though the logic below assumes B=1 for simplicity
            for key in batch:
                batch[key] = batch[key][:1]

        # We will make predictions for prefixes of length 1 to L-1.
        prefix_lengths = torch.arange(1, L, device=device)
        num_prefixes = L - 1

        # 1. Construct the batch of all prefixes for all paths
        # The new batch dimension will be P * num_prefixes

        # Expand paths and prefixes to create all combinations
        # Paths: [P, L, 1] -> [P, 1, L, 1] -> [P, num_prefixes, L, 1]
        # Prefixes: [num_prefixes] -> [1, num_prefixes, 1, 1]

        # Create a mask to select elements for each prefix
        # Shape: [num_prefixes, L] where mask[i, j] is true if j < prefix_lengths[i]
        positions = torch.arange(L, device=device).unsqueeze(0)
        prefix_mask = positions < prefix_lengths.unsqueeze(1)  # [num_prefixes, L]

        # Expand data and mask to create the inference batch
        # New shape: [P, num_prefixes, L, 1]
        inference_times_expanded = batch["event_times"].squeeze(0).unsqueeze(1).expand(-1, num_prefixes, -1, -1)
        inference_types_expanded = batch["event_types"].squeeze(0).unsqueeze(1).expand(-1, num_prefixes, -1, -1)

        # Apply the mask. We use 0 as a padding value.
        # Shape: [P * num_prefixes, L, 1]
        inference_times_batched = (inference_times_expanded * prefix_mask.view(1, num_prefixes, L, 1)).reshape(P * num_prefixes, L, 1)
        inference_types_batched = (inference_types_expanded * prefix_mask.view(1, num_prefixes, L, 1)).reshape(P * num_prefixes, L, 1)
        inference_lengths_batched = prefix_lengths.repeat(P)  # [P * num_prefixes]

        # 2. Prepare the context for each prefix
        # For each of the `num_prefixes` for a path `p`, the context is the same: all other paths.
        context_indices = [torch.arange(p, device=device) for p in range(P)]
        context_masks = [(idx != p_idx) for p_idx, idx in enumerate(context_indices)]

        context_times_list = []
        context_types_list = []
        context_lengths_list = []

        for p_idx in range(P):
            mask = context_masks[p_idx]
            # Context for path p_idx is all other paths
            # Shape: [1, P-1, L, 1]
            ctx_times = batch["event_times"][:, mask, ...]
            ctx_types = batch["event_types"][:, mask, ...]
            ctx_lengths = batch["seq_lengths"][:, mask]

            # Repeat this context for each prefix of path p_idx
            # Shape: [num_prefixes, P-1, L, 1]
            context_times_list.append(ctx_times.expand(num_prefixes, -1, -1, -1))
            context_types_list.append(ctx_types.expand(num_prefixes, -1, -1, -1))
            context_lengths_list.append(ctx_lengths.expand(num_prefixes, -1))

        # Concatenate into a single context batch
        # Shape: [P * num_prefixes, P-1, L, 1]
        context_times_batched = torch.cat(context_times_list, dim=0)
        context_types_batched = torch.cat(context_types_list, dim=0)
        context_lengths_batched = torch.cat(context_lengths_list, dim=0)

        # 3. Assemble the final input dictionary for the single forward pass
        x_batched = {
            "context_event_times": context_times_batched,
            "context_event_types": context_types_batched,
            "context_seq_lengths": context_lengths_batched,
            "inference_event_times": inference_times_batched.unsqueeze(1),  # Add P_inf=1 dim
            "inference_event_types": inference_types_batched.unsqueeze(1),
            "inference_seq_lengths": inference_lengths_batched.unsqueeze(1),
            "intensity_evaluation_times": torch.zeros(P * num_prefixes, 1, 1, device=device),
        }

        # 4. Single, batched forward pass
        with torch.no_grad():
            model_out = self.forward(x_batched)

        # intensity_obj is now a "batched" object, containing parameters for all prefixes
        # mu/alpha/beta shapes: [P*num_prefixes, M, 1, L]
        intensity_obj = model_out["intensity_function"]

        # 5. Batched prediction using the sampler

        # Sampler needs history times and deltas
        hist_times = x_batched["inference_event_times"].squeeze(1).squeeze(-1)  # [P*num_prefixes, L]
        hist_dtimes = torch.zeros_like(hist_times)
        hist_dtimes[:, 1:] = hist_times[:, 1:] - hist_times[:, :-1]
        hist_types = x_batched["inference_event_types"].squeeze(1).squeeze(-1)

        # Create the wrapper for the batched intensity object
        def intensity_fn_for_sampler(query_times, history_times_ignored):
            # query_times shape: [P*num_prefixes, 1, num_samples]
            b_size, _, n_samples = query_times.shape
            query_times_reshaped = query_times.reshape(b_size, 1, n_samples)
            intensity_per_mark = intensity_obj.evaluate(query_times_reshaped)  # [B', M, 1, n_samples]
            total_intensity = intensity_per_mark.sum(dim=1)  # [B', 1, n_samples]
            return total_intensity

        # The sampler needs to be called for each prefix length. We can't fully batch this part
        # because the sampler itself isn't designed to handle ragged histories efficiently.
        # However, we can call it on batches of the *same prefix length*.

        all_dtime_preds = []
        all_type_preds = []

        for i in range(num_prefixes):
            # Select the batch indices corresponding to the current prefix length (i+1)
            # These are indices 0, num_prefixes, 2*num_prefixes, ... for i=0
            # and 1, num_prefixes+1, ... for i=1, etc.
            batch_indices = torch.arange(i, P * num_prefixes, num_prefixes, device=device)

            current_len = i + 1

            # We need to provide the sampler with a view of the history up to the current length
            # The sampler expects [Batch, SeqLen]
            time_seq_for_sampler = hist_times[batch_indices, :current_len]
            dtime_seq_for_sampler = hist_dtimes[batch_indices, :current_len]
            type_seq_for_sampler = hist_types[batch_indices, :current_len]

            # The intensity object needs to be sliced for the current batch
            # Preserve the normalisation constants for the sliced object (if any)
            if hasattr(intensity_obj, "norm_constants") and intensity_obj.norm_constants is not None:
                sliced_norm_consts = intensity_obj.norm_constants[batch_indices]
            else:
                sliced_norm_consts = None

            sliced_intensity_obj = PiecewiseHawkesIntensity(
                event_times=intensity_obj.event_times[batch_indices, :, :current_len].squeeze(1),
                mu=intensity_obj.mu[batch_indices, :, :, :current_len].squeeze(2),
                alpha=intensity_obj.alpha[batch_indices, :, :, :current_len].squeeze(2),
                beta=intensity_obj.beta[batch_indices, :, :, :current_len].squeeze(2),
                norm_constants=sliced_norm_consts,
            )

            def sliced_intensity_fn(query_times, hist_ignored):
                b, _, n_s = query_times.shape
                q_reshaped = query_times.view(b, 1, n_s)
                intensity_per_mark = sliced_intensity_obj.evaluate(q_reshaped)
                return intensity_per_mark.sum(dim=1)

            if getattr(self.event_sampler, "sampling_method", "thinning") == "inverse_transform":
                # Use closed-form inverse transform on the sliced intensity object
                accepted_dtimes, weights = self.event_sampler.draw_next_time_one_step_inverse_transform(
                    sliced_intensity_obj, compute_last_step_only=True
                )
                # The inverse sampler may flatten the (B*P) dimension; reshape back to [P, 1, S]
                if accepted_dtimes.dim() == 3 and accepted_dtimes.shape[0] == time_seq_for_sampler.shape[0]:
                    pass
                else:
                    # best-effort: treat leading dim as batch of paths
                    accepted_dtimes = accepted_dtimes
            else:
                accepted_dtimes, weights = self.event_sampler.draw_next_time_one_step(
                    time_seq=time_seq_for_sampler,
                    time_delta_seq=dtime_seq_for_sampler,
                    event_seq=type_seq_for_sampler,
                    intensity_fn=sliced_intensity_fn,
                    compute_last_step_only=True,
                )  # [P, 1, num_samples]

            # Convert absolute sampled times to inter-event times (delta t)
            t_last_tensor = time_seq_for_sampler[:, -1:].unsqueeze(-1)  # [P, 1, 1]
            delta_samples = accepted_dtimes - t_last_tensor
            delta_samples = torch.clamp(delta_samples, min=0.0)  # Numerical safety

            dtime_pred = torch.sum(delta_samples * weights, dim=-1).squeeze(-1)

            t_last = time_seq_for_sampler[:, -1]  # [P]
            predicted_time = t_last + dtime_pred  # [P]

            intensities_at_pred_time = sliced_intensity_obj.evaluate(predicted_time.view(P, 1, 1))  # [P, M, 1, 1]
            type_pred = torch.argmax(intensities_at_pred_time.squeeze(), dim=1)  # [P]

            all_dtime_preds.append(dtime_pred)
            all_type_preds.append(type_pred)

        # Transpose and stack to get shape [P, L-1]
        final_dtimes = torch.stack(all_dtime_preds, dim=1)
        final_types = torch.stack(all_type_preds, dim=1)

        return {
            "predicted_event_dtimes": final_dtimes,
            "predicted_event_types": final_types,
        }


ModelFactory.register(FIMHawkesConfig.model_type, FIMHawkes)
AutoConfig.register(FIMHawkesConfig.model_type, FIMHawkesConfig)
AutoModel.register(FIMHawkesConfig, FIMHawkes)
