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


ARTIFICIAL_NORM_FACTOR = 1000


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
        kernel_value_var_decoder: dict = None,
        base_intensity_decoder: dict = None,
        base_intensity_var_decoder: dict = None,
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
        self.ts_encoder = ts_encoder
        self.time_dependent_functional_attention = time_dependent_functional_attention
        self.static_functional_attention = static_functional_attention
        self.static_functional_attention_learnable_query = static_functional_attention_learnable_query
        self.kernel_value_decoder = kernel_value_decoder
        self.kernel_value_var_decoder = kernel_value_var_decoder
        self.base_intensity_decoder = base_intensity_decoder
        self.base_intensity_var_decoder = base_intensity_var_decoder
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
        kernel_value_var_decoder = copy.deepcopy(self.config.kernel_value_var_decoder)
        base_intensity_decoder = copy.deepcopy(self.config.base_intensity_decoder)
        base_intensity_var_decoder = copy.deepcopy(self.config.base_intensity_var_decoder)
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
        self.static_functional_attention_learnable_query = create_class_instance(
            static_functional_attention_learnable_query.pop("name"), static_functional_attention_learnable_query
        )

        kernel_value_decoder["in_features"] = self.static_functional_attention_learnable_query.out_features
        kernel_value_decoder["out_features"] = 1
        self.kernel_value_decoder = create_class_instance(kernel_value_decoder.pop("name"), kernel_value_decoder)

        kernel_value_var_decoder["in_features"] = self.static_functional_attention_learnable_query.out_features
        kernel_value_var_decoder["out_features"] = 1
        self.kernel_value_var_decoder = create_class_instance(kernel_value_var_decoder.pop("name"), kernel_value_var_decoder)

        base_intensity_decoder["in_features"] = self.static_functional_attention_learnable_query.out_features
        base_intensity_decoder["out_features"] = 1
        self.base_intensity_decoder = create_class_instance(base_intensity_decoder.pop("name"), base_intensity_decoder)

        base_intensity_var_decoder["in_features"] = self.static_functional_attention_learnable_query.out_features
        base_intensity_var_decoder["out_features"] = 1
        self.base_intensity_var_decoder = create_class_instance(base_intensity_var_decoder.pop("name"), base_intensity_var_decoder)
        if self.config.thinning is not None:
            self.event_sampler = EventSampler(**self.config.thinning)
        else:
            self.event_sampler = EventSampler(num_sample=1, num_exp=500, over_sample_rate=5, num_samples_boundary=5, dtime_max=5)

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
                - "kernel_eval_var_values": Tensor representing the variances of the predicted kernel values. [B, M, L_kernel]
                - "baseline_intensity": Tensor representing the predicted baseline intensity. [B, M]
                - "baseline_intensity_var": Tensor representing the variance of the predicted baseline intensity. [B, M]
                - "losses" (optional): Tensor representing the calculated losses, if the required keys are present in `x`.
        """
        obs_grid = x["event_times"]
        if obs_grid.dim() == 2:
            # for datasets that have only one process and many paths (e.g. easytpp)
            obs_grid = obs_grid.unsqueeze(0).unsqueeze(-1)
            x["event_times"] = obs_grid
            x["event_types"] = x["event_types"].unsqueeze(0).unsqueeze(-1)
            x["seq_lengths"] = x["seq_lengths"].unsqueeze(0)
        B, P, L = obs_grid.shape[:3]

        if "seq_lengths" not in x:
            x["seq_lengths"] = torch.full((B, P), L, device=self.device)

        x["delta_times"] = obs_grid[:, :, 1:] - obs_grid[:, :, :-1]
        # Add a delta time of 0 for the first event
        x["delta_times"] = torch.cat([torch.zeros_like(x["delta_times"][:, :, :1]), x["delta_times"]], dim=2)

        norm_constants = self._normalize_input_times(x)

        sequence_encodings = self._encode_observations(x)  # [B, P, L, D]

        observations_padding_mask = self._generate_padding_mask(x["seq_lengths"], L).unsqueeze(-1)  # [B, P, L, 1]
        time_dependent_encodings = self._time_dependent_encoder(
            x, sequence_encodings, observations_padding_mask=observations_padding_mask
        )  # [B, M, L_kernel, D]

        static_encodings = self._static_encoder(x, sequence_encodings, observations_padding_mask=observations_padding_mask)  # [B, M, D]

        predicted_kernel_values = self._kernel_value_decoder(time_dependent_encodings)  # [B, M, L_kernel]

        log_predicted_kernel_values_var = self._kernel_value_var_decoder(
            time_dependent_encodings.clone().detach()
        )  # [B, M, L_kernel] # Do not backpropagate through this

        predicted_base_intensity = torch.exp(self._base_intensity_decoder(static_encodings))  # [B, M]

        out = {
            "predicted_kernel_values": predicted_kernel_values,
            "log_predicted_kernel_values_var": log_predicted_kernel_values_var,
            "predicted_base_intensity": predicted_base_intensity,
        }

        if "base_intensities" in x and "kernel_evaluations" in x:
            out["losses"] = self.loss(
                out["predicted_kernel_values"],
                out["log_predicted_kernel_values_var"],
                out["predicted_base_intensity"],
                x["kernel_evaluations"],
                x["base_intensities"],
                x["kernel_grids"],
                schedulers,
                step,
            )

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

    def _kernel_value_decoder(self, time_dependent_path_summary: Tensor) -> Tensor:
        B, M, L_kernel, D_3 = time_dependent_path_summary.shape
        time_dependent_path_summary = time_dependent_path_summary.view(B * M * L_kernel, D_3)
        h = self.kernel_value_decoder(time_dependent_path_summary)
        return h.view(B, M, L_kernel)

    def _kernel_value_var_decoder(self, time_dependent_path_summary: Tensor) -> Tensor:
        B, M, L_kernel, D_3 = time_dependent_path_summary.shape
        time_dependent_path_summary = time_dependent_path_summary.view(B * M * L_kernel, D_3)
        h = self.kernel_value_var_decoder(time_dependent_path_summary)
        return h.view(B, M, L_kernel)

    def _base_intensity_decoder(self, static_path_summary: Tensor) -> Tensor:
        B, M, D_2 = static_path_summary.shape
        h = self.base_intensity_decoder(static_path_summary)
        return h.view(-1, M)

    def _base_intensity_var_decoder(self, static_path_summary: Tensor) -> Tensor:
        B, M, D_2 = static_path_summary.shape
        h = self.base_intensity_var_decoder(static_path_summary)
        return h.view(-1, M)

    def _normalize_input_times(self, x: dict) -> dict:
        batch_indices = (
            torch.arange(x["event_times"].size(0), device=x["event_times"].device).view(-1, 1).expand(-1, x["event_times"].size(1))
        )
        path_indices = (
            torch.arange(x["event_times"].size(1), device=x["event_times"].device).view(1, -1).expand(x["event_times"].size(0), -1)
        )
        if self.normalize_by_max_time:
            max_times = x["event_times"][batch_indices, path_indices, x["seq_lengths"] - 1]
            norm_constants = max_times.amax(dim=[1, 2])
        else:
            masked_delta_times = x["delta_times"].clone()
            B, P, L, _ = masked_delta_times.shape

            # Remove last dimension if it's size 1
            masked_delta_times = masked_delta_times.squeeze(-1)  # Now shape is (B, P, L)

            # Create positions tensor
            positions = torch.arange(L).view(1, 1, L).to(masked_delta_times.device)  # Shape: (1, 1, L)

            # Expand seq_lengths to match dimensions
            seq_lengths_expanded = x["seq_lengths"].unsqueeze(2)  # Shape: (B, P, 1)

            # Create mask for invalid positions
            mask = positions >= seq_lengths_expanded  # Shape: (B, P, L)

            # Apply mask to set invalid positions to -inf
            masked_delta_times[mask] = float("-inf")

            # Compute the maximum over the sequence length dimension
            norm_constants = masked_delta_times.amax(dim=[1, 2])
        x["event_times"] = x["event_times"] / norm_constants.view(-1, 1, 1, 1)
        x["delta_times"] = x["delta_times"] / norm_constants.view(-1, 1, 1, 1)
        x["kernel_grids"] = x["kernel_grids"] / norm_constants.view(-1, 1, 1)
        if "base_intensities" in x:
            x["base_intensities"] = x["base_intensities"] / norm_constants.view(-1, 1)
            x["base_intensities"] = (
                x["base_intensities"] * ARTIFICIAL_NORM_FACTOR
            )  # We are testing if the model is less likely to predict constant values then
        if "kernel_evaluations" in x:
            x["kernel_evaluations"] = x["kernel_evaluations"] / norm_constants.view(-1, 1, 1)
            x["kernel_evaluations"] = (
                x["kernel_evaluations"] * ARTIFICIAL_NORM_FACTOR
            )  # We are testing if the model is less likely to predict constant values then
        return norm_constants

    def _denormalize_output(self, x: dict, out: dict, norm_constants: Tensor) -> None:
        out["predicted_kernel_values"] = out["predicted_kernel_values"] * norm_constants.view(-1, 1, 1)
        out["log_predicted_kernel_values_var"] = out["log_predicted_kernel_values_var"] + torch.log(norm_constants).view(-1, 1, 1)
        out["predicted_base_intensity"] = out["predicted_base_intensity"] * norm_constants.view(-1, 1)
        # Rescale by the artifical ARTIFICIAL_NORM_FACTOR factor
        out["predicted_base_intensity"] = out["predicted_base_intensity"] / ARTIFICIAL_NORM_FACTOR
        out["log_predicted_kernel_values_var"] = out["log_predicted_kernel_values_var"] - torch.log(
            torch.tensor(ARTIFICIAL_NORM_FACTOR).to(self.device)
        )
        out["predicted_kernel_values"] = out["predicted_kernel_values"] / ARTIFICIAL_NORM_FACTOR
        x["event_times"] = x["event_times"] * norm_constants.view(-1, 1, 1, 1)
        x["delta_times"] = x["delta_times"] * norm_constants.view(-1, 1, 1, 1)
        x["kernel_grids"] = x["kernel_grids"] * norm_constants.view(-1, 1, 1)
        if "base_intensities" in x:
            x["base_intensities"] = x["base_intensities"] * norm_constants.view(-1, 1)
            x["base_intensities"] = (
                x["base_intensities"] / ARTIFICIAL_NORM_FACTOR
            )  # We are testing if the model is less likely to predict constant values then
        if "kernel_evaluations" in x:
            x["kernel_evaluations"] = x["kernel_evaluations"] * norm_constants.view(-1, 1, 1)
            x["kernel_evaluations"] = (
                x["kernel_evaluations"] / ARTIFICIAL_NORM_FACTOR
            )  # We are testing if the model is less likely to predict constant values then

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

        loss = kernel_loss + base_intensity_loss

        # With a 1% probability plot the kernel function and the ground truth and add the RMSE as a title to the plot
        if torch.rand(1) < 0.01:
            import matplotlib.pyplot as plt

            B, M, T = predicted_kernel_function.shape
            predicted_kernel_function_np = predicted_kernel_function.clone().detach().cpu().float().numpy()
            ground_truth_kernel_function_np = target_kernel_values.clone().detach().cpu().float().numpy()
            # Define scaling factors
            width_per_subplot = 3
            height_per_subplot = 3
            figsize = (width_per_subplot * M, height_per_subplot * B)
            fig, axs = plt.subplots(B, M, figsize=figsize, squeeze=False)
            kernel_rmse_np = torch.sqrt(
                torch.mean(torch.tensor((predicted_kernel_function_np - ground_truth_kernel_function_np)) ** 2, dim=-1)
            )
            kernel_rmse_np = torch.mean(kernel_rmse_np)
            print("RMSE: ", kernel_rmse)
            print("RMSE np: ", kernel_rmse_np)
            for b in range(B):
                for m in range(M):
                    axs[b, m].scatter(
                        kernel_grids[b, m].clone().detach().cpu().float().numpy(), predicted_kernel_function_np[b, m], label="Model"
                    )
                    axs[b, m].scatter(
                        kernel_grids[b, m].clone().detach().cpu().float().numpy(),
                        ground_truth_kernel_function_np[b, m],
                        label="Ground Truth",
                    )
                    axs[b, m].legend()
                    axs[b, m].tick_params(axis="both", which="major", labelsize=8)
            plt.tight_layout()
            plt.savefig("foo.png", dpi=300)
            plt.close()

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

        # [batch_size, seq_len]
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
