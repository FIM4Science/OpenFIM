import logging
import math
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from transformers import BitsAndBytesConfig

from fim.data.utils import make_multi_dim, make_single_dim, reorder_windows_per_sample, split_into_windows
from fim.models.utils import load_model_from_checkpoint
from fim.utils.helper import create_class_instance

from ..trainers.mixed_precision import is_bfloat_supported
from ..utils.logging import RankLoggerAdapter
from .models import FIMODE, AModel, ModelFactory


class FIMWindowed(AModel):
    """
    Wrapper for FIMODE model to allow multidimensional and/or longer input sequences.

    Denoising model is optional. If not provided, the input is not denoised.
    """

    def __init__(self, fim_base: Union[str, Path], denoising_model: dict, window_count: int, overlap: float, **kwargs):
        super().__init__(**kwargs)

        self.window_count = window_count
        if not 0 <= overlap < 1:
            raise ValueError("Overlap (percentage) must be between 0 and 1.")
        self.overlap = overlap

        self._create_model(
            fim_model=fim_base,
            denoising_model=denoising_model,
        )

    def _create_model(self, fim_model: Union[str, Path], denoising_model: Optional[dict] = None):
        if denoising_model is None:
            # create dummy model that does nothing
            self.denoising_model = lambda x, mask: x
        else:
            self.denoising_model = create_class_instance(
                denoising_model.pop("name"),
                denoising_model,
            )
        self.fim_base: FIMODE = (
            load_model_from_checkpoint(fim_model, module=FIMODE) if isinstance(fim_model, (str, Path)) else fim_model
        )

    def forward(self, x: dict) -> dict:
        """
        Args:
            x (dict): input data, with keys
                `coarse_grid_sample_paths`
        """
        noisy_observation_values = x.get("coarse_grid_noisy_sample_paths")  # shape [B, T, D]
        observation_mask = x.get("coarse_grid_observation_mask", None)  #  shape [B, T, 1]
        observation_times = x.get("coarse_grid_grid")  # shape [B, T, 1]
        location_times = x.get("fine_grid_grid")  # shape [B, T, 1]

        if observation_mask is None:
            observation_mask = torch.zeros_like(observation_times)

        batch_size, max_sequence_length, process_dim = noisy_observation_values.shape
        if (
            window_size := int((ws := (math.ceil(max_sequence_length / self.window_count))) + self.overlap * ws)
        ) >= 128:
            raise ValueError(
                f"The window size ({window_size}) is too large for the model. Please increase the number of windows or increase the overlap."
            )

        return_values = {
            "target_solution_paths": x.get("fine_grid_sample_paths"),
        }

        # make single dimensional [B, T, D] -> [B*D, T, 1]
        noisy_observation_values_processed = make_single_dim(noisy_observation_values)
        # repeat mask and times so that it matches process dim
        observation_mask_processed = torch.concat([observation_mask for _ in range(process_dim)], dim=0)
        observation_times_processed = torch.concat([observation_times for _ in range(process_dim)], dim=0)
        location_times_processed = torch.concat([location_times for _ in range(process_dim)], dim=0)

        # for k, v in x.items():
        #     # make single dimensional [B, T, D] -> [B*D, T, 1]
        #     if v.size(-1) > 1:
        #         v = torch.concat(
        #             v.split(1, dim=-1),
        #             dim=0,
        #         )
        #     else:
        #         # repeat for process_dim
        #         v = torch.concat([v for _ in range(process_dim)], dim=0)
        #     x[k] = v
        #     assert v.shape == (
        #         batch_size * process_dim,
        #         max_sequence_length,
        #         1,
        #     ), f"{k} has shape {v.shape} and not {(batch_size * process_dim, max_sequence_length, 1)}"

        # denoise input
        denoised_observation_values = self.denoising_model(
            noisy_observation_values_processed, observation_mask_processed
        )
        # assert denoised_observation_values.shape == (batch_size * process_dim, max_sequence_length, 1)

        return_values.update(
            {
                "observation_times": observation_times,
                "observation_values": noisy_observation_values,
                "denoised_observation_values": denoised_observation_values,
            }
        )

        # split into overlapping windows
        denoised_observation_values, padding_params = split_into_windows(
            denoised_observation_values, self.window_count, self.overlap, max_sequence_length, padding_value=1
        )  # shape [B*D*window_count, window_size+overlap_size, 1]
        observation_mask_processed, _ = split_into_windows(
            observation_mask_processed, self.window_count, self.overlap, max_sequence_length, padding_value=1
        )
        observation_times_processed, _ = split_into_windows(
            observation_times_processed, self.window_count, self.overlap, max_sequence_length, padding_value=1
        )
        location_times_processed, _ = split_into_windows(
            location_times_processed, self.window_count, self.overlap, max_sequence_length, padding_value=None
        )

        self.overlap_size, self.padding_size_windowing_end = padding_params

        # prepare data for FIMODE forward pass
        # for k, v in x.items():
        #     # split into overlapping windows [B*D, T, 1] -> [B*D*window_count, window_size+overlap_size, 1]
        #     v = self._split_into_windows(v, max_sequence_length)
        #     x[k] = v
        assert denoised_observation_values.shape == (batch_size * process_dim * self.window_count, window_size, 1), str(
            denoised_observation_values.shape
        )
        assert (
            observation_mask_processed.shape
            == denoised_observation_values.shape
            == observation_times_processed.shape
            == location_times_processed.shape
        )
        # forward pass for fim base model to get solution paths per window
        fimode_input = {
            "coarse_grid_observation_mask": observation_mask_processed,
            "coarse_grid_noisy_sample_paths": denoised_observation_values,
            "coarse_grid_grid": observation_times_processed,
            "fine_grid_grid": location_times_processed,
        }
        output = self.fim_base(fimode_input, training=False)

        paths = output["visualizations"]["solution"]["learnt"]  # shape [B*D*window_count, window_size+overlap_size, 1]
        times = output["visualizations"]["fine_grid_grid"]  # shape [B*D*window_count, window_size+overlap_size, 1]

        # separate windows: reshape to [B*D, window_count, window_size+overlap_size, 1]
        paths_reordered = reorder_windows_per_sample(
            paths, window_count=self.window_count, batch_size=batch_size, process_dim=process_dim
        )
        times_reordered = reorder_windows_per_sample(
            times, window_count=self.window_count, batch_size=batch_size, process_dim=process_dim
        )

        # old
        # all_samples_values = []
        # all_samples_times = []
        # for sample_id in range(batch_size * process_dim):
        #     sample_values = []
        #     sample_times = []
        #     for w in range(self.window_count):
        #         sample_values.append(paths[sample_id + w * batch_size * process_dim])
        #         sample_times.append(times[sample_id + w * batch_size * process_dim])

        #     all_samples_values.append(torch.stack(sample_values, dim=0))
        #     all_samples_times.append(torch.stack(sample_times, dim=0))
        # paths_reshaped = torch.stack(all_samples_values, dim=0)
        # times_reshaped = torch.stack(all_samples_times, dim=0)

        # combine outputs (solution paths) by interpolation
        combined_paths, combined_times = self._linear_interpolation_windows(paths_reordered, times_reordered)

        # make multidimensional again
        paths_merged_multi_dim = make_multi_dim(combined_paths, batch_size, process_dim)
        times_merged_multi_dim = make_multi_dim(combined_times, batch_size, process_dim)

        # make multidimensional again (old)
        # all_samples_values = []
        # all_samples_times = []

        # for sample_id in range(batch_size):
        #     sample_values = []
        #     sample_times = []
        #     for dim in range(process_dim):
        #         sample_values.append(combined_paths[sample_id + dim * batch_size, :, 0])
        #         sample_times.append(combined_times[sample_id + dim * batch_size, :, 0])
        #     all_samples_values.append(torch.stack(sample_values, dim=-1))
        #     all_samples_times.append(torch.stack(sample_times, dim=-1))

        # paths_merged_multi_dim = torch.stack(all_samples_values, dim=0)  # shape [B, window_count*window_size, D]
        # times_merged_multi_dim = torch.stack(all_samples_times, dim=0)

        # remove padding from windowing
        last_index = -self.padding_size_windowing_end if self.padding_size_windowing_end is not None else None
        paths_merged_multi_dim = paths_merged_multi_dim[
            :, self.overlap_size : last_index, :
        ]  # shape [B, max_sequence_length, D]
        times_merged_multi_dim = times_merged_multi_dim[
            :, self.overlap_size : last_index, :
        ]  # shape [B, max_sequence_length, D]

        assert paths_merged_multi_dim.shape == (
            batch_size,
            max_sequence_length,
            process_dim,
        ), f"got {paths_merged_multi_dim.shape}, expected {(batch_size, max_sequence_length, process_dim)}"

        return_values.update(
            {
                "learnt_solution_paths": paths_merged_multi_dim,
                "solution_times": times_merged_multi_dim,
            }
        )
        return return_values

    def _linear_interpolation_windows(
        self, paths: torch.Tensor, times: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Combine the windows by linear interpolation of the overlapping part.

        Args:
            paths (torch.Tensor): paths of the windows with shape [B*D, window_count, window_size+overlap_size, 1]
            times (torch.Tensor): times of the windows with shape [B*D, window_count, window_size+overlap_size, 1]

        Returns:
            tuple[torch.Tensor, torch.Tensor]: combined paths and times with shape [B*D, approx. window_count*window_size, 1]
        """
        assert paths.dim() == 4 and times.dim() == 4
        assert paths.size(1) == self.window_count and times.size(1) == self.window_count

        combined_values = paths[:, 0, :, :]
        combined_times = times[:, 0, :, :]

        for i in range(1, self.window_count):
            values_b = paths[:, i, :, :]
            times_b = times[:, i, :, :]

            combined_values, combined_times = self._merge_two_windows(
                combined_values, combined_times, values_b, times_b
            )

        return combined_values, combined_times

    def _merge_two_windows(
        self, values_a: torch.Tensor, times_a: torch.Tensor, values_b: torch.Tensor, times_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Combine two windows by linear interpolation of the overlapping part.

        Args:
            values_a (torch.Tensor): values of the first window
            times_a (torch.Tensor): times of the first window
            values_b (torch.Tensor): values of the second window
            times_b (torch.Tensor): times of the second window

        Returns:
            tuple[torch.Tensor, torch.Tensor]: combined values and times
        """
        if self.overlap_size == 0:
            combined_values = torch.cat([values_a, values_b], dim=1)
            combined_times = torch.cat([times_a, times_b], dim=1)

            return combined_values, combined_times

        # overlap-free parts of tensors a and b
        left_window = values_a[:, : -self.overlap_size, :]
        right_window = values_b[:, self.overlap_size :, :]

        # interpolation for the overlapping part
        overlap_t = times_a[:, -self.overlap_size :, :]
        overlap_values_a = values_a[:, -self.overlap_size :, :]
        overlap_values_b = values_b[:, : self.overlap_size, :]

        t_overlap_first = overlap_t[:, :1, :]
        t_overlap_last = overlap_t[:, -1:, :]

        t_ratio = (overlap_t - t_overlap_first) / (t_overlap_last - t_overlap_first + 1e-6)
        interpolated_overlap = (1 - t_ratio) * overlap_values_a + t_ratio * overlap_values_b

        combined_values = torch.cat([left_window, interpolated_overlap, right_window], dim=1)
        combined_times = torch.cat([times_a[:, : -self.overlap_size, :], times_b], dim=1)

        return combined_values, combined_times

    def new_stats(self):
        pass

    def loss(self, *inputs) -> Dict:
        raise NotImplementedError("FIM_windowed does not support loss calculation, as it is not trained.")

    def metric(self, *inputs) -> Dict:
        raise NotImplementedError(
            """FIM_windowed does not support metric calculation, as it is not trained. Metric evaluation happens in evaluation script."""
        )


class FIMImputation(AModel):
    """Imputation and forecasting model based on FIMODE."""

    def __init__(
        self,
        fim_base: Union[Path, str],
        psi_2: dict,
        normalization_values: dict,
        normalization_times: dict,
        scale_feature_mapping: dict,
        use_fim_normalization: bool,
        loss_configs: dict,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_bf16: bool = False,
        device_map: Optional[str] = None,
        resume: bool = False,
        peft: Optional[dict] = None,
        **kwargs,
    ):
        super(FIMImputation, self).__init__()
        self.logger = RankLoggerAdapter(logging.getLogger(self.__class__.__name__))
        self._device_map = device_map
        self._torch_dtype = None
        self._quantization_config = None
        self.resume = resume
        self.peft = peft
        if load_in_8bit and load_in_4bit:
            raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
        elif load_in_8bit or load_in_4bit:
            self._quantization_config = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
            )
            self._device_map = "auto"
        if use_bf16 and is_bfloat_supported:
            self._torch_dtype = torch.float16

        self.use_fim_normalization = use_fim_normalization

        self._create_model(
            fim_base,
            psi_2,
            normalization_values,
            normalization_times,
            scale_feature_mapping,
            loss_configs,
        )

    def _create_model(
        self,
        fim_base: Union[Path, str],
        psi_2: dict,
        normalization_values: dict,
        normalization_times: dict,
        scale_feature_mapping: dict,
        loss_configs: dict,
    ):
        self.fim_base: FIMODE = (
            load_model_from_checkpoint(fim_base, module=FIMODE) if isinstance(fim_base, (Path, str)) else fim_base
        )

        self.fim_base.apply_normalization = self.use_fim_normalization

        self.psi_2 = create_class_instance(
            psi_2.pop("name"),
            psi_2,
        )

        self.normalization_values = create_class_instance(
            normalization_values.pop("name"),
            normalization_values,
        )

        self.normalization_times = create_class_instance(
            normalization_times.pop("name"),
            normalization_times,
        )

        self.scale_feature_mapping = create_class_instance(scale_feature_mapping.pop("name"), scale_feature_mapping)

        self.loss_scale_latent_embedding = loss_configs.get("loss_scale_latent_embedding", 1.0)
        self.loss_scale_drift = loss_configs.get("loss_scale_drift", 1.0)
        self.loss_scale_unsuperv_loss = loss_configs.get("loss_scale_unsuperv_loss", 1.0)

    def forward(
        self,
        x: dict,
        schedulers: Optional[dict] = None,
        step: Optional[int] = None,
        training: bool = False,
    ) -> dict:
        """
        Assume: Input is windowed, i.e. for example observation values shape: [B, window_count, max_window_size, 1]
        """
        # extract data
        # for input sequence (observed windows)
        obs_values = x.get("observation_values")  # shape [B, wc, wlen, 1]
        obs_mask = x.get("observation_mask", None)  # shape [B, wc, wlen, 1]
        if obs_mask is None:
            self.logger.debug("Warning: No mask provided. Assuming no missing values.")
            obs_mask = torch.zeros_like(obs_values, dtype=bool)
        obs_times = x.get("observation_times")  # shape [B, wc, wlen, 1]

        # data for imputation window
        locations = x.get("location_times")  # shape [B, wlen, 1]
        init_conditions_for_impuWindow = x.get("initial_conditions")  # shape [B, 1]
        drift_impuWindow_target = x.get("target_drift", None)  # shape [B, wlen, 1]
        sample_path_impuWindow_target = x.get("target_sample_path", None)  # shape [B, wlen, 1]

        B, wc, wlen, _ = obs_values.shape

        # tor sequences apart on window level, ie. reshape [B, wc, wlen, 1] -> [B*wc, wlen, 1] so FIMODE can handle it
        # obs_values_torn = obs_values.view(B * wc, wlen, 1)
        # obs_times_torn = obs_times.view(B * wc, wlen, 1)

        # normalize observation values & times per sequence
        # TODO: discuss alternative with normalizat per window: problem: how to revert normalization in impu window?
        obs_values_torn_normalized, norm_params_values_torn = self.normalization_values(
            obs_values.view(B, wc * wlen, 1), obs_mask.view(B, wc * wlen, 1)
        )
        obs_values_torn_normalized = obs_values_torn_normalized.view(B, wc, wlen, 1).view(B * wc, wlen, 1)

        obs_times_torn_normalized, norm_params_times_torn = self.normalization_times(
            obs_times.view(B, wc * wlen, 1), obs_mask.view(B, wc * wlen, 1)
        )
        obs_times_torn_normalized = obs_times_torn_normalized.view(B, wc, wlen, 1).view(B * wc, wlen, 1)

        obs_mask_torn = obs_mask.view(B * wc, wlen, 1)

        # normalize locations
        locations_normalized, _ = self.normalization_times(locations, norm_params=norm_params_times_torn)

        if self.use_fim_normalization:
            # locations are not normalized
            (
                obs_values_torn_normalized,  # [B*wc, wlen, 1]
                obs_times_torn_normalized,  # [B*wc, wlen, 1]
                _,
                normalization_parameters_fimbase,
            ) = self.fim_base.normalize_input(
                obs_times=obs_times_torn_normalized,
                obs_values=obs_values_torn_normalized,
                obs_mask=obs_mask_torn,
                loc_times=torch.zeros_like(obs_times_torn_normalized),
            )
        else:
            normalization_parameters_fimbase = {}

        # embedd input windows
        embedded_windows = self.fim_base._encode_input_sequence(
            obs_values=obs_values_torn_normalized, obs_times=obs_times_torn_normalized, obs_mask=obs_mask_torn
        )  # shape [B*wc, 1, dim_latent]

        # reshape back to [B, wc, dim_latent]
        embedded_windows = embedded_windows.view(B, wc, -1)

        # get scale features from non - normalized data
        scale_feature_vectors = self._get_scale_feature_vector(
            obs_values=obs_values, obs_times=obs_times
        )  # shape [B, wc, dim_latent]

        obs_input = torch.concat(
            [
                embedded_windows,
                scale_feature_vectors,
            ],
            dim=-1,
        )  # shape [B*wc, 1, 2*dim_latent]

        # get summarizing embedding of input windows ($\hat{h}$)
        embedded_input_sequence = self.psi_2(obs_input)  # shape [B, 1, dim_latent]

        # get drift concepts of Imputation Window
        drift_concepts_learnt = self.fim_base._get_vector_field_concepts(
            encoded_sequence=embedded_input_sequence, location_times=locations_normalized
        )

        # get solution with dummy initial condition (normalization issues...)
        solution_paths_learnt = self.fim_base.get_solution(
            fine_grid=locations_normalized,
            init_condition=torch.zeros_like(init_conditions_for_impuWindow),
            branch_out=embedded_input_sequence,
            normalization_parameters=normalization_parameters_fimbase,
        )

        #  Revert FIMBase Normalization of drift if necessary
        if self.use_fim_normalization:
            drift_concepts_learnt = self.fim_base._renormalize_vector_field_params(
                vector_field_concepts=drift_concepts_learnt,
                normalization_parameters=normalization_parameters_fimbase,
            )

        # revert normalization (on input sequence level)
        denormalized_solution_paths_learnt = self.normalization_values.revert_normalization(
            x=solution_paths_learnt, data_concepts=norm_params_values_torn
        )
        denormalized_vector_field_concepts = self._renormalize_vector_field_params(
            vector_field_concepts=drift_concepts_learnt,
            norm_params_time=norm_params_times_torn,
            norm_params_values=norm_params_values_torn,
        )

        # move to correct initial condition (hacky solution: subtract current initial value and add desired one -> something better?)
        # TODO Think about: How to handle inference?
        denormalized_solution_paths_learnt = (
            denormalized_solution_paths_learnt
            - denormalized_solution_paths_learnt[:, :1, :]
            + init_conditions_for_impuWindow[:, None, :]
        )

        if drift_impuWindow_target is not None and sample_path_impuWindow_target is not None:
            loss = self.loss(
                vector_field_concepts=denormalized_vector_field_concepts,
                target_drift=drift_impuWindow_target,
                target_sample_path=sample_path_impuWindow_target,
                impu_window_grid=locations,
                latent_embedding_impu_window=embedded_input_sequence,
            )
        else:
            loss = {}
        return {
            "losses": loss,
            "visualizations": {
                "imputation_window": {
                    "locations": locations,
                    "learnt": denormalized_solution_paths_learnt,
                    "target": sample_path_impuWindow_target,
                },
                "observations": {
                    "values": obs_values,
                    "mask": obs_mask,
                    "times": obs_times,
                },
                "drift": {
                    "learnt": denormalized_vector_field_concepts[0],
                    "certainty": torch.exp(denormalized_vector_field_concepts[1]),
                    "target": drift_impuWindow_target,
                    "locations": locations,
                },
            },
        }

    def _get_scale_feature_vector(self, obs_values: torch.Tensor, obs_times: torch.Tensor) -> torch.Tensor:
        """
        Compute scale features per window.

        Features: min & max of time points, range of time, min & max of obs_values & range, first & last obs_value

        Args:
            obs_values (torch.Tensor): shape: [B, wc, wl, 1]
            obs_times (torch.Tensor): shape: [B, wc, wl, 1]

        Returns:
            scale_feature_vector (torch.Tensor): shape [B, wc, 8]
        """
        # Calculate min and max of time points and obs_values
        min_time_points = obs_times.min(dim=-2)[0]
        max_time_points = obs_times.max(dim=-2)[0]
        min_obs_values = obs_values.min(dim=-2)[0]
        max_obs_values = obs_values.max(dim=-2)[0]

        # Calculate ranges of time and obs_values
        time_range = max_time_points - min_time_points
        obs_range = max_obs_values - min_obs_values

        # Get first and last obs_value
        first_obs_value = obs_values[..., 0, :]
        last_obs_value = obs_values[..., -1, :]

        # Stack all features together
        scale_features = torch.concat(
            [
                min_time_points,
                max_time_points,
                time_range,
                min_obs_values,
                max_obs_values,
                obs_range,
                first_obs_value,
                last_obs_value,
            ],
            dim=-1,
        )  # shape [B, wc, 8]

        # pass through lin layer to get correct size
        return self.scale_feature_mapping(scale_features)

    def _renormalize_vector_field_params(
        self,
        vector_field_concepts: tuple[torch.Tensor, torch.Tensor],
        norm_params_time,
        norm_params_values,
    ):
        drift_mean, drift_log_std = vector_field_concepts
        shape = drift_mean.shape

        reversion_factor_time = self.normalization_times.get_reversion_factor(norm_params_time)
        reversion_factor_values = self.normalization_values.get_reversion_factor(norm_params_values)

        # reshape (and repeat) factors to match drift_mean
        if reversion_factor_values.dim() == 2:
            print(" dim 2")
            reversion_factor_values = reversion_factor_values.unsqueeze(-1)
        if reversion_factor_time.dim() == 2:
            print(" dim 2")
            reversion_factor_time = reversion_factor_time.unsqueeze(-1)

        reversion_factor_values_view = reversion_factor_values.repeat(1, shape[1], 1)  # Shape [B, L, 1]
        reversion_factor_times_view = reversion_factor_time.repeat(1, shape[1], 1)  # Shape [B, L, 1]

        assert reversion_factor_values_view.shape == shape == reversion_factor_times_view.shape

        # rescale mean
        drift_mean = drift_mean * reversion_factor_values_view / reversion_factor_times_view  # Shape [B, L, 1]

        assert drift_mean.shape == shape

        # rescale log std if provided
        if drift_log_std is not None:
            learnt_drift_log_std = (
                drift_log_std + torch.log(reversion_factor_values_view) - torch.log(reversion_factor_times_view)
            )  # Shape [B, L, 1]
            assert learnt_drift_log_std.shape == shape
            return drift_mean, learnt_drift_log_std

        else:
            return drift_mean

    def new_stats(self):
        pass

    def loss(
        self,
        vector_field_concepts: tuple,
        target_drift: torch.Tensor,
        target_sample_path: torch.Tensor,
        impu_window_grid: torch.Tensor,
        latent_embedding_impu_window: torch.Tensor,
    ) -> Dict:
        """
        Compute loss for model on the imputation window.

        The loss constsis of three components:
         - negative log likelihood of the vector field values at the fine grid points.
         - unsupervised loss (one-step ahead prediction loss)
         - the L1 loss between the learnt embedding of the imputation window and the embedding of the window as given from the fimbase model.

        The total loss is a weighted sum of all losses. The weights are defined in the loss_configs. (loss_scale_drift, loss_scale_init_cond, loss_scale_unsuperv_loss)

        """
        # nllh drift
        learnt_mean_drift, learnt_log_std_drift = vector_field_concepts
        learnt_var_drift = torch.exp(learnt_log_std_drift) ** 2

        nllh_drift_avg = torch.mean(
            1 / 2 * (target_drift - learnt_mean_drift) ** 2 / learnt_var_drift + learnt_log_std_drift
        )

        # unsupervised loss
        step_size_fine_grid = impu_window_grid[..., 1:, :] - impu_window_grid[..., :-1, :]

        # unsupervised_loss[i] = (target_path[i]-target_path[i-1] - drift[i-1]*step_size)^2
        unsupervised_loss = torch.mean(
            torch.sum(
                (
                    target_sample_path[..., 1:, :]
                    - target_sample_path[..., :-1, :]
                    - learnt_mean_drift[..., :-1, :] * step_size_fine_grid
                )
                ** 2,
                dim=-2,
            )
        )

        # l1 loss between learnt embedding and embedding from fimbase
        target_embedding = self.fim_base._encode_input_sequence(
            obs_values=target_sample_path,
            obs_times=impu_window_grid,
            obs_mask=torch.zeros_like(impu_window_grid, dtype=bool),
        )
        target_embedding = target_embedding.squeeze(1)

        l1_loss_embedding = torch.mean(torch.abs(target_embedding - latent_embedding_impu_window))

        total_loss = (
            self.loss_scale_latent_embedding * l1_loss_embedding
            + self.loss_scale_drift * nllh_drift_avg
            + self.loss_scale_unsuperv_loss * unsupervised_loss
        )

        return {
            "loss": total_loss,
            "nllh_drift": nllh_drift_avg,
            "unsupervised_loss": unsupervised_loss,
            "l1_loss_embedding": l1_loss_embedding,
        }

    def metric(self, *inputs) -> Dict:
        raise NotImplementedError

    def is_peft(self) -> bool:
        return self.peft is not None and self.peft["method"] is not None


ModelFactory.register("FIM_imputation", FIMImputation)
ModelFactory.register("FIM_windowed", FIMWindowed)
