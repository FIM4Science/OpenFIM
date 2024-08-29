import math
from pathlib import Path
from typing import Dict, Optional, Union

import torch

from fim.models.utils import load_model_from_checkpoint
from fim.utils.helper import create_class_instance

from .models import FIMODE, AModel, ModelFactory


class FIMWindowed(AModel):
    """Wrapper for FIMODE model to allow multidimensional and/or longer input sequences."""

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

    def _create_model(self, fim_model: Union[str, Path], denoising_model: dict):
        self.denoising_model = create_class_instance(
            denoising_model.pop("name"),
            denoising_model,
        )
        self.fim_base = (
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
        noisy_observation_values_processed = torch.concat(
            noisy_observation_values.split(1, dim=-1),
            dim=0,
        )
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
        denoised_observation_values = self._split_into_windows(
            denoised_observation_values, max_sequence_length, padding_value=1
        )
        observation_mask_processed = self._split_into_windows(
            observation_mask_processed, max_sequence_length, padding_value=1
        )
        observation_times_processed = self._split_into_windows(
            observation_times_processed, max_sequence_length, padding_value=1
        )
        location_times_processed = self._split_into_windows(
            location_times_processed, max_sequence_length, padding_value=None
        )

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
        all_samples_values = []
        all_samples_times = []
        for sample_id in range(batch_size * process_dim):
            sample_values = []
            sample_times = []
            for w in range(self.window_count):
                sample_values.append(paths[sample_id + w * batch_size * process_dim])
                sample_times.append(times[sample_id + w * batch_size * process_dim])

            all_samples_values.append(torch.stack(sample_values, dim=0))
            all_samples_times.append(torch.stack(sample_times, dim=0))
        paths_reshaped = torch.stack(all_samples_values, dim=0)
        times_reshaped = torch.stack(all_samples_times, dim=0)

        # combine outputs (solution paths) by interpolation
        combined_paths, combined_times = self._linear_interpolation_windows(paths_reshaped, times_reshaped)

        # make multidimensional again
        all_samples_values = []
        all_samples_times = []

        for sample_id in range(batch_size):
            sample_values = []
            sample_times = []
            for dim in range(process_dim):
                sample_values.append(combined_paths[sample_id + dim * batch_size, :, 0])
                sample_times.append(combined_times[sample_id + dim * batch_size, :, 0])
            all_samples_values.append(torch.stack(sample_values, dim=-1))
            all_samples_times.append(torch.stack(sample_times, dim=-1))

        paths_merged_multi_dim = torch.stack(all_samples_values, dim=0)  # shape [B, window_count*window_size, D]
        times_merged_multi_dim = torch.stack(all_samples_times, dim=0)

        # remove padding from windowing
        last_index = -self.padding_size_windowing_end if self.padding_size_windowing_end is not None else None
        paths_merged_multi_dim = paths_merged_multi_dim[
            :, self.overlap_size : last_index, :
        ]  # shape [B, max_sequence_length, D]
        times_merged_multi_dim = times_merged_multi_dim[
            :, self.overlap_size : last_index, :
        ]  # shape [B, max_sequence_length, D]

        return_values.update(
            {
                "learnt_solution_paths": paths_merged_multi_dim,
                "solution_times": times_merged_multi_dim,
            }
        )

        # print("shape ground truth solution times", x.get("fine_grid_grid").shape)
        # print("shape processed solution times", times_merged_multi_dim.shape)
        # print("are the same ", (x.get("fine_grid_grid") == times_merged_multi_dim).all())

        return return_values

    def _split_into_windows(
        self, x: torch.Tensor, max_sequence_length: int, padding_value: Optional[int] = 1
    ) -> torch.Tensor:
        """
        Split the tensor into overlapping windows.

        Therefore, split first into non-overlapping windows, than add overlap to the left for all but the first window.
        Pad with 1 if the window is smaller than the window size + overlap size. (i.e. elements will be masked out)

        Args:
            x (torch.Tensor): input tensor with shape (batch_size*process_dim, max_sequence_length, 1)
            max_sequence_length (int): the maximum length of the sequence
            padding_value (int): value to pad with.
                if None: for the first window the first value and for the last window the last value is used. (intereseting for locations)
                else: the value is used for padding. Recommnedation to use 1 as this automatically masks the values.

        Returns:
            torch.Tensor: tensor with shape (num_windows*batch_size*process_dim, window_size+overlap_size, 1)
        """
        # Calculate the size of each window & overlap
        window_size = math.ceil(max_sequence_length / self.window_count)
        self.overlap_size = int(window_size * self.overlap)

        windows = []

        # Loop to extract non-overlapping windows and add overlap to the left for all but the first window
        start_idx = 0
        self.padding_size_windowing_end = None
        for i in range(self.window_count):
            if i == 0:
                # first window gets special treatment: no overlap hence need to padd it for full size
                window = x[:, start_idx : start_idx + window_size, :]
                if padding_value is not None:
                    padding = padding_value * torch.ones_like(window[:, : self.overlap_size, :], dtype=window.dtype)
                else:
                    first_value = x[:, 0:1, :]
                    padding = first_value.expand(-1, self.overlap_size, -1)
                window = torch.cat([padding, window], dim=1)
            else:
                start_idx = i * window_size - self.overlap_size
                window = x[:, start_idx : start_idx + window_size + self.overlap_size, :]
                # last window might need special treatment: padding to full size
                if (actual_window_size := window.size(1)) < window_size + self.overlap_size:
                    # needed later for padding removal
                    self.padding_size_windowing_end = window_size + self.overlap_size - actual_window_size

                    if padding_value is not None:
                        padding = padding_value * torch.ones_like(
                            window[:, : self.padding_size_windowing_end, :], dtype=window.dtype
                        )
                    else:
                        last_value = x[:, -1:, :]
                        padding = last_value.expand(-1, self.padding_size_windowing_end, -1)

                    window = torch.cat([window, padding], dim=1)

            assert window.size(1) == window_size + self.overlap_size
            windows.append(window)

        return torch.concat(windows, dim=0)

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
        normalization_params: dict,
        scale_feature_mapping: dict,
        use_fim_normalization: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.use_fim_normalization = use_fim_normalization

        self._create_model(fim_base, psi_2, normalization_params, scale_feature_mapping)

    def _create_model(
        self,
        fim_base: Union[Path, str],
        psi_2: dict,
        normalization_params: dict,
        scale_feature_mapping: dict,
    ):
        self.fim_base = (
            load_model_from_checkpoint(fim_base, module=FIMODE) if isinstance(fim_base, (Path, str)) else fim_base
        )

        self.fim_base.normalization = self.use_fim_normalization

        self.psi_2 = create_class_instance(
            psi_2.pop("name"),
            psi_2,
        )

        self.sequence_normalization = create_class_instance(
            normalization_params.pop("name"),
            normalization_params,
        )

        self.scale_feature_mapping = create_class_instance(scale_feature_mapping.pop("name"), scale_feature_mapping)

    def forward(self, x: dict) -> dict:
        """
        Assume: Input is windowed, i.e. for example observation values shape: [B, window_count, max_window_size, 1]
        """
        obs_values = x.get("observation_values")  # shape [B, wc, wlen, 1]
        obs_mask = x.get("observation_mask", None)  # shape [B, wc, wlen, 1]
        if obs_mask is None:
            obs_mask = torch.zeros_like(obs_values)
        obs_times = x.get("observation_times")  # shape [B, wc, wlen, 1]
        locations = x.get("location_times")  # shape [B, wlen, 1]
        # fine_grid_grid = x.get("fine_grid_grid")
        init_conditions = x.get("init_conditions")  # shape [B, 1]

        B, wc, wlen, _ = obs_values.shape

        # normalize observations
        normalized_windows, window_norm_params = self.sequence_normalization(obs_values, obs_mask)

        # reshape [B, wc, wlen, 1] -> [B*wc, wlen, 1] so FIMODE can handle it
        windowed_values = normalized_windows.view(B * wc, wlen, 1)
        windowed_times = obs_times.view(B * wc, wlen, 1)
        windowed_mask = obs_mask.view(B * wc, wlen, 1)

        if self.use_fim_normalization:
            (
                windowed_values,  # [B*wc, wlen, 1]
                windowed_times,  # [B*wc, wlen, 1]
                locations,  # [B*wc, wlen, 1]
                normalization_parameters,
            ) = self.fim_base.normalize_input(
                obs_times=windowed_times,
                obs_values=windowed_values,
                obs_mask=windowed_mask,
                loc_times=locations,
            )
        else:
            normalization_parameters = {}

        # embedd input windows
        embedded_windows = self.fim_base._encode_input_sequence(
            obs_values=windowed_values, obs_times=windowed_times, obs_mask=windowed_mask
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

        # get drift concepts
        drift_concepts = self.fim_base._get_vector_field_concepts(
            encoded_sequence=embedded_input_sequence, location_times=locations
        )

        # get solution
        solution_paths = self.fim_base.get_solution(
            fine_grid=locations,
            init_condition=init_conditions,
            branch_out=embedded_input_sequence,
            normalization_parameters=normalization_parameters,
        )

        # denormalize
        denormalized_solution_paths = self.sequence_normalization.revert_normalization(
            x=solution_paths, data_concepts=window_norm_params
        )

        return denormalized_solution_paths

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

    def new_stats(self):
        pass

    def loss(self, *inputs) -> Dict:
        raise NotImplementedError

    def metric(self, *inputs) -> Dict:
        raise NotImplementedError


ModelFactory.register("FIM_imputation", FIMImputation)
ModelFactory.register("FIM_windowed", FIMWindowed)
