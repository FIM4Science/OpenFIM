import math
from pathlib import Path
from typing import Dict, Union

import torch

from fim.models.models import FIMODE, AModel, ModelFactory
from fim.utils.helper import create_class_instance, load_yaml


class FIMWindowed(AModel):
    """Wrapper for FIMODE model to allow multidimensional and/or longer input sequences."""

    def __init__(self, fim_model: FIMODE, window_count: int, overlap: float, **kwargs):
        super().__init__(**kwargs)

        self.fim_model = fim_model
        self.window_count = window_count
        if not 0 <= overlap < 1:
            raise ValueError("Overlap (percentage) must be between 0 and 1.")
        self.overlap = overlap

    def forward(self, x: dict) -> dict:
        """
        Args:
            x (dict): input data, with keys
                `coarse_grid_sample_paths`
        """
        batch_size, max_sequnce_length, process_dim = x["coarse_grid_sample_paths"].shape
        if (
            window_size := (window_size := (math.ceil(max_sequnce_length / self.window_count)))
            + self.overlap * window_size
        ) >= 128:
            raise ValueError(
                "The window size is too large for the model. Please increase the number of windows or increase the overlap."
            )

        for k, v in x.items():
            # make single dimensional [B, T, D] -> [B*D, T, 1]
            v = v.view(batch_size * process_dim, max_sequnce_length, 1)
            # split into overlapping windows [B*D, T, 1] -> [B*D*window_count, window_size+overlap_size, 1]
            v = self._split_into_windows(v, max_sequnce_length)
            x[k] = v

        # forward pass
        # TODO: check output format. Need solution paths and times
        output = self.fim_model(x)
        paths = output["solution_paths"]

        # combine outputs (solution paths) by interpolation
        combined_paths, combinded_times = self._linear_interpolation_windows(paths, max_sequnce_length, process_dim)

        # make multidimensional again
        paths = combined_paths.view(batch_size, max_sequnce_length, process_dim)
        times = combinded_times.view(batch_size, max_sequnce_length, 1)

        return paths, times

    def _split_into_windows(self, x: torch.Tensor, max_sequence_length: int) -> torch.Tensor:
        """
        Split the tensor into non-overlapping windows, add overlap to the left for all but the first window.
        Pad with 1 if the window is smaller than the window size + overlap size. (i.e. elements will be masked out)

        Args:
            x (torch.Tensor): input tensor with shape (batch_size*process_dim, max_sequence_length, 1)
            max_sequence_length (int): the maximum length of the sequence

        Returns:
            torch.Tensor: tensor with shape (num_windows*batch_size*process_dim, window_size+overlap_size, 1)
        """
        # Calculate the size of each window & overlap
        window_size = max_sequence_length // self.window_count
        self.overlap_size = int(window_size * self.overlap)

        windows = []

        # Loop to extract non-overlapping windows and add overlap to the left for all but the first window
        start_idx = 0
        for i in range(self.window_count):
            if i == 0:
                window = x[:, start_idx : start_idx + window_size, :]
                # padd with 1 for overlap size
                window = torch.cat(
                    [torch.ones_like(window[:, : self.overlap_size, :], dtype=window.dtype), window], dim=1
                )
            else:
                start_idx = i * window_size - self.overlap_size
                window = x[:, start_idx : start_idx + window_size + self.overlap_size, :]
                if window.shape[1] < window_size + self.overlap_size:
                    # padd with 1 if necessary
                    window = torch.cat(
                        [
                            window,
                            torch.ones_like(window[:, : self.overlap_size, :], dtype=window.dtype),
                        ],
                        dim=1,
                    )
            windows.append(window)

        return torch.concat(windows, dim=0)

    def _linear_interpolation_windows(
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

        left_window = values_a[:, : -self.overlap_size, :]
        right_window = values_b[:, self.overlap_size :, :]

        # interpolation for the overlapping window
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


ModelFactory.register("FIM_windowed", FIMWindowed)


class FIMImputation(AModel):
    """Imputation and forecasting model based on FIMODE."""

    def __init__(
        self,
        fim_base: Union[FIMODE, Path, str],
        psi_2: dict,
        window_normalization_params: dict,
        scale_feature_mapping: dict,
        use_fim_normalization: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.use_fim_normalization = use_fim_normalization

        self._create_model(fim_base, psi_2, window_normalization_params, scale_feature_mapping)

    def _create_model(
        self,
        fim_base: Union[FIMODE, Path, str],
        psi_2: dict,
        window_normalization: dict,
        scale_feature_mapping: dict,
    ):
        if isinstance(fim_base, FIMODE):
            self.fim_base = fim_base
        else:
            params_dict_dir = Path(fim_base).parent / "../../train_parameters.yaml"
            if not params_dict_dir.exists():
                raise FileNotFoundError(f"Could not find train_parameters.yaml in {params_dict_dir}")
            params_dict = load_yaml(params_dict_dir)
            model_params = params_dict.get("model")

            if model_params.pop("name") != "FIMODE":
                raise ValueError("Not tested for anything but FIMODE as fim base model!")

            self.fim_base = FIMODE(**model_params)

            checkpoint = torch.load(fim_base)
            self.fim_base.load_state_dict(checkpoint["model_state"])

        # Ensure all parameters of fim_model do not require gradients
        for param in self.fim_base.parameters():
            param.requires_grad = False
        self.fim_base.eval()

        self.fim_base.normalization = self.use_fim_normalization

        self.psi_2 = create_class_instance(
            psi_2.pop("name"),
            psi_2,
        )

        self.window_normalization = create_class_instance(
            window_normalization.pop("name"),
            window_normalization,
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
        fine_grid_grid = x.get("fine_grid_grid")
        init_conditions = x.get("init_conditions")  # shape [B, 1]

        B, wc, wlen, _ = obs_values.shape

        # normalize observations
        normalized_windows, window_norm_params = self.window_normalization(obs_values, obs_mask)

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
        scale_feature_vectors = self._get_scale_feature_vector(obs_values=obs_values, obs_times=obs_times)  # shape [B, wc, dim_latent]

        obs_input = torch.concat(
            [
                embedded_windows,
                scale_feature_vectors,
            ],
            dim=-1
        ) # shape [B*wc, 1, 2*dim_latent]

        # get summarizing embedding of input windows ($\hat{h}$)
        embedded_input_sequence = self.psi_2(obs_input) # shape [B, 1, dim_latent]

        # get drift concepts
        drift_concepts = self.fim_base._get_vector_field_concepts(
            branch_out=embedded_input_sequence, grid_grid=locations
        )

        # get solution
        solution_paths = self.fim_base.get_solution(
            fine_grid=locations,
            init_condition=init_conditions,
            branch_out=embedded_input_sequence,
            normalization_parameters=normalization_parameters,
        )

        # denormalize
        denormalized_solution_paths = self.window_normalization.revert_normalization(
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
        first_obs_value = obs_values[..., 0,:]
        last_obs_value = obs_values[..., -1,:]

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
        ) # shape [B, wc, 8]

        # pass through lin layer to get correct size
        return self.scale_feature_mapping(scale_features)

    def new_stats(self):
        pass

    def loss(self, *inputs) -> Dict:
        raise NotImplementedError

    def metric(self, *inputs) -> Dict:
        raise NotImplementedError


ModelFactory.register("FIM_imputation", FIMImputation)
