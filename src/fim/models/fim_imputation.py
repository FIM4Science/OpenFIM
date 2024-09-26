import logging
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from transformers import BitsAndBytesConfig

from fim.models.utils import load_model_from_checkpoint
from fim.utils.helper import create_class_instance

from ..data.utils import make_multi_dim, make_single_dim, repeat_for_dim
from ..trainers.mixed_precision import is_bfloat_supported
from ..utils.logging import RankLoggerAdapter
from .blocks import MinMaxNormalization
from .fim_ode import FIMODE
from .models import AModel, ModelFactory


class FIMImputation(AModel):
    """Imputation and forecasting model based on FIMODE."""

    def __init__(
        self,
        fim_base: Union[Path, str],
        psi_2: dict,
        global_normalization_values: dict,
        global_normalization_times: dict,
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
        self.loc_normalize_locations = MinMaxNormalization()

        self._create_model(
            fim_base,
            psi_2,
            global_normalization_values,
            global_normalization_times,
            scale_feature_mapping,
            loss_configs,
        )

    def _create_model(
        self,
        fim_base: Union[Path, str],
        psi_2: dict,
        global_normalization_values: dict,
        global_normalization_times: dict,
        scale_feature_mapping: dict,
        loss_configs: dict,
    ):
        self.fim_base: FIMODE = load_model_from_checkpoint(fim_base, module=FIMODE) if isinstance(fim_base, (Path, str)) else fim_base

        self.fim_base.apply_normalization = self.use_fim_normalization

        self.psi_2 = create_class_instance(
            psi_2.pop("name"),
            psi_2,
        )

        self.glob_normalize_values = create_class_instance(
            global_normalization_values.pop("name"),
            global_normalization_values,
        )

        self.glob_normalize_times = create_class_instance(
            global_normalization_times.pop("name"),
            global_normalization_times,
        )

        self.scale_feature_mapping = create_class_instance(scale_feature_mapping.pop("name"), scale_feature_mapping)

        self.loss_scale_latent_embedding = loss_configs.get("loss_scale_latent_embedding", 1.0)
        self.loss_scale_drift = loss_configs.get("loss_scale_drift", 1.0)
        self.loss_scale_unsuperv_loss = loss_configs.get("loss_scale_unsuperv_loss", 1.0)

    def forward(self, x: dict, schedulers: Optional[dict] = None, step: Optional[int] = None, training: bool = False) -> dict:
        """
        Assume: Input is windowed, i.e. for example observation values shape: [B, window_count, max_window_size, 1]
        """
        ##
        # extract data
        ##

        obs_values = x.get("observation_values")  # shape [B, wc, wlen, 1]
        obs_mask = x.get("observation_mask", None)  # shape [B, wc, wlen, 1]
        if obs_mask is None:
            self.logger.debug("Warning: No mask provided. Assuming no missing values.")
            obs_mask = torch.zeros_like(obs_values, dtype=bool)
        obs_times = x.get("observation_times")  # shape [B, wc, wlen, 1]

        locations = x.get("location_times")  # shape [B, wlen_locs, 1]
        init_conditions_for_impuWindow = x.get("initial_conditions")  # shape [B, 1]
        drift_impuWindow_target = x.get("target_drift", None)  # shape [B, wlen_locs, 1]
        sample_path_impuWindow_target = x.get("target_sample_path", None)  # shape [B, wlen_locs, 1]

        B, wc, wlen, _ = obs_values.shape

        ##
        # normalize data (globally and/or optionally locally)
        ##

        obs_values_glob_normalized, glob_norm_params_values = self.glob_normalize_values(
            obs_values.view(B, wc * wlen, 1), obs_mask.view(B, wc * wlen, 1)
        )
        obs_values_glob_normalized = obs_values_glob_normalized.view(B, wc, wlen, 1).view(B * wc, wlen, 1)

        obs_times_glob_normalized, glob_norm_params_times = self.glob_normalize_times(
            obs_times.view(B, wc * wlen, 1), obs_mask.view(B, wc * wlen, 1)
        )
        obs_times_glob_normalized = obs_times_glob_normalized.view(B, wc, wlen, 1).view(B * wc, wlen, 1)

        obs_mask_torn = obs_mask.view(B * wc, wlen, 1)

        if self.use_fim_normalization:
            (
                obs_values_loc_normalized,  # [B*wc, wlen, 1]
                obs_times_loc_normalized,  # [B*wc, wlen, 1]
                _,
                _,
            ) = self.fim_base.normalize_input(
                obs_times=obs_times_glob_normalized,
                obs_values=obs_values_glob_normalized,
                obs_mask=obs_mask_torn,
                loc_times=torch.zeros_like(obs_times_glob_normalized),
            )
            # normalize locations to [0, 1] as well
            locations_loc_normalized, loc_norm_params_locations = self.loc_normalize_locations(locations)

            # need dummy normalization parameters for values. will be implicitly learnt by model
            loc_norm_params_locations = {
                "norm_params_time": loc_norm_params_locations,
                "norm_params_values": (
                    torch.zeros_like(loc_norm_params_locations[0]),
                    torch.ones_like(loc_norm_params_locations[1]),
                ),
            }
        else:
            obs_values_loc_normalized = obs_values_glob_normalized
            obs_times_loc_normalized = obs_times_glob_normalized

            locations_loc_normalized = self.glob_normalize_times(locations, norm_params=glob_norm_params_times)[0]
            loc_norm_params_locations = {}

        ##
        # embedd input sequences
        ##

        embedded_windows = self.fim_base._encode_input_sequence(
            obs_values=obs_values_loc_normalized, obs_times=obs_times_loc_normalized, obs_mask=obs_mask_torn
        )  # shape [B*wc, 1, dim_latent]

        embedded_windows = embedded_windows.view(B, wc, -1)

        # get scale features from globally normalized data
        scale_feature_vectors = self._get_scale_feature_vector(
            obs_values=obs_values_glob_normalized.view(B, wc, wlen, 1),
            obs_times=obs_times_glob_normalized.view(B, wc, wlen, 1),
            obs_mask=obs_mask,
        )  # shape [B, wc, dim_latent]
        # scale_feature_vectors = self._get_scale_feature_vector(
        #     obs_values=obs_values,
        #     obs_times=obs_times,
        #     obs_mask=obs_mask,
        # )  # shape [B, wc, dim_latent]

        obs_input = torch.concat(
            [
                embedded_windows,
                scale_feature_vectors,
            ],
            dim=-1,
        )  # shape [B, wc, 2*dim_latent]

        embedded_input_sequence = self.psi_2(obs_input)  # shape [B, 1, dim_latent]
        assert embedded_input_sequence.dim() == 3 and embedded_input_sequence.size(0) == B

        drift_concepts_learnt = self.fim_base._get_vector_field_concepts(
            encoded_sequence=embedded_input_sequence, location_times=locations_loc_normalized
        )
        assert (
            drift_concepts_learnt[0].dim() == 3
            and drift_concepts_learnt[0].size(0) == B
            and drift_concepts_learnt[0].size(0) == B
            and drift_concepts_learnt[0].size(2) == 1
        )

        # get solution with dummy initial condition
        solution_paths_learnt = self.fim_base.get_solution(
            fine_grid=locations_loc_normalized,
            init_condition=torch.zeros_like(init_conditions_for_impuWindow),
            branch_out=embedded_input_sequence,
            normalization_parameters=loc_norm_params_locations,
        )

        assert solution_paths_learnt.shape == locations.shape

        ##
        # Denormalize
        ##

        if self.use_fim_normalization:
            drift_concepts_learnt = self.fim_base._renormalize_vector_field_params(
                vector_field_concepts=drift_concepts_learnt,
                normalization_parameters=loc_norm_params_locations,
            )

        denormalized_solution_paths_learnt = self.glob_normalize_values.revert_normalization(
            x=solution_paths_learnt, data_concepts=glob_norm_params_values
        )
        # denormalized_solution_paths_learnt = self.glob_normalize_values.revert_normalization(
        #     x=denormalized_solution_paths_learnt, data_concepts=glob_norm_params_values
        # )
        denormalized_vector_field_concepts = self._renormalize_vector_field_params(
            vector_field_concepts=drift_concepts_learnt,
            norm_params_time=glob_norm_params_times,
            norm_params_values=glob_norm_params_values,
        )

        # move to correct initial condition (hacky solution: subtract current initial value and add desired one -> something better?)
        denormalized_solution_paths_learnt = (
            denormalized_solution_paths_learnt - denormalized_solution_paths_learnt[:, :1, :] + init_conditions_for_impuWindow[:, None, :]
        )

        ##
        # Loss
        ##

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

    def _get_scale_feature_vector(self, obs_values: torch.Tensor, obs_times: torch.Tensor, obs_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute scale features per window: extract features & embedd with a linear layer.

        Features: min & max of time points, range of time, min & max of obs_values & range, first & last obs_value & difference between first and list obs_value.

        Args:
            obs_values (torch.Tensor): shape: [B, wc, wl, 1]
            obs_times (torch.Tensor): shape: [B, wc, wl, 1]
            obs_mask (torch.Tensor): shape: [B, wc, wl, 1], 1 indicates masked out value

        Returns:
            scale_feature_vector (torch.Tensor): shape [B, wc, dim_latent]
        """
        assert obs_values.shape == obs_times.shape == obs_mask.shape
        assert obs_values.dim() == 4

        # Calculate min and max of time points and obs_values considering masked values
        min_time_points = torch.where(
            ~obs_mask,
            obs_times,
            torch.tensor(float("inf"), device=obs_values.device),
        ).min(dim=-2)[0]
        max_time_points = torch.where(
            ~obs_mask,
            obs_times,
            torch.tensor(float("-inf"), device=obs_values.device),
        ).max(dim=-2)[0]
        min_obs_values = torch.where(
            ~obs_mask,
            obs_values,
            torch.tensor(float("inf"), device=obs_values.device),
        ).min(dim=-2)[0]
        max_obs_values = torch.where(
            ~obs_mask,
            obs_values,
            torch.tensor(float("-inf"), device=obs_values.device),
        ).max(dim=-2)[0]

        # Calculate ranges of time and obs_values
        time_range = max_time_points - min_time_points
        range_min_max_obs_values = max_obs_values - min_obs_values

        # Get first and last observed value based on mask
        # get index of first observed value
        first_obs_idx = torch.argmax((~obs_mask).int(), dim=2)
        last_obs_idx = (~obs_mask.squeeze(-1)).cumsum(dim=2).argmax(dim=2, keepdim=True)
        first_obs_values = obs_values.gather(dim=-2, index=first_obs_idx.unsqueeze(-1).expand(-1, -1, -1, obs_values.shape[-1])).squeeze(-1)
        last_obs_values = obs_values.gather(dim=-2, index=last_obs_idx.unsqueeze(-1).expand(-1, -1, -1, obs_values.shape[-1])).squeeze(-1)
        range_first_last_obs_values = last_obs_values - first_obs_values

        # Stack all features together
        scale_features = torch.concat(
            [
                min_time_points,
                max_time_points,
                time_range,
                min_obs_values,
                max_obs_values,
                range_min_max_obs_values,
                first_obs_values,
                last_obs_values,
                range_first_last_obs_values,
            ],
            dim=-1,
        )  # shape [B, wc, 9]

        # pass through lin layer
        return self.scale_feature_mapping(scale_features)

    def _renormalize_vector_field_params(
        self,
        vector_field_concepts: tuple[torch.Tensor, torch.Tensor],
        norm_params_time,
        norm_params_values,
    ):
        drift_mean, drift_log_std = vector_field_concepts
        shape = drift_mean.shape

        reversion_factor_time = self.glob_normalize_times.get_reversion_factor(norm_params_time)
        reversion_factor_values = self.glob_normalize_values.get_reversion_factor(norm_params_values)

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

        nllh_drift_avg = torch.mean(1 / 2 * (target_drift - learnt_mean_drift) ** 2 / learnt_var_drift + learnt_log_std_drift)

        # unsupervised loss
        step_size_fine_grid = impu_window_grid[..., 1:, :] - impu_window_grid[..., :-1, :]

        # unsupervised_loss[i] = (target_path[i]-target_path[i-1] - drift[i-1]*step_size)^2
        unsupervised_loss = torch.mean(
            torch.sum(
                (target_sample_path[..., 1:, :] - target_sample_path[..., :-1, :] - learnt_mean_drift[..., :-1, :] * step_size_fine_grid)
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


class FIMImputationWindowed(AModel):
    def __init__(self, fim_imputation: Union[str, Path, FIMImputation], denoising_model: Optional[dict] = None, **kwargs):
        super(FIMImputationWindowed, self).__init__()
        self.logger = RankLoggerAdapter(logging.getLogger(self.__class__.__name__))

        self._create_model(fim_imputation, denoising_model)

    def _create_model(self, fim_imputation: Union[str, Path, FIMImputation], denoising_model: dict):
        if denoising_model is not None:
            self.denoising_model = create_class_instance(denoising_model.pop("name"), denoising_model)
        else:
            self.denoising_model = lambda x, mask: x

        self.fim_imputation: FIMImputation = (
            load_model_from_checkpoint(fim_imputation, module=FIMImputation) if isinstance(fim_imputation, (str, Path)) else fim_imputation
        )

    def forward(self, batch: dict) -> dict:
        """
        Compute solution path for imputation window.

        Assumes the data to be windowed already. Also computes interpolation of observed windows.

        Args:
            batch (dict)
                - observations (values: B, wc, wlen, D; mask and times: B, wc, wlen, 1)
                - locations (B, wlen_locs, 1)
        """
        # get input data
        obs_values = batch.get("observation_values")  # shape [B, wc, wlen, D]
        obs_times = batch.get("observation_times")  # shape [B, wc, wlen, 1]
        obs_mask = batch.get("observation_mask", None)  # shape [B, wc, wlen, D]
        if obs_mask is None:
            self.logger.debug("Warning: No mask provided. Assuming no missing values.")
            obs_mask = torch.zeros_like(obs_times, dtype=bool)

        locations = batch.get("location_times")  # shape [B, wlen_locs, 1]
        init_conditions = batch.get("initial_conditions")  # shape [B, D]

        B, wc, wlen, D = obs_values.shape

        # make single dimensional
        obs_values_processed = make_single_dim(obs_values)  # shape [B*D, wc, wlen, 1]
        obs_times_processed = repeat_for_dim(obs_times, D)  # shape [B*D, wc, wlen, 1]
        obs_mask_processed = repeat_for_dim(obs_mask, D)
        locations_processed = repeat_for_dim(locations, D)  # shape [B*D, wlen_locs, 1]
        init_conditions_processed = make_single_dim(init_conditions)  # shape [B*D, 1]

        # denoise observation values
        obs_values_processed = self.denoising_model(obs_values_processed, obs_mask_processed)

        # compute imputation solution
        fim_imp_input = {
            "observation_values": obs_values_processed,
            "observation_times": obs_times_processed,
            "observation_mask": obs_mask_processed,
            "location_times": locations_processed,
            "initial_conditions": init_conditions_processed,
        }

        output_imputation = self.fim_imputation(fim_imp_input)
        learnt_imp_solution = make_multi_dim(
            output_imputation["visualizations"]["imputation_window"]["learnt"], batch_size=B, process_dim=D
        )
        learnt_imp_drift = make_multi_dim(output_imputation["visualizations"]["drift"]["learnt"], batch_size=B, process_dim=D)
        learnt_imp_certainty = make_multi_dim(output_imputation["visualizations"]["drift"]["certainty"], batch_size=B, process_dim=D)

        # compute interpolation of observed windows
        fim_interpolation_input = {
            "coarse_grid_noisy_sample_paths": obs_values_processed.view(B * D * wc, wlen, 1),
            "coarse_grid_grid": obs_times_processed.view(B * D * wc, wlen, 1),
            "coarse_grid_observation_mask": obs_mask_processed.view(B * D * wc, wlen, 1),
            "fine_grid_grid": obs_times_processed.view(B * D * wc, wlen, 1),
        }

        output_interpolation = self.fim_imputation.fim_base(fim_interpolation_input, training=False)
        interpolation_solution = make_multi_dim(
            output_interpolation["visualizations"]["solution"]["learnt"].view(B * D, wc, wlen, 1),
            batch_size=B,
            process_dim=D,
        )
        interpolation_drift = make_multi_dim(
            output_interpolation["visualizations"]["drift"]["learnt"].view(B * D, wc, wlen, 1),
            batch_size=B,
            process_dim=D,
        )
        interpolation_certainty = make_multi_dim(
            output_interpolation["visualizations"]["drift"]["certainty"].view(B * D, wc, wlen, 1),
            batch_size=B,
            process_dim=D,
        )

        # TODO interpolate at boundaries

        return {
            "imputation_window": {
                "learnt": learnt_imp_solution,
                "target": batch.get("target_sample_path", None),
                "locations": locations,
                "drift": learnt_imp_drift,
                "drift_certainty": learnt_imp_certainty,
                "padding_mask_locations": batch.get("padding_mask_locations", None),
            },
            "observations": {
                "values": obs_values,
                "mask": obs_mask,
                "times": obs_times,
                "denoised_values": obs_values_processed.view(B, wc, wlen, D),
                "interpolation": interpolation_solution,
                "drift": interpolation_drift,
                "drift_certainty": interpolation_certainty,
            },
        }

    def new_stats(self):
        pass

    def loss(self):
        raise NotImplementedError

    def metric(self):
        raise NotImplementedError


ModelFactory.register("FIM_imputation", FIMImputation)
ModelFactory.register("FIM_imputation_windowed", FIMImputationWindowed)
