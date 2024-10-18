import logging
import math
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from transformers import BitsAndBytesConfig

from fim.data.utils import make_multi_dim, make_single_dim, reorder_windows_per_sample, split_into_windows
from fim.models.utils import load_model_from_checkpoint
from fim.utils.helper import create_class_instance
from fim.utils.metrics import compute_metrics

from ..trainers.mixed_precision import is_bfloat_supported
from ..utils.logging import RankLoggerAdapter
from .blocks import AModel, ModelFactory


class FIMODE(AModel):
    def __init__(
        self,
        time_encoding: dict,
        trunk_net: dict,
        branch_net: dict,
        combiner_net: dict,
        init_cond_net: dict,
        vector_field_net: dict,
        loss_configs: dict,
        normalization_time: Optional[dict] = None,
        normalization_values: Optional[dict] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_bf16: bool = False,
        device_map: Optional[str] = None,
        resume: bool = False,
        peft: Optional[dict] = None,
    ):
        super(FIMODE, self).__init__()
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

        if normalization_time is None:
            self.apply_normalization = False
        else:
            self.apply_normalization = True

        self._create_model(
            time_encoding=time_encoding,
            trunk_net=trunk_net,
            branch_net=branch_net,
            combiner_net=combiner_net,
            vector_field_net=vector_field_net,
            init_cond_net=init_cond_net,
            loss_configs=loss_configs,
            normalization_time=normalization_time,
            normalization_values=normalization_values,
        )

        self.to(self._device_map)

    def _create_model(
        self,
        time_encoding: dict,
        trunk_net: dict,
        branch_net: dict,
        combiner_net: dict,
        vector_field_net: dict,
        init_cond_net: dict,
        loss_configs: dict,
        normalization_time: Optional[dict] = None,
        normalization_values: Optional[dict] = None,
    ):
        if self.apply_normalization:
            self.normalization_time = create_class_instance(normalization_time.pop("name"), normalization_time)
            self.normalization_values = create_class_instance(normalization_values.pop("name"), normalization_values)

        self.time_encoding = create_class_instance(time_encoding.pop("name"), time_encoding)

        self.trunk_net = create_class_instance(trunk_net.pop("name"), trunk_net)

        self.branch_net = create_class_instance(branch_net.pop("name"), branch_net)

        if combiner_net.get("in_features") != 2 * combiner_net.get("out_features"):
            raise ValueError("The number of input features for the combiner_net must be twice the number of output features (latent dim).")

        self.combiner_net = create_class_instance(combiner_net.pop("name"), combiner_net)

        self.vector_field_net = create_class_instance(vector_field_net.pop("name"), vector_field_net)

        self.init_cond_net = create_class_instance(init_cond_net.pop("name"), init_cond_net)

        match loss_configs.get("ode_solver"):
            case "rk4":
                from fim.models.utils import rk4

                self.ode_solver = rk4
            case _:
                raise ValueError(f"ODE solver {loss_configs.get('ode_solver')} not supported.")

        self.loss_scale_drift = loss_configs.pop("loss_scale_drift")
        self.loss_scale_init_cond = loss_configs.pop("loss_scale_init_cond")
        self.loss_scale_unsuperv_loss = loss_configs.pop("loss_scale_unsuperv_loss")

    def forward(self, batch, schedulers: Optional[dict] = None, step: Optional[int] = None, training: bool = False) -> dict:
        """
        Args:
            batch (dict): input batch with entries (each torch.Tensor)
                 coarse_grid_(noisy_)sample_paths [B, T, 1] observation values. optionally with noise.
                 coarse_grid_grid [B, T, 1] observation times
                 coarse_grid_observation_mask [B, T, 1] observation mask, dtype: bool (0: value is observed, 1: value is masked out)
                 fine_grid_grid [B, L, 1] time points of fine grid on which the vector field is evaluated
                 fine_grid_sample_paths [B, L, 1] values of the fine grid
                 fine_grid_concept_values [B, L, 1] drift values of the fine grid (optional if training = False)
                with B: batch size, T: number of observation times, L: number of fine grid points (locations)
            training (bool): flag indicating if model is in training mode. Has an impact on the output.

        Returns:
            if training:
                dict: losses
            else:
                dict: losses (if target drift is provided), metrics, visualizations data
        """
        obs_mask = batch.get("coarse_grid_observation_mask").bool()

        if "coarse_grid_noisy_sample_paths" in batch:
            obs_values_origSpace = batch.get("coarse_grid_noisy_sample_paths")
        elif "coarse_grid_sample_paths" in batch:
            obs_values_origSpace = batch.get("coarse_grid_sample_paths")
        else:
            raise ValueError("coarse_grid_noisy_sample_paths or coarse_grid_sample_paths must be provided.")

        obs_times_origSpace = batch.get("coarse_grid_grid")
        fine_grid_grid_origSpace = batch.get("fine_grid_grid")
        fine_grid_sample_paths_origSpace = batch.get("fine_grid_sample_paths", None)

        fine_grid_drift_origSpace = batch.get("fine_grid_concept_values", None)
        if training and fine_grid_drift_origSpace is None:
            raise ValueError("fine_grid_concept_values must be provided for evaluation of loss.")

        if obs_values_origSpace.shape[-1] != 1:
            raise ValueError("Process dimension must be 1 in FIMODE Base model.")

        if obs_mask.all(dim=1).any():
            raise ValueError("Not allowed to have all values masked out.")

        if self.apply_normalization:
            (
                obs_values_normSpace,  # [B, T, 1]
                obs_times_normSpace,  # [B, T, 1]
                fine_grid_grid_normSpace,  # [B, L, 1]
                normalization_parameters,
            ) = self.normalize_input(
                obs_times=obs_times_origSpace, obs_values=obs_values_origSpace, obs_mask=obs_mask, loc_times=fine_grid_grid_origSpace
            )
        else:
            obs_values_normSpace = obs_values_origSpace
            obs_times_normSpace = obs_times_origSpace
            fine_grid_grid_normSpace = fine_grid_grid_origSpace
            normalization_parameters = None

        encoded_input_sequence = self._encode_input_sequence(
            obs_values=obs_values_normSpace, obs_times=obs_times_normSpace, obs_mask=obs_mask
        )  # Shape [B, 1, dim_latent]

        learnt_vector_field_concepts_normSpace = self._get_vector_field_concepts(
            location_times=fine_grid_grid_normSpace, encoded_sequence=encoded_input_sequence
        )  # Shape ([B, L, 1], [B, L, 1]) (normalized space)

        learnt_init_condition_concepts_normSpace = self._get_init_condition_concepts(
            encoded_sequence=encoded_input_sequence, t_0=fine_grid_grid_normSpace[:, :1, :]
        )  # Shape ([B, 1], [B, 1]) (normalized space)

        # renormalize vector field & initial condition distribution parameters
        if self.apply_normalization:
            learnt_vector_field_concepts_origSpace = self._renormalize_vector_field_params(
                vector_field_concepts=learnt_vector_field_concepts_normSpace,
                normalization_parameters=normalization_parameters,
            )  # Shape ([B, L, 1], [B, L, 1])
            learnt_init_condition_concepts_origSpace = self._renormalize_init_condition_params(
                learnt_init_condition_concepts_normSpace, normalization_parameters
            )
        else:
            learnt_vector_field_concepts_origSpace = learnt_vector_field_concepts_normSpace
            learnt_init_condition_concepts_origSpace = learnt_init_condition_concepts_normSpace

        if fine_grid_drift_origSpace is not None:
            losses = self.loss(
                vector_field_concepts=learnt_vector_field_concepts_origSpace,
                init_condition_concepts=learnt_init_condition_concepts_origSpace,
                target_drift_fine_grid=fine_grid_drift_origSpace,
                fine_grid_sample_paths=fine_grid_sample_paths_origSpace,
                fine_grid_grid=fine_grid_grid_origSpace,
            )
        else:
            losses = {}

        model_output = {"losses": losses}

        if not training:
            metrics, solution = self.new_stats(
                normalized_fine_grid_grid=fine_grid_grid_normSpace,
                init_condition_concepts=learnt_init_condition_concepts_origSpace,
                encoded_sequence=encoded_input_sequence,
                normalization_parameters=normalization_parameters,
                fine_grid_sample_path=fine_grid_sample_paths_origSpace,
            )
            model_output["metrics"] = metrics

            model_output["visualizations"] = {
                "fine_grid_grid": fine_grid_grid_origSpace,
            }

            # visualization data of all samples
            model_output["visualizations"]["solution"] = {
                "learnt": solution,
                "target": fine_grid_sample_paths_origSpace,
                "observation_times": obs_times_origSpace,
                "observation_values": obs_values_origSpace,
                "observation_mask": obs_mask,
            }
            model_output["visualizations"]["drift"] = {
                "learnt": learnt_vector_field_concepts_origSpace[0],
                "target": fine_grid_drift_origSpace,
                "certainty": torch.exp(learnt_vector_field_concepts_origSpace[1]),
            }
            model_output["visualizations"]["init_condition"] = {
                "learnt": learnt_init_condition_concepts_origSpace[0],
                "target": fine_grid_sample_paths_origSpace[..., 0, :] if fine_grid_sample_paths_origSpace is not None else None,
                "certainty": torch.exp(learnt_init_condition_concepts_origSpace[1]),
            }
        else:
            model_output["metrics"] = {}
            model_output["visualizations"] = {}

        return model_output

    def _encode_input_sequence(
        self,
        obs_times,
        obs_values,
        obs_mask,
    ):
        """
        Encode input sequence (observation values and times) using the branch net framework of DeepONet.

        Encode observation times, concatenate with normalized observation values and pass through branch net (sequence-to-sequence model).

        Args:
            obs_times (torch.Tensor): observation times [B, T, 1]
            obs_values (torch.Tensor): observation values [B, T, 1]
            obs_mask (torch.Tensor): observation mask [B, T, 1]

        Returns:
            torch.Tensor: encoded input sequence [B, 1, dim_latent]
        """
        # encode times
        encoded_obs_times = self.time_encoding(grid=obs_times)  # Shape [B, T, dim_time]

        # concatenate time encoding with normalized observation values.
        obs_input_latent = torch.cat(
            [
                encoded_obs_times,  # Shape [B, T, dim_time],
                obs_values,  # Shape [B, T, 1]
            ],
            dim=-1,
        )  # Shape [B, T, dim_time + 1]

        encoded_input_sequence = self.branch_net(x=obs_input_latent, key_padding_mask=obs_mask)  # Shape [B, 1, dim_latent]

        return encoded_input_sequence

    def _get_vector_field_concepts(
        self,
        location_times: torch.Tensor,
        encoded_sequence: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mean and log standard deviation of the vector field at given grid times.

        Encode location times, pass through trunk net, combine with branch output in combiner net and compute mean and log_std of the vector field.

        Args:
            location_times (torch.Tensor): fine grid time points [B, L, 1]
            encoded_sequence (torch.Tensor): Encoded input sequence (as issued from branch net) [B, 1, dim_latent]

        Returns:
            tuple: mean and log standard deviation of the vector field ([B, L, 1], [B, L, 1])
        """
        # encode grid times
        times_encoded = self.time_encoding(grid=location_times)  # Shape [B, L, dim_time]

        trunk_out = self.trunk_net(times_encoded)  # Shape [B, L, dim_latent]

        # concat branch and trunk output: append branch output to each time step of trunk output
        combiner_in = torch.cat(
            [
                trunk_out,  # Shape [B, L, dim_latent]
                encoded_sequence.repeat(1, trunk_out.shape[1], 1),  # Shape [B, L, dim_latent]
            ],
            dim=-1,
        )  # Shape [B, L, 2*dim_latent]
        combiner_out = self.combiner_net(combiner_in)  # Shape [B, L, dim_latent]

        # compute mean and log_std of the vector field at every location
        vector_field_concepts = self.vector_field_net(combiner_out)  # Shape [B, L, 2]

        # split into mean and log variance
        vector_field_mean, vector_field_log_std = torch.split(vector_field_concepts, 1, dim=-1)  # Shape each [B, L, 1]

        return vector_field_mean, vector_field_log_std

    def _get_init_condition_concepts(self, encoded_sequence: torch.Tensor, t_0: torch.Tensor) -> tuple:
        """
        Compute mean and log standard deviation of the initial condition.

        Args:
            encoded_sequence (torch.Tensor): embedding of the input sequence (as issued by branch net) [B, 1, dim_latent]
            t_0 (torch.Tensor): initial time point (normalized space) [B, 1, 1]

        Returns:
            tuple: mean and log standard deviation of the initial condition ([B, 1], [B, 1])
        """
        init_condition_input = torch.concat([encoded_sequence, t_0], dim=-1)  # shape [B, 1, dim_latent + 1]
        init_condition_concepts = self.init_cond_net(init_condition_input)  # Shape [B, 1, 2]

        init_condition_concepts = init_condition_concepts.squeeze(1)  # Shape [B, 2]

        # split into mean and log variance
        init_condition_mean, init_condition_log_std = torch.split(init_condition_concepts, 1, dim=-1)  # Shape each [B, 1]

        return init_condition_mean, init_condition_log_std

    def loss(
        self,
        vector_field_concepts: tuple,
        init_condition_concepts: tuple,
        target_drift_fine_grid: torch.Tensor,
        fine_grid_sample_paths: torch.Tensor,
        fine_grid_grid: torch.Tensor,
    ) -> dict:
        """
        Compute the loss of the FIMODE model (in original space).

        The loss consists of supervised losses
            - negative log-likelihood of the vector field values at fine grid points
            - negative log-likelihood of the initial condition
        and an unsupervised loss
            - one-step ahead prediction loss.
        The total loss is a weighted sum of all losses. The weights are defined in the loss_configs. (loss_scale_drift, loss_scale_init_cond, loss_scale_unsuperv_loss)

        Args:
            vector_field_concepts (tuple): mean and log standard deviation of the vector field concepts (in original space) ([B, L, D], [B, L, D])
            init_condition_concepts (tuple): mean and log standard deviation of the initial condition concepts (in original space) ([B, D], [B, D])
            target_drift_fine_grid (torch.Tensor): target values (in original space) [B, L, D]
            fine_grid_grid (torch.Tensor): fine grid time points (in original space) [B, L, 1]

        Returns:
            dict: llh_drift, llh_init_cond, unsupervised_loss, loss = weighted sum of all losses
        """
        # supervised losses
        learnt_mean_drift, learnt_log_std_drift = vector_field_concepts
        learnt_var_drift = torch.exp(learnt_log_std_drift) ** 2

        nllh_drift_avg = torch.mean(1 / 2 * (target_drift_fine_grid - learnt_mean_drift) ** 2 / learnt_var_drift + learnt_log_std_drift)

        learnt_mean_init_cond, learnt_log_std_init_cond = init_condition_concepts
        learnt_var_init_cond = torch.exp(learnt_log_std_init_cond) ** 2

        target_init_cond = fine_grid_sample_paths[..., 0, :]

        # assert (
        #     learnt_mean_init_cond.shape == target_init_cond.shape == learnt_var_init_cond.shape
        # ), "Shapes of initial condition concepts do not match. Expected ? "

        nllh_init_cond_avg = torch.mean(
            1 / 2 * (target_init_cond - learnt_mean_init_cond) ** 2 / learnt_var_init_cond + learnt_log_std_init_cond
        )

        # unsupervised loss (original space)
        step_size_fine_grid = fine_grid_grid[..., 1:, :] - fine_grid_grid[..., :-1, :]

        # unsupervised_loss[i] = (target_path[i]-target_path[i-1] - drift[i-1]*step_size)^2
        unsupervised_loss = torch.mean(
            torch.sum(
                (
                    fine_grid_sample_paths[..., 1:, :]
                    - fine_grid_sample_paths[..., :-1, :]
                    - learnt_mean_drift[..., :-1, :] * step_size_fine_grid
                )
                ** 2,
                dim=-2,
            )
        )

        total_loss = (
            self.loss_scale_drift * nllh_drift_avg
            + self.loss_scale_init_cond * nllh_init_cond_avg
            + self.loss_scale_unsuperv_loss * unsupervised_loss
        )

        return {
            "llh_drift": nllh_drift_avg,
            "llh_init_cond": nllh_init_cond_avg,
            "unsupervised_loss": unsupervised_loss,
            "loss": total_loss,
        }

    def get_solution(
        self,
        fine_grid: torch.Tensor,
        init_condition: torch.Tensor,
        branch_out: torch.Tensor,
        normalization_parameters: dict,
    ) -> torch.Tensor:
        """
        Compute the solution of the ODE using the defined ode_solver.

        Args:
            fine_grid (torch.Tensor): fine grid time points [B, L] (normalized space)
            init_condition (torch.Tensor): initial condition [B, D] (original space)
            branch_out (torch.Tensor): output of the branch network [B, 1, dim_latent] (normalized space)
            normalization_parameters (dict): normalization parameters for time and values

        Returns:
            solution: torch.Tensor: solution at fine grid points [B, L, D] (original space)
        """
        B, L = fine_grid.shape[:-1]

        # need evaluations of FIMODE at fine grid points and one point in between each fine grid point -> add one point in between
        # get mid points between fine grid points
        mid_points = (fine_grid[..., 1:, :] + fine_grid[..., :-1, :]) / 2  # Shape [B, L-1, 1]
        # concat alternating fine grid points and mid points
        super_fine_grid_grid_normSpace = torch.zeros(B, 2 * L - 1, 1, device=fine_grid.device, dtype=fine_grid.dtype)
        super_fine_grid_grid_normSpace[:, ::2] = fine_grid
        super_fine_grid_grid_normSpace[:, 1::2] = mid_points

        # compute drift at super fine grid points (in normalized space)
        learnt_vector_field_concepts_normSpace = self._get_vector_field_concepts(
            location_times=super_fine_grid_grid_normSpace, encoded_sequence=branch_out
        )  # [B, 2*L-1, D]

        # unnormalize learnt drift & underlying time grid
        if self.apply_normalization:
            learnt_vector_field_concepts_origSpace = self._renormalize_vector_field_params(
                normalization_parameters=normalization_parameters,
                vector_field_concepts=learnt_vector_field_concepts_normSpace,
            )
            super_fine_grid_grid_origSpace = self._renormalize_time(
                grid_grid=super_fine_grid_grid_normSpace,
                norm_params=normalization_parameters,
            )
        else:
            super_fine_grid_grid_origSpace = super_fine_grid_grid_normSpace
            learnt_vector_field_concepts_origSpace = learnt_vector_field_concepts_normSpace

        # compute solution using ode solver (unnormalized space)
        solution = self.ode_solver(
            super_fine_grid_grid=super_fine_grid_grid_origSpace,
            super_fine_grid_drift=learnt_vector_field_concepts_origSpace[0],
            initial_condition=init_condition,
        )  # [B, L, D]
        assert solution.shape == (B, L, 1)

        return solution

    def normalize_input(self, obs_values: torch.Tensor, obs_times: torch.Tensor, obs_mask: torch.Tensor, loc_times: torch.Tensor) -> tuple:
        """
        Apply normalization to observation values and times independently.

        Args:
            obs_values (torch.Tensor): observation values
            obs_times (torch.Tensor): observation times
            obs_mask (torch.Tensor): observation mask
            loc_times (torch.Tensor): location times

        Returns:
            tuple: normalized observation values (torch.Tensor),
                   normalized observation times (torch.Tensor),
                   normalized location times (torch.Tensor),
                   normalization parameters (dict) with keys "norm_params_time" and "norm_params_values"
        """
        # normalize time
        obs_times_normSpace, norm_params_time = self.normalization_time(obs_times, obs_mask)

        # normalize location grid with same norm parameters as observation times
        loc_times_normSpace, _ = self.normalization_time(loc_times, obs_mask, norm_params_time)

        # normalize values
        obs_values_normSpace, norm_params_values = self.normalization_values(obs_values, obs_mask)

        normalization_parameters = {
            "norm_params_time": norm_params_time,
            "norm_params_values": norm_params_values,
        }

        return (
            obs_values_normSpace,
            obs_times_normSpace,
            loc_times_normSpace,
            normalization_parameters,
        )

    def normalize_input_old(
        self, obs_values: torch.Tensor, obs_times: torch.Tensor, obs_mask: torch.Tensor, loc_times: torch.Tensor
    ) -> tuple:
        """
        Apply normalization to observation values and times independently.

        Args:
            obs_values (torch.Tensor): observation values
            obs_times (torch.Tensor): observation times
            obs_mask (torch.Tensor): observation mask
            loc_times (torch.Tensor): location times

        Returns:
            tuple: normalized observation values (torch.Tensor),
                   normalized observation times (torch.Tensor),
                   normalized location times (torch.Tensor),
                   normalization parameters (dict) with keys "norm_params_time" and "norm_params_values"
        """

        def get_norm_params(data: torch.Tensor, mask: torch.Tensor) -> tuple:
            """
            Compute normalization parameters for min-max scaling (per sample and dimension/feature).

            Args:
                data (torch.Tensor): data to normalize [B, T, D]
                mask (torch.Tensor): observation mask [B, T, 1]
            Returns:
                tuple: min (torch.Tensor, [B, D]), range (torch.Tensor, [B, D])
            """
            # get min and max values for each feature dimension per batch entry
            data_min = torch.amin(data.masked_fill(mask, float("inf")), dim=1)  # Shape [B, D]
            data_max = torch.amax(data.masked_fill(mask, float("-inf")), dim=1)  # Shape [B, D]

            # compute range, add small value to avoid division by zero
            data_range = data_max - data_min + 1e-6

            return data_min, data_range

        def normalize(data: torch.Tensor, norm_params: tuple) -> torch.Tensor:
            """
            Normalize values using min-max scaling.

            Args:
                data (torch.Tensor): data to normalized [B, T, D]
                norm_params (tuple): min and range of the values ([B, D], [B, D])

            Returns:
                torch.Tensor: normalized values [B, T, D]
            """
            data_min, data_range = norm_params

            # unsqueeze to allow broadcasting
            data_min = data_min.unsqueeze(1)  # Shape [B, 1, D]
            data_range = data_range.unsqueeze(1)  # Shape [B, 1, D]

            return (data - data_min) / data_range

        obs_values_norm_params = get_norm_params(obs_values, obs_mask)  # ([B, D], [B, D])
        obs_times_norm_params = get_norm_params(obs_times, obs_mask)  # ([B, 1], [B, 1])

        normalized_obs_values = normalize(obs_values, obs_values_norm_params)  # [B, T, D]
        normalized_obs_times = normalize(obs_times, obs_times_norm_params)  # [B, T, 1]
        normalized_loc_times = normalize(loc_times, obs_times_norm_params)  # [B, L, 1]

        normalization_parameters = {
            "obs_values_min": obs_values_norm_params[0],
            "obs_values_range": obs_values_norm_params[1],
            "obs_times_min": obs_times_norm_params[0],
            "obs_times_range": obs_times_norm_params[1],
        }
        return (
            normalized_obs_values,
            normalized_obs_times,
            normalized_loc_times,
            normalization_parameters,
        )

    def _renormalize_vector_field_params(
        self,
        vector_field_concepts: tuple,
        normalization_parameters: dict,
    ) -> Union[tuple, torch.Tensor]:
        """
        Rescale vector field (mean and log_std) to original scale based on normalization parameters of observation values and times.

        Args:
            vector_field_concepts (tuple): mean and log standard deviation of the vector field distribution ([B, L, 1], [B, L, 1])
                log std is optional.
            normalization_parameters (dict): holding all normalization parameters, including obs_values_range and obs_times_range

        Returns:
            if drift_log_std != None: return tuple: rescaled mean and log standard deviation of the concept distribution ([B, L, 1], [B, L, 1])
            if drift_log_std == None: return torch.Tensor: rescaled mean of the concept distribution ([B, L, 1])
        """
        drift_mean, drift_log_std = vector_field_concepts

        shape = drift_mean.shape  # [B, L, 1]

        reversion_factor_time = self.normalization_time.get_reversion_factor(normalization_parameters.get("norm_params_time"))
        reversion_factor_values = self.normalization_values.get_reversion_factor(normalization_parameters.get("norm_params_values"))

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
            drift_log_std = (
                drift_log_std + torch.log(reversion_factor_values_view) - torch.log(reversion_factor_times_view)
            )  # Shape [B, L, 1]
            assert drift_log_std.shape == shape
            return drift_mean, drift_log_std

        else:
            return drift_mean

    def _renormalize_init_condition_params(self, init_cond_dist_params: tuple, normalization_parameters: dict) -> tuple:
        """
        Rescale the initial condition (mean and log_std) to original space based on observation values normalization parameters.

        Args:
            init_cond_dist_params (tuple): learnt mean and variance of the initial condition ([B,1], [B, 1])
            normalization_parameters (dict): holding all normalization parameters including obs_values_min and obs_values_range

        Returns:
            tuple: rescaled mean and log standard deviation of the initial condition ([B, 1], [B, 1])
        """
        init_cond_mean, init_cond_log_std = init_cond_dist_params  # [B,1], [B, 1]
        norm_params_values = normalization_parameters.get("norm_params_values")  # ([B, 1, 1], [B, 1, 1])

        # rescale mean and log std
        init_cond_mean_origSpace = self.normalization_values.revert_normalization(
            x=init_cond_mean, data_concepts=norm_params_values
        )  # Shape [B, 1]
        init_cond_log_std_origSpace = self.normalization_values.revert_normalization(
            x=init_cond_log_std, data_concepts=norm_params_values, log_scale=True
        )  # Shape [B, 1]

        assert init_cond_mean_origSpace.shape == init_cond_mean.shape

        return init_cond_mean_origSpace, init_cond_log_std_origSpace

    def _renormalize_time(self, grid_grid: torch.Tensor, norm_params: dict) -> torch.Tensor:
        """Revert time normalization."""
        return self.normalization_time.revert_normalization(grid_grid, norm_params.get("norm_params_time"))

    def _renormalize_init_condition_params_old(self, init_cond_dist_params: tuple, normalization_parameters: dict) -> tuple:
        """
        Rescale the initial condition (mean and log_std) to original space based on observation values normalization parameters.

        Args:
            init_cond_dist_params (tuple): learnt mean and variance of the initial condition ([B,1], [B, 1])
            normalization_parameters (dict): holding all normalization parameters including obs_values_min and obs_values_range

        Returns:
            tuple: rescaled mean and log standard deviation of the initial condition ([B, 1], [B, 1])
        """
        init_cond_mean, init_cond_log_std = init_cond_dist_params  # [B,1], [B, 1]
        obs_values_min = normalization_parameters.get("obs_values_min")  # [B, 1]
        obs_values_range = normalization_parameters.get("obs_values_range")  # [B, 1]

        # rescale mean and log std
        init_cond_mean = init_cond_mean * obs_values_range + obs_values_min  # Shape [B, 1]
        init_cond_log_std = init_cond_log_std + torch.log(obs_values_range)  # Shape [B, 1]

        return init_cond_mean, init_cond_log_std

    def _renormalize_time_old(self, grid_grid: torch.Tensor, normalization_parameters: dict) -> torch.Tensor:
        """Revert min-max scaling of time points."""
        times_min = normalization_parameters.get("obs_times_min")
        times_range = normalization_parameters.get("obs_times_range")

        grid_dim = grid_grid.dim()

        if grid_dim == 3:
            grid_grid = grid_grid.squeeze(-1)

        grid_grid = grid_grid * times_range + times_min

        if grid_dim == 3:
            grid_grid = grid_grid.unsqueeze(-1)

        return grid_grid

    def metric(self, y: Any, y_target: Any) -> Dict:
        # compute MSE, RMSE, MAE, R2 score
        metrics = compute_metrics(y, y_target)
        return metrics

    def new_stats(
        self,
        normalized_fine_grid_grid: torch.Tensor,
        init_condition_concepts: tuple,
        encoded_sequence: torch.Tensor,
        normalization_parameters: dict,
        fine_grid_sample_path: torch.Tensor,
    ) -> tuple[dict, torch.Tensor]:
        """
        Compute metrics betwenn target and predicted solution at fine grid points.

        Args:
            normalized_fine_grid_grid (torch.Tensor): fine grid time points [B, L, 1]
            init_condition_concepts (tuple): mean and log standard deviation of the initial condition ([B, 1], [B, 1])
            encoded_sequence (torch.Tensor): output of the branch network [B, 1, dim_latent]
            normalization_parameters (dict): normalization parameters for time and values
            fine_grid_sample_path (torch.Tensor): target values at fine grid points [B, L, D]

        Returns:
            tuple: metrics (dict), solution (torch.Tensor)
        """
        # get solution
        solution = self.get_solution(
            fine_grid=normalized_fine_grid_grid,
            init_condition=init_condition_concepts[0],
            branch_out=encoded_sequence,
            normalization_parameters=normalization_parameters,
        )  # [B, L, D], [B, L, D] (original space)

        # get metrics
        # metrics = self.metric(
        #     y=solution,
        #     y_target=fine_grid_sample_path,
        # )
        metrics = {}

        return metrics, solution

    def is_peft(self) -> bool:
        return self.peft is not None and self.peft["method"] is not None


class FIMWindowed(AModel):
    """
    Wrapper for FIMODE model to allow multidimensional and/or longer input sequences.

    Denoising model is optional. If not provided, the input is not denoised.
    """

    def __init__(self, fim_base: Union[str, Path, FIMODE], denoising_model: dict, window_count: int, overlap: float, **kwargs):
        super().__init__(**kwargs)

        self.window_count = window_count
        if not 0 <= overlap < 1:
            raise ValueError("Overlap (percentage) must be between 0 and 1.")
        self.overlap = overlap

        self._create_model(
            fim_model=fim_base,
            denoising_model=denoising_model,
        )

    def _create_model(self, fim_model: Union[str, Path, FIMODE], denoising_model: Optional[dict] = None):
        if denoising_model is None:
            # create dummy model that does nothing
            self.denoising_model = lambda x, mask: x
        else:
            self.denoising_model = create_class_instance(
                denoising_model.pop("name"),
                denoising_model,
            )
        self.fim_base: FIMODE = load_model_from_checkpoint(fim_model, module=FIMODE) if isinstance(fim_model, (str, Path)) else fim_model

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
        if (window_size := int((ws := (math.ceil(max_sequence_length / self.window_count))) + self.overlap * ws)) >= 128:
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
        denoised_observation_values = self.denoising_model(noisy_observation_values_processed, observation_mask_processed)
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
        paths_reordered = reorder_windows_per_sample(paths, window_count=self.window_count, batch_size=batch_size, process_dim=process_dim)
        times_reordered = reorder_windows_per_sample(times, window_count=self.window_count, batch_size=batch_size, process_dim=process_dim)

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
        paths_merged_multi_dim = paths_merged_multi_dim[:, self.overlap_size : last_index, :]  # shape [B, max_sequence_length, D]
        times_merged_multi_dim = times_merged_multi_dim[:, self.overlap_size : last_index, :]  # shape [B, max_sequence_length, D]

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

    def _linear_interpolation_windows(self, paths: torch.Tensor, times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

            combined_values, combined_times = self._merge_two_windows(combined_values, combined_times, values_b, times_b)

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


ModelFactory.register("FIMODE", FIMODE)
ModelFactory.register("FIMWindowed", FIMWindowed)
