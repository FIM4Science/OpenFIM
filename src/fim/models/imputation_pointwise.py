import logging
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import einops
import optree
import torch
from torch import Tensor
from transformers import PretrainedConfig

from fim.models.sde import InstanceNormalization, forward_fill_masked_values
from fim.models.utils import load_model_from_checkpoint
from fim.utils.helper import create_class_instance
from fim.utils.metrics import compute_metrics

from ..utils.logging import RankLoggerAdapter
from .blocks import AModel, ModelFactory


class FIMImpPointBaseConfig(PretrainedConfig):
    model_type = "FIMImpPointBase"

    def __init__(
        self,
        time_encoding: dict = None,
        trunk_net: dict = None,
        branch_net: dict = None,
        combiner_net: dict = None,
        init_cond_net: dict = None,
        vector_field_net: dict = None,
        loss_configs: dict = None,
        normalization_time: Optional[dict] = None,
        normalization_values: Optional[dict] = None,
        **kwargs,
    ):
        self.time_encoding = time_encoding
        self.trunk_net = trunk_net
        self.branch_net = branch_net
        self.combiner_net = combiner_net
        self.init_cond_net = init_cond_net
        self.vector_field_net = vector_field_net
        self.loss_configs = loss_configs
        self.normalization_time = normalization_time
        self.normalization_values = normalization_values

        super().__init__(**kwargs)


class FIMImpPointBase(AModel):
    config_class = FIMImpPointBaseConfig

    def __init__(self, config: FIMImpPointBaseConfig, **kwargs):
        super(FIMImpPointBase, self).__init__(config, **kwargs)
        self.logger = RankLoggerAdapter(logging.getLogger(self.__class__.__name__))

        if self.config.normalization_time is None:
            self.apply_normalization = False
        else:
            self.apply_normalization = True

        self._create_model()

    def _create_model(self):
        config = deepcopy(self.config)
        if self.apply_normalization:
            self.normalization_time = create_class_instance(config.normalization_time.pop("name"), config.normalization_time)
            self.normalization_values = create_class_instance(config.normalization_values.pop("name"), config.normalization_values)

        self.time_encoding = create_class_instance(config.time_encoding.pop("name"), config.time_encoding)

        self.trunk_net = create_class_instance(config.trunk_net.pop("name"), config.trunk_net)

        self.branch_net = create_class_instance(config.branch_net.pop("name"), config.branch_net)

        if config.combiner_net.get("in_features") != 2 * config.combiner_net.get("out_features"):
            raise ValueError("The number of input features for the combiner_net must be twice the number of output features (latent dim).")

        self.combiner_net = create_class_instance(config.combiner_net.pop("name"), config.combiner_net)

        self.vector_field_net = create_class_instance(config.vector_field_net.pop("name"), config.vector_field_net)

        self.init_cond_net = create_class_instance(config.init_cond_net.pop("name"), config.init_cond_net)

        match config.loss_configs.get("ode_solver"):
            case "rk4":
                from fim.models.utils import rk4

                self.ode_solver = rk4
            case _:
                raise ValueError(f"ODE solver {config.loss_configs.get('ode_solver')} not supported.")

        self.loss_scale_drift = config.loss_configs.pop("loss_scale_drift")
        self.loss_scale_init_cond = config.loss_configs.pop("loss_scale_init_cond")
        self.loss_scale_unsuperv_loss = config.loss_configs.pop("loss_scale_unsuperv_loss")

    def forward(self, batch, schedulers: Optional[dict] = None, step: Optional[int] = None, training: bool = False) -> dict:
        """
        Args:
            batch (dict): input batch with entries (each Tensor)
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
            raise ValueError("Process dimension must be 1 in FIMImpPointBase Base model.")

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
            obs_times (Tensor): observation times [B, T, 1]
            obs_values (Tensor): observation values [B, T, 1]
            obs_mask (Tensor): observation mask [B, T, 1]

        Returns:
            Tensor: encoded input sequence [B, 1, dim_latent]
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
        location_times: Tensor,
        encoded_sequence: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Compute mean and log standard deviation of the vector field at given grid times.

        Encode location times, pass through trunk net, combine with branch output in combiner net and compute mean and log_std of the vector field.

        Args:
            location_times (Tensor): fine grid time points [B, L, 1]
            encoded_sequence (Tensor): Encoded input sequence (as issued from branch net) [B, 1, dim_latent]

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

    def _get_init_condition_concepts(self, encoded_sequence: Tensor, t_0: Tensor) -> tuple:
        """
        Compute mean and log standard deviation of the initial condition.

        Args:
            encoded_sequence (Tensor): embedding of the input sequence (as issued by branch net) [B, 1, dim_latent]
            t_0 (Tensor): initial time point (normalized space) [B, 1, 1]

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
        target_drift_fine_grid: Tensor,
        fine_grid_sample_paths: Tensor,
        fine_grid_grid: Tensor,
    ) -> dict:
        """
        Compute the loss of the FIMImpPointBase model (in original space).

        The loss consists of supervised losses
            - negative log-likelihood of the vector field values at fine grid points
            - negative log-likelihood of the initial condition
        and an unsupervised loss
            - one-step ahead prediction loss.
        The total loss is a weighted sum of all losses. The weights are defined in the loss_configs. (loss_scale_drift, loss_scale_init_cond, loss_scale_unsuperv_loss)

        Args:
            vector_field_concepts (tuple): mean and log standard deviation of the vector field concepts (in original space) ([B, L, D], [B, L, D])
            init_condition_concepts (tuple): mean and log standard deviation of the initial condition concepts (in original space) ([B, D], [B, D])
            target_drift_fine_grid (Tensor): target values (in original space) [B, L, D]
            fine_grid_grid (Tensor): fine grid time points (in original space) [B, L, 1]

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
        fine_grid: Tensor,
        init_condition: Tensor,
        branch_out: Tensor,
        normalization_parameters: dict,
    ) -> Tensor:
        """
        Compute the solution of the ODE using the defined ode_solver.

        Args:
            fine_grid (Tensor): fine grid time points [B, L] (normalized space)
            init_condition (Tensor): initial condition [B, D] (original space)
            branch_out (Tensor): output of the branch network [B, 1, dim_latent] (normalized space)
            normalization_parameters (dict): normalization parameters for time and values

        Returns:
            solution: Tensor: solution at fine grid points [B, L, D] (original space)
        """
        B, L = fine_grid.shape[:-1]

        # need evaluations of FIMImpPointBase at fine grid points and one point in between each fine grid point -> add one point in between
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

    def normalize_input(self, obs_values: Tensor, obs_times: Tensor, obs_mask: Tensor, loc_times: Tensor) -> tuple:
        """
        Apply normalization to observation values and times independently.

        Args:
            obs_values (Tensor): observation values
            obs_times (Tensor): observation times
            obs_mask (Tensor): observation mask
            loc_times (Tensor): location times

        Returns:
            tuple: normalized observation values (Tensor),
                   normalized observation times (Tensor),
                   normalized location times (Tensor),
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

    def _renormalize_vector_field_params(
        self,
        vector_field_concepts: tuple,
        normalization_parameters: dict,
    ) -> Union[tuple, Tensor]:
        """
        Rescale vector field (mean and log_std) to original scale based on normalization parameters of observation values and times.

        Args:
            vector_field_concepts (tuple): mean and log standard deviation of the vector field distribution ([B, L, 1], [B, L, 1])
                log std is optional.
            normalization_parameters (dict): holding all normalization parameters, including obs_values_range and obs_times_range

        Returns:
            if drift_log_std != None: return tuple: rescaled mean and log standard deviation of the concept distribution ([B, L, 1], [B, L, 1])
            if drift_log_std == None: return Tensor: rescaled mean of the concept distribution ([B, L, 1])
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

    def _renormalize_time(self, grid_grid: Tensor, norm_params: dict) -> Tensor:
        """Revert time normalization."""
        return self.normalization_time.revert_normalization(grid_grid, norm_params.get("norm_params_time"))

    def metric(self, y: Any, y_target: Any) -> Dict:
        # compute MSE, RMSE, MAE, R2 score
        metrics = compute_metrics(y, y_target)
        return metrics

    def new_stats(
        self,
        normalized_fine_grid_grid: Tensor,
        init_condition_concepts: tuple,
        encoded_sequence: Tensor,
        normalization_parameters: dict,
        fine_grid_sample_path: Tensor,
    ) -> tuple[dict, Tensor]:
        """
        Compute metrics betwenn target and predicted solution at fine grid points.

        Args:
            normalized_fine_grid_grid (Tensor): fine grid time points [B, L, 1]
            init_condition_concepts (tuple): mean and log standard deviation of the initial condition ([B, 1], [B, 1])
            encoded_sequence (Tensor): output of the branch network [B, 1, dim_latent]
            normalization_parameters (dict): normalization parameters for time and values
            fine_grid_sample_path (Tensor): target values at fine grid points [B, L, D]

        Returns:
            tuple: metrics (dict), solution (Tensor)
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


@optree.dataclasses.dataclass(namespace="fim_imp_pointwise", eq=False)
class ImputationConcepts:
    """
    Stores Imputation concepts, i.e. evaluation times, initial conditions, the vector field (derivative) and the
    interpolating function, i.e. reconstruction.
    A flag keeps track of the normalization status of these concepts.
    """

    evaluation_times: Tensor  # Shape: [B, ..., 1]
    reconstructed_values: Tensor  # Shape: [B, ..., D]
    init_cond_mean: Tensor  # Shape: [B, ..,  D], one less axis than rest
    init_cond_log_std: Tensor  # Shape: [B, ..,  D], one less axis than rest
    vector_field_mean: Tensor  # Shape: [B, ..., D]
    vector_field_log_std: Tensor  # Shape: [B, ..., D]
    normalized: bool = optree.dataclasses.field(default=False, pytree_node=False)

    def __eq__(self, other: object) -> bool:
        """
        Define equality by closeness of attributes.
        """

        rtol: float = 1e-5
        atol: float = 1e-6

        is_equal: bool = True

        is_equal = is_equal and self._allclose(self.evaluation_times, other.evaluation_times, atol=atol, rtol=rtol)
        is_equal = is_equal and self._allclose(self.reconstructed_values, other.reconstructed_values, atol=atol, rtol=rtol)
        is_equal = is_equal and self._allclose(self.init_cond_mean, other.init_cond_mean, atol=atol, rtol=rtol)
        is_equal = is_equal and self._allclose(self.init_cond_log_std, other.init_cond_log_std, atol=atol, rtol=rtol)
        is_equal = is_equal and self._allclose(self.vector_field_mean, other.vector_field_mean, atol=atol, rtol=rtol)
        is_equal = is_equal and self._allclose(self.vector_field_log_std, other.vector_field_log_std, atol=atol, rtol=rtol)

        is_equal = is_equal and (self.normalized == other.normalized)

        return is_equal

    @staticmethod
    def _allclose(input: Tensor | None, other: Tensor | None, atol: float, rtol=float) -> bool:
        """
        Wraps torch.allclose to handle potential Nones.
        """

        if input is None and other is None:
            return True

        elif input is None or other is None:
            return False

        else:
            return torch.allclose(input, other, atol=atol, rtol=rtol)

    def _assert_shape(self) -> None:
        """
        Assert that all attributes are of same shape.
        """

        shapes_func_evals = []

        for field in [self.reconstructed_values, self.vector_field_mean, self.vector_field_log_std]:
            if field is not None:
                shapes_func_evals.append(field.shape)

        assert len(set(shapes_func_evals)) <= 1, (
            f"Shapes of fields do not match. \
                    reconstructed_values: {self.reconstructed_values.shape},\
                    vector_field_mean: {self.vector_field_mean.shape}, vector_field_log_std: {self.vector_field_log_std.shape}."
        )

        assert self.init_cond_mean is not None and self.init_cond_log_std is not None, (
            f"init_cond_mean is not None: {self.init_cond_mean is not None}, \
                    init_cond_log_std is not None: {self.init_cond_log_std is not None}"
        )

        assert self.init_cond_mean.shape == self.init_cond_log_std.shape, (
            f"init_cond_mean: {self.init_cond_mean.shape}, init_cond_log_std: {self.init_cond_log_std.shape}"
        )

    def _log_std_normalization_map(
        self, norm: InstanceNormalization, norm_stats: Any, log_std: Tensor, mean: Tensor, eps: float = 1e-8
    ) -> Tensor:
        """
        Transforms a log-scaled standard deviation under the forward normalization map.

        Formula: log_std_out = log_std_in + log(|f'(mean)|)

        Args:
            log_std (Tensor): The input log-standard deviation. Shape: [B, ..., D]
            mean (Tensor): The input mean where the derivative is evaluated. Shape: [B, ..., D]
            norm_stats (Any): Precomputed stats defining the normalization map.
            eps (float): Small constant to guarantee numerical stability inside log().
        """
        derivative = norm.normalization_map(mean, norm_stats, derivative_num=1)

        return log_std + torch.log(torch.abs(derivative) + eps)

    def _inverse_log_std_normalization_map(
        self, norm: InstanceNormalization, norm_stats: Any, log_std: Tensor, mean: Tensor, eps: float = 1e-8
    ) -> Tensor:
        """
        Transforms a log-scaled standard deviation under the inverse normalization map
        using its first derivative evaluated at the distribution's mean.

        Formula: log_std_out = log_std_in + log(|(f^-1)'(mean)|)

        Args:
            log_std (Tensor): The input log-standard deviation to re-normalize. Shape: [B, ..., D]
            mean (Tensor): The input mean in normalized space where the derivative is evaluated. Shape: [B, ..., D]
            norm_stats (Any): Precomputed stats defining the normalization map.
            eps (float): Small constant to guarantee numerical stability inside log().
        """
        derivative = norm.inverse_normalization_map(mean, norm_stats, derivative_num=1)

        return log_std + torch.log(torch.abs(derivative) + eps)

    def _values_transformation(self, values_norm: InstanceNormalization, values_norm_stats: Any, normalize: bool) -> None:
        """
        Apply the transformation to concepts induced by the transformation of the values from the InstanceNormalization.

        First transform log standard deviations, then mean values.
        Order is important to apply the general formulas established by _log_std_normalization_map and _inverse_log_std_normalization_map.

        Args:
            values_norm (InstanceNormalization): Underlying transformations of values.
            values_norm_stats (Any): Statistics used by values_norm.
            normalize (bool): If true, applies forward normalization, else inverse normalization.
        """
        self._assert_shape()

        if self.init_cond_mean is not None and self.init_cond_log_std is not None:
            if normalize is True:
                self.init_cond_log_std = self._log_std_normalization_map(
                    values_norm, values_norm_stats, self.init_cond_log_std, self.init_cond_mean
                )
                self.init_cond_mean = values_norm.normalization_map(self.init_cond_mean, values_norm_stats)

            else:
                self.init_cond_log_std = self._inverse_log_std_normalization_map(
                    values_norm, values_norm_stats, self.init_cond_log_std, self.init_cond_mean
                )
                self.init_cond_mean = values_norm.inverse_normalization_map(self.init_cond_mean, values_norm_stats)

        if self.reconstructed_values is not None:
            if normalize is True:
                if self.vector_field_log_std is not None:
                    self.vector_field_log_std = self._log_std_normalization_map(
                        values_norm, values_norm_stats, self.vector_field_log_std, self.reconstructed_values
                    )
                grad = values_norm.normalization_map(self.reconstructed_values, values_norm_stats, derivative_num=1)
                self.reconstructed_values = values_norm.normalization_map(self.reconstructed_values, values_norm_stats)

            else:
                if self.vector_field_log_std is not None:
                    self.vector_field_log_std = self._inverse_log_std_normalization_map(
                        values_norm, values_norm_stats, self.vector_field_log_std, self.reconstructed_values
                    )
                grad = values_norm.inverse_normalization_map(self.reconstructed_values, values_norm_stats, derivative_num=1)
                self.reconstructed_values = values_norm.inverse_normalization_map(self.reconstructed_values, values_norm_stats)

            if self.vector_field_mean is not None:
                self.vector_field_mean = self.vector_field_mean * grad

        self._assert_shape()

    def _times_transformation(self, times_norm: InstanceNormalization, times_norm_stats: Any, normalize: bool) -> None:
        """
        Apply the transformation to concepts induced by the transformation of time from the InstanceNormalization.

        Args:
            times_norm (InstanceNormalization): Underlying transformations of time.
            times_norm_stats (Any): Statistics used by times_norm.
            normalize (bool): If true, applies transformation induced by normalization, else by the inverse of normalization.
        """
        self._assert_shape()

        if self.evaluation_times is not None:
            if normalize is True:
                t_norm = times_norm.normalization_map(self.evaluation_times, times_norm_stats)

                if self.vector_field_log_std is not None:
                    self.vector_field_log_std = self._inverse_log_std_normalization_map(
                        times_norm, times_norm_stats, self.vector_field_log_std, t_norm
                    )

                grad = times_norm.inverse_normalization_map(t_norm, times_norm_stats, derivative_num=1)
                self.evaluation_times = t_norm

            else:
                t_raw = times_norm.inverse_normalization_map(self.evaluation_times, times_norm_stats)

                if self.vector_field_log_std is not None:
                    self.vector_field_log_std = self._log_std_normalization_map(
                        times_norm, times_norm_stats, self.vector_field_log_std, t_raw
                    )

                grad = times_norm.normalization_map(t_raw, times_norm_stats, derivative_num=1)
                self.evaluation_times = t_raw

            if self.vector_field_mean is not None:
                self.vector_field_mean = self.vector_field_mean * grad

        self._assert_shape()

    def normalize(
        self, values_norm: InstanceNormalization, values_norm_stats: Any, times_norm: InstanceNormalization, times_norm_stats: Any
    ) -> None:
        """
        Normalize evaluation times, reconstructed values, initial conditions, and vector fields if not already normalized.

        Args:
            values_norm (InstanceNormalization): Transformation to apply to spatial value states.
            values_norm_stats (Any): Precomputed stats mapping the values/states.
            times_norm (InstanceNormalization): Transformation to apply to time scales.
            times_norm_stats (Any): Precomputed stats mapping timeline axes.
        """

        if self.normalized is False:
            self._values_transformation(values_norm, values_norm_stats, normalize=True)
            self._times_transformation(times_norm, times_norm_stats, normalize=True)

            self.normalized = True

    def renormalize(
        self, values_norm: InstanceNormalization, values_norm_stats: Any, times_norm: InstanceNormalization, times_norm_stats: Any
    ) -> None:
        """
        Renormalize evaluation times, reconstructed values, initial conditions, and vector fields back to original scale.

        Args:
            values_norm (InstanceNormalization): Transformation mapping value states back to target spaces.
            values_norm_stats (Any): Precomputed stats mapping the values/states.
            times_norm (InstanceNormalization): Transformation mapping timeline axes back.
            times_norm_stats (Any): Precomputed stats mapping timeline axes.
        """
        if self.normalized is True:
            self._values_transformation(values_norm, values_norm_stats, normalize=False)
            self._times_transformation(times_norm, times_norm_stats, normalize=False)

            self.normalized = False


class Windowing(ABC):
    """
    Breaking down long time-series sequences into manageable processing windows and recombining their evaluations.
    """

    @abstractmethod
    def get_windows_stats(self, obs_values: Tensor, obs_times: Tensor, obs_mask: Tensor, evaluation_times: Tensor) -> Any:
        """
        Extract stats defining windows from inputs.

        Args:
            obs_... (Tensor): Observations defining windows. Shape: [B, T, D/1]
            evaluation_times (Tensor): Points to evaluate the imputing function at. Shape: [B, G, 1]

        Returns:
            windows_stats (Any): Object containing parameters defining windows and their combination.
        """
        ...

    @abstractmethod
    def split_obs(self, obs: Tensor, windows_stats: Any) -> Tensor:
        """
        Split tensor containing observations into windows according to windows_stats.

        Args:
            obs (Tensor): Vector from input. Shape: [B, T, D/1]
            windows_stats (Any): Object containing parameters defining windows and their combination.

        Returns:
            obs_windowed (Tensor): Input vector split into windows. Shape: [B, W, K, D/1]
        """
        ...

    @abstractmethod
    def split_evaluation_times(self, evaluation_times: Tensor, windows_stats: Any) -> Tensor:
        """
        Split tensor containing evaluation times into windows according to windows_stats.

        Args:
            evaluation_times (Tensor): Shape: [B, G, 1]
            windows_stats (Any): Object containing parameters defining windows and their combination.

        Returns:
            evaluation_times_windowed (Tensor): evaluation_times split into windows. Shape: [B, W, L, 1]
        """
        ...

    @abstractmethod
    def combine_evaluations(self, evaluations_windowed: Tensor, windows_stats: Any) -> Tensor:
        """
        Combine tensor containing windowed evaluations to single output.

        Args:
            evaluations_windowed (Tensor): Shape: [B, W, L, D/1]
            windows_stats (Any): Object containing parameters defining windows and their combination.

        Returns:
            evaluations (Tensor): Shape: [B, G, D/1]
        """
        ...

    def split(self, obs_values: Tensor, obs_times: Tensor, obs_mask: Tensor, evaluation_times: Tensor, windows_stats: Any) -> tuple[Tensor]:
        """
        Split observation tensors and evaluation times into windows according to windows_stats.

        Args:
            obs_values (Tensor): Observed values from input. Shape: [B, T, D/1]
            obs_times (Tensor): Timestamps of observations. Shape: [B, T, 1]
            obs_mask (Tensor): Mask identifying valid observations. Shape: [B, T, 1]
            evaluation_times (Tensor): Points to evaluate the imputing function at. Shape: [B, G, 1]
            windows_stats (Any): Object containing parameters defining windows and their combination.

        Returns:
            obs_..._windowed (Tensor): Observations split into windows. Shape: [B, W, K, D/1]
            evaluation_times_windowed (Tensor): Evaluation times split into windows. Shape: [B, W, L, 1]
        """

        assert obs_values.ndim == obs_times.ndim == obs_mask.ndim == evaluation_times.ndim == 3, (
            f"Got {obs_values.ndim}, {obs_times.ndim}, {obs_mask.ndim}, {evaluation_times.ndim}"
        )

        assert obs_values.shape[:2] == obs_times.shape[:2] == obs_mask.shape[:2], (
            f"Got {obs_values.shape[:2]}, {obs_times.shape[:2]}, {obs_mask.shape[:2]}"
        )

        obs_values = self.split_obs(obs_values, windows_stats)
        obs_times = self.split_obs(obs_times, windows_stats)
        obs_mask = self.split_obs(obs_mask, windows_stats)

        evaluation_times = self.split_evaluation_times(evaluation_times, windows_stats)

        assert obs_values.ndim == obs_times.ndim == obs_mask.ndim == evaluation_times.ndim == 4, (
            f"Got {obs_values.ndim}, {obs_times.ndim}, {obs_mask.ndim}, {evaluation_times.ndim}"
        )

        assert obs_values.shape[:3] == obs_times.shape[:3] == obs_mask.shape[:3], (
            f"Got {obs_values.shape[:3]}, {obs_times.shape[:3]}, {obs_mask.shape[:3]}"
        )

        return obs_values, obs_times, obs_mask, evaluation_times

    def combine(
        self,
        evaluation_times: Tensor,
        reconstructed_values: Tensor,
        vector_field_mean: Tensor,
        vector_field_log_std: Tensor,
        windows_stats: Any,
    ) -> tuple[Tensor]:
        """
        Combine windowed evaluation tensors back into single continuous outputs.

        Args:
            evaluation_times (Tensor): Original evaluation times split into windows. Shape: [B, W, L, 1]
            reconstructed_values (Tensor): Imputed values split into windows. Shape: [B, W, L, D]
            vector_field_mean (Tensor): Mean values of the vector field per window. Shape: [B, W, L, D]
            vector_field_log_std (Tensor): Log standard deviations of the vector field per window. Shape: [B, W, L, D]
            windows_stats (Any): Object containing parameters defining windows and their combination.

        Returns:
            Combined inputs. Shapes: [B, G, D]
        """

        assert evaluation_times.ndim == reconstructed_values.ndim == vector_field_mean.ndim == vector_field_log_std.ndim == 4, (
            f"Got {evaluation_times.ndim}, {reconstructed_values.ndim}, {vector_field_mean.ndim}, {vector_field_log_std.ndim}"
        )

        evaluation_times = self.combine_evaluations(evaluation_times, windows_stats)
        reconstructed_values = self.combine_evaluations(reconstructed_values, windows_stats)
        vector_field_mean = self.combine_evaluations(vector_field_mean, windows_stats)
        vector_field_log_std = self.combine_evaluations(vector_field_log_std, windows_stats)

        assert evaluation_times.ndim == reconstructed_values.ndim == vector_field_mean.ndim == vector_field_log_std.ndim == 3, (
            f"Got {evaluation_times.ndim}, {reconstructed_values.ndim}, {vector_field_mean.ndim}, {vector_field_log_std.ndim}"
        )

        return evaluation_times, reconstructed_values, vector_field_mean, vector_field_log_std


def get_balanced_windows(grid_size: int, windows_count: int, overlap_percentage: float) -> tuple[int, int]:
    """
    Define sizes for windows on a grid with overlap, such that all windows are of equal size.

    Args:
        grid_size (int): Size of grid to split into windows.
        windows_count (int): Number of windows to split grid into.
        overlap_percentage (float): Additional percentage of window to overlap to the PREVIOUS window.

    Returns:
        window_size (int): Actual size of each window (including overlap).
        stride_size (int): Stride of indices defining the first element of each window.
    """

    assert grid_size >= windows_count

    ideal_w = grid_size / (1 + (windows_count - 1) * (1 - overlap_percentage))

    window_size = round(ideal_w)
    stride_size = round(window_size * (1 - overlap_percentage))

    return window_size, stride_size


def get_overlapping_window_slices(total_length: int, windows_count: int, window_size: int, stride_size: int) -> Tensor:
    """
    Build slice indices of windows from count, size, and stride lengths.

    Args:
        total_length (int): Total size of the grid to be split.
        windows_count (int): Number of windows to create.
        window_size (int): Size of each individual window.
        stride_size (int): Stride used to calculate the start of each window.

    Returns:
        window_slices (Tensor): Indices defining start and end of each window. Shape: [w, 2]
    """

    window_slices = torch.zeros((windows_count, 2), dtype=torch.long)

    for i in range(windows_count):
        # last overlap compensates for when split can not be perfectly balanced
        start_idx = i * stride_size if i < windows_count - 1 else total_length - window_size
        end_idx = start_idx + window_size

        window_slices[i, 0] = start_idx
        window_slices[i, 1] = end_idx

    return window_slices


def compress_contiguous_mask(mask: Tensor) -> Tensor:
    """
    Extract compact window slice boundaries from boolean masks containing a single contiguous block of 1s, else 0s.

    Args:
        mask (Tensor): Boolean mask with single contiguous blocks of 1s. Shape: [B, W, G]

    Returns:
        slices (Tensor): Bound indices defining start and end elements of contiguous blocks. Shape: [B, W, 2]
    """

    assert mask.ndim == 3

    start_indices = torch.argmax(mask.to(torch.int), dim=-1)  # [B, w]
    slice_size = torch.sum(mask, dim=-1)  # [B, w]
    end_indices = start_indices + slice_size  # [B, w]

    slices = torch.stack([start_indices, end_indices], dim=-1)  # [B, w, 2]

    return slices


def scatter_contiguous_blocks(windows: Tensor, original_slices: Tensor, original_size: int) -> tuple[Tensor, Tensor]:
    """
    Reverse the model output compression of windows to full grid via coordinate scattering loop.

    Args:
        windows (Tensor): Windowed evaluations. Shape: [B, W, L, D/1]
        original_slices (Tensor): Boundaries for each window. Shape: [B, W, 2]
        original_size (int): The original full grid size G.

    Returns:
        windows_scattered (Tensor): Values expanded back to the original grid. Shape: [B, W, G, D/1]
        scattered_mask (Tensor): Mask identifying valid evaluations after scatter. Shape: [B, W, G, 1]
    """

    B, W, _, D = windows.shape
    G = original_size

    windows_scattered = torch.zeros((B, W, G, D), device=windows.device)
    scattered_mask = torch.zeros((B, W, G, 1), device=windows.device)  # identify evaluations after scatter

    # double for loop could surely be replaced by complex torch operations
    for b in range(B):
        for w in range(W):
            start_idx = original_slices[b, w, 0]
            end_idx = original_slices[b, w, 1]
            length = end_idx - start_idx

            if length > 0:
                target_indices = torch.arange(start_idx, end_idx, device=windows.device)
                windows_scattered[b, w, target_indices, :] = windows[b, w, :length, :]
                scattered_mask[b, w, target_indices, :] = 1.0

    return windows_scattered, scattered_mask


def linear_windows_interpolation(windows_scattered: Tensor, windows_mask: Tensor) -> Tensor:
    """
    Execute linear tensor interpolation over the overlap boundaries of adjacent windows.
    Assumes windows are in order and no more than two windows can overlap at each point.

    Args:
        windows_scattered (Tensor): Scattered evaluation values on a global grid. Shape: [B, W, G, D]
        windows_mask (Tensor): Mask tracking valid entries per window. Shape: [B, W, G, 1]

    Returns:
        combined (Tensor): Unified linear-interpolated tensor. Shape: [B, G, D]
    """

    windows_scattered = windows_scattered * windows_mask

    # points in a single window
    point_single = windows_mask.sum(dim=1) <= 1  # [B, G, 1]
    at_single = torch.where(point_single, windows_scattered.sum(dim=1), 0.0)

    # points_on_overlap
    point_on_overlap = torch.logical_not(point_single)
    overlaps_mask = windows_mask[:, :-1] * windows_mask[:, 1:]  # [B, W-1, G, 1]

    right_window_weight = torch.cumsum(overlaps_mask, dim=-2) * overlaps_mask
    left_window_weight = torch.flip(torch.cumsum(torch.flip(overlaps_mask, dims=[-2]), dim=-2), dims=[-2]) * overlaps_mask
    total_weight = torch.where(left_window_weight + right_window_weight == 0, 1.0, left_window_weight + right_window_weight)

    left_contributions = (left_window_weight / total_weight) * windows_scattered[:, :-1]
    right_contributions = (right_window_weight / total_weight) * windows_scattered[:, 1:]

    summed_overlaps = (left_contributions + right_contributions).sum(dim=1)
    at_overlaps = torch.where(point_on_overlap, summed_overlaps, 0.0)

    return at_single + at_overlaps


@dataclass
class StaticWindowsStats:
    obs_window_slices: Tensor  # [w, 2], defining start_idx and end_idx slicing for obs_...
    obs_windows_size: int
    eval_windows_slices: Tensor  # [B, w, 2], defining start_idx and end_idx slicing for evaluations
    max_eval_window_size: int
    eval_grid_size: int


class StaticWindowing(Windowing):
    """
    Split observation tensors in equal parts with optional overlap, ignoring imbalances caused by masked values.

    Evaluation grid splitting implemented by masks.
    """

    def __init__(self, windows_count: int, overlap_percentage: float):

        if not 0 <= overlap_percentage < 1:
            raise ValueError("Overlap percentage must be between 0 and 1.")

        self.windows_count = windows_count
        self.overlap_percentage = overlap_percentage

    def get_windows_stats(self, obs_values: Tensor, obs_times: Tensor, obs_mask: Tensor, evaluation_times: Tensor) -> StaticWindowsStats:
        """
        Build window stats completely based on the index dimensions of evaluation_times to ensure perfect reversibility.

        Args:
            obs_... (Tensor): Observations defining windows. Shape: [B, T, D/1]
            evaluation_times (Tensor): Points to evaluate the imputing function at. Shape: [B, G, 1]

        Returns:
            windows_stats (StaticWindowsStats): Object containing parameters defining windows and their combination.
        """
        T = obs_times.shape[-2]
        G = evaluation_times.shape[-2]

        # balanced overlapping windows of obs_...
        obs_window_size, stride_size = get_balanced_windows(T, self.windows_count, self.overlap_percentage)

        obs_window_slices = get_overlapping_window_slices(T, self.windows_count, obs_window_size, stride_size)
        obs_window_slices = obs_window_slices.to(obs_times.device)

        # define scattered mask associating each evaluation time to (potentially 0, 1, 2) windows)
        evaluation_times = evaluation_times.squeeze(-1)  # [B, G]
        evaluation_times = einops.repeat(evaluation_times, "B G -> B w G", w=self.windows_count)

        if self.windows_count == 1:
            evaluations_mask = torch.ones_like(evaluation_times, dtype=torch.bool)

        else:
            start_indices = obs_window_slices[:, 0]
            end_indices = obs_window_slices[:, 1] - 1

            raw_lower_bound = obs_times[:, start_indices].clone()
            raw_upper_bound = obs_times[:, end_indices].clone()

            raw_lower_bound[:, 0] = -torch.inf
            raw_upper_bound[:, -1] = torch.inf

            # handling adjacent windows with no overlap
            # define bounds by raw_bounds of adjacent windows
            adjacent_upper_bound = torch.concatenate([raw_lower_bound[:, 1:], raw_upper_bound[:, -1:]], dim=1)
            adjacent_lower_bound = torch.concatenate([raw_lower_bound[:, :1], raw_upper_bound[:, :-1]], dim=1)

            # take extremes of two, to only use adjacents if they are wider than raw bounds
            upper_bound = torch.maximum(raw_upper_bound, adjacent_upper_bound)
            lower_bound = torch.minimum(raw_lower_bound, adjacent_lower_bound)

            evaluations_mask = (evaluation_times >= lower_bound) & (evaluation_times <= upper_bound)

        # compress evaluation mask to pass smallest tensor as possible to the model
        eval_windows_slices = compress_contiguous_mask(evaluations_mask)

        return StaticWindowsStats(
            obs_window_slices=obs_window_slices,
            obs_windows_size=obs_window_size,
            eval_windows_slices=eval_windows_slices,
            max_eval_window_size=torch.amax(eval_windows_slices).item(),
            eval_grid_size=G,
        )

    def split_obs(self, obs: Tensor, windows_stats: StaticWindowsStats) -> Tensor:
        """
        Split tensor containing observations into windows according to windows_stats.

        Args:
            obs (Tensor): Vector from input. Shape: [B, T, D/1]
            windows_stats (StaticWindowsStats): Object containing parameters defining windows and their combination.

        Returns:
            obs_windowed (Tensor): Input vector split into windows. Shape: [B, W, K, D/1]
        """
        if self.windows_count == 1:
            return einops.rearrange(obs, "B T D -> B 1 T D")

        else:
            B, _, D = obs.shape
            W = self.windows_count
            K = windows_stats.obs_windows_size

            obs_windowed = torch.zeros((B, W, K, D), dtype=obs.dtype, device=obs.device)

            for w in range(W):
                start_idx = windows_stats.obs_window_slices[w, 0]
                end_idx = windows_stats.obs_window_slices[w, 1]
                length = end_idx - start_idx

                if length > 0:
                    src_indices = torch.arange(start_idx, end_idx, device=obs.device)
                    obs_windowed[:, w, :length, :] = obs[:, src_indices, :]

            return obs_windowed

    def split_evaluation_times(self, evaluation_times: Tensor, windows_stats: StaticWindowsStats) -> Tensor:
        """
        Split tensor containing evaluation times into windows according to windows_stats.

        Args:
            evaluation_times (Tensor): Shape: [B, G, 1]
            windows_stats (StaticWindowsStats): Object containing parameters defining windows and their combination.

        Returns:
            evaluation_times_windowed (Tensor): evaluation_times split into windows. Shape: [B, W, L, 1]
        """

        if self.windows_count == 1:
            return einops.rearrange(evaluation_times, "B G 1 -> B 1 G 1")

        else:
            B, W, _ = windows_stats.eval_windows_slices.shape
            L = windows_stats.max_eval_window_size

            # prefill with largest time in batch, important for FIMImpPointBase
            evaluation_times_windowed = einops.repeat(torch.amax(evaluation_times, dim=-2), "B X -> B w l X", w=W, l=L).contiguous()

            # double for loop could surely be replaced by complex torch operations
            for b in range(B):
                for w in range(W):
                    start_idx = windows_stats.eval_windows_slices[b, w, 0]
                    end_idx = windows_stats.eval_windows_slices[b, w, 1]
                    length = end_idx - start_idx

                    if length > 0:
                        src_indices = torch.arange(start_idx, end_idx, device=evaluation_times.device)
                        evaluation_times_windowed[b, w, :length] = evaluation_times[b, src_indices]

            return evaluation_times_windowed

    def combine_evaluations(self, evaluations_windowed: Tensor, windows_stats: StaticWindowsStats) -> Tensor:
        """
        Combine tensor containing windowed evaluations to single output.

        Args:
            evaluations_windowed (Tensor): Shape: [B, W, L, D/1]
            windows_stats (StaticWindowsStats): Object containing parameters defining windows and their combination.

        Returns:
            evaluations (Tensor): Shape: [B, G, D/1]
        """

        if self.windows_count == 1:
            G = windows_stats.eval_grid_size
            return evaluations_windowed[:, 0, :G, :]

        else:
            # reverse compression
            evaluations_windowed_scattered, evaluations_mask = scatter_contiguous_blocks(
                evaluations_windowed, windows_stats.eval_windows_slices, windows_stats.eval_grid_size
            )

            evaluations_windowed_scattered.to(evaluations_windowed.device)
            evaluations_mask.to(evaluations_windowed.device)

            # linear interpolation on overlaps
            evaluations = linear_windows_interpolation(evaluations_windowed_scattered, evaluations_mask)

            return evaluations


# Todo: define denoising interface
def no_denoising(obs_values: Tensor, obs_mask: Tensor) -> Tensor:
    return obs_values


no_windowing = StaticWindowing(windows_count=1, overlap_percentage=0)


class FIMImpPoint:
    """
    Wrapper for FIMImpPointBase model to allow multidimensional and/or longer input sequences.

    Denoising model is optional. If not provided, the input is not denoised.

    If windowing scheme is specified, the class can handle longer trajectories than the pre-trained FIMImpPointBase.

    If no denoising and no windowing scheme is specified, this regresses to FIMImpPointBase, if applied to one-dimensional input.
    """

    def __init__(
        self,
        fim_imp_pointwise_base: str | Path | FIMImpPointBase,
        windowing: Windowing | None = None,
        denoising_model: Callable | None = None,
    ):

        self.fim_imp_pointwise_base: FIMImpPointBase = (
            load_model_from_checkpoint(fim_imp_pointwise_base, module=FIMImpPointBase)
            if isinstance(fim_imp_pointwise_base, (str, Path))
            else fim_imp_pointwise_base
        )
        self.windowing: Windowing = windowing if windowing is not None else no_windowing
        self.denoising_model = denoising_model if denoising_model is not None else no_denoising

    @torch.profiler.record_function("fim_imp_pointwise_forward")
    def forward(
        self, obs_times: Tensor, obs_values: Tensor, evaluation_times: Tensor, obs_mask: Tensor | None = None
    ) -> ImputationConcepts:
        """
        Args:
            obs_times (Tensor): Observation times of obs_values, assumed to be ordered. Shape: [B, T, 1]
            obs_values (Tensor): Observation values. Shape: [B, T, D]
            evaluation_times (Tensor): Points to evaluate the imputing function at, assumed to be ordered. Shape: [B, G, 1]
            obs_mask (Tensor): Mask for padded observations. (0: value is observed, 1: value is masked out). Shape: [B, T, 1]

        where B: batch size, T: number of observations, G: number of evaluation points, D: dimensions

        Returns:
            estimated_concepts (ImputationConcepts): Estimated concepts at evaluation_times. Shape: [B, G, D/1], or [B, D] for initial conditions.
        """

        # general preprocessing
        obs_times, obs_values, obs_mask = FIMImpPoint.preprocess_inputs(obs_times, obs_values, obs_mask)

        # denoising
        obs_values = self.denoising_model(obs_values, obs_mask)

        # split into windows: [B, W, K, D/1]
        windows_stats = self.windowing.get_windows_stats(obs_values, obs_times, obs_mask, evaluation_times)
        obs_values, obs_times, obs_mask, evaluation_times = self.windowing.split(
            obs_values, obs_times, obs_mask, evaluation_times, windows_stats
        )

        # FIMImpPointBase per dimension and window
        imputation_concepts = self.apply_fim_imp_pointwise_base(obs_times, obs_values, obs_mask, evaluation_times)

        # combine outputs of windows (evaluation times is reconstructed, but could also be taken from input; should be the same)
        (
            imputation_concepts.evaluation_times,
            imputation_concepts.reconstructed_values,
            imputation_concepts.vector_field_mean,
            imputation_concepts.vector_field_log_std,
        ) = self.windowing.combine(
            imputation_concepts.evaluation_times,
            imputation_concepts.reconstructed_values,
            imputation_concepts.vector_field_mean,
            imputation_concepts.vector_field_log_std,
            windows_stats,
        )

        # select initial condition of first window
        imputation_concepts.init_cond_mean = imputation_concepts.init_cond_mean[:, 0]
        imputation_concepts.init_cond_log_std = imputation_concepts.init_cond_log_std[:, 0]

        return imputation_concepts

    @torch.profiler.record_function("fim_imp_pointwise_preprocess_inputs")
    @staticmethod
    def preprocess_inputs(obs_times: Tensor, obs_values: Tensor, obs_mask: Tensor | None = None) -> tuple[Tensor]:
        """
        Preprocess inputs by creating and applying masks

        Args:
            obs_times (Tensor): Observation times of obs_values, assumed to be ordered. Shape: [B, T, 1]
            obs_values (Tensor): Observation values. Shape: [B, T, D]
            obs_mask (Tensor): Mask for padded observations. (0: value is observed, 1: value is masked out). Shape: [B, T, 1]
        where B: batch size, T: number of observations, G: number of evaluation points, D: dimensions

        Returns: Preprocessed inputs for denoising and FIMImpPointBase model. Shapes: [B, T, D]
        """

        if obs_mask is None:
            obs_mask = torch.zeros_like(obs_times)

        # For sanity, removed masked out values
        obs_times = torch.logical_not(obs_mask) * obs_times
        obs_values = torch.logical_not(obs_mask) * obs_values

        # Then forward fill masked values s.t. obs_times are ordered again
        obs_times = forward_fill_masked_values(obs_times, torch.logical_not(obs_mask))  # expects reverse masking convention
        obs_values = forward_fill_masked_values(obs_values, torch.logical_not(obs_mask))

        return obs_times, obs_values, obs_mask

    @torch.profiler.record_function("fim_imp_pointwise_apply_fim_imp_pointwise_base")
    def apply_fim_imp_pointwise_base(self, obs_times: Tensor, obs_values: Tensor, obs_mask: Tensor, evaluation_times: Tensor):
        """
        Apply pretrained FIMImpPointBase to each window and each dimension.

        Args:
            obs_...: Windowed inputs. Shape: [B, W, K, D/1]
            evaluation_times: Windowed evaluation times. Shape: [B, W, L, 1]

        Returns:
            imputation_concepts (ImputationConcepts): FIMImpPointBase output per dimension and window.
        """
        B, W, _, D = obs_values.shape

        # broadcast, permute and squash windows and D dimension int batch
        fim_inputs = {
            "coarse_grid_observation_mask": einops.repeat(obs_mask, "B W K 1 -> (B W D) K 1", D=D),
            "coarse_grid_noisy_sample_paths": einops.rearrange(obs_values, "B W K D -> (B W D) K 1"),
            "coarse_grid_grid": einops.repeat(obs_times, "B W K 1 -> (B W D) K 1", D=D),
            "fine_grid_grid": einops.repeat(evaluation_times, "B W L 1 -> (B W D) L 1", D=D),
        }

        fim_imp_pointwise_base_output = self.fim_imp_pointwise_base.forward(fim_inputs, training=False)

        # revert axis permutation
        vis = fim_imp_pointwise_base_output["visualizations"]

        return ImputationConcepts(
            evaluation_times=evaluation_times,
            reconstructed_values=einops.rearrange(vis["solution"]["learnt"], "(B W D) K 1 -> B W K D", B=B, W=W, D=D),
            init_cond_mean=einops.rearrange(vis["init_condition"]["learnt"], "(B W D) 1 -> B W D", B=B, W=W, D=D),
            init_cond_log_std=einops.rearrange(torch.log(vis["init_condition"]["certainty"]), "(B W D) 1 -> B W D", B=B, W=W, D=D),
            vector_field_mean=einops.rearrange(vis["drift"]["learnt"], "(B W D) L 1 -> B W L D", B=B, W=W, D=D),
            vector_field_log_std=einops.rearrange(torch.log(vis["drift"]["certainty"]), "(B W D) L 1 -> B W L D", B=B, W=W, D=D),
            normalized=False,
        )


ModelFactory.register(FIMImpPointBaseConfig.model_type, FIMImpPointBase)
ModelFactory.register("FIMImpPoint", FIMImpPoint)
