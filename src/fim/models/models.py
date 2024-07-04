import functools
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from peft.tuners import PrefixEncoder, PromptEmbedding, PromptEncoder
from torch.distributed.fsdp.wrap import (
    _or_policy,
    lambda_auto_wrap_policy,
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from transformers import BitsAndBytesConfig

from fim.utils.helper import create_class_instance

from ..trainers.mixed_precision import is_bfloat_supported
from ..utils.logging import RankLoggerAdapter
from .utils import get_peft_trainable_parameters


def is_distributed() -> bool:
    return dist.is_initialized()


class AModel(nn.Module, ABC):
    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def new_stats(self) -> Dict:
        """
        Create dictionary where it will hold the results (_loss_ and _metrics_) after each training step.
        :return:
        """
        raise NotImplementedError("The new_stats method is not implemented in your class!")

    @abstractmethod
    def loss(self, *inputs) -> Dict:
        raise NotImplementedError("The loss method is not implemented in your class!")

    @abstractmethod
    def metric(self, y: Any, y_target: Any) -> Dict:
        raise NotImplementedError("The metric method is not implemented in your class!")

    def fsdp_wrap_policy(self):
        return None

    @property
    def device(self):
        if is_distributed():
            return int(os.environ["LOCAL_RANK"])
        return next(self.parameters()).device

    @property
    def rank(self) -> int:
        if is_distributed():
            return int(os.environ["RANK"])
        return 0


class ModelFactory:
    model_types = {}

    @classmethod
    def register(
        cls,
        model_type: str,
        model_class: AModel,
    ):
        cls.model_types[model_type] = model_class

    @classmethod
    def create(cls, name: str, **kwargs) -> AModel:
        model_class = cls.model_types.get(name)
        if model_class:
            return model_class(**kwargs)
        else:
            raise ValueError("Invalid model type")


class AR(AModel):
    def __init__(
        self,
        recurrent_module: dict,
        output_head: dict,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_bf16: bool = False,
        device_map: Optional[str] = None,
        resume: bool = False,
        peft: Optional[dict] = None,
    ):
        super(AR, self).__init__()
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

        self._create_model(
            recurrent_module,
            output_head,
        )
        if self.rank == 0:
            self.logger.info("=" * 50)
            self.logger.info(f"AR Model: {get_peft_trainable_parameters(self)} parameters")
            self.logger.info("=" * 50)
            self.logger.info(self)
            self.logger.info("=" * 50)
        self.to(self._device_map)

    def _create_model(
        self,
        recurrent_module: dict,
        output_head: dict,
    ):
        self.rnn = create_class_instance(
            recurrent_module.pop("name"),
            recurrent_module,
        )
        self.output_head = create_class_instance(
            output_head.pop("name"),
            output_head,
        )
        # self.rnn.to(self._device_map)
        # self.output_head.to(self._device_map)

    def forward(
        self,
        batch,
        schedulers: Optional[dict] = None,
        step: Optional[int] = None,
    ):
        """
        Forward step of the  language model.

        Parameters
        ----------
        input (Tensor) of shape [B, T, D]
        z (Tensor) of shape [B, D'] representing global dynamic state

        Returns
        -------
        (logits, hidden_state)
        Notation. B: batch size; T: seq len (== fix_len); D: hidden dimension
        """
        import torch.nn.utils.rnn as rnn_utils

        input = torch.cat(
            [
                batch["target"][..., :-1, :],
                batch["time_feat"][..., 1:, :],
            ],
            dim=-1,
        )
        packed_input = rnn_utils.pack_padded_sequence(
            input,
            batch["seq_len"].cpu() - 1,
            batch_first=True,
            enforce_sorted=False,
        )

        h, _ = self.rnn(packed_input)

        (
            h,
            _,
        ) = rnn_utils.pad_packed_sequence(h, batch_first=True)

        out = self.output_head(h)

        losses = self.loss(out, batch["target"])
        return {
            "losses": losses,
            "predictions": out,
        }

    def loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict:
        import torch.nn.functional as F

        shifted_targets = targets[..., 1:, :].contiguous()
        mse = F.mse_loss(
            predictions[..., 0:1],
            shifted_targets,
        )
        rmse = torch.sqrt(mse)

        return {
            "rmse": rmse,
            "mse": mse,
            "loss": rmse,
        }

    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_new_tokens: int = 20,
        do_sample: bool = False,
    ):
        """
        Forward step of the  language model.

        Parameters
        ----------
        input (Tensor) of shape [B, T, D]
        z (Tensor) of shape [B, D'] representing global dynamic state

        Returns
        -------
        (logits, hidden_state)
        Notation. B: batch size; T: seq len (== fix_len); D: hidden dimension
        """

        out = self.backbone.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )

        return out

    def metric(
        self,
        y: Any,
        y_target: Any,
        seq_len=None,
    ):
        """
        returns a dictionary with metrics
        """

        return {}

    def new_metric_stats(self) -> Dict:
        stats = {}
        return stats

    def fsdp_activation_check_fn(self):
        ...
        # if isinstance(self.backbone, GPT2LMHeadModel):
        #     return lambda submodule: isinstance(submodule, GPT2Block)
        # elif isinstance(self.backbone, LlamaForCausalLM):
        #     return lambda submodule: isinstance(submodule, LlamaDecoderLayer)
        # else:
        #     raise ValueError("Activation checkpoint is not defined for this type of models!")

    def get_transformer_layers(self):
        ...
        # if isinstance(self.config, GPT2Config):
        #     return {GPT2Block}
        # elif isinstance(self.config, LlamaConfig):
        #     return {LlamaDecoderLayer}
        # else:
        #     raise ValueError("Wrapping policy is not defined for this type of models!")

    def fsdp_wrap_policy(self):
        transformer_layers = self.get_transformer_layers()
        return functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_layers,
        )

    def fsdp_peft_wrap_policy(self):
        def lambda_policy_fn(module):
            return (
                len(list(module.named_children())) == 0
                and getattr(
                    module,
                    "weight",
                    None,
                )
                is not None
                and module.weight.requires_grad
            )

        transformer_layers = self.get_transformer_layers()
        lambda_policy = functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda_policy_fn,
        )
        transformer_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                PrefixEncoder,
                PromptEncoder,
                PromptEmbedding,
            }
            | transformer_layers,
        )
        auto_wrap_policy = functools.partial(
            _or_policy,
            policies=[
                lambda_policy,
                transformer_wrap_policy,
            ],
        )
        return auto_wrap_policy
        # return functools.partial(transformer_auto_wrap_policy, transformer_layer_cls=transformer_layers)

    def get_fsdp_policy(self, min_num_params: int):
        try:
            if self.is_peft():
                wrap_policy = self.fsdp_peft_wrap_policy()
            else:
                wrap_policy = self.fsdp_wrap_policy()
        except ValueError:
            self.logger.warning("The model does not have custom wrapping policy. Size based auto policy will be used!")
            wrap_policy = functools.partial(
                size_based_auto_wrap_policy,
                min_num_params=min_num_params,
            )
        return wrap_policy

    def new_stats(self) -> Dict:
        stats = {}
        return stats

    def is_peft(self) -> bool:
        return self.peft is not None and self.peft["method"] is not None


ModelFactory.register("AR", AR)


class DecoderOnly(AModel):
    """
    Based on the paper: "A decoder-only foundation model for time-series forecasting" by Das, A. et. al.
    """

    def __init__(
        self,
        residual_block_input: dict,
        positional_encoding: dict,
        decoder_block: dict,
        residual_block_output: dict,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_bf16: bool = False,
        device_map: Optional[str] = None,
        resume: bool = False,
        peft: Optional[dict] = None,
    ):
        super(DecoderOnly, self).__init__()
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

        self._create_model(
            residual_block_input=residual_block_input,
            positional_encoding=positional_encoding,
            decoder_block=decoder_block,
            residual_block_output=residual_block_output,
        )
        if self.rank == 0:
            self.logger.info("=" * 50)
            self.logger.info(f"DecoderOnly Model: {get_peft_trainable_parameters(self)} parameters")
            self.logger.info("=" * 50)
            self.logger.info(self)
            self.logger.info("=" * 50)
        self.to(self._device_map)

    def _create_model(
        self,
        residual_block_input: dict,
        decoder_block: dict,
        positional_encoding: dict,
        residual_block_output: dict,
    ):
        self.residual_block_input = create_class_instance(
            residual_block_input.pop("name"),
            residual_block_input,
        )

        self.positional_encoding = create_class_instance(
            positional_encoding.pop("name"),
            positional_encoding,
        )

        self.transformer_blocks = nn.ModuleList()
        decoder_block_name = decoder_block.pop("name")
        for _ in range(decoder_block.pop("number_decoder_blocks")):
            self.transformer_blocks.append(create_class_instance(decoder_block_name, decoder_block))

        self.residual_block_output = create_class_instance(
            residual_block_output.pop("name"),
            residual_block_output,
        )

    def forward(
        self,
        batch,
        schedulers: Optional[dict] = None,
        step: Optional[int] = None,
    ):
        """
        Forward step of the DecoderOnly model.

        Args:
            batch: input batch with keys
                - input_values: the input sequence, patched and padded; [batch_size, seq_len, patch_len_in]
                - mask_point_level: the mask on point level; [batch_size, seq_len, patch_len_in], (1: masked, 0: not masked)
                - mask_token_level: the mask on token level; [batch_size, seq_len], (1: masked, 0: not masked)
                - output_values: the output sequence, patched and padded; [batch_size, patch_len_out]
                - seq_len: the sequence length, i.e. the number of patches; [batch_size]
                - time_feat: the time features; [batch_size, seq_len, time_feat_dim]
            schedulers: optional schedulers
            step: optional step
        """
        input_token = self.residual_block_input(batch["input_values"] * (~batch["mask_point_level"]))
        # add positional encoding
        token = self.positional_encoding(input_token)

        # decoder blocks
        for block in self.transformer_blocks:
            token = block(token, batch["mask_token_level"])

        # map to model's output
        output_token = self.residual_block_output(token)

        losses = self.loss(output_token, batch["output_values"])
        return {
            "losses": losses,
            "predictions": output_token,
        }

    def loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict:
        import torch.nn.functional as F

        prediction = predictions[..., -1, :]

        mse = F.mse_loss(
            prediction,
            targets,
        )
        rmse = torch.sqrt(mse)

        mae = F.l1_loss(
            prediction,
            targets,
        )

        return {
            "rmse": rmse,
            "mse": mse,
            "mae": mae,
            "loss": mse,
        }

    def metric(
        self,
        y: Any,
        y_target: Any,
        seq_len=None,
    ):
        raise NotImplementedError("The metric method is not implemented in class DecoderOnly!")

    def new_stats(self) -> Dict:
        raise NotImplementedError("The new_stats method is not implemented in class DecoderOnly!")

    def is_peft(self) -> bool:
        return self.peft is not None and self.peft["method"] is not None


ModelFactory.register("DecoderOnly", DecoderOnly)


class FIMODE(AModel):
    def __init__(
        self,
        time_encoding: dict,
        trunk_net: dict,
        branch_net: dict,
        combiner_net: dict,
        init_cond_mean_net: dict,
        init_cond_var_net: dict,
        vector_field_mean_net: dict,
        vector_field_var_net: dict,
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

        self._create_model(
            time_encoding,
            trunk_net,
            branch_net,
            combiner_net,
            vector_field_mean_net,
            vector_field_var_net,
            init_cond_mean_net,
            init_cond_var_net,
        )

        self.to(self._device_map)

    def _create_model(
        self,
        time_encoding: dict,
        trunk_net: dict,
        branch_net: dict,
        combiner_net: dict,
        init_cond_mean_net: dict,
        init_cond_var_net: dict,
        vector_field_mean_net: dict,
        vector_field_var_net: dict,
    ):
        self.time_encoding = create_class_instance(
            time_encoding.pop("name"),
            time_encoding,
        )

        self.trunk_net = create_class_instance(
            trunk_net.pop("name"),
            trunk_net,
        )

        self.branch_net = create_class_instance(
            branch_net.pop("name"),
            branch_net,
        )

        if combiner_net.get("in_features") != 2 * combiner_net.get("out_features"):
            raise ValueError(
                "The number of input features for the combiner_net must be twice the number of output features (latent dim)."
            )

        self.combiner_net = create_class_instance(
            combiner_net.pop("name"),
            combiner_net,
        )

        self.vector_field_mean_net = create_class_instance(
            vector_field_mean_net.pop("name"),
            vector_field_mean_net,
        )

        self.vector_fiel_var_net = create_class_instance(
            vector_field_var_net.pop("name"),
            vector_field_var_net,
        )

        self.init_cond_mean_net = create_class_instance(
            init_cond_mean_net.pop("name"),
            init_cond_mean_net,
        )

        self.init_cond_var_net = create_class_instance(
            init_cond_var_net.pop("name"),
            init_cond_var_net,
        )

    def forward(
        self,
        batch,
        schedulers: Optional[dict] = None,
        step: Optional[int] = None,
    ):
        """
        Args:
            batch (dict): input batch with entries (each torch.Tensor)
                 obs_values [B, T, D] observation values
                 obs_times [B, T, 1] observation times
                 obs_mask [B, T, 1] observation mask (1: masked, 0: not masked)
                 fine_grid_times [B, L, 1] time points of fine grid on which the vector field is evaluated
                 fine_grid_values [B, L, D] values of the fine grid
                with B: batch size, T: number of observation times, D: process dimension, L: number of fine grid points (locations)
        """
        obs_mask = batch["obs_mask"]
        obs_values = batch["obs_values"]
        obs_times = batch["obs_times"]

        fine_grid = batch["fine_grid_times"]
        fine_grid_values = batch["fine_grid_values"]

        batch_size, max_sequence_length, process_dim = obs_values.shape  # defines B, T, D
        self.process_dim = process_dim

        # TODO: sample locations?
        loc_times, location_values = fine_grid, fine_grid_values

        (
            obs_values,  # [B, T, D]
            obs_times,  # [B, T, 1]
            obs_values_norm_params,  # ([B, D], [B, D])
            obs_times_norm_params,  # ([B, 1], [B, 1])
            loc_times,  # [B, M, 1]
        ) = self.normalize_input(obs_times=obs_times, obs_values=obs_values, obs_mask=obs_mask, loc_times=loc_times)

        # encode times
        obs_times = self.time_encoding(obs_times)  # Shape [B, T, dim_time]
        loc_times = self.time_encoding(loc_times)  # Shape [B, L, dim_time]

        # concatenate time encoding with normalized observation values. Use that dimensions are uncoupled.
        # first: reshape observation times & values to match shapewise
        obs_input = torch.cat(
            [
                obs_times.repeat_interleave(process_dim, 0),  # Shape [D*B, T, dim_time],
                obs_values.reshape(batch_size * process_dim, max_sequence_length, 1),  # Shape [D*B, T, 1]
            ],
            dim=-1,
        )  # Shape [D*B, T, dim_time + 1]

        # repeat obs_mask to match obs_input
        obs_mask = obs_mask.repeat(process_dim, 1, 1)  # Shape [D*B, T, 1]

        # pass through deepOnet
        branch_out = self.branch_net(obs_input, obs_mask)  # Shape [D*B, 1, dim_latent]
        trunk_out = self.trunk_net(loc_times)  # Shape [B, M, dim_latent]

        # concat branch and trunk output: append branch output to each time step of trunk output
        combiner_in = torch.cat(
            [
                trunk_out.repeat(process_dim, 1, 1),  # Shape [D*B, M, dim_latent]
                branch_out.repeat(1, trunk_out.shape[1], 1),  # Shape [D*B, M, dim_latent]
            ],
            dim=-1,
        )  # Shape [D*B, M, 2*dim_latent]
        combiner_out = self.combiner_net(combiner_in)  # Shape [D*B, M, dim_latent]

        # compute mean and log variance of the vector field at every location
        vector_field_mean = self.vector_field_mean_net(combiner_out)  # Shape [D*B, M, 1]
        vector_field_var = self.vector_fiel_var_net(combiner_out)  # Shape [D*B, M, 1]

        # compute mean and log variance of the initial condition
        init_condition_mean = self.init_cond_mean_net(branch_out)  # Shape [D*B, 1, 1]
        init_condition_var = self.init_cond_var_net(branch_out)  # Shape [D*B, 1, 1]

        # renormalize vector field & initial condition distribution parameters
        vector_field_concepts = self.renormalize_concept_dist_params(
            (vector_field_mean, vector_field_var), obs_values_norm_params, obs_times_norm_params
        )
        init_condition_mean, init_condition_var = self.renormalize_init_condition_params(
            (init_condition_mean, init_condition_var), obs_values_norm_params
        )

        # TODO compute loss
        return {
            **self.loss(
                vector_field_concepts=vector_field_concepts,
                init_condition_concepts=(init_condition_mean, init_condition_var),
                targets=location_values,
            ),
            "vector_field_concepts": vector_field_concepts,
            "init_condition_concepts": (init_condition_mean, init_condition_var),
        }

        # TODO setup return value

    def loss(self, vector_field_concepts: tuple, init_condition_concepts: tuple, targets: torch.Tensor) -> Dict:
        """
        Compute the loss function of the FIMODE model: supervised and unsupervised loss.

        Args:
            vector_field_concepts (tuple): mean and log standard deviation of the vector field concepts ([T, D], [T, D])
            init_condition_concepts (tuple): mean and log standard deviation of the initial condition concepts ([D], [D])
            targets (torch.Tensor): target values [P, T, D]

        Returns:
            dict: supervised loss, unsupervised loss, (total) loss
        """
        # supervised loss: maximize log-likelihood of values taken by vector field at observation times
        mu, log_std = vector_field_concepts
        std = torch.exp(log_std)
        # TODO check if shapes are correct
        supervised_loss = torch.sum(
            -1 / 2 * torch.log(2 * torch.pi) - log_std - 1 / 2 * (targets - mu) ** 2 / std**2, dim=-1
        )

        # unsupervised loss: minimize one-step ahead reconstrucion error of integrated solution
        unsupervised_loss = None

        return {
            "supervised_loss": supervised_loss,
            "unsupervised_loss": unsupervised_loss,
            "loss": -supervised_loss + unsupervised_loss,
        }

    def normalize_input(
        self, obs_values: torch.Tensor, obs_times: torch.Tensor, obs_mask: torch.Tensor, loc_times: torch.Tensor
    ) -> tuple:
        """
        Apply min-max scaling to observation values and times.

        Args:
            obs_values (torch.Tensor): observation values
            obs_times (torch.Tensor): observation times
            obs_mask (torch.Tensor): observation mask
            loc_times (torch.Tensor): location times

        Returns:
            tuple: normalized observation values (torch.Tensor),
                   normalized observation times (torch.Tensor),
                   normalization parameters for values (tuple),
                   normalization parameters for times (tuple)
                   normalized location times (torch.Tensor)
        """

        def get_norm_params(values: torch.Tensor, mask: torch.Tensor) -> tuple:
            """
            Compute normalization parameters for min-max scaling.

            Args:
                values (torch.Tensor): observation values [B, T, D]
                mask (torch.Tensor): observation mask [B, T, 1]

            Returns:
                tuple: min (torch.Tensor, [B, D]), range (torch.Tensor, [B, D])
            """
            # get min and max values for each feature dimension per batch entry
            min_values = torch.amin(values.masked_fill(mask, float("inf")), dim=1)  # Shape [B, D]
            max_values = torch.amax(values.masked_fill(mask, float("-inf")), dim=1)  # Shape [B, D]

            # compute range, add small value to avoid division by zero
            values_range = max_values - min_values + 1e-6

            return min_values, values_range

        def normalize(values: torch.Tensor, norm_params: tuple) -> torch.Tensor:
            """
            Normalize values using min-max scaling.

            Args:
                values (torch.Tensor): observation values [B, T, D]
                norm_params (tuple): min and range of the values ([B, D], [B, D])

            Returns:
                torch.Tensor: normalized values [B, T, D]
            """
            min_values, values_range = norm_params

            # unsqueeze to allow broadcasting
            min_values = min_values.unsqueeze(1)  # Shape [B, 1, D]
            values_range = values_range.unsqueeze(1)  # Shape [B, 1, D]

            return (values - min_values) / values_range

        obs_values_norm_params = get_norm_params(obs_values, obs_mask)  # ([B, D], [B, D])
        obs_times_norm_params = get_norm_params(obs_times, obs_mask)  # ([B, 1], [B, 1])

        normalized_obs_values = normalize(obs_values, obs_values_norm_params)  # [B, T, D]
        normalized_obs_times = normalize(obs_times, obs_times_norm_params)  # [B, T, 1]
        normalized_loc_times = normalize(loc_times, obs_times_norm_params)  # [B, M, 1]

        return (
            normalized_obs_values,
            normalized_obs_times,
            obs_values_norm_params,
            obs_times_norm_params,
            normalized_loc_times,
        )

    def renormalize_concept_dist_params(
        self, concept_dist_params: tuple, obs_values_norm_params: tuple, obs_times_norm_params: tuple
    ) -> tuple:
        """
        Rescale concepts based on observation values and observation times normalization parameters.

        Args:
            concept_dist_params (tuple): mean and variance of the concept distribution (learnt) ([B*D, T, 1], [B*D, T, 1])
            obs_values_norm_params (tuple): mean and variance of the observation values ([B, D], [B, D])
            obs_times_norm_params (tuple): mean and variance of the observation times ([B, 1], [B, 1])

        Returns:
            tuple: rescaled mean and log standard deviation of the concept distribution ([B*D, M, 1], [B*D, M, 1])
        """
        learnt_drift_mean, learnt_drift_log_std = concept_dist_params  # [D*B, M, 1], [D*B, M, 1]
        _, obs_values_range = obs_values_norm_params  # [B, D]
        _, obs_times_range = obs_times_norm_params  # [B, 1]

        # reshape (and repeat) values_range to match drift_mean
        values_range_view = obs_values_range.reshape(-1, 1, 1).expand(learnt_drift_mean.shape)  # Shape [D*B, M, 1]
        times_range_view = (
            obs_times_range.unsqueeze(1).repeat(self.process_dim, 1, 1).expand(learnt_drift_mean.shape)
        )  # Shape [D*B, M, 1]
        # rescale mean and log std
        learnt_drift_mean = learnt_drift_mean * values_range_view / times_range_view  # Shape [D*B, M, 1]
        learnt_drift_log_std = (
            learnt_drift_log_std + torch.log(values_range_view) - torch.log(times_range_view)
        )  # Shape [D*B, M, 1]

        return learnt_drift_mean, learnt_drift_log_std

    def renormalize_init_condition_params(self, init_cond_dist_params: tuple, obs_values_norm_params: tuple) -> tuple:
        """
        Rescale the initial condition based on observation values normalization parameters.

        Args:
            init_cond_dist_params (tuple): mean and variance of the initial condition ([B*D, 1, 1], [B*D, 1, 1])
            obs_values_norm_params (tuple): mean and variance of the observation values ([B, D], [B, D])

        Returns:
            tuple: rescaled mean and log standard deviation of the initial condition ([D*B, 1], [D*B, 1])
        """
        init_cond_mean, init_cond_log_std = init_cond_dist_params  # [B*D, 1, 1], [B*D, 1, 1]
        obs_values_min, obs_values_range = obs_values_norm_params  # [B, D], [B, D]

        # reshape so that they match
        init_cond_mean = init_cond_mean.squeeze(-1)
        init_cond_log_std = init_cond_log_std.squeeze(-1)
        obs_values_min = obs_values_min.reshape(-1, 1)
        obs_values_range = obs_values_range.reshape(-1, 1)

        # rescale mean and log std
        init_cond_mean = init_cond_mean * obs_values_range + obs_values_min  # Shape [D*B, 1]
        init_cond_log_std = init_cond_log_std + torch.log(obs_values_range)  # Shape [D*B, 1]

        return init_cond_mean, init_cond_log_std

    def metric(self, y: Any, y_target: Any, seq_len=None):
        raise NotImplementedError("The metric method is not implemented in class FIMODE!")

    def new_stats(self) -> Dict:
        raise NotImplementedError("The new_stats method is not implemented in class FIMODE!")


ModelFactory.register("FIMODE", FIMODE)
