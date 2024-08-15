import functools
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

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
from fim.utils.metrics import compute_metrics

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
        init_cond_net: dict,
        vector_field_net: dict,
        loss_configs: dict,
        normalization: bool = False,
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

        self.normalization = normalization

        self._create_model(
            time_encoding=time_encoding,
            trunk_net=trunk_net,
            branch_net=branch_net,
            combiner_net=combiner_net,
            vector_field_net=vector_field_net,
            init_cond_net=init_cond_net,
            loss_configs=loss_configs,
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

        self.vector_field_net = create_class_instance(
            vector_field_net.pop("name"),
            vector_field_net,
        )

        self.init_cond_net = create_class_instance(
            init_cond_net.pop("name"),
            init_cond_net,
        )

        match loss_configs.get("ode_solver"):
            case "rk4":
                from fim.models.utils import rk4

                self.ode_solver = rk4
            case _:
                raise ValueError(f"ODE solver {loss_configs.get('ode_solver')} not supported.")

        self.loss_scale_drift = loss_configs.pop("loss_scale_drift")
        self.loss_scale_init_cond = loss_configs.pop("loss_scale_init_cond")
        self.loss_scale_unsuperv_loss = loss_configs.pop("loss_scale_unsuperv_loss")

    def forward(
        self,
        batch,
        schedulers: Optional[dict] = None,
        step: Optional[int] = None,
        training: bool = False,
    ) -> dict:
        """
        Args:
            batch (dict): input batch with entries (each torch.Tensor)
                 obs_values [B, T, D] observation values
                 obs_times [B, T, 1] observation times
                 obs_mask [B, T, 1] observation mask, dtype: bool (0: value is observed, 1: value is masked out)
                 fine_grid_times [B, L, 1] time points of fine grid on which the vector field is evaluated
                 fine_grid_values [B, L, D] values of the fine grid
                with B: batch size, T: number of observation times, D: process dimension, L: number of fine grid points (locations)

        Returns:
            dict: losses, predictions (solutions at fine grid points)
        """
        obs_mask = batch["coarse_grid_observation_mask"].bool()
        unnormalized_obs_values = batch["coarse_grid_sample_paths"]
        unnormalized_obs_times = batch["coarse_grid_grid"]
        unnormalized_fine_grid_grid = batch["fine_grid_grid"]
        unnormalized_fine_grid_drift = batch["fine_grid_concept_values"]
        unnormalized_fine_grid_sample_paths = batch["fine_grid_sample_paths"]  # unnormalized

        batch_size, max_sequence_length, process_dim = unnormalized_obs_values.shape  # defines B, L, D=1

        if process_dim != 1:
            raise ValueError("Process dimension must be 1 in FIMODE Base model.")

        assert not obs_mask.all(), "Not allowed to have all values masked out."

        if self.normalization:
            (
                obs_values,  # [B, T, 1]
                obs_times,  # [B, T, 1]
                fine_grid_grid,  # [B, L, 1]
                normalization_parameters,
            ) = self.normalize_input(
                obs_times=unnormalized_obs_times,
                obs_values=unnormalized_obs_values,
                obs_mask=obs_mask,
                loc_times=unnormalized_fine_grid_grid,
            )
        else:
            obs_values = unnormalized_obs_values
            obs_times = unnormalized_obs_times
            fine_grid_grid = unnormalized_fine_grid_grid
            normalization_parameters = None

        encoded_input_sequence = self._encode_input_sequence(
            obs_values=obs_values, obs_times=obs_times, obs_mask=obs_mask
        )  # Shape [B, 1, dim_latent]

        learnt_vector_field_concepts = self._get_vector_field_concepts(
            grid_grid=fine_grid_grid, branch_out=encoded_input_sequence
        )  # Shape ([B, L, 1], [B, L, 1]) (normalized space)

        learnt_init_condition_concepts = self._get_init_condition_concepts(
            branch_out=encoded_input_sequence
        )  # Shape ([B, 1], [B, 1]) (normalized space)

        # renormalize vector field & initial condition distribution parameters
        if self.normalization:
            vector_field_concepts = self._renormalize_vector_field_params(
                vector_field_concepts=learnt_vector_field_concepts,
                normalization_parameters=normalization_parameters,
            )
            init_condition_concepts = self._renormalize_init_condition_params(
                learnt_init_condition_concepts, normalization_parameters
            )
        else:
            vector_field_concepts = learnt_vector_field_concepts
            init_condition_concepts = learnt_init_condition_concepts

        model_output: dict = {}

        losses: dict = self.loss(
            vector_field_concepts=vector_field_concepts,
            init_condition_concepts=init_condition_concepts,
            fine_grid_grid=fine_grid_grid,
            target_drift_fine_grid=unnormalized_fine_grid_drift,
            normalization_parameters=normalization_parameters,
            fine_grid_sample_paths=unnormalized_fine_grid_sample_paths,
        )

        model_output["losses"] = losses

        if not training:
            metrics, solution = self.new_stats(
                normalized_fine_grid_grid=fine_grid_grid,
                init_condition_concepts=learnt_init_condition_concepts,
                branch_out=encoded_input_sequence,
                normalization_parameters=normalization_parameters,
                fine_grid_sample_path=unnormalized_fine_grid_sample_paths,
            )
            model_output["metrics"] = metrics

            model_output["visualizations"] = {
                "fine_grid_grid": unnormalized_fine_grid_grid,
            }

            # visualization data of all samples
            model_output["visualizations"]["solution"] = {
                "learnt": solution,
                "target": unnormalized_fine_grid_sample_paths,
                "observation_times": unnormalized_obs_times,
                "observation_values": unnormalized_obs_values,
                "observation_mask": obs_mask,
            }
            model_output["visualizations"]["drift"] = {
                "learnt": learnt_vector_field_concepts[0],
                "target": unnormalized_fine_grid_drift,
                "certainty": torch.exp(learnt_vector_field_concepts[1]),
            }
            model_output["visualizations"]["init_condition"] = {
                "learnt": learnt_init_condition_concepts[0],
                "target": unnormalized_fine_grid_sample_paths[..., 0, :],
                "certainty": torch.exp(learnt_init_condition_concepts[1]),
            }
        else:
            model_output["metrics"] = {}

            # only output the drift of the first sample of the batch and init conditions of the first 10 samples
            # sample_id = 0
            # model_output["visualizations"] = {
            #     "fine_grid_grid": fine_grid_grid[0],
            # }
            # model_output["visualizations"]["drift"] = {
            #     "learnt": learnt_vector_field_concepts[0][sample_id],
            #     "target": unnormalized_fine_grid_drift[sample_id],
            #     "certainty": torch.exp(learnt_vector_field_concepts[1][sample_id]),
            # }
            # model_output["visualizations"]["init_condition"] = {
            #     "learnt": learnt_init_condition_concepts[0][:10],
            #     "target": unnormalized_fine_grid_sample_paths[:10, 0, :],
            #     "certainty": torch.exp(learnt_init_condition_concepts[1][:10]),
            # }

            model_output["visualizations"] = {}

        return model_output

    def _encode_input_sequence(
        self,
        obs_times,
        obs_values,
        obs_mask,
    ):
        # encode times
        encoded_obs_times = self.time_encoding(grid=obs_times)  # Shape [B, T, dim_time]

        # concatenate time encoding with normalized observation values. Use that dimensions are uncoupled.
        # first: reshape observation times & values to match shapewise
        obs_input_latent = torch.cat(
            [
                encoded_obs_times,  # Shape [B, T, dim_time],
                obs_values,  # Shape [B, T, 1]
            ],
            dim=-1,
        )  # Shape [B, T, dim_time + 1]

        # repeat obs_mask to match obs_input
        # obs_mask = obs_mask.repeat_interleave(self.process_dim, 1, 1)  # Shape [D*B, T, 1]

        encoded_input_sequence = self.branch_net(
            x=obs_input_latent, key_padding_mask=obs_mask
        )  # Shape [B, 1, dim_latent]

        return encoded_input_sequence

    def _get_vector_field_concepts(
        self,
        grid_grid: torch.Tensor,
        branch_out: torch.Tensor,
    ) -> tuple:
        """
        Compute mean and log standard deviation of the vector field at given grid times.

        Encode location times, pass through trunk net, combine with branch output in combiner net and compute mean and log standard deviation of the vector field concepts.

        Args:
            grid_grid (torch.Tensor): fine grid time points [B, L, 1]
            branch_out (torch.Tensor): output of the branch network [B*D, 1, dim_latent]
            process_dim (int): process dimension (= D)

        Returns:
            tuple: mean and log standard deviation of the vector field concepts ([B, L, D], [B, L, D)
        """
        # encode grid times
        loc_times = self.time_encoding(grid=grid_grid)  # Shape [B, L, dim_time]

        trunk_out = self.trunk_net(loc_times)  # Shape [B, L, dim_latent]

        # concat branch and trunk output: append branch output to each time step of trunk output
        combiner_in = torch.cat(
            [
                trunk_out,  # Shape [B, L, dim_latent]
                branch_out.repeat(1, trunk_out.shape[1], 1),  # Shape [B, L, dim_latent]
            ],
            dim=-1,
        )  # Shape [B, L, 2*dim_latent]
        combiner_out = self.combiner_net(combiner_in)  # Shape [B, L, dim_latent]

        # compute mean and log variance of the vector field at every location
        vector_field_concepts = self.vector_field_net(combiner_out)  # Shape [B, L, 2]

        # split into mean and log variance
        vector_field_mean, vector_field_log_std = torch.split(vector_field_concepts, 1, dim=-1)  # Shape each [B, L, 1]

        # TODO reshape to [B, L, D] B*D, L, 1 -> B, D, L, 1 -> B, L, D
        # vector_field_mean = vector_field_mean.reshape((*trunk_out.shape[:2], self.process_dim))
        # vector_field_var = vector_field_log_std.reshape((*trunk_out.shape[:2], self.process_dim))

        return vector_field_mean, vector_field_log_std

    def _get_init_condition_concepts(self, branch_out: torch.Tensor) -> tuple:
        """Compute mean and log standard deviation of the initial condition"""
        init_condition_concepts = self.init_cond_net(branch_out)  # Shape [B, 1, 2]

        init_condition_concepts = init_condition_concepts.squeeze(1)  # Shape [B, 2]

        # split into mean and log variance
        init_condition_mean, init_condition_log_std = torch.split(
            init_condition_concepts, 1, dim=-1
        )  # Shape each [B, 1]

        # reshape to [B, D]
        # init_condition_mean = init_condition_mean.reshape(-1, self.process_dim)
        # init_condition_var = init_condition_log_std.reshape(-1, self.process_dim)

        return init_condition_mean, init_condition_log_std

    def loss(
        self,
        vector_field_concepts: tuple,
        init_condition_concepts: tuple,
        target_drift_fine_grid: torch.Tensor,
        fine_grid_sample_paths: torch.Tensor,
        fine_grid_grid: torch.Tensor,
        normalization_parameters: dict,
    ) -> dict:
        """
        Compute the loss of the FIMODE model, also returns the solution of the inferred ODE at fine grid points.

        Args:
            vector_field_concepts (tuple): mean and log standard deviation of the vector field concepts (unnormalized) ([B, L, D], [B, L, D])
            init_condition_concepts (tuple): mean and log standard deviation of the initial condition concepts (unnormalized) ([B, D], [B, D])
            target_drift_fine_grid (torch.Tensor): target values (unnormalized) [B, L, D]
            normalized_fine_grid_grid (torch.Tensor): fine grid time points [B, L, 1]
            normalization_parameters (dict): normalization parameters for time and values

        Returns:
            dict: losses: supervised loss, unsupervised loss, (total) loss; solution at fine grid points
        """
        # supervised loss: maximize log-likelihood of values taken by vector field at observation times
        learnt_mean_drift, learnt_log_std_drift = vector_field_concepts
        learnt_var_drift = torch.exp(learnt_log_std_drift) ** 2

        learnt_mean_init_cond, learnt_log_std_init_cond = init_condition_concepts
        learnt_var_init_cond = torch.exp(learnt_log_std_init_cond) ** 2

        nllh_drift_avg = torch.mean(
            1 / 2 * (target_drift_fine_grid - learnt_mean_drift) ** 2 / learnt_var_drift + learnt_log_std_drift
        )

        nllh_init_cond_avg = torch.mean(
            1 / 2 * (fine_grid_sample_paths[..., 0, :] - learnt_mean_init_cond) ** 2 / learnt_var_init_cond
            + learnt_log_std_init_cond
        )

        # unsupervised loss (unnormalized space)
        if self.normalization:
            fine_grid_grid = self._renormalize_time(
                grid_grid=fine_grid_grid,
                normalization_parameters=normalization_parameters,
            )
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
            init_condition (torch.Tensor): initial condition [B, D] (unnormalized space)
            branch_out (torch.Tensor): output of the branch network [B, 1, dim_latent] (normalized space)
            normalization_parameters (dict): normalization parameters for time and values

        Returns:
            solution: torch.Tensor: solution at fine grid points [B, L, D] (unnormalized space)
        """
        B, L = fine_grid.shape[:-1]

        # need evaluations at fine grid points and one point in between each fine grid point -> add one point in between
        # get mid points between fine grid points
        mid_points = (fine_grid[..., 1:, :] + fine_grid[..., :-1, :]) / 2  # Shape [B, L-1, 1]
        # concat alternating fine grid points and mid points
        super_fine_grid_grid = torch.zeros(B, 2 * L - 1, 1, device=fine_grid.device, dtype=fine_grid.dtype)
        super_fine_grid_grid[:, ::2] = fine_grid
        super_fine_grid_grid[:, 1::2] = mid_points

        # compute drift at super fine grid points (in normalized space)
        super_fine_grid_drift, super_fine_grid_log_std = self._get_vector_field_concepts(
            grid_grid=super_fine_grid_grid, branch_out=branch_out
        )  # [B, 2*L-1, D]

        # unnormalize learnt drift & underlying time grid
        if self.normalization:
            super_fine_grid_drift, _ = self._renormalize_vector_field_params(
                normalization_parameters=normalization_parameters,
                vector_field_concepts=(super_fine_grid_drift, super_fine_grid_log_std),
            )
            super_fine_grid_grid = self._renormalize_time(
                grid_grid=super_fine_grid_grid, normalization_parameters=normalization_parameters
            )

        # compute solution using ode solver (unnormalized space)
        solution = self.ode_solver(
            super_fine_grid_grid=super_fine_grid_grid,
            super_fine_grid_drift=super_fine_grid_drift,
            initial_condition=init_condition,
        )  # [B, L, D]

        return solution

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
                   normalized location times (torch.Tensor),
                   normalization parameters (dict) with keys "obs_values_min", "obs_values_range", "obs_times_min", "obs_times_range"
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
        # TODO : Caution. Changed here.
        locations_norm_params = get_norm_params(loc_times, torch.zeros_like(loc_times, dtype=bool))

        normalized_obs_values = normalize(obs_values, obs_values_norm_params)  # [B, T, D]
        normalized_obs_times = normalize(obs_times, obs_times_norm_params)  # [B, T, 1]
        normalized_loc_times = normalize(loc_times, locations_norm_params)  # [B, L, 1]

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
        Rescale vector field concepts based on normalization parameters of observation values and times.

        Args:
            normalization_parameters (dict): holding all normalization parameters including obs_values_range and obs_times_range
            vector_field_concepts (tuple): mean and log standard deviation of the vector field distribution ([B, L, D], [B, L, D])
                log std is optional.

        Returns:
            if drift_log_std != None: return tuple: rescaled mean and log standard deviation of the concept distribution ([B, L, D], [B, L, D])
            if drift_log_std == None: return torch.Tensor: rescaled mean of the concept distribution ([B, L, D])
        """
        drift_mean, drift_log_std = vector_field_concepts

        shape = drift_mean.shape  # [B, L, D]

        # reshape (and repeat) values_range to match drift_mean
        values_range_view = (
            normalization_parameters["obs_values_range"].unsqueeze(1).repeat(1, shape[1], 1)
        )  # Shape [B, L, D]
        times_range_view = (
            normalization_parameters["obs_times_range"].unsqueeze(1).repeat(1, shape[1], shape[2])
        )  # Shape [B, L, D]

        # rescale  mean
        drift_mean = drift_mean * values_range_view / times_range_view  # Shape [B, L, D]

        # rescale log std if provided
        if drift_log_std is not None:
            learnt_drift_log_std = (
                drift_log_std + torch.log(values_range_view) - torch.log(times_range_view)
            )  # Shape [B, L, D]
            return drift_mean, learnt_drift_log_std

        else:
            return drift_mean

    def _renormalize_init_condition_params(self, init_cond_dist_params: tuple, normalization_parameters: dict) -> tuple:
        """
        Rescale the initial condition based on observation values normalization parameters.

        Args:
            init_cond_dist_params (tuple): mean and variance of the initial condition ([B,1], [B, 1])
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

    def _renormalize_time(self, grid_grid: torch.Tensor, normalization_parameters: dict) -> torch.Tensor:
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
        branch_out: torch.Tensor,
        normalization_parameters: dict,
        fine_grid_sample_path: torch.Tensor,
    ) -> Dict:
        """
        Get
            - solution
            - compute metrics between solution and target sample path
        """
        # get solution
        solution = self.get_solution(
            fine_grid=normalized_fine_grid_grid,
            init_condition=init_condition_concepts[0],
            branch_out=branch_out,
            normalization_parameters=normalization_parameters,
        )  # [B, L, D], [B, L, D] (unnormalized space)

        # get metrics
        metrics = self.metric(
            y=solution,
            y_target=fine_grid_sample_path,
        )

        return metrics, solution

    def is_peft(self) -> bool:
        return self.peft is not None and self.peft["method"] is not None


ModelFactory.register("FIMODE", FIMODE)
