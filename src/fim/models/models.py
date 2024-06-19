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
        deeponet: dict,
        init_cond_distr_net: dict,
        vector_field_distr_net: dict,
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
            deeponet,
            init_cond_distr_net,
            vector_field_distr_net,
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

        self.combiner_net = create_class_instance(
            combiner_net.pop("name"),
            combiner_net,
        )

        self.vector_fiel_mean_net, = create_class_instance(
            vector_field_mean_net.pop("name"),
            vector_field_mean_net,
        )

        self.vector_fiel_var_net, = create_class_instance(
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
        # TODO: normalize values & times

        # sample target
        location_times, location_values = self.sample_locations(batch["fine_grid"], batch["fine_grid_values"])
        # encode time
        observation_times = self.time_encoding(batch["obs_times"])
        # pass through deeoOnet
        branch_out = self.branch_net(observation_times, batch["obs_values"])
        trunk_out = self.trunk_net(location_times)

        combined_out = self.combiner_net(trunk_out, branch_out)

        # compute mean and log variance of the vector field
        vector_field_mean = self.vector_fiel_mean_net(combined_out)
        vector_field_var = self.vector_fiel_var_net(combined_out)


        # compute mean and log variance of the initial condition
        init_condition_mean = self.init_cond_mean_net(branch_out)
        init_condition_var = self.init_cond_var_net(branch_out)

        # TODO denormalize

        # TODO compute loss

        # TODO setup return value

        torch.nn.SELU()

    def loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> Dict:
        raise NotImplementedError("The loss method is not implemented in class FIMODE!")

    def sample_locations(self, fine_grid, fine_grid_values):
        # TODO: implement
        raise NotImplementedError("The sample_locations method is not implemented in class FIMODE!")
