import copy
import logging

import torch
import torch.nn as nn
from peft import LoraConfig, PeftConfig
from transformers import PreTrainedModel

from ..utils.logging import RankLoggerAdapter


logger = RankLoggerAdapter(logging.getLogger("__main__"))


def get_peft_config(config: dict) -> PeftConfig:
    config = copy.deepcopy(config)
    config.pop("method")
    peft_config = LoraConfig(**config)
    return peft_config


def get_peft_trainable_parameters(model):
    """
    Gets the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    return f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"


def add_peft_adapter(model: PreTrainedModel, config: dict, adapter_name: str = None):
    adapter_config = get_peft_config(config)

    model.add_adapter(adapter_config, adapter_name)

    logger.info("Added PEFT addapter `%s` to model!", adapter_name)
    logger.info(get_peft_trainable_parameters(model))


def freeze_transformer_layers(model: nn.Module, num_layers: int = 0):
    """Freeze the layers of a model.

    Args:
        model (nn.Model): which layers we want to freeze.
        num_layer (int): the first `num_layers` will be frozen.
    """
    if num_layers == 0:
        return
    for i, layer in enumerate(model.model.layers):
        if num_layers == -1 or i < num_layers:
            for param in layer.parameters():
                param.requires_grad = False


class SinActivation(nn.Module):
    # def __init__(self):
    #     super(SinActivation, self).__init__()

    def forward(self, x):
        return torch.sin(x)
