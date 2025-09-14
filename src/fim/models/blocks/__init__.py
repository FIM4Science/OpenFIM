import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torchinfo
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel

from ...trainers.utils import is_distributed
from ...utils.logging import RankLoggerAdapter
from .base import MLP, IdentityBlock, MultiHeadLearnableQueryAttention, RNNEncoder, Transformer
from .normalization import MinMaxNormalization
from .positional_encodings import DeltaTimeEncoding, SineTimeEncoding


logger = RankLoggerAdapter(logging.getLogger(__name__))
__all__ = [
    MLP,
    SineTimeEncoding,
    Transformer,
    MinMaxNormalization,
    DeltaTimeEncoding,
    RNNEncoder,
    MultiHeadLearnableQueryAttention,
    IdentityBlock,
]


class AModel(PreTrainedModel, ABC):
    def __init__(self, config: PretrainedConfig, **kwargs):
        super().__init__(config)
        self.config = config

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

    def summary(self, x: dict):
        return torchinfo.summary(self, input_data=[x], mode="train")

    @classmethod
    def load_model(cls, model_path: Path):
        model_path = Path(model_path)
        config_path = model_path / "config.json"
        try:
            model_config = AutoConfig.from_pretrained(config_path)
        except Exception:
            # Fallback: if config.json lacks a recognizable model_type, default to fimhawkes
            with open(config_path, "r") as f:
                cfg_dict = json.load(f)
            if not cfg_dict.get("model_type"):
                cfg_dict["model_type"] = "fimhawkes"
            model_config = PretrainedConfig.from_dict(cfg_dict)
        model_weights_path = model_path / "model-checkpoint.pth"

        if not model_path.exists() or not config_path.exists() or not model_weights_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        # Load weights safely on systems without CUDA by mapping tensors to CPU
        try:
            model_state = torch.load(model_path / "model-checkpoint.pth", weights_only=True)
        except RuntimeError as e:
            if "Attempting to deserialize object on a CUDA device" in str(e) or (not torch.cuda.is_available()):
                model_state = torch.load(
                    model_path / "model-checkpoint.pth",
                    map_location=torch.device("cpu"),
                    weights_only=True,
                )
            else:
                raise
        # model_config = cls.config_class.from_pretrained(config_path)
        model = ModelFactory.create(model_config)
        model.load_state_dict(model_state)
        return model


class ModelFactory:
    model_types = {}
    model_types_with_data_params = {}

    @classmethod
    def register(cls, model_type: str, model_class: AModel, with_data_params: bool = False):
        if with_data_params:
            cls.model_types_with_data_params[model_type] = model_class
        else:
            cls.model_types[model_type] = model_class

    @classmethod
    def create(cls, config: dict | PretrainedConfig, dataset_config: dict = None) -> AModel:
        if isinstance(config, dict):
            config = PretrainedConfig.from_dict(config)
        model_class = cls.model_types.get(config.model_type)
        if model_class:
            return model_class(config) if dataset_config is None else model_class(config, dataset_config)
        else:
            raise ValueError(f"Invalid model type: {config.model_type}")
