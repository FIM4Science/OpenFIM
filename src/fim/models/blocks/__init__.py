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
from .base import MLP, MultiHeadLearnableQueryAttention, RNNEncoder, Transformer, TransformerBlock, TransformerEncoder, IdentityBlock
from .normalization import MinMaxNormalization
from .positional_encodings import DeltaTimeEncoding, SineTimeEncoding


logger = RankLoggerAdapter(logging.getLogger(__name__))
__all__ = [
    MLP,
    SineTimeEncoding,
    Transformer,
    MinMaxNormalization,
    TransformerBlock,
    TransformerEncoder,
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
        return torchinfo.summary(self, input_data=[x])

    @classmethod
    def load_model(cls, model_path: Path):
        model_path = Path(model_path)
        config_path = model_path / "config.json"
        model_weights_path = model_path / "model-checkpoint.pth"
        if not model_path.exists() or not config_path.exists() or not model_weights_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model_state = torch.load(model_path / "model-checkpoint.pth", weights_only=True)
        model_config = cls.config_class.from_pretrained(config_path)
        model = cls(model_config)
        model.load_state_dict(model_state)
        return model


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
    def create(cls, config: dict | PretrainedConfig) -> AModel:
        model_class = cls.model_types.get(config.get("model_type"))
        if isinstance(config, dict):
            config = model_class.config_class.from_dict(config)
        if model_class:
            return model_class(config)
        else:
            raise ValueError(f"Invalid model type: {config.model_type}")
