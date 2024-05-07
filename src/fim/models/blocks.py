import logging
import os
from pathlib import Path
from typing import List, Optional
import math

from fim.utils.helper import create_class_instance
import torch
from torch import nn
from transformers import (
    BitsAndBytesConfig,
    PreTrainedModel,
    TimeSeriesTransformerForPrediction,
)

from ..trainers.mixed_precision import is_bfloat_supported
from ..trainers.utils import is_distributed
from ..utils.logging import RankLoggerAdapter
from .utils import add_peft_adapter, freeze_transformer_layers


class Block(nn.Module):
    def __init__(self, resume: bool = False, **kwargs):
        super(Block, self).__init__(**kwargs)

        self.resume = resume

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

    def param_init(self):
        """
        Parameters initialization.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="leaky_relu")
                if module.bias.data is not None:
                    nn.init.zeros_(module.bias)


class Mlp(Block):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_layers: List[int],
        hidden_act: nn.Module | dict = nn.ReLU(),
        output_act: nn.Module | dict = None,
        **kwargs,
    ):
        super(Mlp, self).__init__(**kwargs)

        if isinstance(hidden_act, dict):
            hidden_act = create_class_instance(hidden_act.pop("name"), hidden_act)

        self.layers = nn.Sequential()
        in_size = in_features
        for i, h_size in enumerate(hidden_layers):
            self.layers.add_module(f"linear_{i}", nn.Linear(in_size, h_size))
            self.layers.add_module(f"activation_{i}", hidden_act)
            in_size = h_size
        self.layers.add_module("output", nn.Linear(hidden_layers[-1], out_features))
        if output_act is not None:
            if isinstance(output_act, dict):
                output_act = create_class_instance(output_act.pop("name"), output_act)
            self.layers.add_module("output_activation", output_act)

    def forward(self, x):
        return self.layers(x)


class HFBlock(Block):
    """
    Hugging Face Transformer-based neural network block.

    Args:
        backbone (str): Name of the Hugging Face transformer backbone.
        pad_token_id (int): Padding token ID.
        num_added_tokens (int): Number of additional tokens.
        load_in_8bit (bool): Load the model with 8-bit quantization.
        load_in_4bit (bool): Load the model with 4-bit quantization.
        use_bf16 (bool): Use bfloat16 data type for model weights.
        **kwargs: Additional keyword arguments for the base class.

    Attributes:
        _torch_dtype: Torch data type for model weights (None if not used).
        _quantization_config: Quantization configuration (None if not used).
        backbone: Hugging Face transformer backbone model.
        conf: AutoConfig instance.
    """

    def __init__(
        self,
        backbone: str | PreTrainedModel,
        backbone_path: Optional[Path] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_bf16: bool = False,
        device_map: Optional[str] = None,
        peft: Optional[dict] = None,
        freeze_first_n_layers: Optional[int] = None,
        freeze_backbone: Optional[bool] = False,
        adapter_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.logger = RankLoggerAdapter(logging.getLogger(self.__class__.__name__))
        self._device_map = device_map
        self._torch_dtype = None
        self._quantization_config = None
        self.is_peft = peft is not None and peft["method"] is not None
        self.backbone = None
        if isinstance(backbone, str):
            self._load_backbone(backbone, backbone_path, load_in_8bit, load_in_4bit, use_bf16)
        else:
            self.config = backbone.config
        if self.is_peft and (not self.resume or self.rank != 0):
            add_peft_adapter(
                backbone if self.backbone is None else self.backbone,
                peft,
                adapter_name,
            )

        assert not (freeze_backbone and freeze_first_n_layers is not None), self.logger.error(
            "Both `freeze_backbone` and `freeze_first_n_layers` are set at the same time!"
        )
        if not self.is_peft and freeze_backbone:
            for params in self.backbone.parameters():
                params.require_grad = False

        if not self.is_peft and freeze_first_n_layers is not None:
            freeze_transformer_layers(self.backbone, freeze_first_n_layers)

    def _load_backbone(
        self,
        backbone,
        backbone_path,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_bf16: bool = False,
    ):
        if load_in_8bit and load_in_4bit:
            raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
        elif load_in_8bit or load_in_4bit:
            self._quantization_config = BitsAndBytesConfig(load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit)
            self._device_map = "auto"
        if use_bf16 and is_bfloat_supported:
            self._torch_dtype = torch.float16

        self.backbone = (
            TimeSeriesTransformerForPrediction()
        )  # TODO: Change this to the actual model it is just example it is not tested

    def __str__(self):
        """
        Get a string representation of the HFBlock instance.

        Returns:
            str: A string representation of the instance.
        """
        return f"HFBlock(backbone={self.backbone.config.architectures[0]}, dtype={self._torch_dtype}, quantization={self._quantization_config})"

    def forward(self, **kwargs):
        return self.backbone(**kwargs)


class ResidualBlock(Block):
    """Residual block as defined in 'Das, A. et al. Long-term forecasting with TiDE: Time- series dense encoder'."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: List[int],
        hidden_act: nn.Module | dict = nn.ReLU(),
        output_act: nn.Module | dict = None,
        dropout: float = 0.0,
        **kwargs,
    ):
        super(ResidualBlock, self).__init__(**kwargs)

        self.mlp = Mlp(
            in_features,
            out_features,
            hidden_features,
            hidden_act,
            output_act,
            **kwargs,
        )
        self.dropout = nn.Dropout(dropout)
        self.skip_connection = nn.Linear(in_features, out_features)
        self.layer_norm = nn.LayerNorm(out_features)

    def forward(self, x):
        out_mlp = self.mlp(x)
        out_mlp = self.dropout(out_mlp)
        out_skip = self.skip_connection(x)

        out_final = out_mlp + out_skip
        out_final = self.layer_norm(out_final)

        return out_final


class PositionalEncoding(torch.nn.Module):
    """Positional encoding as defined in 'Vaswani, A. et al. Attention is all you need'."""

    def __init__(self, d_model: int, max_len: int):
        super(PositionalEncoding, self).__init__()

        # note: this implementation works only with even d_model.
        # if d_model is odd, we add an additional dimension to the encoding and remove it
        #  after the calculation of the encoding
        if d_model % 2 != 0:
            self.encoding = torch.zeros(max_len, d_model + 1)
        else:
            self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)

        if d_model % 2 != 0:
            self.encoding = self.encoding[:, :-1]

        self.encoding = self.encoding.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        return x + self.encoding[: x.size(0), :]


class DecoderBlock(nn.Module):
    """Consists of causal self-attention block and feed-forward network and layer normalization"""

    def __init__(self, d_model: int, num_heads: int, dropout: float, batch_first: bool = True):
        super(DecoderBlock, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=batch_first, dropout=dropout)

        # ffn layer
        self.mlp = Mlp(d_model, d_model, [d_model], nn.ReLU())
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = self.self_attn(x, x, x, key_padding_mask=mask, is_causal=False, need_weights=False)[0]
        x = self.mlp(x)
        x = self.dropout(x)
        x = self.layer_norm(x)
        return x
