import copy
import logging
import math
import os
from pathlib import Path
from typing import List, Optional

import torch
from torch import nn
from transformers import (
    BitsAndBytesConfig,
    PreTrainedModel,
    TimeSeriesTransformerForPrediction,
)

from fim.utils.helper import create_class_instance

from ..trainers.mixed_precision import is_bfloat_supported
from ..trainers.utils import is_distributed
from ..utils.logging import RankLoggerAdapter
from .utils import SinActivation, add_peft_adapter, freeze_transformer_layers


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
        dropout: float = 0.0,
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

        # if no hidden layers are provided, the output layer is directly connected to the input layer
        if len(hidden_layers) == 0:
            hidden_layers = [in_features]
        self.layers.add_module("output", nn.Linear(hidden_layers[-1], out_features))

        if output_act is not None:
            if isinstance(output_act, dict):
                output_act = create_class_instance(output_act.pop("name"), output_act)
            self.layers.add_module("output_activation", output_act)

        if dropout > 0:
            self.layers.add_module("dropout", nn.Dropout(dropout))

    def forward(self, x):
        return self.layers(x)


class ResidualMlp(Block):
    """Implements a MLP with residual connections."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_layers: List[int],
        hidden_act: nn.Module | dict = nn.ReLU(),
        output_act: nn.Module | dict = None,
        skip_connections: list[int] = None,
        **kwargs,
    ):
        """
        Implements a MLP with optional skip connections and optional add+layer normalization.
        skip_connections: list of indices of layers where skip connections should be added. If no provided, this is a standard MLP Block
            Note: linear and activation layer are counted as one layer. Only one layer is skipped at a time.
        """
        super(Mlp, self).__init__(**kwargs)

        if isinstance(hidden_act, dict):
            hidden_act = create_class_instance(hidden_act.pop("name"), hidden_act)

        self.skip_connections = skip_connections if skip_connections is not None else []

        self.layers = nn.ModuleList()

        # hidden layers
        in_size = in_features
        for i, h_size in enumerate(hidden_layers):
            layer = nn.Sequential(nn.Linear(in_size, h_size), hidden_act)
            self.layers.add_module(f"layer_{i}", layer)

            in_size = h_size

        # output layer
        if output_act is not None:
            if isinstance(output_act, dict):
                output_act = create_class_instance(output_act.pop("name"), output_act)
            layer = nn.Sequential(nn.Linear(hidden_layers[-1], out_features), output_act)
        else:
            layer = nn.Linear(hidden_layers[-1], out_features)
        self.layers.add_module("output", layer)

    def forward(self, x):
        residual = x
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.skip_connections:
                x += residual
            residual = x
        return x


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


class PositionalEncoding(Block):
    """Positional encoding as defined in 'Vaswani, A. et al. Attention is all you need'."""

    def __init__(self, d_model: int, max_len: int):
        """
        Args:
            d_model (int): Dimension of the model.
                note: this implementation works only with even d_model.
                if d_model is odd, we add an additional dimension to the encoding and remove it
                after the calculation of the encoding
            max_len (int): Maximum length of the input sequence. (including train and test iteration!)
        """
        super(PositionalEncoding, self).__init__()

        if d_model % 2 != 0:
            encoding = torch.zeros(max_len, d_model + 1)
        else:
            encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)

        if d_model % 2 != 0:
            encoding = encoding[:, :-1]

        encoding = encoding.unsqueeze(0).transpose(0, 1)

        # register the encoding as buffer to make it available on all devices
        self.register_buffer("encoding", encoding, persistent=False)

    @property
    def device(self):
        return self.encoding.device

    def forward(self, x):
        return x + self.encoding[: x.size(0), :]


class DecoderBlock(Block):
    """Consists of causal self-attention block and feed-forward network and layer normalization"""

    def __init__(self, d_model: int, num_heads: int, dropout: float, batch_first: bool = True):
        super(DecoderBlock, self).__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            batch_first=batch_first,
            dropout=dropout,
        )

        # ffn layer
        self.mlp = Mlp(d_model, d_model, [d_model], nn.ReLU())
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        if mask.dim() == 3:
            mask = mask.squeeze(-1)
        x, _ = self.self_attn(
            x,
            x,
            x,
            key_padding_mask=mask,
            is_causal=False,
            need_weights=False,
        )
        x = self.mlp(x)
        x = self.dropout(x)
        x = self.layer_norm(x)
        return x


class TimeEncoding(Block):
    """
    Implements the time encoding as described in "Multi-time attention networks for irregularly sampled time series, Shukla & Marlin, 2020".

    Each time point t is encoded as a vector of dimension d_time:
        - first element: linear embedding of t: w_0*t + b_0
        - remaining elements: sinusoidal embedding of t with different frequencies: sin(w_i*t + b_i) for i in {1, ..., d_time-1}
    w_j and b_j are learnable parameters.
    """

    def __init__(self, dim_time: int):
        """
        Args:
            d_time (int): Dimension of the time representation
        """
        super(TimeEncoding, self).__init__()

        self.d_time = dim_time

        self.linear_embedding = nn.Linear(1, 1, bias=True)
        self.periodic_embedding = nn.Sequential(nn.Linear(1, dim_time - 1, bias=True), SinActivation())

    def forward(self, grid: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            grid (torch.Tensor): Grid of time points, shape (batch_size, seq_len, 1)
            mask (torch.Tensor): Mask for the grid, shape (batch_size, seq_len)
                where 1 indicates that the time point is masked out.

        Returns:
            torch.Tensor: Time encoding, shape (batch_size, seq_len, d_time)
        """
        if mask is None:
            mask = torch.zeros_like(grid)

        grid = grid * (1 - mask)

        linear = self.linear_embedding(grid)
        periodic = self.periodic_embedding(grid)

        return torch.cat([linear, periodic], dim=-1)


class Transformer(Block):
    """The encoder block of the transformer model as defined in 'Vaswani, A. et al. Attention is all you need'."""

    def __init__(
        self,
        num_encoder_blocks: int,
        dim_model: int,
        dim_time: int,
        num_heads: int,
        dropout: float,
        residual_mlp: dict,
        batch_first: bool = True,
    ):
        super(Transformer, self).__init__()

        self.num_heads = num_heads

        self.input_projection = nn.Linear(dim_time + 1, dim_model)

        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(dim_model, num_heads, dropout, copy.deepcopy(residual_mlp), batch_first=batch_first)
                for _ in range(num_encoder_blocks - 1)
            ]
        )

        self.final_query_vector = nn.Parameter(torch.randn(1, 1, dim_model))
        self.final_attention = nn.MultiheadAttention(dim_model, num_heads, dropout=dropout, batch_first=batch_first)

    def forward(self, x, key_padding_mask):
        """
        Args:
            x (torch.Tensor): Input tensor, shape (batch_size, seq_len, dim_time)
            mask (torch.Tensor): Mask for the input tensor, shape (batch_size, seq_len) with 1 indicating that the time point is masked out.
        """
        # prepare masks: create attention masks and ensure, both are of type float
        if key_padding_mask.dim() == 3:
            key_padding_mask = key_padding_mask.squeeze(-1)  #  (batch_size, seq_len)#

        attn_mask = self._pad2attn_mask(key_padding_mask)  # (batch_size * num_heads, seq_len, seq_len)
        attn_mask_summarizing_query = self._pad2attn_mask(
            key_padding_mask, target_seq_len=1
        )  # (batch_size * num_heads, 1, seq_len)
        key_padding_mask = key_padding_mask.float().masked_fill(key_padding_mask, float("-inf"))

        x = self.input_projection(x)  # (batch_size, seq_len, dim_model)

        # pass through encoder blocks
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, key_padding_mask, attn_mask)

        # use learnable query vector to get final embedding
        query = self.final_query_vector.repeat(x.size(0), 1, 1)

        attn_output, _ = self.final_attention(
            query,
            x,
            x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask_summarizing_query,
            is_causal=False,
            need_weights=False,
        )  # (batch_size, 1, dim_model)

        return attn_output

    def _pad2attn_mask(self, key_padding_mask, target_seq_len: Optional[int] = None):
        """
        Args:
            key_padding_mask (torch.Tensor): Mask for the input tensor, shape (batch_size, seq_len) with 1 indicating that the time point is masked out.
            target_seq_len (int): Target sequence length. If None, the sequence length is the same as the input sequence length.

        Returns:
            torch.Tensor: Attention mask, float valued, shape (batch_size * num_heads, seq_len, seq_len)
        """
        batch_size, seq_len = key_padding_mask.size()
        if target_seq_len is None:
            target_seq_len = seq_len

        expanded_mask = (
            key_padding_mask.unsqueeze(1).unsqueeze(2).expand(-1, self.num_heads, target_seq_len, -1)
        )  # Shape: (B, num_heads, seq_len, seq_len)

        # Reshape to (batch_size * num_heads, target_seq_len, seq_len)
        attention_mask = expanded_mask.reshape(batch_size * self.num_heads, target_seq_len, seq_len)

        # Convert boolean mask to float mask (1 -> -inf)
        attention_mask = attention_mask.float().masked_fill(attention_mask, float("-inf"))

        return attention_mask


class EncoderBlock(Block):
    """The encoder block of the transformer model as defined in 'Vaswani, A. et al. Attention is all you need'."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        residual_mlp: dict,
        batch_first: bool = True,
    ):
        super(EncoderBlock, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=batch_first)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.residual_mlp = create_class_instance(
            residual_mlp.pop("name"),
            residual_mlp,
        )
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask, attention_mask):
        attn_out, _ = self.self_attn(
            x,
            x,
            x,
            key_padding_mask=padding_mask,
            attn_mask=attention_mask,
            is_causal=False,
            need_weights=False,
        )
        attn_out = self.dropout(attn_out)
        x = self.layer_norm1(x + attn_out)

        mlp_out = self.residual_mlp(x)
        mlp_out = self.dropout(mlp_out)
        x = self.layer_norm2(x + mlp_out)

        return x
