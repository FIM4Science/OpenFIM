import copy
import os
from typing import List, Optional, Union

import torch
from torch import nn

from fim.utils.helper import create_class_instance

from ..trainers.utils import is_distributed
from ..utils.logging import RankLoggerAdapter
from .utils import SinActivation, add_peft_adapter, freeze_transformer_layers


eps = 0  # 1e-6


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
    """
    Implement a multi-layer perceptron (MLP) with optional dropout.

    If defined dropout will be applied after each hidden layer but the final hidden and the output layer.
    """

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
        nr_hidden_layers = len(hidden_layers)
        for i, h_size in enumerate(hidden_layers):
            self.layers.add_module(f"linear_{i}", nn.Linear(in_size, h_size))
            self.layers.add_module(f"activation_{i}", hidden_act)
            if dropout != 0 and i < nr_hidden_layers - 1:
                self.layers.add_module(f"dropout_{i}", nn.Dropout(dropout))
            in_size = h_size

        # if no hidden layers are provided, the output layer is directly connected to the input layer
        if len(hidden_layers) == 0:
            hidden_layers = [in_features]
        self.layers.add_module("output", nn.Linear(hidden_layers[-1], out_features))

        if output_act is not None:
            if isinstance(output_act, dict):
                output_act = create_class_instance(output_act.pop("name"), output_act)
            self.layers.add_module("output_activation", output_act)

    def forward(self, x):
        return self.layers(x)


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

    def forward(self, grid: torch.Tensor):
        """
        Args:
            grid (torch.Tensor): Grid of time points, shape (batch_size, seq_len, 1)

        Returns:
            torch.Tensor: Time encoding, shape (batch_size, seq_len, d_time)
        """
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

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        """
        Args:
            x (torch.Tensor): Input tensor, shape (batch_size, seq_len, dim_time)
            mask (torch.Tensor): Mask for the input tensor, shape (batch_size, seq_len) with 1 indicating that the time point is masked out. If None, nothing is masked.
        """
        if key_padding_mask is None:
            key_padding_mask = torch.zeros_like(x[:, :, 0])
        elif key_padding_mask.dim() == 3:
            key_padding_mask = key_padding_mask.squeeze(-1)  #  (batch_size, seq_len)#z

        x = self.input_projection(x)  # (batch_size, seq_len, dim_model)

        # pass through encoder blocks
        for encoder_block in self.encoder_blocks:
            x = encoder_block(
                x,
                key_padding_mask,
            )

        # use learnable query vector to get final embedding
        query = self.final_query_vector.repeat(x.size(0), 1, 1)

        attn_output, _ = self.final_attention(
            query,
            x,
            x,
            key_padding_mask=key_padding_mask,
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

    def forward(self, x, padding_mask):
        # TODO ? Need attention mask?
        # x shape: B, sequence length, d_model, padding_mask shape: B, sequence length
        attn_out, _ = self.self_attn(
            x,
            x,
            x,
            key_padding_mask=padding_mask,
            is_causal=False,
            need_weights=False,
        )
        x = self.layer_norm1(x + attn_out)

        mlp_out = self.residual_mlp(x)
        x = self.layer_norm2(x + mlp_out)

        return x


### Normalization Blocks ###
class Standardization(Block):
    """Standardization block for normalizing input data via mean and std."""

    def __init__(self, mean_target: float = 0.0, std_target: float = 1.0):
        super(Standardization, self).__init__()
        self.mean_target = mean_target
        self.std_target = std_target if std_target != 0 else eps

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, tuple]:
        """
        Change statistics of given data x to target mean and std (Default: 0 and 1) with `X^ = std_target / std_data * (x - mean_data) + mean_target`.

        Standardization is applied along dim 1 (if x.dim()==3) or along dim=(1,2) (if x.dim()==4).

        Args:
            x: (torch.Tensor), shape [B, w, 1] or [B, wc, wlen, 1].
            mask: (torch.Tensor), shape [B, w, 1] or [B, wc, wlen, 1]. 1 indicating that value is masked out. Default: no values masked out.

        Returns:
            normalized_x
        """
        if mask is None:
            mask = torch.zeros_like(x, dtype=bool)

        x_dim = x.dim()
        if x_dim == 4:
            B, wc, wlen, D = x.shape
            x = x.view(B, wc * wlen, D)
            mask = mask.view(B, wc * wlen, D)

        # invert mask
        mask_inverted = ~mask

        # get masked mean and std per window
        mean_data = ((mask_inverted * x).sum(dim=1) / mask_inverted.sum(dim=1)).unsqueeze(-1)  # shape [B, 1, 1]
        var_data = ((mask_inverted * (x - mean_data) ** 2).sum(dim=1) / mask_inverted.sum(dim=1)).unsqueeze(
            -1
        )  # shape [B, 1, 1]

        normalized_x = (x - mean_data) / torch.sqrt(var_data + eps) * self.std_target + self.mean_target

        if x_dim == 4:
            normalized_x = normalized_x.view(B, wc, wlen, D)

        return normalized_x, (mean_data.squeeze(-1), var_data.squeeze(-1))

    def revert_normalization(self, x: torch.Tensor, data_concepts: Union[tuple, torch.Tensor]) -> torch.Tensor:
        """
        Revert above's standardization using the formula `X_out = std_data / std_target * (X - mean_target) + mean_data` where `X` is the input tensor.

        Args:
            x: torch.Tensor. Either 3 or 4 dimensional.
            data_concepts: mean and std of original data (statistics of data after applying this function).

        Returns:
            x_renormalized: torch.Tensor. Data with given mean and std.
        """
        if isinstance(data_concepts, tuple):
            mean_data, var = data_concepts
        elif isinstance(data_concepts, torch.Tensor) and data_concepts.shape[-1] == 2:
            mean_data, var = data_concepts.split(1, dim=-1)
        else:
            raise ValueError("Wrong format of data concept for reverting the standardization.")

        std_data = torch.sqrt(var.unsqueeze(-1) + eps)
        mean_data = mean_data.unsqueeze(-1)

        x_dim = x.dim()
        if x_dim == 4:
            B, wc, wlen, D = x.shape
            x = x.view(B, wc * wlen, D)

        x_renormalized = std_data / self.std_target * (x - self.mean_target) + mean_data

        if x_dim == 4:
            # need to reshape
            x_renormalized = x_renormalized.view(B, wc, wlen, D)
        return x_renormalized


class StandardizationSERIN(Standardization):
    """
    Standardization following "Statistics Embedding: Compensate for the Lost Part of Normalization in Time Series Forecasting" by xyz.

    Idea: Fuse normalized data with embedded statistics (learnable embedding).

    Idea: linear combination of "normal" standardization and embedding of statistics (mean and std of input data).
    Revertion of normalization is same as in Standardization (learnable part is ignored).
    """

    def __init__(
        self,
        mean_target: float = 0.0,
        std_target: float = 2.0,
        lin_factor: float = 0.5,
        network: dict = {},
    ):
        std_target = 3
        super(StandardizationSERIN, self).__init__(mean_target, std_target)

        self.linear_factor = torch.tensor(lin_factor)
        mlp = create_class_instance(network.pop("name"), network)
        layer_norm = nn.LayerNorm(network.get("out_features"), elementwise_affine=False)
        self.statistics_embedder = nn.Sequential(mlp, layer_norm)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, tuple]:
        """
        Standardize data by linear combination of Standardization and linearly embedded data statistics.

        Standardization is applied along dim 1 (if x.dim()==3) or along dim=(1,2) (if x.dim()==4).

        Args:
            x: torch.Tensor. Either 3 or 4 dimensional.
            mask: torch.Tensor, optional, default: no values are masked out. 1 indicates that a value is masked out.

        returns:
            standardized x: torch.Tensor
        """
        # x can either be 3 or 4 dimensional we need 3 dim.
        x_dim = x.dim()
        if x_dim == 4:
            B, wc, wlen, D = x.shape
            x = x.view(B, wc * wlen, D)
            mask = mask.view(B, wc * wlen, D)
        standardized_out, statistics = super().forward(x, mask)  # shape [B, w, 1], ([B,1], [B,1])
        embedded_statistics = self.statistics_embedder(torch.concat(statistics, dim=1)).unsqueeze(-1)  # shape [B, w, 1]

        # fuse normalized x and embedded statistics
        x_fused = (
            torch.sqrt(1 - self.linear_factor) * standardized_out + torch.sqrt(self.linear_factor) * embedded_statistics
        )
        if x_dim == 4:
            # need to reshape
            x_fused = x_fused.view(B, wc, wlen, D)

        return x_fused, statistics
