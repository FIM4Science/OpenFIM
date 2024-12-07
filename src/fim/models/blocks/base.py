import copy
import os
from typing import List, Optional, Union

import torch
from torch import Tensor, nn
from torch.nn.functional import scaled_dot_product_attention

from ...trainers.utils import is_distributed
from ...utils.helper import create_class_instance


class Block(nn.Module):
    def __init__(self, resume: bool = False, **kwargs):
        super(Block, self).__init__()
        self.initialization_scheme = kwargs.get("initialization_scheme", "kaiming_normal")
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
                if self.initialization_scheme == "kaiming_normal":
                    nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="leaky_relu")
                elif self.initialization_scheme == "lecun_normal":
                    nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="linear")
                if module.bias.data is not None:
                    nn.init.zeros_(module.bias)


class MLP(Block):
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
        super(MLP, self).__init__(**kwargs)

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
        self.param_init()

    def forward(self, x):
        return self.layers(x)


class MultiHeadLearnableQueryAttention(Block):
    def __init__(self, n_queries: int, n_heads: int, embed_dim: int, kv_dim: int = None, output_projection: bool = False, **kwargs):
        super().__init__(**kwargs)

        self.n_queries = n_queries
        self.n_heads = n_heads
        self.kv_dim = kv_dim if kv_dim is not None else embed_dim
        self.embed_dim = embed_dim
        self.output_projection = output_projection

        self.head_dim = self.kv_dim // n_heads if kv_dim else embed_dim // n_heads
        assert self.head_dim * n_heads == (self.kv_dim if kv_dim else embed_dim), "Dimension must be divisible by n_heads"

        self.q = nn.Parameter(torch.randn(1, self.n_heads, self.n_queries, self.head_dim))
        self.W_k = nn.Linear(self.embed_dim, self.kv_dim if kv_dim else self.embed_dim, bias=False)
        self.W_v = nn.Linear(self.embed_dim, self.kv_dim if kv_dim else self.embed_dim, bias=False)

        if self.output_projection:
            self.W_o = nn.Linear(self.kv_dim if kv_dim else self.embed_dim, self.embed_dim, bias=False)

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:  # FIXME: I think it would be less confusing if we not take some placeholder q as input
        B, L, _ = k.size()

        q = self.q.expand(B, -1, -1, -1).reshape(B * self.n_heads, self.n_queries, self.head_dim)
        k = self.W_k(k).reshape(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3).reshape(B * self.n_heads, L, self.head_dim)
        v = self.W_v(v).reshape(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3).reshape(B * self.n_heads, L, self.head_dim)

        h = scaled_dot_product_attention(q, k, v, mask)
        h = h.view(B, self.n_heads, self.n_queries, self.head_dim).permute(0, 2, 1, 3).contiguous()
        if self.output_projection:
            h = self.W_o(h.view(B, self.n_queries, self.n_heads * self.head_dim))[:, -1]
        else:
            h = h.view(B, -1)

        return h

    @property
    def out_features(self):
        return self.n_heads * self.head_dim * self.n_queries if not self.output_projection else self.n_heads * self.head_dim


class TransformerBlock(Block):
    def __init__(
        self,
        in_features: int,
        ff_dim: int,
        dropout: float,
        attention_head: Union[dict, nn.Module] = nn.MultiheadAttention,
        activation: Union[dict, nn.Module] = nn.ReLU(),
        normalization: Union[dict, nn.Module] = nn.LayerNorm,
        initialization_scheme: str = "kaiming_normal",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_dim = in_features
        self.attention_head = attention_head
        if isinstance(attention_head, dict):
            self.attention_head = create_class_instance(attention_head.pop("name"), attention_head)

        if isinstance(normalization, dict):
            norm_type = normalization.pop("name")
            self.norm1 = create_class_instance(norm_type, normalization)
            self.norm2 = create_class_instance(norm_type, normalization)
        else:
            self.norm1 = normalization(in_features)
            self.norm2 = normalization(in_features)

        self.ff = MLP(in_features, in_features, [ff_dim, in_features], hidden_act=activation, initialization_scheme=initialization_scheme)
        self.dropout = nn.Dropout(dropout)
        self.dropout_attention = nn.Dropout(dropout)

    def forward(self, x: Tensor, padding_mask: Optional[Tensor] = None, mask: Optional[Tensor] = None):
        x = self.dropout_attention(self.attention_head(x, x, x, key_padding_mask=padding_mask, attn_mask=mask)[0]) + x
        x = self.norm1(x)
        x = self.dropout(self.ff(x)) + x
        x = self.norm2(x)

        return x


class RNNEncoder(Block):
    def __init__(self, in_features: int, rnn: dict | nn.Module, **kwargs):
        super().__init__(**kwargs)
        self.rnn = rnn
        if isinstance(rnn, dict):
            rnn_name = rnn.pop("name")
            rnn["input_size"] = in_features
            self.rnn = create_class_instance(rnn_name, rnn)
        self.init_params()

    def forward(self, x: Tensor, seq_len: Tensor) -> Tensor:
        x = torch.nn.utils.rnn.pack_padded_sequence(x, seq_len.cpu(), batch_first=True, enforce_sorted=False)
        carry = self.get_init_carry(x.batch_sizes[0])
        out, _ = self.rnn(x, carry)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return out

    def get_init_carry(self, batch_size: int):
        return (
            torch.zeros(self.rnn.num_layers * (2 if self.rnn.bidirectional else 1), batch_size, self.rnn.hidden_size)
            .to(self.device)
            .contiguous(),
            torch.ones(self.rnn.num_layers * (2 if self.rnn.bidirectional else 1), batch_size, self.rnn.hidden_size)
            .to(self.device)
            .contiguous(),
        )

    @property
    def out_features(self):
        return self.rnn.hidden_size * 2 if self.rnn.bidirectional else self.rnn.hidden_size

    def init_params(self):
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)


class TransformerEncoder(Block):
    def __init__(self, num_layers: int, in_features: int, transformer_block: dict | TransformerBlock, **kwargs):
        super().__init__(**kwargs)
        if isinstance(transformer_block, dict):
            name = transformer_block.pop("name")
            transformer_block["in_features"] = in_features
            self.layers = MaskedSequential(*(create_class_instance(name, copy.deepcopy(transformer_block)) for _ in range(num_layers)))
        else:
            self.layers = MaskedSequential(*(transformer_block(**kwargs) for _ in range(num_layers)))

    def forward(self, x: Tensor, padding_mask: Optional[Tensor] = None) -> Tensor:
        return self.layers(x, padding_mask=padding_mask)

    @property
    def out_features(self):
        return self.layers[0].model_dim


class MaskedSequential(nn.Sequential):
    def forward(self, x: Tensor, mask: Optional[Tensor] = None, padding_mask: Optional[Tensor] = None) -> Tensor:
        for module in self._modules.values():
            x = module(x, mask=mask, padding_mask=padding_mask)
        return x


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
            key_padding_mask = torch.zeros_like(x[:, :, 0], dtype=bool)
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
        # x shape: [B, sequence length, d_model], padding_mask shape: [B, sequence length, 1]
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


class IdentityBlock(Block):
    """
    A dummy block that represents the identity function.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "out_features" in kwargs:
            self.out_features = kwargs["out_features"]

    def forward(self, x: dict | torch.Tensor) -> dict | torch.Tensor:
        return x
