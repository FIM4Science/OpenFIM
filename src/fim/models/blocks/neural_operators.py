from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from fim.models.blocks.base import Block
from fim.utils.helper import create_class_instance


torch.set_float32_matmul_precision("high")


class InducedSetAttentionLayer(Block):
    def __init__(self, d_model: int, num_induced_points: int, **layer):
        super(InducedSetAttentionLayer, self).__init__()

        learnable_queries = torch.ones(1, num_induced_points, d_model)
        learnable_queries = nn.init.xavier_uniform(learnable_queries)
        learnable_queries = torch.arange(num_induced_points).view(1, -1, 1).repeat(1, 1, d_model).float()
        self.learnable_queries = nn.Parameter(learnable_queries, requires_grad=True)  # [1, Q, H]

        layer.update({"d_model": d_model})
        self.res_attn_layer_0 = ResidualAttentionLayer(batch_first=True, **deepcopy(layer))
        self.res_attn_layer_1 = ResidualAttentionLayer(batch_first=True, **deepcopy(layer))

    @torch.compile()
    def forward(self, values: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Applies sinduced set attention with num_induced_points induced points.

        Args:
            values (Tensor): Shape: [B, V, H]
            key_padding_mask (Optional[Tensor]). Shape: [B, V, 1], True indicates value is padded

        Returns:
            transformed_values: [B, V, H]
        """
        B, V, H = values.shape

        queries = self.learnable_queries.expand(B, -1, -1)  # [B, Q, H]

        ind_H = self.res_attn_layer_0(queries, values, values, key_padding_mask)  # [B, Q, H]
        transformed_values = self.res_attn_layer_1(values, ind_H, ind_H)
        assert transformed_values.shape == (B, V, H), f"Got {transformed_values.shape}."

        return transformed_values


class InducedSetTransformerEncoder(Block):
    def __init__(self, d_model: int, num_layers: int, layer: Optional[dict] = {}):
        super(InducedSetTransformerEncoder, self).__init__()

        self.layers = nn.ModuleList([InducedSetAttentionLayer(d_model=d_model, **layer) for _ in range(num_layers)])

    def forward(self, values: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Applies num_layers of InducedSetAttentionLayer to values.

        Args:
            values (Tensor): Shape: [B, V, H]
            key_padding_mask (Optional[Tensor]). Shape: [B, V, 1], True indicates value is padded

        Returns:
            transformed_values: [B, V, H]
        """
        B, V, H = values.shape

        transformed_values = values

        for layer in self.layers:
            transformed_values = layer(transformed_values, key_padding_mask)

        assert transformed_values.shape == (B, V, H), f"Got {transformed_values.shape}."

        return transformed_values


class AttentionOperator(Block):
    def __init__(
        self,
        embed_dim,
        out_features,
        attention: dict = {},
        projection: dict = {},
        paths_block_attention: bool = True,
        num_res_layers: int = 1,
    ):
        super().__init__()

        self.paths_block_attention: bool = paths_block_attention

        if self.paths_block_attention is True:
            # first summarize each path with single query attention, then summarize all paths
            self.paths_summary_attention = PathsSummaryBlockAttention(embed_dim, **attention)

        else:
            # repeated transformations of query embeddings from observations (idea taken from GNOT paper)
            attention.pop("locations_as_final_query", None)

            self.res_layers = nn.ModuleList(
                [ResidualAttentionLayer(d_model=embed_dim, batch_first=True, **deepcopy(attention)) for _ in range(num_res_layers)]
            )

        projection.update({"in_features": embed_dim, "out_features": out_features})
        self.projection = create_class_instance(projection.pop("name"), projection)

    @torch.profiler.record_function("attention_operator")
    def forward(
        self,
        locations_encoding: Tensor,
        observations_encoding: Tensor,
        observations_padding_mask: Optional[Tensor] = None,
        paths_padding_mask: Optional[Tensor] = None,
    ):
        """
        Combines trunk and branch net (locations_encoding and observations_encoding) to evaluate neural operator at locations.

        Args:
            locations_encoding (Tensor): Shape: [B, G, H]
            observations_encoding (Tensor): Shape: [B, P, T, H]
            observations_padding_mask (Optional[Tensor]): Shape: [B, P, T, 1], True indicates value is padded
            paths_padding_mask (Optional[Tensor]): Shape: [B, P, 1], True indicates path is padded

        Returns:
            func_at_locations: Shape:[B, G, out_features]
        """

        if self.paths_block_attention is True:
            paths_dependent_locations_encoding = self.paths_summary_attention(
                locations_encoding, observations_encoding, observations_padding_mask, paths_padding_mask
            )  # [B, G, H]

        else:
            # repeated query attention over ALL observations yields encoding per location
            B, P, T, H = observations_encoding.shape

            observations_encoding = observations_encoding.view(B, P * T, H)  # [B, P * T, H]

            if observations_padding_mask is not None:
                observations_padding_mask = observations_padding_mask.view(B, P * T, 1)  # [B, P * T, 1]

            paths_dependent_locations_encoding = locations_encoding

            for res_layer in self.res_layers:
                paths_dependent_locations_encoding = res_layer(
                    paths_dependent_locations_encoding, observations_encoding, observations_encoding, observations_padding_mask
                )  # [B, G, H]

        func_at_locations = self.projection(paths_dependent_locations_encoding)  # [B, G, out_features]

        return func_at_locations


class PathsSummaryBlockAttention(Block):
    def __init__(self, embed_dim, **kwargs):  # args and kwargs are passed directly to nn.TransformerEncoderLayer
        super(PathsSummaryBlockAttention, self).__init__()

        self.locations_as_final_query: bool = kwargs.pop("locations_as_final_query", True)

        self.omega_1 = ResidualAttentionLayer(d_model=embed_dim, batch_first=True, query_residual=False, **kwargs)
        self.omega_2 = ResidualAttentionLayer(d_model=embed_dim, batch_first=True, query_residual=False, **kwargs)

    def forward(
        self,
        locations_encoding: Tensor,
        observations_encoding: Tensor,
        observations_padding_mask: Optional[Tensor] = None,
        paths_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Return paths summary encoding for each location via 2 layer block attention.
        With paths_enc as values, the shapes are: [B, P, T, H] -> [B, P, G, H] -> [B, G, H]

        Args:
            locations_encoding (Tensor): Shape: [B, G, H]
            observations_encoding (Tensor): Shape: [B, P, T, H]
            observations_padding_mask (Optional[Tensor]): Shape: [B, P, T, 1], True indicates value is padded
            paths_padding_mask (Optional[Tensor]): Shape: [B, P, 1], True indicates path is padded

        Returns:
            paths_dep_loc_enc (Tensor): Paths dependent location encoding. Shape: [B, G, H]
        """

        B, G, _ = locations_encoding.shape
        B, P, T, _ = observations_encoding.shape

        # first summary per path
        locations_encoding_all_paths = locations_encoding[:, None, :, :].repeat(1, P, 1, 1)  # [B, P, G, embed_dim]
        locations_encoding_all_paths = locations_encoding_all_paths.view(B * P, G, -1)
        observations_encoding = observations_encoding.view(B * P, T, -1)

        # attention yields location dep paths encoding
        if observations_padding_mask is not None:
            observations_padding_mask = observations_padding_mask.view(B * P, T, 1)

        loc_dep_path_enc = self.omega_1(
            locations_encoding_all_paths, observations_encoding, observations_encoding, observations_padding_mask
        )  # [B * P, G, H]
        loc_dep_path_enc = loc_dep_path_enc.reshape(B, P, G, -1)
        loc_dep_path_enc = torch.transpose(loc_dep_path_enc, 1, 2).reshape(B * G, P, -1)  # [B * G, P, H]

        # single query attention yields encoding per location
        if self.locations_as_final_query is True:
            query = locations_encoding.reshape(B * G, 1, -1)

        else:
            query = torch.ones_like(loc_dep_path_enc[..., 0, :][..., None, :])  # [B * G, 1, H]

        if paths_padding_mask is not None:
            paths_padding_mask = paths_padding_mask[:, None, :, :].repeat(1, G, 1, 1)  # [B, G, P, 1]
            paths_padding_mask = paths_padding_mask.view(B * G, P, 1)

        paths_dep_locations_encoding = self.omega_2(query, loc_dep_path_enc, loc_dep_path_enc, paths_padding_mask)  # [B * G, 1, H]
        paths_dep_locations_encoding = paths_dep_locations_encoding.view(B, G, -1)

        return paths_dep_locations_encoding


class ResidualAttentionLayer(Block):
    """
    Attention and residual feedforward like torch.nn.TransformerEncoderLayer. With key, query, values as inputs.
    Optional residual connection after attention is wrt. queries.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "torch.nn.ReLU",
        bias: bool = True,
        batch_first=True,
        query_residual=True,
        attn_method: str = "nn_multihead",
        lin_feature_map: str = "elu",
        lin_normalize: bool = True,
    ):
        super(ResidualAttentionLayer, self).__init__()
        self.batch_first = batch_first

        # Attention
        assert attn_method in ["nn_multihead", "linear"]

        if attn_method == "nn_multihead":
            self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias, batch_first=batch_first)

        elif attn_method == "linear":
            self.attn = LinearAttention(d_model, nhead, dropout=dropout, bias=bias, feature_map=lin_feature_map, normalize=lin_normalize)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)

        self.norm1 = nn.LayerNorm(d_model, bias=bias)
        self.norm2 = nn.LayerNorm(d_model, bias=bias)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Activation from str
        self.activation = create_class_instance(activation, {})

        # Flag for resiual connection with query
        self.query_residual = query_residual

    @torch.compile()
    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Apply attention and layer norm, followed by a residual feedforward block.

        Args:
            queries (Tensor): Shape: [B, Q, H] keys, values (Tensor): Shape: [B, KV, H]
            key_padding_mask (Optional[Tensor]). Shape: [B, KV, 1], True indicates value is padded
        key_padding_mask: [B, K, 1], True if observed
        """
        assert queries.ndim == keys.ndim == values.ndim == 3

        if key_padding_mask is not None:  #
            B, K, _ = key_padding_mask.shape  # [B, K, 1]
            key_padding_mask = key_padding_mask.reshape(B, K)  # [B, K]

        if self.query_residual is True:
            x = self.norm1(queries + self._attn_block(queries, keys, values, key_padding_mask))
        else:
            x = self.norm1(self._attn_block(queries, keys, values, key_padding_mask))

        x = self.norm2(x + self._ff_block(x))

        return x

    def _attn_block(self, queries: Tensor, keys: Tensor, values: Tensor, key_padding_mask: Optional[Tensor] = None):
        x = self.attn(queries, keys, values, key_padding_mask=key_padding_mask)[0]
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class LinearAttention(Block):
    """
    Linear attention (Fast Autoregressive Transformers with Linear Attention, Katharopoulos 2020) variants.
    If feature_map == elu and normalization == True, close to original idea.
    If feature_map == softmax and normalization == True, close to  (GNOT: A General Neural Operator Transformer for Operator Learning, Hao 2023).
    """

    def __init__(
        self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True, feature_map: str = "elu", normalize: bool = True
    ) -> None:
        super(LinearAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = embed_dim // num_heads

        # no dropout in reference implementations

        self.linear_Q = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.linear_K = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.linear_V = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)

        assert feature_map in ["elu", "softmax"], f"Got {feature_map}."
        self.feature_map = feature_map
        self.normalize = normalize

    @torch.compile()
    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None):
        B, Tq, _ = query.shape
        B, Tk, _ = key.shape

        q = self.linear_Q(query).reshape(B, Tq, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, Tq, head_dim]
        k = self.linear_K(key).reshape(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, Tk, head_dim]
        v = self.linear_V(value).reshape(B, Tk, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, Tk, head_dim]

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (B, Tk), f"Got {key_padding_mask.shape}."

            # masked keys and values will be multplied for v_
            key_padding_mask = key_padding_mask.view(B, 1, Tk, 1).bool()
            key_padding_mask = torch.broadcast_to(key_padding_mask, k.shape)
            k = torch.where(key_padding_mask, -torch.finfo(k.dtype).max, k)
            v = torch.where(key_padding_mask, 0, v)  # these values do not get used

        if self.feature_map == "softmax":
            q_ = q.softmax(dim=-1, dtype=torch.float32)
            k_ = k.softmax(dim=-1, dtype=torch.float32)

        else:  # elu
            q_ = torch.nn.functional.elu(q) + 1
            k_ = torch.nn.functional.elu(k) + 1

        # normmaliztion coefficient
        if self.normalize is True:
            k_summed = k_.sum(dim=-2, keepdim=True).expand(-1, -1, Tq, -1)  # [B, num_heads, Tq, head_dim]
            norm_coeff = (q_ * k_summed).sum(dim=-1, keepdim=True)  # [B, num_heads, Tq, 1]

        else:
            norm_coeff = 1

        # context
        kv = k_.transpose(-2, -1) @ v  # [B, num_heads, head_dim, head_dim]
        assert kv.shape == (B, self.num_heads, self.head_dim, self.head_dim), f"Got, {kv.shape}."

        # apply query
        attn_output = (1 / norm_coeff) * q_ @ kv  # [B, num_heads, Tq, head_dim]
        attn_output = attn_output.transpose(1, 2).reshape(B, Tq, self.embed_dim)  # [B, Tq, embed_dim]

        return self.out_proj(attn_output)


class ResidualEncoderLayer(Block):
    """
    Self-attention interface of ResidualAttentionLayer for TransformerEncoder.
    Mainly for standard TransformerEncoder with EncoderLayers other that nn.MultiheadAttention.
    """

    def __init__(self, d_model, batch_first=True, **kwargs):
        super(ResidualEncoderLayer, self).__init__()

        self.self_attn = ResidualAttentionLayer(d_model=d_model, batch_first=batch_first, **kwargs)

    def forward(
        self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, is_causal: bool = False
    ):
        return self.self_attn(src, src, src, src_key_padding_mask)
