from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from fim.models.blocks.base import MLP, Block
from fim.utils.helper import create_class_instance


class AttentionOperator(Block):
    def __init__(self, embed_dim, out_features, attention: dict = {}, projection: dict = {}):
        super().__init__()

        self.paths_summary_attention = PathsSummaryBlockAttention(embed_dim, **attention)
        self.projection = MLP(in_features=embed_dim, out_features=out_features, **projection)

    def forward(
        self,
        locations_encoding: Tensor,
        observations_encoding: Tensor,
        observations_mask: Optional[Tensor] = None,
        paths_mask: Optional[Tensor] = None,
    ):
        """
        Combines trunk and branch net (locations_encoding and observations_encoding) to evaluate neural operator at locations.

        Args:
            locations_encoding (Tensor): Shape: [B, G, H]
            observations_encoding (Tensor): Shape: [B, P, T, H]
            observations_mask (Optional[Tensor]): Shape: [B, P, T, 1], True indicates value is observed, i.e. not padded
            paths_mask (Optional[Tensor]): Shape: [B, P, 1], True indicates path is observed, i.e. not padded

        Returns:
            func_at_locations: Shape:[B, G, out_features]
        """

        paths_dependent_locations_encoding = self.paths_summary_attention(
            locations_encoding, observations_encoding, observations_mask, paths_mask
        )
        func_at_locations = self.projection(paths_dependent_locations_encoding)

        return func_at_locations


class PathsSummaryBlockAttention(Block):
    def __init__(self, embed_dim, **kwargs):  # args and kwargs are passed directly to nn.TransformerEncoderLayer
        super().__init__()

        self.locations_as_final_query: bool = kwargs.pop("locations_as_final_query", True)

        self.omega_1 = NonResidualAttentionLayer(d_model=embed_dim, batch_first=True, **kwargs)
        self.omega_2 = NonResidualAttentionLayer(d_model=embed_dim, batch_first=True, **kwargs)

    def forward(
        self,
        locations_encoding: Tensor,
        observations_encoding: Tensor,
        observations_mask: Optional[Tensor] = None,
        paths_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Return paths summary encoding for each location via 2 layer block attention.
        With paths_enc as values, the shapes are: [B, P, T, H] -> [B, P, G, H] -> [B, G, H]

        Args:
            locations_encoding (Tensor): Shape: [B, G, H]
            observations_encoding (Tensor): Shape: [B, P, T, H]
            observations_mask (Optional[Tensor]): Shape: [B, P, T, 1], True indicates value is observed, i.e. not padded
            paths_mask (Optional[Tensor]): Shape: [B, P, 1], True indicates path is observed, i.e. not padded

        Returns:
            paths_dep_loc_enc (Tensor): Paths dependent location encoding. Shape: [B, G, H]
        """

        B, G = locations_encoding.shape[:2]
        B, P, T = observations_encoding.shape[:3]

        # first summary per path
        locations_encoding_all_paths = locations_encoding[:, None, :, :].repeat(1, P, 1, 1)  # [B, P, G, embed_dim]
        locations_encoding_all_paths = locations_encoding_all_paths.view(B * P, G, -1)
        observations_encoding = observations_encoding.view(B * P, T, -1)

        # attention yields location dep paths encoding
        if observations_mask is not None:
            observations_mask = observations_mask.view(B * P, T, 1)

        loc_dep_path_enc = self.omega_1(
            locations_encoding_all_paths, observations_encoding, observations_encoding, observations_mask
        )  # [B * P, G, H]
        loc_dep_path_enc = loc_dep_path_enc.reshape(B, P, G, -1)
        loc_dep_path_enc = torch.transpose(loc_dep_path_enc, 1, 2).reshape(B * G, P, -1)  # [B * G, P, H]

        # single query attention yields encoding per location
        if self.locations_as_final_query is True:
            query = locations_encoding.reshape(B * G, 1, -1)

        else:
            query = torch.ones_like(loc_dep_path_enc[..., 0, :][..., None, :])  # [B * G, 1, H]

        if paths_mask is not None:
            paths_mask = paths_mask[:, None, :, :].repeat(1, G, 1, 1)  # [B, G, P, 1]
            paths_mask = paths_mask.view(B * G, P, 1)

        paths_dep_locations_encoding = self.omega_2(query, loc_dep_path_enc, loc_dep_path_enc, paths_mask)  # [B * G, 1, H]
        paths_dep_locations_encoding = paths_dep_locations_encoding.view(B, G, -1)

        return paths_dep_locations_encoding


class NonResidualAttentionLayer(Block):
    """
    Attention and residual feedforward like torch.nn.TransformerEncoderLayer.
    With key, query, values as inputs.
    Without residual connection after attention.
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
    ):
        super().__init__()
        # Attention
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=bias, batch_first=batch_first)

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

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Apply attention and layer norm, followed by a residual feedforward block.

        Args:
            queries (Tensor): Shape: [B, Q, H]
            keys, values (Tensor): Shape: [B, KV, H]
            key_padding_mask (Optional[Tensor]). Shape: [B, KV, 1], True indicates value is observed
        key_padding_mask: [B, K, 1], True if observed
        """

        if key_padding_mask is not None:  #
            B, K, _ = key_padding_mask.shape  # [B, K, 1]
            key_padding_mask.reshape(B, K)
            key_padding_mask = torch.logical_not(key_padding_mask.bool())  # we follow the opposite convention throughout the model!

        x = self.norm1(self._attn_block(queries, keys, values, key_padding_mask))  # no residual connection because shapes might not be same
        x = self.norm2(x + self._ff_block(x))

        return x

    def _attn_block(self, queries: Tensor, keys: Tensor, values: Tensor, key_padding_mask: Optional[Tensor] = None):
        x = self.attn(queries, keys, values, key_padding_mask=key_padding_mask)[0]
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
