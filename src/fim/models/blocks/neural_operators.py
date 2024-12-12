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

    def forward(self, locations_encoding: Tensor, paths_encoding: Tensor):
        """
        Combines trunk and branch net (locations_encoding and paths_encoding) to evaluate neural operator at locations.

        Args:
            locations_encoding (Tensor): Shape: [B, G, H]
            paths_encoding (Tensor): Shape: [B, P, T, H]

        Returns:
            func_at_locations: Shpaee:[B, G, out_features]
        """

        paths_dependent_locations_encoding = self.paths_summary_attention(locations_encoding, paths_encoding)
        func_at_locations = self.projection(paths_dependent_locations_encoding)

        return func_at_locations


class PathsSummaryBlockAttention(Block):
    def __init__(self, embed_dim, **kwargs):  # args and kwargs are passed directly to nn.TransformerEncoderLayer
        super().__init__()

        self.locations_as_final_query: bool = kwargs.pop("locations_as_final_query", True)

        self.omega_1 = NonResidualAttentionLayer(d_model=embed_dim, batch_first=True, **kwargs)
        self.omega_2 = NonResidualAttentionLayer(d_model=embed_dim, batch_first=True, **kwargs)

    def forward(self, locations_encoding: Tensor, paths_encoding: Tensor) -> Tensor:
        """
        Return paths summary encoding for each location via 2 layer block attention.
        With paths_enc as values, the shapes are: [B, P, T, H] -> [B, P, G, H] -> [B, G, H]

        Args:
            loc_enc (Tensor): Encoding of G locations. Shape: [B, G, H]
            paths_enc (Tensor): Encoding of P paths. Shape: [B, P, T, H]

        Returns:
            paths_dep_loc_enc (Tensor): Paths dependent location encoding. Shape: [B, G, H]
        """

        B, G = locations_encoding.shape[:2]
        B, P, T = paths_encoding.shape[:3]

        # first summary per path
        locations_encoding_all_paths = locations_encoding[:, None, :, :].repeat(1, P, 1, 1)  # [B, P, G, embed_dim]
        locations_encoding_all_paths = locations_encoding_all_paths.view(B * P, G, -1)
        paths_encoding = paths_encoding.view(B * P, T, -1)

        # attention yields location dep paths encoding
        loc_dep_path_enc = self.omega_1(locations_encoding_all_paths, paths_encoding, paths_encoding)  # [B * P, G, H]
        loc_dep_path_enc = loc_dep_path_enc.reshape(B, P, G, -1)
        loc_dep_path_enc = torch.transpose(loc_dep_path_enc, 1, 2).reshape(B * G, P, -1)  # [B * G, P, H]

        # single query attention yields encoding per location
        if self.locations_as_final_query is True:
            query = locations_encoding.reshape(B * G, 1, -1)

        else:
            query = torch.ones_like(loc_dep_path_enc[..., 0, :][..., None, :])  # [B * G, 1, H]

        paths_dep_locations_encoding = self.omega_2(query, loc_dep_path_enc, loc_dep_path_enc)  # [B * G, 1, H]
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

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        """
        Apply attention and layer norm, followed by a residual feedforward block.
        """

        x = self.norm1(self._attn_block(queries, keys, values, attn_mask))  # no residual connection because shapes might not be same
        x = self.norm2(x + self._ff_block(x))

        return x

    def _attn_block(self, queries: Tensor, keys: Tensor, values: Tensor, attn_mask: Optional[Tensor] = None):
        x = self.attn(queries, keys, values, attn_mask=attn_mask)[0]
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
