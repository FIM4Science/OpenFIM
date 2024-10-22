import torch
from torch import nn

from fim.models.blocks.base import Block
from fim.models.utils import SinActivation


class SineTimeEncoding(Block):
    """
    Implements the time encoding as described in "Multi-time attention networks for irregularly sampled time series, Shukla & Marlin, 2020".

    Each time point t is encoded as a vector of dimension d_time:
        - first element: linear embedding of t: w_0*t + b_0
        - remaining elements: sinusoidal embedding of t with different frequencies: sin(w_i*t + b_i) for i in {1, ..., d_time-1}
    w_j and b_j are learnable parameters.
    """

    def __init__(self, out_dim: int):
        """
        Args:
            d_time (int): Dimension of the time representation
        """
        super(SineTimeEncoding, self).__init__()

        self.out_dim = out_dim

        self.linear_embedding = nn.Linear(1, 1, bias=True)
        self.periodic_embedding = nn.Sequential(nn.Linear(1, out_dim - 1, bias=True), SinActivation())

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid (torch.Tensor): Grid of time points, shape (batch_size, seq_len, 1)

        Returns:
            torch.Tensor: Time encoding, shape (batch_size, seq_len, d_time)
        """
        linear = self.linear_embedding(grid)
        periodic = self.periodic_embedding(grid)

        return torch.cat([linear, periodic], dim=-1)


class DeltaTimeEncoding(Block):
    def __init__(self, resume: bool = False, **kwargs):
        super().__init__(resume, **kwargs)
        self.out_dim = 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x

        delta_t = torch.diff(x, dim=-2, prepend=x[..., :1, :])
        x = torch.cat([x, delta_t], dim=-1)
        return x
