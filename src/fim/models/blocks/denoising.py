from typing import Optional

import numpy as np
import torch
from scipy.signal import savgol_filter

from fim.models.blocks.base import Block


class SavGolFilter(Block):
    def __init__(self, window_length: int = 15, polyorder: int = 3):
        super(SavGolFilter, self).__init__()

        self.window_length = window_length
        self.polyorder = polyorder

        self.filter = savgol_filter

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Apply Savitzky-Golay filter to input tensor.

        As scipy's savgol-filter is numpy-based, we need to convert the input tensor to numpy & to cpu and back.
        """
        if mask is None:
            mask = torch.zeros_like(x, dtype=bool)
        x_dim = x.dim()
        if x_dim == 3:
            if x.size(-1) == 1:
                x = x.squeeze(-1)
                mask = mask.squeeze(-1) if mask is not None else None
            else:
                raise ValueError("Input tensor must have shape [B, T, 1] or [B, T].")

        # convert to numpy and cpu
        x_device = x.device
        x = x.cpu().numpy()
        mask = mask.cpu().numpy()

        # apply filter to each sample
        denoised_samples = []
        for sample, mask_sample in zip(x, mask):
            denoised_samples.append(self.apply_savgol_to_sample(sample.copy(), mask_sample))

        x_denoised = np.stack(denoised_samples)

        # convert back to torch tensor and send to original device
        x_denoised = torch.tensor(x_denoised, device=x_device)

        if x_dim == 3:
            x_denoised = x_denoised.unsqueeze(-1)
        return x_denoised

    def apply_savgol_to_sample(self, values: np.array, mask: np.array) -> np.array:
        """
        Apply Savitzky-Golay filter to a single sample.

        Args:
            values: np.array, values to be filtered.
            mask: np.array, mask indicating which values are masked out. 1 indicates that value is masked out..

        Returns:
            np.array: denoised values. Same shape as input.
        """
        obs_values = values[mask == 0]
        window_length = min(self.window_length, len(obs_values) - 1)
        polyorder = min(self.polyorder, window_length - 1)
        denoised_values = savgol_filter(obs_values, window_length, polyorder)
        values[mask == 0] = denoised_values
        return values
