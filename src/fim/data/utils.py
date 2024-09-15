"""Utility functions for working with data."""

import math
import os
import pickle
from typing import Optional

import numpy as np
import torch


def load_ODEBench_as_torch(directory: str) -> dict:
    """Loads data from the ODEBench dataset (given in cphefs dir) as torch tensors."""
    data = {}
    for filename in os.listdir(directory):
        if filename.endswith(".pickle"):
            key, _, _ = filename.rpartition(".")
            with open(os.path.join(directory, filename), "rb") as f:
                data[key] = pickle.load(f)

    # convert numpy arrays to torch tensors
    for key, value in data.items():
        if isinstance(value, tuple):
            value = value[0]
        if isinstance(value, np.ndarray):
            data[key] = torch.tensor(value, dtype=torch.float64)
        else:
            raise TypeError(f"Expected numpy array, got {type(value)}")
        if "mask" in key:
            # need to invert mask for correct usage of this implementation (1 indicates masked out)
            data[key] = ~data[key].bool()
    return data


def split_into_windows(
    x: torch.Tensor, window_count: int, overlap: float, max_sequence_length: int, padding_value: Optional[int] = 1
) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Split the tensor into overlapping windows.

    Therefore, split first into non-overlapping windows, than add overlap to the left for all but the first window.
    Pad with 1 if the window is smaller than the window size + overlap size. (i.e. elements will be masked out)

    Args:
        x (torch.Tensor): input tensor with shape (batch_size*process_dim, max_sequence_length, 1)
        max_sequence_length (int): the maximum length of the sequence
        padding_value (int): value to pad with.
            if None: for the first window the first value and for the last window the last value is used. (intereseting for locations)
            else: the value is used for padding. Recommnedation to use 1 as this automatically masks the values.

    Returns:
        torch.Tensor: tensor with shape (num_windows*batch_size*process_dim, window_size+overlap_size, 1)
        tuple[int, int]: overlap_size, padding_size_window
    """
    # Calculate the size of each window & overlap
    window_size = math.ceil(max_sequence_length / window_count)
    overlap_size = int(window_size * overlap)

    windows = []

    # Loop to extract non-overlapping windows and add overlap to the left for all but the first window
    start_idx = 0
    padding_size_windowing_end = None
    for i in range(window_count):
        if i == 0:
            # first window gets special treatment: no overlap to the left hence need to padd it for full size if overlap > 0
            window = x[:, start_idx : start_idx + window_size, :]
            if overlap_size > 0:
                if padding_value is not None:
                    padding = padding_value * torch.ones_like(window[:, :overlap_size, :], dtype=window.dtype)
                else:
                    first_value = x[:, 0:1, :]
                    padding = first_value.expand(-1, overlap_size, -1)
                window = torch.cat([padding, window], dim=1)
        else:
            start_idx = i * window_size - overlap_size
            window = x[:, start_idx : start_idx + window_size + overlap_size, :]
            # last window might need special treatment: padding to full size
            if (actual_window_size := window.size(1)) < window_size + overlap_size:
                # needed later for padding removal
                padding_size_windowing_end = window_size + overlap_size - actual_window_size

                if padding_value is not None:
                    padding = padding_value * torch.ones_like(
                        window[:, :padding_size_windowing_end, :], dtype=window.dtype
                    )
                else:
                    last_value = x[:, -1:, :]
                    padding = last_value.expand(-1, padding_size_windowing_end, -1)

                window = torch.cat([window, padding], dim=1)

        assert window.size(1) == window_size + overlap_size
        windows.append(window)

    return torch.concat(windows, dim=0), (overlap_size, padding_size_windowing_end)


def reorder_windows_per_sample(x: torch.Tensor, window_count: int, batch_size: int, process_dim: int):
    """
    Rearange windows to have all windows of one sample in a row.

    Follow-up function to split_into_windows. This function is needed to reorder the windows to have all windows of one sample in a row.

    Args:
        x (torch.Tensor): tensor with shape (num_windows*batch_size*process_dim, window_size+overlap_size, 1)
        window_count (int): number of windows
        batch_size (int): batch size (actual batch size, not multiplied with process_dim)
        process_dim (int): process dimension

    Returns:
        torch.Tensor: tensor with shape (batch_size*process_dim, window_count, window_size+overlap_size, 1)
    """
    all_samples = []
    for sample_id in range(batch_size * process_dim):
        sample = []
        for w in range(window_count):
            sample.append(x[sample_id + w * batch_size * process_dim])

        all_samples.append(torch.stack(sample, dim=0))
    x_reshaped = torch.stack(all_samples, dim=0)

    return x_reshaped


def make_single_dim(x: torch.Tensor):
    """Split last dimension of tensor in pieces of size 1 and concatenate along first dimension.
    Args:
        x (torch.Tensor): tensor with shape (B, ..., D)

    Returns:
        torch.Tensor: tensor with shape (B*D, ..., 1)
    """
    return torch.concat(
        x.split(1, dim=-1),
        dim=0,
    )


def make_multi_dim(x: torch.Tensor, batch_size: int, process_dim: int):
    """
    Reversion of `make_single_dim`.

    Args:
        x (torch.Tensor): tensor with shape (B*D, ..., 1)
        batch_size (int): batch size
        process_dim (int): process dimension

    Returns:
        torch.Tensor: tensor with shape (B, ..., D)
    """
    assert x.size(-1) == 1

    all_samples = []

    for sample_id in range(batch_size):
        sample_values = []
        for dim in range(process_dim):
            sample_values.append(x[sample_id + dim * batch_size, ..., 0])
        all_samples.append(torch.stack(sample_values, dim=-1))

    return torch.stack(all_samples, dim=0)


def repeat_for_dim(t, process_dim):
    """
    Repeat tensor for process_dim times along first dimension.
    """
    return torch.concat([t for _ in range(process_dim)], dim=0)
