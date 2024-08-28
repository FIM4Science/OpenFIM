"""Utility functions for working with data."""

import os
import pickle

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
