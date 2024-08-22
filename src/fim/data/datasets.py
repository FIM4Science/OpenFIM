import logging
from pathlib import Path
from typing import Optional, Union

import torch
from datasets import DatasetDict, DownloadMode, get_dataset_split_names, load_dataset

from ..utils.helper import verify_str_arg
from ..utils.logging import RankLoggerAdapter


class BaseDataset(torch.utils.data.Dataset):
    """
    Base class for time series datasets.

    Args:
        path (Union[str, Path]): The path to the dataset.
        name (Optional[str]): The name of the dataset. Defaults to None.
        split (Optional[str]): The split of the dataset. Defaults to "train".
        **kwargs: Additional keyword arguments to be passed to the `load_dataset` function.

    Attributes:
        logger: The logger object for logging messages.
        split (str): The split of the dataset.
        data (DatasetDict): The loaded dataset.

    Methods:
        __getitem__(self, idx): Returns the item at the given index.
        __str__(self): Returns a string representation of the dataset.

    """

    def __init__(
        self,
        path: Union[str, Path],
        ds_name: Optional[str] = None,
        split: Optional[str] = "train",
        download_mode: Optional[DownloadMode | str] = None,
        **kwargs,
    ):
        super().__init__()

        self.logger = RankLoggerAdapter(logging.getLogger(__class__.__name__))
        self.path = path
        self.name = ds_name
        self.logger.debug(f"Loading dataset from {path} with name {ds_name} and split {split}.")
        self.split = verify_str_arg(split, arg="split", valid_values=get_dataset_split_names(path, ds_name) + [None])
        self.data: DatasetDict = load_dataset(path, ds_name, split=split, download_mode=download_mode, **kwargs)

    def __post_init__(self):
        self.logger.debug("Base Dataset loaded successfully.")

    def __getitem__(self, idx):
        out = self.data[idx]
        if isinstance(out["target"], torch.Tensor):
            out["target"] = out["target"].unsqueeze(-1)
        return out | {"seq_len": len(out["target"])}

    def map(self, function, **kwargs):
        self.data = self.data.map(function, **kwargs)

    def __str__(self):
        return f"BaseDataset(path={self.path}, name={self.name}, split={self.split}, dataset={self.data})"

    def __len__(self):
        return len(self.data)


class TimeSeriesDataset(BaseDataset):
    """
    Base class for time series datasets.

    Args:
        path (Union[str, Path]): The path to the dataset.
        name (Optional[str]): The name of the dataset. Defaults to None.
        split (Optional[str]): The split of the dataset. Defaults to "train".
        **kwargs: Additional keyword arguments to be passed to the `load_dataset` function.

    Attributes:
        logger: The logger object for logging messages.
        split (str): The split of the dataset.
        data (DatasetDict): The loaded dataset.

    Methods:
        __getitem__(self, idx): Returns the item at the given index.
        __str__(self): Returns a string representation of the dataset.

    """

    def __init__(
        self,
        path: Union[str, Path],
        ds_name: Optional[str] = None,
        split: Optional[str] = "train",
        download_mode: Optional[DownloadMode | str] = None,
        debugging_data_range: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(path=path, ds_name=ds_name, split=split, download_mode=download_mode, **kwargs)
        # use only the first debugging_data_range time series
        if debugging_data_range is not None:
            debugging_data_range = min(debugging_data_range, len(self))
            self.data = self.data.select(range(debugging_data_range))

    def __post_init__(self):
        self.logger.debug("Time Series Dataset loaded successfully.")

    def __getitem__(self, idx):
        out = self.data[idx]

        return out | {"seq_len": len(out["coarse_grid_observation_mask"])}

    def __str__(self):
        return f"TimeSeriesDataset(path={self.path}, name={self.name}, split={self.split}, dataset={self.data})"


class TimeSeriesDatasetTorch(BaseDataset):
    """
    Base class for time series datasets where the data is given in torch format.

    Args:
        path (Union[str, Path]): The path to the dataset.
        name (Optional[str]): The name of the dataset. Defaults to None.
        split (Optional[str]): The split of the dataset. Defaults to "train".
        output_fields (Optional[list]): The columns to include in the output. Defaults to None i.e. all columns.
        **kwargs: Additional keyword arguments to be passed to the `load_dataset` function.

    Attributes:
        logger: The logger object for logging messages.
        split (str): The split of the dataset.
        data (DatasetDict): The loaded dataset.

    Methods:
        __getitem__(self, idx): Returns the item at the given index.
        __str__(self): Returns a string representation of the dataset.

    """

    def __init__(
        self,
        path: Union[str, Path],
        ds_name: Optional[str] = None,
        split: Optional[str] = "train",
        debugging_data_range: Optional[int] = None,
        output_fields: Optional[list] = None,
        **kwargs,
    ):
        self.logger = RankLoggerAdapter(logging.getLogger(__class__.__name__))
        self.path = path
        self.name = ds_name
        self.logger.debug(f"Loading dataset from {path} with name {ds_name} and split {split}.")
        self.split = verify_str_arg(split, arg="split", valid_values=["train", "test", "validation", None])
        self.data = torch.load(path + f"{split}.pt")

        if output_fields is not None:
            self.data = {k: v for k, v in self.data.items() if k in output_fields}

        if debugging_data_range is not None:
            debugging_data_range = min(debugging_data_range, len(self))
            self.data = {k: v[:debugging_data_range] for k, v in self.data.items()}

    def __post_init__(self):
        self.logger.debug("Time Series Dataset Torch loaded successfully.")

    def map(self, function, **kwargs):
        self.data = self.data.map(function, **kwargs)

    def __len__(self):
        key = list(self.data.keys())[0]
        return len(self.data[key])

    def __getitem__(self, idx):
        out = {k: (v[idx] if isinstance(v, torch.Tensor) else v[0][idx]) for k, v in self.data.items()}
        return out

    def __str__(self):
        return f"TimeSeriesDatasetTorch(path={self.path}, name={self.name}, split={self.split}, dataset_keys={list(self.data.keys())})"
