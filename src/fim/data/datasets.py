import logging
import math
from pathlib import Path
from typing import Optional, Union

import torch
from datasets import DatasetDict, DownloadMode, get_dataset_split_names, load_dataset

from fim.data.utils import split_into_windows

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
        loading_function: Optional[callable] = None,
        **kwargs,
    ):
        self.logger = RankLoggerAdapter(logging.getLogger(__class__.__name__))
        self.path = path
        self.name = ds_name
        self.logger.debug(f"Loading dataset from {path} with name {ds_name} and split {split}.")
        self.split = verify_str_arg(split, arg="split", valid_values=["train", "test", "validation", None])

        if ds_name is None:
            ds_name = ""
        if loading_function is None:
            self.data = torch.load(path + ds_name + f"/{split}.pt")
        else:
            self.data = loading_function(path + ds_name)

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


class TimeSeriesImputationDatasetTorch(TimeSeriesDatasetTorch):
    def __init__(
        self,
        path: Union[str, Path],
        ds_name: Optional[str] = None,
        split: Optional[str] = "train",
        debugging_data_range: Optional[int] = None,
        output_fields: Optional[list] = None,
        output_fields_fimbase: Optional[list] = None,
        loading_function: Optional[callable] = None,
        key_mapping_fct: Optional[callable] = None,
        window_count: int = 3,
        overlap: int = 0,
        max_sequence_length: int = 256,
        imputation_mask: Optional[list[bool]] = None,
        **kwargs,
    ):
        super().__init__(
            path=path,
            ds_name=ds_name,
            split=split,
            debugging_data_range=debugging_data_range,
            output_fields=output_fields_fimbase,
            loading_function=loading_function,
            **kwargs,
        )

        self.output_fields = output_fields
        self.key_mapping_fct = key_mapping_fct

        self.window_count = window_count
        self.overlap = overlap
        self.window_size = math.ceil(max_sequence_length / window_count)
        self.overlap_size = int(self.window_size * overlap)
        self.imputation_mask = torch.tensor(imputation_mask) if imputation_mask is not None else None

        # get padding parameters:
        # if overlap>0: the first window is padded with overlap_size many elements (in the front)
        # last window is padded with remaining number of elements to reach window_size+overlap_size (in the back)
        size_last_window = max_sequence_length - (self.window_count - 1) * self.window_size
        self.padding_params = (self.overlap_size, self.window_size + self.overlap_size - size_last_window)

    def __post_init__(self):
        self.logger.debug(
            f"Time Series Dataset (Torch) for Imputation with {self.window_count} windows and {int(100*self.overlap)}% overlap loaded successfully."
        )

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        for k, v in item.items():
            if isinstance(v, torch.Tensor):
                item[k] = v.unsqueeze(0)

        # apply key mapping function if provided
        if self.key_mapping_fct is not None:
            k = list(item.keys())
            item = self.key_mapping_fct(item)

        # need to split into windows and provide observation part and location part
        max_sequence_length = item["coarse_grid_grid"].size(1)

        # observations. all shapes: from  [1, M, 1] to  [wc, wlen, 1]
        observation_values, padding_params = split_into_windows(
            item["coarse_grid_noisy_sample_paths"],
            self.window_count,
            self.overlap,
            max_sequence_length=max_sequence_length,
        )
        assert padding_params == self.padding_params
        observation_times, _ = split_into_windows(
            item["coarse_grid_grid"], self.window_count, self.overlap, max_sequence_length=max_sequence_length
        )
        observation_mask, _ = split_into_windows(
            item["coarse_grid_observation_mask"].bool(),
            self.window_count,
            self.overlap,
            max_sequence_length=max_sequence_length,
        )
        # imputation window data
        location_times, _ = split_into_windows(
            item["fine_grid_grid"],
            self.window_count,
            self.overlap,
            max_sequence_length=max_sequence_length,
            padding_value=None,
        )
        target_drift, _ = split_into_windows(
            item["fine_grid_concept_values"],
            self.window_count,
            self.overlap,
            max_sequence_length=max_sequence_length,
            padding_value=None,
        )
        target_sample_path, _ = split_into_windows(
            item["fine_grid_sample_paths"],
            self.window_count,
            self.overlap,
            max_sequence_length=max_sequence_length,
            padding_value=None,
        )
        assert (observation_mask.size(0) == self.window_count) and (observation_mask.size(2) == 1)
        assert (
            observation_mask.shape
            == observation_values.shape
            == observation_times.shape
            == target_drift.shape
            == target_sample_path.shape
            == location_times.shape
        )

        if self.imputation_mask is None:
            # generate window level mask: exactly one window is masked out (==1). shape: wc, 1
            mask = torch.zeros(self.window_count, dtype=torch.bool)
            masked_window_index = torch.randint(0, self.window_count, (1,)).item()
            mask[masked_window_index] = True
        else:
            mask = self.imputation_mask

        assert mask.sum() == 1
        assert mask.dim() == 1

        # select masked out window as imputation window and observed windows as observation windows
        # take all but masked out window
        observation_values = observation_values[~mask]
        observation_times = observation_times[~mask]
        observation_mask = observation_mask[~mask]

        # take only masked out window
        location_times = location_times[mask].squeeze(0)
        target_drift = target_drift[mask].squeeze(0)
        target_sample_path = target_sample_path[mask].squeeze(0)
        initial_conditions = target_sample_path[0, :]

        return {
            "observation_values": observation_values,
            "observation_times": observation_times,
            "observation_mask": observation_mask.bool(),
            "location_times": location_times,
            "target_drift": target_drift,
            "target_sample_path": target_sample_path,
            "initial_conditions": initial_conditions,
        }

    def __str__(self):
        return f"TimeSeriesImputationDatasetTorch(path={self.path}, name={self.name}, split={self.split},  window_count={self.window_count}, overlap={self.overlap}, dataset_keys={self.output_fields})"
