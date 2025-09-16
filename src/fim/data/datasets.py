import itertools
import json
import logging
import math
import operator
import os
import pathlib
import random
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from functools import reduce
from typing import Any, List, Optional, Union

import h5py
import numpy as np
import torch
import torch.distributed as dist
import torch.utils
import torch.utils.data
from datasets import Dataset, DatasetDict, DownloadMode, get_dataset_split_names, load_dataset
from torch import Tensor
from torch.utils.data import default_collate

from fim import data_path
from fim.data.config_dataclasses import FIMDatasetConfig
from fim.data.utils import load_h5

from ..typing import Path, Paths
from ..utils.helper import verify_str_arg
from ..utils.logging import RankLoggerAdapter
from .utils import clean_split_from_size_info, load_file, split_into_variable_windows


class HFDataset(torch.utils.data.Dataset):
    """
    Base class for time series datasets.

    Args:
        path (Union[str, Path]): The path to the dataset.
        config (Optional[str]): The config of the dataset. Defaults to None.
        split (Optional[str]): The split of the dataset. Defaults to "train".
        visible_columns (Optional[List[str]]): List of columns to be visible in the dataset. Defaults to None, meaning all columns are visible.
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
        conf: Optional[str] = None,
        split: Optional[str] = "train",
        download_mode: Optional[DownloadMode | str] = None,
        rename_columns: Optional[dict[str, str]] = None,
        output_columns: Optional[List[str]] = None,
        batch_columns: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__()

        self.logger = RankLoggerAdapter(logging.getLogger(__class__.__name__))
        self.path = path
        self.name = conf
        self.rename_columns = rename_columns
        self.visible_columns = output_columns
        self.logger.debug(f"Loading dataset from {path} with config {conf} and split {split}.")
        split_clean = clean_split_from_size_info(split)
        self.split = verify_str_arg(
            split_clean, arg="split", valid_values=get_dataset_split_names(path, conf, trust_remote_code=True) + [None]
        )
        self.data: DatasetDict | Dataset = load_dataset(path, conf, split=split, download_mode=download_mode, **kwargs)
        self.logger.debug(f"Dataset from {path} with config {conf} and split {split} loaded successfully.")
        self.data.set_format(type="torch")
        self.batch_columns = batch_columns
        if rename_columns:
            self.data = self.data.rename_columns(rename_columns)
        if output_columns:
            self.data = self.data.select_columns(output_columns)

    def __getitem__(self, idx):
        out = self.data[idx]
        if self.batch_columns:
            out = {k: v for k, v in out.items() if k in self.batch_columns}
        return out

    def map(self, function, **kwargs):
        self.data = self.data.map(function, **kwargs)

    def __str__(self):
        return f"HFDataset(path={self.path}, name={self.name}, split={self.split}, dataset={self.data})"

    def __repr__(self):
        return f"HFDataset(path={self.path}, name={self.name}, split={self.split}, dataset={self.data})"

    def __len__(self):
        return len(self.data)


class TimeSeriesDataset(torch.utils.data.Dataset):
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


class FIMDataset(torch.utils.data.Dataset):
    """
    FimDataset is a custom dataset class for loading and handling data from a specified path.

    Each file in the directory is loaded as a separate key-value pair in the dataset dictionary. In case _files_to_load_ is
    specified only the files in the dictionary are loaded. The data is loaded from the files and stored in the dataset

    The file type is automatically detected. Currently, the supported file types are .h5, .pickle, and .pt.

    Example:
        ```python
        dataset = FimDataset(path="data/mjp/train")
        print(dataset)

        # example with files_to_load
        files_to_load = {
            "fine_grid": "fine_grid_grid.pt",
            "fine_grid_masks": "fine_grid_masks.pt",
        }
        dataset = FimDataset(path="data/mjp/train", files_to_load=files_to_load)
        print(dataset)
        ```
    Attributes:
        path (Union[Path, Paths]): The path or list of paths to the dataset files.
        files_to_load (Optional[dict]): A dictionary specifying which files to load.
        data_limit (Optional[int]): An optional limit on the number of data entries to load from each file.
        logger (RankLoggerAdapter): Logger for the dataset class.
        data (dict): A dictionary containing the loaded data.
    Methods:
        __init__(path: Union[Path, Paths], files_to_load: Optional[dict] = None, data_limit: Optional[int] = None):
            Initializes the FIMDataset with the given path, files to load, and data limit.
        __load_data() -> dict:
            Loads the data from the specified files and returns it as a dictionary.
        __getitem__(idx):
            Returns the data entry at the specified index.
        __get_files() -> list[tuple[str, Path]]:
            Retrieves the list of files to load based on the specified files_to_load or all files in the path.
        path:
            Property getter and setter for the dataset path.
        __len__():
            Returns the number of data entries in the dataset.
        __str__():
            Returns a string representation of the FIMDataset.
    """

    def __init__(
        self,
        path: Path | Paths,
        files_to_load: Optional[dict] = None,
        data_limit: Optional[int] = None,
        field_name_for_dimension_grouping: Optional[str] = None,
    ):
        super().__init__()

        self.logger = RankLoggerAdapter(logging.getLogger(__class__.__name__))
        self.path: Paths = path
        self.files_to_load = files_to_load
        self.field_name_for_dimension_grouping = field_name_for_dimension_grouping
        self.data_limit = data_limit
        self.logger.debug(f"Loading dataset from {path} with files {files_to_load}.")
        self.__different_last_dim = False
        self.data = self.__load_data()
        if self.__different_last_dim:
            self.group_lengths = {group: len(next(iter(data.values()))) for group, data in self.data.items()}
        self.logger.debug(f"Dataset from {path} loaded successfully.")

    def __load_data(self) -> dict:
        data = defaultdict(list)
        files_to_load = self.__get_files()
        for file_name, file_path in files_to_load:
            data[file_name].append(load_file(file_path)[: self.data_limit])

        if self.field_name_for_dimension_grouping:
            self.__different_last_dim = self.__check_dimension_consistency(data)
            if self.__different_last_dim:
                grouped_data = defaultdict(lambda: defaultdict(list))
                idx_to_dim = [v.shape[-1] for v in data[self.field_name_for_dimension_grouping]]
                for k, v in data.items():
                    for i, tensor in enumerate(v):
                        grouped_data[idx_to_dim[i]][k].append(tensor)
                return {dim: {k: torch.cat(v) for k, v in group.items()} for dim, group in grouped_data.items()}
        return {k: torch.cat(v) for k, v in data.items()}

    def __check_dimension_consistency(self, data):
        different_last_dim = False
        base_dim = data[self.field_name_for_dimension_grouping][0].shape[-1]
        for v in data[self.field_name_for_dimension_grouping][1:]:
            if v.shape[-1] != base_dim:
                different_last_dim = True
                break
        return different_last_dim

    def __getitem__(self, idx):
        if self.__different_last_dim:
            group_idx, group_key = self.__retrieve_group_index(idx)
            data = {k: v[group_idx] for k, v in self.data[group_key].items()}
            data["_group_dim"] = group_key
            return data

        return {k: v[idx] for k, v in self.data.items()}

    def __retrieve_group_index(self, idx):
        group_idx = 0
        cumulative_length = 0
        for length in self.group_lengths.values():
            if idx < cumulative_length + length:
                break
            cumulative_length += length
            group_idx += 1
        group_key = list(self.group_lengths.keys())[group_idx]
        group_idx = idx - cumulative_length
        return group_idx, group_key

    def __get_files(self) -> Paths:
        if self.files_to_load is not None:
            files_to_load = [(key, path / file_name) for key, file_name in self.files_to_load.items() for path in self.path]
        else:
            files_to_load = [(f.stem, f) for path in self.path for f in path.iterdir() if f.is_file()]
        return files_to_load

    @property
    def is_last_dim_varying(self):
        return self.__different_last_dim

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path: Path | Paths):
        assert isinstance(path, (str, Path, list, tuple)), f"Expected path to be of type str, Path, or list, but got {type(path)}."
        if not isinstance(path, (tuple, list)):
            path = [path]
        path = [pathlib.Path(p) for p in path]
        if not all(p.exists() for p in path):
            missing_paths = [str(p) for p in path if not p.exists()]
            self.logger.error(f"Paths {', '.join(missing_paths)} do not exist.")
            raise AssertionError(f"Paths {', '.join(missing_paths)} do not exist.")
        self._path = path

    def __len__(self):
        if self.__different_last_dim:
            return sum([len(next(iter(v.values()))) for v in self.data.values()])
        else:
            return len(next(iter(self.data.values())))

    def __str__(self):
        return f"FimDataset(path={self.path}, files_to_load={self.files_to_load})"


class HawkesDataset(FIMDataset):
    """
    Specialized dataset for Hawkes processes that handles variable numbers of marks.
    Inherits from FIMDataset but adds mark-specific padding functionality.
    """

    def __init__(
        self,
        path: Path | Paths,
        files_to_load: Optional[dict] = None,
        data_limit: Optional[int] = None,
        field_name_for_dimension_grouping: Optional[str] = None,
    ):
        # Initialize parent class attributes without calling its __load_data method
        self.logger = RankLoggerAdapter(logging.getLogger(__class__.__name__))
        self.path = path
        self.files_to_load = files_to_load or {}
        self.data_limit = data_limit
        self.field_name_for_dimension_grouping = field_name_for_dimension_grouping
        self._FIMDataset__different_last_dim = False

        # Load data using our custom method
        self.data = self._HawkesDataset__load_data()

    def _HawkesDataset__load_data(self) -> dict:
        data = defaultdict(list)
        files_to_load = self._FIMDataset__get_files()
        for file_name, file_path in files_to_load:
            data[file_name].append(load_file(file_path)[: self.data_limit])

        if self.field_name_for_dimension_grouping:
            # Handle both single field name and list of field names for dimension grouping
            grouping_fields = self.field_name_for_dimension_grouping
            if isinstance(grouping_fields, list):
                grouping_field = grouping_fields[0]  # Use first field for consistency check
            else:
                grouping_field = grouping_fields

            self._FIMDataset__different_last_dim = self.__check_hawkes_dimension_consistency(data, grouping_field)
            if self._FIMDataset__different_last_dim:
                grouped_data = defaultdict(lambda: defaultdict(list))
                # For Hawkes processes, marks dimension is the second dimension (shape[1])
                idx_to_dim = [v.shape[1] for v in data[grouping_field]]
                for k, v in data.items():
                    for i, tensor in enumerate(v):
                        grouped_data[idx_to_dim[i]][k].append(tensor)
                return {dim: {k: torch.cat(v) for k, v in group.items()} for dim, group in grouped_data.items()}

        # Always check if we need to pad tensors with different mark dimensions
        tensors_to_pad = ["base_intensity_functions", "kernel_functions"]
        need_padding = False
        max_marks = 0

        for key in tensors_to_pad:
            if key in data:
                # For Hawkes processes, marks dimension is the second dimension (index 1)
                marks_dims = [tensor.shape[1] for tensor in data[key]]
                if len(set(marks_dims)) > 1:  # Different mark dimensions found
                    need_padding = True
                    max_marks = max(max_marks, max(marks_dims))
        if need_padding:
            # Pad tensors to have the same number of marks
            for key in tensors_to_pad:
                if key in data:
                    padded_tensors = []
                    for tensor in data[key]:
                        current_marks = tensor.shape[1]
                        if current_marks < max_marks:
                            # Pad the second dimension (marks dimension)
                            pad_size = max_marks - current_marks
                            # Create padding with shape [N, pad_size, remaining_dims...]
                            pad_shape = (tensor.shape[0], pad_size) + tensor.shape[2:]
                            padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
                            tensor = torch.cat([tensor, padding], dim=1)
                        padded_tensors.append(tensor)
                    data[key] = padded_tensors

        return {k: torch.cat(v) for k, v in data.items()}

    def __check_hawkes_dimension_consistency(self, data, grouping_field=None):
        """Check consistency for Hawkes-specific mark dimensions."""
        if grouping_field is None:
            grouping_field = self.field_name_for_dimension_grouping

        different_marks_dim = False
        if grouping_field in data and len(data[grouping_field]) > 0:
            # For Hawkes processes, check the second dimension (marks dimension)
            base_dim = data[grouping_field][0].shape[1]
            for v in data[grouping_field][1:]:
                if v.shape[1] != base_dim:
                    different_marks_dim = True
                    break
        return different_marks_dim

    @property
    def is_last_dim_varying(self):
        return self._FIMDataset__different_last_dim

    def __getitem__(self, idx):
        if self.is_last_dim_varying:
            # Handle grouped data structure
            group_idx, data_idx = self._retrieve_group_index(idx)
            item = {k: v[data_idx] for k, v in self.data[group_idx].items()}
            item["_group_dim"] = group_idx  # Add group dimension for collate function
            return item
        else:
            # Handle flat data structure
            return {k: v[idx] for k, v in self.data.items()}

    def _retrieve_group_index(self, idx):
        """Retrieve group index and data index for variable dimension datasets."""
        cumulative_size = 0
        for group_idx, group_data in self.data.items():
            group_size = len(next(iter(group_data.values())))
            if cumulative_size + group_size > idx:
                data_idx = idx - cumulative_size
                return group_idx, data_idx
            cumulative_size += group_size
        raise IndexError(f"Index {idx} out of range for dataset of size {cumulative_size}")

    def __len__(self):
        if self.is_last_dim_varying:
            # For grouped data, sum up the sizes of all groups
            total_size = 0
            for group_data in self.data.values():
                if group_data:
                    total_size += len(next(iter(group_data.values())))
            return total_size
        else:
            # For flat data, get the size from any key
            if self.data:
                return len(next(iter(self.data.values())))
            return 0


class TimeSeriesDatasetTorch(torch.utils.data.Dataset):
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
            self.data = torch.load(path + ds_name + f"/{split}.pt", weights_only=True)
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
        out = {k: (v[idx] if isinstance(v, Tensor) else v[0][idx]) for k, v in self.data.items()}
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
        min_iwindow_percentage: float = 0.1,
        max_iwindow_percentage: float = 0.3,
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
        self.min_iwindow_percentage = min_iwindow_percentage
        self.max_iwindow_percentage = max_iwindow_percentage

        self.imputation_mask = torch.tensor(imputation_mask, dtype=torch.bool) if imputation_mask is not None else None
        if self.imputation_mask is not None:
            assert sum(self.imputation_mask) == 1, "Only one window can be masked out for imputation."

        size_last_window = max_sequence_length - (self.window_count - 1) * self.window_size
        self.padding_params = (self.overlap_size, self.window_size + self.overlap_size - size_last_window)

    def __post_init__(self):
        self.logger.debug(
            f"Time Series Dataset (Torch) for Imputation with {self.window_count} windows and {int(100 * self.overlap)}% overlap loaded successfully."
        )

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        for k, v in item.items():
            if isinstance(v, Tensor):
                item[k] = v.unsqueeze(0)

        # apply key mapping function if provided
        if self.key_mapping_fct is not None:
            k = list(item.keys())
            item = self.key_mapping_fct(item)
        return item

    def _sample_iwindow_size(self, max_sequence_length):
        min_iwindow_percentage = self.min_iwindow_percentage
        max_iwindow_percentage = self.max_iwindow_percentage
        iwindow_size = torch.randint(
            int(min_iwindow_percentage * max_sequence_length), int(max_iwindow_percentage * max_sequence_length), (1,)
        ).item()
        return iwindow_size

    @classmethod
    def sample_imputation_mask(cls, window_count):
        iwindow_index = torch.randint(1, window_count - 1, (1,)).item()
        mask = torch.zeros(window_count, dtype=torch.bool)
        mask[iwindow_index] = True
        assert mask.sum() == 1, f"Number of masked windows {mask.sum()} does not match expected {1}."
        return mask

    @staticmethod
    def collate_fn(batch, dataset):
        mask = dataset.imputation_mask
        max_sequence_length = batch[0]["coarse_grid_grid"].size(1)
        iwindow_size = dataset._sample_iwindow_size(max_sequence_length)
        output = []
        for item in batch:
            if dataset.imputation_mask is None:
                mask = dataset.sample_imputation_mask(dataset.window_count)

            iwindow_index = mask.nonzero().item()
            output.append(dataset._get_windowed_item(item, iwindow_size, iwindow_index, mask))

        return default_collate(output)

    def _get_windowed_item(self, item: dict, iwindow_size: int, iwindow_index: int, mask: Tensor) -> dict:
        max_sequence_length = item["coarse_grid_grid"].size(1)
        observation_values = split_into_variable_windows(
            item["coarse_grid_noisy_sample_paths"], iwindow_size, iwindow_index, self.window_count, max_sequence_length=max_sequence_length
        )

        observation_times = split_into_variable_windows(
            item["coarse_grid_grid"], iwindow_size, iwindow_index, self.window_count, max_sequence_length=max_sequence_length
        )
        observation_mask = split_into_variable_windows(
            item["coarse_grid_observation_mask"].bool(),
            iwindow_size,
            iwindow_index,
            self.window_count,
            max_sequence_length=max_sequence_length,
        )
        # imputation window data
        location_times = split_into_variable_windows(
            item["fine_grid_grid"],
            iwindow_size,
            iwindow_index,
            self.window_count,
            max_sequence_length=max_sequence_length,
            padding_value=None,
        )
        target_drift = item.get("fine_grid_concept_values", None)
        if target_drift is not None:
            target_drift = split_into_variable_windows(
                target_drift, iwindow_size, iwindow_index, self.window_count, max_sequence_length=max_sequence_length, padding_value=None
            )
        target_sample_path = split_into_variable_windows(
            item["fine_grid_sample_paths"],
            iwindow_size,
            iwindow_index,
            self.window_count,
            max_sequence_length=max_sequence_length,
            padding_value=None,
        )
        assert (observation_mask.size(0) == self.window_count) and (observation_mask.size(2) == 1)
        assert (
            observation_mask.shape[:2]
            == observation_values.shape[:2]
            == observation_times.shape[:2]
            == target_sample_path.shape[:2]
            == location_times.shape[:2]
        )

        # select masked out window as imputation window and observed windows as observation windows
        # take all but masked out window
        observation_values = observation_values[~mask]
        observation_times = observation_times[~mask]
        observation_mask = observation_mask[~mask]

        # take only masked out window
        location_times = location_times[mask].squeeze(0)
        if target_drift is not None:
            target_drift = target_drift[mask].squeeze(0)
        linitial_conditions = observation_values[iwindow_index - 1, ~observation_mask[iwindow_index - 1].bool()][-1:]
        rinitial_conditions = observation_values[iwindow_index, ~observation_mask[iwindow_index].bool()][:1]
        target_sample_path = target_sample_path[mask].squeeze(0)

        return {
            "observation_values": observation_values,
            "observation_times": observation_times,
            "observation_mask": observation_mask.bool(),
            "location_times": location_times[:iwindow_size],
            "target_drift": target_drift[:iwindow_size],
            "target_sample_path": target_sample_path[:iwindow_size],
            "linitial_conditions": linitial_conditions,
            "rinitial_conditions": rinitial_conditions,
            "imputation_window_index": iwindow_index,
        }

    def __str__(self):
        return f"TimeSeriesImputationDatasetTorch(path={self.path}, name={self.name}, split={self.split},  window_count={self.window_count}, overlap={self.overlap}, dataset_keys={self.output_fields})"


# FIMSDE ---------------------------------------------------------


@dataclass
class FIMSDEDatabatch:
    obs_times: Tensor | np.ndarray
    obs_values: Tensor | np.ndarray
    obs_noisy_values: Tensor | np.ndarray

    drift_at_locations: Tensor | np.ndarray
    diffusion_at_locations: Tensor | np.ndarray
    locations: Tensor | np.ndarray

    drift_at_obs_values: Tensor | np.ndarray
    diffusion_at_obs_values: Tensor | np.ndarray

    obs_mask: Tensor | np.ndarray = None

    diffusion_parameters: Tensor | np.ndarray = None
    drift_parameters: Tensor | np.ndarray = None
    process_label: Tensor | np.ndarray = None
    process_dimension: Tensor | np.ndarray = None

    def convert_to_tensors(self):
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, np.ndarray):
                try:
                    setattr(self, field, Tensor(value))
                except Exception:
                    print(f"Problem for field {field}")
                    setattr(self, field, None)


# Define the named tuple
FIMSDEDatabatchTuple = namedtuple(
    "FIMSDEDatabatchTuple",
    [
        "obs_times",
        "obs_values",
        "obs_noisy_values",
        "obs_mask",
        "diffusion_at_locations",
        "drift_at_locations",
        "locations",
        "dimension_mask",
    ],
)


class FIMSDEDataset(torch.utils.data.Dataset):
    """
    First simple dataset to train a Neural Operator
    """

    def __init__(self, data_config: FIMDatasetConfig = None, data_paths: Optional[List[str]] = None):
        """
        Args:
            data_paths (List[str]): list of locations of .h5 files required to load the data
        """
        # To keep track of the number of samples in each file
        self.data = []
        self.lengths = []
        self.data_config = data_config

        # Load data and compute cumulative lengths
        self.read_files(data_config, data_paths)

    @classmethod
    def from_data_paths(cls, data_paths: list[str], data_in_files: Optional[dict] = {}):
        data_config = FIMDatasetConfig(data_in_files=data_in_files)
        return cls(data_config, data_paths)

    @classmethod
    def from_data_batches(cls, data_batch: list[FIMSDEDatabatch]):
        print("------------------------------------------------------------------------------")
        print("There seems to be a bug somewhere. Paths, drifts and diffusions of one element don't 'belong' together.")
        print("------------------------------------------------------------------------------")
        return cls(None, data_batch)

    def _prepare_file_path(self, file_path: Path | str | FIMSDEDatabatch) -> Path | str | FIMSDEDatabatch:
        """
        Translate data paths format to os. Prepend fim data_dir path if passed paths are not absolute.
        """
        if isinstance(file_path, str):
            # translate path format
            if os.name == "posix":
                file_path = file_path.replace("\\", "/")

            elif os.name == "nt":
                file_path = file_path.replace("/", "\\")

            # # prepend fim data dir
            # if not os.path.isabs(file_path):
            #     file_path = os.path.join(data_path, file_path)

        return file_path

    def _read_one_bulk(self, data: str | FIMSDEDatabatch | Path) -> FIMSDEDatabatch:
        from pathlib import Path

        data_dict = {}
        if isinstance(data, FIMSDEDatabatch):
            return data
        elif isinstance(data, (str, Path)):
            data = Path(data)
            for key, value in self.data_config.data_in_files.__dict__.items():
                key_file_path = data / value
                data_dict[key] = load_h5(key_file_path)

            data_bulk = FIMSDEDatabatch(**data_dict)
            data_bulk.convert_to_tensors()
            return data_bulk

    def read_files(self, params, file_paths: List[str]):
        """
        Reads the files and organize data such that during item selection
        the dataset points to the file and then to the location within that file
        of the particular datapoint
        """
        if params is not None:
            self.max_num_paths = params.max_num_paths
            self.max_time_steps = params.max_time_steps
            self.max_dimension = params.max_dimension
            self.max_location_size = params.max_location_size
        else:
            self.max_num_paths = 0
            self.max_time_steps = 0
            self.max_dimension = 0
            self.max_location_size = 0
            self.max_drift_param_size = 0
            self.max_diffusion_param_size = 0

        for file_path in file_paths:
            file_path = self._prepare_file_path(file_path)
            if isinstance(file_path, (Path, str)):
                one_data_bulk: FIMSDEDatabatch = self._read_one_bulk(file_path)  # Adjust loading method as necessary
            elif isinstance(file_path, FIMSDEDatabatch):
                one_data_bulk = file_path

            self.data.append(one_data_bulk)
            self.lengths.append(one_data_bulk.obs_values.size(0))  # Number of samples in this file

            # Update max dimensions
            self.max_num_paths = max(self.max_dimension, one_data_bulk.obs_values.size(1))
            self.max_time_steps = max(self.max_time_steps, one_data_bulk.obs_values.size(2))
            self.max_dimension = max(self.max_dimension, one_data_bulk.obs_values.size(3))

            self.max_location_size = max(self.max_location_size, one_data_bulk.locations.size(1))

            if one_data_bulk.drift_parameters is not None:
                self.max_drift_param_size = max(self.max_drift_param_size, one_data_bulk.drift_parameters.size(1))
            if one_data_bulk.diffusion_parameters is not None:
                self.max_diffusion_param_size = max(self.max_diffusion_param_size, one_data_bulk.diffusion_parameters.size(1))

        if self.data_config is not None:
            self.data_config.max_dimension = self.max_dimension
            self.data_config.max_time_steps = self.max_time_steps
            self.data_config.max_location_size = self.max_location_size
            self.data_config.max_num_paths = self.max_num_paths

        self.cumulative_lengths = np.cumsum(self.lengths)

    def __len__(self):
        return sum(self.lengths)  # Total number of samples

    def __getitem__(self, idx) -> FIMSDEDatabatchTuple:
        # Obtains index of the associated file and item within the file
        file_idx, sample_idx = self._get_file_and_sample_index(idx)
        data_bulk: FIMSDEDatabatch = self.data[file_idx]
        # Get the tensor from the appropriate file
        obs_times = data_bulk.obs_times[sample_idx]
        obs_values = data_bulk.obs_values[sample_idx]
        if hasattr(data_bulk, "obs_mask"):
            obs_mask = data_bulk.obs_mask[sample_idx]
        else:
            None
        diffusion_at_locations = data_bulk.diffusion_at_locations[sample_idx]
        drift_at_locations = data_bulk.drift_at_locations[sample_idx]
        locations = data_bulk.locations[sample_idx]
        # Pad and Obtain Mask of The tensors if necessary
        obs_times, obs_values, obs_mask = self._pad_obs_tensors(obs_times, obs_values, obs_mask)
        drift_at_locations, diffusion_at_locations, locations, mask = self._pad_locations_tensors(
            drift_at_locations, diffusion_at_locations, locations
        )
        if len(obs_values.shape) == 4:
            obs_values = obs_values[:, :, :, 0]

        # select a smaller set of paths
        # obs_values,obs_times,diffusion_at_locations,drift_at_locations,locations,mask = self._select_paths_and_grid(obs_values,obs_times,diffusion_at_locations,drift_at_locations,locations,mask)

        # Create and return the named tuple
        return FIMSDEDatabatchTuple(
            obs_times=obs_times,
            obs_values=obs_values,
            obs_mask=obs_mask,
            drift_at_locations=drift_at_locations,
            diffusion_at_locations=diffusion_at_locations,
            locations=locations,
            dimension_mask=mask,
        )

    def _select_paths_and_grid(
        self,
        obs_values: Tensor,
        obs_times: Tensor,
        drift_at_locations: Tensor,
        diffusion_at_locations: Tensor,
        locations: Tensor,
        dimension_mask: Tensor,
    ):
        P = obs_values.size(0)
        G = locations.size(0)

        number_of_paths = torch.randint(self.min_number_of_paths_per_batch, min(self.max_number_of_paths_per_batch, P), size=(1,))[0]
        number_of_grids = torch.randint(self.min_number_of_grid_per_batch, min(G, self.max_number_of_grid_per_batch), size=(1,))[0]

        obs_values = obs_values[:number_of_paths]
        obs_times = obs_times[:number_of_paths]

        drift_at_locations = drift_at_locations[:number_of_grids]
        diffusion_at_locations = diffusion_at_locations[:number_of_grids]
        locations = locations[:number_of_grids]
        dimension_mask = dimension_mask[:number_of_grids]

        return (obs_values, obs_times, drift_at_locations, diffusion_at_locations, locations, dimension_mask)

    def _get_file_and_sample_index(self, idx):
        """Helper function to determine the file index and sample index."""
        file_idx = np.searchsorted(self.cumulative_lengths, idx, "right")
        sample_idx = idx if file_idx == 0 else idx - self.cumulative_lengths[file_idx - 1]
        return file_idx, sample_idx

    def _pad_obs_tensors(self, obs_times, obs_values, obs_mask):
        """ """
        current_dimension = obs_values.size(2)
        current_time_steps = obs_values.size(1)

        dim_padding_size = self.max_dimension - current_dimension
        time_dim_padding_size = self.max_time_steps - current_time_steps

        if dim_padding_size > 0 or time_dim_padding_size > 0:
            if len(obs_values.shape) == 4:  # coming from h5 files
                obs_values = torch.nn.functional.pad(obs_values, (0, 0, 0, dim_padding_size, 0, time_dim_padding_size))
            elif len(obs_values.shape) == 3:  # coming from target data simulation
                obs_values = torch.nn.functional.pad(obs_values, (0, dim_padding_size, 0, time_dim_padding_size))

            obs_times = torch.nn.functional.pad(obs_times, (0, 0, 0, time_dim_padding_size))

            if obs_mask is not None:
                obs_mask = torch.nn.functional.pad(obs_mask, (0, 0, 0, time_dim_padding_size))

            else:
                obs_mask = torch.ones_like(obs_times).bool()

        return obs_times, obs_values, obs_mask

    def _pad_locations_tensors(self, drift_at_locations, diffusion_at_locations, locations):
        """ """
        current_dimension = drift_at_locations.size(1)
        current_location = drift_at_locations.size(0)
        location_padding_size = self.max_location_size - current_location
        dim_padding_size = self.max_dimension - current_dimension

        if dim_padding_size > 0 or location_padding_size > 0:
            diffusion_at_locations = torch.nn.functional.pad(diffusion_at_locations, (0, dim_padding_size, 0, location_padding_size))
            drift_at_locations = torch.nn.functional.pad(drift_at_locations, (0, dim_padding_size, 0, location_padding_size))
            locations = torch.nn.functional.pad(locations, (0, dim_padding_size, 0, location_padding_size))

            mask = self._create_mask(drift_at_locations, current_location, current_dimension)
        else:
            mask = torch.ones_like(drift_at_locations)

        return drift_at_locations, diffusion_at_locations, locations, mask

    def _create_mask(self, drift_at_locations, current_location, current_dimension):
        """Create a mask for the observations.
        Args:
            drift_at_hypercube (Tensor) [B,H,D], current_hyper  (int), current_dimension (int)
        Returns:
            mask [B,H,D] will do 0 for hypercube positions and dimensions not on batch
        """
        mask = torch.ones_like(drift_at_locations)
        mask[:, current_dimension:] = 0.0
        mask[current_location:, :] = 0.0
        return mask

    def update_parameters(self, param):
        param.max_dimension = self.max_dimension
        param.max_hypercube_size = self.max_location_size
        param.max_num_steps = self.max_num_steps


class JsonSDEDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        batch_size: int,
        json_paths: Path | Paths | dict,
        keys_to_load: dict,
        paths_per_batch_element: int | None = None,
        **kwargs,
    ):
        self.batch_size = batch_size
        self.num_batches = None  # determined by self.__len__()

        self.keys_to_load = keys_to_load
        self.paths_per_batch_element = paths_per_batch_element

        self.data = self.__load_data(json_paths, keys_to_load, paths_per_batch_element)

    def __load_data(self, data: Path | Paths | dict | list[dict], keys_to_load: dict, paths_per_batch_element: int | None) -> dict:
        """
        Load dicts from jsons (all containing keys_to_load as keys), concatenate their Tensor values along first dimension.

        Args:
            data (Path | Paths | dict | list[dict]): (List of) Path to jsons containing dicts with Tensor values or dicts.
            keys_to_load (dict): Keys to load from dicts in jsons. Rename loaded keys to the corresponding values in keys_to_load.
            paths_per_batch_element (int): Group loaded paths into batch eleemnts with this many paths for easier FIM processing.
        """
        if not isinstance(data, list):
            data = [data]

        # data: list[dict] = torch.utils._pytree.tree_map(
        #     partial(self._load_dict_from_json, keys_to_load=keys_to_load, paths_per_batch_element=paths_per_batch_element), data
        # )
        data: list[dict] = [self._load_dict_from_json(d, keys_to_load, paths_per_batch_element) for d in data]

        if len(data) > 1:
            data: dict = torch.utils._pytree.tree_map(lambda *x: torch.concatenate(x, dim=0), *data)  # tuple of length 1

        return data[0]

    @staticmethod
    def _load_dict_from_json(data, keys_to_load: dict, paths_per_batch_element: int | None = None) -> dict:
        """
        Load dict with Tensor values from json.
        Extract relevant keys and (optionally) add a specific path dimension for FIM processing.
        """
        if isinstance(data, Path):
            data: Path = pathlib.Path(data)  # to be sure
            if not data.is_absolute():
                data = pathlib.Path(data_path) / data

            assert data.exists(), f"{data} does not exist."

            data: dict = json.load(open(data, "r"))

        elif isinstance(data, dict):
            data: dict = data

        else:
            raise ValueError(f"Must pass Path or dict, got {type(data)}.")

        data: dict = {key: Tensor(value) if isinstance(value, list) else value for key, value in data.items()}

        assert all(json_key in data.keys() for json_key in keys_to_load.values()), (
            f"Dict has keys {data.keys()} and does not contain all keys to load {keys_to_load}."
        )
        data: dict = {renamed_key: data[json_key] for renamed_key, json_key in keys_to_load.items()}

        assert all(isinstance(value, Tensor) for value in data.values()), (
            f"Can only load Tensors, got types {(type(value) for value in data.values())} in json {data}."
        )

        if paths_per_batch_element is not None:
            data_size = list(data.values())[0].shape[-3]
            num_batches = data_size // paths_per_batch_element

            data: dict = {key: value[: num_batches * paths_per_batch_element] for key, value in data.items()}
            data: dict = {key: value.reshape(num_batches, paths_per_batch_element, *value.shape[1:]) for key, value in data.items()}

        return data

    def __len__(self):
        """
        Length is number of batches.
        """
        if self.num_batches is None:
            B = list(self.data.values())[0].shape[0]
            self.num_batches = B // self.batch_size

        return self.num_batches

    def __iter__(self):
        """
        Return iterator yielding batches of self.batch_size.
        """
        return tensor_dict_iterator(self.data, self.batch_size, process_data=None, process_batch=None)


class PaddedFIMSDEDataset(torch.utils.data.IterableDataset):
    """
    Load data from separate files, pad them along observed dimension, sequence length and loations, and concatenate the resulting tensors.
    If num_workers > 1, split files among workers.

    Best used if whole data fits into memory. Loading is very fast, even with 1 worker.
    """

    def __init__(
        self,
        batch_size: int,
        data: Optional[dict] = None,
        data_dirs: Optional[Path | Paths] = None,
        files_to_load: Optional[dict] = None,
        data_limit: Optional[int] = None,
        max_dim: Optional[int] = None,
        dim_mask_key: Optional[str] = "dimension_mask",
        add_paths_keys: Optional[list[str]] = [],
        add_loc_keys: Optional[list[str]] = [],
        add_dim_keys: Optional[list[str]] = [],
        shuffle_locations: Optional[bool] = True,
        shuffle_paths: Optional[bool] = True,
        shuffle_elements: Optional[bool] = True,
        load_data_at_init: Optional[bool] = True,  # use if persistent_workers = False
        num_locations: Optional[int] = None,
        num_observations: Optional[tuple] = None,
    ):
        # group values by keys, so they can be easily selected
        self.paths_keys = ["obs_values", "obs_times", "obs_mask"] + list(add_paths_keys)
        self.loc_keys = ["locations", "drift_at_locations", "diffusion_at_locations", dim_mask_key] + list(add_loc_keys)
        self.dim_keys = ["obs_values", "locations", "drift_at_locations", "diffusion_at_locations", dim_mask_key] + list(add_dim_keys)
        self.dim_mask_key = dim_mask_key

        # shuffle data per iter setup
        self.shuffle_locations = shuffle_locations
        self.shuffle_paths = shuffle_paths
        self.shuffle_elements = shuffle_elements

        if data is not None:
            # loaded data is passed directly as dict
            self.data = data
            self.load_data_at_init = False

        else:
            # paths to data and loading / padding config
            self.data_dirs = data_dirs  # one or multiple paths to directories containing files_to_load
            self.files_to_load = files_to_load  # maps keys to filenames (in path)
            self.data_limit = data_limit  # number of equations (first dim)  to load from each file
            self.max_dim = max_dim

            # prepare all paths to load under each
            self.all_file_paths: dict[str, list[Path]] = get_file_paths(self.data_dirs, self.files_to_load)

            self.load_data_at_init = load_data_at_init
            if self.load_data_at_init is True:
                self.data: dict[str, Tensor] = self.__load_data(self.all_file_paths)

            else:
                self.data = None

        self.batch_size: int = batch_size
        self.num_batches: int = None

        self.num_locations: int = num_locations
        self.num_observations: tuple = num_observations

    def __load_data(self, file_paths: dict[str, list[Path]]) -> dict[str, Tensor]:
        """
        Load data from paths, pad and concatenate them per key in self.files_to_load.

        Returns:
            data (dict): Keys are keys of `files_to_load`. Values are Tensors associated to the key.
        """
        # load data from paths
        data: dict[str, list[Tensor]] = torch.utils._pytree.tree_map(load_file, file_paths)

        # minimally required data
        assert "obs_times" in data.keys()
        assert "obs_values" in data.keys()

        if data.get("obs_mask") is None:
            data["obs_mask"] = torch.utils._pytree.tree_map(torch.ones_like, data["obs_times"])

        # padding observations (shape [B, paths, seq_len, X]) to largest sequence length
        data = pad_data_in_dict(data, self.paths_keys, dim=-2, mode="constant")

        if "locations" in data.keys():
            # padding location data (shape [B, G, X] to largest sequence length (using mode "circular", so without mask for now)
            data = pad_data_in_dict(data, self.loc_keys, dim=-2, mode="circular")

            # to pad dimension of observation space to max_dim
            if self.dim_mask_key not in data.keys():
                data[self.dim_mask_key] = torch.utils._pytree.tree_map(torch.ones_like, data["locations"])

            data[self.dim_mask_key] = torch.utils._pytree.tree_map(lambda x: x.bool(), data[self.dim_mask_key])

        # padding dimension of observation space to max_dim
        data = pad_data_in_dict(data, self.dim_keys, dim=-1, mode="constant", max_length=self.max_dim)

        # set masks dtype to bool
        data["obs_mask"] = torch.utils._pytree.tree_map(lambda x: x.bool(), data["obs_mask"])

        # concatenate data per key
        data: dict[str, Tensor] = {k: torch.concatenate(v, dim=0) for k, v in data.items()}

        # return only part of data
        if self.data_limit is not None:
            data = torch.utils._pytree.tree_map(lambda x: x[: self.data_limit], data)

        return data

    def __process_data(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Data processing at beginning of each iterator.
        """
        # shuffle data
        data = shuffle_sde_data(
            data,
            paths_keys=self.paths_keys,
            loc_keys=self.loc_keys,
            shuffle_paths=self.shuffle_paths,
            shuffle_locations=self.shuffle_locations,
            shuffle_elements=self.shuffle_elements,
        )

        # truncate locations, can be done once per iterator as is a fixed number and has been shuffled before
        if self.num_locations is not None:
            data = truncate_locations(data, self.loc_keys, self.num_locations)

        return data

    def __process_batch(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Data processing after each batch has been created.
        """
        # truncate number of observations
        if self.num_observations is not None:
            min_num_obs, max_num_obs = self.num_observations

            obs_vals = data["obs_values"]
            B, P, T, D = obs_vals.shape

            _min_obs = max(min_num_obs, T)
            _max_obs = min(max_num_obs, P * T)

            if _min_obs >= _max_obs:
                num_obs = P * T
            else:
                num_obs = torch.randint(_min_obs, _max_obs, size=(1,)).item()  # want at least one path
            num_paths = num_obs // T

            data = truncate_paths(data, self.paths_keys, num_paths)

        return data

    def __len__(self):
        """
        Length is number of batches.
        If not already computed, open all files related to "obs_values", sum their sizes and divide by batch_size.
        """
        if self.num_batches is None:
            self.num_batches = get_iterable_dataset_length(self.all_file_paths) // self.batch_size + 1  # account for remaining batch

        return self.num_batches

    def __iter__(self):
        """
        Return iterator through data per worker. Optionally load data per worker first.
        """
        worker_info = torch.utils.data.get_worker_info()

        if self.load_data_at_init is True:
            # assign chunks of data to each worker
            if worker_info is not None:  # in a worker process, split workload
                total_num_elements = self.data["obs_values"].size(0)

                # distribute number of elements evenly over workers
                num_workers: int = worker_info.num_workers
                min_num_elements_per_worker: int = total_num_elements // num_workers
                num_elements_per_worker: list[int] = [min_num_elements_per_worker] * num_workers

                # add remaining elements evenly to some workers
                num_elements_remaining: int = total_num_elements % num_workers
                if num_elements_remaining != 0:
                    for i in range(num_elements_remaining):
                        num_elements_per_worker[i] = num_elements_per_worker[i] + 1

                # calculate start and end index of current worker
                worker_id = worker_info.id

                start_index = int(reduce(operator.add, [0] + num_elements_per_worker[:worker_id], 0))
                end_index = int(reduce(operator.add, num_elements_per_worker[: worker_id + 1], 0))

                # Truncate self.data for current worker (to save some space)
                self.data = torch.utils._pytree.tree_map(lambda x: x[start_index:end_index], self.data)

        else:
            # load some files per worker
            if self.data is not None:
                # data already loaded for this worker
                pass

            elif worker_info is not None:
                num_workers: int = worker_info.num_workers
                files_per_worker: list[dict[str, list[Path]]] = distribute_file_paths_among_workers(self.all_file_paths, num_workers)

                # select files for current worker
                worker_id = worker_info.id
                worker_files_to_load = files_per_worker[worker_id]

                # load data for current worker
                total_num_files: int = len(self.all_file_paths["obs_values"])
                if worker_id >= total_num_files:  # worker has not associated files
                    self.data = None

                else:
                    self.data = self.__load_data(worker_files_to_load)

            else:  # no worker
                self.data = self.__load_data(self.all_file_paths)

        return tensor_dict_iterator(self.data, self.batch_size, process_data=self.__process_data, process_batch=self.__process_batch)


class HeterogeneousFIMSDEDataset(torch.utils.data.IterableDataset):
    """
    To load files with different number of paths, lengths of paths and number of locations.
    Load data from separate files, pad them along observed dimension.
    Iterate over data per files. Each worker chains its respective iterators.

    Best used if whole data fits into memory, but number of paths, sequence length, locations are different for each set of files
    """

    def __init__(
        self,
        data_dirs: Path | Paths,
        batch_size: int,
        files_to_load: Optional[dict] = None,
        data_limit: Optional[int] = None,
        max_dim: Optional[int] = None,
        dim_mask_key: Optional[str] = "dimension_mask",
        add_paths_keys: Optional[list[str]] = [],
        add_loc_keys: Optional[list[str]] = [],
        add_dim_keys: Optional[list[str]] = [],
        shuffle_locations: Optional[bool] = True,
        shuffle_paths: Optional[bool] = True,
        shuffle_elements: Optional[bool] = True,
        num_locations: Optional[int] = None,
        num_observations: Optional[tuple] = None,
        **kwargs,
    ):
        # group values by keys, so they can be easily selected
        self.paths_keys = ["obs_values", "obs_times", "obs_mask"] + list(add_paths_keys)
        self.loc_keys = ["locations", "drift_at_locations", "diffusion_at_locations", dim_mask_key] + list(add_loc_keys)
        self.dim_keys = ["obs_values", "locations", "drift_at_locations", "diffusion_at_locations", dim_mask_key] + list(add_dim_keys)
        self.dim_mask_key = dim_mask_key

        # paths to data and loading / padding config
        self.data_dirs = data_dirs  # one or multiple paths to directories containing files_to_load
        self.files_to_load = files_to_load  # maps keys to filenames (in path)
        self.data_limit = data_limit  # number of equations (first dim)  to load from each file
        self.max_dim = max_dim

        # shuffle data per iter setup
        self.shuffle_locations = shuffle_locations
        self.shuffle_paths = shuffle_paths
        self.shuffle_elements = shuffle_elements

        # prepare all paths to load under each
        self.all_file_paths: dict[str, list[Path]] = get_file_paths(self.data_dirs, self.files_to_load)

        self.data = None
        self.batch_size: int = batch_size
        self.num_batches: int = None

        self.num_locations: int = num_locations
        self.num_observations: tuple = num_observations

    def __load_data(self, file_paths: dict[str, Path]) -> dict[str, Tensor]:
        """
        Load data from path and pad them to max_dim per key in self.files_to_load.

        Returns:
            data (dict): Keys are keys of `files_to_load`. Values are Tensors associated to the key.
        """
        # load data from paths
        data: dict[str, Tensor] = torch.utils._pytree.tree_map(load_file, file_paths)

        # minimally required data
        assert "obs_times" in data.keys()
        assert "obs_values" in data.keys()

        if data.get("obs_mask") is None:
            data["obs_mask"] = torch.utils._pytree.tree_map(torch.ones_like, data["obs_times"])

        if "locations" in data.keys():
            # masks for padding dimension of observation space to max_dim
            if self.dim_mask_key not in data.keys():
                data[self.dim_mask_key] = torch.utils._pytree.tree_map(torch.ones_like, data["locations"])

            data[self.dim_mask_key] = torch.utils._pytree.tree_map(lambda x: x.bool(), data[self.dim_mask_key])

        # padding observed dimension to max_dim
        data: dict[str, Tensor] = pad_data_in_dict(data, self.dim_keys, dim=-1, mode="constant", max_length=self.max_dim)

        # set masks dtype to bool
        data["obs_mask"] = torch.utils._pytree.tree_map(lambda x: x.bool(), data["obs_mask"])

        # return only part of data
        if self.data_limit is not None:
            data = torch.utils._pytree.tree_map(lambda x: x[: self.data_limit], data)

        return data

    def __process_data(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Data processing at beginning of each iterator.
        """
        # shuffle data
        data = shuffle_sde_data(
            data,
            paths_keys=self.paths_keys,
            loc_keys=self.loc_keys,
            shuffle_paths=self.shuffle_paths,
            shuffle_locations=self.shuffle_locations,
            shuffle_elements=self.shuffle_elements,
        )

        # truncate locations, can be done once per iterator as is a fixed number and has been shuffled before
        if self.num_locations is not None:
            data = truncate_locations(data, self.loc_keys, self.num_locations)

        return data

    def __process_batch(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Data processing after each batch has been created.
        """
        # truncate number of observations
        if self.num_observations is not None:
            min_num_obs, max_num_obs = self.num_observations

            obs_vals = data["obs_values"]
            B, P, T, D = obs_vals.shape

            num_obs = torch.randint(max(min_num_obs, T), min(max_num_obs, P * T), size=(1,)).item()  # want at least one path
            num_paths = num_obs // T

            data = truncate_paths(data, self.paths_keys, num_paths)

        return data

    def __len__(self):
        """
        Length is number of batches.
        If not already computed, open all files related to "obs_values", sum their sizes and divide by batch_size.
        """
        if self.num_batches is None:
            self.num_batches = get_iterable_dataset_length(self.all_file_paths) // self.batch_size + 32  # account for max workers

        return self.num_batches

    def __iter__(self):
        """
        Return iterator through data per worker. Optionally load data per worker first.
        """
        worker_info = torch.utils.data.get_worker_info()

        if self.data is not None:
            # data (per worker) is already loaded
            pass
        else:
            if worker_info is not None:  # in a worker process, split workload
                num_workers: int = worker_info.num_workers
                files_per_worker: list[dict[str, list[Path]]] = distribute_file_paths_among_workers(self.all_file_paths, num_workers)

                # select files for current worker
                worker_id = worker_info.id
                worker_files_to_load = files_per_worker[worker_id]

                # load data (per file!) for current worker
                total_num_files: int = len(self.all_file_paths["obs_values"])
                if worker_id >= total_num_files:  # worker has not associated files
                    self.data = None

                else:
                    # each set of files loaded and padded (only) to max_dim
                    self.data: dict[str, Tensor] = self.__load_data(worker_files_to_load)

            else:  # no worker
                self.data: dict[str, Tensor] = self.__load_data(self.all_file_paths)

        if self.data is None:
            return iter([])

        else:
            # data iter per files of worker
            num_files: int = len(self.data["obs_values"])

            data_iters = []
            for i in range(num_files):
                data_for_iter = {k: v[i] for k, v in self.data.items()}
                data_iter = tensor_dict_iterator(
                    data_for_iter, self.batch_size, process_data=self.__process_data, process_batch=self.__process_batch
                )
                data_iters.append(data_iter)

            # randomize order
            random.shuffle(data_iters)

            # chain data iters of worker
            combined_data_iter = itertools.chain(*data_iters)

            return combined_data_iter


class StreamingFIMSDEDataset(torch.utils.data.IterableDataset):
    """
    Lazy load data from separate files. Each worker is responsible for different files.
    Each iterator opens one set of data streams, extracts batches, pads along observed dimension and yields them.
    Each worker chains multiple iterators together, to cover complete data.

    Currently must be .h5 files!

    Best used if whole data does not fit into memory.
    """

    def __init__(
        self,
        data_dirs: Path | Paths,
        batch_size: int,
        files_to_load: Optional[dict] = None,
        data_limit: Optional[int] = None,
        max_dim: Optional[int] = None,
        dim_mask_key: Optional[str] = "dimension_mask",
        add_paths_keys: Optional[list[str]] = [],
        add_loc_keys: Optional[list[str]] = [],
        add_dim_keys: Optional[list[str]] = [],
        shuffle_locations: Optional[bool] = True,
        shuffle_paths: Optional[bool] = True,
        num_locations: Optional[int] = None,
        num_observations: Optional[tuple] = None,
        **kwargs,
    ):
        # group values by keys, so they can be easily selected
        self.paths_keys = ["obs_values", "obs_times", "obs_mask"] + list(add_paths_keys)
        self.loc_keys = ["locations", "drift_at_locations", "diffusion_at_locations", dim_mask_key] + list(add_loc_keys)
        self.dim_keys = ["obs_values", "locations", "drift_at_locations", "diffusion_at_locations", dim_mask_key] + list(add_dim_keys)
        self.dim_mask_key = dim_mask_key

        # paths to data and loading / padding config
        self.data_dirs = data_dirs  # one or multiple paths to directories containing files_to_load
        self.data_limit = data_limit
        self.files_to_load = files_to_load  # maps keys to filenames (in path)
        self.max_dim = max_dim

        # shuffle data per iter setup
        self.shuffle_locations = shuffle_locations
        self.shuffle_paths = shuffle_paths

        # prepare all paths to load under each
        self.all_file_paths: dict[str, list[Path]] = get_file_paths(self.data_dirs, self.files_to_load)
        self.batch_size: int = batch_size

        self.num_batches: int = None

        self.num_locations: int = num_locations
        self.num_observations: tuple = num_observations

    def __process_batch(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Preprocess data batch by adding masks and pad to max_dim per key.

        Returns:
            data (dict): Keys are keys of `files_to_load`. Values are Tensors associated to the key.
        """
        # minimally required data
        assert "obs_times" in data.keys()
        assert "obs_values" in data.keys()

        if data.get("obs_mask") is None:
            data["obs_mask"] = torch.utils._pytree.tree_map(torch.ones_like, data["obs_times"])

        if "locations" in data.keys():
            # masks for padding dimension of observation space to max_dim
            if self.dim_mask_key not in data.keys():
                data[self.dim_mask_key] = torch.utils._pytree.tree_map(torch.ones_like, data["locations"])

            data[self.dim_mask_key] = torch.utils._pytree.tree_map(lambda x: x.bool(), data[self.dim_mask_key])

        # padding observed dimension to max_dim
        data: dict[str, Tensor] = pad_data_in_dict(data, self.dim_keys, dim=-1, mode="constant", max_length=self.max_dim)

        # set masks dtype to bool
        data["obs_mask"] = torch.utils._pytree.tree_map(lambda x: x.bool(), data["obs_mask"])

        # shuffle
        data = shuffle_sde_data(data, self.paths_keys, self.loc_keys, self.shuffle_paths, self.shuffle_locations, shuffle_elements=False)

        #  truncate locations
        if self.num_locations is not None:
            data = truncate_locations(data, self.loc_keys, self.num_locations)

        # truncate paths
        if self.num_observations is not None:
            min_num_obs, max_num_obs = self.num_observations

            obs_vals = data["obs_values"]
            B, P, T, D = obs_vals.shape

            num_obs = torch.randint(max(min_num_obs, T), min(max_num_obs, P * T), size=(1,)).item()  # want at least one path
            num_paths = num_obs // T

            data = truncate_paths(data, self.paths_keys, num_paths)

        return data

    def __len__(self):
        """
        Length is number of batches.
        If not already computed, open all files related to "obs_values", sum their sizes and divide by batch_size.
        """
        if self.num_batches is None:
            self.num_batches = get_iterable_dataset_length(self.all_file_paths) // self.batch_size + 128  # account for max workers

        return self.num_batches

    def __iter__(self):
        """
        Return iterator through data per worker. Optionally load data per worker first.
        """
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            worker_files = self.all_file_paths

        else:  # in a worker process, split workload
            num_workers: int = worker_info.num_workers
            files_per_worker: list[dict[str, list[Path]]] = distribute_file_paths_among_workers(self.all_file_paths, num_workers)

            # select files for current worker
            worker_id = worker_info.id
            worker_files = files_per_worker[worker_id]

        # data iter per files of worker
        num_files: int = len(worker_files["obs_values"])

        if num_files == 0:
            return iter([])

        else:
            iterators = []
            for i in range(num_files):
                files_for_iter = {k: v[i] for k, v in worker_files.items()}
                iterator = h5_files_dict_iterator(
                    files_for_iter,
                    self.batch_size,
                    process_batch=self.__process_batch,
                )
                if self.data_limit is not None:
                    iterator = itertools.islice(iterator, self.data_limit)

                iterators.append(iterator)

            # randomize order
            random.shuffle(iterators)

            # chain data iters of worker
            combined_iterator = itertools.chain(*iterators)

            return combined_iterator


class StreamingHawkesDataset(torch.utils.data.IterableDataset):
    """
    Stream Hawkes process datasets from directories of .pt files without loading everything into memory.

    Expects per-directory files specified via `files_to_load`, e.g.:
      {
        "base_intensity_functions": "base_intensity_functions.pt",
        "event_times": "event_times.pt",
        "event_types": "event_types.pt",
        "kernel_functions": "kernel_functions.pt",
        "time_offsets": "time_offsets.pt"  # optional
      }

    Yields single items (dicts) which are later collated by the HawkesDataLoader's custom collate.
    """

    def __init__(
        self,
        data_dirs: Path | Paths,
        batch_size: int,
        files_to_load: Optional[dict] = None,
        data_limit: Optional[int] = None,
        field_name_for_dimension_grouping: Optional[str | list[str]] = None,
        prefetch_rows: Optional[int] = None,
        shuffle: bool = False,
        **kwargs,
    ):
        # config
        self.data_dirs = data_dirs
        self.files_to_load = files_to_load or {}
        self.data_limit = data_limit
        self.batch_size = batch_size
        # prefetch multiple rows per IO to reduce h5 decompression overhead
        self.prefetch_rows = prefetch_rows if prefetch_rows is not None else max(1, 4 * int(batch_size))
        self.field_name_for_dimension_grouping = field_name_for_dimension_grouping

        # collect file paths per key
        self.all_file_paths: dict[str, list[Path]] = get_file_paths(self.data_dirs, self.files_to_load)

        # choose a reference key to determine per-directory sample counts
        # prefer event_times, otherwise fall back to any available key
        if "event_times" in self.all_file_paths:
            self._ref_key = "event_times"
        else:
            self._ref_key = next(iter(self.all_file_paths.keys()))

        # lazily computed
        self._num_dirs: int = len(self.all_file_paths[self._ref_key])
        self._num_items_per_dir: Optional[list[int]] = None
        self._num_batches: Optional[int] = None
        self._marks_per_dir: Optional[list[int]] = None
        self._different_marks_dim: bool = False

        # shuffling across epochs (like DistributedSampler)
        self._shuffle = shuffle
        self._epoch: int = 0
        self._seed: int = 0

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    # HawkesDataLoader relies on this attribute
    @property
    def is_last_dim_varying(self):
        # True if different number of marks across directories
        if self._marks_per_dir is None:
            self._compute_marks_per_dir()
        return self._different_marks_dim

    def __len__(self):
        # Return the total number of items. The DataLoader will compute the number of
        # batches based on its own batch_size. Returning batches here would cause an
        # unintended double division by batch_size.
        return self._compute_total_items()

    def _compute_total_items(self) -> int:
        if self._num_items_per_dir is None:
            self._num_items_per_dir = []
            for ref_path in self.all_file_paths[self._ref_key]:
                # Prefer .h5 if available (same basename)
                h5_path = ref_path.with_suffix(".h5")
                if h5_path.exists():
                    with h5py.File(h5_path, "r") as f:
                        num_items = int(f["data"].shape[0])
                else:
                    tensor = torch.load(ref_path, weights_only=True)
                    num_items = int(tensor.shape[0])
                if self.data_limit is not None:
                    num_items = min(num_items, int(self.data_limit))
                self._num_items_per_dir.append(num_items)
        return int(sum(self._num_items_per_dir))

    def _compute_marks_per_dir(self):
        marks_key = None
        if "kernel_functions" in self.all_file_paths:
            marks_key = "kernel_functions"
        elif "base_intensity_functions" in self.all_file_paths:
            marks_key = "base_intensity_functions"

        self._marks_per_dir = []
        if marks_key is None:
            # fallback: assume same marks everywhere
            self._marks_per_dir = [0] * self._num_dirs
            self._different_marks_dim = False
            return

        for p in self.all_file_paths[marks_key]:
            h5p = p.with_suffix(".h5")
            if h5p.exists():
                with h5py.File(h5p, "r") as f:
                    shape = f["data"].shape
            else:
                t = torch.load(p, weights_only=True)
                shape = t.shape
            # For Hawkes, marks dimension is at index 1
            marks_dim = int(shape[1])
            self._marks_per_dir.append(marks_dim)

        self._different_marks_dim = len(set(self._marks_per_dir)) > 1

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        # shard by rank first, then by worker
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            rank_indices = [i for i in range(self._num_dirs) if (i % world_size) == rank]
        else:
            rank_indices = list(range(self._num_dirs))

        if worker_info is None:
            dir_indices = rank_indices
        else:
            num_workers = worker_info.num_workers
            dir_indices = [i for i in rank_indices if (i % num_workers) == worker_info.id]

        # optional deterministic shuffle per epoch
        if self._shuffle and len(dir_indices) > 1:
            g = torch.Generator()
            g.manual_seed(self._seed + self._epoch)
            perm = torch.randperm(len(dir_indices), generator=g).tolist()
            dir_indices = [dir_indices[j] for j in perm]

        # ensure marks info known
        if self._marks_per_dir is None:
            self._compute_marks_per_dir()

        # iterate directories assigned to this rank/worker
        for dir_idx in dir_indices:
            # load all keys for this directory
            dir_files = {k: v[dir_idx] for k, v in self.all_file_paths.items()}
            # Prefer h5 files if available; open streams once per directory
            streams = {}
            use_h5 = False
            for key, path in dir_files.items():
                h5p = path.with_suffix(".h5")
                if h5p.exists():
                    streams[key] = h5py.File(h5p, "r")
                    use_h5 = True
                elif path.exists():
                    streams[key] = torch.load(path, weights_only=True)
                else:
                    streams[key] = None

            # determine number of items from reference
            if use_h5 and streams.get(self._ref_key) is not None and isinstance(streams[self._ref_key], h5py.File):
                num_items = int(streams[self._ref_key]["data"].shape[0])
            else:
                ref_tensor = streams.get(self._ref_key)
                if ref_tensor is None:
                    # close any opened h5 files
                    for s in streams.values():
                        if isinstance(s, h5py.File):
                            s.close()
                    continue
                num_items = int(ref_tensor.shape[0])

            if self.data_limit is not None:
                num_items = min(num_items, int(self.data_limit))

            try:
                # vectorized block reads to minimize I/O and decompression overhead
                rows_per_block = max(int(self.prefetch_rows), 1)
                for batch_start in range(0, num_items, rows_per_block):
                    batch_end = min(batch_start + rows_per_block, num_items)

                    batch_block: dict[str, torch.Tensor] = {}
                    for key, src in streams.items():
                        if src is None:
                            continue
                        if isinstance(src, h5py.File):
                            arr = src["data"][batch_start:batch_end]
                            batch_block[key] = torch.as_tensor(arr)
                        else:
                            batch_block[key] = src[batch_start:batch_end]

                    # yield individual items from the preloaded block
                    local_rows = batch_end - batch_start
                    for i in range(local_rows):
                        item = {k: v[i] for k, v in batch_block.items()}

                        # add grouping key if marks differ across directories
                        if self._different_marks_dim:
                            item["_group_dim"] = self._marks_per_dir[dir_idx]

                        yield item
            finally:
                # close any opened h5 files
                for s in streams.values():
                    if isinstance(s, h5py.File):
                        s.close()


def h5_files_dict_iterator(files_dict: dict, batch_size: int, process_batch: Optional[callable]):
    """
    Return (consecutive) batches of data extracted from file paths in dict.

    Args:
        files_dict (dict[str, Path]): Maps data keys to file paths to load.
        batch_size (int): Maximal batch size. Remaining data is returned as smaller batch.
        process_batch (Optional[callable]): function pre-processing a data batch dict.
    """
    if files_dict is None:
        pass

    else:
        # open all required files
        files_streams = torch.utils._pytree.tree_map(lambda path: h5py.File(path, "r") if path.exists() else None, files_dict)
        files_data = torch.utils._pytree.tree_map(lambda stream: stream["data"] if stream is not None else None, files_streams)

        # iterate through data consecutive
        num_elements = files_data["obs_values"].shape[0]

        batch_start = 0
        batch_end = batch_size

        while batch_start < num_elements:
            batch_end = min(batch_end, num_elements)

            # get data of batch from file
            batch = torch.utils._pytree.tree_map(lambda f: f[batch_start:batch_end] if f is not None else None, files_data)
            batch = torch.utils._pytree.tree_map(lambda x: torch.from_numpy(x) if x is not None else None, batch)

            # Optional pre-processing and shuffle
            if process_batch is not None:
                batch = process_batch(batch)

            yield torch.utils._pytree.tree_map(lambda x: x.contiguous() if x is not None else None, batch)

            batch_start = batch_end
            batch_end = batch_end + batch_size

        # close all files
        torch.utils._pytree.tree_map(lambda stream: stream.close() if stream is not None else None, files_streams)


def tensor_dict_iterator(data_dict: dict, batch_size: int, process_data: Optional[callable], process_batch: Optional[callable]):
    """
    Return (consecutive) batches of data in a dict with tensors.

    Args:
        data_dict (dict[str, Tensor]): Maps data keys to Tensors.
        batch_size (int): Maximal batch size. Remaining data is returned as smaller batch.
        process_data (Optional[callable]): Apply processing to data dict at start of iterator.
        process_batch (Optional[callable]): Apply processing to batch at each point of iterator.
    """
    if data_dict is None:
        pass

    else:
        if process_data is not None:
            data_dict = process_data(data_dict)

        num_elements = data_dict["obs_values"].shape[0]

        batch_start = 0
        batch_end = batch_size

        while batch_start < num_elements:
            batch_end = min(batch_end, num_elements)

            # select data for current batch
            batch = torch.utils._pytree.tree_map(lambda x: x[batch_start:batch_end], data_dict)

            if process_batch is not None:
                batch = process_batch(batch)

            yield torch.utils._pytree.tree_map(lambda x: x.contiguous(), batch)

            batch_start = batch_end
            batch_end = batch_end + batch_size


def get_file_paths(dir_paths: list[Path], file_names: dict[str, str]) -> dict[str, list[Path]]:
    """
    For list of directories, return full paths of files in directories.

    Example:
        file_names = {"a": "a.h5", "b": "b.h5"}
        return: {"a": [path_1 / "a.h5", path_2 / "a.h5"],
                 "b": [path_1 / "b.h5", path_2 / "b.h5"],}

    Args:
        dir_paths (list[Paths]): Paths to directories containing files to load.
        file_names (dict): Maps keys to filenames in directories.

    Returns:
        file_paths (dict): Maps keys to list of paths to the same kind of file.

    """
    if not isinstance(dir_paths, list | tuple):
        dir_paths = [dir_paths]

    dir_paths: list[Path] = [pathlib.Path(dir_path) for dir_path in dir_paths]  # to be sure

    file_paths = {k: [] for k in file_names.keys()}

    for dir_path in dir_paths:
        for file_key, file_name in file_names.items():
            file_path = dir_path / file_name

            if not file_path.exists() and "values" in file_path.__str__():  # TODO: remove this, just needs to work for now
                file_path = dir_path / "obs_values.h5"

            file_paths[file_key].append(file_path)

    return file_paths


def get_subdict(d: dict, keys: list[str]):
    """
    Returns subsdict with keys that exist in keys of d.

    Args:
        d (dict): Dict to exctract subdict from.
        keys (list[str]): Keys to extract from d, key in d.

    Returns:
        subdict (dict): Values from d, if key from keys exist in d.keys()
    """
    return {k: d[k] for k in keys if k in d.keys()}


def pad_data_in_dict(
    data: dict, keys_to_pad: list[str], dim: int, mode: Optional[str] = "constant", max_length: Optional[int] = None
) -> dict:
    """
    Pad of values in data along dim = -1 or -2.

    Args:
        data (dict): Contains data to pad.
        keys_to_pad (list[str]): Keys of data to pad.
        dim (int): Dimension to pad. Must be -1 or -2.
        mode (Optional[str]): Passed to torch.nn.functional.pad.
        max_length (Optional[int]): Target length. Defaults to maximum found size along dim.

    Return:
        data (dict): Input data with updated keys_to_pad values.
    """
    assert dim in [-1, -2]

    # get data to pad
    data_to_pad = get_subdict(data, keys_to_pad)

    # default padding length to maximum found length
    if max_length is None:
        max_length = max([v.size(dim) for v in torch.utils._pytree.tree_flatten(data_to_pad)[0]])

    # apply padding
    if dim == -2:
        data_to_pad = torch.utils._pytree.tree_map(
            lambda x: torch.nn.functional.pad(x, (0, 0, 0, max_length - x.size(-2)), mode=mode), data_to_pad
        )

    elif dim == -1:
        data_to_pad = torch.utils._pytree.tree_map(
            lambda x: torch.nn.functional.pad(x, (0, max_length - x.size(-1)), mode=mode), data_to_pad
        )

    data.update(data_to_pad)

    return data


def shuffle_at_dim(tree: Any, dim: int) -> Tensor:
    """
    Permute leaf tensors of tree at dim, independently for each batch element, but jointly for each leaf.
    Assumption: all tensors in trees have same shape to (and including) dim.

    Example:
        Tree leaf shapes = {"a": [B0, ..., BN, T, ...], "b": [B0, ..., BN, T, ...]} where T is at dim.
        Different permutation of T per [B0, ..., BN].
        Same permutation for values of "a" and "b".

    Args:
        tree (Any): Tree with tensor leafs.
        dim (int): Dimension to shuffle.

    Return:
        shuffled_tree (Any): Tree with tensor leafs shuffled at dim. (i.e. T)
    """
    # get shape of leafs to dim
    leafs = torch.utils._pytree.tree_flatten(tree)[0]

    shape_to_dim = leafs[0].shape[: dim + 1]

    # Assert assumption of same shapes to (and including) dim
    for leaf in leafs:
        assert leaf.shape[: dim + 1] == shape_to_dim, (
            f"Expected {shape_to_dim}, got {torch.utils._pytree.tree_map(lambda x: x.shape[: dim + 1], tree)}"
        )

    # squash dimensions to dim
    if dim == 0:
        # permutation of  B
        example_leaf: Tensor = torch.utils._pytree.tree_flatten(tree)[0][0]
        rand_indices = torch.argsort(torch.randn(example_leaf.size(0)), dim=0)  # [B, T]

        # apply permutation per B
        shuffled_tree = torch.utils._pytree.tree_map(lambda x: x[rand_indices], tree)  # [B, T, ...]

    else:
        tree = torch.utils._pytree.tree_map(lambda x: x.view((-1,) + x.shape[dim:]), tree)  # [B, T, ...]

        # permutation of T per B
        example_leaf: Tensor = torch.utils._pytree.tree_flatten(tree)[0][0]
        rand_indices = torch.argsort(torch.randn(example_leaf.shape[:2]), dim=1)  # [B, T]

        # vmapped torch.take_along_dim (in first dimension)
        def _take_along_first_dim(x, ind):
            return x[ind]

        vmapped_take_along_first_dim = torch.vmap(_take_along_first_dim)

        # apply permutation per B
        shuffled_tree = torch.utils._pytree.tree_map(lambda x: vmapped_take_along_first_dim(x, rand_indices), tree)  # [B, T, ...]

        # reshape leafs to original shape
        shuffled_tree = torch.utils._pytree.tree_map(lambda x: x.view(shape_to_dim + x.shape[dim + 1 :]), shuffled_tree)

    return shuffled_tree


def shuffle_sde_data(
    data: dict,
    paths_keys: list[str],
    loc_keys: list[str],
    shuffle_paths: bool,
    shuffle_locations: bool,
    shuffle_elements: bool,
):
    """
    Shuffle (some keys of) sde data along paths, locations or batch elements.

    Args:
        data (dict): Contains all sde data.
        paths_keys (list[str]): Keys of data that have paths.
        loc_keys (list[str]): Keys of data that have locations.
        shuffle_X (bool): Flag to shuffle paths, locations or batch elements.

    Return:
        data (dict): Shuffled data
    """
    if data is not None:
        if shuffle_paths is True:
            obs_seq_data = get_subdict(data, paths_keys)
            obs_seq_data = shuffle_at_dim(obs_seq_data, dim=-3)

            data.update(obs_seq_data)
            del obs_seq_data

        if shuffle_locations is True:
            loc_seq_data = get_subdict(data, loc_keys)
            loc_seq_data = shuffle_at_dim(loc_seq_data, dim=-2)

            data.update(loc_seq_data)
            del loc_seq_data

        if shuffle_elements is True:
            data = shuffle_at_dim(data, dim=0)

    return data


def truncate_locations(data: dict[str, Tensor], loc_keys: list[str], trunc_size: int):
    """
    Truncate tensors at location keys in dim=-2 to trunc_size.

    Args:
        data (dict): ALL data.
        keys (list[str]): Keys in data dict to truncate.
        trunc_size (list[str]): Target size.

    Return:
        updated data dict
    """
    if data is not None:
        loc_data = get_subdict(data, loc_keys)
        loc_data = torch.utils._pytree.tree_map(lambda x: torch.narrow(x, dim=-2, start=0, length=trunc_size), loc_data)
        data.update(loc_data)

    return data


def truncate_paths(data: dict[str, Tensor], path_keys: list[str], trunc_size: int):
    """
    Truncate tensors at location keys in dim=-3 to trunc_size.

    Args:
        data (dict): ALL data.
        keys (list[str]): Keys in data dict to truncate.
        trunc_size (list[str]): Target size.

    Return:
        updated data dict
    """
    if data is not None:
        path_data = get_subdict(data, path_keys)
        path_data = torch.utils._pytree.tree_map(lambda x: torch.narrow(x, dim=-3, start=0, length=trunc_size), path_data)
        data.update(path_data)

    return data


def append_to_lists_in_dict(d: dict[str : list[Any]], to_append: dict[str, Any]):
    """
    Append dict to_append to a dict with same key, but lists as values.

    Args:
        d (dict): Has list as values.
        to_append (dict): has same keys as d.

    Returns:
        d_appended (dict). d with appended values in each value.

    """
    for key, value in to_append.items():
        d[key].append(value)

    return d


def distribute_file_paths_among_workers(file_paths: dict[str, list[Path]], num_workers: int) -> list[dict[str, list[Path]]]:
    """
    Distribute file paths in dict evenyl among number of workers.

    Args:
        file_paths (dict): Maps keys to list of paths to the same kind of file. Output of `get_file_paths`.
        num_workers (int): Number of workers to distribute paths among.

    Returns:
        file_paths_per_worker (list[dict]): lengh == num_workers, each entry contains file_paths style dict with only selected paths.
    """
    total_num_files: int = len(file_paths["obs_values"])

    file_paths_per_worker = [{k: [] for k in file_paths.keys()} for _ in range(num_workers)]

    for file_nr in range(total_num_files):
        file_worker_id = file_nr % num_workers

        file_paths_per_worker[file_worker_id] = append_to_lists_in_dict(
            file_paths_per_worker[file_worker_id], {k: v[file_nr] for k, v in file_paths.items()}
        )

    return file_paths_per_worker


def get_iterable_dataset_length(file_paths: dict[str, list[Path]]) -> int:
    """
    Open all files related to "obs_values" and sum their sizes at dim 0.

    Args:
        file_paths (dict): Maps keys to list of paths to the same kind of file. Output of `get_file_paths`.

    Returns:
        dataset_length (int): Summed sizes at dim 0 of tensors.
    """
    file_streams: list = torch.utils._pytree.tree_map(lambda path: h5py.File(path, "r"), file_paths["obs_values"])
    vals_sizes: list = torch.utils._pytree.tree_map(lambda file: file["data"].shape[0], file_streams)
    torch.utils._pytree.tree_map(lambda stream: stream.close(), file_streams)

    return sum(vals_sizes)
