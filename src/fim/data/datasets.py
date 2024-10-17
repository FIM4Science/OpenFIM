import logging
import math
from pathlib import Path
from typing import Optional, Union

import torch
import torch.utils
import torch.utils.data
from datasets import DatasetDict, DownloadMode, get_dataset_split_names, load_dataset
from torch.utils.data import default_collate

from ..utils.helper import verify_str_arg
from ..utils.logging import RankLoggerAdapter
from .utils import load_file, split_into_variable_windows


class HFDataset(torch.utils.data.Dataset):
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
        self.logger.debug(f"Dataset from {path} with name {ds_name} and split {split} loaded successfully.")

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


class FimDataset(torch.utils.data.Dataset):
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
        logger (RankLoggerAdapter): Logger for the dataset class.
        _path (Union[str, Path]): Path to the dataset directory.
        files_to_load (Optional[dict]): Dictionary specifying files to load.
        data_limit (Optional[int]): Limit on the number of data entries to load from each file.
        data (dict): Loaded data from the specified files.
    Methods:
        __init__(path: Union[str, Path], files_to_load: Optional[dict] = None, data_limit: Optional[int] = None):
            Initializes the FimDataset with the given path, files to load, and data limit.
        __load_data() -> dict:
            Loads data from the specified files and returns it as a dictionary.
        __getitem__(idx):
            Returns the data entry at the specified index.
        __get_files() -> list[tuple[str, Path]]:
            Retrieves the list of files to load based on the specified files or all files in the directory.
        path:
            Property to get and set the dataset path.
        __len__():
            Returns the number of data entries in the dataset.
        __str__():
            Returns a string representation of the FimDataset.
    """

    def __init__(self, path: Union[str, Path], files_to_load: Optional[dict] = None, data_limit: Optional[int] = None):
        super().__init__()

        self.logger = RankLoggerAdapter(logging.getLogger(__class__.__name__))
        self._path = path
        self.files_to_load = files_to_load
        self.data_limit = data_limit
        self.logger.debug(f"Loading dataset from {path} with files {files_to_load}.")
        self.data = self.__load_data()
        self.logger.debug(f"Dataset from {path} loaded successfully.")

    def __load_data(self) -> dict:
        files_to_load = self.__get_files()
        return {file_name: load_file(file_path)[: self.data_limit] for file_name, file_path in files_to_load}

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

    def __get_files(self) -> list[tuple[str, Path]]:
        if self.files_to_load is not None:
            files_to_load = self.files_to_load.items()
        else:
            files_to_load = [(f.stem, f) for f in self.path.iterdir() if f.is_file()]
        return files_to_load

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path: Union[str, Path]):
        assert isinstance(path, (str, Path)), f"Expected path to be of type str or Path, but got {type(path)}."
        assert Path(path).exists(), f"Path {path} does not exist."
        self._path = path

    def __len__(self):
        return len(next(iter(self.data.values())))

    def __str__(self):
        return f"FimDataset(path={self.path}, files_to_load={self.files_to_load})"


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

        self.imputation_mask = torch.tensor(imputation_mask) if imputation_mask is not None else None
        if self.imputation_mask is not None:
            assert sum(self.imputation_mask) == 1, "Only one window can be masked out for imputation."

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
        max_sequence_length = batch[0]["coarse_grid_grid"].size(1)
        iwindow_size = dataset._sample_iwindow_size(max_sequence_length)
        output = []
        for item in batch:
            if dataset.imputation_mask is None:
                mask = dataset.sample_imputation_mask(dataset.window_count)

            iwindow_index = mask.nonzero().item()
            output.append(dataset._get_windowed_item(item, iwindow_size, iwindow_index, mask))

        return default_collate(output)

    def _get_windowed_item(self, item: dict, iwindow_size: int, iwindow_index: int, mask: torch.Tensor) -> dict:
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
