import logging
import math
import os
import pathlib
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch
import torch.utils
import torch.utils.data
from datasets import DatasetDict, DownloadMode, get_dataset_split_names, load_dataset
from torch.utils.data import default_collate

from fim import data_path
from fim.data.config_dataclasses import FIMDatasetConfig
from fim.data.utils import load_h5

from ..typing import Path, Paths
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
        self.split = verify_str_arg(
            split, arg="split", valid_values=get_dataset_split_names(path, ds_name, trust_remote_code=True) + [None]
        )
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

    def __init__(self, path: Path | Paths, files_to_load: Optional[dict] = None, data_limit: Optional[int] = None):
        super().__init__()

        self.logger = RankLoggerAdapter(logging.getLogger(__class__.__name__))
        self.path: Paths = path
        self.files_to_load = files_to_load
        self.data_limit = data_limit
        self.logger.debug(f"Loading dataset from {path} with files {files_to_load}.")
        self.data = self.__load_data()
        self.logger.debug(f"Dataset from {path} loaded successfully.")

    def __load_data(self) -> dict:
        data = defaultdict(list)
        files_to_load = self.__get_files()
        for file_name, file_path in files_to_load:
            data[file_name].append(load_file(file_path)[: self.data_limit])
        return {k: torch.cat(v) for k, v in data.items()}

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}

    def __get_files(self) -> Paths:
        if self.files_to_load is not None:
            files_to_load = [(key, path / file_name) for key, file_name in self.files_to_load.items() for path in self.path]
        else:
            files_to_load = [(f.stem, f) for path in self.path for f in path.iterdir() if f.is_file()]
        return files_to_load

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
            raise AssertionError(f"Paths {', '.join(missing_paths)} do not exist.")
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


# FIMSDE ---------------------------------------------------------


@dataclass
class FIMSDEDatabatch:
    obs_values: torch.Tensor | np.ndarray
    obs_times: torch.Tensor | np.ndarray

    drift_at_locations: torch.Tensor | np.ndarray
    diffusion_at_locations: torch.Tensor | np.ndarray
    locations: torch.Tensor | np.ndarray

    diffusion_parameters: torch.Tensor | np.ndarray = None
    drift_parameters: torch.Tensor | np.ndarray = None
    process_label: torch.Tensor | np.ndarray = None
    process_dimension: torch.Tensor | np.ndarray = None

    def convert_to_tensors(self):
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, np.ndarray):
                try:
                    setattr(self, field, torch.tensor(value))
                except Exception:
                    print(f"Problem for field {field}")
                    setattr(self, field, None)


# Define the named tuple
FIMSDEDatabatchTuple = namedtuple(
    "FIMSDEDatabatchTuple",
    [
        "obs_values",
        "obs_times",
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
            data_paths (List[str]): list of locations of .h5 files requiered to load the data
        """
        # To keep track of the number of samples in each file
        self.data = []
        self.lengths = []
        self.data_config = data_config

        # Load data and compute cumulative lengths
        self.read_files(data_config, data_paths)

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

            # prepend fim data dir
            if not os.path.isabs(file_path):
                file_path = os.path.join(data_path, file_path)

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
        # Obtains index of the associated file and item whithin the file
        file_idx, sample_idx = self._get_file_and_sample_index(idx)
        data_bulk: FIMSDEDatabatch = self.data[file_idx]
        # Get the tensor from the appropriate file
        obs_values = data_bulk.obs_values[sample_idx]
        obs_times = data_bulk.obs_times[sample_idx]
        diffusion_at_locations = data_bulk.diffusion_at_locations[sample_idx]
        drift_at_locations = data_bulk.drift_at_locations[sample_idx]
        locations = data_bulk.locations[sample_idx]
        # Pad and Obtain Mask of The tensors if necessary
        obs_values, obs_times = self._pad_obs_tensors(obs_values, obs_times)
        drift_at_locations, diffusion_at_locations, locations, mask = self._pad_locations_tensors(
            drift_at_locations, diffusion_at_locations, locations
        )
        if len(obs_values.shape) == 4:
            obs_values = obs_values[:, :, :, 0]

        # select a smaller set of paths
        # obs_values,obs_times,diffusion_at_locations,drift_at_locations,locations,mask = self._select_paths_and_grid(obs_values,obs_times,diffusion_at_locations,drift_at_locations,locations,mask)

        # Create and return the named tuple
        return FIMSDEDatabatchTuple(
            obs_values=obs_values,
            obs_times=obs_times,
            drift_at_locations=drift_at_locations,
            diffusion_at_locations=diffusion_at_locations,
            locations=diffusion_at_locations,
            dimension_mask=mask,
        )

    def _select_paths_and_grid(
        self,
        obs_values: torch.Tensor,
        obs_times: torch.Tensor,
        drift_at_locations: torch.Tensor,
        diffusion_at_locations: torch.Tensor,
        locations: torch.Tensor,
        dimension_mask: torch.Tensor,
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

    def _pad_obs_tensors(self, obs_values, obs_times):
        """ """
        current_dimension = obs_values.size(2)
        current_time_steps = obs_values.size(1)

        dim_padding_size = self.max_dimension - current_dimension
        time_dim_padding_size = self.max_time_steps - current_time_steps

        if dim_padding_size > 0 or time_dim_padding_size > 0:
            if len(obs_values.shape) == 4:  # comming from h5 files
                obs_values = torch.nn.functional.pad(obs_values, (0, 0, 0, dim_padding_size, 0, time_dim_padding_size))
            elif len(obs_values.shape) == 3:  # comming from target data simulation
                obs_values = torch.nn.functional.pad(obs_values, (0, dim_padding_size, 0, time_dim_padding_size))

            obs_times = torch.nn.functional.pad(obs_times, (0, 0, 0, time_dim_padding_size))

        return obs_values, obs_times

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
