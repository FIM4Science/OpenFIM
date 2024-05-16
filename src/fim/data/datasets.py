import logging
from pathlib import Path
from typing import Optional, Union

import torch
from datasets import DatasetDict, DownloadMode, get_dataset_split_names, load_dataset, Dataset

from ..utils.helper import verify_str_arg
from ..utils.logging import RankLoggerAdapter
from .tokenizers import PatcherDecoderOnlyStyle


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


class SyntheticDataset(BaseDataset):
    def __init__(
        self,
        path: Union[str, Path],
        ds_name: Optional[list[str]] = None,
        split: Optional[str] = "train",
        noise: Optional[tuple[float]] = None,
        **kwargs,
    ):
        self.logger = RankLoggerAdapter(logging.getLogger(__class__.__name__))
        self.path = path
        self.name = ds_name
        self.noise_params = noise
        self.logger.debug(f"Loading dataset from {path} and split {split}.")
        self.split = verify_str_arg(split, valid_values=["train", "test", None])
        self.data: DatasetDict = self._load_synthetic_dataset(path, split=split, ds_name=ds_name)

        self.function_types = sorted(set(self.data["function_type"]))

        self.logger.debug("Synthetic Dataset loaded successfully.")

    def __str__(self):
        return f"""SyntheticDataset(
                path={self.path},
                name={self.name},
                split={self.split},
                dataset={self.data},
                function types={self.function_types},
                noise=N(0, {self.noise_params}))
                """

    def _load_synthetic_dataset(self, path, split, ds_name: Optional[list[str]] = None) -> DatasetDict:
        """
        Custom method to load synthetic dataset from cephfs directory.

        Args:
            path: path to dataset root.
                Expected structure: subfolders with function types / < train | test > / < function_samples_grid.pickle | function_samples_values.pickle>
            split: 'train' or 'test'
            ds_name: List of names of data sets aka function types. If None (default) all datasets in subfolder of given path are loaded

        Returns:
            DatasetDict with keys 'target' and 'start' and 'function_type'
        """
        import os
        import pickle
        import numpy as np

        synthetic_data = []
        # iterate over folders in path
        for function_type in sorted(os.listdir(path)):
            if ds_name is not None and function_type not in ds_name:
                continue

            function_path = os.path.join(path, function_type, split)

            with open(os.path.join(function_path, "function_samples_values.pickle"), "rb") as f:
                jax_data = pickle.load(f)

            # convert Jax -> numpy (need copy to make it writeable) -> torch
            np_data = np.asarray(jax_data).copy()
            data = torch.from_numpy(np_data)  # type: torch.FloatTensor

            # squeeze last dimesion -> shape: [nr_functions, len functions = 640]
            data = data.squeeze(-1)

            # TODO: load grid if ever necessary

            # TODO: make larger when deploying
            synthetic_data.extend({"target": ts, "start": 0, "function_type": function_type} for ts in data[:100])

        return Dataset.from_list(synthetic_data)


class PatchedDatasetSynthetic(SyntheticDataset):
    """
    Split each time series into (overlapping) context windows, then into patches. Store corresponding target (horizon) values.
    """

    def __init__(
        self,
        path: Union[str, Path],
        ds_name: Optional[str] = None,
        split: Optional[str] = "train",
        patch_len_in: Optional[int] = 32,
        max_context_len: Optional[int] = None,
        patch_len_out: Optional[int] = 1,
        overlap_context_windows: Optional[int] = None,
        noise_param: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(path=path, ds_name=ds_name, split=split, **kwargs)

        self.logger = RankLoggerAdapter(logging.getLogger(__class__.__name__))

        self.tokenizer = PatcherDecoderOnlyStyle(
            max_context_len=max_context_len,
            patch_len_in=patch_len_in,
            patch_len_out=patch_len_out,
            overlap_context_windows=overlap_context_windows,
        )

        self.data = self.tokenizer.split_data(self.data)

        if noise_param is not None:
            self.noise_param = noise_param
            # add gaussian noise to non-masked input values
            self._add_gaussian_noise()

        self.logger.debug("Dataset successfully divided into context windows and patches.")

    def _add_gaussian_noise(self):
        """
        Add Gaussian noise to the non-masked input values with mean 0 and std noise_param.
        """
        processed_data = []

        for item in self.data:
            input = torch.tensor(item["input"])
            mask = torch.tensor(item["mask_point_level"])

            noise = torch.normal(mean=0, std=self.noise_param, size=input.shape)
            input[~mask] += noise[~mask]
            item["input"] = input
            processed_data.append(item)

        self.data = Dataset.from_list(processed_data)

        self.logger.debug("Gaussian Noise added.")

    def __str__(self):
        return f"""PatchedDatasetSynthetic(\n\tdata={self.data},\t\ntokenizer={self.tokenizer}\n)"""

    def __getitem__(self, idx):
        """
        Get item at index `idx`.

        Args:
            idx (int): The index of the item.

        Returns:
            dict: The item at the given index with keys
                - "input_values" (List[List[float]]): The input values as sequence of patches; [n_patches, patch_len_in]
                - "output_values" (List[float]): The output values; [patch_len_out]
                - "start" (int): The start of the time series.
                - "seq_len" (int): The sequence length, i.e. the number of patches (without padding).
                - "mask_point_level" (List[List[bool]]): The mask on point level; [n_patches, patch_len_in]
                - "mask_token_level" (List[bool]): The mask on token level; [n_patches]
        """
        item = self.data[idx]
        return {
            "input_values": item["input"],
            "output_values": item["output"],
            "time_feat": item["time_feat"],
            "seq_len": sum(~item["mask_token_level"]),
            "mask_point_level": item["mask_point_level"],
            "mask_token_level": item["mask_token_level"],
        }


class PatchedDatasetBase(BaseDataset):
    def __init__(
        self,
        path: Union[str, Path],
        ds_name: Optional[str] = None,
        split: Optional[str] = "train",
        patch_len_in: Optional[int] = 32,
        max_context_len: Optional[int] = None,
        patch_len_out: Optional[int] = 1,
        overlap_context_windows: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(path=path, ds_name=ds_name, split=split, **kwargs)

        self.logger = RankLoggerAdapter(logging.getLogger(__class__.__name__))

        self.tokenizer = PatcherDecoderOnlyStyle(
            max_context_len=max_context_len,
            patch_len_in=patch_len_in,
            patch_len_out=patch_len_out,
            overlap_context_windows=overlap_context_windows,
        )

        self.data = self.tokenizer.split_data(self.data)

        self.logger.debug("Dataset successfully divided into context windows and patches.")

    def __getitem__(self, idx):
        """
        Get item at index `idx`.

        Args:
            idx (int): The index of the item.

        Returns:
            dict: The item at the given index with keys
                - "input_values" (List[List[float]]): The input values as sequence of patches; [n_patches, patch_len_in]
                - "output_values" (List[float]): The output values; [patch_len_out]
                - "start" (int): The start of the time series.
                - "seq_len" (int): The sequence length, i.e. the number of patches (without padding).
                - "mask_point_level" (List[List[bool]]): The mask on point level; [n_patches, patch_len_in]
                - "mask_token_level" (List[bool]): The mask on token level; [n_patches]
        """
        item = self.data[idx]
        return {
            "input_values": item["input"],
            "output_values": item["output"],
            "time_feat": item["time_feat"],
            "seq_len": sum(~item["mask_token_level"]),
            "mask_point_level": item["mask_point_level"],
            "mask_token_level": item["mask_token_level"],
        }

    def __str__(self):
        return f"""PatchedDatasetBase(
data={self.data}
tokenizer={self.tokenizer}
)"""
