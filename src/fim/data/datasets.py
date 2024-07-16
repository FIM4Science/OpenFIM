import logging
from pathlib import Path
from typing import Optional, Union

import torch
from datasets import Dataset, DatasetDict, DownloadMode, get_dataset_split_names, load_dataset
from tqdm import tqdm

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
        debugging_cut_off: Optional[int] = None,
        **kwargs,
    ):
        self.logger = RankLoggerAdapter(logging.getLogger(__class__.__name__))
        self.path = path
        self.name = ds_name
        self.noise_params = noise
        self.logger.debug(f"Loading dataset from {path} and split {split}.")
        self.split = verify_str_arg(split, valid_values=["train", "test", None])
        self.data: DatasetDict = self._load_synthetic_dataset(
            path, split=split, ds_name=ds_name, debugging_cut_off=debugging_cut_off
        )

        self.function_types = sorted(set(self.data["function_type"]))

        self.logger.debug("Synthetic Dataset loaded successfully.")

    def __str__(self):
        return f"""SyntheticDataset(
                path={self.path},
                name={self.name},
                split={self.split},
                dataset={self.data},
                noise=N(0, {self.noise_params}),
                function types={self.function_types})
                """

    def _load_synthetic_dataset(
        self,
        path,
        split,
        ds_name: Optional[list[str]] = None,
        debugging_cut_off: Optional[int] = None,
    ) -> DatasetDict:
        """
        Custom method to load synthetic dataset from cephfs directory.

        Args:
            path: path to dataset root.
                Expected structure: subfolders with function types / < train | test > / < function_samples_grid.pickle | function_samples_values.pickle>
            split: 'train' or 'test'
            ds_name: name of data set to load. If None (default) all datasets in subfolder of given path are loaded
            debugging_cut_off: Optional cut off for debugging purposes to reduce the size of the dataset

        Returns:
            DatasetDict with keys 'target' and 'start' and 'function_type'
        """
        import os
        import pickle

        import numpy as np
        import pandas as pd
        import pyarrow as pa
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq

        def read_pickle(file_path: str) -> torch.FloatTensor:
            with open(file_path, "rb") as f:
                jax_data = pickle.load(f)
                # convert Jax -> numpy (need copy to make it writeable) -> torch
                if isinstance(jax_data, tuple):
                    # needed for concepts
                    return (
                        torch.from_numpy(np.asarray(jax_data[0]).copy()),
                        torch.from_numpy(np.asarray(jax_data[1]).copy()) if jax_data[1] is not None else None,
                    )

                return torch.from_numpy(np.asarray(jax_data).copy())  # type: torch.FloatTensor

        obs_times, obs_values, obs_mask = [], [], []
        fine_grid_times, fine_grid_values, fine_grid_concept_values = [], [], []

        # iterate over folders in path
        l = len(os.listdir(path)) if ds_name is None else 1
        for function_type in tqdm(
            sorted(os.listdir(path)),
            desc=f"Loading synthetic data ({split})",
            total=l,
            leave=True,
        ):
            if ds_name is not None and function_type not in ds_name:
                continue

            function_path = os.path.join(path, function_type, split)

            obs_times.append(
                read_pickle(
                    os.path.join(function_path, "coarse_grid_grid.pickle"),
                )[:debugging_cut_off]
            )

            obs_values.append(
                read_pickle(
                    os.path.join(function_path, "coarse_grid_noisy_sample_paths.pickle"),
                )[:debugging_cut_off]
            )
            obs_mask.append(
                read_pickle(
                    os.path.join(function_path, "coarse_grid_observation_mask.pickle"),
                )[:debugging_cut_off]
            )
            fine_grid_times.append(
                read_pickle(
                    os.path.join(function_path, "fine_grid_grid.pickle"),
                )[:debugging_cut_off]
            )
            fine_grid_values.append(
                read_pickle(
                    os.path.join(function_path, "fine_grid_noisy_sample_paths.pickle"),
                )[:debugging_cut_off]
            )

            fine_grid_concept_values.append(
                read_pickle(
                    os.path.join(function_path, "fine_grid_concept_values.pickle"),
                )[0][:debugging_cut_off]
            )

        print("Data loaded successfully.", flush=True)

        obs_times = torch.cat(obs_times, dim=0).tolist()
        obs_values = torch.cat(obs_values, dim=0).tolist()
        obs_mask = torch.cat(obs_mask, dim=0).tolist()
        fine_grid_times = torch.cat(fine_grid_times, dim=0).tolist()
        fine_grid_values = torch.cat(fine_grid_values, dim=0).tolist()
        fine_grid_concept_values = torch.cat(fine_grid_concept_values, dim=0).tolist()

        df = pd.DataFrame(
            {
                "obs_times": obs_times,
                "obs_values": obs_values,
                "obs_mask": obs_mask,
                "fine_grid_times": fine_grid_times,
                "fine_grid_values": fine_grid_values,
                "fine_grid_concept_values": fine_grid_concept_values,
            }
        )
        print("Data converted to pandas DataFrame.", flush=True)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, "synthetic_data.parquet")
        ds = ds.dataset("synthetic_data.parquet", format="parquet")
        print("Dataset created.", flush=True)
        return ds

    def __getitem__(self, idx):
        """
        Get item at index `idx`.

        Args:
            idx (int): The index of the item.

        Returns:
            dict: The item at the given index with keys
                - "obs_values" (List[List[float]]): observed values; [seq_len, D]
                - "obs_times" (List[List[float]]): observed times; [seq_len, dim_time]
                - "obs_mask" (List[bool]): mask for observed values; [seq_len]
                - fine_grid_times (List[float]): fine grid times; [fine_grid_len]
                - "fine_grid_values" (List[float]): fine grid values; [fine_grid_len]
        """
        item = self.data[idx]

        non_masked_values = int(torch.sum(~torch.Tensor(item["obs_mask"]).bool()))

        return item | {"seq_len": non_masked_values}


class DummyDataset(BaseDataset):
    """Class for simple dummy datasets to test the implementation of a model. Expects the data to be in <split>.pt files with appropriate keys."""

    def __init__(self, path: str, name: str, split: Optional[str]):
        self.logger = RankLoggerAdapter(logging.getLogger(__class__.__name__))
        self.logger.warn(f"name {name} not used in Dummy Dataset")

        self.path = path
        self.logger.debug(f"Loading dataset from {path} and split {split}.")
        self.split = verify_str_arg(split, valid_values=["train", "test", "validation", None])
        self.data: DatasetDict = self._load_dummy_dataset(
            path,
            split=split,
        )

        self.logger.debug("Dummy Dataset loaded successfully.")

    def _load_dummy_dataset(self, path, split) -> DatasetDict:
        data: dict = torch.load(path + split + ".pt")
        data["coarse_grid_observation_mask"] = data["coarse_grid_observation_mask"].bool()
        data["fine_grid_sample_paths"] = data["coarse_grid_sample_paths"]
        return Dataset.from_dict(data)

    def __getitem__(self, idx):
        """
        Get item at index `idx`.

        Args:
            idx (int): The index of the item.

        Returns:
            dict: The item at the given index with keys
                - "obs_values" (List[List[float]]): observed values; [seq_len, D]
                - "obs_times" (List[List[float]]): observed times; [seq_len, dim_time]
                - "obs_mask" (List[bool]): mask for observed values; [seq_len]
                - "fine_grid_times" (List[float]): fine grid times; [fine_grid_len]
                - "fine_grid_values" (List[float]): fine grid values; [fine_grid_len]
                - "fine_grid_sample_paths" (List[float]): fine grid sample paths, ie. ground truth solution; [fine_grid_len]
        """
        item = self.data[idx]

        # non_masked_values = int(torch.sum(~torch.Tensor(item["coarse_grid_observation_mask"]).bool()))
        # return item | {"seq_len": non_masked_values}

        return item | {"seq_len": len(item["coarse_grid_observation_mask"])}


class PatchedDatasetSynthetic(SyntheticDataset):
    # TODO check if still compatible
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

        for item in tqdm(self.data, desc="Adding Gaussian Noise"):
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
