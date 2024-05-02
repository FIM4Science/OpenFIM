import logging
from pathlib import Path
from typing import Optional, Union
import random

import torch
from datasets import DatasetDict, DownloadMode, get_dataset_split_names, load_dataset, Dataset

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
        self.split = verify_str_arg(split, arg="split", valid_values=get_dataset_split_names(path, ds_name) + [None])
        self.path = path
        self.name = ds_name
        self.logger.debug(f"Loading dataset from {path} with name {ds_name} and split {split}.")
        self.data: DatasetDict = load_dataset(path, ds_name, split=split, download_mode=download_mode, **kwargs)
        self.logger.debug("Dataset loaded successfully.")

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


class PatchedDataset(BaseDataset):
    """
    Split each time series into sequence of patches. Store corresponding target (horizon) values.
    """

    def __init__(
        self,
        path: Union[str, Path],
        ds_name: Optional[str] = None,
        split: Optional[str] = "train",
        download_mode: Optional[DownloadMode | str] = None,
        patch_len_in: Optional[int] = 32,
        max_context_len: Optional[int] = None,
        patch_len_out: Optional[int] = 1,
        **kwargs,
    ):
        super().__init__(path, ds_name, split, download_mode, **kwargs)

        self.max_context_len = max_context_len
        self.patch_len_out = patch_len_out
        self.patch_len_in = patch_len_in

        max_nr_patches_per_context_window = max_context_len // patch_len_in
        new_data = []

        # split time series into patches
        for row in self.data:
            time_series = row["target"]
            patches = [
                time_series[i : i + patch_len_in]
                for i in range(0, len(time_series) - patch_len_in - patch_len_out, patch_len_in)
            ]
            predictions = [
                time_series[i : i + patch_len_out]
                for i in range(patch_len_in, len(time_series) - patch_len_out, patch_len_in)
            ]
            # TODO treat test / validation differently ?
            for context_id in range(len(patches) - max_nr_patches_per_context_window + 1):
                for nr_patches in range(1, max_nr_patches_per_context_window):
                    # include padding
                    cur_patches = patches[context_id : context_id + nr_patches] + [[0] * patch_len_in] * (
                        max_nr_patches_per_context_window - nr_patches
                    )
                    cur_predictions = predictions[context_id + nr_patches - 1]

                    # create mask on time-point level: mask out random first r values of first patch + padded patches
                    r = random.randint(0, patch_len_in - 1)
                    mask_point_level = (
                        [[True] * r + [False] * (patch_len_in - r)]
                        + [[False] * patch_len_in] * (nr_patches - 1)
                        + [[True] * patch_len_in] * (max_nr_patches_per_context_window - nr_patches)
                    )

                    # create mask on token level: mask out tokens that are completely masked out by mask_point_level
                    mask_token_level = [all(m) for m in mask_point_level]
                    # TODO: fix "start" if time feature is ever relevant
                    new_data.append(
                        {
                            "start": row["start"],
                            "input": cur_patches,
                            "output": cur_predictions,
                            "mask_point_level": mask_point_level,
                            "mask_token_level": mask_token_level,
                        }
                    )
        self.data = Dataset.from_list(new_data)

    def __str__(self):
        return f"""PatchedDataset(
                path={self.path},
                name={self.name},
                split={self.split},
                patch_len_in={self.patch_len_in},
                max_context_len={self.max_context_len},
                patch_len_out={self.patch_len_out},
                dataset={self.data}
                )"""

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
                - "seq_len" (int): The sequence length, i.e. the number of patches.
                - "mask"
        """
        item = self.data[idx]
        input_values = item["input"]
        sequence_length = len(input_values)
        return {
            "input_values": input_values,
            "output_values": item["output"],
            "time_feat": item["time_feat"],
            "seq_len": sequence_length,
            "mask_point_level": item["mask_point_level"],
            "mask_token_level": item["mask_token_level"],
        }

    def __len__(self):
        return len(self.data)
