import logging
from pathlib import Path
from typing import Optional, Union
import random
import math
from itertools import pairwise

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


class PatchedDataset(BaseDataset):
    """
    Split each time series into context windows, y sequence of patches. Store corresponding target (horizon) values.
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
        overlap_context_windows: Optional[int] = 0,
        **kwargs,
    ):
        super().__init__(path, ds_name, split, download_mode, **kwargs)

        self.logger = RankLoggerAdapter(logging.getLogger(__class__.__name__))
        self.max_context_len = max_context_len
        self.patch_len_out = patch_len_out
        self.patch_len_in = patch_len_in
        self.overlap_context_windows = overlap_context_windows
        self.skipped_time_series_counter = 0

        processed_data = []

        for row in self.data:
            processed_data.extend(self._split_time_series(time_series=row["target"], time_start=row["start"]))

        if self.skipped_time_series_counter > 0:
            self.logger.warning(
                f"Skipped {self.skipped_time_series_counter} of {len(self.data)} time series due to insufficient length for context and horizon windows."
            )
        self.data = Dataset.from_list(processed_data)
        self.logger.debug("Dataset successfully divided into context windows and patches.")

    def _split_time_series(self, time_series, time_start):
        """
        Split a time series into (overlapping) context windows, then into patches. Also pad patches and create masks.

        Args:
            time_series (List[float]): The time series to split.
            time_start (int): The start time of the time series.

        Returns:
            List[dict]]: The data entries for the given time series. One entry consists of:
                'input': the input sequence of patches, padded as necessary. [max_nr_patches_per_context_window, patch_len_in]
                'output': the output sequence of target values, subsequent to the last non-padded input token/time series point. [patch_len_out]
                'mask_point_level': the mask on point level. [max_nr_patches_per_context_window, patch_len_in]
                    Following the masking strategy of Das et al, the first r (random number) points of the fist patch are
                    masked out. Further, if the last patch is not full length, the remaining points are masked out.
                    True indicates that the point is masked out.
                'mask_token_level': the mask on token level. [max_nr_patches_per_context_window]
                    Mask for each patch, indicating whether the patch is fully masked out.
                    True indicates that the patch is fully masked out.
                'start': the start time of the time series. Note: needs to be updated if time feature is ever relevant.
        """
        # Safety check to ensure context length can accommodate at least one input patch
        if self.max_context_len < self.patch_len_in:
            raise ValueError("max context length < input patch length.")
        if len(time_series) <= self.max_context_len + self.patch_len_out:
            self.skipped_time_series_counter += 1
            return []

        # Calculate the maximum number of patches per context window
        max_nr_patches_per_context_window = math.ceil(self.max_context_len / self.patch_len_in)
        processed_data = []

        # Determine the starting indices for each context window in the series
        context_start_indices = list(
            range(
                0,
                len(time_series) - self.max_context_len - self.patch_len_out,
                self.max_context_len - self.overlap_context_windows,
            )
        )

        for context_start_id in context_start_indices:
            # Calculate the start indices for each patch within the current context
            patch_start_indices = [
                context_start_id + patch_nr * self.patch_len_in
                for patch_nr in range(0, max_nr_patches_per_context_window)
            ]
            # append index of last element of last patch
            patch_start_indices.append(context_start_id + self.max_context_len)

            # extract patches + subsequent predictions
            patches = [time_series[start:end] for start, end in pairwise(patch_start_indices)]
            predictions = [
                time_series[patch_end : patch_end + self.patch_len_out] for patch_end in patch_start_indices[1:]
            ]

            for last_patch_id in range(0, len(patches)):
                # make last patch ull length + create mask on point level for this patch
                mask_point_level_final_patch = [False] * len(patches[last_patch_id]) + [True] * (
                    self.patch_len_in - len(patches[last_patch_id])
                )
                if len(patches[last_patch_id]) < self.patch_len_in:
                    patches[last_patch_id].extend([0] * (self.patch_len_in - len(patches[last_patch_id])))

                for nr_patches in range(last_patch_id + 1):
                    # Assemble context from selected patches with padding as necessary
                    cur_patched_context = patches[last_patch_id - nr_patches : last_patch_id + 1] + [
                        [0] * self.patch_len_in
                    ] * (max_nr_patches_per_context_window - nr_patches - 1)

                    # mask first patch on point level
                    r = random.randint(0, self.patch_len_in - 1)
                    if nr_patches == 0:
                        mask_point_level = [[True] * r + mask_point_level_final_patch[r:]] + [
                            [True] * self.patch_len_in
                        ] * (max_nr_patches_per_context_window - nr_patches - 1)
                    else:
                        mask_point_level = (
                            [[True] * r + [False] * (self.patch_len_in - r)]
                            + [[False] * self.patch_len_in] * (nr_patches - 1)
                            + [mask_point_level_final_patch]
                            + [[True] * self.patch_len_in] * (max_nr_patches_per_context_window - nr_patches - 1)
                        )
                    assert len(mask_point_level) == len(cur_patched_context)

                    # get mask on token level
                    mask_token_level = [all(m) for m in mask_point_level]

                    # Append processed data entry
                    # TODO: fix "start" if time feature is ever relevant
                    processed_data.append(
                        {
                            "input": cur_patched_context,
                            "output": predictions[last_patch_id],
                            "mask_point_level": mask_point_level,
                            "mask_token_level": mask_token_level,
                            "start": time_start,
                        }
                    )
        return processed_data

    def __str__(self):
        return f"""PatchedDataset(
                path={self.path},
                name={self.name},
                split={self.split},
                max_context_len={self.max_context_len},
                patch_len_in={self.patch_len_in},
                patch_len_out={self.patch_len_out},
                overlap_context_windows={self.overlap_context_windows},
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

    def __len__(self):
        return len(self.data)
