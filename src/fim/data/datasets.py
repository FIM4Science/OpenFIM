import logging
from pathlib import Path
from typing import Optional, Union

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


class ContextualizedDataset(BaseDataset):
    """
    Present data so that each entry is a time series snippet of length max_context_len + prediction_len.
    """

    def __init__(
        self,
        path: Union[str, Path],
        ds_name: Optional[str] = None,
        split: Optional[str] = "train",
        download_mode: Optional[DownloadMode | str] = None,
        max_context_len: Optional[int] = None,
        prediction_len: Optional[int] = 1,
        **kwargs,
    ):
        super().__init__(path, ds_name, split, download_mode, **kwargs)

        self.max_context_len = max_context_len
        self.prediction_len = prediction_len

        start = []
        context_and_horizon_windows = []
        for row in self.data:
            for i in range(len(row["target"]) - max_context_len - prediction_len + 1):
                context_and_horizon_windows.append(row["target"][i : i + max_context_len + prediction_len])
                start.append(row["start"])
        self.data = Dataset.from_dict({"start": start, "target": context_and_horizon_windows})

    def __str__(self):
        return f"""ContextualizedDataset(
                path={self.path},
                name={self.name},
                split={self.split},
                max_context_len={self.max_context_len},
                prediction_len={self.prediction_len},
                dataset={self.data}
                )"""

    def __getitem__(self, idx):
        item, _ = super().__getitem__(idx)
        input_values = item["target"][: -self.prediction_len]
        target_values = item["target"][-self.prediction_len :]
        sequence_length = len(input_values)
        return {
            "input_values": input_values,
            "target_values": target_values,
            "start": item["start"],
            "seq_len": sequence_length,
        }

    def __len__(self):
        return len(self.data)
