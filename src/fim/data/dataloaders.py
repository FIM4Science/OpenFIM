import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import torch
import torch.distributed as dist
from datasets import get_dataset_split_names, load_dataset_builder
from torch.utils.data.dataloader import DataLoader

from fim.utils.collate import pad_data_collator
from fim.utils.helper import verify_str_arg

from ..data.datasets import BaseDataset, PatchedDatasetSynthetic, PatchedDatasetBase
from ..trainers.utils import is_distributed
from ..utils.logging import RankLoggerAdapter

DistributedSampler = torch.utils.data.distributed.DistributedSampler


def convert_to_pandas_data_range(date: List[datetime], periods: List[int], freq: str):
    pr = [pd.date_range(d, periods=p, freq=freq) for d, p in zip(date, periods)]

    month_and_year_pairs = [[list(pair) for pair in zip(r.month.tolist(), r.year.tolist())] for r in pr]

    return month_and_year_pairs


def transform_start_field_to_time_features(batch: dict, freq: str = "1M", key: str = "target"):
    periods = list(map(len, batch[key]))
    batch["time_feat"] = convert_to_pandas_data_range(batch["start"], periods, freq)
    return batch


class BaseDataLoader:
    def __init__(
        self,
        path: Union[str, Path],
        ds_name: Optional[str] = None,
        split: Optional[str] = None,
        batch_size: Optional[int] = 32,
        test_batch_size: Optional[int] = 32,
        output_fields: Optional[List[str]] = None,
        loader_kwargs: Optional[dict] = {},
        dataset_kwargs: Optional[dict] = {},
    ):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.dataset_kwargs = dataset_kwargs
        self.loader_kwargs = loader_kwargs
        self.iter = {}
        self.path = path
        self.name = ds_name

        self.logger = RankLoggerAdapter(logging.getLogger(__class__.__name__))

        self.split = verify_str_arg(split, arg="split", valid_values=get_dataset_split_names(path, ds_name) + [None])

        if self.split is not None:
            self.dataset = {self.split: BaseDataset(self.path, self.name, self.split, **self.dataset_kwargs)}
        else:
            self.dataset = {
                split_: BaseDataset(self.path, self.name, split_, **self.dataset_kwargs)
                for split_ in get_dataset_split_names(self.path, self.name)
            }
        for dataset in self.dataset.values():
            dataset.map(transform_start_field_to_time_features, batched=True)
            dataset.data.set_format(type="torch", columns=output_fields)

        self._init_dataloaders(self.dataset)

    def _init_dataloaders(self, dataset):
        for n, d in dataset.items():
            sampler = None
            if is_distributed():
                sampler = DistributedSampler(
                    d, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=n == "train"
                )
            batch_size = self.batch_size
            if n != "train":
                batch_size = self.test_batch_size
            self.iter[n] = DataLoader(
                d,
                drop_last=False,
                sampler=sampler,
                shuffle=sampler is None,
                batch_size=batch_size,
                collate_fn=pad_data_collator,
                **self.loader_kwargs,
            )

    def __str__(self) -> str:
        ds_info = load_dataset_builder(self.path, self.name)
        return f"{ds_info.info.description}\n{ds_info.info.features}"

    @property
    def train(self):
        return self.dataset["train"]

    @property
    def train_it(self) -> DataLoader:
        return self.iter["train"]

    @property
    def validation(self):
        return self.dataset["validation"]

    @property
    def validation_it(self) -> DataLoader:
        return self.iter["validation"]

    @property
    def test(self):
        return self.dataset["test"]

    @property
    def test_it(self) -> DataLoader:
        return self.iter["test"]

    @property
    def n_train_batches(self):
        return len(self.train_it)

    @property
    def n_validation_batches(self):
        return len(self.validation_it)

    @property
    def n_test_batches(self):
        return len(self.test_it)

    @property
    def train_set_size(self):
        return len(self.train)

    @property
    def validation_set_size(self):
        return len(self.validation)

    @property
    def test_set_size(self):
        return len(self.test)


class PatchedDataLoader(BaseDataLoader):
    def __init__(
        self,
        path: Union[str, Path],
        ds_name: Optional[str] = None,
        synthetic_data: Optional[bool] = False,
        split: Optional[str] = None,
        batch_size: Optional[int] = 32,
        test_batch_size: Optional[int] = 32,
        output_fields: Optional[List[str]] = None,
        loader_kwargs: Optional[dict] = {},
        dataset_kwargs: Optional[dict] = {},
    ):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.dataset_kwargs = dataset_kwargs
        self.loader_kwargs = loader_kwargs
        self.iter = {}
        self.path = path
        self.name = ds_name

        self.logger = RankLoggerAdapter(logging.getLogger(__class__.__name__))

        self.split = verify_str_arg(split, arg="split", valid_values=get_dataset_split_names(path, ds_name) + [None])

        if synthetic_data:
            dataset_type = PatchedDatasetSynthetic
        else:
            dataset_type = PatchedDatasetBase

        if self.split is not None:
            self.dataset = {
                self.split: dataset_type(path=self.path, ds_name=self.name, split=self.split, **self.dataset_kwargs)
            }
        else:
            self.dataset = {
                split_: dataset_type(
                    path=self.path,
                    ds_name=self.name,
                    split=split_,
                    **self.dataset_kwargs,
                )
                for split_ in get_dataset_split_names(self.path, self.name)
            }
        if "validation" not in self.dataset.keys() and "test" in self.dataset.keys() and "train" in self.dataset.keys():
            self.dataset["validation"] = self.dataset["test"]
            del self.dataset["test"]
            self.logger.warn('No validation set found. Setting changing key "test" to "validation".')

        for dataset in self.dataset.values():
            dataset.map(transform_start_field_to_time_features, batched=True, fn_kwargs={"key": "input"})
            dataset.data.set_format(type="torch", columns=output_fields)

        self._init_dataloaders(self.dataset)

    def _init_dataloaders(self, dataset):
        for n, d in dataset.items():
            sampler = None
            if is_distributed():
                sampler = DistributedSampler(
                    d, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=n == "train"
                )
            batch_size = self.batch_size
            if n != "train":
                batch_size = self.test_batch_size
            self.iter[n] = DataLoader(
                d,
                drop_last=False,
                sampler=sampler,
                shuffle=sampler is None,
                batch_size=batch_size,
                **self.loader_kwargs,
            )


class DataLoaderFactory:
    """Dataloader factory class."""

    object_types = {}

    @classmethod
    def register(cls, object_type: str, object_class: BaseDataLoader) -> None:
        """Register new dataloader type to the factory.

        Args:
            object_type (str): name of the object
            object_class (BaseDataLoader): class that is registered
        """
        cls.object_types[object_type] = object_class

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseDataLoader:
        """Create new dataloader object.

        Args:
            object_type (str): name of the object type that is created

        Raises:
            ValueError: if the object type is not registered

        Returns:
            BaseDataLoader: new instance of the dataloader object
        """
        object_class = cls.object_types.get(name)
        if object_class:
            return object_class(**kwargs)
        else:
            raise ValueError("Invalid object type!")


DataLoaderFactory.register("base_dataloader", BaseDataLoader)
DataLoaderFactory.register("patched_dataloader", PatchedDataLoader)
