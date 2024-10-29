import logging
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import torch
import torch.distributed as dist
import torch.utils
from datasets import load_dataset_builder
from torch import Tensor
from torch.utils.data import default_collate
from torch.utils.data.dataloader import DataLoader

from ..data.datasets import FIMDataset, TimeSeriesImputationDatasetTorch
from ..trainers.utils import is_distributed
from ..utils.helper import create_class_instance, verify_str_arg
from ..utils.logging import RankLoggerAdapter
from .utils import get_path_counts


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
    def __init__(self, dataset_kwargs: dict, loader_kwargs: dict):
        self.logger = RankLoggerAdapter(logging.getLogger(__class__.__name__))

        self.batch_size = loader_kwargs.pop("batch_size")
        self.test_batch_size = loader_kwargs.pop("test_batch_size")
        self.dataset_kwargs = dataset_kwargs
        self.loader_kwargs = loader_kwargs
        self.iter = {}
        self.dataset = {}
        self.samplers = {}

    def _init_dataloaders(self, dataset: dict[str, torch.utils.data.Dataset]):
        for n, d in dataset.items():
            sampler = None
            if is_distributed():
                sampler = DistributedSampler(d, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=n == "train")
            self.samplers[n] = sampler
            batch_size = self.batch_size
            if n != "train":
                batch_size = self.test_batch_size
            self.iter[n] = DataLoader(
                d,
                drop_last=False,
                sampler=sampler,
                shuffle=sampler is None,
                batch_size=batch_size,
                collate_fn=self._get_collate_fn(n, d),
                **self.loader_kwargs,
            )

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

    def _get_collate_fn(self, dataset_name: str, dataset: torch.utils.data.Dataset) -> Union[None, callable]:
        return None

    def __str__(self) -> str:
        ds_info = load_dataset_builder(self.path, self.name)
        return f"{ds_info.info.description}\n{ds_info.info.features}"


class FIMDataLoader(BaseDataLoader):
    def __init__(self, path_collections: dict[str, list[str | Path]], dataset_kwargs: dict, loader_kwargs: dict):
        self.max_path_count = loader_kwargs.pop("max_path_count", None)
        self.max_number_of_minibatch_sizes = loader_kwargs.pop("max_number_of_minibatch_sizes", None)
        self.variable_num_of_paths = loader_kwargs.pop("variable_num_of_paths", False)
        super().__init__(dataset_kwargs, loader_kwargs)
        if self.variable_num_of_paths:
            assert (
                self.max_number_of_minibatch_sizes is not None
            ), "max_number_of_minibatch_sizes must be provided if variable_num_of_paths is True"
            assert self.max_path_count is not None, "max_path_conunt must be provided if variable_num_of_paths is True"
            self.batch_path_sizes = {}

        self.path_collections = path_collections
        for name, paths in path_collections.items():
            self.dataset[name] = FIMDataset(paths, **dataset_kwargs)
            if self.variable_num_of_paths:
                self.batch_path_sizes[name] = get_path_counts(
                    len(self.dataset[name]),
                    self.batch_size if name == "train" else self.test_batch_size,
                    self.max_path_count,
                    self.max_number_of_minibatch_sizes,
                )

        self._init_dataloaders(self.dataset)

    @staticmethod
    def var_path_collate_fun(batch: List[dict], paths_per_batch: List[int], max_path_count: int):
        batch_data = []
        paths_to_use = paths_per_batch.pop(0)  # Take the number of paths for this batch
        path_idxs = torch.randint(0, max_path_count, (paths_to_use,))
        for item in batch:
            selected_paths = {}
            for k, v in item.items():
                if isinstance(v, Tensor) and v.dim() != 0 and v.size(0) == max_path_count:
                    selected_paths[k] = v[path_idxs]
                else:
                    selected_paths[k] = v
            batch_data.append(selected_paths)

        return default_collate(batch_data)

    def _get_collate_fn(self, dataset_name: str, dataset: torch.utils.data.Dataset) -> Union[None, callable]:
        if self.variable_num_of_paths:
            return partial(
                FIMDataLoader.var_path_collate_fun,
                paths_per_batch=list(self.batch_path_sizes[dataset_name]).copy(),
                max_path_count=self.max_path_count,
            )
        return super()._get_collate_fn(dataset_name, dataset)


class TimeSeriesDataLoaderTorch:
    """Datalaoder for time series data in torch format."""

    def __init__(
        self,
        path: Union[str, Path],
        ds_name: Optional[str] = None,
        split: Optional[str] = None,
        batch_size: Optional[int] = 32,
        test_batch_size: Optional[int] = 32,
        output_fields: Optional[List[str]] = None,
        loader_kwargs: Optional[dict] = {},
        dataset_name: str = "fim.data.datasets.TimeSeriesDatasetTorch",
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

        dataset_split_names = ["train", "test", "validation"]

        self.split = verify_str_arg(split, arg="split", valid_values=dataset_split_names + [None])

        if self.split is not None:
            self.dataset = {
                self.split: create_class_instance(
                    dataset_name,
                    {
                        "path": self.path,
                        "ds_name": self.name,
                        "split": self.split,
                        "output_fields": output_fields,
                        **self.dataset_kwargs,
                    },
                )
            }
        else:
            self.dataset = {
                split_: create_class_instance(
                    dataset_name,
                    {
                        "path": self.path,
                        "ds_name": self.name,
                        "split": split_,
                        "output_fields": output_fields,
                        **self.dataset_kwargs,
                    },
                )
                for split_ in dataset_split_names
            }

        self._init_dataloaders(self.dataset)

    def _init_dataloaders(self, dataset):
        for n, d in dataset.items():
            sampler = None
            if is_distributed():
                sampler = DistributedSampler(d, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=n == "train")
            batch_size = self.batch_size
            if n != "train":
                batch_size = self.test_batch_size
            self.iter[n] = DataLoader(
                d,
                drop_last=False,
                sampler=sampler,
                shuffle=sampler is None and n == "train",
                batch_size=batch_size,
                collate_fn=partial(TimeSeriesImputationDatasetTorch.collate_fn, dataset=d)
                if isinstance(d, TimeSeriesImputationDatasetTorch)
                else None,
                **self.loader_kwargs,
            )

    def __str__(self) -> str:
        dataset_desc = {k: str(v) for k, v in self.dataset.items()}
        return f"TimeSeriesDataLoaderTorch=(batch_size={self.batch_size}, test_batch_size={self.test_batch_size}, dataset={dataset_desc})"

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


DataLoaderFactory.register("ts_torch_dataloader", TimeSeriesDataLoaderTorch)
DataLoaderFactory.register("FIMDataLoader", FIMDataLoader)
