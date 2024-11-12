from dataclasses import dataclass, field
from typing import List, Type, TypeVar, Any

T = TypeVar('T')

@dataclass
class DataInFiles:
    obs_times: str
    obs_values: str
    locations: str
    drift_at_locations: str
    diffusion_at_locations: str

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        return cls(**data)

@dataclass
class DatasetPathCollections:
    train: List[str]
    test: List[str]
    validation: List[str]

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        return cls(**data)

@dataclass
class FIMDatasetConfig:
    dataset_description: str
    total_minibatch_size: int
    total_minibatch_size_test: int
    data_loading_processes_count: int
    data_in_files: DataInFiles
    dataset_path_collections: DatasetPathCollections
    tensorboard_figure_data: str
    plot_paths_count: int
    loader_kwargs:dict

    name: str = "FIMSDEDataloader"
    max_dimension:int = 0
    max_time_steps:int = 0
    max_location_size:int = 0
    max_num_paths:int = 0

    def __post_init__(self):
        if isinstance(self.data_in_files, dict):
            self.data_in_files = DataInFiles.from_dict(self.data_in_files)
        if isinstance(self.dataset_path_collections, dict):
            self.dataset_path_collections = DatasetPathCollections.from_dict(self.dataset_path_collections)

