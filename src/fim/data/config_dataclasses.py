from dataclasses import dataclass, field
from typing import List, Type, TypeVar, Any, Dict
from fim import data_path  # Assumes `data_path` provides the base path for your data files
from fim import project_path  # Assumes `data_path` provides the base path for your data files
import yaml

T = TypeVar('T')

@dataclass
class DataInFiles:
    obs_times: str = "obs_times.h5"
    obs_values: str = "obs_values.h5"
    locations: str = "hypercube_locations.h5"
    drift_at_locations: str = "drift_functions_at_hypercube.h5"
    diffusion_at_locations: str = "scaled_diffusion_functions_at_hypercube.h5"

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        return cls(**data)

@dataclass
class DatasetPathCollections:
    train: List[str] = field(default_factory=lambda: [
        f"{data_path}/processed/state_sde/linear/dim-1/1",
        f"{data_path}/processed/state_sde/linear/dim-2/1",
        f"{data_path}/processed/state_sde/linear/dim-3/1"
    ])
    test: List[str] = field(default_factory=lambda: [
        f"{data_path}/processed/state_sde/linear/dim-2/1"
    ])
    validation: List[str] = field(default_factory=lambda: [
        f"{data_path}/processed/state_sde/linear/dim-2/1"
    ])

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        return cls(**data)

@dataclass
class FIMDatasetConfig:
    name: str = "FIMSDEDataloader"
    type:str = "synthetic" # synthetic, theory
    dataset_description: str = "dynamical_systems"
    #data_path:str = str(data_path)

    dynamical_systems_hyperparameters_file:str = r"\configs\train\fim-sde\sde-systems-hyperparameters.yaml"

    total_minibatch_size: int = 2
    total_minibatch_size_test: int = 2
    data_loading_processes_count: int = 0

    data_in_files: DataInFiles = field(default_factory=DataInFiles)
    dataset_path_collections: DatasetPathCollections = field(default_factory=DatasetPathCollections)

    plot_paths_count: int = 100
    tensorboard_figure_data: str = "test"
    loader_kwargs: Dict[str, Any] = field(default_factory=lambda: {"num_workers": 2})

    max_dimension: int = 3
    max_time_steps: int = 128
    max_location_size: int = 1024
    max_num_paths: int = 30

    def __post_init__(self):
        if isinstance(self.data_in_files, dict):
            self.data_in_files = DataInFiles.from_dict(self.data_in_files)
        if isinstance(self.dataset_path_collections, dict):
            self.dataset_path_collections = DatasetPathCollections.from_dict(self.dataset_path_collections)
        
        self.dynamical_systems_hyperparameters_file = rf"{project_path}"+self.dynamical_systems_hyperparameters_file

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'FIMDatasetConfig':
        with open(yaml_path, 'r') as file:
            params_dict = yaml.safe_load(file)
        return cls(**params_dict)
