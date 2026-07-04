from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import torch


Vec = torch.Tensor | np.ndarray


@dataclass
class FimDataset:
    coordinates: torch.Tensor = None
    trajectories: torch.Tensor = None
    fx: torch.Tensor = None
    times: torch.Tensor = None
    fx_at_obs_values: torch.Tensor = None

    def __getitem__(self, idx):
        initial_conditions = self.trajectories[idx, :, 0, :]

        return self.fx[idx], self.coordinates[idx], initial_conditions, self.trajectories[idx], self.times[idx]

    def __len__(self):
        if self.trajectories is None:
            return 0
        else:
            return self.trajectories.shape[0]

    @property
    def is_loaded_and_dims_match(self):
        is_loaded = (
            self.coordinates is not None
            and self.trajectories is not None
            and self.fx is not None
            and self.times is not None
            and self.trajectories is not None
        )

        if not is_loaded:
            return False

        bt, tt, nt, dt = self.trajectories.shape
        bts, tts, nts, _ = self.times.shape
        bf, pf, df = self.fx.shape
        bc, pc, dc = self.coordinates.shape

        dims_match = (
            bt == bts == bf == bc  # batches match
            and tt == tts  # number of trajectory
            and nt == nts  # number of trajectory points match
            and pf == pc  # number of coords match number of function points
            and dt == df == dc
        )  # dim of trajectories and fx and coords match

        return dims_match

    def filter_funcs_by_range(self, range: Tuple[int, int], idx_paths: Tuple[int, int] = (0, 1)):
        a, b = range
        i, j = idx_paths

        self.fx = self.fx[a:b, ...]
        self.coordinates = self.coordinates[a:b, ...]
        self.trajectories = self.trajectories[a:b, i:j, ...]
        self.fx_at_obs_values = self.fx_at_obs_values[a:b, i:j, ...]
        self.times = self.times[a:b, i:j, ...]

    def print_shapes(self):
        if not self.is_loaded_and_dims_match:
            msg = "Data not loaded or dimensions do not match."
        else:
            msg = f"""Trajectories shape: {self.trajectories.shape},
Times shape: {self.times.shape}
Fx shape: {self.fx.shape},
Coordinates shape: {self.coordinates.shape},
"""

        print(msg)


class FimDataloader:
    base_path: Path
    torch_dtype: torch.dtype
    data: FimDataset

    def __init__(self, base_path: Path, torch_dtype: torch.dtype = torch.float32):
        self.base_path = base_path
        self.torch_dtype = torch_dtype

    def _get_h5_file_contents(self, filename: str):
        with h5py.File(self.base_path / filename, "r") as h5file:
            keys = list(h5file.keys())
            assert len(keys) == 1 and keys[0] == "data"

            # astype as we have problems with package versions regarding dtype versions
            data = h5file["data"].astype(np.float32)[:]

            return torch.tensor(data, dtype=self.torch_dtype)

    def _load_locations(self):
        self.data.coordinates = self._get_h5_file_contents("locations.h5")

    def _load_drift_at_locations(self):
        self.data.fx = self._get_h5_file_contents("drift_at_locations.h5")

    def _load_obs_times(self):
        self.data.times = self._get_h5_file_contents("obs_times.h5")

    def _load_obs_values(self):
        self.data.trajectories = self._get_h5_file_contents("obs_values.h5")

    def _load_drift_at_obs_values(self):
        try:
            self.data.fx_at_obs_values = self._get_h5_file_contents("drift_at_observations.h5")
        except FileNotFoundError:
            print("drift_at_observations.h5 not found, using drift at locations instead.")
            self.data.fx_at_obs_values = self.data.fx

    def load_dataset(self):
        self.data = FimDataset()
        self._load_locations()
        self._load_drift_at_locations()
        self._load_obs_times()
        self._load_obs_values()
        self._load_drift_at_obs_values()

        # self._filter_data()

        return self.data

    def _filter_data(self):
        trajs = self.data.trajectories

        # in area of initial condition
        # a, b = -4.5, 4.5
        # is_between = ((trajs >= a) & (trajs <= b)).view(trajs.shape[0], -1)
        # all_between = torch.all(is_between, dim=1)

        # not stationary
        # TODO MM: this filter should be dynamic to boundaries of fx
        eps = 0.5
        diffs = torch.diff(trajs, dim=2)
        norm_diffs = torch.norm(diffs, dim=3)
        metric_length_diffs = torch.sum(norm_diffs, dim=2)

        is_stationary = metric_length_diffs < eps
        all_not_stationary = ~torch.all(is_stationary, dim=1)

        mask = all_not_stationary  # & all_between

        print(f"num ok examples: {mask.sum()}, percent okay: {mask.sum() / mask.shape[0] * 100:.2f}%")

        n = int(mask.sum())
        num_paths = 9
        self.data.trajectories = self.data.trajectories[mask][:n, :num_paths, ...]
        self.data.times = self.data.times[mask][:n, :num_paths, ...]
        self.data.fx = self.data.fx[mask][:n, ...]
        self.data.coordinates = self.data.coordinates[mask][:n, ...]


class SpecificDimFimDataset(torch.utils.data.Dataset):
    data_path: Path
    dataset: FimDataset

    def __init__(self, data_path: Path, expected_dim: int):
        self.data_path = data_path

        self._load_data()
        assert self.dataset.trajectories.shape[-1] == expected_dim, (
            f"Expected dimension {expected_dim}, but got {self.dataset.trajectories.shape[-1]}"
        )

    def _load_data(self):
        dl = FimDataloader(self.data_path)
        dataset = dl.load_dataset()
        assert dataset.is_loaded_and_dims_match, "Problem loading dataset."

        self.dataset = dataset

    def __len__(self):
        return self.dataset.trajectories.shape[0]

    def __getitem__(self, idx):
        d = self.dataset

        fx, co, tr, ts = d.fx[idx], d.coordinates[idx], d.trajectories[idx], d.times[idx]

        return fx, co, tr, ts
