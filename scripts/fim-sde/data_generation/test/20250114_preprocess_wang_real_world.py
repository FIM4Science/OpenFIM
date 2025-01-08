import pickle
from pathlib import Path

import numpy as np
import optree

from fim import data_path
from fim.data_generation.sde.preprocess_utils import save_arrays_from_dict


def process_single_trajectory(file_path: Path) -> dict:
    """
    Return data from single trajectory as one-path + batched data (s.t. it can be loaded by dataloader).

    Args:
        file_path: Path to .pickle containing array of shape [T].

    Returns:
        dict with keys:
            obs_times: Artifial observation times. Shape: [1, 1, T, 1]
            obs_values: Content of file path. Shape: [1, 1, T, 1]
            obs_fluctuations: (Recomputed) fluctuation of obs_values. Shape: [1, 1, T-1, 1]
    """
    data = pickle.load(open(file_path, "rb"))  # [T]
    assert data.ndim == 1

    # Create artificial time
    T = data.shape[0]
    times = np.arange(T)

    obs_times = times.reshape(1, 1, T, 1)
    obs_values = data.reshape(1, 1, T, 1)
    obs_fluctuations = obs_values[:, :, 1:, :] - obs_values[:, :, :-1, :]

    return {"obs_times": obs_times, "obs_values": obs_values, "obs_fluctuations": obs_fluctuations}


def split_trajectory_data(single_traj_data: dict, split_path_length: int, reset_time_per_path: bool) -> dict:
    """
    Split pre-processed single trajectory data into paths.

    Args:
        single_traj_data (dict): Returned from `process_single_trajectory`.
        split_path_length (int): (Max.) length of each path after splitting.
        reset_time_per_path (bool): If true, resets first time of each path to 0.

    Returns:
        Similar to `process_single_trajectory`, but with non-unit path dimension.
    """
    obs_times = single_traj_data.get("obs_times")  # [1, 1, T, 1]
    obs_values = single_traj_data.get("obs_values")  # [1, 1, T, 1]
    obs_fluctuations = single_traj_data.get("obs_fluctuations")  # [1, 1, T-1, 1]

    T = obs_times.shape[-2]

    obs_mask = np.ones((1, 1, T, 1)).astype(bool)

    # if required: pad before reshaping
    if T % split_path_length != 0:
        pad_width = split_path_length - (T % split_path_length)
        obs_times, obs_values, obs_mask, obs_fluctuations = optree.tree_map(
            lambda x: np.pad(x, pad_width=((0, 0), (0, 0), (0, pad_width), (0, 0)), mode="constant", constant_values=0),
            (obs_times, obs_values, obs_mask, obs_fluctuations),
        )

    # always pad obs_fluctuations to T
    obs_fluctuations = np.pad(obs_fluctuations, pad_width=((0, 0), (0, 0), (0, 1), (0, 0)), mode="constant", constant_values=0)

    # reshape into paths
    obs_times, obs_values, obs_mask, obs_fluctuations = optree.tree_map(
        lambda x: x.reshape(1, -1, split_path_length, 1), (obs_times, obs_values, obs_mask, obs_fluctuations)
    )

    # optionally reset time per path to start at 0
    if reset_time_per_path is True:
        obs_times = obs_times - obs_times[:, :, 0, :][..., None, :]

    obs_mask = obs_mask.astype(bool)

    return {"obs_times": obs_times, "obs_values": obs_values, "obs_mask": obs_mask, "obs_fluctuations": obs_fluctuations}


if __name__ == "__main__":
    # set paths
    save_dir = Path("processed/test/20250114_preprocessed_wang_real_world")
    base_data_dir = Path("raw/SDE_data_driven_BISDE_datasets")

    # set split options
    split_path_length = 128
    reset_time_per_path = True

    # process paths
    if not base_data_dir.is_absolute():
        base_data_dir = Path(data_path) / base_data_dir

    if not save_dir.is_absolute():
        save_dir = Path(data_path) / save_dir

    wind_price_path = base_data_dir / "wind" / "wind_speeds.pickle"
    oil_price_path = base_data_dir / "oil" / "oil_prices.pickle"
    fb_price_path = base_data_dir / "stonks" / "fb_stock_price.pickle"
    tsla_price_path = base_data_dir / "stonks" / "tsla_stock_price.pickle"

    # process data
    datasets_paths = [wind_price_path, oil_price_path, fb_price_path, tsla_price_path]
    datasets_labels = ["wind", "oil", "fb_stock", "tsla_stock"]

    save_dir.mkdir(parents=True, exist_ok=True)

    for dataset_path, dataset_label in zip(datasets_paths, datasets_labels):
        dataset_save_dir: Path = save_dir / dataset_label

        # full trajectory as single path
        single_traj_data: dict = process_single_trajectory(dataset_path)
        save_arrays_from_dict(dataset_save_dir / "full_traj_single_path", single_traj_data)

        # full trajectory as split paths
        split_data: dict = split_trajectory_data(single_traj_data, split_path_length, reset_time_per_path)
        save_arrays_from_dict(dataset_save_dir / ("split_into_paths_of_length_" + str(split_path_length)), split_data)
