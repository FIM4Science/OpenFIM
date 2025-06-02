import pickle
from pathlib import Path

import numpy as np

from fim import data_path
from fim.data_generation.sde.preprocess_utils import save_arrays_from_dict


def process_single_trajectory(file_path: Path, dt: float) -> dict:
    """
    Return data from single trajectory as one-path + batched data (s.t. it can be loaded by dataloader).

    Args:
        file_path: Path to .pickle containing array of shape [T].
        dt (float): dt between two observations, for artificial time grid.

    Returns:
        dict with keys:
            obs_times: Regular observation times in [0, max_time]. Shape: [1, 1, T, 1]
            obs_values: Content of file path. Shape: [1, 1, T, 1]
            obs_values_fluctuations: (Recomputed) fluctuation of obs_values. Shape: [1, 1, T-1, 1]
    """
    data = pickle.load(open(file_path, "rb"))  # [T]
    assert data.ndim == 1

    # remove nan values
    data = data[np.logical_not(np.isnan(data))]

    # Create artificial time
    T = data.shape[0]
    times = dt * np.arange(T)

    obs_times = times.reshape(1, 1, T, 1)
    obs_values = data.reshape(1, 1, T, 1)
    obs_times_fluctuations = obs_times[:, :, :-1, :]
    obs_values_fluctuations = obs_values[:, :, 1:, :] - obs_values[:, :, :-1, :]

    return {
        "obs_times": obs_times,
        "obs_values": obs_values,
        "obs_values_fluctuations": obs_values_fluctuations,
        "obs_times_fluctuations": obs_times_fluctuations,
    }


# def split_trajectory_data(single_traj_data: dict, split_path_length: int, reset_time_per_path: bool) -> dict:
#     """
#     Split pre-processed single trajectory data into paths.
#
#     Args:
#         single_traj_data (dict): Returned from `process_single_trajectory`.
#         split_path_length (int): (Max.) length of each path after splitting.
#         reset_time_per_path (bool): If true, resets first time of each path to 0.
#
#     Returns:
#         Similar to `process_single_trajectory`, but with non-unit path dimension.
#     """
#     obs_times = single_traj_data.get("obs_times")  # [1, 1, T, 1]
#     obs_values = single_traj_data.get("obs_values")  # [1, 1, T, 1]
#     obs_times_fluctuations = single_traj_data.get("obs_times_fluctuations")  # [1, 1, T-1, 1]
#     obs_values_fluctuations = single_traj_data.get("obs_values_fluctuations")  # [1, 1, T-1, 1]
#
#     T = obs_times.shape[-2]
#
#     obs_mask = np.ones((1, 1, T, 1)).astype(bool)
#
#     # if required: pad before reshaping
#     if T % split_path_length != 0:
#         pad_width = split_path_length - (T % split_path_length)
#         obs_times, obs_values, obs_mask, obs_times_fluctuations, obs_values_fluctuations = optree.tree_map(
#             lambda x: np.pad(x, pad_width=((0, 0), (0, 0), (0, pad_width), (0, 0)), mode="constant", constant_values=0),
#             (obs_times, obs_values, obs_mask, obs_times_fluctuations, obs_values_fluctuations),
#         )
#
#     # always pad obs_..._fluctuations to T
#     obs_values_fluctuations = np.pad(
#         obs_values_fluctuations, pad_width=((0, 0), (0, 0), (0, 1), (0, 0)), mode="constant", constant_values=0
#     )
#     obs_times_fluctuations = np.pad(obs_times_fluctuations, pad_width=((0, 0), (0, 0), (0, 1), (0, 0)), mode="constant", constant_values=0)
#
#     # reshape into paths
#     obs_times, obs_values, obs_mask, obs_times_fluctuations, obs_values_fluctuations = optree.tree_map(
#         lambda x: x.reshape(1, -1, split_path_length, 1), (obs_times, obs_values, obs_mask, obs_times_fluctuations, obs_values_fluctuations)
#     )
#
#     # optionally reset time per path to start at 0
#     if reset_time_per_path is True:
#         obs_times = obs_times - obs_times[:, :, 0, :][..., None, :]
#         obs_times_fluctuations = obs_times_fluctuations - obs_times_fluctuations[:, :, 0, :][..., None, :]
#
#     obs_mask = obs_mask.astype(bool)
#
#     return {
#         "obs_times": obs_times,
#         "obs_values": obs_values,
#         "obs_mask": obs_mask,
#         "obs_values_fluctuations": obs_values_fluctuations,
#         "obs_times_fluctuations": obs_times_fluctuations,
#     }
#
#
if __name__ == "__main__":
    # set paths
    save_dir = Path("processed/test/20250125_preprocessed_wang_real_world")
    base_data_dir = Path("/cephfs_projects/foundation_models/data/SDE/raw/BISDE_datasets")

    # # set split options
    # split_path_length = 128
    # reset_time_per_path = True

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
    # datasets_max_time = [6 / 12, 5 / 12, 3 / 12, 3 / 12]  # measured in years
    datasets_dt = [1 / 6, 1, 1 / (252 * 390), 1 / (252 * 390)]  # dt from BISDE code

    save_dir.mkdir(parents=True, exist_ok=True)

    for dataset_path, dataset_label, dataset_dt in zip(datasets_paths, datasets_labels, datasets_dt):
        dataset_save_dir: Path = save_dir / dataset_label

        # full trajectory as single path
        single_traj_data: dict = process_single_trajectory(dataset_path, dataset_dt)
        save_arrays_from_dict(dataset_save_dir / "full_traj_single_path", single_traj_data)

        # # full trajectory as split paths
        # split_data: dict = split_trajectory_data(single_traj_data, split_path_length, reset_time_per_path)
        # split_data: dict = optree.tree_map(lambda x: np.nan_to_num(x, nan=0), split_data)
        # save_arrays_from_dict(dataset_save_dir / ("split_into_paths_of_length_" + str(split_path_length)), split_data)
