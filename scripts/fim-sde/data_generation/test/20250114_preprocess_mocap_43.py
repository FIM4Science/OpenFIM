import math
from pathlib import Path

import numpy as np
import optree
from scipy.io import loadmat

from fim import data_path
from fim.data_generation.sde.preprocess_utils import save_arrays_from_dict


def load_mocap_43_data(path_to_mocap43: Path) -> list:
    """
    Load data from all trajectories and their projection to the first 3 PCA dimensions.
    """
    # load data
    if not path_to_mocap43.is_absolute():
        path_to_mocap43: Path = Path(data_path) / path_to_mocap43

    with open(path_to_mocap43, "rb") as f:
        data = loadmat(f)

    def _load_and_squeeze(data_key: str):
        values = data[data_key]
        return [v[0] for v in values]

    obs = _load_and_squeeze("Ys")  # list of lengt 43, each array of shape [T_obs, 50]
    eigenvectors = _load_and_squeeze("us")
    eigenvalues = _load_and_squeeze("vs")

    # downsample by strides of length 4 as Wang et al. (2008)
    obs = [traj[::4] for traj in obs]

    # project to first 3 pca dimensions
    pca_space_trajectory = [
        (traj @ eigenvec)[:, :3] / np.sqrt(eigenval).squeeze()[:3] for (traj, eigenvec, eigenval) in zip(obs, eigenvectors, eigenvalues)
    ]

    # reconstruct from first 3 pca dimensions
    obs_reconst_from_pca_space_trajectory = [
        traj * np.sqrt(eigenval).squeeze()[:3] @ (np.transpose(eigenvec, axes=(1, 0))[:3, :])
        for (traj, eigenvec, eigenval) in zip(pca_space_trajectory, eigenvectors, eigenvalues)
    ]

    return obs, eigenvectors, eigenvalues, pca_space_trajectory, obs_reconst_from_pca_space_trajectory


def get_imputation_mask(traj: np.array, imputation_percentage: float):
    """
    Mask imputation_percentage points from the middle of a trajectory.

    Args:
        traj (Array): Trajectory to base mask on. Shape: [traj_length, D]
        imputation_percentage (float): Percentage of traj_length to mask for imputation.

    Returns:
        imputation_mask (Array): 1s at imputation points in middle of trajectory. Shape: [traj_length, 1]
        before_imputation_mask (Array): 1s at observed points before imputation points. Shape: [traj_length, 1]
        after_imputation_mask (Array): 1s at observed points after imputation points. Shape: [traj_length, 1]
    """
    # half_percentage before and after middle index of trajectory
    half_imputation_percentage = imputation_percentage / 2
    traj_length = traj.shape[0]

    half_masking_indices_range = int(half_imputation_percentage * traj_length)
    middle_index = traj_length // 2

    before_imputation_length = middle_index - half_masking_indices_range
    after_imputation_length = traj.shape[0] - (middle_index + half_masking_indices_range)
    imputation_length = 2 * half_masking_indices_range

    # mask with 1s at the imputation points
    imputation_mask = np.concatenate(
        [
            np.zeros(before_imputation_length),
            np.ones(imputation_length),
            np.zeros(after_imputation_length),
        ],
        axis=0,
    ).reshape(-1, 1)

    # masks with 1s before the imputation points
    before_imputation_mask = np.concatenate(
        [np.ones(before_imputation_length), np.zeros(traj_length - before_imputation_length)], axis=0
    ).reshape(-1, 1)

    # mask with 1s after the imputation points
    after_imputation_mask = np.concatenate(
        [np.zeros(traj_length - after_imputation_length), np.ones(after_imputation_length)], axis=0
    ).reshape(-1, 1)

    return imputation_mask, before_imputation_mask, after_imputation_mask


def get_mocap_imputation_data(mocap43_path: Path, imputation_percentage: float) -> None:
    """
    Process CMU 43 with imputation region in the middle of each trajectory.

    Args:
        mocap43_path (Path): Path to 'mocap43.mat'
        imputation_percentage (float): Percentage of observations per trajectory to use as imputation targets.

    Returns:
        imputation_mocap_data (dict): Contains padded, batched and masked mocap trajectory data.
    """

    # load trajectories in list with arrays of different lengths
    (
        data_space_trajectory_list,
        eigenvectors_list,
        eigenvalues_list,
        pca_space_trajectory_list,
        data_space_from_pca_space_trajectory_list,
    ) = load_mocap_43_data(mocap43_path)

    max_len = max([len(traj) for traj in data_space_trajectory_list])

    # Get masks for each trajectory
    all_imputation_masks = [get_imputation_mask(traj, imputation_percentage) for traj in data_space_trajectory_list]
    imputation_masks, before_imputation_masks, after_imputation_masks = zip(*all_imputation_masks)

    # observation mask is everything but imputation mask
    observation_masks = [np.logical_not(mask.astype(bool)) for mask in imputation_masks]

    # observation grids end with 1 at last observation
    observation_grid = [
        np.linspace(start=0, stop=traj.shape[0], num=traj.shape[0]).reshape(-1, 1) / traj.shape[0] for traj in data_space_trajectory_list
    ]

    # pad and stack masks from all trajectories
    def _pad_masks_to_max_len(traj):
        return np.pad(traj, [(0, max_len - traj.shape[0]), (0, 0)], mode="constant", constant_values=False)

    all_masks = imputation_masks, observation_masks, before_imputation_masks, after_imputation_masks
    all_masks = optree.tree_map(_pad_masks_to_max_len, all_masks)
    all_masks = [np.stack(x, axis=0).astype(bool) for x in all_masks]
    imputation_masks, observation_masks, before_imputation_masks, after_imputation_masks = all_masks

    # pad and stack obs from all trajectories
    def _pad_obs_to_max_len(traj):
        return np.pad(traj, [(0, max_len - traj.shape[0]), (0, 0)], mode="constant", constant_values=0)

    all_obs_data = (observation_grid, data_space_trajectory_list, pca_space_trajectory_list, data_space_from_pca_space_trajectory_list)
    all_obs_data = optree.tree_map(_pad_obs_to_max_len, all_obs_data)
    all_obs_data = [np.stack(x, axis=0) for x in all_obs_data]
    observation_grid, data_space_trajectory, pca_space_trajectory, data_space_from_pca_space_trajectory = all_obs_data

    # prepare pca parameters
    eigenvalues = np.stack(eigenvalues_list, axis=0).squeeze()
    eigenvectors = np.stack(eigenvectors_list, axis=0).squeeze()

    imputation_mocap_data = {  # B=43, T = 125
        "observation_grid": observation_grid,  # [B, T, 1]
        "observation_values": pca_space_trajectory,  # [B, T, 3]
        "observation_mask": observation_masks,  # [B, T, 1]
        "imputation_mask": imputation_masks,  # [B, T, 1]
        "before_imputation_mask": before_imputation_masks,  # [B, T, 1]
        "after_imputation_mask": after_imputation_masks,  # [B, T, 1]
        "eigenvectors": eigenvectors,  # [B, 50, 50]
        "eigenvalues": eigenvalues,  # [B, 50]
        "high_dim_trajectory": data_space_trajectory,  # [B, T, 50]
        "high_dim_reconst_from_3_pca": data_space_from_pca_space_trajectory,  # [B, T, 50]
    }

    return imputation_mocap_data


def get_forecasting_mask(traj: np.array, forecasting_percentage: float):
    """
    Mask forecasting_percentage points from the end of a trajectory.

    Args:
        traj (Array): Trajectory to base mask on. Shape: [traj_length, D]
        forecasting_percentage (float): Percentage of traj_length to mask for forecasting.

    Returns:
        forecasting_mask (Array): 1s at forecasting points at the of trajectory. Shape: [traj_length, 1]
    """
    traj_length = traj.shape[0]

    num_forecasting_points = math.ceil(traj_length * forecasting_percentage)
    num_context_points = traj_length - num_forecasting_points

    forecasting_mask = np.concatenate([np.zeros(num_context_points), np.ones(num_forecasting_points)], axis=0).reshape(-1, 1)

    return forecasting_mask


def get_mocap_forecasting_data(mocap43_path: Path, forecasting_percentage: float) -> None:
    """
    Process CMU 43 with forecasting region at the end of each trajectory.

    Args:
        mocap43_path (Path): Path to 'mocap43.mat'
        forecasting_percentage (float): Percentage of observations per trajectory to use as forecasting targets.

    Returns:
        forecasting_mocap_data (dict): Contains padded, batched and masked mocap trajectory data.
    """

    # load trajectories in list with arrays of different lengths
    (
        data_space_trajectory_list,
        eigenvectors_list,
        eigenvalues_list,
        pca_space_trajectory_list,
        data_space_from_pca_space_trajectory_list,
    ) = load_mocap_43_data(mocap43_path)

    max_len = max([len(traj) for traj in data_space_trajectory_list])

    # Get masks for each trajectory
    all_forecasting_masks = [get_forecasting_mask(traj, forecasting_percentage) for traj in data_space_trajectory_list]

    # observation mask is everything but forecasting mask
    observation_masks = [np.logical_not(mask.astype(bool)) for mask in all_forecasting_masks]

    # observation grids end with 1 at last observation
    observation_grid = [
        np.linspace(start=0, stop=traj.shape[0], num=traj.shape[0]).reshape(-1, 1) / traj.shape[0] for traj in data_space_trajectory_list
    ]

    # pad and stack masks from all trajectories
    def _pad_masks_to_max_len(traj):
        return np.pad(traj, [(0, max_len - traj.shape[0]), (0, 0)], mode="constant", constant_values=False)

    all_masks = all_forecasting_masks, observation_masks
    all_masks = optree.tree_map(_pad_masks_to_max_len, all_masks)
    all_masks = [np.stack(x, axis=0).astype(bool) for x in all_masks]
    forecasting_masks, observation_masks = all_masks

    # pad and stack obs from all trajectories
    def _pad_obs_to_max_len(traj):
        return np.pad(traj, [(0, max_len - traj.shape[0]), (0, 0)], mode="constant", constant_values=0)

    all_obs_data = (observation_grid, data_space_trajectory_list, pca_space_trajectory_list, data_space_from_pca_space_trajectory_list)
    all_obs_data = optree.tree_map(_pad_obs_to_max_len, all_obs_data)
    all_obs_data = [np.stack(x, axis=0) for x in all_obs_data]
    observation_grid, data_space_trajectory, pca_space_trajectory, data_space_from_pca_space_trajectory = all_obs_data

    # prepare pca parameters
    eigenvalues = np.stack(eigenvalues_list, axis=0).squeeze()
    eigenvectors = np.stack(eigenvectors_list, axis=0).squeeze()

    forecasting_mocap_data = {  # B=43, T = 125
        "observation_grid": observation_grid,  # [B, T, 1]
        "observation_values": pca_space_trajectory,  # [B, T, 3]
        "observation_mask": observation_masks,  # [B, T, 1]
        "forecasting_mask": forecasting_masks,  # [B, T, 1]
        "eigenvectors": eigenvectors,  # [B, 50, 50]
        "eigenvalues": eigenvalues,  # [B, 50]
        "high_dim_trajectory": data_space_trajectory,  # [B, T, 50]
        "high_dim_reconst_from_3_pca": data_space_from_pca_space_trajectory,  # [B, T, 50]
    }

    return forecasting_mocap_data


if __name__ == "__main__":
    # set generation config
    mocap43_path = Path("/Users/patrickseifner/repos/FIM/data/raw/SDE_mocap_43/mocap43.mat")
    save_dir = Path("/Users/patrickseifner/repos/FIM/data/processed/test/20250115_preprocessed_mocap43")

    imputation_percentages = [0.2, 0.15, 0.10, 0.03]
    forecasting_percentages = [0.5, 0.2]

    # prepare paths
    if not mocap43_path.is_absolute():
        mocap43_path = Path(data_path) / mocap43_path

    if not save_dir.is_absolute():
        save_dir = Path(data_path) / save_dir

    # preprocess for imputation
    for imputation_percentage in imputation_percentages:
        imputation_data: dict = get_mocap_imputation_data(mocap43_path, imputation_percentage)

        # each trajectory as own batch element with 1 path
        single_path_imputation_data = optree.tree_map(lambda x: x[:, None], imputation_data)
        save_arrays_from_dict(
            save_dir / ("imputation_" + str(imputation_percentage) + "_perc") / "single_path_per_element", single_path_imputation_data
        )

        # all trajectories as one element with 50 paths
        all_paths_imputation_data = optree.tree_map(lambda x: x[None], imputation_data)
        save_arrays_from_dict(
            save_dir / ("imputation_" + str(imputation_percentage) + "_perc") / "all_paths_one_element", all_paths_imputation_data
        )

    # preprocess for forecasting
    for forecasting_percentage in forecasting_percentages:
        forecasting_data: dict = get_mocap_forecasting_data(mocap43_path, forecasting_percentage)

        # each trajectory as own batch element with 1 path
        single_path_forecasting_data = optree.tree_map(lambda x: x[:, None], forecasting_data)
        save_arrays_from_dict(
            save_dir / ("forecasting_" + str(forecasting_percentage) + "_perc") / "single_path_per_element", single_path_forecasting_data
        )

        # all trajectories as one element with 50 paths
        all_paths_forecasting_data = optree.tree_map(lambda x: x[None], forecasting_data)
        save_arrays_from_dict(
            save_dir / ("forecasting_" + str(forecasting_percentage) + "_perc") / "all_paths_one_element", all_paths_forecasting_data
        )
