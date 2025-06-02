import pickle
from pathlib import Path

import numpy as np
import optree

from fim import data_path
from fim.data_generation.sde.preprocess_utils import RandomizedSVD, save_arrays_from_dict


def load_navier_stokes_data(vortex_path: Path) -> dict:
    """
    Load data from vortex into a dict.

    Args:
        vortex_path (Path): Path to `vortex.pkl` provided by SVISE repo (Course 2023).

    Returns: dict with keys
        u, v: Shape: [3051, 199, 1499]
        u_and_v: Shape: [3051, 199, 1499, 2]
        time: Shape: [3051, 1]
    """
    if not vortex_path.is_absolute():
        vortex_path: Path = Path(data_path) / vortex_path

    with open(vortex_path, "rb") as f:
        data = pickle.load(f)  # data["x"] = [..., 5] with x, y, u, v, vorticity

    u = data["x"][..., 2]
    v = data["x"][..., 3]

    x_dim = 199
    y_dim = 1499

    (u, v) = optree.tree_map(lambda x: x.reshape(-1, x_dim, y_dim), (u, v))  # [3051, 199, 1499]

    data = {"u": u, "v": v, "u_and_v": np.stack([u, v], axis=-1)}

    steps = u.shape[0]
    time = np.linspace(0, 305, steps, endpoint=False)  # yiels exactly times from paper, after warmup and test removal
    data["time"] = time.reshape(-1, 1)

    return data


def get_pca_params(u_and_v: np.array) -> dict:
    """
    Get randomized svd parameters from navier stokes data.

    Args:
        u_and_v (Array): Shape: [T, 199, 1499, 2]

    Returns: dict with pca parameters and keys:
        left_eigenvectors: Shape: [T, 3]
        right_eigenvectors: Shape: [596602, 3]
        eigenvalues: Shape: [3]
        data_mean: Shape: [1, 596602]
    """
    assert u_and_v.ndim == 4

    # collaps data dimensions
    T = u_and_v.shape[0]
    u_and_v = u_and_v.reshape(T, -1)

    # need centered data (over time dimension)
    u_and_v_mean = np.mean(u_and_v, axis=-2, keepdims=True)
    centered_u_and_v = u_and_v - np.broadcast_to(u_and_v_mean, u_and_v.shape)

    # apply randomized svd
    svd = RandomizedSVD(components_count=3)
    pca_params = svd.train_pca(centered_u_and_v)

    # save mean for reconstruction
    pca_params["data_mean"] = u_and_v_mean

    return pca_params


def apply_pca(data: np.array, pca_params: dict) -> np.array:
    """
    Project data with randomized pca params.

    Args:
        data (Array): Data to project. Shape: [T, 199, 1499, 2]
        pca_params (dict): Returned from `get_pca_params`.

    Returns:
        projected_data (Array). Shape: [T, 3]

    """
    # center data from pca params
    T = data.shape[0]
    data = data.reshape(T, -1)
    centered_data = data - np.broadcast_to(pca_params["data_mean"], data.shape)

    # apply projection with predefinde parameters
    svd = RandomizedSVD(components_count=3)
    projected_data = svd.get_time_coefficients(pca_params, centered_data)

    return projected_data


def preprocess_navier_stokes(vortex_path: Path, save_dir: Path) -> None:
    """
    Preprocessing and train-test split from 'State estimation of a physical system with unknown governing equations' (Course, 2023),
    but with 3 pca dimensions.

    Args:
        vortex_path (Path): Path to 'vortex.pkl', provided by (Course, 2023).
        save_dir (Path): Path to directory to save train and test splits in.
    """

    # load and data
    data = load_navier_stokes_data(vortex_path)

    # remove warmup like Course 2023
    warmup_percentage = 0.2
    warmup_index = int(data["u"].shape[0] * warmup_percentage)
    data = optree.tree_map(lambda x: x[warmup_index:], data)  # [2441, 199, 1499, ...]

    # split trajectory into train and test like Course 2023
    test_percentage = 0.2
    test_index = -int(data["u"].shape[0] * test_percentage)
    train_data: dict = optree.tree_map(lambda x: x[:test_index], data)  # [1953, 199, 1499, ...]
    test_data: dict = optree.tree_map(lambda x: x[test_index:], data)  # [488, 199, 1499, ...]

    # project to 3 pca dimensions
    pca_params: dict = get_pca_params(train_data["u_and_v"])
    train_pca_trajectory = apply_pca(train_data["u_and_v"], pca_params)  # [1953, 3]
    test_pca_trajectory = apply_pca(test_data["u_and_v"], pca_params)  # [488, 3]

    # build train and test data to save
    train_data = {  # T = 1953
        "obs_times": train_data["time"],  # [T, 1]
        "obs_values_pca": train_pca_trajectory,  # [T, 3]
        "obs_values_high_dim": train_data["u_and_v"],  # [T, 199, 1499, 2]
        "pca_left_eigenvectors": pca_params["left_eigenvectors"],  # [T, 3]
        "pca_right_eigenvectors": pca_params["right_eigenvectors"],  # [596602, 3]
        "pca_eigenvalues": pca_params["eigenvalues"],  # [3]
        "pca_data_mean": pca_params["data_mean"],  # [1, 596602]
    }

    test_data = {  # T = 488
        "obs_times": test_data["time"],
        "obs_values_pca": test_pca_trajectory,
        "obs_values_high_dim": test_data["u_and_v"],
        "pca_left_eigenvectors": pca_params["left_eigenvectors"],
        "pca_right_eigenvectors": pca_params["right_eigenvectors"],
        "pca_eigenvalues": pca_params["eigenvalues"],
        "pca_data_mean": pca_params["data_mean"],
    }

    # add single batch and path dimension for easier model application
    train_data, test_data = optree.tree_map(lambda x: x[None, None, :], (train_data, test_data))

    # save data
    save_arrays_from_dict(save_dir / "train", train_data)
    save_arrays_from_dict(save_dir / "test", test_data)


if __name__ == "__main__":
    vortex_path = Path("/cephfs_projects/foundation_models/data/SDE/raw/vortex.pkl")
    save_dir = Path("")  # full path to a dir

    preprocess_navier_stokes(vortex_path, save_dir)
