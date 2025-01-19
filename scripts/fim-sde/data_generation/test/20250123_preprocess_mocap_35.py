from pathlib import Path

import numpy as np
import optree
from scipy.io import loadmat

from fim import data_path
from fim.data_generation.sde.preprocess_utils import save_arrays_from_dict


def load_mocap_35_data(path_to_mocap35: Path, include_val: bool) -> list:
    """
    Load data from train (and optionally validation), compute pca, project train and test to 3 PCA components.

    Args:
        path_to_mocap35 (Path): Path to `mocap35.mat`
        include_val (bool): If true, include validation in train set. (We don`t train, so don't need validation)

    Returns:
        Data from test and train, in high dim. and pca space.
    """
    # load data
    if not path_to_mocap35.is_absolute():
        path_to_mocap35: Path = Path(data_path) / path_to_mocap35

    with open(path_to_mocap35, "rb") as f:
        data = loadmat(f)

    train = data["Xtr"]  # [16, 300, 50]
    test = data["Xtest"]  # [4, 300, 50]

    if include_val is True:
        val = data["Xval"]  # [3, 300, 50]
        train = np.concatenate([train, val], axis=0)

    # pca via svd; data is centralized
    u, s, vh = np.linalg.svd(train.reshape(-1, 50))
    eigenvalues = s**2  # [50]
    eigenvectors = np.transpose(np.conjugate(vh), axes=(-1, -2))  # [50, 50]

    pca_train = (train @ eigenvectors)[:, :, :3] / np.sqrt(eigenvalues)[None, None, :3]  # [X, 300, 3]
    pca_test = (test @ eigenvectors)[:, :, :3] / np.sqrt(eigenvalues)[None, None, :3]  # [4, 300, 3]

    reconst_from_pca_train = (pca_train * np.sqrt(eigenvalues)[:3]) @ np.transpose(eigenvectors, axes=(-1, -2))[:3, :]  # [X, 300, 50]
    reconst_from_pca_test = (pca_test * np.sqrt(eigenvalues)[:3]) @ np.transpose(eigenvectors, axes=(-1, -2))[:3, :]  # [4, 300, 50]

    return train, test, pca_train, pca_test, eigenvectors, eigenvalues, reconst_from_pca_train, reconst_from_pca_test


def get_mocap_35_forecasting_data(path_to_mocap35: Path, include_val: bool):
    """
    Mocap train and test data, including masks to evaluate forecasting at.
    """
    (
        high_dim_train,
        high_dim_test,
        pca_train,
        pca_test,
        eigenvectors,
        eigenvalues,
        reconst_from_pca_train,
        reconst_from_pca_test,
    ) = load_mocap_35_data(path_to_mocap35, include_val=include_val)

    # MSE computed on 297 frames of test set
    forecasting_mask = np.concatenate([np.zeros(3), np.ones(297)], axis=0)
    forecasting_mask = np.broadcast_to(forecasting_mask.reshape(1, 300, 1), (4, 300, 1)).copy().astype(bool)

    # solve from 2nd observation
    last_context_value = pca_test[:, 2, :]  # [4, 3]
    inference_mask = np.concatenate([np.zeros(2), np.ones(298)], axis=0)
    inference_mask = np.broadcast_to(inference_mask.reshape(1, 300, 1), (4, 300, 1)).copy().astype(bool)

    # regular observation grid
    B = 16 if include_val is False else 19
    observation_grid_train = np.broadcast_to(np.arange(300).reshape(1, 300, 1), (B, 300, 1)).copy()
    observation_grid_test = np.broadcast_to(np.arange(300).reshape(1, 300, 1), (4, 300, 1)).copy()

    data = {
        "high_dim_train": high_dim_train,
        "obs_values_train": pca_train,
        "reconst_from_pca_train": reconst_from_pca_train,
        "obs_grid_train": observation_grid_train,
        "high_dim_test": high_dim_test,
        "obs_values_test": pca_test,
        "reconst_from_pca_test": reconst_from_pca_test,
        "obs_grid_test": observation_grid_test,
        "last_context_value_test": last_context_value,
        "eigenvectors": eigenvectors,
        "eigenvalues": eigenvalues,
        "forecasting_mask": forecasting_mask,
        "inference_mask": inference_mask,
    }

    # add batch dimension
    data = optree.tree_map(lambda x: x[None], data)

    return data


if __name__ == "__main__":
    # set generation config
    path_to_mocap_35 = Path("raw/SDE_mocap_35/mocap35.mat")
    save_dir = Path("processed/test/20250123_preprocessed_mocap35")

    # prepare paths
    if not path_to_mocap_35.is_absolute():
        path_to_mocap_35 = Path(data_path) / path_to_mocap_35

    if not save_dir.is_absolute():
        save_dir = Path(data_path) / save_dir

    # pure train split as train data
    pure_train_data = get_mocap_35_forecasting_data(path_to_mocap_35, include_val=False)
    save_arrays_from_dict(save_dir / "pure_train_data", pure_train_data)

    # add validation to train
    train_and_val_data = get_mocap_35_forecasting_data(path_to_mocap_35, include_val=True)
    save_arrays_from_dict(save_dir / "train_and_val_data", train_and_val_data)
