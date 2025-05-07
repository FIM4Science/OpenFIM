from pathlib import Path
from pprint import pprint

import numpy as np
import optree
from scipy.io import loadmat

from fim import data_path
from fim.data_generation.sde.preprocess_utils import save_arrays_from_dict


def load_mocap_35_data(path_to_mocap35: Path) -> list:
    """
    Load data from train (and optionally validation), compute pca, project train and test to 3 PCA components.

    Args:
        path_to_mocap35 (Path): Path to `mocap35.mat`

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
    val = data["Xval"]  # [3, 300, 50]

    # pca via svd; data is centralized
    u, s, vh = np.linalg.svd(train.reshape(-1, 50))
    eigenvalues = s**2  # [50]
    eigenvectors = np.transpose(np.conjugate(vh), axes=(-1, -2))  # [50, 50]

    pca_train = (train @ eigenvectors)[:, :, :3] / np.sqrt(eigenvalues)[None, None, :3]  # [X, 300, 3]
    pca_test = (test @ eigenvectors)[:, :, :3] / np.sqrt(eigenvalues)[None, None, :3]  # [4, 300, 3]
    pca_val = (val @ eigenvectors)[:, :, :3] / np.sqrt(eigenvalues)[None, None, :3]  # [3, 300, 3]

    reconst_from_pca_train = (pca_train * np.sqrt(eigenvalues)[:3]) @ np.transpose(eigenvectors, axes=(-1, -2))[:3, :]  # [X, 300, 50]
    reconst_from_pca_test = (pca_test * np.sqrt(eigenvalues)[:3]) @ np.transpose(eigenvectors, axes=(-1, -2))[:3, :]  # [4, 300, 50]
    reconst_from_pca_val = (pca_val * np.sqrt(eigenvalues)[:3]) @ np.transpose(eigenvectors, axes=(-1, -2))[:3, :]  # [3, 300, 50]

    mocap_35_data = {
        "high_dim": train,
        "high_dim_test": test,
        "high_dim_val": val,
        "obs_values": pca_train,
        "obs_values_test": pca_test,
        "obs_values_val": pca_val,
        "eigenvectors": eigenvectors,
        "eigenvalues": eigenvalues,
        "reconst_high_dim_from_pca": reconst_from_pca_train,
        "reconst_high_dim_from_pca_test": reconst_from_pca_test,
        "reconst_high_dim_from_pca_val": reconst_from_pca_val,
    }

    return mocap_35_data


def get_mocap_35_forecasting_data(path_to_mocap35: Path, last_context_index: int):
    """
    Mocap train and test data, including masks to evaluate forecasting at.
    """
    mocap_35_data = load_mocap_35_data(path_to_mocap35)

    # regular observation grid
    train_size = 16
    test_size = 4
    val_size = 3

    obs_grid_train = np.broadcast_to(np.arange(300).reshape(1, 300, 1), (train_size, 300, 1)).copy()
    obs_grid_test = np.broadcast_to(np.arange(300).reshape(1, 300, 1), (test_size, 300, 1)).copy()
    obs_grid_val = np.broadcast_to(np.arange(300).reshape(1, 300, 1), (val_size, 300, 1)).copy()
    mocap_35_data.update({"obs_grid": obs_grid_train, "obs_grid_test": obs_grid_test, "obs_grid_val": obs_grid_val})

    # inference grid includes last context value as initial state
    inference_grid_size = 300 - last_context_index

    last_context_value = mocap_35_data.get("obs_values_test")[:, last_context_index, :]  # [test_size, 3]
    paths_inference_grid = obs_grid_test[:, last_context_index:, :]  # [test_size, inference_grid_size, 1]
    mocap_35_data.update({"last_context_value_test": last_context_value, "paths_inference_grid": paths_inference_grid})

    # MSE computed on last 297 frames of test set, reference values shape: [test_size, inference_grid_size, 50]
    paths_metrics_high_dim_reference_values = mocap_35_data.get("high_dim_test")[:, last_context_index:, :]
    paths_metrics_mask = np.concatenate([np.zeros(3 - last_context_index), np.ones(297)], axis=0)
    paths_metrics_mask = (
        np.broadcast_to(paths_metrics_mask.reshape(1, inference_grid_size, 1), (test_size, inference_grid_size, 1)).copy().astype(bool)
    )
    mocap_35_data.update(
        {
            "paths_metrics_high_dim_reference_values": paths_metrics_high_dim_reference_values,
            "paths_metrics_mask": paths_metrics_mask,
        }
    )
    assert paths_metrics_high_dim_reference_values.shape[:-1] == paths_metrics_mask.shape[:-1] == paths_inference_grid.shape[:-1]

    # truncate reference value reconstruction from first 3 PCA dimensions
    mocap_35_data.update(
        {"reconst_high_dim_from_pca_test": mocap_35_data.get("reconst_high_dim_from_pca_test")[..., last_context_index:, :]}
    )

    # add mask for 297 forecasting values on obs_grid_test
    forecasting_mask_test = np.concatenate([np.zeros(3), np.ones(297)], axis=0)
    forecasting_mask_test = np.broadcast_to(forecasting_mask_test.reshape(1, 300, 1), (test_size, 300, 1)).copy().astype(bool)
    mocap_35_data.update({"forecasting_mask_test": forecasting_mask_test})

    # add batch dimension for convenient model application
    mocap_35_data = optree.tree_map(lambda x: x[None], mocap_35_data)

    return mocap_35_data


if __name__ == "__main__":
    # set generation config
    path_to_mocap_35 = Path("raw/SDE_mocap_35/mocap35.mat")
    save_dir = Path("processed/test/20250510_preprocessed_mocap35")

    # prepare paths
    if not path_to_mocap_35.is_absolute():
        path_to_mocap_35 = Path(data_path) / path_to_mocap_35

    if not save_dir.is_absolute():
        save_dir = Path(data_path) / save_dir

    for last_context_index in range(3):
        # pure train split as train data
        data = get_mocap_35_forecasting_data(path_to_mocap_35, last_context_index=last_context_index)
        save_arrays_from_dict(save_dir / ("last_context_index_" + str(last_context_index)), data)

        print("last_context_index_" + str(last_context_index))
        pprint(optree.tree_map(lambda x: x.shape, data))
        print("\n\n")
