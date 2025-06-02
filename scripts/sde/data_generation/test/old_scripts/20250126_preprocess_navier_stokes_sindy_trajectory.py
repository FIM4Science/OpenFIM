from pathlib import Path

import numpy as np
from scipy.io import loadmat

from fim import data_path
from fim.data_generation.sde.preprocess_utils import save_arrays_from_dict


def get_navier_stokes_data(path_to_pod_coeffs_mat: Path, path_to_pod_coeffs_run1_mat: Path):
    """
    Load and preprocess navier stokes trajectories from SINDy.
    Combine both simulations for training.
    Add (artificial) time grid.
    Also return each trajectory individually.

    Args:
        path to "PODcoefficients.mat"
        path to "PODcoefficients_run1.mat"
    """
    # adapted from PySindy documentation.
    dt = 0.02

    data_run1 = loadmat(path_to_pod_coeffs_mat)
    x_run1 = np.concatenate((data_run1["alpha"][:5000, :2], data_run1["alphaS"][:5000, 0:1]), axis=1)  # [5000, 3]
    t_run1 = np.arange(0, dt * x_run1.shape[0], dt).reshape(-1, 1)  # [5000, 1]
    mask_run1 = np.ones_like(t_run1)

    data_run2 = loadmat(path_to_pod_coeffs_run1_mat)
    x_run2 = np.concatenate((data_run2["alpha"][:3000, :2], data_run2["alphaS"][:3000, 0:1]), axis=1)  # [3000, 3]
    t_run2 = np.arange(0, dt * x_run2.shape[0], dt).reshape(-1, 1)  # [3000, 1]
    mask_run2 = np.ones_like(t_run2)

    # combine both trajectories
    x_run2 = np.pad(x_run2, pad_width=((0, 2000), (0, 0)), constant_values=0)
    t_run2 = np.pad(t_run2, pad_width=((0, 2000), (0, 0)), constant_values=0)
    mask_run2 = np.pad(mask_run2, pad_width=((0, 2000), (0, 0)), constant_values=0)

    obs_times = np.stack([t_run1, t_run2], axis=0)  # [2, 5000, 1]
    obs_values = np.stack([x_run1, x_run2], axis=0)  # [2, 5000, 3]
    obs_mask = np.stack([mask_run1, mask_run2], axis=0)  # [2 5000, 1]

    navier_stokes_data = {  # add first "batch" dimension for convenience
        "obs_times": obs_times[None],
        "obs_values": obs_values[None],
        "obs_mask": obs_mask[None].astype(bool),
    }

    return navier_stokes_data


if __name__ == "__main__":
    # set generation config
    base_data_dir = Path("raw/SDE_navier_stokes_sindy_trajectory")
    path_to_pod_coeffs_mat = base_data_dir / "PODcoefficients.mat"
    path_to_pod_coeffs_run1_mat = base_data_dir / "PODcoefficients_run1.mat"

    save_dir = Path("processed/test/20250126_preprocessed_navier_stokes_sindy_trajectory")

    # prepare paths
    if not path_to_pod_coeffs_mat.is_absolute():
        path_to_pod_coeffs_mat = Path(data_path) / path_to_pod_coeffs_mat

    if not path_to_pod_coeffs_run1_mat.is_absolute():
        path_to_pod_coeffs_run1_mat = Path(data_path) / path_to_pod_coeffs_run1_mat

    if not save_dir.is_absolute():
        save_dir = Path(data_path) / save_dir

    navier_stokes_data: dict = get_navier_stokes_data(path_to_pod_coeffs_mat, path_to_pod_coeffs_run1_mat)
    save_arrays_from_dict(save_dir, navier_stokes_data)
