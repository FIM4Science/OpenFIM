from pathlib import Path

import numpy as np
import torch
from sklearn.utils.extmath import randomized_svd

from fim.data.utils import save_h5


def save_arrays_from_dict(save_dir: Path, arrays: dict[str, np.array]) -> None:
    """
    Save arrays from dict as h5 with keys as file names.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    for key, array in arrays.items():
        save_h5(torch.from_numpy(array), save_dir / (key + ".h5"))


class RandomizedSVD:
    "From https://api-depositonce.tu-berlin.de/server/api/core/bitstreams/2aeebc0d-c451-4171-9e24-3480c69b76d6/content"

    def __init__(self, components_count: 3) -> None:
        self.components_count = components_count

    def train_pca(self, train_data: np.array) -> dict:
        "train_data has to be centered, i.e. have mean = 0"
        T = train_data.shape[-2]

        L, sigma, R_conjugate_transpose = randomized_svd(
            train_data / np.sqrt(T - 1), n_components=self.components_count, random_state=None
        )  # [..., T, T], [..., T, D], [..., D, D]

        R = np.transpose(np.conjugate(R_conjugate_transpose), axes=(-1, -2))
        eigenvalues = sigma**2

        pca_params = {
            "left_eigenvectors": L,
            "right_eigenvectors": R,
            "eigenvalues": eigenvalues,
        }

        return pca_params

    def _unpack_pca_params(self, pca_params: dict) -> tuple:
        left_eigenvectors = pca_params.get("left_eigenvectors")
        right_eigenvectors = pca_params.get("right_eigenvectors")
        eigenvalues = pca_params.get("eigenvalues")
        return eigenvalues, left_eigenvectors, right_eigenvectors

    def get_time_coefficients(self, pca_params: dict, apply_data: np.array) -> dict:
        "apply_data has to be centered, i.e. have mean = 0"
        eigenvalues, left_eigenvectors, right_eigenvectors = self._unpack_pca_params(pca_params)

        # create snapshot matrix
        time_coefficients = apply_data @ right_eigenvectors  # [..., T, D]
        return time_coefficients

    def reconstruct_from_time_coefficients(self, pca_params: dict, time_coefficients: np.array) -> np.array:
        "return reconstruction based on the number of time coefficients provided"
        # time_coefficients [..., T, components_count] with components_count <= D
        eigenvalues, left_eigenvectors, right_eigenvectors = self._unpack_pca_params(pca_params)

        inverse_base_change = np.transpose(right_eigenvectors, axes=(-1, -2))

        # select base change based on number of provided time coefficients
        inverse_base_change = inverse_base_change[..., : self.components_count, :]  # [..., components_count, D]
        time_coefficients = time_coefficients[..., : self.components_count]  # [..., components_count, D]
        reconstruction = time_coefficients @ inverse_base_change  # [..., T, D]

        # adjust for centering of data
        reconstruction = reconstruction + pca_params.get("data_mean")

        return reconstruction
