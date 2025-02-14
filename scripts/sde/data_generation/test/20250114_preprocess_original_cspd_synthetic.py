from pathlib import Path

import numpy as np
import optree

from fim import data_path
from fim.data_generation.sde.preprocess_utils import save_arrays_from_dict


def load_cspd_synthetic_data(file_path: Path) -> dict:
    """
    Load times and values from .npz files from CSPD data.
    """
    with open(file_path, "rb") as f:
        npz_file = np.load(f)
        times = npz_file["t"]  # [10.000, T, 1]
        values = npz_file["x"]  # [10.000, T, D]

    return {"obs_times": times, "obs_values": values}


if __name__ == "__main__":
    # set paths
    save_dir = Path("processed/test/20250114_preprocessed_original_SCPD_synthetic_data")
    data_dir = Path("raw/SDE_SCPD_synthetic_data_from_their_repo")

    split_number_of_paths = [50, 100, 500, 1000, 10000]

    process_names = ["cir", "ou", "sine", "lorenz", "predator_prey", "sink"]

    if not save_dir.is_absolute():
        save_dir = Path(data_path) / save_dir

    if not data_dir.is_absolute():
        data_dir = Path(data_path) / data_dir

    for process_name in process_names:
        data: tuple = load_cspd_synthetic_data(data_dir / (process_name + ".npz"))

        # split trajectories into batches with different number of paths
        for num_paths in split_number_of_paths:
            data_split = optree.tree_map(lambda x: x.reshape((-1, num_paths) + x.shape[1:]), data)
            save_arrays_from_dict(save_dir / ("num_paths_" + str(num_paths)) / process_name, data_split)
