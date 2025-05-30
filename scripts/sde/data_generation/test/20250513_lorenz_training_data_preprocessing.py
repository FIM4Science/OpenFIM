from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

from fim import data_path
from fim.data.utils import save_h5


def get_paths_data(lorenz_data: dict, num_paths: Optional[int] = None) -> tuple:
    """
    Unpacks and pre-processes Lorenz data from NeuralSDE.

    Args:
        lorenz_data (dict): Data of NeuralSDE Lorenz system loaded from pth.
        num_paths (int): Number of paths to return. Defaults to all

    Returns:
        obs_times (torch.Tensor): Observation times. Shape: [1, P, T, 1]
        obs_values (torch.Tensor): Observation values. Shape: [1, P, T, 3]
    """

    obs_times = lorenz_data["ts"]  # [T]
    obs_values = lorenz_data["xs"]  # [T, P, 3]

    # preprocess to our convention of [batch, paths, T, D]
    T, P, _ = obs_values.shape
    obs_times = torch.broadcast_to(obs_times.reshape(1, 1, T, 1), (1, P, T, 1))  # [1, P, T, 1]
    obs_values = torch.swapaxes(obs_values, 0, 1).reshape(1, P, T, 3)

    # obs_values = obs_values[:, torch.randperm(obs_values.shape[1])]

    if num_paths is not None:
        obs_times = obs_times[:, :num_paths, :, :]
        obs_values = obs_values[:, :num_paths, :, :]

    return obs_times, obs_values


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    # set save paths
    save_dir = Path("processed/test")
    subdir_label = "lorenz_train_data"

    lorenz_data_linear_diffusion_pth = "/cephfs_projects/foundation_models/data/SDE/external_evaluations_and_data/20250512_lorenz_data_from_neural_sde_github/20250513_lorenz_data_linear_diffusion_from_neural_sde_github_setup_40_path_length.pth"
    lorenz_data_constant_diffusion_pth = "/cephfs_projects/foundation_models/data/SDE/external_evaluations_and_data/20250512_lorenz_data_from_neural_sde_github/20250514_lorenz_data_constant_diffusion_from_neural_sde_paper_setup_generated_with_adapted_github_code.pth"

    total_num_paths = [128, 256, 512, 1024]
    # --------------------------------------------------------------------------------------------------------------------------------- #
    if not save_dir.is_absolute():
        save_dir = Path(data_path) / save_dir

    # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
    time: str = str(datetime.now().strftime("%Y%m%d%H%M%S"))
    save_data_dir: Path = save_dir / (time + "_" + subdir_label)
    save_data_dir.mkdir(parents=True, exist_ok=True)

    # save path to training data
    with open(save_data_dir / "linear_diffusion_pth_file_path.txt", "w") as f:
        f.write(lorenz_data_linear_diffusion_pth)

    with open(save_data_dir / "constant_diffusion_pth_file_path.txt", "w") as f:
        f.write(lorenz_data_constant_diffusion_pth)

    # load pth
    linear_diffusion_data: dict = torch.load(lorenz_data_linear_diffusion_pth, map_location=torch.device("cpu"))
    constant_diffusion_data: dict = torch.load(lorenz_data_constant_diffusion_pth, map_location=torch.device("cpu"))

    # save pth for reference
    torch.save(linear_diffusion_data, save_data_dir / "linear_diffusion_lorenz_data.pth")
    torch.save(constant_diffusion_data, save_data_dir / "constant_diffusion_lorenz_data.pth")

    # preprocess and save data with multiple paths
    for num_paths in total_num_paths:
        for label, data in zip(["linear", "constant"], [linear_diffusion_data, constant_diffusion_data]):
            label_data_dir = save_data_dir / (label + "_diffusion_data")
            label_data_dir.mkdir(exist_ok=True, parents=True)

            num_paths_dir = label_data_dir / f"{num_paths}_paths"
            num_paths_dir.mkdir(exist_ok=True, parents=True)

            obs_times, obs_values = get_paths_data(data, num_paths)

            save_h5(obs_times, num_paths_dir / "obs_times.h5")
            save_h5(obs_values, num_paths_dir / "obs_values.h5")
