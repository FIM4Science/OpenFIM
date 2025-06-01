from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from fim import project_path
from fim.data.utils import load_h5
from fim.models.sde import FIMSDE
from fim.utils.sde.evaluation import save_fig


def delta_times_histogram(train_set_dir: Path, save_dir: Path):
    """
    Save histograms of delta times.

    Args:
        train_set_dir (Path): Directory that contains many subdirs, each containing obs_times.h5 and obs_mask.h5
        save_dir (Path): Path to dir in which to save histogram in.
    """
    subdirs = [subdir for subdir in train_set_dir.iterdir() if subdir.is_dir()]

    subdirs = subdirs[:2]

    all_delta_times = []

    for subdir in tqdm(subdirs, total=len(subdirs), desc="Subdir delta times processed"):
        data = {
            "obs_values": load_h5(subdir / "obs_values.h5"),
            "obs_times": load_h5(subdir / "obs_times.h5"),
            "obs_mask": load_h5(subdir / "obs_mask.h5").bool(),
        }

        # fill masked values backwards, now all masked values will have delta time ==  0
        obs_times, obs_values, obs_mask = FIMSDE._fill_masked_values(data)

        delta_times = obs_times[:, :, 1:, :] - obs_times[:, :, :-1, :]  # obs_times are backward filled
        delta_times = delta_times.reshape(-1)

        all_delta_times.append(delta_times[delta_times != 0.0])

    all_delta_times = torch.concat(all_delta_times)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    ax.hist(all_delta_times, bins=len(all_delta_times.unique()), log=True)

    save_fig(fig, save_dir, "delta_times_histogram")
    plt.close(fig)


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    description = "train_set_statistics"
    experiment_descr = "develop"
    train_set_dir = Path(
        "/cephfs_projects/foundation_models/data/SDE/train/202501XX_icml_submission_model_600k_polynomials_drift_deg_3_diffusion_deg_2_100_paths_half_with_noise/train"
    )
    # --------------------------------------------------------------------------------------------------------------------------------- #

    # Save dir setup: project_path / evaluations / train_set_statistics / time_stamp + _ + experiment_descr
    evaluation_path = Path(project_path) / "evaluations"
    time: str = str(datetime.now().strftime("%m%d%H%M"))
    evaluation_dir: Path = evaluation_path / description / (time + "_" + experiment_descr)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    delta_times_histogram(train_set_dir, evaluation_dir)
