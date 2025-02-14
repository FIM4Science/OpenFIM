from pathlib import Path

from fim.data_generation.sde.dynamical_systems_to_files import save_dynamical_system_from_yaml


def get_600k_100_paths_half_noisy(project_path: Path, data_path: Path):
    yaml_path = project_path / Path("configs/data_generation/sde/20250117_600k_drift_deg_3_diffusion-deg_2_100_paths_half_with_noise.yaml")
    save_dir = data_path / Path("600k_polynomials_drift_deg_3_diffusion_deg_2_100_paths_half_with_noise")
    labels_to_use = ["train", "test", "validation"]
    return yaml_path, labels_to_use, save_dir


def get_600k_100_paths_half_noisy_set_2(project_path: Path, data_path: Path):
    yaml_path = project_path / Path(
        "configs/data_generation/sde/20250120_600k_drift_deg_3_diffusion-deg_2_100_paths_half_with_noise_set_2.yaml"
    )
    save_dir = data_path / Path("600k_polynomials_drift_deg_3_diffusion_deg_2_100_paths_half_with_noise_set_2")
    labels_to_use = ["train", "test", "validation"]
    return yaml_path, labels_to_use, save_dir


def get_600k_100_paths_half_noisy_with_delta_tau_1e_1_to_1e_3(project_path: Path, data_path: Path):
    yaml_path = project_path / Path(
        "configs/data_generation/sde/20250220_600k_drift_deg_3_diffusion_deg_2_100_paths_half_with_noise_delta_tau_1e-1_to_1e-3.yaml"
    )
    save_dir = data_path / Path("600k_polynomials_drift_deg_3_diffusion_deg_2_100_paths_half_with_noise_delta_tau_1e-1_to_1e-3")
    labels_to_use = ["train", "test", "validation"]
    return yaml_path, labels_to_use, save_dir


if __name__ == "__main__":
    project_path = Path("/home/seifner/repos/FIM/")
    data_path = Path("/home/seifner/repos/FIM/")

    # project_path = Path("/home/seifnerp_hpc/repos/FIM/")
    # data_path = Path("/lustre/scratch/data/seifnerp_hpc-fim_data/data/processed/train/")
    # tr_save_dir = data_path / "save_dynamical_system_tr"

    # project_path = Path("/Users/patrickseifner/repos/FIM")
    # data_path = Path("/Users/patrickseifner/repos/FIM")
    tr_save_dir = data_path / "save_dynamical_system_tr"
    tr_save_dir.mkdir(exist_ok=True)

    # print("600k, half noisy")
    # yaml_path, labels_to_use, save_dir = get_600k_100_paths_half_noisy(project_path, data_path)
    # save_dynamical_system_from_yaml(yaml_path, labels_to_use, save_dir)

    # print("600k, half noisy set 2")
    # yaml_path, labels_to_use, save_dir = get_600k_100_paths_half_noisy_set_2(project_path, data_path)
    # save_dynamical_system_from_yaml(yaml_path, labels_to_use, save_dir, tr_save_dir=tr_save_dir)

    print("600k, half noisy, delta tau 1e-1 to 1e-3")
    yaml_path, labels_to_use, save_dir = get_600k_100_paths_half_noisy_with_delta_tau_1e_1_to_1e_3(project_path, data_path)
    save_dynamical_system_from_yaml(yaml_path, labels_to_use, save_dir, tr_save_dir=tr_save_dir)
