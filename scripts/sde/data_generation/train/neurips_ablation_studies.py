from pathlib import Path

from fim.data_generation.sde.dynamical_systems_to_files import save_dynamical_system_from_yaml


def get_100k_100_paths_half_noisy(project_path: Path, data_path: Path):
    yaml_path = project_path / Path("configs/data_generation/sde/20250313_100k_drift_deg_3_diffusion_deg_2_100_paths_half_with_noise.yaml")
    save_dir = data_path / Path("100k_polynomials_drift_deg_3_diffusion_deg_2_100_paths_half_with_noise")
    # labels_to_use = ["train", "test", "validation"]
    labels_to_use = ["train"]
    return yaml_path, labels_to_use, save_dir


def get_30k_100_paths_half_noisy(project_path: Path, data_path: Path):
    yaml_path = project_path / Path("configs/data_generation/sde/20250313_30k_drift_deg_3_diffusion_deg_2_100_paths_half_with_noise.yaml")
    save_dir = data_path / Path("30k_polynomials_drift_deg_3_diffusion_deg_2_100_paths_half_with_noise")
    labels_to_use = ["train", "test", "validation"]
    return yaml_path, labels_to_use, save_dir


def get_deg_4_drift_30k_100_paths_half_noisy(project_path: Path, data_path: Path):
    yaml_path = project_path / Path("configs/data_generation/sde/20250314_30k_drift_deg_4_diffusion_deg_2_100_paths_half_with_noise.yaml")
    save_dir = data_path / Path("30k_polynomials_drift_deg_4_diffusion_deg_2_100_paths_half_with_noise")
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

    # print("100k, half noisy")
    # yaml_path, labels_to_use, save_dir = get_100k_100_paths_half_noisy(project_path, data_path)
    # save_dynamical_system_from_yaml(yaml_path, labels_to_use, save_dir, tr_save_dir=tr_save_dir)

    # print("30k, half noisy")
    # yaml_path, labels_to_use, save_dir = get_30k_100_paths_half_noisy(project_path, data_path)
    # save_dynamical_system_from_yaml(yaml_path, labels_to_use, save_dir, tr_save_dir=tr_save_dir)

    # print("100k, half noisy")
    # yaml_path, labels_to_use, save_dir = get_100k_100_paths_half_noisy(project_path, data_path)
    # save_dynamical_system_from_yaml(yaml_path, labels_to_use, save_dir, tr_save_dir=tr_save_dir)

    print("degree 4 drift, 30k, half noisy")
    yaml_path, labels_to_use, save_dir = get_deg_4_drift_30k_100_paths_half_noisy(project_path, data_path)
    save_dynamical_system_from_yaml(yaml_path, labels_to_use, save_dir, tr_save_dir=tr_save_dir)
