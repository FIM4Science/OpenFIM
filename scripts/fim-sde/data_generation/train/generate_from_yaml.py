from pathlib import Path

from fim.data_generation.sde.dynamical_systems_to_files import save_dynamical_system_from_yaml


def get_sde_drift_deg_2_diffusion_deg_1_no_scale(project_path: Path, data_path: Path):
    yaml_path = project_path / Path("configs/train/fim-sde/train_data_configs/sde-drift-deg-2-diffusion-deg-1-no-scale.yaml")
    save_dir = data_path / Path("data/processed/train/sde-drift-deg-2-diffusion-deg-1-no-scale/")
    labels_to_use = ["train", "test", "validation"]
    return yaml_path, labels_to_use, save_dir


def get_sde_drift_deg_2_diffusion_deg_0_no_scale(project_path: Path, data_path: Path):
    yaml_path = project_path / Path("configs/train/fim-sde/train_data_configs/sde-drift-deg-2-diffusion-deg-0-no-scale.yaml")
    save_dir = data_path / Path("data/processed/train/sde-drift-deg-2-diffusion-deg-0-no-scale/")
    labels_to_use = ["train", "test", "validation"]
    return yaml_path, labels_to_use, save_dir


def get_sde_drift_deg_2_diffusion_deg_0_no_scale_100_paths(project_path: Path, data_path: Path):
    yaml_path = project_path / Path("configs/train/fim-sde/train_data_configs/sde-drift-deg-2-diffusion-deg-0-no-scale-100-paths.yaml")
    save_dir = data_path / Path("data/processed/train/sde-drift-deg-2-diffusion-deg-0-no-scale-100-paths/")
    labels_to_use = ["train", "test", "validation"]
    return yaml_path, labels_to_use, save_dir


def get_sde_drift_deg_3_diffusion_deg_0_50_paths(project_path: Path, data_path: Path):
    yaml_path = project_path / Path("configs/train/fim-sde/train_data_configs/sde-drift-deg-3-diffusion-deg-0-50-paths.yaml")
    save_dir = data_path / Path("data/processed/train/sde-drift-deg-3-diffusion-deg-0-50-paths")
    labels_to_use = ["train", "test", "validation"]
    return yaml_path, labels_to_use, save_dir


def get_sde_drift_deg_3_diffusion_deg_0_50_paths_30_perc_hypercube(project_path: Path, data_path: Path):
    yaml_path = project_path / Path(
        "configs/train/fim-sde/train_data_configs/sde-drift-deg-3-diffusion-deg-0-50-paths-30-perc-larger-hypercube.yaml"
    )
    save_dir = data_path / Path("data/processed/train/sde-drift-deg-3-diffusion-deg-0-50-paths-30-perc-larger-hypercube")
    labels_to_use = ["train", "test", "validation"]
    return yaml_path, labels_to_use, save_dir


def get_sde_drift_deg_3_diffusion_deg_0_50_paths_30_perc_hypercube_10_noise_5_mask(project_path: Path, data_path: Path):
    yaml_path = project_path / Path(
        "configs/train/fim-sde/train_data_configs/sde-drift-deg-3-diffusion-deg-0-50-paths-30-perc-larger-hypercube-10-noise-5-mask.yaml"
    )
    save_dir = data_path / Path("data/processed/train/sde-drift-deg-3-diffusion-deg-0-50-paths-30-perc-larger-hypercube-5-mask")
    labels_to_use = ["train", "test", "validation"]
    return yaml_path, labels_to_use, save_dir


def get_sde_test_resources_data(project_path: Path, data_path: Path):
    yaml_path = project_path / Path("configs/train/fim-sde/train_data_configs/sde-for-test-resources.yaml")
    save_dir = data_path / Path("data/processed/train/sde-for-test-resources")
    labels_to_use = ["train", "test", "validation"]
    return yaml_path, labels_to_use, save_dir


if __name__ == "__main__":
    project_path = Path("/home/cvejoski/Projects/FoundationModels/FIM/")
    data_path = Path("/home/cvejoski/Projects/FoundationModels/FIM/")

    # project_path = Path("/home/seifnerp_hpc/repos/FIM/")
    # data_path = Path("/lustre/scratch/data/seifnerp_hpc-fim_data/")

    # project_path = Path("/Users/patrickseifner/repos/FIM")
    # data_path = Path("/Users/patrickseifner/repos/FIM")

    yaml_path, labels_to_use, save_dir = get_sde_drift_deg_3_diffusion_deg_0_50_paths_30_perc_hypercube_10_noise_5_mask(
        project_path, data_path
    )

    save_dynamical_system_from_yaml(yaml_path, labels_to_use, save_dir)
