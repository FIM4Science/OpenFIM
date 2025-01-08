from pathlib import Path

from fim.data_generation.sde.dynamical_systems_to_files import save_dynamical_system_from_yaml


def get_450k_100_paths_monomials_survive_uniformly(project_path: Path, data_path: Path):
    yaml_path = project_path / Path(
        "configs/train/fim-sde/train_data_configs/20250110_450k_drift_deg_3_diffusion-deg-0-100-paths-monomials-survive-uniformly.yaml"
    )
    save_dir = data_path / Path("data/processed/train/450k_drift_deg_3_diffusion_deg_0_100_paths_monomial_survival_uniform")
    labels_to_use = ["train", "test", "validation"]
    return yaml_path, labels_to_use, save_dir


def get_450k_100_paths_monomials_coeffs_std_10(project_path: Path, data_path: Path):
    yaml_path = project_path / Path(
        "configs/train/fim-sde/train_data_configs/20250113_450k_drift_deg_3_diffusion_deg_0_100_paths_monomial_coefficients_std_10.yaml"
    )
    save_dir = data_path / Path("data/processed/train/450k_drift_deg_3_diffusion_deg_0_100_paths_monomial_coeffs_std_10")
    labels_to_use = ["train", "test", "validation"]
    return yaml_path, labels_to_use, save_dir


if __name__ == "__main__":
    project_path = Path("/home/seifner/repos/FIM/")
    data_path = Path("/home/seifner/repos/FIM/")

    # print("450k degree and monomial survival uniform")
    # yaml_path, labels_to_use, save_dir = get_450k_100_paths_monomials_survive_uniformly(project_path, data_path)
    # save_dynamical_system_from_yaml(yaml_path, labels_to_use, save_dir)

    print("450k degree and monomial coeffs from N(0, 10)")
    yaml_path, labels_to_use, save_dir = get_450k_100_paths_monomials_coeffs_std_10(project_path, data_path)
    save_dynamical_system_from_yaml(yaml_path, labels_to_use, save_dir)
