from pathlib import Path

from fim.data_generation.sde.dynamical_systems_to_files import save_dynamical_system_from_yaml


def get_600k_100_paths_half_noisy_with_delta_tau_1e_1_to_1e_3(project_path: Path, data_path: Path):
    yaml_path = project_path / Path("configs/data_generation/sde/test_saving_vf_at_obs.yaml")
    save_dir = data_path / Path("test_saving_vf_at_obs")
    labels_to_use = ["train"]
    return yaml_path, labels_to_use, save_dir


if __name__ == "__main__":
    project_path = Path("/Users/patrickseifner/repos/FIM")
    data_path = Path("/Users/patrickseifner/repos/FIM")

    tr_save_dir = data_path / "save_dynamical_system_tr"
    tr_save_dir.mkdir(exist_ok=True)

    print("600k, half noisy, delta tau 1e-1 to 1e-3")
    yaml_path, labels_to_use, save_dir = get_600k_100_paths_half_noisy_with_delta_tau_1e_1_to_1e_3(project_path, data_path)
    save_dynamical_system_from_yaml(yaml_path, labels_to_use, save_dir, tr_save_dir=tr_save_dir)
