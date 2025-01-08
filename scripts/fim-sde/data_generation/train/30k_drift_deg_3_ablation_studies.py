from pathlib import Path

from fim.data_generation.sde.dynamical_systems_to_files import save_dynamical_system_from_yaml


def get_degree_surv_rate_075_monomial_surv_rate_05(project_path: Path, data_path: Path):
    yaml_path = project_path / Path(
        "configs/train/fim-sde/train_data_configs/30k_drift_deg_3_ablation_studies/sde-drift-deg-3-diffusion-deg-0-50-paths-degree-survives-075-monomials-survive-05.yaml"
    )

    save_dir = data_path / Path("data/processed/train/30k_drift_deg_3_ablation_studies/degree_surv_rate_075_monomial_surv_rate_05")
    labels_to_use = ["train", "test", "validation"]
    return yaml_path, labels_to_use, save_dir


def get_degree_surv_rate_05_monomial_surv_rate_025(project_path: Path, data_path: Path):
    yaml_path = project_path / Path(
        "configs/train/fim-sde/train_data_configs/30k_drift_deg_3_ablation_studies/sde-drift-deg-3-diffusion-deg-0-50-paths_degree_survives-05-monomials-survive-025.yaml"
    )
    save_dir = data_path / Path("data/processed/train/30k_drift_deg_3_ablation_studies/degree_surv_rate_05_monomial_surv_rate_025")
    labels_to_use = ["train", "test", "validation"]
    return yaml_path, labels_to_use, save_dir


def get_uniform_init_cond(project_path: Path, data_path: Path):
    yaml_path = project_path / Path(
        "configs/train/fim-sde/train_data_configs/30k_drift_deg_3_ablation_studies/sde-drift-deg-3-diffusion-deg-0-50-paths-uniform-init-cond.yaml"
    )

    save_dir = data_path / Path("data/processed/train/30k_drift_deg_3_ablation_studies/uniform_init_cond")
    labels_to_use = ["train", "test", "validation"]
    return yaml_path, labels_to_use, save_dir


def get_diffusion_degree_2(project_path: Path, data_path: Path):
    yaml_path = project_path / Path(
        "configs/train/fim-sde/train_data_configs/30k_drift_deg_3_ablation_studies/sde-drift-deg-3-diffusion-deg-2-50-paths.yaml"
    )

    save_dir = data_path / Path("data/processed/train/30k_drift_deg_3_ablation_studies/diffusion_degree_2")
    labels_to_use = ["train", "test", "validation"]
    return yaml_path, labels_to_use, save_dir


def get_300_paths(project_path: Path, data_path: Path):
    yaml_path = project_path / Path(
        "configs/train/fim-sde/train_data_configs/30k_drift_deg_3_ablation_studies/sde-drift-deg-3-diffusion-deg-0-300-paths.yaml"
    )

    save_dir = data_path / Path("data/processed/train/30k_drift_deg_3_ablation_studies/300_paths")
    labels_to_use = ["train", "test", "validation"]
    return yaml_path, labels_to_use, save_dir


def get_init_cond_mean_from_uniform(project_path: Path, data_path: Path):
    yaml_path = project_path / Path(
        "configs/train/fim-sde/train_data_configs/30k_drift_deg_3_ablation_studies/sde-drift-deg-3-diffusion-deg-0-init-cond-mean-from-uniform.yaml"
    )
    save_dir = data_path / Path("data/processed/train/30k_drift_deg_3_ablation_studies/init_cond_mean_from_uniform")
    labels_to_use = ["train", "test", "validation"]
    return yaml_path, labels_to_use, save_dir


def get_one_long_path_data(project_path: Path, data_path: Path):
    yaml_path = project_path / Path(
        "configs/train/fim-sde/train_data_configs/30k_drift_deg_3_ablation_studies/sde-drift-deg-3-diffusion-deg-0-1-path-50-paths-length-128-equivalent.yaml"
    )
    save_dir = data_path / Path("data/processed/train/30k_drift_deg_3_ablation_studies/one_long_path")
    labels_to_use = ["train", "test", "validation"]
    return yaml_path, labels_to_use, save_dir


def get_one_degree_monomial_uniform_surivial(project_path: Path, data_path: Path):
    yaml_path = project_path / Path(
        "configs/train/fim-sde/train_data_configs/30k_drift_deg_3_ablation_studies/sde-drift-deg-3-diffusion-deg-0-monomial-survival-uniform.yaml"
    )
    save_dir = data_path / Path("data/processed/train/30k_drift_deg_3_ablation_studies/degree_and_monomial_survival_uniform")
    labels_to_use = ["train", "test", "validation"]
    return yaml_path, labels_to_use, save_dir


def get_degree_4_drift_data(project_path: Path, data_path: Path):
    yaml_path = project_path / Path(
        "configs/train/fim-sde/train_data_configs/30k_drift_deg_3_ablation_studies/sde-drift-deg-4-diffusion-deg-0-50-paths.yaml"
    )
    save_dir = data_path / Path("data/processed/train/30k_drift_deg_3_ablation_studies/degree_4_drift")
    labels_to_use = ["train", "test", "validation"]
    return yaml_path, labels_to_use, save_dir


if __name__ == "__main__":
    project_path = Path("/home/seifner/repos/FIM/")
    data_path = Path("/home/seifner/repos/FIM/")

    # project_path = Path("/home/seifnerp_hpc/repos/FIM/")
    # data_path = Path("/lustre/scratch/data/seifnerp_hpc-fim_data/")

    # project_path = Path("/Users/patrickseifner/repos/FIM")
    # data_path = Path("/Users/patrickseifner/repos/FIM")

    # print("degree survival rate 0.75, monomial survival rate 0.5")
    # yaml_path, labels_to_use, save_dir = get_degree_surv_rate_075_monomial_surv_rate_05(project_path, data_path)
    # save_dynamical_system_from_yaml(yaml_path, labels_to_use, save_dir)

    # print("degree survival rate 0.5, monomial survival rate 0.25")
    # yaml_path, labels_to_use, save_dir = get_degree_surv_rate_05_monomial_surv_rate_025(project_path, data_path)
    # save_dynamical_system_from_yaml(yaml_path, labels_to_use, save_dir)

    # print("uniform init cond")
    # yaml_path, labels_to_use, save_dir = get_uniform_init_cond(project_path, data_path)
    # save_dynamical_system_from_yaml(yaml_path, labels_to_use, save_dir)

    # print("diffusion degree 2")
    # yaml_path, labels_to_use, save_dir = get_diffusion_degree_2(project_path, data_path)
    # save_dynamical_system_from_yaml(yaml_path, labels_to_use, save_dir)

    # print("300 paths")
    # yaml_path, labels_to_use, save_dir = get_300_paths(project_path, data_path)
    # save_dynamical_system_from_yaml(yaml_path, labels_to_use, save_dir)

    # print("init cond mean from uniform")
    # yaml_path, labels_to_use, save_dir = get_init_cond_mean_from_uniform(project_path, data_path)
    # save_dynamical_system_from_yaml(yaml_path, labels_to_use, save_dir)

    # print("one long path")
    # yaml_path, labels_to_use, save_dir = get_one_long_path_data(project_path, data_path)
    # save_dynamical_system_from_yaml(yaml_path, labels_to_use, save_dir)

    # print("degree and monomial survival uniform")
    # yaml_path, labels_to_use, save_dir = get_one_degree_monomial_uniform_surivial(project_path, data_path)
    # save_dynamical_system_from_yaml(yaml_path, labels_to_use, save_dir)

    print("up to degree 4 drift")
    yaml_path, labels_to_use, save_dir = get_degree_4_drift_data(project_path, data_path)
    save_dynamical_system_from_yaml(yaml_path, labels_to_use, save_dir)
