#!/usr/bin/env python
from pathlib import Path

from latent_sde_train_on_lorenz import sample_lorenz_paths_from_trained_model


if __name__ == "__main__":
    neural_sde_paper_path = Path("/cephfs_projects/foundation_models/data/SDE/test/20250629_lorenz_systems/neural_sde_paper/set_0/")
    neural_sde_github_path = Path("/cephfs_projects/foundation_models/data/SDE/test/20250629_lorenz_systems/neural_sde_github/set_0/")

    test_data_setups = {
        "neural_sde_paper": {
            "train_data_jsons": neural_sde_paper_path / "train_data.json",
            "(1,1,1)": {
                "inference_data": neural_sde_paper_path / "(1,1,1)_inference_data.json",
                "reference_data": neural_sde_paper_path / "(1,1,1)_reference_data.json",
            },
            "N(0,1)": {
                "inference_data": neural_sde_paper_path / "N(0,1)_inference_data.json",
                "reference_data": neural_sde_paper_path / "N(0,1)_reference_data.json",
            },
            "N(0,2)": {
                "inference_data": neural_sde_paper_path / "N(0,2)_inference_data.json",
                "reference_data": neural_sde_paper_path / "N(0,2)_reference_data.json",
            },
        },
        "neural_sde_github": {
            "train_data_jsons": neural_sde_github_path / "train_data.json",
            "(1,1,1)": {
                "inference_data": neural_sde_github_path / "(1,1,1)_inference_data.json",
                "reference_data": neural_sde_github_path / "(1,1,1)_reference_data.json",
            },
            "N(0,1)": {
                "inference_data": neural_sde_github_path / "N(0,1)_inference_data.json",
                "reference_data": neural_sde_github_path / "N(0,1)_reference_data.json",
            },
            "N(0,2)": {
                "inference_data": neural_sde_github_path / "N(0,2)_inference_data.json",
                "reference_data": neural_sde_github_path / "N(0,2)_reference_data.json",
            },
        },
    }

    exp_name = "latent_sde_paper_setup_posterior_equation_record_every_5_epochs_convergence_speed_post_neurips"
    checkpoint_dir = Path(
        "/cephfs/users/seifner/repos/FIM/results/latent_sde_paper_setup_record_every_5_epochs_for_convergence_speed_comparison_08-21-0058/checkpoints"
    )

    n_epochs = 5000
    train_data_label = "neural_sde_paper"
    sample_every = 5

    sample_lorenz_paths_from_trained_model(checkpoint_dir, n_epochs, exp_name, train_data_label, test_data_setups, sample_every)
