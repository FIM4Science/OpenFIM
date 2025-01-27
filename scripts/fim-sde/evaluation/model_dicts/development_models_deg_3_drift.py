from pathlib import Path


def get_model_dicts_20241230_trained_on_30k_deg_3_drift_deg_0_diffusion_50_paths_streaming_dataloader() -> tuple[dict, dict, str]:
    """
    Set up evaluation for ablation studies on 30k degree 3 drift and constant diffusion, trained on cluster with streaming dataloader
    Ablations include:
    """
    base_dir = Path("/cephfs_projects/foundation_models/models/FIMSDE/")

    # Models to load
    model_dict = {
        "1000_threshold_linear_softmax_attn-2_layers_GNOT_repeated_4_layers": "30K_deg_3_drift_deg_0_diff_mlp_out_projection_2_lin_self_att_4_repeated_loc_query_streaming_dataloader_01-05-2310",
    }
    model_dict = {key: base_dir / exp_name / "checkpoints" for key, exp_name in model_dict.items()}

    model_display_ids = {
        "1000_threshold_linear_softmax_attn-2_layers_GNOT_repeated_4_layers": "Lin. Softmax Self-Attn. 2 Layers + GNOT, 4 times location query",
    }

    return model_dict, model_display_ids


def get_models_dict_20250116_traind_on_30k_deg_3_drift_deg_0_diffusion_50_paths_with_noise_and_mask() -> tuple[dict, dict, str]:
    base_dir = Path("/cephfs_projects/foundation_models/models/FIMSDE/")
    # Models to load
    model_dict = {
        "ablation_30K_deg_3_03_noise": "30K_deg_3_drift_ablation_train_on_50_locations_03_noise_01-14-1729",
        "ablation_30K_deg_3_10_noise": "30K_deg_3_drift_ablation_train_on_50_locations_10_noise_01-14-1728",
        "ablation_30K_deg_3_05_mask": "30K_deg_3_drift_ablation_train_on_50_locations_05_mask_01-16-1503",
        "ablation_30K_deg_3_20_mask": "30K_deg_3_drift_ablation_train_on_50_locations_20_mask_01-16-1503",
    }
    model_dict = {key: base_dir / exp_name / "checkpoints" for key, exp_name in model_dict.items()}

    model_display_ids = {
        "ablation_30K_deg_3_03_noise": "30K deg 3 drift ablation train on 50 locations 03 noise",
        "ablation_30K_deg_3_05_mask": "30K deg 3 drift ablation train on 50 locations 05 mask",
        "ablation_30K_deg_3_10_noise": "30K deg 3 drift ablation train on 50 locations 10 noise",
        "ablation_30K_deg_3_20_mask": "30K deg 3 drift ablation train on 50 locations 20 mask",
    }

    return model_dict, model_display_ids


def get_models_dict_20250116_traind_on_30k_deg_3_drift_deg_0_diffusion_50_paths_with_noise_and_mask_new_loss() -> tuple[dict, dict, str]:
    base_dir = Path("/cephfs_projects/foundation_models/models/FIMSDE/")
    # Models to load
    model_dict = {
        "ablation_30K_deg_3_03_noise": "30K_deg_3_drift_ablation_obs_noise_03_01-18-2113",
        "ablation_30K_deg_3_10_noise": "30K_deg_3_drift_ablation_obs_noise_10_01-18-2112",
        "ablation_30K_deg_3_05_mask": "30K_deg_3_drift_ablation_dropout_05_01-18-2126",
        "ablation_30K_deg_3_20_mask": "30K_deg_3_drift_ablation_dropout_20_01-18-2127",
    }
    model_dict = {key: base_dir / exp_name / "checkpoints" for key, exp_name in model_dict.items()}

    model_display_ids = {
        "ablation_30K_deg_3_03_noise": "30K deg 3 drift ablation train on 50 locations 03 noise",
        "ablation_30K_deg_3_05_mask": "30K deg 3 drift ablation train on 50 locations 05 mask",
        "ablation_30K_deg_3_10_noise": "30K deg 3 drift ablation train on 50 locations 10 noise",
        "ablation_30K_deg_3_20_mask": "30K deg 3 drift ablation train on 50 locations 20 mask",
    }

    return model_dict, model_display_ids
