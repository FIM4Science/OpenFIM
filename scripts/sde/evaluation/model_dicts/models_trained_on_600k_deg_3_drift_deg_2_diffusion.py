def get_model_dicts_ablation_models() -> tuple[dict, dict, str]:
    """ """
    model_dicts = {
        # "ablation_30k_deg_4_drift_500k_steps": {
        #     "checkpoint_dir": "/cephfs_projects/foundation_models/models/FIMSDE/NeurIPS_submission_models/ablation_models/30k_drift_deg_4_diff_deg_2_5M_params_1_layer_enc_4_layer_128_hidden_size_03-24-1650/checkpoints",
        #     "checkpoint_name": "epoch-475",
        # },
        # "ablation_30k_train_size_500k_steps": {
        #     "checkpoint_dir": "/cephfs_projects/foundation_models/models/FIMSDE/NeurIPS_submission_models/ablation_models/30k_drift_deg_3_diff_deg_2_5M_params_1_layer_enc_4_layer_128_hidden_size_03-16-1744/checkpoints",
        #     "checkpoint_name": "epoch-494",
        # },
        # "ablation_100k_train_size_500k_steps": {
        #     "checkpoint_dir": "/cephfs_projects/foundation_models/models/FIMSDE/NeurIPS_submission_models/ablation_models/100k_drift_deg_3_diff_deg_2_10M_params_1_layer_enc_4_layer_256_hidden_size_03-18-1123/checkpoints",
        #     "checkpoint_name": "epoch-159",
        # },
        # "ablation_600k_train_size_500k_steps": { # TRAINED ON DELTA TAU; DON'T USE
        #     "checkpoint_dir": "/cephfs_projects/foundation_models/models/FIMSDE/NeurIPS_submission_models/old_development_models/600k_drift_deg_3_diff_deg_2_delta_tau_1e-1_to_1e-3_2_layer_enc_8_layer_with_fixed_linear_attn_04-28-0941/checkpoints/",
        #     "checkpoint_name": "epoch-70",
        # },
        "ablation_600k_train_size_500k_steps_from_ICML_model_epoch_70": {
            "checkpoint_dir": "/cephfs_projects/foundation_models/models/FIMSDE/ICML_submission_models/20M_params_icml_submission/checkpoints/",
            "checkpoint_name": "epoch-84",  # checkpoint 70 go delted, replace with earliest we have
        },
    }
    models_display_ids = {
        "ablation_30k_deg_4_drift_500k_steps": "Abl: deg 4 drift, 30k train size, 5M params, 500k steps",
        "ablation_30k_train_size_500k_steps": "Abl: 30k train size, 5M params, 500k steps",
        "ablation_100k_train_size_500k_steps": "Abl: 100k train size, 10M params, 500k steps",
        "ablation_600k_train_size_500k_steps": "Abl: 600k train size, 20M params, 500k steps",
        "ablation_600k_train_size_500k_steps_from_ICML_model_epoch_70": "Abl: 600k train size, 20M params, 500k steps (ICML model, no delta tau, epoch 70)",
    }

    return model_dicts, models_display_ids


def get_model_dicts_neurips_submission_checkpoint() -> tuple[dict, dict, str]:
    """ """
    model_dicts = {
        "FIM_fixed_softmax_dim_epoch_139": {
            # "checkpoint_dir": "/cephfs_projects/foundation_models/models/FIMSDE/NeurIPS_submission_models/600k_drift_deg_3_diff_deg_2_delta_tau_fixed_linear_attn_softmax_no_extra_normalization_and_fix_in_residual_layer_05-06-2300/checkpoints/",
            # "checkpoint_name": "epoch-139",
            "checkpoint_dir": "/cephfs_projects/foundation_models/models/FIMSDE/NeurIPS_submission_models/600k_drift_deg_3_diff_deg_2_delta_tau_fixed_linear_attn_softmax_no_extra_normalization_and_fix_in_residual_layer_05-06-2300_for_refactored_model_code/checkpoints/",
            "checkpoint_name": "epoch-139_refactored",
        },
    }
    models_display_ids = {
        "FIM_fixed_softmax_dim_epoch_139": "FIM, fixed Softmax dim., Epoch 139",
    }

    return model_dicts, models_display_ids


def get_model_dicts_post_neurips_submission_checkpoint() -> tuple[dict, dict, str]:
    """ """
    model_dicts = {
        "FIM_half_locs_from_obs_epoch_139": {
            "checkpoint_dir": "/cephfs_projects/foundation_models/models/FIMSDE/Post_NeurIPS_models/600k_32_locations_at_observations_32_locations_randomly_07-14-1850/checkpoints",
            "checkpoint_name": "epoch-139",
        },
    }
    models_display_ids = {
        "FIM_half_locs_from_obs_epoch_139": "FIM, 1/2 locations from observations, Epoch 139",
    }

    return model_dicts, models_display_ids
