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
