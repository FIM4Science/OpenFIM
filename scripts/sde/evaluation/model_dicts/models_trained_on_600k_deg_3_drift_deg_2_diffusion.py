# def get_model_dicts_600k_deg_3_drift_deg_2_diff() -> tuple[dict, dict, str]:
#     """
#     Set up evaluation for ablation studies on 30k degree 3 drift and degree 2 diffusion.
#     Ablations include:
#     """
#     model_dict = {
#         "20M_params_trained_even_longer": "/cephfs_projects/foundation_models/models/FIMSDE/ICML_submission_models/20M_params_icml_submission/checkpoints",
#     }
#
#     model_display_ids = {
#         "20M_params_trained_even_longer": "20M Paramters trained even longer (ICML Paper)",
#     }
#
#     return model_dict, model_display_ids
#
#
# def get_model_dicts_600k_post_submission_models() -> tuple[dict, dict, str]:
#     """ """
#     model_dicts = {
#         # "20M_params_trained_even_longer": "/cephfs_projects/foundation_models/models/FIMSDE/ICML_submission_models/20M_params_icml_submission/checkpoints",
#         # "20M_params_trained_delta_tau_1e-1_to-1e-3 (checkpoint 03-13-1415)": "/cephfs_projects/foundation_models/models/FIMSDE/NeurIPS_submission_models/old_development_models/600k_drift_deg_3_diff_deg_2_delta_tau_1e-1_to_1e-3_2_layer_enc_8_layer_03-13-1415/checkpoints",
#         # "20M_params_trained_delta_tau_1e-1_to-1e-3 (checkpoint 03-18-1205)": "/cephfs_projects/foundation_models/models/FIMSDE/NeurIPS_submission_models/old_development_models/600k_drift_deg_3_diff_deg_2_delta_tau_1e-1_to_1e-3_2_layer_enc_8_layer_03-18-1205/checkpoints",
#         "20M_params_trained_delta_tau_1e-1_to-1e-3 (checkpoint 03-23-1747)": "/cephfs_projects/foundation_models/models/FIMSDE/NeurIPS_submission_models/old_development_models/600k_drift_deg_3_diff_deg_2_delta_tau_1e-1_to_1e-3_2_layer_enc_8_layer_03-23-1747/checkpoints",
#     }
#     models_display_ids = {
#         # "20M_params_trained_even_longer": "20M Paramters trained even longer (ICML Paper)",
#         # "20M_params_trained_delta_tau_1e-1_to-1e-3 (checkpoint 03-13-1415)": "20M Params, trained on delta tau from 1e-1 to 1e-3 (checkpoint 03-13-1415)",
#         # "20M_params_trained_delta_tau_1e-1_to-1e-3 (checkpoint 03-18-1205)": "20M Params, trained on delta tau from 1e-1 to 1e-3 (checkpoint 03-18-1205)",
#         "20M_params_trained_delta_tau_1e-1_to-1e-3 (checkpoint 03-23-1747)": "20M Params, trained on delta tau from 1e-1 to 1e-3 (checkpoint 03-23-1747)",
#     }
#
#     return model_dicts, models_display_ids
#


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


# def get_model_dicts_600k_fixed_linear_attn() -> tuple[dict, dict, str]:
#     """ """
#     model_dicts = {
#         # "20M_params_600k_polys_delta_tau_fixed_attn (checkpoint 04-28-0941)": "/cephfs_projects/foundation_models/models/FIMSDE/NeurIPS_submission_models/old_development_models/600k_drift_deg_3_diff_deg_2_delta_tau_1e-1_to_1e-3_2_layer_enc_8_layer_with_fixed_linear_attn_04-28-0941/checkpoints/",
#         # "20M_params_600k_polys_delta_tau_fixed_attn (checkpoint 05-01-1207)": "/cephfs_projects/foundation_models/models/FIMSDE/NeurIPS_submission_models/old_development_models/600k_drift_deg_3_diff_deg_2_delta_tau_1e-1_to_1e-3_2_layer_enc_8_layer_with_fixed_linear_attn_05-01-1207/checkpoints/",
#         "20M_params_600k_polys_delta_tau_fixed_attn_fixed_softmax (checkpoint 05-03-2033)": "/cephfs_projects/foundation_models/models/FIMSDE/NeurIPS_submission_models/600k_drift_deg_3_diff_deg_2_delta_tau_fixed_linear_attn_softmax_no_extra_normalization_and_fix_in_residual_layer_05-03-2033/checkpoints/",
#         "20M_params_600k_polys_delta_tau_fixed_attn_fixed_softmax (checkpoint 05-06-2300)": "/cephfs_projects/foundation_models/models/FIMSDE/NeurIPS_submission_models/600k_drift_deg_3_diff_deg_2_delta_tau_fixed_linear_attn_softmax_no_extra_normalization_and_fix_in_residual_layer_05-06-2300/checkpoints/",
#         "20M_params_600k_polys_delta_tau_fixed_attn_fixed_softmax (checkpoint 05-10-0015)": "/cephfs_projects/foundation_models/models/FIMSDE/NeurIPS_submission_models/600k_drift_deg_3_diff_deg_2_delta_tau_fixed_linear_attn_softmax_no_extra_normalization_and_fix_in_residual_layer_05-10-0015/checkpoints/",
#     }
#     models_display_ids = {
#         # "20M_params_600k_polys_delta_tau_fixed_attn (checkpoint 04-28-0941)": "20M Params, trained on 600k polys, with delta tau, fixed linear attn (checkpoint 04-28-0941)",
#         # "20M_params_600k_polys_delta_tau_fixed_attn (checkpoint 05-01-1207)": "20M Params, trained on 600k polys, with delta tau, fixed linear attn (checkpoint 05-01-1207)",
#         "20M_params_600k_polys_delta_tau_fixed_attn_fixed_softmax (checkpoint 05-03-2033)": "20M Params, trained on 600k polys, with delta tau, fixed linear attn and softmax (checkpoint 05-03-2033)",
#         "20M_params_600k_polys_delta_tau_fixed_attn_fixed_softmax (checkpoint 05-06-2300)": "20M Params, trained on 600k polys, with delta tau, fixed linear attn and softmax (checkpoint 05-06-2300)",
#         "20M_params_600k_polys_delta_tau_fixed_attn_fixed_softmax (checkpoint 05-10-0015)": "20M Params, trained on 600k polys, with delta tau, fixed linear attn and softmax (checkpoint '5-10-0015)",
#     }
#
#     return model_dicts, models_display_ids


def get_model_dicts_neurips_submission_checkpoint() -> tuple[dict, dict, str]:
    """ """
    model_dicts = {
        "FIM_fixed_softmax_dim_epoch_139": {
            "checkpoint_dir": "/cephfs_projects/foundation_models/models/FIMSDE/NeurIPS_submission_models/600k_drift_deg_3_diff_deg_2_delta_tau_fixed_linear_attn_softmax_no_extra_normalization_and_fix_in_residual_layer_05-06-2300/checkpoints/",
            "checkpoint_name": "epoch-139",
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
