def get_model_dicts_600k_deg_3_drift_deg_2_diff() -> tuple[dict, dict, str]:
    """
    Set up evaluation for ablation studies on 30k degree 3 drift and degree 2 diffusion.
    Ablations include:
    """
    model_dict = {
        # "11M_params": "/home/seifner/repos/FIM/saved_results/20250120_11M_params_trained_on_600k_deg_3/600k_drift_deg_3_diff_deg_2_embed_size_256_cont_with_grad_clip_lr_1e-5_weight_decay_1e-4_01-19-2228/checkpoints",
        # "20M_params": "/home/seifner/repos/FIM/saved_results/20250124_20M_params_trained_on_600k_deg_3/600k_drift_deg_3_diff_deg_2_mse_no_div_by_diff_softplus_2_layer_enc_8_layer_dec_01-21-1106/checkpoints",
        # "20M_params_trained_longer": "/home/seifner/repos/FIM/saved_results/20250124_20M_params_trained_on_600k_deg_3/600k_drift_deg_3_diff_deg_2_mse_no_div_by_diff_softplus_2_layer_enc_8_layer_dec_01-21-1106_trained_longer/checkpoints",
        "20M_params_trained_even_longer": "/home/seifner/repos/FIM/saved_results/20250124_20M_params_trained_on_600k_deg_3/600k_drift_deg_3_diff_deg_2_mse_no_div_by_diff_softplus_2_layer_enc_8_layer_dec_continued_01-25-1301_trained_even_longer/checkpoints",
    }

    model_display_ids = {
        # "11M_params": "11M Parameters",
        # "20M_params": "20M Parameters",
        # "20M_params_trained_longer": "20M Parameters trained longer",
        "20M_params_trained_even_longer": "20M Paramters trained even longer",
    }

    return model_dict, model_display_ids


def get_model_dicts_600k_post_submission_models() -> tuple[dict, dict, str]:
    """ """
    model_dicts = {
        # "20M_params_trained_even_longer": "/cephfs_projects/foundation_models/models/FIMSDE/20M_parameter_model_for_icml_submission/checkpoints",
        # "20M_params_trained_even_longer": "/home/seifner/repos/FIM/saved_results/20250124_20M_params_trained_on_600k_deg_3/600k_drift_deg_3_diff_deg_2_mse_no_div_by_diff_softplus_2_layer_enc_8_layer_dec_continued_01-25-1301_trained_even_longer/checkpoints",
        # "20M_params_cont_train_unary_binary": "/home/seifner/repos/FIM/saved_results/20250124_20M_params_trained_on_600k_deg_3/600k_drift_deg_3_diff_deg_2_mse_no_div_by_diff_softplus_2_layer_enc_8_layer_dec_continued_more_data_NOT_USED_IN_PAPER_01-29-2111/checkpoints",
        # "20M_params_paper_model_cont_training_unary_binary": "/cephfs_projects/foundation_models/models/FIMSDE/600k_drift_deg_3_diff_deg_2_cont_icml_submission_model_with_unary_binary_data/checkpoints",
        # "20M_params_trained_delta_tau_1e-1_to-1e-3 (checkpoint 03-13-1415)": "/cephfs/users/seifner/repos/FIM/saved_results/20250124_20M_params_trained_on_600k_deg_3/600k_drift_deg_3_diff_deg_2_delta_tau_1e-1_to_1e-3_2_layer_enc_8_layer_03-13-1415/checkpoints",
        # "20M_params_trained_delta_tau_1e-1_to-1e-3 (checkpoint 03-18-1205)": "/cephfs/users/seifner/repos/FIM/saved_results/20250124_20M_params_trained_on_600k_deg_3/600k_drift_deg_3_diff_deg_2_delta_tau_1e-1_to_1e-3_2_layer_enc_8_layer_03-18-1205/checkpoints",
        "20M_params_trained_delta_tau_1e-1_to-1e-3 (checkpoint 03-23-1747)": "/cephfs/users/seifner/repos/FIM/saved_results/20250124_20M_params_trained_on_600k_deg_3/600k_drift_deg_3_diff_deg_2_delta_tau_1e-1_to_1e-3_2_layer_enc_8_layer_03-23-1747/checkpoints",
    }
    models_display_ids = {
        # "20M_params_trained_even_longer": "20M Paramters trained even longer (Paper)",
        # "20M_params_trained_even_longer": "20M Params, Paper Model",
        # "20M_params_cont_train_unary_binary": "20M Params, cont. train on unary-binary trees",
        # "20M_params_paper_model_cont_training_unary_binary": "20M Paras, cont. train mixed polynomials and unary-binary",
        # "20M_params_trained_delta_tau_1e-1_to-1e-3 (checkpoint 03-13-1415)": "20M Params, trained on delta tau from 1e-1 to 1e-3 (checkpoint 03-13-1415)",
        # "20M_params_trained_delta_tau_1e-1_to-1e-3 (checkpoint 03-18-1205)": "20M Params, trained on delta tau from 1e-1 to 1e-3 (checkpoint 03-18-1205)",
        "20M_params_trained_delta_tau_1e-1_to-1e-3 (checkpoint 03-23-1747)": "20M Params, trained on delta tau from 1e-1 to 1e-3 (checkpoint 03-23-1747)",
    }

    return model_dicts, models_display_ids


def get_model_dicts_ablation_models() -> tuple[dict, dict, str]:
    """ """
    model_dicts = {
        # "20M_params_trained_even_longer": "/home/seifner/repos/FIM/saved_results/20250124_20M_params_trained_on_600k_deg_3/600k_drift_deg_3_diff_deg_2_mse_no_div_by_diff_softplus_2_layer_enc_8_layer_dec_continued_01-25-1301_trained_even_longer/checkpoints",
        # "20M_params_trained_delta_tau_1e-1_to-1e-3 (checkpoint 03-13-1415)": "/cephfs/users/seifner/repos/FIM/saved_results/20250124_20M_params_trained_on_600k_deg_3/600k_drift_deg_3_diff_deg_2_delta_tau_1e-1_to_1e-3_2_layer_enc_8_layer_03-13-1415/checkpoints",
        # "20M_params_trained_delta_tau_1e-1_to-1e-3 (checkpoint 03-18-1205)": "/cephfs/users/seifner/repos/FIM/saved_results/20250124_20M_params_trained_on_600k_deg_3/600k_drift_deg_3_diff_deg_2_delta_tau_1e-1_to_1e-3_2_layer_enc_8_layer_03-18-1205/checkpoints",
        # "20M_params_trained_delta_tau_1e-1_to-1e-3 (checkpoint 03-23-1747)": "/cephfs/users/seifner/repos/FIM/saved_results/20250124_20M_params_trained_on_600k_deg_3/600k_drift_deg_3_diff_deg_2_delta_tau_1e-1_to_1e-3_2_layer_enc_8_layer_03-23-1747/checkpoints",
        # "ablation_30k_deg_4_drift_500k_steps": {
        #     "checkpoint_dir": "/home/seifner/repos/FIM/saved_results/20250318_icml_rebuttal_ablation/30k_drift_deg_4_diff_deg_2_5M_params_1_layer_enc_4_layer_128_hidden_size_03-24-1650/checkpoints",
        #     "checkpoint_name": "epoch-475",
        # },
        # "ablation_30k_train_size_500k_steps": {
        #     "checkpoint_dir": "/home/seifner/repos/FIM/saved_results/20250318_icml_rebuttal_ablation/30k_drift_deg_3_diff_deg_2_5M_params_1_layer_enc_4_layer_128_hidden_size_03-16-1744/checkpoints",
        #     "checkpoint_name": "epoch-494",
        # },
        # "ablation_100k_train_size_500k_steps": {
        #     "checkpoint_dir": "/home/seifner/repos/FIM/saved_results/20250318_icml_rebuttal_ablation/100k_drift_deg_3_diff_deg_2_10M_params_1_layer_enc_4_layer_256_hidden_size_03-18-1123/checkpoints",
        #     "checkpoint_name": "epoch-159",
        # },
        "ablation_600k_train_size_500k_steps": {
            "checkpoint_dir": "/home/seifner/repos/FIM/saved_results/20250430_20M_params_trained_on_600k_deg_3_with_delta_tau_fixed_linear_attn/600k_drift_deg_3_diff_deg_2_delta_tau_1e-1_to_1e-3_2_layer_enc_8_layer_with_fixed_linear_attn_04-28-0941/checkpoints/",
            "checkpoint_name": "epoch-70",
        },
    }
    models_display_ids = {
        "20M_params_trained_even_longer": "20M Params, Paper Model",
        "20M_params_trained_delta_tau_1e-1_to-1e-3 (checkpoint 03-13-1415)": "20M Params, trained on delta tau from 1e-1 to 1e-3 (checkpoint 03-13-1415)",
        "20M_params_trained_delta_tau_1e-1_to-1e-3 (checkpoint 03-18-1205)": "20M Params, trained on delta tau from 1e-1 to 1e-3 (checkpoint 03-18-1205)",
        "20M_params_trained_delta_tau_1e-1_to-1e-3 (checkpoint 03-23-1747)": "20M Params, trained on delta tau from 1e-1 to 1e-3 (checkpoint 03-23-1747)",
        "ablation_30k_deg_4_drift_500k_steps": "Abl: deg 4 drift, 500k steps",
        "ablation_30k_train_size_500k_steps": "Abl: 30k train size, 5M params, 500k steps",
        "ablation_100k_train_size_500k_steps": "Abl: 100k train size, 10M params, 500k steps",
        "ablation_600k_train_size_500k_steps": "Abl: 600k train size, 20M params, 500k steps",
    }

    return model_dicts, models_display_ids


def get_model_dicts_600k_fixed_linear_attn() -> tuple[dict, dict, str]:
    """ """
    model_dicts = {
        "20M_params_600k_polys_delta_tau_fixed_attn (checkpoint 04-28-0941)": "/home/seifner/repos/FIM/saved_results/20250430_20M_params_trained_on_600k_deg_3_with_delta_tau_fixed_linear_attn/600k_drift_deg_3_diff_deg_2_delta_tau_1e-1_to_1e-3_2_layer_enc_8_layer_with_fixed_linear_attn_04-28-0941/checkpoints/",
    }
    models_display_ids = {
        "20M_params_600k_polys_delta_tau_fixed_attn (checkpoint 04-28-0941)": "20M Params, trained on 600k polys, with delta tau, fixed linear attn (checkpoint 04-28-0941)",
    }

    return model_dicts, models_display_ids
