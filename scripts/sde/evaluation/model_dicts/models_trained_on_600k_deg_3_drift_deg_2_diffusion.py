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
        "20M_params_trained_even_longer": "/cephfs_projects/foundation_models/models/FIMSDE/20M_parameter_model_for_icml_submission/checkpoints",
        # "20M_params_trained_even_longer": "/home/seifner/repos/FIM/saved_results/20250124_20M_params_trained_on_600k_deg_3/600k_drift_deg_3_diff_deg_2_mse_no_div_by_diff_softplus_2_layer_enc_8_layer_dec_continued_01-25-1301_trained_even_longer/checkpoints",
        # "20M_params_cont_train_unary_binary": "/home/seifner/repos/FIM/saved_results/20250124_20M_params_trained_on_600k_deg_3/600k_drift_deg_3_diff_deg_2_mse_no_div_by_diff_softplus_2_layer_enc_8_layer_dec_continued_more_data_NOT_USED_IN_PAPER_01-29-2111/checkpoints",
        "20M_params_paper_model_cont_training_unary_binary": "/cephfs_projects/foundation_models/models/FIMSDE/600k_drift_deg_3_diff_deg_2_cont_icml_submission_model_with_unary_binary_data/checkpoints",
    }
    models_display_ids = {
        "20M_params_trained_even_longer": "20M Paramters trained even longer (Paper)",
        # "20M_params_trained_even_longer": "20M Params, Paper Model",
        # "20M_params_cont_train_unary_binary": "20M Params, cont. train on unary-binary trees",
        "20M_params_paper_model_cont_training_unary_binary": "20M Paras, cont. train mixed polynomials and unary-binary",
    }

    return model_dicts, models_display_ids
