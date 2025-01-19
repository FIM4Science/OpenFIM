def get_model_dicts_600k_deg_3_drift_deg_2_diff() -> tuple[dict, dict, str]:
    """
    Set up evaluation for ablation studies on 30k degree 3 drift and degree 2 diffusion.
    Ablations include:
    """
    model_dict = {
        "11M_params": "/home/seifner/repos/FIM/saved_results/20250120_11M_params_trained_on_600k_deg_3/600k_drift_deg_3_diff_deg_2_embed_size_256_cont_with_grad_clip_lr_1e-5_weight_decay_1e-4_01-19-2228/checkpoints",
        "20M_params": "/home/seifner/repos/FIM/saved_results/20250120_11M_params_trained_on_600k_deg_3/600k_drift_deg_3_diff_deg_2_mse_no_div_by_diff_softplus_2_layer_enc_8_layer_dec_01-21-1106/checkpoints",
        "20M_params_trained_longer": "/home/seifner/repos/FIM/results/600k_drift_deg_3_diff_deg_2_mse_no_div_by_diff_softplus_2_layer_enc_8_layer_dec_01-21-1106/checkpoints",
    }

    model_display_ids = {
        "11M_params": "11M Parameters",
        "20M_params": "20M Parameters",
        "20M_params_trained_longer": "20M Parameters trained longer",
    }

    return model_dict, model_display_ids
