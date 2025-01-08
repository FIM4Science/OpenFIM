def get_model_dicts_450k_deg_3_drift_deg_0_diff() -> tuple[dict, dict, str]:
    """
    Set up evaluation for ablation studies on 30k degree 3 drift and constant diffusion.
    Ablations include:
    """
    model_dict = {
        "survival_uniform_30k": "/home/seifner/repos/FIM/saved_results/20250108_30k_drift_deg_3_ablation_studies/30K_deg_3_drift_ablation_deg_mon_survive_uniformly_01-08-2351/checkpoints",
        "survival_uniform_450k": "/home/seifner/repos/FIM/saved_results/20250112_450k_deg_3_drift_deg_0_diffusion/450k_deg_3_drift_deg_0_diff_monomial_survive_uniform_2_lin_self_att_4_loc_query_01-11-1050/checkpoints",
    }

    model_display_ids = {
        "survival_uniform_30k": "Train data: 30k, Survival degree + monomial uniform",
        "survival_uniform_450k": "Train data: 450k, Survival degree + monomial uniform",
    }

    return model_dict, model_display_ids
