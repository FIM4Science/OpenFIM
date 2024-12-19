from pathlib import Path


##########################   On Marvin ##################################################################################################


def get_model_dicts_20241222_trained_on_15k_deg2_drift_deg1_diffusion() -> tuple[dict, dict, str]:
    """
    Set up evaluation for first model trained on 15k drift degree 2 and diffusion degree 1 (+bernoulli 0.5).
    """
    # Models to load
    model_dict = {"trained_on_15k_up_to_deg_2_drift": "/home/seifnerp_hpc/repos/FIM/results/test_new_dataloader_12-22-1925/checkpoints"}
    model_display_ids = {"trained_on_15k_up_to_deg_2_drift": "Trained on up to deg2 drift."}

    # How to name experiments
    experiment_descr = "model_trained_on_15k_up_to_deg_2_drift_deg_1_diffusion"

    return model_dict, model_display_ids, experiment_descr


def get_model_dicts_20241224_trained_on_15k_deg_2_drift_deg_0_diffusion_ablation_studies_architecture_and_nrmse() -> tuple[dict, dict, str]:
    """
    Set up evaluation for ablation studies on 15k degree 2 drift and constant diffusion.
    Ablations include:
    """

    base_dir = Path(
        "/home/seifnerp_hpc/repos/FIM/saved_results/20241224_trained_on_15k_deg_2_drift_deg_0_diffusion_ablation_studies_architecture_and_nrmse"
    )

    # Models to load
    model_dict = {
        "base_model_rmse_obj_delta_time_enc_mlp_location_and_obs_enc": "15k_deg_2_drift_deg_1_diff_rmse_delta_time_only_mlp_location_enc_base_model_12-25-0036",
        "rmse_obj_delta_time_enc_sine_time_enc_for_location_and_obs_enc": "15k_deg_2_drift_deg_1_diff_rmse_delta_time_only_sine_location_enc_12-25-0757",
        "rmse_obj_delta_time_enc_linear_enc_for_locataion_and_obs_enc": "15k_deg_2_drift_deg_1_diff_rmse_delta_time_only_linear_location_enc_12-25-1219",
        "rmse_obj_absolute_time_enc_mlp_location_and_obs_enc": "15k_deg_2_drift_deg_1_diff_rmse_absolute_time_mlp_location_enc_base_model_12-25-0758",
        "normalized_rmse_obj_delta_time_enc_mlp_location_and_obs_enc": "15k_deg_2_drift_deg_1_diff_normalized_rmse_denom_shifted_delta_time_only_mlp_location_enc_base_model_12-25-1136",
        "rmse_obj_delta_time_enc_mlp_location_and_obs_enc_instance_standardize": "15k_deg_2_drift_deg_1_diff_rmse_delta_time_only_mlp_location_enc_standardization_12-26-1930",
        "rmse_obj_delta_time_enc_mlp_locations_and_obs_enc_no_psi_1": "15k_deg_2_drift_deg_1_diff_rmse_delta_time_only_mlp_location_enc_no_psi_1_transfomer_tests_12-27-1559",
        "rmse_obj_delta_time_enc_mlp_locations_and_obs_enc_no_psi_1_larger_nets": "15k_deg_2_drift_deg_1_diff_rmse_delta_time_only_linear_location_enc_no_psi_1_transfomer_model_embd_256_hidden_layers_256_8_heads_12-27-1601",
    }
    model_dict = {key: base_dir / exp_name / "checkpoints" for key, exp_name in model_dict.items()}

    model_display_ids = {
        "base_model_rmse_obj_delta_time_enc_mlp_location_and_obs_enc": "Base Model. Obj: RMSE, Time enc.: Delta only, Spatial enc.: MLP",
        "rmse_obj_delta_time_enc_sine_time_enc_for_location_and_obs_enc": "Ablation: Spatial enc.: Sine(Time)Encoding",
        "rmse_obj_delta_time_enc_linear_enc_for_locataion_and_obs_enc": "Ablation: Spatial enc.: Linear",
        "rmse_obj_absolute_time_enc_mlp_location_and_obs_enc": "Ablation: Time enc.: Absolute + Delta",
        "normalized_rmse_obj_delta_time_enc_mlp_location_and_obs_enc": "Ablation: Obj: Normalized RMSE",
        "rmse_obj_delta_time_enc_mlp_location_and_obs_enc_instance_standardize": "Ablation: Instance Standardization",
        "rmse_obj_delta_time_enc_mlp_locations_and_obs_enc_no_psi_1": "Ablation: No Psi1",
        "rmse_obj_delta_time_enc_mlp_locations_and_obs_enc_no_psi_1_larger_nets": "Ablation: No Psi1, More Params",
    }

    # How to name experiments
    experiment_descr = "model_trained_on_15k_up_to_deg_2_drift_deg_0_diffusion_ablation_studies_architecture_and_nrmse"

    return model_dict, model_display_ids, experiment_descr


def get_model_dicts_20241228_trained_on_15k_deg_2_drift_deg_0_diffusion_100_paths() -> tuple[dict, dict, str]:
    """
    Set up evaluation for ablation studies on 15k degree 2 drift and constant diffusion.
    Ablations include:
    """

    base_dir = Path(
        "/home/seifnerp_hpc/repos/FIM/saved_results/20241224_trained_on_15k_deg_2_drift_deg_0_diffusion_ablation_studies_architecture_and_nrmse"
    )

    # Models to load
    model_dict = {
        "trained_on_100_paths_max": "15k_deg_2_drift_deg_1_diff_rmse_delta_time_only_mlp_location_enc_100_paths_12-27-1210",
    }
    model_dict = {key: base_dir / exp_name / "checkpoints" for key, exp_name in model_dict.items()}

    model_display_ids = {
        "trained_on_100_paths_max": "Trained on 100 paths",
    }

    # How to name experiments
    experiment_descr = "model_trained_on_100_paths_evaluated_on_100_paths_if_possible"

    return model_dict, model_display_ids, experiment_descr


def get_model_dicts_20241230_trained_on_30k_deg_3_drift_deg_0_diffusion_50_paths() -> tuple[dict, dict, str]:
    """
    Set up evaluation for ablation studies on 30k degree 3 drift and constant diffusion.
    Ablations include:
    """

    base_dir = Path("/home/seifnerp_hpc/repos/FIM/saved_results/20241231_trained_on_30k_deg_3_drift_deg_0_diffusion")

    # Models to load
    model_dict = {
        # "linear_output_projection_1000_threshold": "30K_deg_3_drift_deg_0_diff_linear_out_projection_norm_threshold_1000_50_gridpoints_12-30-2235",
        "linear_output_projection_1000_threshold_finetuned_larger_hypercube": "30K_deg_3_drift_deg_0_diff_linear_out_projection_norm_threshold_1000_50_gridpoints_fintune_larger_hypercube_12-31-1045",
        "linear_output_projection_1000_threshold_set_transformer_psi1": "30K_deg_3_drift_deg_0_diff_linear_out_projection_set_transformer_psi_1",
        " 1000_threshold_GNOT_repeated_location_query_4_layers": "30K_deg_3_drift_deg_0_diff_mlp_out_projection_GNOT_repeated_location_query_attention_01-02-0913",
        "1000_threshold_linear_softmax_attn_4_layers": "linear_self_attention_softmax_feature_map_with_normalization_4_layers_01-02-1534",
        "1000_threshold_linear_softmax_attn-2_layers_GNOT_repeated_4_layers": "30K_deg_3_drift_deg_0_diff_mlp_out_projection_GNOT_repeated_location_query_attention_separate_encoders_2_layer_linear_self_attention_01-02-1850",
    }
    model_dict = {key: base_dir / exp_name / "checkpoints" for key, exp_name in model_dict.items()}

    model_display_ids = {
        # "linear_output_projection_1000_threshold": "Linear Output Projection, 1000 Threshold",
        "linear_output_projection_1000_threshold_finetuned_larger_hypercube": "Linear Output Projection, 1000 Threshold, Finetuned larger hypercube",
        "linear_output_projection_1000_threshold_set_transformer_psi1": "Linear Output Projection, 1000 Threshold, Set Transformer Psi1",
        " 1000_threshold_GNOT_repeated_location_query_4_layers": "GNOT, 4 times location query layers",
        "1000_threshold_linear_softmax_attn_4_layers": "Linear Softmax Self-Attention",
        "1000_threshold_linear_softmax_attn-2_layers_GNOT_repeated_4_layers": "Lin. Softmax Self-Attn. 2 Layers + GNOT, 4 times location query",
    }

    # How to name experiments
    experiment_descr = "models_trained_on_deg_3_drift_deg_0_diff"

    return model_dict, model_display_ids, experiment_descr
