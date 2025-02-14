from pathlib import Path


def get_model_dicts_30k_deg_3_drift_ablation_studies() -> tuple[dict, dict, str]:
    """
    Set up evaluation for ablation studies on 30k degree 3 drift and constant diffusion.
    Ablations include:
    """
    base_dir = Path("/home/seifner/repos/FIM/saved_results/20250108_30k_drift_deg_3_ablation_studies")

    # Models to load
    model_dict = {
        "base_model": "30K_deg_3_drift_ablation_base_model",
        "survival_rate_degree_075_monomial_05": "30K_deg_3_drift_ablation_deg_wurv_rate_075_monomial_surv_rate_05_01-07-2314",
        "survival_rate_degree_05_monomial_025": "30K_deg_3_drift_ablation_deg_wurv_rate_05_monomial_surv_rate_025_01-08-1935",
        "diffusion_degree_2": "30K_deg_3_drift_ablation_degree_2_diffusion_01-07-2320",
        "init_cond_normal_with_uniformly_shifted_mean": "30K_deg_3_drift_ablation_init_cond_mean_from_uniform_01-08-0004",
        "init_cond_uniform": "30K_deg_3_drift_ablation_uniform_init_cond_01-07-2339",
        "rel_l2_norm_clipped_0_1": "30K_deg_3_drift_ablation_rel_l2_clipped_at_1_01-08-1109",
        "rel_l2_norm_clipped_1": "30K_deg_3_drift_ablation_rel_l2_clipped_at_1_01-08-1109",
        "trained_on_300_paths": "30K_deg_3_drift_ablation_300_paths_max_01-07-2330",
        "data_split_one_long_path": "30K_deg_3_drift_ablation_one_long_path_data_01-08-1930",
        "survival_uniform": "30K_deg_3_drift_ablation_deg_mon_survive_uniformly_01-08-2351",
        "no_target_norm_threshold": "30K_deg_3_drift_ablation_no_norm_threshold_in_objective_01-09-0001",
        "50_locations_per_batch": "30K_deg_3_drift_ablation_train_on_50_locations_01-09-0917",
        "more_params_lr_weight_decay": "30K_deg_3_drift_ablation_larger_nets_warmup_lr_decay_weight_decay_01-09-0856",
        "trained_on_degree_4_polynomials": "30K_deg_4_drift_ablation_train_on_50_locations_and_max_degree_4_01-11-1258",
        "divide_drift_loss_by_target_diffusion": "30K_deg_3_drift_ablation_train_on_50_locations_and_divide_drift_loss_by_target_diffusion_01-11-1736",
        "learnable_loss_scales_and_rmse": "30K_deg_3_drift_ablation_train_on_50_locations_and_learnable_loss_scales_and_rmse_01-12-0823",
        "trained_on_unnormalized_rmse": "30K_deg_3_drift_ablation_train_on_50_locations_and_unnormalized_rmse_01-12-1029",
        "dropout_0_2_weight_decay_1e_3": "30K_deg_3_drift_ablation_train_on_50_locations_dropout_0.2_weight_decay_1e-3_01-13-1049",
        "learn_loss_scale_single_detached_head": "30K_deg_3_drift_ablation_learn_loss_scales_combined_with_detached_head_01-13-1752",
        "learn_loss_scale_single_undetached_head": "30K_deg_3_drift_ablation_learn_loss_scales_combined_without_detached_head_01-13-1809",
        "learn_loss_scale_single_detached_head_4_layers": "30K_deg_3_drift_ablation_learn_loss_scales_combined_with_detached_head_4_layers_01-14-1152",
    }
    model_dict = {key: base_dir / exp_name / "checkpoints" for key, exp_name in model_dict.items()}

    # model_dict.update({
    #     "base_model": Path(
    #         "/home/seifner/repos/FIM/saved_results/20250106_sde_30k_deg_3_drift_deg_0_diff_2_layer_lin_att_4_layer_repeated_loc_query_streaming_dataloader/30K_deg_3_drift_deg_0_diff_mlp_out_projection_2_lin_self_att_4_repeated_loc_query_streaming_dataloader_contiguous_01-06-2114/checkpoints"
    #     )
    # })

    model_display_ids = {
        "base_model": "Base Model",
        "survival_rate_degree_075_monomial_05": "Train data: Survival rates: degree 0.75, monomial 0.5",
        "survival_rate_degree_05_monomial_025": "Train data: Survival rates: degree 0.5, monomial 0.25",
        "survival_uniform": "Train data: Survival degree + monomial uniform",
        "diffusion_degree_2": "Train data: Diffusion: degree 2",
        "init_cond_normal_with_uniformly_shifted_mean": "Train data: Init. cond.: normal, mean shifted uniformly [-10, 10]",
        "init_cond_uniform": "Train data: Init. cond.: uniform [-10, 10]",
        "trained_on_300_paths": "Train data: Max. paths: 300",
        "data_split_one_long_path": "Train data: split one long path into smaller paths",
        "trained_on_degree_4_polynomials": "Train data: Degree 4 drift",
        "rel_l2_norm_clipped_0_1": "Objective: Rel. l2, clipped at 0.1",
        "rel_l2_norm_clipped_1": "Objective: Rel. l2, clipped at 1",
        "divide_drift_loss_by_target_diffusion": "Objective: Divide drift loss by target diffusion",
        "learnable_loss_scales_and_rmse": "Objective: RMSE with learnable scales",
        "trained_on_unnormalized_rmse": "Objective: unnormalized RMSE",
        "no_target_norm_threshold": "Train setup: No target norm threshold",
        "50_locations_per_batch": "Train setup: 50 locations per batch",
        "more_params_lr_weight_decay": "Train setup: More Params, LR + Weight Decay",
        "dropout_0_2_weight_decay_1e_3": "Train setup: Dropout 0.2, Weight Decay 1e-3",
        "learn_loss_scale_single_detached_head": "Objective: learnable scales, single head, detached",
        "learn_loss_scale_single_undetached_head": "Objective: learnable scales, single head, NOT detached",
        "learn_loss_scale_single_detached_head_4_layers": "Objective: learnable scales, single head, detached, 4 Layers",
    }

    return model_dict, model_display_ids


def get_model_dicts_learn_scale() -> tuple[dict, dict, str]:
    """
    Set up evaluation for ablation studies only with learnable scale
    Ablations include:
    """
    base_dir = Path("/home/seifner/repos/FIM/saved_results/20250108_30k_drift_deg_3_ablation_studies")

    # Models to load
    model_dict = {
        "learnable_loss_scales_and_rmse": "30K_deg_3_drift_ablation_train_on_50_locations_and_learnable_loss_scales_and_rmse_01-12-0823",
        "single_detached_head": "30K_deg_3_drift_ablation_learn_loss_scales_combined_with_detached_head_01-13-1752",
        "single_undetached_head": "30K_deg_3_drift_ablation_learn_loss_scales_combined_without_detached_head_01-13-1809",
        "single_detached_head_4_layers": "30K_deg_3_drift_ablation_learn_loss_scales_combined_with_detached_head_4_layers_01-14-1152",
        "single_detached_head_rmse_divide_each_dim_by_diffusion": "30K_deg_3_drift_ablation_divide_drift_RMSE_by_diffusion_value_01-15-2345",
        "single_detached_head_rmse_without_division_by_dimension": "30K_deg_3_drift_ablation_RMSE_without_dividing_by_dim_01-15-2350",
        "single_detached_head_trained_on_mse": "30K_deg_3_drift_ablation_objective_MSE_train_objective_01-15-2359",
    }
    model_dict = {key: base_dir / exp_name / "checkpoints" for key, exp_name in model_dict.items()}

    # model_dict.update({
    #     "base_model": Path(
    #         "/home/seifner/repos/FIM/saved_results/20250106_sde_30k_deg_3_drift_deg_0_diff_2_layer_lin_att_4_layer_repeated_loc_query_streaming_dataloader/30K_deg_3_drift_deg_0_diff_mlp_out_projection_2_lin_self_att_4_repeated_loc_query_streaming_dataloader_contiguous_01-06-2114/checkpoints"
    #     )
    # })

    model_display_ids = {
        "learnable_loss_scales_and_rmse": "Objective: RMSE with learnable scales",
        "single_detached_head": "Objective: learnable scales, single head, detached",
        "single_undetached_head": "Objective: learnable scales, single head, NOT detached",
        "single_detached_head_4_layers": "Objective: learnable scales, single head, detached, 4 Layers",
        "single_detached_head_rmse_divide_each_dim_by_diffusion": "Objective: learnable scales, drift_loss / g.t.-diffusion per dimension",
        "single_detached_head_rmse_without_division_by_dimension": "Objective: learnable scales, RMSE, no division by dimension count",
        "single_detached_head_trained_on_mse": "Objective: learnable scales, MSE",
    }

    return model_dict, model_display_ids
