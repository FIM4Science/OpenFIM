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
