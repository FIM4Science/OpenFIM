import json
from pathlib import Path

import torch


if __name__ == "__main__":
    # !!! CAREFUL: CODE CHANGES FILES !!!
    # !!! Only apply this script to copies of saved checkpoints !!!

    path_to_checkpoint_dir = Path(
        # "/cephfs_projects/foundation_models/models/FIMSDE/NeurIPS_submission_models/600k_drift_deg_3_diff_deg_2_delta_tau_fixed_linear_attn_softmax_no_extra_normalization_and_fix_in_residual_layer_05-06-2300_for_refactored_model_code/checkpoints/epoch-139_refactored"
        "/cephfs_projects/foundation_models/models/FIMSDE/Post_NeurIPS_models/600k_32_locations_at_observations_32_locations_randomly_07-14-1850_for_refactored_model_code/checkpoints/epoch-139"
    )

    path_to_model_config_json = path_to_checkpoint_dir / "config.json"
    path_to_model_checkpoint = path_to_checkpoint_dir / "model-checkpoint.pth"

    # remove unused keys from config
    model_config: dict = json.load(open(path_to_model_config_json, "r"))

    model_config.pop("delta_time_only", None)
    model_config.pop("detach_learnable_loss_scale_heads", None)
    model_config.pop("divide_drift_loss_by_diffusion", None)
    model_config.pop("dt_pipeline", None)
    model_config.pop("evaluate_with_unnormalized_heads", None)
    model_config.pop("layer_norms_in_phi_0", None)
    model_config.pop("learn_vf_var", None)
    model_config.pop("loss_filter_nans", None)
    model_config.pop("loss_type", None)
    model_config.pop("non_negative_diffusion_by", None)
    model_config.pop("number_of_time_steps_pipeline", None)
    model_config.pop("operator_specificity", None)
    model_config.pop("separate_phi_0_encoders", None)
    model_config.pop("single_learnable_loss_scale_head", None)
    model_config.pop("times_norm_on_deltas", None)
    model_config.pop("train_with_normalized_head", None)

    with open(path_to_model_config_json, "w") as f:
        json.dump(model_config, f)

    # rename scale operator module in checkpoint
    # prev: used same network for both scales, under different name
    # remove one and rename the other

    ckp = torch.load(path_to_model_checkpoint)

    keys = list(ckp.keys())
    for key in keys:
        if key.startswith("operator_loss_scale_diffusion"):
            ckp.pop(key)

    remaining_keys = list(ckp.keys())
    for key in remaining_keys:
        if key.startswith("operator_loss_scale_drift"):
            weight = ckp.pop(key)
            key = key.replace("operator_loss_scale_drift", "operator_loss_scale")
            ckp[key] = weight

    torch.save(ckp, path_to_model_checkpoint)
