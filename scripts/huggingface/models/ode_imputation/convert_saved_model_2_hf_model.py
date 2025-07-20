from fim.models.imputation import FIMImputationWindowed


# ode_path = Path("/cephfs_projects/foundation_models/models/FIMODE/fim_ode_minMax/model-checkpoint.pth")
# model = load_model_from_checkpoint(ode_path, module=FIMODE, for_eval=True)
# print(model)
# model.save_pretrained("/cephfs_projects/foundation_models/models/FIMODE/fim_ode_minMax_hf")
# model.push_to_hub("FIM4Science/fim-ode", private=False)

# model_checkpoint_path = Path("/cephfs_projects/foundation_models/models/FIMImputation/fim_imputation_5windows_minMax/model-checkpoint.pth")
# model = load_model_from_checkpoint(model_checkpoint_path, module=FIMImputation, for_eval=True)
# print(f"Model loaded from {model_checkpoint_path}")
# model.save_pretrained("/cephfs_projects/foundation_models/models/FIMImputation/fim_imputation_5windows_minMax_hf")
# model.push_to_hub("FIM4Science/fim-imputation", private=False)

# model_checkpoint_path = Path("/cephfs_projects/foundation_models/models/FIMImputation/fim_imputation_5windows_minMax/model-checkpoint.pth")
# model = load_model_from_checkpoint(model_checkpoint_path, module=FIMImputation, for_eval=True)
# print(f"Model loaded from {model_checkpoint_path}")
# model.save_pretrained("/cephfs_projects/foundation_models/models/FIMImputation/fim_imputation_5windows_minMax_hf")
# model.push_to_hub("FIM4Science/fim-imputation", private=False)

# model_config = {
#     "model_type": FIMImputationWindowedConfig.model_type,
#      "fim_imputation": "FIM4Science/fim-imputation",
#      "denoising_model": None,
# }

# model = ModelFactory.create(model_config)

# model.push_to_hub("FIM4Science/fim-windowed-imputation", private=False)

model = FIMImputationWindowed.from_pretrained("FIM4Science/fim-windowed-imputation")


print(model)
