from fim.models.imputation_temporal import FIMImpTemp


# ode_path = Path("/cephfs_projects/foundation_models/models/FIMImpTempBase/fim_ode_minMax/model-checkpoint.pth")
# model = load_model_from_checkpoint(ode_path, module=FIMImpPointBase, for_eval=True)
# print(model)
# model.save_pretrained("/cephfs_projects/foundation_models/models/FIMImpTempBase/fim_ode_minMax_hf")
# model.push_to_hub("FIM4Science/fim-ode", private=False)

# model_checkpoint_path = Path("/cephfs_projects/foundation_models/models/FIMImpTempBase/fim_imputation_5windows_minMax/model-checkpoint.pth")
# model = load_model_from_checkpoint(model_checkpoint_path, module=FIMImpTempBase, for_eval=True)
# print(f"Model loaded from {model_checkpoint_path}")
# model.save_pretrained("/cephfs_projects/foundation_models/models/FIMImpTempBase/fim_imputation_5windows_minMax_hf")
# model.push_to_hub("FIM4Science/fim-imp-temporal-base", private=False)

# model_checkpoint_path = Path("/cephfs_projects/foundation_models/models/FIMImpTempBase/fim_imputation_5windows_minMax/model-checkpoint.pth")
# model = load_model_from_checkpoint(model_checkpoint_path, module=FIMImpTempBase, for_eval=True)
# print(f"Model loaded from {model_checkpoint_path}")
# model.save_pretrained("/cephfs_projects/foundation_models/models/FIMImpTempBase/fim_imputation_5windows_minMax_hf")
# model.push_to_hub("FIM4Science/fim-imp-temporal-base", private=False)

# model_config = {
#     "model_type": FIMImpTempConfig.model_type,
#      "fim_imputation": "FIM4Science/fim-imp-temporal-base",
#      "denoising_model": None,
# }

# model = ModelFactory.create(model_config)

# model.push_to_hub("FIM4Science/fim-imp-temporal", private=False)

model = FIMImpTemp.from_pretrained("FIM4Science/fim-imp-temporal")


print(model)
