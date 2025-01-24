# if __name__ == "__main__":
#     # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
#     eval_description = "icml_ablation_studies"
#
#     # How to name experiments
#     experiment_descr = "develop"
#
#     model_dicts, models_display_ids = get_model_dicts_600k_deg_3_drift_deg_2_diff()
#
#     results_to_load: list[str] = [
#         # "/home/seifner/repos/FIM/evaluations/2D_dense_data_from_wang_opper_svise/01281104_1000_paths_500_length_initial_states_from_paths_subsampled_1_or_10/model_evaluations"
#     ]
#
#     # data to load
#     path_to_data = Path("/home/seifner/repos/FIM/data/processed/test/20250126_2D_dense_data_from_wang_opper_svise/systems_data/")
#
#     # --------------------------------------------------------------------------------------------------------------------------------- #
#
#     # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
#     evaluation_path = Path(project_path) / "evaluations"
#     time: str = str(datetime.now().strftime("%m%d%H%M"))
#     evaluation_dir: Path = evaluation_path / dataset_descr / (time + "_" + experiment_descr)
#     evaluation_dir.mkdir(parents=True, exist_ok=True)
#
#     # Get dataloaders inits and their display ids (for ModelEvaluation)
#     dataloaders = {
#         (system, stride): get_dataset_from_opper_generated_data(path_to_data / system, stride, num_initial_states=sample_paths_count)
#         for system in systems_to_load
#         for stride in strides
#     }
#     dataloaders_display_ids = {system: system for system in systems_to_load}  # for now
#
#     # Setup inits for models and dataloaders
#     model_map = model_map_from_dict(model_dicts)
#
#     # Load previous evaluations that don't need to be evaluated anymore
#     loaded_evaluations: list[ModelEvaluation] = load_evaluations(results_to_load)
#
#     # Evaluate all models on all datasets
#     all_evaluations: list[ModelEvaluation] = [
#         ModelEvaluation(model_id, dataloader_id) for model_id, dataloader_id in itertools.product(model_dicts.keys(), dataloaders.keys())
#     ]
#     to_evaluate: list[ModelEvaluation] = [evaluation for evaluation in all_evaluations if evaluation not in loaded_evaluations]
#
#     # Add model evaluations from GP model
#     gp_model_evaluations: list[ModelEvaluation] = load_gp_model_evaluations(path_to_gp_results, systems_to_load, dataloaders)
#
#     # Create, run and save evaluations
#     evaluated: list[ModelEvaluation] = run_evaluations(
#         to_evaluate, model_map, dataloaders, sample_model_paths, sample_paths_count, dt, sample_path_steps
#     )
#
#     # Add loaded evaluations
#     all_evaluations: list[ModelEvaluation] = loaded_evaluations + evaluated
#     save_evaluations(all_evaluations, evaluation_dir / "model_evaluations")
#     all_evaluations = all_evaluations + gp_model_evaluations
