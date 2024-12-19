from datetime import datetime
from pathlib import Path

from dataloader_inits.synthetic_test_equations import get_svise_dataloaders_inits
from model_dicts.development_models_deg_3_drift import (
    get_model_dicts_20241230_trained_on_30k_deg_3_drift_deg_0_diffusion_50_paths_streaming_dataloader,
)

from fim import project_path
from fim.utils.evaluation_sde import (
    EvaluationConfig,
    ModelEvaluation,
    get_maps_from_dicts,
    load_evaluations,
)


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    BATCH_SIZE = 128
    NUM_WORKERS = 8

    # How to name experiments
    experiment_descr = "testing"

    # Asumption: Models are evaluated already, paths are generated, saved as ModelEvaluation
    results_to_load: list[str] = [
        "synthetic_datasets/01062334_testing/model_evaluations",
    ]

    # get model_dicts and dataloader_ids results have been generated with
    streaming_model_dict, streaming_model_display_id = (
        get_model_dicts_20241230_trained_on_30k_deg_3_drift_deg_0_diffusion_50_paths_streaming_dataloader()
    )
    svise_dataloaders_dict, svise_dataloaders_display_ids = get_svise_dataloaders_inits(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # --------------------------------------------------------------------------------------------------------------------------------- #
    # Save dir setup: project_path / ksig / dataset_d / time_stamp + _ + experiment_descr
    evaluation_path = Path(project_path) / "evaluations"
    time: str = str(datetime.now().strftime("%m%d%H%M"))
    evaluation_dir: Path = evaluation_path / "ksig" / (time + "_" + experiment_descr)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # combine all models dicts
    models_dicts = {}
    models_display_ids = {}

    models_dicts.update(streaming_model_dict)
    models_display_ids.update(streaming_model_display_id)

    # combine all dataloader dicts
    dataloaders_dict = {}
    dataloaders_display_id = {}

    dataloaders_dict.update(svise_dataloaders_dict)
    dataloaders_display_id.update(svise_dataloaders_display_ids)

    # Setup inits for models and dataloaders
    model_map, dataloader_map, models_display_id_map, dataloaders_display_id_map = get_maps_from_dicts(
        models_dicts, dataloaders_dict, models_display_ids, dataloaders_display_id
    )

    # Load evaluations
    loaded_evaluations: list[ModelEvaluation] = load_evaluations(results_to_load)

    # Create EvaluationConfig
    evaluation_config = EvaluationConfig(model_map, dataloader_map, evaluation_dir, models_display_ids, dataloaders_display_id)

    # compute ksig per ModelEvaluation
    # all_evaluations = []
    # for model_evaluation in loaded_evaluations:
    #     if "ksig" not in model_eval.results.keys():  # check if it has been compute before
    #         ### add ksig to model evaluation
    #         model_eval: ModelEvaluation = compute_ksig(model_evaluation, evaluation_config)
    #
    #     all_evaluations.append(model_eval)

    # # Save
    # save_evaluations(all_evaluations, evaluation_dir / "model_evaluations")

    # Maybe create table?
