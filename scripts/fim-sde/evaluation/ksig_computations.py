from datetime import datetime
from pathlib import Path

import tabulate
from compute_mmd import get_mmd
from dataloader_inits.synthetic_test_equations import (
    get_cspd_dataloaders_inits,
    get_difficult_synth_equations_dataloaders_inits,
    get_opper_or_wang_dataloaders_inits,
    get_svise_dataloaders_inits,
    get_up_to_deg_3_polynomial_test_sets_init,
)
from model_dicts.development_models_deg_3_drift import (
    get_model_dicts_20241230_trained_on_30k_deg_3_drift_deg_0_diffusion_50_paths_streaming_dataloader,
)

from fim import project_path
from fim.utils.evaluation_sde import (
    EvaluationConfig,
    ModelEvaluation,
    get_data_from_model_evaluation,
    get_maps_from_dicts,
    load_evaluations,
    save_evaluations,
)


def compute_mmd(model_evaluation, evaluation_config):
    model_sample_paths = model_evaluation.results["sample_paths"]  # [1, num_paths, num_steps, num_dim]
    # model_sample_path_grid = model_evaluation.results["sample_paths_grid"]
    ground_truth_data = get_data_from_model_evaluation(model_evaluation, evaluation_config)
    ground_truth_sample_paths = ground_truth_data["obs_values"]  # [1, num_paths, num_steps, num_dim]
    # ground_truth_sample_path_grid = ground_truth_data["obs_times"]
    # assert torch.allclose(model_sample_path_grid, ground_truth_sample_path_grid), f"Sample path grids do not match: {model_evaluation.__repr__()}"
    assert model_sample_paths.shape == ground_truth_sample_paths.shape, f"Sample paths do not match in shape {model_evaluation.__repr__()}"
    mmd = get_mmd(model_sample_paths[0], ground_truth_sample_paths[0])
    model_evaluation.results["mmd"] = mmd

    return model_evaluation


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    BATCH_SIZE = 128
    NUM_WORKERS = 8

    # How to name experiments
    experiment_descr = "testing"

    # Asumption: Models are evaluated already, paths are generated, saved as ModelEvaluation
    results_to_load: list[str] = [
        "synthetic_datasets/01151152_testing/model_evaluations",
    ]

    # get model_dicts and dataloader_ids results have been generated with
    streaming_model_dict, streaming_model_display_id = (
        get_model_dicts_20241230_trained_on_30k_deg_3_drift_deg_0_diffusion_50_paths_streaming_dataloader()
    )
    svise_dataloaders_dict, svise_dataloaders_display_ids = get_svise_dataloaders_inits(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    poly_dataloaders_dict, poly_dataloaders_display_ids = get_up_to_deg_3_polynomial_test_sets_init(
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )
    opper_or_wang_dataloaders_dict, opper_or_wang_dataloaders_display_ids = get_opper_or_wang_dataloaders_inits(
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )
    cspd_dataloaders_dict, cspd_dataloaders_display_ids = get_cspd_dataloaders_inits(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    difficult_dataloaders_dict, difficult_dataloaders_display_ids = get_difficult_synth_equations_dataloaders_inits(
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    # choose data to evaluate
    dataloader_dict = {}
    dataloader_display_ids = {}

    dataloader_dict.update(svise_dataloaders_dict)
    dataloader_display_ids.update(svise_dataloaders_display_ids)

    dataloader_dict.update(opper_or_wang_dataloaders_dict)
    dataloader_display_ids.update(opper_or_wang_dataloaders_display_ids)

    dataloader_dict.update(cspd_dataloaders_dict)
    dataloader_display_ids.update(cspd_dataloaders_display_ids)

    dataloader_dict.update(difficult_dataloaders_dict)
    dataloader_display_ids.update(difficult_dataloaders_display_ids)

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

    # Setup inits for models and dataloaders
    model_map, dataloader_map, models_display_id_map, dataloaders_display_id_map = get_maps_from_dicts(
        models_dicts, dataloader_dict, models_display_ids, dataloader_display_ids
    )

    # Load evaluations
    loaded_evaluations: list[ModelEvaluation] = load_evaluations(results_to_load)

    # Create EvaluationConfig
    evaluation_config = EvaluationConfig(model_map, dataloader_map, evaluation_dir, models_display_ids, dataloader_display_ids)

    # compute ksig per ModelEvaluation
    all_evaluations = []
    for model_evaluation in loaded_evaluations:
        if "mmd" not in model_evaluation.results.keys():  # check if it has been compute before
            ### add ksig to model evaluation
            model_evaluation: ModelEvaluation = compute_mmd(model_evaluation, evaluation_config)

        all_evaluations.append(model_evaluation)

    # Save
    save_evaluations(all_evaluations, evaluation_dir / "model_evaluations")

    # Print a tabulate table
    evaluation_names = [model_evaluation.__repr__() for model_evaluation in all_evaluations]
    mmds = [model.results["mmd"] for model in all_evaluations]
    table = tabulate.tabulate(zip(evaluation_names, mmds), headers=["Model", "MMD"], tablefmt="fancy_grid")

    # print table to file
    with open(evaluation_dir / "mmd_table.txt", "w") as f:
        f.write(table)
