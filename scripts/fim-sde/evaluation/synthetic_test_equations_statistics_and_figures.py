import itertools
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Optional

import optree
import torch
from dataloader_inits.synthetic_test_equations import (
    get_cspd_dataloaders_inits,
    get_opper_or_wang_dataloaders_inits,
    get_svise_dataloaders_inits,
)
from model_dicts.models_trained_on_600k_deg_3_drift_deg_2_diffusion import get_model_dicts_600k_deg_3_drift_deg_2_diff
from tqdm import tqdm

from fim import project_path
from fim.models.sde import FIMSDE
from fim.pipelines.sde_sampling_from_model import fimsde_sample_paths
from fim.utils.evaluation_sde import (
    EvaluationConfig,
    ModelEvaluation,
    evaluate_with_step_function,
    get_maps_from_dicts,
    load_evaluations,
    run_model_evaluations,
    save_evaluations,
)
from fim.utils.evaluation_sde_synthetic_datasets import (
    add_vector_field_metrics,
    get_synthetic_dataloader,
    synthetic_data_metrics,
    synthetic_data_plots,
    synthetic_dataset_statistics,
)


def evaluate_sde_model_step(model: FIMSDE, batch: dict, device, sample_paths: Optional[bool] = True) -> dict:
    """
    Evaluate FIMSDE on one batch. Batch comes from PaddedFIMSDEDataset.

    Returns dict with evaluation results from batch, including:
        estimated_concepts (SDEConcepts): FIMSDE output from evaluation on batch.
        sample_paths, sample_paths_grid (Tensor): Sample paths from model.
    """
    batch = optree.tree_map(lambda x: x.to(device), batch, namespace="fimsde")

    # get vector fields at locations
    estimated_concepts = model(batch, training=False, return_losses=False)
    estimated_concepts = optree.tree_map(lambda x: x.to("cpu"), estimated_concepts, namespace="fimsde")
    step_results = {"estimated_concepts": estimated_concepts}

    # optionally: get sample paths
    if sample_paths is True:
        initial_time = torch.amin(torch.where(batch["obs_mask"].bool(), batch["obs_times"], torch.inf), dim=-2)
        end_time = torch.amax(torch.where(batch["obs_mask"].bool(), batch["obs_times"], -torch.inf), dim=-2)
        sample_paths, sample_paths_grid = fimsde_sample_paths(
            model, batch, initial_time=initial_time, end_time=end_time, grid_size=batch["obs_times"].shape[-2]
        )
        sample_paths, sample_paths_grid = optree.tree_map(lambda x: x.to("cpu"), (sample_paths, sample_paths_grid), namespace="fimsde")

        step_results.update({"sample_paths": sample_paths, "sample_paths_grid": sample_paths_grid})

    return step_results


if __name__ == "__main__":
    # ------------------------------------ General Setup ------------------------------------------------------------------------------ #
    dataset_descr = "synthetic_datasets"

    # How to name experiments
    # experiment_descr = "models_trained_on_deg_3_drift_deg_0_diff_streaming_dataloader_development"
    # experiment_descr = "30k_deg_3_ablation_studies_extended"
    # experiment_descr = "450k_deg_3_drift_deg_0_diffusion"
    # experiment_descr = "30k_deg_3_ablation_studies_only_learn_scale"
    experiment_descr = "600k_deg_3_drift_deg_2_diffusion_larger_model"

    model_dicts, models_display_ids = get_model_dicts_600k_deg_3_drift_deg_2_diff()

    # train_test_split_dir = [
    #     "/lustre/scratch/data/seifnerp_hpc-fim_data/data/processed/train/sde-drift-deg-3-diffusion-deg-0-50-paths/test/0_test_deg_2",
    #     "/lustre/scratch/data/seifnerp_hpc-fim_data/data/processed/train/sde-drift-deg-3-diffusion-deg-0-50-paths/test/1_test_deg_3",
    #     "/lustre/scratch/data/seifnerp_hpc-fim_data/data/processed/train/sde-drift-deg-3-diffusion-deg-0-50-paths/test/2_test_deg_1",
    # ]
    train_test_split_dir = None

    results_to_load: list[str] = [
        # "/home/cvejoski/Projects/FoundationModels/FIM/evaluations/synthetic_datasets/01171007_testing/model_evaluations",
        # "/home/seifner/repos/FIM/evaluations/synthetic_datasets/01231533_600k_deg_3_drift_deg_2_diffusion/model_evaluations"
    ]

    BATCH_SIZE = 128
    NUM_WORKERS = 8
    dataset_stats_precision = 2
    indices_to_plot = 20  # per equation
    random_indices = False
    sample_model_paths = True

    plot_max_ground_truth_paths = None  # number of paths to plot
    plot_max_model_paths = 50
    # --------------------------------------------------------------------------------------------------------------------------------- #

    # Save dir setup: project_path / evaluations / synthetic_datasets / time_stamp + _ + experiment_descr
    evaluation_path = Path(project_path) / "evaluations"
    time: str = str(datetime.now().strftime("%m%d%H%M"))
    evaluation_dir: Path = evaluation_path / dataset_descr / (time + "_" + experiment_descr)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    # Get dataloaders inits and their display ids (for ModelEvaluation)
    svise_dataloaders_dict, svise_dataloaders_display_ids = get_svise_dataloaders_inits(
        Path("/cephfs_projects/foundation_models/data/SDE/test/20250124_svise/sde_relative_diffusion_1_perc_20_paths/"),
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    # poly_dataloaders_dict, poly_dataloaders_display_ids = get_up_to_deg_3_polynomial_test_sets_init(
    #     batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    # )
    opper_or_wang_dataloaders_dict, opper_or_wang_dataloaders_display_ids = get_opper_or_wang_dataloaders_inits(
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )
    cspd_dataloaders_dict, cspd_dataloaders_display_ids = get_cspd_dataloaders_inits(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    # difficult_dataloaders_dict, difficult_dataloaders_display_ids = get_difficult_synth_equations_dataloaders_inits(
    #     batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    # )

    # choose data to evaluate
    dataloader_dict = {}
    dataloader_display_ids = {}

    dataloader_dict.update(svise_dataloaders_dict)
    dataloader_display_ids.update(svise_dataloaders_display_ids)

    dataloader_dict.update(opper_or_wang_dataloaders_dict)
    dataloader_display_ids.update(opper_or_wang_dataloaders_display_ids)

    dataloader_dict.update(cspd_dataloaders_dict)
    dataloader_display_ids.update(cspd_dataloaders_display_ids)

    # dataloader_dict.update(difficult_dataloaders_dict)
    # dataloader_display_ids.update(difficult_dataloaders_display_ids)

    # dataloader_dict.update(poly_dataloaders_dict)
    # dataloader_display_ids.update(poly_dataloaders_display_ids)

    if train_test_split_dir is not None:
        test_dl = get_synthetic_dataloader(train_test_split_dir, BATCH_SIZE, NUM_WORKERS)
        test_dl_dict = {"test_split_of_train_set": test_dl}
        test_dl_display_dict = {"test_split_of_train_set": "Test split of train set"}

        dataloader_dict.update(test_dl_dict)
        dataloader_display_ids.update(test_dl_display_dict)

    # Setup inits for models and dataloaders
    model_map, dataloader_map, models_display_id_map, dataloaders_display_id_map = get_maps_from_dicts(
        model_dicts, dataloader_dict, models_display_ids, dataloader_display_ids
    )

    # Load previous evaluations that don't need to be evaluated anymore
    loaded_evaluations: list[ModelEvaluation] = load_evaluations(results_to_load)

    # Evaluate all models on all datasets
    all_evaluations: list[ModelEvaluation] = [
        ModelEvaluation(model_id, dataloader_id)
        for model_id, dataloader_id in itertools.product(model_dicts.keys(), dataloader_dict.keys())
    ]
    to_evaluate: list[ModelEvaluation] = [evaluation for evaluation in all_evaluations if evaluation not in loaded_evaluations]

    # Create, run and save EvaluationConfig
    _evaluate_sde_model_step = partial(evaluate_sde_model_step, sample_paths=sample_model_paths)
    evaluate_sde_model = partial(evaluate_with_step_function, step_func=_evaluate_sde_model_step)
    evaluated: list[ModelEvaluation] = run_model_evaluations(to_evaluate, model_map, dataloader_map, evaluate_sde_model)

    # Add loaded evaluations
    all_evaluations: list[ModelEvaluation] = loaded_evaluations + evaluated

    evaluation_config = EvaluationConfig(model_map, dataloader_map, evaluation_dir, models_display_ids, dataloader_display_ids)

    # Add metrics
    metrics = ["mse", "mae", "norm_mse", "norm_mae"]
    estimated_concepts_key = "estimated_concepts"
    precision = 2
    all_evaluations: list[ModelEvaluation] = [
        add_vector_field_metrics(evaluation_config, model_eval, metrics, estimated_concepts_key) for model_eval in all_evaluations
    ]

    save_evaluations(all_evaluations, evaluation_dir / "model_evaluations")

    # Postprocessing of results from task
    synthetic_dataset_statistics(evaluation_config, all_evaluations, precision=dataset_stats_precision)  # many statistics
    synthetic_data_metrics(evaluation_config, all_evaluations, precision)

    # Figures with subplot grid containing results from multiple equations per dataset
    if sample_model_paths is True:
        for model_evaluation in (pbar := tqdm(all_evaluations, total=len(all_evaluations), leave=False)):
            pbar.set_description(
                f"Saving figure grids for model {model_evaluation.model_id} and dataloader {model_evaluation.dataloader_id}."
            )
            synthetic_data_plots(
                evaluation_config,
                model_evaluation,
                plot_indices=indices_to_plot,
                random_indices=random_indices,
                max_ground_truth_paths=plot_max_ground_truth_paths,
                max_model_paths=plot_max_model_paths,
            )

        pbar.close()
