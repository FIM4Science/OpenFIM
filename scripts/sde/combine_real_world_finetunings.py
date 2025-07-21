import json
from pathlib import Path


def combine_outputs(results_dir: Path, epoch: str):
    results_dir = Path(results_dir)

    all_result_files = list(finetuning_results_dir.glob("*.json"))

    epoch_result_files = []
    for file in all_result_files:
        stripped_file_name = str(file).split(".")[-2]
        if stripped_file_name.endswith(epoch):
            epoch_result_files.append(file)

    epoch_results = [json.load(open(file, "r")) for file in epoch_result_files]

    epoch_results_json = json.dumps(epoch_results)

    with open(finetuning_results_dir / f"combined_outputs_{epoch}.json", "w") as f:
        f.write(epoch_results_json)


if __name__ == "__main__":
    # epochs = ["epoch_49", "epoch_99", "epoch_199", "epoch_499", "best_model"]
    epochs = ["epoch_9", "epoch_19", "epoch_29", "epoch_39", "epoch_49", "epoch_59", "epoch_69", "epoch_79", "epoch_89", "epoch_99"]

    finetuning_results_dir: Path = Path(
        "/cephfs/users/seifner/repos/FIM/evaluations/real_world_cross_validation_vf_and_paths_evaluation/finetune_one_step_ahead_one_em_step_nll_512_points_lr_1e_6_every_10_epochs/"
    )

    for epoch in epochs:
        combine_outputs(finetuning_results_dir, epoch)
