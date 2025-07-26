import json
from pathlib import Path

import numpy as np


def _check_shape(result: dict):
    """
    Paths should have 4 dimensions: [1, B, T, 1]
    """

    paths = np.array(result["synthetic_paths"])

    if paths.ndim == 3:
        paths = paths[None, :, :, :]

    elif paths.ndim != 4:
        raise ValueError(
            f"Result name {result['name']}, split {result['split']} of has ndim {paths.ndim} with shape {paths.shape}. Expect 4 dims."
        )

    if not (paths.shape[0] == 1 and paths.shape[-1] == 1):
        raise ValueError(f"Result name {result['name']}, split {result['split']} of has shape {paths.shape}. Expect [1, _, _, 1]. ")

    result["synthetic_paths"] = paths.tolist()

    return result


def combine_outputs(results_dir: Path, epoch: str):
    results_dir = Path(results_dir)

    all_result_files = list(finetuning_results_dir.glob("*.json"))

    epoch_result_files = []
    for file in all_result_files:
        stripped_file_name = str(file).split(".")[-2]
        if stripped_file_name.endswith(epoch):
            epoch_result_files.append(file)

    epoch_results = [json.load(open(file, "r")) for file in epoch_result_files]

    if not len(epoch_results) == 20:
        raise ValueError(f"{results_dir=} has {len(epoch_results)} results of epoch {epoch}. Expect 20.")

    epoch_results = list(map(_check_shape, epoch_results))

    epoch_results_json = json.dumps(epoch_results)

    with open(finetuning_results_dir / f"combined_outputs_{epoch}.json", "w") as f:
        f.write(epoch_results_json)


if __name__ == "__main__":
    # epochs = ["epoch_49", "epoch_99", "epoch_199", "epoch_499", "best_model"]
    # epochs = ["epoch_9", "epoch_19", "epoch_29", "epoch_39", "epoch_49", "epoch_59", "epoch_69", "epoch_79", "epoch_89", "epoch_99"]
    # epochs = ["epoch_9"]
    # epochs = ["epoch_499", "epoch_999", "epoch_1999", "epoch_4999"]
    epochs = ["epoch_499", "epoch_999", "epoch_1999", "epoch_4999"]
    # epochs = [
    #     "epoch_9",
    #     "epoch_19",
    #     "epoch_29",
    #     "epoch_39",
    #     "epoch_49",
    #     "epoch_59",
    #     "epoch_69",
    #     "epoch_79",
    #     "epoch_89",
    #     "epoch_99",
    #     "epoch_199",
    #     "epoch_499",
    # ]
    finetuning_results_dir: Path = Path(
        # "/cephfs/users/seifner/repos/FIM/evaluations/real_world_cross_validation_vf_and_paths_evaluation/finetune_one_step_ahead_one_em_step_nll_512_points_lr_1e_6_every_10_epochs/"
        "/cephfs/users/seifner/repos/FIM/evaluations/real_world_cross_validation_vf_and_paths_evaluation/latent_sde_latent_dim_4_context_dim_100_decoder_NLL_train_subsplits_10/"
    )

    for epoch in epochs:
        combine_outputs(finetuning_results_dir, epoch)
