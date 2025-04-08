from pathlib import Path

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from loguru import logger
from tqdm import tqdm


SOURCE_SPLIT = "val"
PATH = "/cephfs_projects/foundation_models/hawkes/data/1D_easytpp/" + SOURCE_SPLIT

OUT_PATH = "/cephfs_projects/foundation_models/hawkes/data/"
OUT_PATH = Path(OUT_PATH)
OUT_PATH.mkdir(parents=True, exist_ok=True)
NUM_PROCESSES = 1

TOTAL_NUM_PATHS = 1200
if SOURCE_SPLIT == "train":
    NUM_PATHS_FOR_INFERENCE = 1200
    TOTAL_NUM_PROCESSES = 1
else:
    TOTAL_NUM_PROCESSES = 1
    NUM_PATHS_FOR_INFERENCE = 0
NUM_OF_PATHS_FOR_TEST = TOTAL_NUM_PATHS - NUM_PATHS_FOR_INFERENCE
idx = list(range(NUM_PROCESSES))
data = {}
for file in Path(PATH).rglob("*.pt"):
    with open(file, "rb") as f:
        logger.info(f"Loading file: {file}")
        tmp = torch.load(f, weights_only=False)
        if not isinstance(tmp, torch.Tensor) and file.stem != "intensities" and file.stem != "intensity_times":
            tmp = torch.tensor(np.array(tmp))
        data[file.stem] = tmp

our_to_easytpp = {
    "event_times": "time_since_start",
    "event_types": "type_event",
    "delta_times": "time_since_last_event",
    "kernel_evaluations": "target_kernel_evaluations",
    "kernel_grids": "target_kernel_grids",
    "base_intensities": "target_base_intensities",
    "intensities": "target_intensities",
    "intensity_times": "target_intensity_times",
}

delta_times = torch.diff(data["event_times"], dim=-2)
delta_times = torch.cat((torch.zeros_like(delta_times[..., :1, :]), delta_times), dim=-2)
data["delta_times"] = delta_times

for key, value in data.items():
    if isinstance(value, torch.Tensor):
        logger.info(f"Shape of {key}: {value.shape}")


for i, process_idx in tqdm(enumerate(idx), total=len(idx), desc="Processing processes"):
    train_data = {}
    test_data = {}
    if NUM_PATHS_FOR_INFERENCE != 0:
        train_data = {
            k: v[i][:NUM_PATHS_FOR_INFERENCE] if k == "intensity_times" or k == "intensities" or v.dim() == 4 else v[i]
            for k, v in data.items()
        }
        train_data["dim_process"] = torch.ones(train_data["event_times"].size(0), dtype=torch.int32) * train_data["event_times"].size(1)
        train_data["seq_len"] = torch.ones(train_data["event_times"].size(0), dtype=torch.int32) * train_data["event_times"].size(1)
        train_data["kernel_evaluations"] = train_data["kernel_evaluations"].unsqueeze(0).repeat(NUM_PATHS_FOR_INFERENCE, 1, 1)
        train_data["kernel_grids"] = train_data["kernel_grids"].unsqueeze(0).repeat(NUM_PATHS_FOR_INFERENCE, 1, 1)
        train_data["base_intensities"] = train_data["base_intensities"].unsqueeze(0).repeat(NUM_PATHS_FOR_INFERENCE, 1)
        train_data["event_times"] = train_data["event_times"].squeeze(-1)
        train_data["event_types"] = train_data["event_types"].squeeze(-1)
        train_data["seq_idx"] = torch.arange(NUM_PATHS_FOR_INFERENCE).int()

    if NUM_OF_PATHS_FOR_TEST != 0:
        test_data = {
            k: v[i][NUM_PATHS_FOR_INFERENCE:] if k == "intensity_times" or k == "intensities" or v.dim() == 4 else v[i]
            for k, v in data.items()
        }

        test_data["dim_process"] = torch.ones(test_data["event_times"].size(0), dtype=torch.int32) * test_data["event_times"].size(1)
        test_data["seq_len"] = torch.ones(test_data["event_times"].size(0), dtype=torch.int32) * test_data["event_times"].size(1)
        test_data["kernel_evaluations"] = test_data["kernel_evaluations"].unsqueeze(0).repeat(NUM_OF_PATHS_FOR_TEST, 1, 1)
        test_data["kernel_grids"] = test_data["kernel_grids"].unsqueeze(0).repeat(NUM_OF_PATHS_FOR_TEST, 1, 1)
        test_data["base_intensities"] = test_data["base_intensities"].unsqueeze(0).repeat(NUM_OF_PATHS_FOR_TEST, 1)
        test_data["event_times"] = test_data["event_times"].squeeze(-1)
        test_data["event_types"] = test_data["event_types"].squeeze(-1).int()
        test_data["seq_idx"] = torch.arange(NUM_OF_PATHS_FOR_TEST).int()

    # Rename keys to match EasyTPP format
    train_data = {our_to_easytpp.get(k, k): v for k, v in train_data.items()}
    test_data = {our_to_easytpp.get(k, k): v for k, v in test_data.items()}

    logger.info(f"Exporting process {process_idx}")

    try:
        dataset = load_dataset(
            "FIM4Science/easytpp-synthetic-1d",
            download_mode="force_redownload",
        )
    except Exception:
        dataset = DatasetDict()

    if NUM_OF_PATHS_FOR_TEST != 0:
        test_dataset = Dataset.from_dict(test_data, split=SOURCE_SPLIT)
        dataset[SOURCE_SPLIT] = test_dataset
    if NUM_PATHS_FOR_INFERENCE != 0:
        train_dataset = Dataset.from_dict(train_data, split="train")
        dataset["train"] = train_dataset

    DATASET_NAME = "FIM4Science/easytpp-synthetic-1d"
    # CONFIG_NAME = f"{SOURCE_SPLIT}_process_{process_idx}"

    # dataset_dict.push_to_hub(DATASET_NAME, private=False, config_name=CONFIG_NAME)
    dataset.push_to_hub(DATASET_NAME, private=False)
