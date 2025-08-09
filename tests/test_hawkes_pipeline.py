import runpy
from pathlib import Path

import pytest

from fim.data.dataloaders import DataLoaderFactory
from fim.utils.helper import expand_params, load_yaml


TEST_3D_DIR = Path(__file__).resolve().parent / "resources" / "data" / "hawkes" / "3D_hawkes_test_data"
TEST_10D_DIR = Path(__file__).resolve().parent / "resources" / "data" / "hawkes" / "10D_hawkes_test_data"


def _make_minimal_hawkes_dl_config():
    return {
        "name": "HawkesDataLoader",
        "path": {
            "train": [TEST_3D_DIR, TEST_10D_DIR],
            "validation": [TEST_3D_DIR, TEST_10D_DIR],
        },
        "loader_kwargs": {
            "batch_size": 2,
            "test_batch_size": 1,
            "num_workers": 0,
            "num_inference_paths": 1,
            "num_inference_times": 8,
            "variable_num_of_paths": True,
            "min_path_count": 10,
            "max_path_count": 10,
            "max_number_of_minibatch_sizes": 1,
            "variable_sequence_lens": {"train": True, "validation": False},
            "min_sequence_len": 4,
            "max_sequence_len": 8,
        },
        "dataset_kwargs": {
            "files_to_load": {
                "base_intensity_functions": "base_intensity_functions.pt",
                "event_times": "event_times.pt",
                "event_types": "event_types.pt",
                "kernel_functions": "kernel_functions.pt",
            },
            "field_name_for_dimension_grouping": [
                "base_intensity_functions",
                "kernel_functions",
            ],
            "data_limit": 8,
        },
    }


@pytest.mark.short
def test_hawkes_dataloader_smoke():
    cfg = _make_minimal_hawkes_dl_config()
    dl = DataLoaderFactory.create(**cfg)

    batch = next(iter(dl.train_it))

    # If dataset groups by variable number of marks, pick one group
    if isinstance(batch, dict) and len(batch) > 0 and isinstance(next(iter(batch.keys())), int):
        batch = next(iter(batch.values()))

    # Collate splits to inference/context and synthesizes evaluation times
    assert "inference_event_times" in batch
    assert "context_event_times" in batch
    assert "intensity_evaluation_times" in batch
    # Hawkes tensors should be present
    assert "kernel_functions" in batch
    assert "base_intensity_functions" in batch
    # Sanity: shapes are batchable
    assert batch["inference_event_times"].ndim in (3, 4)
    assert batch["intensity_evaluation_times"].ndim == 3  # [B, P_infer, T_eval]


@pytest.mark.short
def test_hawkes_training_pipeline(tmp_path):
    # Load training script utilities dynamically
    mod = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "train_model.py"))
    train_single = mod["train_single"]

    # Load tiny test config from tests resources
    cfg_path = Path(__file__).resolve().parents[0] / "resources" / "config" / "hawkes" / "mini_test.yaml"
    base_cfg = load_yaml(cfg_path)
    cfg = expand_params(base_cfg)[0]

    # Make sure runs fast and writes to tmp
    cfg.trainer.experiment_dir = str(tmp_path)
    cfg.experiment.device_map = "cpu"
    cfg.experiment.name_add_date = False

    # Run one short epoch end-to-end
    train_single(cfg, resume=None)

    # Verify artifacts written
    exp_dir = Path(cfg.trainer.experiment_dir) / cfg.experiment.name
    assert (exp_dir / "train_parameters.yaml").exists()
