from pathlib import Path

import pytest

from fim import test_data_path
from fim.data.dataloaders import DataLoaderFactory
from fim.models import FIMMJP, FIMSDE, FIMImpPointBase, FIMImpPointBaseConfig, FIMMJPConfig, FIMSDEConfig
from fim.models.blocks import ModelFactory
from fim.trainers.trainer import Trainer, TrainLossTracker
from fim.utils.helper import load_yaml


class TestLossTracker:
    @pytest.fixture
    def loss_tracker(self):
        return TrainLossTracker()

    def test_add_batch_loss(self, loss_tracker):
        loss_tracker.add_batch_loss("loss1", 0.5)
        loss_tracker.add_batch_loss("loss1", 0.5)
        assert loss_tracker.get_batch_losses("loss1") == 1.0

    def test_add_batch_losses_dict(self, loss_tracker: TrainLossTracker):
        losses_dict = {"loss1": 0.5, "loss2": 0.7}
        loss_tracker.add_batch_losses(losses_dict)
        assert loss_tracker.get_batch_losses("loss1") == 0.5
        assert loss_tracker.get_batch_losses("loss2") == 0.7

    def test_add_epoch_loss(self, loss_tracker: TrainLossTracker):
        loss_tracker.add_batch_loss("loss1", 0.5)
        loss_tracker.add_batch_loss("loss1", 0.5)
        loss_tracker.add_batch_loss("loss2", 0.7)
        loss_tracker.summarize_epoch()
        assert loss_tracker.get_average_epoch_loss("loss1") == 0.5
        assert loss_tracker.get_average_epoch_loss("loss2") == 0.7

    def test_get_last_epoch_loss(self, loss_tracker: TrainLossTracker):
        loss_tracker.add_batch_loss("loss1", 0.5)
        loss_tracker.add_batch_loss("loss1", 0.5)
        loss_tracker.add_batch_loss("loss2", 0.7)
        loss_tracker.summarize_epoch()
        assert loss_tracker.get_average_epoch_loss("loss1") == 0.5
        assert loss_tracker.get_average_epoch_loss("loss2") == 0.7
        epoch_losses = loss_tracker.get_last_epoch_stats()["losses"]
        assert epoch_losses["loss1"] == 0.5
        assert epoch_losses["loss2"] == 0.7
        loss_tracker.add_batch_loss("loss1", 0.5)
        loss_tracker.add_batch_loss("loss1", 0.5)
        loss_tracker.add_batch_loss("loss2", 0.7)
        loss_tracker.add_batch_loss("loss2", 1.7)
        loss_tracker.summarize_epoch()
        epoch_losses = loss_tracker.get_last_epoch_stats()["losses"]
        assert epoch_losses["loss1"] == 0.5
        assert epoch_losses["loss2"] == 1.2


@pytest.mark.parametrize(
    "config_subpath, config_cls, model_cls, checkpoint_subpath",
    [
        (
            Path("imputation") / "fim_imp_pointwise_base_mini_test.yaml",
            FIMImpPointBaseConfig,
            FIMImpPointBase,
            "DEBUG_fim_ode_noisy_MinMax-experiment-seed-10_08-23-1331/checkpoints/best-model",
        ),
        (
            Path("mjp") / "mjp_homogeneous_mini.yaml",
            FIMMJPConfig,
            FIMMJP,
            "FIM_MJP_Homogeneous_Mini/checkpoints/best-model",
        ),
        (
            Path("sde") / "sde_mini.yaml",
            FIMSDEConfig,
            FIMSDE,
            "sde/checkpoints/best-model",
        ),
    ],
    ids=["imputation", "mjp", "sde"],
)
def test_trainer_per_model(tmp_path, config_subpath, config_cls, model_cls, checkpoint_subpath):

    train_conf = test_data_path / "config" / config_subpath
    config = load_yaml(train_conf, True)
    config.trainer.experiment_dir = tmp_path

    dataloader = DataLoaderFactory.create(**config.dataset.to_dict())
    model_config = config_cls(**config.model.to_dict())
    model = ModelFactory.create(model_config)

    trainer = Trainer(model, dataloader, config)
    assert trainer is not None
    trainer.train()

    checkpoint_path = tmp_path / checkpoint_subpath
    loaded_model = model_cls.from_pretrained(checkpoint_path)

    assert loaded_model is not None
    assert isinstance(loaded_model, model_cls)
