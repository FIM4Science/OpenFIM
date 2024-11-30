# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long


import pytest

from fim import test_data_path
from fim.data.dataloaders import DataLoaderFactory
from fim.models import FIMMJP, FIMMJPConfig, FIMODEConfig
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


def test_trainer_fimode(tmp_path):
    TRAIN_CONF = test_data_path / "config" / "fim_ode_mini_test.yaml"
    config = load_yaml(TRAIN_CONF, True)
    config.trainer.experiment_dir = tmp_path

    dataloader = DataLoaderFactory.create(**config.dataset.to_dict())
    model = ModelFactory.create(FIMODEConfig(**config.model.to_dict()))

    trainer = Trainer(model, dataloader, config)

    trainer.train()
    assert trainer is not None


class TestTrainMJP:
    @pytest.fixture(scope="module")
    def results_dir(self, tmp_path_factory):
        return tmp_path_factory.mktemp("results")

    def test_trainer_mjp(self, results_dir):
        TRAIN_CONF = test_data_path / "config" / "mjp_homogeneous_mini.yaml"

        config = load_yaml(TRAIN_CONF, True)
        config.trainer.experiment_dir = results_dir
        dataloader = DataLoaderFactory.create(**config.dataset.to_dict())

        model = ModelFactory.create(FIMMJPConfig(**config.model.to_dict()))

        trainer = Trainer(model, dataloader, config)

        trainer.train()
        assert trainer is not None
        model = FIMMJP.load_model(results_dir / "FIM_MJP_Homogeneous_Mini/checkpoints/best-model")
        assert model is not None
        assert isinstance(model, FIMMJP)
