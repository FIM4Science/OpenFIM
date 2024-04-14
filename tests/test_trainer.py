# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long


import pytest

# from fim.data.dataloaders import DataLoaderFactory
# from fim.models import ModelFactory
from fim.trainers.trainer import TrainLossTracker

from fim.utils.logging import setup_logging

# from fim.data.dataloaders import DataLoaderFactory


setup_logging()


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


# def test_trainer_llm():
#     TRAIN_CONF = test_data_path / "config" / "gpt2_commonsense_qa.yaml"
#     config = load_yaml(TRAIN_CONF, True)
#     device_map = config.experiment.device_map
#     tokenizer = load_tokenizer(**config.tokenizer.__dict__)
#     dataloader = DataLoaderFactory.create(**config.dataset.__dict__, tokenizer=tokenizer)
#     num_added_tokens = len(tokenizer.added_tokens_decoder)
#     model = ModelFactory.create(
#         **config.model.to_dict(), pad_token_id=tokenizer.pad_token_id, num_added_tokens=num_added_tokens, device_map=device_map
#     )

#     trainer = Trainer(model, dataloader, config)

#     # trainer.train()
#     assert trainer is not None
