# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import pytest
import torch

from fim import test_data_path
from fim.data.dataloaders import DataLoaderFactory
from fim.models import ModelFactory

# from reasoningschema.data import load_tokenizer
# from reasoningschema.data.dataloaders import DataLoaderFactory
# from reasoningschema.models import HSN, AModel, ModelFactory
# from reasoningschema.models.blocks.decoders import Decoder
# from reasoningschema.models.blocks.encoders import EncoderModelA
from fim.models.models import AModel
from fim.utils.helper import GenericConfig, create_schedulers, load_yaml


class TestModelFactory:
    @pytest.fixture
    def train_config(self):
        conf_path = test_data_path / "config" / "ar_lstm_vanila.yaml"
        train_config = load_yaml(conf_path, True)
        return train_config

    def test_init_ar(self, train_config: GenericConfig):
        model = ModelFactory.create(
            name="AR", recurrent_module=train_config.model.recurrent_module.to_dict(), output_head=train_config.model.output_head.to_dict()
        )
        assert model is not None
        assert model.device == torch.device("cpu")
        del model
        model = ModelFactory.create(
            name="AR",
            recurrent_module=train_config.model.recurrent_module.to_dict(),
            output_head=train_config.model.output_head.to_dict(),
            device_map="cuda",
        )
        assert model is not None
        assert model.device == torch.device("cuda:0")
        del model

    def test_model_factory_ar(self):
        model_params = {
            "recurrent_module": {"name": "torch.nn.LSTM", "input_size": 2, "hidden_size": 10, "num_layers": 1, "batch_first": True},
            "output_head": {
                "name": "fim.models.blocks.Mlp",
                "in_features": 10,
                "out_features": 2,
                "hidden_layers": (256, 256),
                "hidden_act": {"name": "torch.nn.ReLU"},
                "output_act": {"name": "torch.nn.Sigmoid"},
            },
        }
        model = ModelFactory.create("AR", **model_params)
        assert model is not None

    @pytest.fixture
    def model(self, train_config):
        self.device_map = train_config.experiment.device_map
        model = ModelFactory.create(**train_config.model.to_dict())

        return model.to(self.device_map)

    @pytest.fixture
    def dataset(self, train_config):
        dataloader = DataLoaderFactory.create(**train_config.dataset.to_dict())
        return dataloader.train_it

    @pytest.fixture
    def schedulers(self, train_config, dataset):
        schedulers_config = train_config.trainer.schedulers
        max_steps = train_config.trainer.epochs * len(dataset)
        return create_schedulers(schedulers_config, max_steps, len(dataset))

    def test_forward(self, model: AModel, dataset):
        for batch in dataset:
            for key in batch.keys():
                batch[key] = batch[key].to(self.device_map)
            out = model(batch)
            break
        assert isinstance(out["losses"], dict)
        max_len = batch["seq_len"].max().item() - 1
        bs = batch["target"].shape[0]
        assert out["predictions"].shape == (bs, max_len, 2)
