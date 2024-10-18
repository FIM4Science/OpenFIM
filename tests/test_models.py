# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import pytest
import torch

from fim import test_data_path
from fim.data.dataloaders import DataLoaderFactory

# from reasoningschema.data import load_tokenizer
# from reasoningschema.data.dataloaders import DataLoaderFactory
# from reasoningschema.models import HSN, AModel, ModelFactory
# from reasoningschema.models.blocks.decoders import Decoder
# from reasoningschema.models.blocks.encoders import EncoderModelA
from fim.models.blocks import AModel, ModelFactory
from fim.utils.helper import GenericConfig, create_schedulers, load_yaml
from fim.models import FIMMjp

class TestModelFactory:
    @pytest.fixture
    def train_config(self):
        conf_path = test_data_path / "config" / "fim_ode_mini_test.yaml"
        train_config = load_yaml(conf_path, True)
        return train_config

    def test_init_fim(self, train_config: GenericConfig):
        model = ModelFactory.create(
            name="FIMODE",
            time_encoding=train_config.model.time_encoding.to_dict(),
            trunk_net=train_config.model.trunk_net.to_dict(),
            branch_net=train_config.model.branch_net.to_dict(),
            combiner_net=train_config.model.combiner_net.to_dict(),
            vector_field_net=train_config.model.vector_field_net.to_dict(),
            init_cond_net=train_config.model.init_cond_net.to_dict(),
            loss_configs=train_config.model.loss_configs.to_dict(),
            normalization_time=train_config.model.normalization_time.to_dict(),
            normalization_values=train_config.model.normalization_values.to_dict(),
            load_in_8bit=False,
            load_in_4bit=False,
            use_bf16=False,
            device_map="cpu",
            resume=False,
            peft=None,
        )
        assert model is not None
        assert model.device == torch.device("cpu")
        del model

        if torch.cuda.is_available():
            model = ModelFactory.create(
                name="FIMODE",
                time_encoding=train_config.model.time_encoding.to_dict(),
                trunk_net=train_config.model.trunk_net.to_dict(),
                branch_net=train_config.model.branch_net.to_dict(),
                combiner_net=train_config.model.combiner_net.to_dict(),
                vector_field_net=train_config.model.vector_field_net.to_dict(),
                init_cond_net=train_config.model.init_cond_net.to_dict(),
                loss_configs=train_config.model.loss_configs.to_dict(),
                normalization_time=train_config.model.normalization_time.to_dict(),
                normalization_values=train_config.model.normalization_values.to_dict(),
                load_in_8bit=False,
                load_in_4bit=False,
                use_bf16=False,
                device_map="cuda",
                resume=False,
                peft=None,
            )
            assert model is not None
            assert model.device == torch.device("cuda:0")
            del model

    def test_model_factory_fim(self, train_config: GenericConfig):
        model = ModelFactory.create(**train_config.model.to_dict())
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
        bs = batch["coarse_grid_grid"].shape[0]
        assert out["visualizations"]["solution"]["learnt"].shape == (bs, 128, 1)
        assert out["visualizations"]["solution"]["target"].shape == (bs, 128, 1)


class TestMJP:
    def config(self):
        conf_path = test_data_path / "config" / "fim_mjp_mini_test.yaml"
        train_config = load_yaml(conf_path, True)
        return train_config
    def test_init(self):
        assert True
