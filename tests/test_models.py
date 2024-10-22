# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import pytest
import torch

from fim import test_data_path
from fim.data.dataloaders import DataLoaderFactory
from fim.models import FIMMJP
from fim.models.blocks import AModel, ModelFactory, TransformerEncoder
from fim.utils.helper import GenericConfig, create_schedulers, load_yaml


class TestModelFactory:
    @pytest.fixture
    def device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    @pytest.fixture
    def train_config(self):
        conf_path = test_data_path / "config" / "fim_ode_mini_test.yaml"
        train_config = load_yaml(conf_path, True)
        return train_config

    def test_init_fim(self, train_config: GenericConfig, device):
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
            device_map=device,
            resume=False,
            peft=None,
        )
        assert model is not None
        assert model.device == torch.device(f"{str(device)}:0")
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
    @pytest.fixture
    def device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    @pytest.fixture
    def config(self):
        conf_path = test_data_path / "config" / "mjp_homogeneous_mini.yaml"
        train_config = load_yaml(conf_path, True)
        return train_config

    @pytest.fixture
    def dataloader(self, config):
        dataloader = DataLoaderFactory.create(**config.dataset.to_dict())
        return dataloader

    @pytest.fixture
    def model(self, config, device):
        model = ModelFactory.create(**config.model.to_dict())
        return model.to(device)

    def test_init(self):
        n_states = 6
        use_adjacency_matrix = False
        transformer_block = {
            "name": "fim.models.blocks.TransformerBlock",
            "model_dim": 64,
            "ff_dim": 256,
            "dropout": 0.1,
            "attention_head": {"name": "torch.nn.MultiheadAttention", "embed_dim": 64, "num_heads": 8, "batch_first": True},
            "activation": {"name": "torch.nn.ReLU"},
            "normalization": {"name": "torch.nn.LayerNorm", "normalized_shape": 64},
        }
        pos_encodings = {"name": "fim.models.blocks.SineTimeEncoding", "model_dim": 64}
        timeseries_encoder = TransformerEncoder(4, transformer_block)
        path_attn = ({"name": "torch.nn.MultiheadAttention", "embed_dim": 64, "num_heads": 8, "batch_first": True},)
        intensity_matrix_decoder = {"name": "fim.models.blocks.MLP", "hidden_layers": [64, 64], "dropout": 0.1}
        initial_distribution_decoder = {"name": "fim.models.blocks.MLP", "hidden_layers": [64, 64], "dropout": 0.1}
        assert (
            FIMMJP(
                n_states,
                use_adjacency_matrix,
                timeseries_encoder,
                pos_encodings,
                path_attn,
                intensity_matrix_decoder,
                initial_distribution_decoder,
            )
            is not None
        )

    def test_forward(self, dataloader, model, device):
        batch = next(iter(dataloader.train_it))
        print(model)
        batch = {key: val.to(device) for key, val in batch.items()}
        out = model(batch)

        assert isinstance(out, dict)
        assert "losses" in out
        assert "im" in out
        assert "log_var_im" in out
        assert "loss" in out["losses"]
        assert "loss_gauss" in out["losses"]
        assert "loss_initial" in out["losses"]
        assert "loss_missing_link" in out["losses"]
        assert out["im"].shape == (batch["observation_grid"].shape[0], 6, 6)
        assert out["log_var_im"].shape == (batch["observation_grid"].shape[0], 6, 6)
        assert out["init_cond"].shape == (batch["observation_grid"].shape[0], 6)
