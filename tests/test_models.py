# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long


from pathlib import Path
import pytest
import torch

from fim import test_data_path
from fim.data.dataloaders import DataLoaderFactory
from fim.models import FIMMJP, FIMHawkes, FIMHawkesConfig, FIMMJPConfig, FIMODEConfig
from fim.models.blocks import AModel, ModelFactory
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
        ode_config = FIMODEConfig(**train_config.model.to_dict())
        model = ModelFactory.create(config=ode_config)
        assert model is not None
        assert model.device == torch.device("cpu")
        del model

    @pytest.fixture
    def model(self, train_config):
        self.device_map = train_config.experiment.device_map
        model = ModelFactory.create(FIMODEConfig(**train_config.model.to_dict()))

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
        conf_path = test_data_path / "config" / "mjp" / "mjp_homogeneous_mini.yaml"
        train_config = load_yaml(conf_path, True)
        return train_config

    @pytest.fixture
    def config_rnn(self):
        conf_path = test_data_path / "config" / "mjp" / "mjp_homogeneous_rnn.yaml"
        train_config = load_yaml(conf_path, True)
        return train_config

    @pytest.fixture
    def dataloader(self, config):
        dataloader = DataLoaderFactory.create(**config.dataset.to_dict())
        return dataloader

    @pytest.fixture
    def model(self, config, device):
        model = ModelFactory.create(FIMMJPConfig(**config.model.to_dict()))
        return model.to(device)

    @pytest.fixture
    def model_rnn(self, config_rnn, device):
        model = ModelFactory.create(FIMMJPConfig(**config_rnn.model.to_dict()))
        return model.to(device)

    def test_init(self):
        n_states = 6
        use_adjacency_matrix = False
        transformer_encoder_layer = {
            "name": "torch.nn.TransformerEncoderLayer",
            "d_model": 64,
            "nhead": 8,
            "dim_feedforward": 256,
            "dropout": 0.1,
        }
        pos_encodings = {"name": "fim.models.blocks.SineTimeEncoding", "out_features": 64}
        timeseries_encoder = {
            "name": "torch.nn.TransformerEncoder",
            "num_layers": 4,
            "encoder_layer": transformer_encoder_layer,
        }
        path_attn = {"name": "torch.nn.MultiheadAttention", "embed_dim": 64, "num_heads": 8, "batch_first": True}
        intensity_matrix_decoder = {"name": "fim.models.blocks.MLP", "hidden_layers": [64, 64], "dropout": 0.1}
        initial_distribution_decoder = {"name": "fim.models.blocks.MLP", "hidden_layers": [64, 64], "dropout": 0.1}
        assert (
            FIMMJP(
                FIMMJPConfig(
                    n_states,
                    use_adjacency_matrix,
                    timeseries_encoder,
                    pos_encodings,
                    path_attn,
                    intensity_matrix_decoder,
                    initial_distribution_decoder,
                )
            )
            is not None
        )

    def test_forward(self, dataloader, model, device):
        batch = next(iter(dataloader.train_it))
        batch = {key: val.to(device) for key, val in batch.items()}
        out = model(batch)

        assert isinstance(out, dict)
        assert "losses" in out
        assert "intensity_matrices" in out
        assert "intensity_matrices_variance" in out
        assert "loss" in out["losses"]
        assert "loss_gauss" in out["losses"]
        assert "loss_initial" in out["losses"]
        assert "loss_missing_link" in out["losses"]
        assert out["intensity_matrices"].shape == (batch["observation_grid"].shape[0], 6, 6)
        assert out["intensity_matrices_variance"].shape == (batch["observation_grid"].shape[0], 6, 6)
        assert out["initial_condition"].shape == (batch["observation_grid"].shape[0], 6)

    def test_init_rnn(self):
        n_states = 6
        use_adjacency_matrix = False
        pos_encodings = {"name": "fim.models.blocks.DeltaTimeEncoding"}
        # rnn = torch.nn.RNN(2 + n_states, 64, 1, batch_first=True, bidirectional=True)
        rnn = {"name": "torch.nn.RNN", "hidden_size": 64, "num_layers": 1, "bidirectional": True, "batch_first": True}
        timeseries_encoder = {"name": "fim.models.blocks.RNNEncoder", "encoder_layer": rnn}
        path_attn = {
            "name": "fim.models.blocks.MultiHeadLearnableQueryAttention",
            "n_queries": 16,
            "n_heads": 1,
            "embed_dim": 64,
            "kv_dim": 128,
        }
        intensity_matrix_decoder = {"name": "fim.models.blocks.MLP", "hidden_layers": [64, 64], "dropout": 0.1}
        initial_distribution_decoder = {"name": "fim.models.blocks.MLP", "hidden_layers": [64, 64], "dropout": 0.1}
        assert (
            FIMMJP(
                FIMMJPConfig(
                    n_states,
                    use_adjacency_matrix,
                    timeseries_encoder,
                    pos_encodings,
                    path_attn,
                    intensity_matrix_decoder,
                    initial_distribution_decoder,
                )
            )
            is not None
        )

    def test_forward_rnn(self, dataloader, model_rnn, device):
        batch = next(iter(dataloader.train_it))
        batch = {key: val.to(device) for key, val in batch.items()}
        out = model_rnn(batch)

        assert isinstance(out, dict)
        assert "losses" in out
        assert "intensity_matrices" in out
        assert "intensity_matrices_variance" in out
        assert "loss" in out["losses"]
        assert "loss_gauss" in out["losses"]
        assert "loss_initial" in out["losses"]
        assert "loss_missing_link" in out["losses"]
        assert out["intensity_matrices"].shape == (batch["observation_grid"].shape[0], 6, 6)
        assert out["intensity_matrices_variance"].shape == (batch["observation_grid"].shape[0], 6, 6)
        assert out["initial_condition"].shape == (batch["observation_grid"].shape[0], 6)

    def test_summary(self, model, dataloader):
        import torchinfo

        x = dataloader.train_it.dataset[0]
        x = {key: val.unsqueeze(0).to(model.device) for key, val in x.items()}
        print(torchinfo.summary(model, input_data=[x]))

    def test_save_load_model(self, model, tmp_path):
        model.save_pretrained(tmp_path)

        loaded_model = FIMMJP.from_pretrained(tmp_path)
        # assert model.config == loaded_model.config
        assert model.state_dict().keys() == loaded_model.state_dict().keys()


class TestFIMHawkes5ST:
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
        conf_path = Path("test_data/config/hawkes/mini_5_st.yaml")
        train_config = load_yaml(conf_path, True)
        return train_config

    @pytest.fixture
    def dataloader(self, config):
        return DataLoaderFactory.create(**config.dataset.to_dict())

    @pytest.fixture
    def model(self, config, device):
        model_config = FIMHawkesConfig(**config.model.to_dict())
        model = FIMHawkes(model_config).to(device)
        return model

    def test_init(self):
        num_marks = 5
        hidden_dim = 32
        config = FIMHawkesConfig(
            num_marks=num_marks,
            hidden_dim=hidden_dim,
            mark_encoder={"name": "torch.nn.Linear", "out_features": hidden_dim},
            time_encoder={"name": "torch.nn.Linear", "out_features": hidden_dim},
            delta_time_encoder={"name": "torch.nn.Linear", "out_features": hidden_dim},
            kernel_time_encoder={"name": "torch.nn.Linear", "out_features": hidden_dim},
            evaluation_mark_encoder={"name": "torch.nn.Linear", "out_features": hidden_dim},
            ts_encoder={
                "name": "torch.nn.TransformerEncoder",
                "num_layers": 4,
                "encoder_layer": {
                    "name": "torch.nn.TransformerEncoderLayer",
                    "d_model": hidden_dim,
                    "nhead": 4,
                    "dropout": 0.1,
                    "batch_first": True,
                },
            },
            time_dependent_functional_attention={
                "paths_block_attention": False,
                "num_res_layers": 8,
                "attention": {"nhead": 1},
                "projection": {"name": "fim.models.blocks.base.MLP", "hidden_layers": (hidden_dim, hidden_dim)},
            },
            static_functional_attention={
                "paths_block_attention": False,
                "num_res_layers": 8,
                "attention": {"nhead": 1},
                "projection": {"name": "fim.models.blocks.base.MLP", "hidden_layers": (hidden_dim, hidden_dim)},
            },
            static_functional_attention_learnable_query={"name": "torch.nn.Linear", "out_features": hidden_dim},
            kernel_value_decoder={
                "name": "fim.models.blocks.base.MLP",
                "hidden_layers": (hidden_dim, hidden_dim),
            },
            base_intensity_decoder={
                "name": "fim.models.blocks.base.MLP",
                "hidden_layers": (hidden_dim, hidden_dim),
            },
        )
        model = FIMHawkes(config)
        assert model.num_marks == num_marks
        assert model.mark_encoder is not None
        assert model.time_encoder is not None
        assert model.kernel_time_encoder is not None
        assert model.evaluation_mark_encoder is not None
        assert model.ts_encoder is not None
        assert model.time_dependent_functional_attention is not None
        assert model.static_functional_attention is not None
        assert model.static_functional_attention_learnable_query is not None
        assert model.kernel_value_decoder is not None
        assert model.base_intensity_decoder is not None

    def test_forward(self, dataloader, model, device):
        batch = next(iter(dataloader.train_it))
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(batch)

        assert isinstance(output, dict)
        assert "predicted_kernel_values" in output
        assert "predicted_base_intensity" in output

        if "base_intensities" in batch and "kernel_evaluations" in batch:
            assert "losses" in output
            assert "loss" in output["losses"]
            assert "kernel_rmse" in output["losses"]
            assert "base_intensity_rmse" in output["losses"]

        batch_size, num_marks, L_kernel = batch["kernel_grids"].shape
        assert output["predicted_kernel_values"].shape == (batch_size, num_marks, L_kernel)
        assert output["predicted_base_intensity"].shape == (batch_size, num_marks)

    def test_save_load_model(self, model, tmp_path):
        model.save_pretrained(tmp_path)
        loaded_model = FIMHawkes.from_pretrained(tmp_path)
        assert model.config.num_marks == loaded_model.config.num_marks
        assert set(model.state_dict().keys()) == set(loaded_model.state_dict().keys())