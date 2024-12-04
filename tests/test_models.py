# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long


import pytest
import torch

from fim import test_data_path
from fim.data.dataloaders import DataLoaderFactory
from fim.data.datasets import FIMSDEDatabatchTuple
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
        transformer_block = {
            "name": "fim.models.blocks.TransformerBlock",
            "in_features": 64,
            "ff_dim": 256,
            "dropout": 0.1,
            "attention_head": {"name": "torch.nn.MultiheadAttention", "embed_dim": 64, "num_heads": 8, "batch_first": True},
            "activation": {"name": "torch.nn.ReLU"},
            "normalization": {"name": "torch.nn.LayerNorm", "normalized_shape": 64},
        }
        pos_encodings = {"name": "fim.models.blocks.SineTimeEncoding", "out_features": 64}
        timeseries_encoder = {
            "name": "fim.models.blocks.base.TransformerEncoder",
            "num_layers": 4,
            "embed_dim": 64,
            "transformer_block": transformer_block,
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
        print(model)
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
        rnn = torch.nn.RNN(2 + n_states, 64, 1, batch_first=True, bidirectional=True)
        rnn = {"name": "torch.nn.RNN", "hidden_size": 64, "num_layers": 1, "bidirectional": True, "batch_first": True}
        timeseries_encoder = {"name": "fim.models.blocks.RNNEncoder", "rnn": rnn}
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
        print(model_rnn)
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


@pytest.mark.skip(reason=r"Config yaml includes paths in Windows format, using `\` instead of `/`, so the test fails.")
class TestFIMHawkes:
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
        conf_path = test_data_path / "config" / "hawkes" / "mini.yaml"
        train_config = load_yaml(conf_path, True)
        return train_config

    # @pytest.fixture
    # def config_rnn(self):
    #     conf_path = Path("test_data/config/fimhawkes_rnn.yaml")
    #     train_config = load_yaml(conf_path, True)
    #     return train_config

    @pytest.fixture
    def dataloader(self, config):
        dataloader = DataLoaderFactory.create(**config.dataset.to_dict())
        return dataloader

    @pytest.fixture
    def model(self, config, device):
        model_config = FIMHawkesConfig(**config.model.to_dict())
        model = FIMHawkes(model_config)
        return model.to(device)

    # @pytest.fixture
    # def model_rnn(self, config_rnn, device):
    #     model_config = FIMHawkesConfig(**config_rnn.model.to_dict())
    #     model = FIMHawkes(model_config)
    #     return model.to(device)

    def test_init(self):
        num_marks = 1
        config = FIMHawkesConfig(
            num_marks=num_marks,
            time_encodings={"name": "fim.models.blocks.positional_encodings.DeltaTimeEncoding"},
            event_type_embedding={"name": "fim.models.blocks.IdentityBlock", "out_features": num_marks},
            ts_encoder={
                "name": "fim.models.blocks.base.RNNEncoder",
                "rnn": {"name": "torch.nn.LSTM", "hidden_size": 64, "batch_first": True, "bidirectional": True},
            },
            trunk_net={
                "name": "fim.models.blocks.base.MLP",
                "out_features": 8,
                "hidden_layers": [8, 8],
                "hidden_act": {"name": "torch.nn.SELU"},
                "dropout": 0,
                "initialization_scheme": "lecun_normal",
            },
            Omega_1_encoder={"name": "torch.nn.MultiheadAttention", "embed_dim": 8, "num_heads": 1, "batch_first": True},
            Omega_2_encoder={
                "name": "fim.models.blocks.MultiHeadLearnableQueryAttention",
                "n_queries": 16,
                "n_heads": 1,
                "embed_dim": 8,
                "kv_dim": 8,
            },
            Omega_3_encoder={
                "name": "fim.models.blocks.MultiHeadLearnableQueryAttention",
                "n_queries": 16,
                "n_heads": 1,
                "embed_dim": 8,
                "kv_dim": 8,
            },
            Omega_4_encoder={
                "name": "fim.models.blocks.MultiHeadLearnableQueryAttention",
                "n_queries": 16,
                "n_heads": 1,
                "embed_dim": 2048,
                "kv_dim": 8,
            },
            kernel_value_decoder={
                "name": "fim.models.blocks.base.MLP",
                "in_features": 2048,
                "hidden_layers": [8, 8],
                "hidden_act": {"name": "torch.nn.SELU"},
                "dropout": 0,
                "initialization_scheme": "lecun_normal",
            },
            kernel_parameter_decoder={
                "name": "fim.models.blocks.base.MLP",
                "in_features": 2048,
                "hidden_layers": [8, 8],
                "hidden_act": {"name": "torch.nn.SELU"},
                "dropout": 0,
                "initialization_scheme": "lecun_normal",
            },
            loss="TBD",
        )
        model = FIMHawkes(config)
        assert model.num_marks == num_marks
        assert model.ts_encoder is not None
        assert model.kernel_value_decoder is not None
        assert isinstance(model.Omega_1_encoder, torch.nn.MultiheadAttention)
        assert isinstance(model.Omega_2_encoder, torch.nn.Module)
        assert isinstance(model.Omega_3_encoder, torch.nn.Module)
        assert isinstance(model.Omega_4_encoder, torch.nn.Module)

    def test_forward(self, dataloader, model, device):
        batch = next(iter(dataloader.train_it))
        batch = {key: val.to(device) for key, val in batch.items()}
        out = model(batch)

        assert isinstance(out, dict)
        assert "predicted_kernel_values" in out
        assert "predicted_base_intensity" in out
        assert "predicted_kernel_decay" in out

        if "base_intensities" in batch and "kernel_evaluations" in batch:
            assert "losses" in out
            assert "loss" in out["losses"]
            assert "kernel_rmse" in out["losses"]
            assert "base_intensity_rmse" in out["losses"]

        batch_size = batch["kernel_grids"].shape[0]
        num_marks = batch["kernel_grids"].shape[1]
        num_kernel_eval_points = batch["kernel_grids"].shape[2]
        assert out["predicted_kernel_values"].shape == (batch_size, num_marks, num_kernel_eval_points, num_marks)
        assert out["predicted_base_intensity"].shape == (batch_size, model.num_marks)

    def test_init_rnn(self):
        num_marks = 1
        config_rnn = FIMHawkesConfig(
            num_marks=num_marks,
            time_encodings={"name": "fim.models.blocks.positional_encodings.DeltaTimeEncoding"},
            event_type_embedding={"name": "fim.models.blocks.IdentityBlock", "out_features": num_marks},
            ts_encoder={
                "name": "fim.models.blocks.base.RNNEncoder",
                "rnn": {"name": "torch.nn.LSTM", "hidden_size": 64, "batch_first": True, "bidirectional": True},
            },
            trunk_net={
                "name": "fim.models.blocks.base.MLP",
                "out_features": 8,
                "hidden_layers": [8, 8],
                "hidden_act": {"name": "torch.nn.SELU"},
                "dropout": 0,
                "initialization_scheme": "lecun_normal",
            },
            Omega_1_encoder={"name": "torch.nn.MultiheadAttention", "embed_dim": 8, "num_heads": 1, "batch_first": True},
            Omega_2_encoder={
                "name": "fim.models.blocks.MultiHeadLearnableQueryAttention",
                "n_queries": 16,
                "n_heads": 1,
                "embed_dim": 8,
                "kv_dim": 8,
            },
            Omega_3_encoder={
                "name": "fim.models.blocks.MultiHeadLearnableQueryAttention",
                "n_queries": 16,
                "n_heads": 1,
                "embed_dim": 8,
                "kv_dim": 8,
            },
            Omega_4_encoder={
                "name": "fim.models.blocks.MultiHeadLearnableQueryAttention",
                "n_queries": 16,
                "n_heads": 1,
                "embed_dim": 2048,
                "kv_dim": 8,
            },
            kernel_value_decoder={
                "name": "fim.models.blocks.base.MLP",
                "in_features": 2048,
                "hidden_layers": [8, 8],
                "hidden_act": {"name": "torch.nn.SELU"},
                "dropout": 0,
                "initialization_scheme": "lecun_normal",
            },
            kernel_parameter_decoder={
                "name": "fim.models.blocks.base.MLP",
                "in_features": 2048,
                "hidden_layers": [8, 8],
                "hidden_act": {"name": "torch.nn.SELU"},
                "dropout": 0,
                "initialization_scheme": "lecun_normal",
            },
            loss="TBD",
        )
        model = FIMHawkes(config_rnn)
        assert model.ts_encoder is not None
        assert isinstance(model.ts_encoder.rnn, torch.nn.LSTM)
        assert model.ts_encoder.rnn.hidden_size == 64
        assert model.ts_encoder.rnn.bidirectional is True

    # def test_forward_rnn(self, dataloader, model_rnn, device):
    #     batch = next(iter(dataloader.train_it))
    #     batch = {key: val.to(device) for key, val in batch.items()}
    #     out = model_rnn(batch)

    #     assert isinstance(out, dict)
    #     assert "predicted_kernel_values" in out
    #     assert "predicted_base_intensity" in out
    #     assert "predicted_kernel_decay" in out

    #     if "base_intensities" in batch and "kernel_evaluations" in batch:
    #         assert "losses" in out
    #         assert "loss" in out["losses"]
    #         assert "kernel_rmse" in out["losses"]
    #         assert "base_intensity_rmse" in out["losses"]

    #     batch_size = batch["kernel_grids"].shape[0]
    #     num_marks = batch["kernel_grids"].shape[1]
    #     num_kernel_eval_points = batch["kernel_grids"].shape[2]
    #     assert out["predicted_kernel_values"].shape == (batch_size, num_marks, num_kernel_eval_points)
    #     assert out["predicted_base_intensity"].shape == (batch_size, num_marks)

    def test_summary(self, model, dataloader):
        batch = next(iter(dataloader.train_it))
        x = {key: val.unsqueeze(0).to(model.device) for key, val in batch.items()}
        print(x)

    def test_save_load_model(self, model, tmp_path):
        model.save_pretrained(tmp_path)

        loaded_model = FIMHawkes.from_pretrained(tmp_path)
        assert model.config.num_marks == loaded_model.config.num_marks
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
        conf_path = conf_path = test_data_path / "config" / "hawkes" / "mini_5_st.yaml"
        train_config = load_yaml(conf_path, True)
        return train_config

    @pytest.fixture
    def dataloader(self, config):
        dataloader = DataLoaderFactory.create(**config.dataset.to_dict())
        return dataloader

    @pytest.fixture
    def model(self, config, device):
        model_config = FIMHawkesConfig(**config.model.to_dict())
        model = FIMHawkes(model_config)
        return model.to(device)

    def test_init(self):
        num_marks = 5
        config = FIMHawkesConfig(
            num_marks=num_marks,
            time_encodings={"name": "fim.models.blocks.positional_encodings.DeltaTimeEncoding"},
            event_type_embedding={"name": "fim.models.blocks.IdentityBlock", "out_features": num_marks},
            ts_encoder={
                "name": "fim.models.blocks.base.RNNEncoder",
                "rnn": {"name": "torch.nn.LSTM", "hidden_size": 64, "batch_first": True, "bidirectional": True},
            },
            trunk_net={
                "name": "fim.models.blocks.base.MLP",
                "out_features": 8,
                "hidden_layers": [8, 8],
                "hidden_act": {"name": "torch.nn.SELU"},
                "dropout": 0,
                "initialization_scheme": "lecun_normal",
            },
            Omega_1_encoder={"name": "torch.nn.MultiheadAttention", "embed_dim": 8, "num_heads": 1, "batch_first": True},
            Omega_2_encoder={
                "name": "fim.models.blocks.MultiHeadLearnableQueryAttention",
                "n_queries": 2,
                "n_heads": 1,
                "embed_dim": 8,
                "kv_dim": 8,
            },
            Omega_3_encoder={
                "name": "fim.models.blocks.MultiHeadLearnableQueryAttention",
                "n_queries": 2,
                "n_heads": 1,
                "embed_dim": 8,
                "kv_dim": 8,
            },
            Omega_4_encoder={
                "name": "fim.models.blocks.MultiHeadLearnableQueryAttention",
                "n_queries": 2,
                "n_heads": 1,
                "embed_dim": 32,
                "kv_dim": 8,
            },
            kernel_value_decoder={
                "name": "fim.models.blocks.base.MLP",
                "in_features": 32,
                "hidden_layers": [8, 8],
                "hidden_act": {"name": "torch.nn.SELU"},
                "dropout": 0,
                "initialization_scheme": "lecun_normal",
            },
            kernel_parameter_decoder={
                "name": "fim.models.blocks.base.MLP",
                "in_features": 32,
                "hidden_layers": [8, 8],
                "hidden_act": {"name": "torch.nn.SELU"},
                "dropout": 0,
                "initialization_scheme": "lecun_normal",
            },
            loss="TBD",
        )
        model = FIMHawkes(config)
        assert model.num_marks == num_marks
        assert model.ts_encoder is not None
        assert model.kernel_value_decoder is not None
        assert isinstance(model.Omega_1_encoder, torch.nn.MultiheadAttention)
        assert isinstance(model.Omega_2_encoder, torch.nn.Module)
        assert isinstance(model.Omega_3_encoder, torch.nn.Module)
        assert isinstance(model.Omega_4_encoder, torch.nn.Module)

    def test_forward(self, dataloader, model, device):
        batch = next(iter(dataloader.train_it))
        batch = {key: val.to(device) for key, val in batch.items()}
        out = model(batch)

        assert isinstance(out, dict)
        assert "predicted_kernel_values" in out
        assert "predicted_base_intensity" in out
        assert "predicted_kernel_decay" in out

        if "base_intensities" in batch and "kernel_evaluations" in batch:
            assert "losses" in out
            assert "loss" in out["losses"]
            assert "kernel_rmse" in out["losses"]
            assert "base_intensity_rmse" in out["losses"]

        batch_size = batch["kernel_grids"].shape[0]
        num_marks = batch["kernel_grids"].shape[1]
        num_kernel_eval_points = batch["kernel_grids"].shape[2]
        assert out["predicted_kernel_values"].shape == (batch_size, num_marks, num_kernel_eval_points, num_marks)
        assert out["predicted_base_intensity"].shape == (batch_size, model.num_marks)

    def test_init_rnn(self):
        num_marks = 5
        config_rnn = FIMHawkesConfig(
            num_marks=num_marks,
            time_encodings={"name": "fim.models.blocks.positional_encodings.DeltaTimeEncoding"},
            event_type_embedding={"name": "fim.models.blocks.IdentityBlock", "out_features": num_marks},
            ts_encoder={
                "name": "fim.models.blocks.base.RNNEncoder",
                "rnn": {"name": "torch.nn.LSTM", "hidden_size": 64, "batch_first": True, "bidirectional": True},
            },
            trunk_net={
                "name": "fim.models.blocks.base.MLP",
                "out_features": 8,
                "hidden_layers": [8, 8],
                "hidden_act": {"name": "torch.nn.SELU"},
                "dropout": 0,
                "initialization_scheme": "lecun_normal",
            },
            Omega_1_encoder={"name": "torch.nn.MultiheadAttention", "embed_dim": 8, "num_heads": 1, "batch_first": True},
            Omega_2_encoder={
                "name": "fim.models.blocks.MultiHeadLearnableQueryAttention",
                "n_queries": 16,
                "n_heads": 1,
                "embed_dim": 8,
                "kv_dim": 8,
            },
            Omega_3_encoder={
                "name": "fim.models.blocks.MultiHeadLearnableQueryAttention",
                "n_queries": 16,
                "n_heads": 1,
                "embed_dim": 8,
                "kv_dim": 8,
            },
            Omega_4_encoder={
                "name": "fim.models.blocks.MultiHeadLearnableQueryAttention",
                "n_queries": 16,
                "n_heads": 1,
                "embed_dim": 32,
                "kv_dim": 8,
            },
            kernel_value_decoder={
                "name": "fim.models.blocks.base.MLP",
                "in_features": 32,
                "hidden_layers": [8, 8],
                "hidden_act": {"name": "torch.nn.SELU"},
                "dropout": 0,
                "initialization_scheme": "lecun_normal",
            },
            kernel_parameter_decoder={
                "name": "fim.models.blocks.base.MLP",
                "in_features": 32,
                "hidden_layers": [8, 8],
                "hidden_act": {"name": "torch.nn.SELU"},
                "dropout": 0,
                "initialization_scheme": "lecun_normal",
            },
            loss="TBD",
        )
        model = FIMHawkes(config_rnn)
        assert model.ts_encoder is not None
        assert isinstance(model.ts_encoder.rnn, torch.nn.LSTM)
        assert model.ts_encoder.rnn.hidden_size == 64
        assert model.ts_encoder.rnn.bidirectional is True

    def test_summary(self, model, dataloader):
        batch = next(iter(dataloader.train_it))
        x = {key: val.unsqueeze(0).to(model.device) for key, val in batch.items()}
        print(x)

    def test_save_load_model(self, model, tmp_path):
        model.save_pretrained(tmp_path)

        loaded_model = FIMHawkes.from_pretrained(tmp_path)
        assert model.config.num_marks == loaded_model.config.num_marks
        assert model.state_dict().keys() == loaded_model.state_dict().keys()
