import pytest
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from fim.models.blocks.base import MLP


class TestMlp:
    input_size = 100
    output_size = 8
    hidden_sizes = [64, 32]

    @pytest.fixture
    def mlp(self):
        hidden_activation = torch.nn.ReLU()
        output_activation = torch.nn.Sigmoid()
        mlp = MLP(self.input_size, self.output_size, self.hidden_sizes, hidden_activation, output_activation)
        return mlp

    def test_mlp_architecture(self, mlp: MLP):
        assert mlp.layers[0].in_features == self.input_size
        assert mlp.layers[-2].out_features == self.output_size
        for i in range(len(self.hidden_sizes)):
            assert mlp.layers.get_submodule(f"linear_{i}").out_features == self.hidden_sizes[i]
            assert mlp.layers.get_submodule(f"linear_{i}").in_features == self.input_size if i == 0 else self.hidden_sizes[i - 1]

    def test_forward(self, mlp):
        x = torch.randn(10, 100)  # input tensor of shape (batch_size, input_size)
        output = mlp.forward(x)
        print(mlp)
        assert output.shape == (10, 8)


class TestTransformer:
    @pytest.fixture
    def transformer_layer(self):
        return TransformerEncoderLayer(d_model=64, nhead=8, dim_feedforward=256, dropout=0.1)

    def test_transformer_block_forward(self, transformer_layer):
        batch_size = 2
        seq_len = 10
        dim_model = 64

        x = torch.randn(batch_size, seq_len, dim_model)

        output = transformer_layer(x)
        assert output.shape == (batch_size, seq_len, dim_model)

    def test_transformer_block_forward_with_mask(self, transformer_layer):
        batch_size = 2
        seq_len = 10
        dim_model = 64

        x = torch.randn(batch_size, seq_len, dim_model)

        output = transformer_layer(x)
        assert output.shape == (batch_size, seq_len, dim_model)

    @pytest.fixture
    def transformer_encoder(self, transformer_layer):
        return TransformerEncoder(encoder_layer=transformer_layer, num_layers=2)

    def test_transformer_encoder_initialization(self, transformer_encoder):
        assert transformer_encoder is not None
        assert len(transformer_encoder.layers) == 2
        for layer in transformer_encoder.layers:
            assert isinstance(layer, TransformerEncoderLayer)

    def test_transformer_encoder_forward(self, transformer_encoder):
        batch_size = 2
        seq_len = 10
        dim_model = 64

        x = torch.randn(batch_size, seq_len, dim_model)
        output = transformer_encoder(x)
        assert output.shape == (batch_size, seq_len, dim_model)
