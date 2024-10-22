import pytest
import torch

from fim.models.blocks.base import MLP, TransformerBlock, TransformerEncoder


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
        for i in range(len(self.hidden_sizes) - 1):
            assert mlp.layers[i + 2].in_features == self.hidden_sizes[i]
            assert mlp.layers[i + 2].out_features == self.hidden_sizes[i + 1]

    def test_forward(self, mlp):
        x = torch.randn(10, 100)  # input tensor of shape (batch_size, input_size)
        output = mlp.forward(x)
        print(mlp)
        assert output.shape == (10, 8)


class TestTransformer:
    @pytest.fixture
    def transformer_block(self):
        dim = 64
        num_heads = 8

        attention_head = torch.nn.MultiheadAttention(dim, num_heads, batch_first=True)
        return TransformerBlock(
            model_dim=64,
            ff_dim=256,
            dropout=0.1,
            attention_head=attention_head,
            activation=torch.nn.ReLU(),
            normalization=torch.nn.LayerNorm,
        )

    def test_transformer_block_initialization(self, transformer_block):
        assert transformer_block is not None
        assert isinstance(transformer_block.attention_head, torch.nn.MultiheadAttention)
        assert isinstance(transformer_block.norm1, torch.nn.LayerNorm)
        assert isinstance(transformer_block.norm2, torch.nn.LayerNorm)
        assert isinstance(transformer_block.ff, MLP)

    def test_transformer_block_initialization_with_dict(self):
        dim = 64
        num_heads = 8

        attention_head = {"name": "torch.nn.MultiheadAttention", "embed_dim": dim, "num_heads": num_heads, "batch_first": True}
        transformer_block = TransformerBlock(
            model_dim=64,
            ff_dim=256,
            dropout=0.1,
            attention_head=attention_head,
            activation={"name": "torch.nn.ReLU"},
            normalization={"name": "torch.nn.LayerNorm", "normalized_shape": dim},
        )

        assert transformer_block is not None
        assert isinstance(transformer_block.attention_head, torch.nn.MultiheadAttention)
        assert isinstance(transformer_block.norm1, torch.nn.LayerNorm)
        assert isinstance(transformer_block.norm2, torch.nn.LayerNorm)
        assert isinstance(transformer_block.ff, MLP)

    def test_transformer_block_forward(self, transformer_block):
        batch_size = 2
        seq_len = 10
        dim_model = 64

        x = torch.randn(batch_size, seq_len, dim_model)

        output = transformer_block(x)
        assert output.shape == (batch_size, seq_len, dim_model)

    def test_transformer_block_forward_with_mask(self, transformer_block):
        batch_size = 2
        seq_len = 10
        dim_model = 64

        x = torch.randn(batch_size, seq_len, dim_model)
        mask = torch.tril(torch.ones(seq_len, seq_len), 1).bool()

        output = transformer_block(x, mask=mask)
        assert output.shape == (batch_size, seq_len, dim_model)

    @pytest.fixture
    def transformer_encoder(self):
        transformer_block = {
            "name": "fim.models.blocks.TransformerBlock",
            "model_dim": 64,
            "ff_dim": 256,
            "dropout": 0.1,
            "attention_head": {"name": "torch.nn.MultiheadAttention", "embed_dim": 64, "num_heads": 8, "batch_first": True},
            "activation": {"name": "torch.nn.ReLU"},
            "normalization": {"name": "torch.nn.LayerNorm", "normalized_shape": 64},
        }
        return TransformerEncoder(num_layers=2, transformer_block=transformer_block)

    def test_transformer_encoder_initialization(self, transformer_encoder):
        assert transformer_encoder is not None
        assert len(transformer_encoder.layers) == 2
        for layer in transformer_encoder.layers:
            assert isinstance(layer, TransformerBlock)

    def test_transformer_encoder_forward(self, transformer_encoder):
        batch_size = 2
        seq_len = 10
        dim_model = 64

        x = torch.randn(batch_size, seq_len, dim_model)
        output = transformer_encoder(x)
        assert output.shape == (batch_size, seq_len, dim_model)
