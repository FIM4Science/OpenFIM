import pytest
import torch
from fim.models.blocks import Mlp


class TestMlp:
    input_size = 100
    output_size = 8
    hidden_sizes = [64, 32]
    @pytest.fixture
    def mlp(self):
        hidden_activation = torch.nn.ReLU()
        output_activation = torch.nn.Sigmoid()
        mlp = Mlp(self.input_size, self.output_size, self.hidden_sizes, hidden_activation, output_activation)
        return mlp

    

    def test_mlp_architecture(self, mlp: Mlp):

        assert mlp.layers[0].in_features == self.input_size
        assert mlp.layers[-2].out_features == self.output_size
        for i in range(len(self.hidden_sizes)-1):
            assert mlp.layers[i+2].in_features == self.hidden_sizes[i]
            assert mlp.layers[i+2].out_features == self.hidden_sizes[i+1]



    def test_forward(self, mlp):
        x = torch.randn(10, 100)  # input tensor of shape (batch_size, input_size)
        output = mlp.forward(x)
        print(mlp)
        assert output.shape == (10, 8)
