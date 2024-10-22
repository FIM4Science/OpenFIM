import torch

from fim.models.blocks.positional_encodings import DeltaTimeEncoding


class TestDeltaTimeEncoding:
    def test_forward(self):
        model = DeltaTimeEncoding()
        input_tensor = torch.tensor([[[1.0], [2.0], [4.0]]])
        expected_output = torch.tensor([[[1.0, 0.0], [2.0, 1.0], [4.0, 2.0]]])

        output = model(input_tensor)
        assert torch.allclose(output, expected_output), f"Expected {expected_output}, but got {output}"

    def test_forward_batch(self):
        model = DeltaTimeEncoding()
        input_tensor = torch.tensor([[[1.0], [2.0], [4.0]], [[0.0], [1.0], [3.0]]])
        expected_output = torch.tensor([[[1.0, 0.0], [2.0, 1.0], [4.0, 2.0]], [[0.0, 0.0], [1.0, 1.0], [3.0, 2.0]]])

        output = model(input_tensor)
        assert torch.allclose(output, expected_output), f"Expected {expected_output}, but got {output}"

    def test_forward_empty(self):
        model = DeltaTimeEncoding()
        input_tensor = torch.tensor([])
        expected_output = torch.tensor([])

        output = model(input_tensor)
        assert torch.allclose(output, expected_output), f"Expected {expected_output}, but got {output}"
