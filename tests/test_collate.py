import torch

from fim.utils.collate import pad_data_collator


def test_pad_collate():
    # Define a batch of input data
    batch = [
        {"seq_len": 3, "target": torch.tensor([1, 2, 3]), "time_feat": torch.tensor([[1, 2021], [2, 2021], [3, 2021]])},
        {"seq_len": 2, "target": torch.tensor([4, 5]), "time_feat": torch.tensor([[1, 2021], [2, 2021]])},
        {"seq_len": 4, "target": torch.tensor([6, 7, 8, 9]), "time_feat": torch.tensor([[1, 2021], [2, 2021], [3, 2021], [4, 2021]])},
    ]

    padded_batch_target = {
        "seq_len": torch.tensor([3, 2, 4]),
        "target": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0], [6, 7, 8, 9]]),
        "time_feat": torch.tensor(
            [
                [[1, 2021], [2, 2021], [3, 2021], [0, 0]],
                [[1, 2021], [2, 2021], [0, 0], [0, 0]],
                [[1, 2021], [2, 2021], [3, 2021], [4, 2021]],
            ]
        ),
    }

    padded_batch = pad_data_collator(batch)

    assert padded_batch.keys() == padded_batch_target.keys()
    assert padded_batch["seq_len"].equal(padded_batch_target["seq_len"])
    assert padded_batch["target"].equal(padded_batch_target["target"])
    assert padded_batch["time_feat"].equal(padded_batch_target["time_feat"])
