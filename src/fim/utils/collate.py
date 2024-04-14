import torch


def pad_data_collator(batch):
    """
    Collates a batch of data samples by padding sequences to the maximum sequence length.

    Args:
        batch (list): A list of dictionaries, where each dictionary represents a data sample.

    Returns:
        dict: A dictionary containing the padded batch of data samples.

    """
    max_seq_len = max([x["seq_len"] for x in batch])

    # Initialize the padded batch dictionary
    padded_batch = {}

    for key in batch[0].keys():
        if key == "seq_len":
            padded_batch[key] = torch.tensor([x[key] for x in batch])
        else:
            padded_batch[key] = torch.zeros(len(batch), max_seq_len, *batch[0][key].shape[1:])

            for i, item in enumerate(batch):
                padded_batch[key][i, : item["seq_len"]] = item[key]

    return padded_batch
