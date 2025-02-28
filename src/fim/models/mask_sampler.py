import torch


class KernelGridSubSampler:
    """
    Sample a random subset of points on which we evaluate the kernel function.
    """

    def __init__(self, **kwargs) -> None:
        self.num_points = kwargs["num_points"]  # Number of points on which we evaluate the kernel function

    def __call__(self, x) -> None:
        """
        Sample a mask for the kernel function evaluated on the grid x.
        """
        batch_size, _, grid_size = x["kernel_grids"].shape
        device = x["kernel_grids"].device

        # Generate random scores for each grid point
        rand_scores = torch.rand(batch_size, grid_size, device=device)

        # Select the top `num_points` indices per batch
        selected_idxs = torch.topk(rand_scores, self.num_points, dim=1, largest=False).indices

        # Sort the selected indices
        selected_idxs, _ = torch.sort(selected_idxs, dim=1)

        # Expand indices for gathering
        expanded_idxs = selected_idxs.unsqueeze(1).expand(-1, x["kernel_grids"].shape[1], -1)

        # Subsample the kernel grid and evaluations
        x["kernel_grids"] = x["kernel_grids"].gather(2, expanded_idxs)
        x["kernel_evaluations"] = x["kernel_evaluations"].gather(2, expanded_idxs)


def sample_exponential_indices(size, scale, num_samples):
    """
    Sample indices from 0 to size-1 using an exponential distribution.

    :param size: The total number of indices (N).
    :param scale: The scale parameter for the exponential distribution.
                  A smaller scale means a steeper distribution, leading to lower indices being more likely.
    :param num_samples: Number of indices to sample.
    :return: Tensor of sampled indices.
    """
    # Generate exponential random variables
    exp_samples = torch.distributions.Exponential(rate=1 / scale).sample((num_samples,))

    # Normalize to fit within the range of indices
    exp_samples_normalized = exp_samples / torch.max(exp_samples)
    indices = (exp_samples_normalized * (size - 1)).long()

    return indices


class ObservationMaskSampler:
    """
    Create a mask for the observation values.
    """

    def __init__(self, **kwargs) -> None:
        self.min_max_seq_len = kwargs["min_max_seq_len"]  # Minimum maximum sequence length
        self.max_max_seq_len = kwargs["max_max_seq_len"]
        self.scale = kwargs["scale"]  # Scale parameter for the exponential distribution

    def __call__(self, x) -> None:
        batch_size, num_paths, num_points, _ = x["event_times"].shape
        # Sample the maximum sequence length from a uniform distribution
        max_seq_len = torch.randint(self.min_max_seq_len, self.max_max_seq_len + 1, (batch_size,))
        observation_masks = torch.zeros(batch_size, num_paths, num_points, 1, dtype=torch.bool)
        for i in range(batch_size):
            min_seq_len = torch.randint(1, max_seq_len[i] // 2, (1,))
            # Sample the number of observed events
            seq_lens = sample_exponential_indices(max_seq_len[i] - min_seq_len, self.scale, num_paths) + min_seq_len
            for j in range(num_paths):
                observation_masks[i, j, : seq_lens[j]] = True
        # Move observation masks to the device
        observation_masks = observation_masks.to(x["event_times"].device)
        x["obervation_masks"] = observation_masks
        # Also mask out the observation values to make sure that the information is not leaked
        x["event_types"] = x["event_types"] * observation_masks.float()
        x["event_times"] = x["event_times"] * observation_masks.float()
