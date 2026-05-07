import numpy as np
import torch
from torch import Tensor


def sample_exponential_within_range(rate: float, max_value: float, size: tuple) -> torch.Tensor:
    """
    Samples values from an exponential distribution within the range [0, max_value].

    Parameters:
        rate (float): The rate parameter of the exponential distribution (1/scale).
        max_value (float): The maximum value for the sampled data.
        size (tuple): The shape of the output tensor.

    Returns:
        torch.Tensor: A tensor with sampled values between 0 and max_value.
    """
    # Sample from the exponential distribution
    samples = torch.distributions.Exponential(rate).sample(size)

    # Clip the samples to ensure they are within [0, max_value]
    samples = torch.clamp(samples, max=max_value)
    zeros = torch.zeros((*samples.shape[:-1], 1), device=samples.device, dtype=samples.dtype)
    return torch.sort(torch.cat((zeros, samples), dim=-1))[0]


def sample_exponential_indices(size: int, scale: float, num_samples: int) -> Tensor:
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


def sample_kernel_grid(distribution_type: str, **kwargs) -> torch.Tensor:
    """Sample points from a specified distribution within a specified range.

    Args:
        distribution_type (str): The type of distribution to sample from (e.g., "exponential").
        **kwargs: Additional keyword arguments specific to the distribution type.

    Returns:
        torch.Tensor: Sampled points from the specified distribution.
    """
    if distribution_type == "exponential":
        rate = kwargs.get("rate", 1.0)
        max_value = kwargs.get("max_value", 1.0)
        size = kwargs.get("size", (1,))
        return sample_exponential_within_range(rate, max_value, size)
    else:
        raise ValueError(f"Unsupported distribution type: {distribution_type}")


class BernoulliMaskSampler:
    def __init__(self, **kwargs) -> None:
        """
        BernoulliMaskSampler

        Generates a mask for the input data using a Bernoulli distribution.
        Parameters:
           survival_probability (float): The MINIMUM probability of each element surviving. The actual probability used per series is sampled
                                         uniformly between this value and 1.0.
           min_survival_count (int, optional): The minimum number of surviving elements required per sample. Defaults to 0.
        Callable:
           __call__(data: Tensor) -> Tensor:
              Generates a mask for the given data tensor, ensuring at least min_survival_count elements survive along the last axis.
        """
        assert "survival_probability" in kwargs, "survival_probability is a required parameter"
        self.survival_probability = kwargs.get("survival_probability")
        self.min_survival_count = kwargs.get("min_survival_count", 0)

    def __call__(self, data: Tensor) -> Tensor:
        mask_shape = data.shape
        survival_probability = np.random.uniform(self.survival_probability, 1.0, size=(mask_shape[0], 1, 1))
        mask = np.random.binomial(size=mask_shape[:-1], n=1, p=survival_probability)

        while (mask.sum(axis=-1) < self.min_survival_count).any() is True:
            resample_mask = mask.sum(axis=-1) < self.min_survival_count
            resample_count = mask[resample_mask].shape
            mask[resample_mask] = np.random.binomial(size=resample_count, n=1, p=survival_probability[resample_mask])

        return torch.from_numpy(mask).unsqueeze(-1).bool()
