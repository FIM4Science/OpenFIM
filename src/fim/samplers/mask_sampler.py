import numpy as np
from torch import Tensor


class BernoulliMaskSampler:
    def __init__(self, **kwargs) -> None:
        """
        BernoulliMaskSampler

        Generates a mask for the input data using a Bernoulli distribution.
        Parameters:
           survival_probability (float): The probability of each element surviving (i.e., being 1 in the mask).
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
        mask = np.random.binomial(size=mask_shape[:-1], n=1, p=self.survival_probability)

        while (mask.sum(axis=-1) < self.min_survival_count).any() is True:
            resample_mask = mask.sum(axis=-1) < self.min_survival_count
            resample_count = mask[resample_mask].shape
            mask[resample_mask] = np.random.binomial(size=resample_count, n=1, p=self.survival_probability)

        return mask
