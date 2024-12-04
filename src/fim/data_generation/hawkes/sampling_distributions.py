import numpy as np


class Uniform:
    def __init__(self, **kwargs) -> None:
        self.low = kwargs["low"]
        self.high = kwargs["high"]

    def __call__(self):
        return np.random.uniform(self.low, self.high)
