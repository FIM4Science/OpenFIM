import numpy as np


class Uniform:
    def __init__(self, **kwargs) -> None:
        self.low = kwargs["low"]
        self.high = kwargs["high"]

    def __call__(self):
        return np.random.uniform(self.low, self.high)
    
class LogUniform:
    """
    Sample from a log-uniform distribution.
    This distribution is uniform in log space, making it suitable for parameters
    that should be sampled across multiple orders of magnitude.
    """
    def __init__(self, **kwargs) -> None:
        self.low = kwargs["low"]
        self.high = kwargs["high"]
        
        # Store log values for efficient sampling
        self.log_low = np.log(self.low)
        self.log_high = np.log(self.high)
        
    def __call__(self):
        # Sample uniformly in log space, then exponentiate
        log_val = np.random.uniform(self.log_low, self.log_high)
        return np.exp(log_val)
