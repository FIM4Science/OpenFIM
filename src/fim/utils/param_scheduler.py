import numpy as np


# TODO: It is better if they inherit from torch scheduler class so we have save_state. At the moment the state of the schedulers is not saved during checkpoining
class ExponentialScheduler:
    def __init__(self, max_steps: int, decay_rate: float = 0.1, **kwargs):
        self.max_steps = max_steps
        self.decay_rate = decay_rate

    def __call__(self, step):
        return float(1.0 / (1.0 + np.exp(-self.decay_rate * (step - self.max_steps))))


class ExponentialIncrease:
    """
    Increases exponentially from zero to max_value
    """

    def __init__(self, max_steps: int, training_fraction_to_reach_max: float = 0.5, max_value: float = 1.0, **kwargs):
        n_steps_to_rich_maximum = training_fraction_to_reach_max * max_steps

        self.max_value = max_value
        self.decay_rate = -np.log(1.0 - 0.99) / n_steps_to_rich_maximum

    def __call__(self, step):
        return self.max_value * float(1.0 - np.exp(-self.decay_rate * step))


class ExponentialSchedulerGumbel:
    """
    Exponential annealing for Gumbel-Softmax temperature
    """

    def __init__(
        self, max_steps: int, init_temperature: float, min_temperature: float, training_fraction_to_reach_min: float = 0.5
    ):
        self.init_temperature = init_temperature
        self.min_temperature = min_temperature

        n_steps_to_rich_minimum = training_fraction_to_reach_min * max_steps

        self.decay_rate = -np.log(self.min_temperature) / n_steps_to_rich_minimum

    def __call__(self, step):
        t = np.maximum(self.init_temperature * np.exp(-self.decay_rate * step), self.min_temperature)
        return t


class ConstantScheduler:
    def __init__(self, beta: float = 1.0, **kwargs):
        self.beta = beta

    def __call__(self, step):
        return self.beta


class LinearScheduler:
    def __init__(self, max_steps: int, start_step: int = 0, **kwargs):
        self.max_steps = max_steps
        self.start_step = start_step

    def __call__(self, step):
        if self.start_step == 0:
            return min(1.0, float(step) / self.max_steps)
        else:
            return min(1.0, self.start_step + float(step) / self.max_steps * (1 - self.start_step))


class MultiplicativeScheduler:
    """
    Multiplies current value by multiplier each step until end_value is reached
    """

    def __init__(self, start_step: int = 0, end_value: int = 0, multiplier: float = 0.9, **kwargs):
        self.start_value = start_step
        self.end_value = end_value
        self.multiplier = multiplier

    def __call__(self, step):
        beta = self.start_value * self.multiplier**step
        return min(self.end_value, beta) if self.multiplier > 1 else max(self.end_value, beta)


class PeriodicScheduler:
    """ """

    def __init__(self, max_steps: int, max_value: float = 1):
        self.max_steps = max_steps
        self.max_value = max_value

        self.quarter_epoch_length = self.max_steps * 0.25

    def __call__(self, step):
        step = step % self.max_steps
        if step < self.max_steps * 0.5:
            return 0
        elif step < self.max_steps * 0.75:
            return (step - 2 * self.quarter_epoch_length) / self.quarter_epoch_length * self.max_value
        else:
            return self.max_value
