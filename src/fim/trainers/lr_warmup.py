from torch.optim.lr_scheduler import LRScheduler, _warn_get_lr_called_within_step
from torch.optim.optimizer import Optimizer


class LinearWarmupScheduler(LRScheduler):
    def __init__(self, optimizer: Optimizer, init_lr=0.0, target_lr=0.1, warmup_steps: int = 100, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.init_lr = init_lr
        self.target_lr = target_lr
        self.__warmup_factor = (target_lr - init_lr) / warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        _warn_get_lr_called_within_step(self)
        return [self.init_lr + min(self._step_count, self.warmup_steps) * self.__warmup_factor for _ in self.optimizer.param_groups]
