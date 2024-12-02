# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=line-too-long

import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from fim.trainers.lr_warmup import LinearWarmupScheduler
from fim.utils.helper import GenericConfig, create_lr_schedulers


class TestWarmupSchedulers:
    @pytest.fixture
    def optimizer(self):
        model = nn.Linear(2, 1)

        return torch.optim.SGD(model.parameters(), lr=0.1)

    def test_LinearWarmupScheduler(self, optimizer):
        def get_target_lr_fn(init_lr, target_lr, warmup_steps, step_count):
            if step_count < warmup_steps:
                return init_lr + (target_lr - init_lr) / warmup_steps * step_count
            else:
                return target_lr

        scheduler = LinearWarmupScheduler(optimizer, target_lr=1.0, warmup_steps=100)
        target_lrs = [get_target_lr_fn(0, 1, 100, i) for i in range(1, 201)]
        lrs = []
        for _ in range(200):
            optimizer.step()
            lrs.append(scheduler.get_last_lr()[0])
            scheduler.step()

        assert lrs == target_lrs

    def test_init_warmup_scheduler(self, optimizer):
        scheduler_config = {
            "name": "torch.optim.lr_scheduler.SequentialLR",
            "schedulers": [
                GenericConfig({"name": "fim.trainers.lr_warmup.LinearWarmupScheduler", "target_lr": 1.0, "warmup_steps": 100}),
                GenericConfig(
                    {"name": "torch.optim.lr_scheduler.CosineAnnealingLR", "T_max": "*epochs", "eta_min": 0.0000001, "last_epoch": -1}
                ),
            ],
            "milestones": [100],
        }
        scheduler = create_lr_schedulers(GenericConfig(scheduler_config), optimizer)[0][1]

        assert isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR)
        assert len(scheduler._schedulers) == 2
        assert isinstance(scheduler._schedulers[0], LinearWarmupScheduler)
        assert isinstance(scheduler._schedulers[1], CosineAnnealingLR)
