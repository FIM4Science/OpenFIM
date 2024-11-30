from fim.trainers.lr_warmup import LinearWarmupScheduler


def test_LinearWarmupScheduler():

    def get_target_lr_fn(init_lr, target_lr, warmup_steps, step_count):
        if step_count < warmup_steps:
            return init_lr + (target_lr - init_lr) / warmup_steps * step_count
        else:
            return target_lr

    import torch
    from torch import nn

    model = nn.Linear(2, 1)

    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    scheduler = LinearWarmupScheduler(optimizer, target_lr=1.0, warmup_steps=100)
    target_lrs = [get_target_lr_fn(0, 1, 100, i) for i in range(1, 201)]
    lrs = []
    for i in range(200):
        optimizer.step()
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

    assert lrs == target_lrs
