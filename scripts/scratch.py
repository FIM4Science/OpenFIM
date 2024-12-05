import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim.lr_scheduler import SequentialLR

from fim.trainers.lr_warmup import LinearWarmupScheduler


model = nn.Linear(2, 1)


optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
l_warmup = LinearWarmupScheduler(optimizer, warmup_steps=10000)
cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50000)
seq_scheduler = SequentialLR(optimizer, [l_warmup, cosine_scheduler], [10000])


# Collect learning rates
lrs = []
for i in range(50000):
    seq_scheduler.step()
    lrs.append(optimizer.param_groups[0]["lr"])


plt.plot(lrs)
plt.xlabel("Iteration")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule")
plt.grid(True)
plt.savefig("lr_schedule.png")
plt.close()
