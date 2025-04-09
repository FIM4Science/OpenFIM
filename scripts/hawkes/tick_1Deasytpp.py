import matplotlib.pyplot as plt
import numpy as np
from tick.hawkes import HawkesEM, HawkesKernelTimeFunc, SimuHawkes
from tick.plot import plot_hawkes_kernels


def kernel(t):
    return 0.8 * np.exp(-t)


kernel_grid = np.linspace(0, 7, 100)

kernel_evaluations = kernel(kernel_grid)

baselines = np.array([0.2])

NUM_PATHS = 1200
n_events_per_path = 250
n_events = NUM_PATHS * n_events_per_path

hawkes = SimuHawkes(baseline=baselines, max_jumps=n_events, seed=0, verbose=False)

kernel = HawkesKernelTimeFunc(t_values=kernel_grid, y_values=kernel_evaluations)
hawkes.set_kernel(0, 0, kernel)

hawkes.simulate()

model = HawkesEM(4, kernel_size=16, n_threads=8, verbose=False, tol=1e-3)
model.fit(hawkes.timestamps)
# model = HawkesExpKern(decays=1.0)
# model.fit(hawkes.timestamps)
print("Model Baseline:", model.baseline)

fig = plot_hawkes_kernels(model, hawkes=hawkes, show=False)

plt.savefig("tick_1Deasytpp.png")
