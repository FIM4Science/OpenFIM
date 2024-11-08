import itertools
import numpy as np
import matplotlib.pyplot as plt
import time

from tick.plot import plot_basis_kernels, plot_hawkes_kernels
from tick.hawkes import SimuHawkes, HawkesKernelTimeFunc, HawkesBasisKernels

end_time = 1e3
num_marks = 1
num_paths = 300

def lambd(t, alpha, beta):
    """
    Use the kernel that has been used in previous papers for the synthetic dataset.
    """
    return alpha * beta * np.exp(-beta * t)

t_values = np.linspace(0, 20, 1000)

alphas = [0.4]*num_marks
betas = [1.0]*num_marks
mus = [0.2]*num_marks

function_values = []
for i in range(len(alphas)):
    function_values.append(lambd(t_values, alphas[i], betas[i]))

hawkes = SimuHawkes(baseline=mus, seed=1093, verbose=False)
for i in range(len(alphas)):
    kernel = HawkesKernelTimeFunc(t_values=t_values, y_values=function_values[i])
    hawkes.set_kernel(i, i, kernel)

hawkes.end_time = end_time
hawkes.simulate()
ticks = hawkes.timestamps

print("number of events: %f" % hawkes.n_total_jumps)

t = time.time()
for _ in range(num_paths):
    hawkes.reset()
    hawkes.simulate()
print("Time taken: %f" % (time.time() - t))

import pdb
pdb.set_trace()

