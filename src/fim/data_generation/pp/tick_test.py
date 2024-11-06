import itertools
import numpy as np
import matplotlib.pyplot as plt

from tick.plot import plot_basis_kernels, plot_hawkes_kernels
from tick.hawkes import SimuHawkes, HawkesKernelTimeFunc, HawkesBasisKernels

end_time = 1e5
C = 1e-3
kernel_size = 100
max_iter = 100

def lambd(t, alpha, beta):
    """
    Use the kernel that has been used in previous papers for the synthetic dataset.
    """
    return alpha * beta * np.exp(-beta * t)

t_values = np.linspace(0, 20, 1000)

# # Configuration 1
# alphas = [0.8]
# betas = [1.0]
# mus = [1]

# Configuration 2
alphas = [0.4,0.4]
betas = [1.0,20.]
mus = [0.2,0.2]

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

# import pdb
# pdb.set_trace()

# And then perform estimation using expectation maximization with two basis kernels
kernel_support = 20

em = HawkesBasisKernels(kernel_support, n_basis=len(alphas),
                        kernel_size=kernel_size, C=C, n_threads=4,
                        max_iter=max_iter, verbose=False, ode_tol=1e-5)
em.fit(ticks)

# fig = plot_hawkes_kernels(em, hawkes=hawkes, support=19.9, show=False)
# for ax in fig.axes:
#     ax.set_ylim([0.0, 0.5])
    
def g1(t):
    return lambd(t, alphas[0], betas[0])

def g2(t):
    return lambd(t, alphas[1], betas[1])

fig = plot_basis_kernels(em, basis_kernels=[g2, g1], show=False)
for ax in fig.axes:
    ax.set_ylim([0.0, 0.5])

plt.show()
plt.savefig('tick_test.png')