import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def gaussian_kernel(t, t_i, alpha=0.8, sigma=1.0):
    """
    Computes the Gaussian kernel influence from event time t_i at time t.

    Parameters:
    - t (float): Current time.
    - t_i (float): Past event time.
    - alpha (float): Scaling factor for the excitation.
    - sigma (float): Standard deviation of the Gaussian kernel.

    Returns:
    - float: Kernel influence at time t.
    """
    return alpha * norm.pdf(t - t_i, scale=sigma)


def hawkes_intensity(t, event_times, lambda_0=0.5, alpha=0.8, sigma=1.0):
    """
    Computes the Hawkes process intensity function using a Gaussian kernel.

    Parameters:
    - t (float): Time at which to compute the intensity.
    - event_times (list): List of past event times.
    - lambda_0 (float): Baseline intensity.
    - alpha (float): Excitation factor.
    - sigma (float): Spread of the Gaussian kernel.

    Returns:
    - float: Intensity value at time t.
    """
    excitation = np.sum([gaussian_kernel(t, t_i, alpha, sigma) for t_i in event_times if t_i < t])
    return lambda_0 + excitation


# Generate synthetic event times
np.random.seed(42)
event_times = np.sort(np.random.uniform(0, 10, size=10))  # 10 random event times in [0, 10]

# Compute intensity over time
t_values = np.linspace(0, 10, 1000)
intensity_values = [hawkes_intensity(t, event_times) for t in t_values]

# Plot intensity function
plt.figure(figsize=(8, 4))
plt.plot(t_values, intensity_values, label="Hawkes Intensity (Gaussian Kernel)")
plt.scatter(event_times, np.zeros_like(event_times), color="red", marker="x", label="Events")
plt.xlabel("Time")
plt.ylabel("Intensity λ(t)")
plt.title("Hawkes Process with Gaussian Kernel")
plt.legend()
plt.savefig("hawkes_intensity.png")
