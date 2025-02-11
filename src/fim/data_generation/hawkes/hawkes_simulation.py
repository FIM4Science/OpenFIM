import numpy as np
from tick.hawkes import HawkesKernelTimeFunc, SimuHawkes


def run_hawkes_simulation(baselines, kernel_grids, kernel_evaluations, num_paths, n_events_per_path, seed=0):
    """
    Run a Hawkes simulation with the given kernels.

    Args:
    baselines: np.array of length num_marks
        The time independent intensities.
    kernel_grids: np.array of length num_marks
        The time grids on which the kernels get evaluated.
    kernel_evaluations: np.array of length num_marks
        The (diagonal) kernel evaluations.
    num_paths: int
        The number of paths to simulate.
    n_events_per_path: int
        The number of events per path.

    Returns:
    event_times: np.array [num_paths, n_events_per_path]
        The event times.
    event_types: np.array [num_paths, n_events_per_path]
        The event types.
    """
    hawkes = SimuHawkes(baseline=baselines, max_jumps=n_events_per_path, seed=seed, verbose=False)
    for i in range(len(kernel_grids)):
        kernel = HawkesKernelTimeFunc(t_values=kernel_grids[i], y_values=kernel_evaluations[i])
        hawkes.set_kernel(i, i, kernel)

    event_times = np.zeros((num_paths, n_events_per_path))
    event_types = np.zeros((num_paths, n_events_per_path))
    try:
        for i in range(num_paths):
            hawkes.reset()
            hawkes.simulate()
            event_times[i], event_types[i] = tick_timestamps_to_single_timeseries(hawkes.timestamps)
    except Exception as e:
        print(f"Simulation failed with error: {e}")
        return None, None

    return event_times, event_types


def tick_timestamps_to_single_timeseries(tick_timestamps):
    """
    The tick library returns the timestamps for every event type.
    We want to have a single time series.
    """
    # Create a list of all event timestamps and their corresponding event types
    event_times = np.concatenate(tick_timestamps)
    event_types = np.concatenate([[event_type] * len(events) for event_type, events in enumerate(tick_timestamps)])

    # Sort indices based on event_times
    sorted_indices = np.argsort(event_times)

    return event_times[sorted_indices], event_types[sorted_indices]
