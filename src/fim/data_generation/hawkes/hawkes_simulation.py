import numpy as np
import torch
from tick.hawkes import HawkesKernelTimeFunc, SimuHawkes


def run_hawkes_simulation(baselines, kernel_grids, kernel_evaluations, num_paths, n_events_per_path, track_intensity, seed=0):
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
    track_intensity: bool
        Whether to track the intensity.

    Returns:
    event_times: np.array [num_paths, n_events_per_path]
        The event times.
    event_types: np.array [num_paths, n_events_per_path]
        The event types.
    """
    hawkes = SimuHawkes(baseline=baselines, max_jumps=n_events_per_path, seed=seed, verbose=False)
    hawkes.threshold_negative_intensity(allow=True)
    for i in range(len(kernel_grids)):
        kernel = HawkesKernelTimeFunc(t_values=kernel_grids[i], y_values=kernel_evaluations[i])
        hawkes.set_kernel(i, i, kernel)

    if track_intensity:
        smallest_kernel_timescale = np.min([kernel_grids[i][-1] for i in range(len(kernel_grids))])
        hawkes.track_intensity(intensity_track_step=smallest_kernel_timescale / 10)

    event_times = np.zeros((num_paths, n_events_per_path))
    event_types = np.zeros((num_paths, n_events_per_path))
    intensities = []
    intensity_times = []
    try:
        for i in range(num_paths):
            hawkes.reset()
            hawkes.simulate()
            event_times[i], event_types[i] = tick_timestamps_to_single_timeseries(hawkes.timestamps)
            if track_intensity:
                tracked_intensity = torch.tensor(np.array(hawkes.tracked_intensity))
                tracked_intensity_times = torch.tensor(np.array(hawkes.intensity_tracked_times))
                tracked_intensity_times = tracked_intensity_times - event_times[i][0]
                intensities.append(tracked_intensity)
                intensity_times.append(tracked_intensity_times)
    except Exception as e:
        print(f"Simulation failed with error: {e}")
        return None, None, None, None

    # Make sure that the events always start at 0
    event_times = event_times - event_times[:, 0][:, None]

    return event_times, event_types, intensity_times, intensities


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
