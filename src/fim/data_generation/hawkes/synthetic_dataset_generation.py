import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

from fim.leftovers_from_old_library import create_class_instance
from fim.data_generation.hawkes.hawkes_simulation import run_hawkes_simulation


class HawkesDatasetGenerator:
    def __init__(self, **kwargs) -> None:
        self.num_samples_train = kwargs["num_samples_train"]
        self.num_samples_test = kwargs["num_samples_test"]
        self.num_paths = kwargs["num_paths"]
        self.n_events_per_path = kwargs["n_events_per_path"]
        self.num_procs = kwargs["num_procs"]
        self.kernel_sampler = create_class_instance(kwargs["kernel_sampler"])

    def _generate_sample(self, _):
        baselines, kernel_grids, kernel_evaluations = self.kernel_sampler()
        event_time, event_type = run_hawkes_simulation(baselines, kernel_grids, kernel_evaluations, self.num_paths, self.n_events_per_path)
        return baselines, kernel_grids, kernel_evaluations, event_time, event_type

    def assemble(self, dtype=np.float32):
        num_samples = self.num_samples_train + self.num_samples_test
        num_marks = self.kernel_sampler.num_marks
        kernel_grid_size = self.kernel_sampler.kernel_grid_size

        baseline_data = []
        kernel_grid_data = []
        kernel_evaluation_data = []
        event_time_data = []
        event_type_data = []

        with Pool(self.num_procs) as pool:
            for result in tqdm(pool.imap(self._generate_sample, range(num_samples)), total=num_samples):
                baselines, kernel_grids, kernel_evaluations, event_time, event_type = result
                baseline_data.append(baselines)
                kernel_grid_data.append(kernel_grids)
                kernel_evaluation_data.append(kernel_evaluations)
                event_time_data.append(event_time)
                event_type_data.append(event_type)

        baseline_data = np.array(baseline_data, dtype=dtype)
        kernel_grid_data = np.array(kernel_grid_data, dtype=dtype)
        kernel_evaluation_data = np.array(kernel_evaluation_data, dtype=dtype)
        event_time_data = np.array(event_time_data, dtype=dtype)
        event_type_data = np.array(event_type_data, dtype=dtype)

        # Detect invalid samples, that is, samples with no events
        valid_samples = np.sum(event_time_data, axis=(1, 2)) != 0
        print("Ratio of invalid samples: ", (num_samples - np.sum(valid_samples)) / num_samples)

        # Remove invalid samples
        baseline_data = baseline_data[valid_samples]
        kernel_grid_data = kernel_grid_data[valid_samples]
        kernel_evaluation_data = kernel_evaluation_data[valid_samples]
        event_time_data = event_time_data[valid_samples]
        event_type_data = event_type_data[valid_samples]

        res = {
            "baseline_data": baseline_data,
            "kernel_grid_data": kernel_grid_data,
            "kernel_evaluation_data": kernel_evaluation_data,
            "event_time_data": event_time_data,
            "event_type_data": event_type_data,
        }
        return res


class HawkesKernelSampler:
    def __init__(self, **kwargs) -> None:
        self.num_marks = kwargs["num_marks"]
        self.kernel_grid_size = kwargs["kernel_grid_size"]
        self.baseline_sampler = create_class_instance(kwargs["baseline_sampler"])
        self.kernel_function_samplers = [
            create_class_instance(kernel_function_sampler) for kernel_function_sampler in kwargs["kernel_function_samplers"].values()
        ]

    def __call__(self):
        """
        Sample the parameters for the Hawkes kernel.

        Returns:
        kernel_grids: np.array
            The time grids on which the kernels get evaluated.
        kernel_evaluations: np.array
            The (diagonal) kernel evaluations.
        """
        kernel_grids = []
        kernel_evaluations = []
        baselines = []
        for _ in range(self.num_marks):
            # Randomly sample one of the kernel function samplers
            kernel_function_sampler = np.random.choice(self.kernel_function_samplers)
            grid, values = kernel_function_sampler(self.kernel_grid_size)
            kernel_grids.append(grid)
            kernel_evaluations.append(values)
            baselines.append(self.baseline_sampler())
        return baselines, np.array(kernel_grids), np.array(kernel_evaluations)
