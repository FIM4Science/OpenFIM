import time
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from fim.data_generation.hawkes.hawkes_simulation import run_hawkes_simulation
from fim.leftovers_from_old_library import create_class_instance


class HawkesDatasetGenerator:
    def __init__(self, **kwargs) -> None:
        self.num_samples_train = kwargs["num_samples_train"]
        self.num_samples_val = kwargs["num_samples_val"]
        self.num_samples_test = kwargs["num_samples_test"]
        self.num_paths = kwargs["num_paths"]
        self.n_events_per_path = kwargs["n_events_per_path"]
        self.num_procs = kwargs["num_procs"]
        self.num_chunks = kwargs["num_chunks"]
        self.track_intensity = kwargs.get("track_intensity", False)
        self.kernel_sampler = create_class_instance(kwargs["kernel_sampler"])

    def _generate_sample(self, seed=0):
        event_time, event_type = None, None
        failed_attempts = 0
        while event_time is None or event_type is None:
            baselines, kernel_grids, kernel_evaluations = self.kernel_sampler()
            event_time, event_type, intensitity_time, intensity = run_hawkes_simulation(
                baselines, kernel_grids, kernel_evaluations, self.num_paths, self.n_events_per_path, self.track_intensity, seed=seed
            )
            if event_time is None:
                failed_attempts += 1
                # Change np seed based on current wallclock time
                np.random.seed(int(time.time() * 1e6) % 2**32)
                print(f"Simulation failed for the {failed_attempts} time.")
        return baselines, kernel_grids, kernel_evaluations, event_time, event_type, intensitity_time, intensity

    def assemble(self, dtype=np.float32):
        num_samples = self.num_samples_train + self.num_samples_val + self.num_samples_test

        baseline_data = []
        kernel_grid_data = []
        kernel_evaluation_data = []
        event_time_data = []
        event_type_data = []
        intensity_time_data = []
        intensity_data = []

        num_chunks = min(self.num_chunks, num_samples)
        samples_per_chunk = num_samples // num_chunks
        chunks = [range(i * samples_per_chunk, (i + 1) * samples_per_chunk) for i in range(num_chunks)]

        # Handle any remaining samples
        if num_samples % num_chunks != 0:
            chunks.append(range(num_chunks * samples_per_chunk, num_samples))

        with Pool(self.num_procs) as pool:
            for result in tqdm(pool.imap(self._generate_chunk, chunks), total=len(chunks)):
                baselines, kernel_grids, kernel_evaluations, event_time, event_type, intensitity_times, intensities = result

                baseline_data.extend(baselines)
                kernel_grid_data.extend(kernel_grids)
                kernel_evaluation_data.extend(kernel_evaluations)
                event_time_data.extend(event_time)
                event_type_data.extend(event_type)
                intensity_time_data.extend(intensitity_times)
                intensity_data.extend(intensities)

        baseline_data = np.array(baseline_data, dtype=dtype)
        kernel_grid_data = np.array(kernel_grid_data, dtype=dtype)
        kernel_evaluation_data = np.array(kernel_evaluation_data, dtype=dtype)
        event_time_data = np.array(event_time_data, dtype=dtype)
        event_type_data = np.array(event_type_data, dtype=dtype)

        # Reshape the data to match our requirements
        event_time_data = event_time_data[:, :, :, None]
        event_type_data = event_type_data[:, :, :, None]

        res = {
            "base_intensities": baseline_data,  # [B, M]
            "kernel_grids": kernel_grid_data,  # [B, M, L_kernel]
            "kernel_evaluations": kernel_evaluation_data,  # [B, M, L_kernel]
            "event_times": event_time_data,  # [B, P, L, 1]
            "event_types": event_type_data,  # [B, P, L, 1]
        }
        if self.track_intensity:
            res["intensity_times"] = intensity_time_data
            res["intensities"] = intensity_data

        return res

    def _generate_chunk(self, chunk_range):
        baselines = []
        kernel_grids = []
        kernel_evaluations = []
        event_times = []
        event_types = []
        intensitity_times = []
        intensities = []

        for sample_idx in chunk_range:
            np.random.seed(sample_idx)  # TODO: Check if we can make this faster
            baseline, kernel_grid, kernel_evaluation, event_time, event_type, intensitity_time, intensity = self._generate_sample(
                seed=sample_idx
            )
            baselines.append(baseline)
            kernel_grids.append(kernel_grid)
            kernel_evaluations.append(kernel_evaluation)
            event_times.append(event_time)
            event_types.append(event_type)
            intensitity_times.append(intensitity_time)
            intensities.append(intensity)

        return baselines, kernel_grids, kernel_evaluations, event_times, event_types, intensitity_times, intensities


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
