import warnings  # Added for warning formatting

import torch


# --- Interpolation Functions ---


def interpolate_1d_scalar_grid(grids: torch.Tensor, values: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    Performs linear interpolation for 1D data where grids/values are 1D
    and points are multi-dimensional. Handles broadcasting correctly.

    Args:
        grids (torch.Tensor): The x-coordinates of the grid points, shape [L_grid]. Must be sorted.
        values (torch.Tensor): The y-coordinates (values) at the grid points, shape [L_grid].
        points (torch.Tensor): The x-coordinates at which to interpolate, shape [...].

    Returns:
        torch.Tensor: The interpolated values at the query points, shape [...].
    """
    if grids.dim() != 1 or values.dim() != 1:
        raise ValueError(f"grids and values must be 1D, but got shapes {grids.shape} and {values.shape}")
    if grids.shape != values.shape:
        raise ValueError(f"Shapes of grids {grids.shape} and values {values.shape} must be the same.")
    L_grid = grids.shape[0]
    eps = torch.finfo(grids.dtype).eps

    # Ensure points are within the grid range, clamp if outside
    points_clamped = torch.clamp(points, grids[0], grids[-1])

    # Find indices: searchsorted handles broadcasting of 1D grids over points
    # indices will have the same shape as points
    indices = torch.searchsorted(grids, points_clamped, right=True)

    # Handle boundary cases: indices == 0 or indices == L_grid
    # Clamp indices to be valid for accessing grid elements (1 to L_grid - 1)
    indices = torch.clamp(indices, 1, L_grid - 1)

    # Gather surrounding grid points and values using the multi-dim indices
    # Shape of indices: [...]
    x0 = grids[indices - 1]  # Uses advanced indexing
    x1 = grids[indices]
    y0 = values[indices - 1]
    y1 = values[indices]

    # Calculate interpolation weights (handle division by zero if x0 == x1)
    denom = x1 - x0
    # Add small epsilon ONLY where denom is zero (or close to it) to avoid NaNs if points=x0=x1
    weight = (points_clamped - x0) / (denom + eps * (denom < eps))  # Use eps based on dtype
    weight = torch.clamp(weight, 0.0, 1.0)  # Ensure weight is in [0, 1] after division

    # Perform linear interpolation
    interpolated_values = y0 + weight * (y1 - y0)

    # Ensure that points exactly matching grid points return the exact grid value
    # Use a small tolerance for float comparison
    match_x0 = torch.isclose(points, x0, atol=eps * 10)
    match_x1 = torch.isclose(points, x1, atol=eps * 10)
    interpolated_values[match_x0] = y0[match_x0]
    interpolated_values[match_x1] = y1[match_x1]

    return interpolated_values


def interpolate_1d(grids: torch.Tensor, values: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    Performs linear interpolation for 1D data where grids/values and points share leading dimensions.
    Handles broadcasting.

    Args:
        grids (torch.Tensor): The x-coordinates of the grid points, shape [..., L_grid]. Must be sorted.
        values (torch.Tensor): The y-coordinates (values) at the grid points, shape [..., L_grid].
                               Leading dimensions must match `grids`.
        points (torch.Tensor): The x-coordinates at which to interpolate, shape [...L_points].
                               Leading dimensions must be broadcastable with grids.

    Returns:
        torch.Tensor: The interpolated values at the query points, shape depends on broadcasting.
                      Example: grids[B,P,Lk], points[B,P,Lp] -> output[B,P,Lp]
    """
    grid_shape = grids.shape
    points_shape = points.shape
    value_shape = values.shape
    L_grid = grid_shape[-1]
    eps = torch.finfo(grids.dtype).eps

    if not grid_shape[:-1] == value_shape[:-1]:
        raise ValueError(f"Leading dimensions of grids {grid_shape} and values {value_shape} must match.")

    # Check if broadcasting is needed and possible
    try:
        target_leading_shape = torch.broadcast_shapes(grid_shape[:-1], points_shape[:-1])
    except RuntimeError as e:
        raise ValueError(f"Cannot broadcast leading dimensions of grids {grid_shape[:-1]} and points {points_shape[:-1]}") from e

    # Expand dimensions if necessary for broadcasting rules of searchsorted/gather
    grids_b = grids.expand(*target_leading_shape, L_grid)
    values_b = values.expand(*target_leading_shape, L_grid)
    points_b = points.expand(*target_leading_shape, points_shape[-1])

    # --- Perform Interpolation ---
    points_clamped = torch.clamp(points_b, grids_b[..., 0:1], grids_b[..., -1:])

    # Find indices - searchsorted works with broadcasted shapes if leading dims match
    indices = torch.searchsorted(grids_b, points_clamped, right=True)
    indices = torch.clamp(indices, 1, L_grid - 1)

    # Gather surrounding points and values
    x0 = torch.gather(grids_b, -1, indices - 1)
    x1 = torch.gather(grids_b, -1, indices)
    y0 = torch.gather(values_b, -1, indices - 1)
    y1 = torch.gather(values_b, -1, indices)

    # Calculate weights
    denom = x1 - x0
    weight = (points_clamped - x0) / (denom + eps * (denom < eps))
    weight = torch.clamp(weight, 0.0, 1.0)

    # Interpolate
    interpolated_values = y0 + weight * (y1 - y0)

    # Correct for exact matches
    match_x0 = torch.isclose(points_b, x0, atol=eps * 10)
    match_x1 = torch.isclose(points_b, x1, atol=eps * 10)
    interpolated_values[match_x0] = y0[match_x0]
    interpolated_values[match_x1] = y1[match_x1]

    return interpolated_values


# --- Main Log-Likelihood Function ---


def hawkes_log_likelihood(
    event_times: torch.Tensor,
    event_types: torch.Tensor,
    kernel_grids: torch.Tensor,
    base_intensities: torch.Tensor,
    kernel_evaluations: torch.Tensor,
    num_mc_samples: int = 20,
    observation_end_time: float = -1.0,
) -> torch.Tensor:
    """
    Computes the log-likelihood of multivariate Hawkes process sequences using Monte Carlo integration for the integral term.
    Optimized for performance by vectorizing the intensity term calculation.

    Args:
        event_times (torch.Tensor): Absolute event times. Shape: [P, L, 1]. Padded with large values.
        event_types (torch.Tensor): Event types (marks), 0 to M-1. Shape: [P, L, 1]. Padded with a specific ID (e.g., M).
        kernel_grids (torch.Tensor): Time points relative to an event at which the kernel is evaluated.
                                     Shape: [M, L_kernel]. Assumed sorted.
        base_intensities (torch.Tensor): Base intensity `mu_m` for each event type. Shape: [M].
        kernel_evaluations (torch.Tensor): Kernel values `alpha_{m, *} * phi(delta_t)` evaluated at `kernel_grids`.
                                          Shape: [M, L_kernel]. M corresponds to the *target* event type.
        num_mc_samples (int): Number of samples per interval for Monte Carlo integration. Defaults to 20.
        observation_end_time (float): The end time T of the observation window. If -1.0 (default),
                                      it's inferred as the time of the last event in each sequence.

    Returns:
        torch.Tensor: The log-likelihood for each sequence in the batch. Shape: [P].
    """
    P, L, _ = event_times.shape
    M = base_intensities.shape[0]
    L_kernel = kernel_grids.shape[-1]
    device = event_times.device
    dtype = event_times.dtype
    eps = torch.finfo(dtype).eps

    # --- Input Validation and Setup ---
    # (Same as before)
    if kernel_grids.dim() != 2 or kernel_evaluations.dim() != 2:
        raise ValueError("kernel_grids and kernel_evaluations must be 2D tensors.")
    if kernel_grids.shape[0] != M or kernel_evaluations.shape[0] != M:
        raise ValueError(f"kernel_grids and kernel_evaluations must have leading dimension M={M}")
    if kernel_grids.shape[1] != L_kernel or kernel_evaluations.shape[1] != L_kernel:
        raise ValueError(f"kernel_grids and kernel_evaluations must have last dimension L_kernel={L_kernel}")
    if base_intensities.shape != (M,):
        raise ValueError(f"base_intensities must have shape [{M}], got {base_intensities.shape}")
    if event_times.shape != (P, L, 1) or event_types.shape != (P, L, 1):
        raise ValueError(f"event_times and event_types must have shape [P,L,1], got {event_times.shape}, {event_types.shape}")

    max_type_found = -1
    if (event_times < 1e9).any():  # Check if there are any non-padding times before taking max
        max_type_found = int(event_types[event_times.squeeze(-1) < 1e9].max().item())

    if max_type_found >= M:
        pad_token_id = M
        warnings.warn(f"event_types contains values >= M ({M}). Assuming {M} is the padding ID.", stacklevel=2)
    else:
        pad_token_id = -1

    if pad_token_id != -1:
        is_padding = (event_types >= pad_token_id).squeeze(-1)
    else:
        valid_times_exist = (event_times < 1e9).any()
        max_valid_time = event_times[event_times < 1e9].max().item() if valid_times_exist else 0.0
        is_padding = (event_times > (max_valid_time + 1.0)).squeeze(-1)
        if is_padding.any() and valid_times_exist:
            warnings.warn(
                f"No explicit padding ID detected (max_type={max_type_found} < M={M}), but large event times found relative to max valid time {max_valid_time:.2f}. Assuming these are padding.",
                stacklevel=2,
            )

    valid_event_mask = ~is_padding
    event_types_clamped = torch.clamp(event_types.squeeze(-1), 0, M - 1)
    seq_lengths = valid_event_mask.sum(dim=1)

    if observation_end_time < 0:
        last_event_indices = torch.clamp(seq_lengths - 1, min=0)
        T = event_times[torch.arange(P, device=device), last_event_indices, 0]
        T[seq_lengths == 0] = 0.0
    else:
        T = torch.full((P,), observation_end_time, device=device, dtype=dtype)

    # --- 1. Compute Log-Intensity Sum (VECTORIZED) ---
    event_times_flat = event_times.squeeze(-1)  # Shape [P, L]
    delta_times = event_times_flat.unsqueeze(2) - event_times_flat.unsqueeze(1)  # Shape [P, L, L]
    causal_mask = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1).unsqueeze(0)  # Shape [1, L, L]
    history_mask = valid_event_mask.unsqueeze(1)  # Shape [P, 1, L]
    intensity_contribution_mask = history_mask & causal_mask  # Shape [P, L, L]
    valid_delta_times_mask = intensity_contribution_mask & (delta_times > eps)  # Shape [P, L, L]
    target_types_for_kernel = event_types_clamped  # Shape [P, L]

    # Select kernel grids/evals based on target type i
    # Shape [P, L, Lk]
    selected_kernel_grids = kernel_grids[target_types_for_kernel]
    selected_kernel_evals = kernel_evaluations[target_types_for_kernel]

    # Interpolate using the general interpolate_1d function
    # shapes: grids[P,L,Lk], values[P,L,Lk], points[P,L,L] -> output[P,L,L]
    interpolated_kernels = interpolate_1d(selected_kernel_grids, selected_kernel_evals, delta_times)

    # Apply mask and sum over history j
    interpolated_kernels = interpolated_kernels * valid_delta_times_mask  # Shape [P, L, L]
    kernel_sum = torch.sum(interpolated_kernels, dim=2)  # Shape [P, L]

    # Add base intensity
    event_base_intensities = base_intensities[event_types_clamped]  # Shape [P, L]
    intensities_at_events = event_base_intensities + kernel_sum  # Shape [P, L]

    # Mask out padded events and sum log intensities (skip i=0)
    intensities_at_events_masked = intensities_at_events * valid_event_mask  # Shape [P, L]
    log_intensities_term = torch.log(intensities_at_events_masked[:, 1:] + eps) * valid_event_mask[:, 1:]  # Shape [P, L-1]
    log_intensity_sum = log_intensities_term.sum(dim=1)  # Shape [P]

    # --- 2. Compute Integral Term (MC - using scalar grid interpolation) ---
    integral_term = torch.zeros(P, device=device, dtype=dtype)
    time_deltas = event_times_flat.diff(dim=1)  # Shape [P, L-1]
    valid_interval_mask = valid_event_mask[:, 1:] & valid_event_mask[:, :-1]  # Shape [P, L-1]
    valid_interval_mask_expanded = valid_interval_mask.unsqueeze(-1)  # Shape [P, L-1, 1]

    # --- Integral over intervals [t_{i-1}, t_i] ---
    if valid_interval_mask.any():  # Only compute if there are valid intervals
        random_uniform = torch.rand(P, L - 1, num_mc_samples, device=device, dtype=dtype)  # [P, L-1, Ns]
        sample_times = event_times_flat[:, :-1].unsqueeze(2) + time_deltas.unsqueeze(2) * random_uniform  # [P, L-1, Ns]
        delta_times_samples = sample_times.unsqueeze(3) - event_times_flat.unsqueeze(1).unsqueeze(2)  # [P, L-1, Ns, L]

        valid_history_j_mask = valid_event_mask.unsqueeze(1).unsqueeze(2)  # [P, 1, 1, L]
        causal_sample_mask = delta_times_samples > eps  # [P, L-1, Ns, L]
        history_leq_interval_start_mask = torch.arange(L, device=device).view(1, 1, 1, L) <= torch.arange(L - 1, device=device).view(
            1, L - 1, 1, 1
        )  # [1, L-1, 1, L]
        sample_contribution_mask = valid_history_j_mask & causal_sample_mask & history_leq_interval_start_mask  # [P, L-1, Ns, L]

        total_intensity_at_samples = torch.zeros_like(sample_times)  # Shape [P, L-1, Ns]

        for m in range(M):  # Iterate over target types m for lambda_total
            m_kernel_grids = kernel_grids[m]  # Shape [Lk]
            m_kernel_evals = kernel_evaluations[m]  # Shape [Lk]
            interpolated_kernels_m = interpolate_1d_scalar_grid(
                m_kernel_grids, m_kernel_evals, delta_times_samples
            )  # Shape [P, L-1, Ns, L]
            interpolated_kernels_m = interpolated_kernels_m * sample_contribution_mask  # Shape [P, L-1, Ns, L]
            kernel_sum_m = torch.sum(interpolated_kernels_m, dim=3)  # Shape [P, L-1, Ns]
            intensity_m_at_samples = base_intensities[m] + kernel_sum_m  # Shape [P, L-1, Ns]
            total_intensity_at_samples += intensity_m_at_samples

        # Calculate MC integral estimate for intervals
        total_intensity_at_samples = total_intensity_at_samples * valid_interval_mask_expanded  # Shape [P, L-1, Ns]
        mean_intensity_in_interval = total_intensity_at_samples.mean(dim=2)  # Shape [P, L-1]
        integral_estimate_intervals = mean_intensity_in_interval * time_deltas  # Shape [P, L-1]
        integral_term = integral_estimate_intervals.sum(dim=1)  # Shape [P]

    # --- Integral from last event to T ---
    last_valid_indices = torch.clamp(seq_lengths - 1, min=0)  # Shape [P]
    last_event_times = event_times_flat[torch.arange(P, device=device), last_valid_indices]  # Shape [P]
    delta_t_to_T = T - last_event_times  # Shape [P]
    valid_last_interval_mask = (delta_t_to_T > eps) & (seq_lengths > 0)  # Shape [P]

    if valid_last_interval_mask.any():
        valid_P_indices = torch.where(valid_last_interval_mask)[0]
        P_valid = len(valid_P_indices)

        if P_valid > 0:
            last_event_times_valid = last_event_times[valid_P_indices]  # [Pv]
            delta_t_to_T_valid = delta_t_to_T[valid_P_indices]  # [Pv]
            random_uniform_last_valid = torch.rand(P_valid, num_mc_samples, device=device, dtype=dtype)  # [Pv, Ns]
            sample_times_last = (
                last_event_times_valid.unsqueeze(1) + delta_t_to_T_valid.unsqueeze(1) * random_uniform_last_valid
            )  # [Pv, Ns]

            event_times_hist_last = event_times_flat[valid_P_indices, :]  # [Pv, L]
            valid_event_mask_hist_last = valid_event_mask[valid_P_indices, :]  # [Pv, L]
            last_valid_indices_pv = last_valid_indices[valid_P_indices]  # [Pv]
            delta_times_samples_last = sample_times_last.unsqueeze(2) - event_times_hist_last.unsqueeze(1)  # [Pv, Ns, L]

            valid_history_j_mask_last = valid_event_mask_hist_last.unsqueeze(1)  # [Pv, 1, L]
            causal_sample_mask_last = delta_times_samples_last > eps  # [Pv, Ns, L]
            history_leq_last_event_mask = torch.arange(L, device=device).view(1, 1, L) <= last_valid_indices_pv.view(-1, 1, 1)  # [Pv, 1, L]
            sample_contribution_mask_last = valid_history_j_mask_last & causal_sample_mask_last & history_leq_last_event_mask  # [Pv, Ns, L]

            total_intensity_at_samples_last = torch.zeros_like(sample_times_last)  # [Pv, Ns]

            for m in range(M):  # Iterate over target types
                m_kernel_grids = kernel_grids[m]
                m_kernel_evals = kernel_evaluations[m]
                interpolated_kernels_m_last = interpolate_1d_scalar_grid(
                    m_kernel_grids, m_kernel_evals, delta_times_samples_last
                )  # [Pv, Ns, L]
                interpolated_kernels_m_last = interpolated_kernels_m_last * sample_contribution_mask_last  # [Pv, Ns, L]
                kernel_sum_m_last = torch.sum(interpolated_kernels_m_last, dim=2)  # [Pv, Ns]
                intensity_m_at_samples_last = base_intensities[m] + kernel_sum_m_last  # [Pv, Ns]
                total_intensity_at_samples_last += intensity_m_at_samples_last

            mean_intensity_last_interval = total_intensity_at_samples_last.mean(dim=1)  # [Pv]
            integral_last_interval = mean_intensity_last_interval * delta_t_to_T_valid  # [Pv]
            integral_term.scatter_add_(0, valid_P_indices, integral_last_interval)

    # --- Final Log-Likelihood ---
    log_likelihood = log_intensity_sum - integral_term
    # Ensure logL is 0 for sequences with no valid events (log_sum=0, integral=0)
    log_likelihood[seq_lengths == 0] = 0.0

    return log_likelihood


# --- Example Usage ---
if __name__ == "__main__":
    P = 3
    L = 6
    M = 3
    L_kernel = 20
    N_MC = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Setup ---
    times = torch.full((P, L, 1), 1000.0, device=device, dtype=torch.float32)
    times[0, :3, 0] = torch.tensor([0.0, 1.1, 2.5])
    times[1, :6, 0] = torch.tensor([0.0, 0.8, 1.5, 3.0, 4.2, 5.0])
    times[2, :, 0] = 1000.0
    event_times = times.to(device)

    types = torch.full((P, L, 1), M, dtype=torch.long, device=device)
    types[0, :3, 0] = torch.tensor([0, 1, 0])
    types[1, :6, 0] = torch.tensor([1, 2, 0, 1, 2, 1])
    event_types = types.to(device)

    # --- Model Parameters ---
    kernel_grids = torch.zeros(M, L_kernel, device=device)
    kernel_evaluations = torch.zeros(M, L_kernel, device=device)
    base_intensities = torch.rand(M, device=device, dtype=torch.float32) * 0.1 + 0.05

    for m in range(M):
        beta_m = (m + 1) * 0.5
        alpha_m = (m + 1) * 0.2
        grid_max_time = 5.0
        grid_points = torch.linspace(0, grid_max_time, L_kernel, device=device)
        kernel_grids[m] = grid_points
        kernel_evaluations[m] = alpha_m * beta_m * torch.exp(-beta_m * grid_points)

    # --- Calculate Log Likelihood ---
    print("\n--- Calculating Log Likelihood (T = last event) ---")
    log_likelihood_default_T = hawkes_log_likelihood(
        event_times=event_times,
        event_types=event_types,
        kernel_grids=kernel_grids,
        base_intensities=base_intensities,
        kernel_evaluations=kernel_evaluations,
        num_mc_samples=N_MC,
    )
    print("Log Likelihood (T=last event):", log_likelihood_default_T)

    print("\n--- Calculating Log Likelihood (T = 7.0) ---")
    log_likelihood_fixed_T = hawkes_log_likelihood(
        event_times=event_times,
        event_types=event_types,
        kernel_grids=kernel_grids,
        base_intensities=base_intensities,
        kernel_evaluations=kernel_evaluations,
        num_mc_samples=N_MC,
        observation_end_time=7.0,
    )
    print("Log Likelihood (T=7.0):", log_likelihood_fixed_T)

    print("\n--- Calculating Log Likelihood (T = 2.0) ---")
    log_likelihood_short_T = hawkes_log_likelihood(
        event_times=event_times,
        event_types=event_types,
        kernel_grids=kernel_grids,
        base_intensities=base_intensities,
        kernel_evaluations=kernel_evaluations,
        num_mc_samples=N_MC,
        observation_end_time=2.0,
    )
    print("Log Likelihood (T=2.0):", log_likelihood_short_T)
