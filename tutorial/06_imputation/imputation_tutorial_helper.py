import matplotlib.pyplot as plt
import numpy as np
import torch


def prepare_data(ts, values, mask=None, window_count=3, imp_start=65, imp_end=70):
    """
    Prepare simple time series data for FIM imputation model.

    Args:
        ts: Time points tensor of shape [1, 1, T, 1]
        values: Value tensor of shape [1, 1, T, 1]
        mask: Boolean mask tensor of shape [1, 1, T, 1] (True = observed, False = missing)
        window_count: Number of windows (3 or 5)
        imputation_start: Start index for imputation window
        imputation_end: End index for imputation window

    Returns:
        batch: Dictionary suitable for FIM model
    """
    if mask is None:
        mask = torch.zeros_like(ts).bool()

    B, _, T, D = ts.shape  # Should be [1, 1, T, 1]

    # Create location times and values (imputation window)
    loc_times = ts[0, 0, imp_start:imp_end].unsqueeze(0)  # [1, imp_len, 1]
    loc_values = values[0, 0, imp_start:imp_end].unsqueeze(0)  # [1, imp_len, 1]

    # Create context windows
    if window_count == 3:
        # Left context, imputation window (index 1), right context
        left_context_values = values[:, :, :imp_start]  # [1, 1, left_len, 1]
        right_context_values = values[:, :, imp_end:]  # [1, 1, right_len, 1]

        left_context_times = ts[:, :, :imp_start]
        right_context_times = ts[:, :, imp_end:]

        left_context_mask = mask[:, :, :imp_start]
        right_context_mask = mask[:, :, imp_end:]

        # Pad to same length (max_length_window)
        max_len = max(left_context_values.shape[2], right_context_values.shape[2])
        max_len = max(max_len, 64)  # Minimum window size

        # Pad left context
        left_pad_size = max_len - left_context_values.shape[2]
        if left_pad_size > 0:
            left_context_values = torch.cat([torch.zeros(B, 1, left_pad_size, D), left_context_values], dim=2)
            left_context_times = torch.cat([torch.zeros(B, 1, left_pad_size, 1), left_context_times], dim=2)
            left_context_mask = torch.cat([torch.zeros(B, 1, left_pad_size, 1, dtype=torch.bool), left_context_mask], dim=2)
        else:
            # Truncate if too long
            left_context_values = left_context_values[:, :, -max_len:]
            left_context_times = left_context_times[:, :, -max_len:]
            left_context_mask = left_context_mask[:, :, -max_len:]

        # Pad right context
        right_pad_size = max_len - right_context_values.shape[2]
        if right_pad_size > 0:
            right_context_values = torch.cat([right_context_values, torch.zeros(B, 1, right_pad_size, D)], dim=2)
            right_context_times = torch.cat([right_context_times, torch.zeros(B, 1, right_pad_size, 1)], dim=2)
            right_context_mask = torch.cat([right_context_mask, torch.zeros(B, 1, right_pad_size, 1, dtype=torch.bool)], dim=2)
        else:
            # Truncate if too long
            right_context_values = right_context_values[:, :, :max_len]
            right_context_times = right_context_times[:, :, :max_len]
            right_context_mask = right_context_mask[:, :, :max_len]

        # Stack context windows: [B, window_count-1, max_len, D]
        context_values = torch.stack([left_context_values.squeeze(1), right_context_values.squeeze(1)], dim=1)
        context_times = torch.stack([left_context_times.squeeze(1), right_context_times.squeeze(1)], dim=1)
        context_mask = torch.stack([left_context_mask.squeeze(1), right_context_mask.squeeze(1)], dim=1)

    elif window_count == 5:
        # Split left and right contexts into two parts each
        left_len = imp_start
        right_len = T - imp_end

        # Left contexts
        left_mid = left_len // 2
        left_context1_values = values[:, :, :left_mid]
        left_context2_values = values[:, :, left_mid:imp_start]

        # Right contexts
        right_mid = (T + imp_end) // 2
        right_context1_values = values[:, :, imp_end:right_mid]
        right_context2_values = values[:, :, right_mid:]

        # Similar for times and masks
        left_context1_times = ts[:, :, :left_mid]
        left_context2_times = ts[:, :, left_mid:imp_start]
        right_context1_times = ts[:, :, imp_end:right_mid]
        right_context2_times = ts[:, :, right_mid:]

        left_context1_mask = mask[:, :, :left_mid]
        left_context2_mask = mask[:, :, left_mid:imp_start]
        right_context1_mask = mask[:, :, imp_end:right_mid]
        right_context2_mask = mask[:, :, right_mid:]

        # Pad all to same length
        contexts = [left_context1_values, left_context2_values, right_context1_values, right_context2_values]
        times = [left_context1_times, left_context2_times, right_context1_times, right_context2_times]
        masks = [left_context1_mask, left_context2_mask, right_context1_mask, right_context2_mask]

        max_len = max([c.shape[2] for c in contexts])
        max_len = max(max_len, 64)

        padded_contexts = []
        padded_times = []
        padded_masks = []

        for i, (ctx_val, ctx_time, ctx_mask) in enumerate(zip(contexts, times, masks)):
            pad_size = max_len - ctx_val.shape[2]
            if pad_size > 0:
                if i < 2:  # Left contexts - pad at beginning
                    ctx_val = torch.cat([torch.zeros(B, 1, pad_size, D), ctx_val], dim=2)
                    ctx_time = torch.cat([torch.zeros(B, 1, pad_size, 1), ctx_time], dim=2)
                    ctx_mask = torch.cat([torch.zeros(B, 1, pad_size, 1, dtype=torch.bool), ctx_mask], dim=2)
                else:  # Right contexts - pad at end
                    ctx_val = torch.cat([ctx_val, torch.zeros(B, 1, pad_size, D)], dim=2)
                    ctx_time = torch.cat([ctx_time, torch.zeros(B, 1, pad_size, 1)], dim=2)
                    ctx_mask = torch.cat([ctx_mask, torch.zeros(B, 1, pad_size, 1, dtype=torch.bool)], dim=2)
            else:
                # Truncate if too long
                if i < 2:  # Left contexts - keep end
                    ctx_val = ctx_val[:, :, -max_len:]
                    ctx_time = ctx_time[:, :, -max_len:]
                    ctx_mask = ctx_mask[:, :, -max_len:]
                else:  # Right contexts - keep beginning
                    ctx_val = ctx_val[:, :, :max_len]
                    ctx_time = ctx_time[:, :, :max_len]
                    ctx_mask = ctx_mask[:, :, :max_len]

            padded_contexts.append(ctx_val.squeeze(1))
            padded_times.append(ctx_time.squeeze(1))
            padded_masks.append(ctx_mask.squeeze(1))

        context_values = torch.stack(padded_contexts, dim=1)
        context_times = torch.stack(padded_times, dim=1)
        context_mask = torch.stack(padded_masks, dim=1)

    # Find initial conditions (last observed point in left context, first in right context)
    if window_count == 3:
        # Find last True in left context
        left_mask = context_mask[:, 0].squeeze(-1)  # [B, max_len]
        reversed_left = torch.flip(left_mask, dims=[1])
        last_idx_left = torch.argmax(reversed_left.int(), dim=1)
        last_idx_left = left_mask.shape[1] - 1 - last_idx_left

        # Find first True in right context
        right_mask = context_mask[:, 1].squeeze(-1)  # [B, max_len]
        first_idx_right = torch.argmax(right_mask.int(), dim=1)

        batch_indices = torch.arange(B)
        linitial_conditions = context_values[:, 0][batch_indices, last_idx_left]  # [B, D]
        rinitial_conditions = context_values[:, 1][batch_indices, first_idx_right]  # [B, D]

    else:  # window_count == 5
        # Use last point of second left context and first point of third context (first right)
        left2_mask = context_mask[:, 1].squeeze(-1)
        reversed_left2 = torch.flip(left2_mask, dims=[1])
        last_idx_left2 = torch.argmax(reversed_left2.int(), dim=1)
        last_idx_left2 = left2_mask.shape[1] - 1 - last_idx_left2

        right1_mask = context_mask[:, 2].squeeze(-1)
        first_idx_right1 = torch.argmax(right1_mask.int(), dim=1)

        batch_indices = torch.arange(B)
        linitial_conditions = context_values[:, 1][batch_indices, last_idx_left2]
        rinitial_conditions = context_values[:, 2][batch_indices, first_idx_right1]

    # Create padding mask for locations (all False for no padding in this simple case)
    padding_mask_locations = torch.zeros(B, loc_times.shape[1], dtype=torch.bool)

    # Prepare batch dictionary
    batch = {
        "location_times": loc_times.float(),
        "target_sample_path": loc_values.float(),
        "initial_conditions": loc_values[:, 0].float(),  # First point of imputation window
        "observation_values": context_values.float(),
        "linitial_conditions": linitial_conditions.float(),
        "rinitial_conditions": rinitial_conditions.float(),
        "observation_mask": context_mask,
        "observation_times": context_times.float(),
        "padding_mask_locations": padding_mask_locations,
    }

    return batch


def prepare_data_base(ts, values, mask=None, fine_grid_times=None):
    """
    Prepare data for the base fim-imp-pointwise-base model.

    Args:
        ts: Time points tensor of shape [B, T, 1]
        values: Value tensor of shape [B, T, 1]
        mask: Boolean mask tensor of shape [B, T, 1] (True = missing, False = observed)
        fine_grid_times: Fine grid for evaluation, if None uses same as ts

    Returns:
        batch: Dictionary suitable for base fim-imp-pointwise-base model
    """
    if mask is None:
        mask = torch.zeros_like(ts).bool()  # If no mask is given, all data was observed

    if fine_grid_times is None:
        fine_grid_times = ts.clone()

    # The model fim-imp-pointwise-base expects this format based on its forward method
    B, T, D = ts.shape
    assert T % 4 == 0, f"T dimension ({T}) must be divisible by 4"

    T_chunk = T // 4

    # Split tensors along T dimension into 4 parts: [B, T, D] -> [B, 4, T/4, D]
    ts_split = ts.reshape(B, 4, T_chunk, D)
    values_split = values.reshape(B, 4, T_chunk, D)
    mask_split = mask.reshape(B, 4, T_chunk, D)
    fine_grid_times_split = fine_grid_times.reshape(B, 4, T_chunk, D)

    # Merge batch and split dimension: [B, 4, T/4, D] -> [B*4, T/4, D]
    ts_merged = ts_split.reshape(B * 4, T_chunk, D)
    values_merged = values_split.reshape(B * 4, T_chunk, D)
    mask_merged = mask_split.reshape(B * 4, T_chunk, D)
    fine_grid_times_merged = fine_grid_times_split.reshape(B * 4, T_chunk, D)

    # The model fim-imp-pointwise-base expects this format
    batch = {
        # Coarse grid (observations)
        "coarse_grid_noisy_sample_paths": values_merged,  # [B*4, T/4, 1]
        "coarse_grid_grid": ts_merged,  # [B*4, T/4, 1]
        "coarse_grid_observation_mask": mask_merged,  # [B*4, T/4, 1]
        # Fine grid (evaluation points)
        "fine_grid_grid": fine_grid_times_merged,  # [B*4, T/4, 1]
        "fine_grid_sample_paths": values_merged,  # [B*4, T/4, 1] - target if available
    }

    return batch


def prepare_data_base(ts, values, mask=None, fine_grid_times=None, num_windows=4):
    """
    Prepare data for the base fim-imp-pointwise-base model.

    Args:
        ts: Time points tensor of shape [B, T, 1]
        values: Value tensor of shape [B, T, 1]
        mask: Boolean mask tensor of shape [B, T, 1] (True = missing, False = observed)
        fine_grid_times: Fine grid for evaluation, if None uses same as ts
        num_windows: Number of windows to split data into (default: 4)

    Returns:
        batch: Dictionary suitable for base fim-imp-pointwise-base model
    """
    if mask is None:
        mask = torch.zeros_like(ts).bool()  # If no mask is given, all data was observed

    if fine_grid_times is None:
        fine_grid_times = ts.clone()

    # The model fim-imp-pointwise-base expects this format based on its forward method
    B, T, D = ts.shape
    assert T % num_windows == 0, f"T dimension ({T}) must be divisible by num_windows ({num_windows})"

    T_chunk = T // num_windows

    # Split tensors along T dimension into num_windows parts: [B, T, D] -> [B, num_windows, T/num_windows, D]
    ts_split = ts.reshape(B, num_windows, T_chunk, D)
    values_split = values.reshape(B, num_windows, T_chunk, D)
    mask_split = mask.reshape(B, num_windows, T_chunk, D)
    fine_grid_times_split = fine_grid_times.reshape(B, num_windows, T_chunk, D)

    # Merge batch and split dimension: [B, num_windows, T/num_windows, D] -> [B*num_windows, T/num_windows, D]
    ts_merged = ts_split.reshape(B * num_windows, T_chunk, D)
    values_merged = values_split.reshape(B * num_windows, T_chunk, D)
    mask_merged = mask_split.reshape(B * num_windows, T_chunk, D)
    fine_grid_times_merged = fine_grid_times_split.reshape(B * num_windows, T_chunk, D)

    # The model fim-imp-pointwise-base expects this format
    batch = {
        # Coarse grid (observations)
        "coarse_grid_noisy_sample_paths": values_merged,  # [B*num_windows, T/num_windows, 1]
        "coarse_grid_grid": ts_merged,  # [B*num_windows, T/num_windows, 1]
        "coarse_grid_observation_mask": mask_merged,  # [B*num_windows, T/num_windows, 1]
        # Fine grid (evaluation points)
        "fine_grid_grid": fine_grid_times_merged,  # [B*num_windows, T/num_windows, 1]
        "fine_grid_sample_paths": values_merged,  # [B*num_windows, T/num_windows, 1] - target if available
    }

    return batch


def visualize_prediction(ts, ys_true, ys_noisy, mask, eval_times, prediction):
    plt.plot(ts[~mask], ys_noisy[~mask], label="Observed Values", c="grey", zorder=1)
    plt.plot(ts, ys_true, label="True Values", linestyle="dashed", c="black", zorder=5)
    plt.plot(eval_times, prediction, label="Predicted Values", c="#0072B2", zorder=10)
    plt.xlabel("t")
    plt.ylabel("x")
    plt.legend()
    plt.show()


def plot_trajectories(eval_times, obs_times, pred_x, obs_x, pred_v, obs_v, title_suffix=""):
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    for i in range(pred_x.shape[0]):
        ax1.plot(obs_times[i].flatten(), obs_x[i].flatten(), c="black", label="Observed" if i == 0 else None, zorder=1, linestyle="dashed")
        ax1.plot(eval_times[i].flatten(), pred_x[i].flatten(), c="#0072B2", label="Estimated" if i == 0 else None, zorder=5)

    ax1.set_title(f"Position {title_suffix}".strip())
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("x")
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    for i in range(pred_v.shape[0]):
        ax2.plot(obs_times[i].flatten(), obs_v[i].flatten(), c="black", label="True" if i == 0 else None, zorder=1, linestyle="dashed")
        ax2.plot(eval_times[i].flatten(), pred_v[i].flatten(), c="#0072B2", label="Estimated" if i == 0 else None, zorder=5)

    ax2.set_title(f"Velocity {title_suffix}".strip())
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("x'")
    ax2.legend()

    plt.show()
