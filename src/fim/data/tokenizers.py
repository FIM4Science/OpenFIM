from typing import Optional
from datasets import Dataset
import math
from itertools import pairwise
import random


class PatcherDecoderOnlyStyle:
    """
    Split each time series into (overlapping) context windows, then into patches. Store corresponding target (horizon) values.

    Steps:
    1. split time series into context + horizon windows
    2. each context window: split into patches and corresponding target values
    3. select training/testing input: per context window: select patch 1 & 1. prediction; 1-2 & 2. prediction; 1-3 & 3. prediction,...
    4. create masks
        - point level: first patch: mask first r (random) points; last patch: mask points to fill up to patch_len_in; remaining patches: not masked.
            True indicates that the point is masked out. Bases on masking strategy of Das et al. in Decoder-only paper.
        - token level: indicates if corresponding patch is fully masked out
    5. resulting features of a data entry:
        - input: sequence of patches, padded with 0 if necessary, [max_nr_patches_per_context_window, patch_len_in]
        - output: target values subsequent to the last non-padded input value, [patch_len_out]
        - mask_point_level: mask on point level, [max_nr_patches_per_context_window, patch_len_in]
        - mask_token_level: mask on token level, [max_nr_patches_per_context_window]
        - start: start time of the considered patch sequence (note: currently dummy time, needs to be fixed it ever necessary)
    """

    def __init__(
        self, max_context_len: int, patch_len_in: int, patch_len_out: int, overlap_context_windows: Optional[int] = None
    ):
        self.max_context_len = max_context_len
        self.patch_len_in = patch_len_in
        self.patch_len_out = patch_len_out
        self.overlap_context_windows = overlap_context_windows if overlap_context_windows is not None else 0

        self.max_nr_patches_per_context_window = math.ceil(self.max_context_len / self.patch_len_in)

    def split_data(self, data) -> Dataset:
        processed_data = []

        for row in data:
            processed_data.extend(self._process_single_time_series(time_series=row["target"], time_start=row["start"]))

        data = Dataset.from_list(processed_data)

        return data

    def _process_single_time_series(self, time_series, time_start) -> list[dict]:
        """Split time series into context windows and trigger patching function."""
        if len(time_series) <= self.patch_len_out + 1:
            return []

        processed_time_series = []

        # get start indices for each context window
        context_start_indices = list(
            range(
                0,
                len(time_series) - self.max_context_len - self.patch_len_out,
                self.max_context_len - self.overlap_context_windows,
            )
        )

        for context_start_id in context_start_indices:
            processed_time_series.extend(
                self._process_single_context_window(
                    time_series[context_start_id : context_start_id + self.max_context_len + self.patch_len_out],
                    time_start,
                )
            )
        return processed_time_series

    def _split_into_patches(self, context_window) -> tuple[list[list[float]], list[list[float]]]:
        """
        Split a context window into patches of length `patch_len_in` and subsequent `patch_len_out` points as prediction.

        Args:
            context_window (list[float]): The context window to split.

        Returns:
            tuple[list[list[float]], list[list[float]]]: The input patches and the corresponding predictions.
        """
        patch_start_indices = [
            patch_id * self.patch_len_in for patch_id in range(0, self.max_nr_patches_per_context_window)
        ]
        patch_start_indices.append(self.max_context_len)

        patches_in = [context_window[start:end] for start, end in pairwise(patch_start_indices)]
        patches_out = [
            context_window[patch_end : patch_end + self.patch_len_out] for patch_end in patch_start_indices[1:]
        ]

        return patches_in, patches_out

    def _process_single_context_window(self, context_window, time_start) -> list[dict]:
        """Patch a context window, compute masks and return data entries."""
        processed_context_window = []

        patches_in, predictions = self._split_into_patches(context_window)

        for nr_patches in range(1, len(patches_in) + 1):
            cur_patched_context = patches_in[:nr_patches]
            mask_point_level = self._create_mask_point_level(nr_patches, cur_patched_context)
            # pad last patch to full length
            if len(patches_in[nr_patches - 1]) < self.patch_len_in:
                patches_in[nr_patches - 1].extend([0] * (self.patch_len_in - len(patches_in[nr_patches - 1])))
            # pad patch sequence to max_nr_patches_per_context_window
            cur_patched_context += [[0] * self.patch_len_in] * (self.max_nr_patches_per_context_window - nr_patches)
            mask_point_level += [[True] * self.patch_len_in] * (self.max_nr_patches_per_context_window - nr_patches)

            mask_token_level = self._create_mask_token_level(mask_point_level)

            processed_context_window.append(
                {
                    "input": cur_patched_context,
                    "output": predictions[nr_patches - 1],
                    "mask_point_level": mask_point_level,
                    "mask_token_level": mask_token_level,
                    "start": time_start,
                }
            )
        return processed_context_window

    def _create_mask_point_level(self, nr_patches, patches_in) -> list[list[bool]]:
        """
        Create the mask on point level.

        The first r (random number) points of the first patch are masked out & the last values of last patch if it is not of full length.

        Returns:
            list[list[bool]]: The mask on point level.
            datetime.timedelta: The time delta to the start of the first patch (due to masking of first r points in first patch)
        """
        # create first patch mask
        r = random.randint(0, len(patches_in[0]) - 1)
        mask_point_level = [
            [True] * r + [False] * (len(patches_in[0]) - r) + [True] * (self.patch_len_in - len(patches_in[0]))
        ]

        # create patch masks for all patches except the last one
        if nr_patches > 2:
            mask_point_level.extend([[False] * self.patch_len_in] * (nr_patches - 2))

        # append padding mask for last patch if necessary
        if nr_patches > 1:
            mask_point_level.extend(
                [[False] * len(patches_in[-1]) + [True] * (self.patch_len_in - len(patches_in[-1]))]
            )

        return mask_point_level

    def _create_mask_token_level(self, mask_point_level) -> list[bool]:
        """Create the mask on token level."""
        return [all(mask) for mask in mask_point_level]

    def __str__(self):
        return f"""PatcherDecoderOnlyStyle(
max_context_len={self.max_context_len},
overlap_context_windows={self.overlap_context_windows},
patch_len_in={self.patch_len_in},
patch_len_out={self.patch_len_out}
)"""
