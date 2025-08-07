import logging

import torch
import torch.nn as nn

from ...trainers.utils import get_accel_type
from ...utils.logging import RankLoggerAdapter


class EventSampler(nn.Module):
    """Event Sequence Sampler based on thinning algorithm, which corresponds to Algorithm 2 of
    The Neural Hawkes Process: A Neurally Self-Modulating Multivariate Point Process,
    https://arxiv.org/abs/1612.09328.

    The implementation uses code from https://github.com/yangalan123/anhp-andtt/blob/master/anhp/esm/thinning.py.
    """

    def __init__(
        self,
        num_sample: int = 1,
        num_exp: int = 500,
        over_sample_rate: float = 5,
        num_samples_boundary: int = 5,
        dtime_max: float = 5,
        patience_counter: int = 5,
        device=None,
    ):
        """Initialize the event sampler.

        Args:
            num_sample (int): number of sampled next event times via thinning algo for computing predictions.
            num_exp (int): number of i.i.d. Exp(intensity_bound) draws at one time in thinning algorithm
            over_sample_rate (float): multiplier for the intensity up bound.
            num_samples_boundary (int): number of sampled event times to compute the boundary of the intensity.
            dtime_max (float): max value of delta times in sampling
            patience_counter (int): the maximum iteration used in adaptive thinning.
            device (torch.device): torch device index to select.
        """
        super(EventSampler, self).__init__()
        self.num_sample = num_sample
        self.num_exp = num_exp
        self.over_sample_rate = over_sample_rate
        self.num_samples_boundary = num_samples_boundary
        self.dtime_max = dtime_max
        self.patience_counter = patience_counter
        self.device = device if device is not None else get_accel_type()
        self.logger = RankLoggerAdapter(logging.getLogger(self.__class__.__name__))

    def compute_intensity_upper_bound(self, time_seq, time_delta_seq, event_seq, intensity_fn, compute_last_step_only):
        """Compute the upper bound of intensity at each event timestamp.

        Args:
            time_delta_seq (tensor): [batch_size, seq_len], time delta seqs.
            intensity_fn (fn): a function that computes the intensity.
            compute_last_step_only (bool): wheter to compute the last time step pnly.

        Returns:
            tensor: [batch_size, seq_len]
        """
        time_deltas_for_bound = torch.linspace(start=1e-5, end=self.dtime_max, steps=self.num_samples_boundary, device=self.device)

        # Reshape for broadcasting.
        # Shape: [1, 1, num_samples_boundary]
        time_deltas_for_bound = time_deltas_for_bound.view(1, 1, -1)

        # We need to compute absolute times to query the intensity function.
        # `time_seq` contains the absolute time of each event in the history.
        # Shape of `time_seq`: [batch_size, seq_len]
        # Shape of `query_times`: [batch_size, seq_len, num_samples_boundary]
        query_times = time_seq.unsqueeze(-1) + time_deltas_for_bound

        intensities_for_bound = intensity_fn(query_times, time_seq)

        # `intensities_for_bound` is expected to have shape [B, L, T, M] where L is
        # seq_len, T is num_samples_boundary and M is num_marks.
        # Sum over marks (M), find max over sampled times (T).
        bounds = intensities_for_bound.sum(dim=-1).max(dim=-1)[0] * self.over_sample_rate

        return bounds

    def sample_exp_distribution(self, sample_rate: torch.Tensor) -> torch.Tensor:
        """
        Draw samples from an exponential distribution using the inversion method.

        Args:
            sample_rate (torch.Tensor): The rate parameter (lambda) of the exponential distribution.
                                      Can be of shape [batch_size] or [batch_size, seq_len].

        Returns:
            torch.Tensor: Samples from the exponential distribution.
        """
        # Ensure sample_rate is at least 2D for consistent processing
        if sample_rate.dim() == 1:
            sample_rate = sample_rate.unsqueeze(-1)

        batch_size, seq_len = sample_rate.size()

        # Generate uniform random numbers
        uniform_samples = torch.rand(batch_size, seq_len, self.num_exp, device=sample_rate.device)

        # Inverse transform sampling: -log(U) / lambda
        # Add a small epsilon to prevent log(0)
        exp_samples = -torch.log(uniform_samples + 1e-9) / (sample_rate.unsqueeze(-1) + 1e-9)

        return exp_samples

    def sample_uniform_distribution(self, intensity_upper_bound):
        """Sample an uniform distribution

        Args:
            intensity_upper_bound (tensor): upper bound intensity computed in the previous step.

        Returns:
            tensor: [batch_size, seq_len, num_sample, num_exp]
        """
        batch_size, seq_len = intensity_upper_bound.size()

        unif_numbers = torch.empty(size=[batch_size, seq_len, self.num_sample, self.num_exp], dtype=torch.float32, device=self.device)
        unif_numbers.uniform_(0.0, 1.0)

        return unif_numbers

    def sample_accept(self, unif_numbers, sample_rate, total_intensities, exp_numbers):
        """Do the sample-accept process.

        For the accumulated exp (delta) samples drawn for each event timestamp, find (from left to right) the first
        that makes the criterion < 1 and accept it as the sampled next-event time. If all exp samples are rejected
        (criterion >= 1), then we set the sampled next-event time dtime_max.

        Args:
            unif_numbers (tensor): [batch_size, max_len, num_sample, num_exp], sampled uniform random number.
            sample_rate (tensor): [batch_size, max_len], sample rate (intensity).
            total_intensities (tensor): [batch_size, seq_len, num_sample, num_exp]
            exp_numbers (tensor): [batch_size, seq_len, num_sample, num_exp]: sampled exp numbers (delta in Algorithm 2).

        Returns:
            result (tensor): [batch_size, seq_len, num_sample], sampled next-event times.
        """

        # [batch_size, max_len, num_sample, num_exp]
        criterion = unif_numbers * sample_rate[:, :, None, None] / total_intensities

        # [batch_size, max_len, num_sample, num_exp]
        masked_crit_less_than_1 = torch.where(criterion < 1, 1, 0)

        # [batch_size, max_len, num_sample]
        non_accepted_filter = (1 - masked_crit_less_than_1).all(dim=3)

        # [batch_size, max_len, num_sample]
        first_accepted_indexer = masked_crit_less_than_1.argmax(dim=3)

        # [batch_size, max_len, num_sample,1]
        # indexer must be unsqueezed to 4D to match the number of dimensions of exp_numbers
        result_non_accepted_unfiltered = torch.gather(exp_numbers, 3, first_accepted_indexer.unsqueeze(3))

        # [batch_size, max_len, num_sample,1]
        result = torch.where(non_accepted_filter.unsqueeze(3), torch.tensor(self.dtime_max), result_non_accepted_unfiltered)

        # [batch_size, max_len, num_sample]
        result = result.squeeze(dim=-1)

        return result

    def draw_next_time_one_step(self, time_seq, time_delta_seq, event_seq, intensity_fn, compute_last_step_only=False):
        """Compute next event time based on Thinning algorithm.

        Args:
            time_seq (tensor): [batch_size, seq_len], timestamp seqs.
            time_delta_seq (tensor): [batch_size, seq_len], time delta seqs.
            event_seq (tensor): [batch_size, seq_len], event type seqs.
            dtime_boundary (tensor): [batch_size, seq_len], dtime upper bound.
            intensity_fn (fn): a function to compute the intensity.
            compute_last_step_only (bool, optional): whether to compute last event timestep only. Defaults to False.

        Returns:
            tuple: next event time prediction and weight.
        """
        # If only computing for the last step, slice the history here
        if compute_last_step_only:
            time_seq = time_seq[:, -1:]
            time_delta_seq = time_delta_seq[:, -1:]
            event_seq = event_seq[:, -1:]

        # 1. compute the upper bound of the intensity at each timestamp
        # [batch_size, seq_len] (or [batch_size, 1] if last_step_only)
        intensity_upper_bound = self.compute_intensity_upper_bound(
            time_seq, time_delta_seq, event_seq, intensity_fn, compute_last_step_only
        )

        # 2. draw exp distribution with intensity = intensity_upper_bound
        # we apply fast approximation, i.e., re-use exp sample times for computation
        # [batch_size, seq_len, num_exp]
        exp_numbers = self.sample_exp_distribution(intensity_upper_bound)
        exp_numbers = torch.cumsum(exp_numbers, dim=-1)
        exp_numbers = time_seq[:, :, None] + exp_numbers
        # 3. compute intensity at sampled times from exp distribution
        # [batch_size, seq_len, num_exp, event_num]
        intensities_at_sampled_times = intensity_fn(exp_numbers, time_seq)

        # [batch_size, seq_len, num_exp]
        total_intensities = intensities_at_sampled_times.sum(dim=-1)

        # add one dim of num_sample: re-use the intensity for samples for prediction
        # [batch_size, seq_len, num_sample, num_exp]
        total_intensities = torch.tile(total_intensities[:, :, None, :], [1, 1, self.num_sample, 1])

        # [batch_size, seq_len, num_sample, num_exp]
        exp_numbers = torch.tile(exp_numbers[:, :, None, :], [1, 1, self.num_sample, 1])

        # 4. draw uniform distribution
        # [batch_size, seq_len, num_sample, num_exp]
        unif_numbers = self.sample_uniform_distribution(intensity_upper_bound)

        # 5. find out accepted intensities
        # [batch_size, seq_len, num_sample]
        res = self.sample_accept(unif_numbers, intensity_upper_bound, total_intensities, exp_numbers)

        # [batch_size, seq_len, num_sample]
        weights = torch.ones_like(res) / res.shape[2]

        # add a upper bound here in case it explodes, e.g., in ODE models
        return res.clamp(max=1e5), weights
