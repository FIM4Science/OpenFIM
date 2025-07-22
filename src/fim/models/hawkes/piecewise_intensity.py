from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


class PiecewiseHawkesIntensity(torch.nn.Module):
    """A piece-wise defined multivariate Hawkes intensity function.

    For each event in every inference path we store the parameters
    (\\mu_i, \alpha_i, \beta_i) that define the local intensity dynamics for
    all possible target marks **after** that event.  Between two consecutive
    events the intensity is assumed to follow the functional form

        \\lambda_i(t) = Softplus(\\mu_i + (\alpha_i - \\mu_i) \\exp(-\beta_i (t - t_i)))

    where ``t_i`` is the timestamp of the *latest* event before *t*.

    Parameters are expected to be already *positive* (e.g. after a Softplus).
    The tensors have the following shapes

        event_times: [B, P, L]
        mu, alpha, beta: [B, M, P, L]

    with
        B – batch size
        P – number of inference paths per sample
        L – number of (historical) events per path
        M – number of marks
    """

    def __init__(
        self,
        event_times: Tensor,
        mu: Tensor,
        alpha: Tensor,
        beta: Tensor,
    ) -> None:
        super().__init__()

        # Store event_times as buffer (not trainable); parameters tensors keep their
        # gradient information to allow back-prop through the intensity evaluation.
        self.register_buffer("event_times", event_times)  # [B, P, L]
        self.mu = mu  # [B, M, P, L]
        self.alpha = alpha  # [B, M, P, L]
        self.beta = beta  # [B, M, P, L]

    def evaluate(self, query_times: Tensor) -> Tensor:
        r"""Evaluate \lambda at ``query_times``.

        query_times must have shape [B, P, L_eval].  The returned tensor has
        shape [B, M, P, L_eval].
        """
        device = query_times.device
        B, P, L_eval = query_times.shape
        _, M, _, L = self.mu.shape

        # Reshape and broadcast to compare each query time with all event times
        # per path: [B, P, L_eval, 1] vs [B, P, 1, L]
        past_mask = self.event_times.unsqueeze(2) < query_times.unsqueeze(3)  # [B,P,L_eval,L]

        # Build indices tensor 0..L-1 for max operation
        idx = torch.arange(L, device=device, dtype=torch.long).view(1, 1, 1, L)
        idx = idx.expand(B, P, L_eval, L)

        # Replace *future* events by -1 so that max selects the last *past* event
        idx_masked = torch.where(past_mask, idx, torch.full_like(idx, -1))
        last_idx = idx_masked.max(dim=3).values  # [B, P, L_eval], -1 if no past event

        # Clamp to at least 0 so that gather works; the corresponding delta_t
        # will be set to 0 and intensity will reduce to Softplus(mu).
        last_idx_clamped = last_idx.clamp(min=0)

        # Gather parameters of the last event
        # Expand indices to [B, M, P, L_eval] for gathering
        gather_idx = last_idx_clamped.unsqueeze(1).expand(-1, M, -1, -1)  # [B,M,P,L_eval]

        mu_last = torch.gather(self.mu, dim=3, index=gather_idx)
        alpha_last = torch.gather(self.alpha, dim=3, index=gather_idx)
        beta_last = torch.gather(self.beta, dim=3, index=gather_idx)

        # Gather event times of the last event: [B,P,L_eval]
        t_last = torch.gather(self.event_times, dim=2, index=last_idx_clamped)
        # If no past event exists (last_idx == -1) we fallback to t_last = 0 so that
        # the intensity reduces to Softplus(mu) with delta_t = query_times.
        t_last = torch.where(last_idx.eq(-1), torch.zeros_like(t_last), t_last)
        delta_t = query_times - t_last  # [B,P,L_eval]
        delta_t = delta_t.unsqueeze(1)  # -> [B,1,P,L_eval]

        # Piece-wise intensity formula
        exponent = torch.exp(-beta_last * delta_t)
        base = mu_last + (alpha_last - mu_last) * exponent
        intensity = F.softplus(base)

        return intensity

    # Alias ``forward`` to ``evaluate`` so that the module can be called like a
    # regular function.
    def forward(self, query_times: Tensor) -> Tensor:  # type: ignore[override]
        return self.evaluate(query_times)

    def integral(self, t_end: Tensor, t_start: Tensor | None = None, num_samples: int = 100) -> Tensor:
        r"""Estimate the integral of \lambda from ``t_start`` to ``t_end`` via Monte-Carlo.

        A simple Monte-Carlo estimator is used:
            \int_{t_start}^{t_end} \lambda(t) dt \approx (t_end - t_start) * MEAN(\lambda(t_samples))
        where t_samples are drawn uniformly from [t_start, t_end].

        This method supports broadcasting for ``t_start`` and ``t_end``. For example,
        to compute the integrated intensity \int_0^t \lambda(s) ds for multiple t,
        pass ``t_end`` with shape [B, P, L_eval] and ``t_start=None``.

        Args:
            t_end (Tensor): The end of the integration interval(s).
                Can be e.g. [B, P] or [B, P, L_eval].
            t_start (Tensor | None): The start of the integration interval(s).
                If None, defaults to 0. Must be broadcastable to ``t_end.shape``.
            num_samples (int): The number of samples for the Monte-Carlo estimation.

        Returns:
            Tensor: The estimated integral for each mark. Shape is [B, M, *t_end.shape[1:]].
        """
        device = t_end.device
        if t_start is None:
            t_start = torch.zeros_like(t_end)

        # We will add a sample dimension at the end
        # Shape: [*t_end.shape, num_samples]
        random_samples = torch.rand(*t_end.shape, num_samples, device=device)

        # Scale samples to be in [t_start, t_end]
        # t_start and t_end are broadcastable.
        interval_len = t_end - t_start
        # Add a dimension to t_start and interval_len for broadcasting with random_samples
        t_samples = t_start.unsqueeze(-1) + random_samples * interval_len.unsqueeze(-1)

        # To call evaluate, we need to flatten the evaluation and sample dimensions
        # Original shape: [B, P, (L_eval), num_samples]
        # Target shape for evaluate: [B, P, L_eval * num_samples]
        B, P, *rest = t_samples.shape
        num_total_samples = t_samples.shape[2:].numel()
        t_samples_flat = t_samples.reshape(B, P, num_total_samples)

        # Evaluate intensity at the sampled times
        intensity_at_samples_flat = self.evaluate(t_samples_flat)  # [B, M, P, num_total_samples]

        # Reshape back to include the original evaluation and sample dimensions
        # Target shape: [B, M, P, (L_eval), num_samples]
        _, M, _, _ = self.mu.shape
        intensity_at_samples = intensity_at_samples_flat.reshape(B, M, P, *t_end.shape[2:], num_samples)

        # Compute the mean intensity over the samples (the last dimension)
        mean_intensity = intensity_at_samples.mean(dim=-1)  # [B, M, P, (L_eval)]

        # Multiply by interval length to get the integral estimate
        # interval_len has shape [B, P, (L_eval)], needs unsqueezing at dim 1 for marks
        integral_estimate = mean_intensity * interval_len.unsqueeze(1)

        return integral_estimate
