"""
scripts/ode/finetune.py
================================
Gradient-based finetuning of FIM-ODE on small reference datasets.

Training objective — trajectory reconstruction
-----------------------------------------------
Given a short context trajectory x(t_0), x(t_1), …, x(t_T):

  1. Encode ALL observations once: (D_enc, feature_mask, concept) = trajectory_encoding(ctx)
  2. For every consecutive pair (x_i, x_{i+1}) with step Δt_i:
         y_pred = x_i + Δt_i · f_θ(x_i)          [Euler]
         loss  += L1(y_pred, x_{i+1})              [normalised state space]
  3. Back-propagate; Adam updates θ.

The model predicts x_{i+1} by integrating its own inferred vector field from
x_i.  Matching the observed x_{i+1} forces the vector field to be consistent
with the data, improving forecasting downstream.  All pairs are integrated in
parallel (one batch), so the cost scales as O(T) model evaluations per step.
The context encoding is shared and reused for all pairs.

Supported tasks
---------------
  vdp-u     Van der Pol uniform
  vdp-nu    Van der Pol non-uniform
  fhn       FitzHugh-Nagumo (segmented, missing quadrant)
  mocap     MoCap (5-dim PCA)

Usage
-----
    python scripts/ode/finetune.py --task vdp-u --epochs 200 --lr 3e-5

    # Monitor in another terminal:
    tensorboard --logdir results/ode/logs
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.integrate import solve_ivp
from torch.optim import Adam

from fim.models.ode import FIMODE, load_fim_ode_hf, load_fim_ode_local


# ── project root on sys.path ──────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


# ── optional TensorBoard ──────────────────────────────────────────────────────
try:
    from torch.utils.tensorboard import SummaryWriter

    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False
    SummaryWriter = None


# =============================================================================
# 1. Tensor preparation helpers
# =============================================================================


def _t(arr: np.ndarray, device: str, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(arr, dtype=dtype, device=device)


def prepare_context_tensors(
    traj: np.ndarray,  # (n_paths, T, D)  or  (T, D) for single path
    times: np.ndarray,  # (T,) shared across paths
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build (traj_t, time_t, mask_t) for model_forward / trajectory_encoding.

    Returns
    -------
    traj_t : (1, n_paths, T, D)
    time_t : (1, n_paths, T, 1)
    mask_t : (1, n_paths, T, 1)
    """
    if traj.ndim == 2:
        traj = traj[np.newaxis]
    n_paths, T, D = traj.shape

    traj_t = _t(traj, device).unsqueeze(0)
    time_t = _t(times, device).unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(1, n_paths, T, 1).contiguous()
    mask_t = torch.ones(1, n_paths, T, 1, dtype=torch.bool, device=device)
    return traj_t, time_t, mask_t


def prepare_context_tensors_segmented(
    traj: np.ndarray,  # (n_segs, max_len, D)
    times: np.ndarray,  # (n_segs, max_len)
    mask: np.ndarray,  # (n_segs, max_len) bool
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build context tensors for segmented context (e.g. FHN missing quadrant).

    Returns
    -------
    traj_t : (1, n_segs, max_len, D)
    time_t : (1, n_segs, max_len, 1)
    mask_t : (1, n_segs, max_len, 1)
    """
    traj_t = _t(traj, device).unsqueeze(0)
    time_t = _t(times, device).unsqueeze(0).unsqueeze(-1)
    mask_t = _t(mask, device, dtype=torch.bool).unsqueeze(0).unsqueeze(-1)
    return traj_t, time_t, mask_t


# =============================================================================
# 2. Core integration helpers (mirroring TrainIntegrator, no config needed)
# =============================================================================


def _euler(f, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """One Euler step: y + h · f(y)."""
    return y + h * f(y)


def _improved_euler(f, h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """One improved-Euler step: y + h · f(y + h/2 · f(y))."""
    f0 = f(y)
    y_mid = y + (h / 2) * f0
    return y + h * f(y_mid)


def _make_f(
    model: FIMODE,
    D_enc,
    feature_mask: torch.Tensor,
    concept,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Build the ODE right-hand side from the cached context encoding.

    The returned function f maps
        y  : (1, N_locs, D_pad)  →  drift : (1, N_locs, D_pad)
    in the model's normalized state space.
    Gradients flow through function_decoding into both the trajectory encoder
    (via D_enc) and the functional decoder.
    """

    def f(y: torch.Tensor) -> torch.Tensor:
        result = model.function_decoding(y, feature_mask, D_enc, copy.deepcopy(concept))
        return result.predictions.drift

    return f


def _collect_consecutive_pairs(
    traj_norm_padded: torch.Tensor,  # (1, n_segs, max_len, D_pad)
    time_norm: torch.Tensor,  # (1, n_segs, max_len, 1)
    mask: torch.Tensor,  # (1, n_segs, max_len, 1)  bool
    D_orig: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract all valid consecutive observation pairs across all segments.

    Returns
    -------
    y0_batch     : (1, N_pairs, D_pad)  — normalised start points
    y_tgt_batch  : (1, N_pairs, D_pad)  — normalised end points
    h_batch      : (1, N_pairs, 1)      — Δt in normalised time
    """
    n_segs = traj_norm_padded.shape[1]
    max_len = traj_norm_padded.shape[2]

    y0_list, yt_list, h_list = [], [], []

    for s in range(n_segs):
        m = mask[0, s, :, 0].bool()  # (max_len,)
        # valid consecutive pairs: both endpoints must be observed
        for i in range(max_len - 1):
            if m[i] and m[i + 1]:
                y0_list.append(traj_norm_padded[0, s, i])  # (D_pad,)
                yt_list.append(traj_norm_padded[0, s, i + 1])  # (D_pad,)
                dt = time_norm[0, s, i + 1, 0] - time_norm[0, s, i, 0]
                h_list.append(dt)

    if not y0_list:
        raise ValueError("No valid consecutive pairs found in context.")

    y0 = torch.stack(y0_list, dim=0).unsqueeze(0)  # (1, N_pairs, D_pad)
    yt = torch.stack(yt_list, dim=0).unsqueeze(0)  # (1, N_pairs, D_pad)
    h = torch.stack(h_list).view(1, -1, 1)  # (1, N_pairs, 1)

    return y0, yt, h


# =============================================================================
# 3. Finetuner
# =============================================================================


class Finetuner:
    """
    Lightweight trajectory-reconstruction finetuner for FIM-ODE.

    For each consecutive observation pair (x_i, x_{i+1}) in the context:

        1. Encode ALL context trajectories once → shared vector field f_θ
        2. Integrate one step:  x̂_{i+1} = x_i + Δt · f_θ(x_i)
        3. Loss:  L1( x̂_{i+1},  x_{i+1} )   in normalised state space

    All pairs are integrated **in parallel** (batched), so encoding happens
    once per gradient step.  Adam momentum smooths out any observation noise.

    Parameters
    ----------
    model          : FIMODE already on the correct device
    device         : "cpu" | "cuda" | "mps"
    lr             : Adam learning rate (default 3e-5)
    weight_decay   : Adam L2 regularisation (default 1e-4)
    freeze_encoder : if True, only the functional decoder and location_proj
                     are updated (safer, less catastrophic forgetting)
    integrator     : "euler" or "improved_euler"
    """

    def __init__(
        self,
        model: FIMODE,
        device: str = "cpu",
        lr: float = 1e-5,
        weight_decay: float = 1e-4,
        freeze_encoder: bool = False,
        integrator: str = "euler",
        n_inner_steps: int = 5,
    ):
        self.model = model
        self.device = device
        self.n_inner_steps = n_inner_steps
        self._integrate_step = _euler if integrator == "euler" else _improved_euler

        if freeze_encoder:
            params = list(model.functional_decoder.parameters()) + list(model.location_proj.parameters())
        else:
            params = list(model.parameters())

        self.optimizer = Adam(params, lr=lr, weight_decay=weight_decay)

    # ------------------------------------------------------------------
    # One gradient step — single contiguous trajectory
    # ------------------------------------------------------------------

    def step(
        self,
        traj_np: np.ndarray,  # (T, D)  or  (n_paths, T, D)
        times_np: np.ndarray,  # (T,)  shared across paths
    ) -> float:
        """
        One gradient step on a single (or multi-path) contiguous context.

        All consecutive observation pairs are integrated in parallel.
        """
        if traj_np.ndim == 2:
            traj_np = traj_np[np.newaxis]  # (1, T, D)
        n_paths, T, D = traj_np.shape

        traj_t, time_t, mask_t = prepare_context_tensors(traj_np, times_np, self.device)

        self.model.train()
        self.optimizer.zero_grad()

        # ── Encode context (once) ─────────────────────────────────────
        D_enc, feature_mask, concept = self.model.trajectory_encoding(traj_t, time_t, mask_t)

        # ── Normalise trajectory & times (same stats as encoding) ──────
        # _prepare_input_features pads THEN normalises, so we must do the same:
        # pad first → norm stats were computed on padded D
        x_padded = self.model.pad_if_necessary(traj_t)  # (1, n_paths, T, D_pad)
        x_norm = self.model.spatial_norm.normalization_map(x_padded, concept._states_norm_stats)  # (1, n_paths, T, D_pad)

        t_norm = self.model.temporal_norm.normalization_map(time_t, concept._times_norm_stats)  # (1, n_paths, T, 1)

        # ── Collect all consecutive pairs ─────────────────────────────
        # mask: all ones for contiguous trajectory
        full_mask = torch.ones(1, n_paths, T, 1, dtype=torch.bool, device=self.device)
        y0_batch, y_tgt_batch, h_batch = _collect_consecutive_pairs(x_norm, t_norm, full_mask, D, self.device)

        # ── Integrate one step per pair (in parallel) ─────────────────
        f = _make_f(self.model, D_enc, feature_mask, concept)
        y_pred_batch = self._integrate_step(f, h_batch, y0_batch)  # (1, N_pairs, D_pad)

        # ── L1 loss in normalised space, trimmed to original D dims ────
        loss = F.l1_loss(y_pred_batch[0, :, :D], y_tgt_batch[0, :, :D])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    # ------------------------------------------------------------------
    # One gradient step — full-trajectory single shooting
    # ------------------------------------------------------------------

    def step_full_trajectory(
        self,
        traj_np: np.ndarray,  # (T, D)
        times_np: np.ndarray,  # (T,)
    ) -> float:
        """
        Single-shooting trajectory reconstruction over the full training window.

        Encodes the full context once, then integrates from x_0 through all T
        observation times, matching each observed x_i along the way.

            y_0 = x_0  (first observation, normalised)
            for i = 0 … T-2:
                for _ in range(n_inner_steps):
                    y ← y + (h_i / n_inner) · f(y)     [Euler]
            loss = mean L1( y_i_pred , x_i_obs )  for i = 1 … T-1

        Integrating through the full window means gradient signal accounts for
        error accumulation over the whole trajectory.  With n_inner_steps > 1
        the effective step is Δt/n_inner, reducing Euler discretisation error.
        Adam momentum averages the per-observation residuals, implicitly
        weighting each target by 1/noise-variance.

        Parameters
        ----------
        traj_np  : (T, D)  observed (noisy) training trajectory
        times_np : (T,)    corresponding time stamps
        n_inner_steps : set via constructor; sub-steps between observations
        """
        if traj_np.ndim != 2:
            raise ValueError("step_full_trajectory expects a single (T, D) trajectory.")
        T, D = traj_np.shape

        traj_t, time_t, mask_t = prepare_context_tensors(
            traj_np[np.newaxis],
            times_np,
            self.device,  # (1, 1, T, D)
        )

        self.model.train()
        self.optimizer.zero_grad()

        # ── Encode context (once) ─────────────────────────────────────
        D_enc, feature_mask, concept = self.model.trajectory_encoding(traj_t, time_t, mask_t)

        # ── Normalise trajectory & times ──────────────────────────────
        x_padded = self.model.pad_if_necessary(traj_t)  # (1, 1, T, D_pad)
        x_norm = self.model.spatial_norm.normalization_map(x_padded, concept._states_norm_stats)  # (1, 1, T, D_pad)
        t_norm = self.model.temporal_norm.normalization_map(time_t, concept._times_norm_stats)  # (1, 1, T, 1)

        # ── Build ODE right-hand side ─────────────────────────────────
        f = _make_f(self.model, D_enc, feature_mask, concept)

        # ── Single-shoot from x_0 through all T observations ─────────
        y = x_norm[0, 0, 0].unsqueeze(0).unsqueeze(0)  # (1, 1, D_pad)
        preds: List[torch.Tensor] = []

        for i in range(T - 1):
            dt = t_norm[0, 0, i + 1, 0] - t_norm[0, 0, i, 0]
            h = (dt / self.n_inner_steps).view(1, 1, 1)  # (1, 1, 1)
            for _ in range(self.n_inner_steps):
                y = self._integrate_step(f, h, y)  # (1, 1, D_pad)
            preds.append(y[0, 0, :D])  # (D,)

        preds_t = torch.stack(preds, dim=0)  # (T-1, D)
        targets = x_norm[0, 0, 1:, :D]  # (T-1, D)

        loss = F.l1_loss(preds_t, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    # ------------------------------------------------------------------
    # One gradient step — segmented context (e.g. FHN)
    # ------------------------------------------------------------------

    def step_segmented(
        self,
        traj_np: np.ndarray,  # (n_segs, max_len, D)
        times_np: np.ndarray,  # (n_segs, max_len)
        mask_np: np.ndarray,  # (n_segs, max_len) bool
    ) -> float:
        """
        Gradient step for gappy / segmented context.

        Valid consecutive pairs are extracted per segment; all are integrated
        in parallel with a single shared context encoding.
        """
        n_segs, max_len, D = traj_np.shape

        traj_t, time_t, mask_t = prepare_context_tensors_segmented(traj_np, times_np, mask_np, self.device)

        self.model.train()
        self.optimizer.zero_grad()

        # ── Encode full segmented context ─────────────────────────────
        D_enc, feature_mask, concept = self.model.trajectory_encoding(traj_t, time_t, mask_t)

        # ── Normalise using shared stats (pad first, then normalise) ──
        x_padded = self.model.pad_if_necessary(traj_t)
        x_norm = self.model.spatial_norm.normalization_map(x_padded, concept._states_norm_stats)

        t_norm = self.model.temporal_norm.normalization_map(time_t, concept._times_norm_stats)

        # ── Collect valid consecutive pairs across all segments ────────
        y0_batch, y_tgt_batch, h_batch = _collect_consecutive_pairs(x_norm, t_norm, mask_t, D, self.device)

        # ── Integrate & loss ──────────────────────────────────────────
        f = _make_f(self.model, D_enc, feature_mask, concept)
        y_pred_batch = self._integrate_step(f, h_batch, y0_batch)

        loss = F.l1_loss(y_pred_batch[0, :, :D], y_tgt_batch[0, :, :D])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    # ------------------------------------------------------------------
    # One gradient step — multiple paths, full-trajectory single shooting
    # ------------------------------------------------------------------

    def step_full_trajectory_batch(
        self,
        traj_np: np.ndarray,  # (n_paths, T, D)
        times_np: np.ndarray,  # (T,) shared across paths
    ) -> float:
        """
        Single-shooting trajectory reconstruction over all n_paths simultaneously.

        All n_paths are encoded together as a shared context. Each path is then
        integrated forward from its own x_0 through all T observations using the
        same ODE RHS (shared context encoding).  Loss = mean L1 over all paths
        and all time steps in the normalised state space.

        This is the natural mode for MoCap where we have multiple training
        trajectories from the same subject.
        """
        if traj_np.ndim != 3:
            raise ValueError("step_full_trajectory_batch expects (n_paths, T, D).")
        n_paths, T, D = traj_np.shape

        traj_t, time_t, mask_t = prepare_context_tensors(traj_np, times_np, self.device)
        # traj_t : (1, n_paths, T, D),  time_t : (1, n_paths, T, 1)

        self.model.train()
        self.optimizer.zero_grad()

        # ── Encode all paths as shared context ────────────────────────
        D_enc, feature_mask, concept = self.model.trajectory_encoding(traj_t, time_t, mask_t)

        # ── Normalise ─────────────────────────────────────────────────
        x_padded = self.model.pad_if_necessary(traj_t)  # (1, n_paths, T, D_pad)
        x_norm = self.model.spatial_norm.normalization_map(x_padded, concept._states_norm_stats)
        t_norm = self.model.temporal_norm.normalization_map(time_t, concept._times_norm_stats)

        # ── Build ODE RHS ─────────────────────────────────────────────
        f = _make_f(self.model, D_enc, feature_mask, concept)

        # ── Initialise all paths at their first observation ───────────
        y = x_norm[0, :, 0, :].unsqueeze(0)  # (1, n_paths, D_pad)
        preds: List[torch.Tensor] = []

        for i in range(T - 1):
            # All paths share the same time grid → same dt
            dt = t_norm[0, 0, i + 1, 0] - t_norm[0, 0, i, 0]
            h = (dt / self.n_inner_steps).view(1, 1, 1)
            for _ in range(self.n_inner_steps):
                y = self._integrate_step(f, h, y)  # (1, n_paths, D_pad)
            preds.append(y[0, :, :D])  # (n_paths, D)

        preds_t = torch.stack(preds, dim=0)  # (T-1, n_paths, D)
        targets = x_norm[0, :, 1:, :D].permute(1, 0, 2)  # (T-1, n_paths, D)

        loss = F.l1_loss(preds_t, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def finetune(
        self,
        traj_np: np.ndarray,
        times_np: np.ndarray,
        n_epochs: int,
        mask_np: Optional[np.ndarray] = None,
        mode: str = "full_trajectory",
        eval_fn: Optional[Callable[[FIMODE], float]] = None,
        eval_every: int = 20,
        writer=None,
        ckpt_dir: Optional[Path] = None,
        ckpt_every: int = 50,
        log_every: int = 10,
        task_label: str = "task",
    ) -> Dict:
        """
        Full finetuning loop.

        Parameters
        ----------
        traj_np    : (T, D) for single-context tasks; (n_segs, max_len, D)
                     when mask_np is provided
        times_np   : (T,) or (n_segs, max_len)
        n_epochs   : number of gradient steps
        mask_np    : boolean mask for segmented context (FHN); implies segmented mode
        mode       : "full_trajectory" — single-shooting over whole training window
                     "consecutive"     — independent 1-step pairs in parallel
                     (segmented context always uses consecutive-pairs logic)
        eval_fn    : callable model → float; returns MSE on the test window
        eval_every : evaluate every this many epochs
        writer     : TensorBoard SummaryWriter (or None)
        ckpt_dir   : directory for checkpoints; None = no saving
        ckpt_every : save checkpoint every this many epochs
        log_every  : print interval
        task_label : prefix for TensorBoard scalars and print statements

        Returns
        -------
        history : dict with keys 'train_loss', 'eval_mse', 'eval_epochs'
        """
        segmented = mask_np is not None
        history: Dict[str, List] = {"train_loss": [], "eval_mse": [], "eval_epochs": []}

        best_mse = float("inf")
        best_ckpt_path = None

        for epoch in range(n_epochs):
            if segmented:
                loss = self.step_segmented(traj_np, times_np, mask_np)
            elif mode == "batch_trajectory":
                loss = self.step_full_trajectory_batch(traj_np, times_np)
            elif mode == "full_trajectory":
                loss = self.step_full_trajectory(traj_np, times_np)
            else:
                loss = self.step(traj_np, times_np)

            history["train_loss"].append(loss)
            if writer:
                writer.add_scalar(f"{task_label}/train_recon_loss", loss, epoch)

            if eval_fn is not None and (epoch + 1) % eval_every == 0:
                self.model.eval()
                with torch.no_grad():
                    mse = eval_fn(self.model)
                self.model.train()
                history["eval_mse"].append(mse)
                history["eval_epochs"].append(epoch + 1)
                if writer:
                    writer.add_scalar(f"{task_label}/eval_mse", mse, epoch)

                improved = mse < best_mse
                if improved:
                    best_mse = mse
                    if ckpt_dir is not None:
                        best_ckpt_path = self.save_checkpoint(ckpt_dir, epoch + 1, task_label, suffix="best")

                marker = "  ★" if improved else ""
                if (epoch + 1) % log_every == 0 or improved:
                    print(f"[{task_label}] epoch {epoch + 1:>5d}/{n_epochs}  loss={loss:.5e}  eval_mse={mse:.5e}{marker}")
            elif (epoch + 1) % log_every == 0:
                print(f"[{task_label}] epoch {epoch + 1:>5d}/{n_epochs}  loss={loss:.5e}")

            if ckpt_dir is not None and (epoch + 1) % ckpt_every == 0:
                self.save_checkpoint(ckpt_dir, epoch + 1, task_label)

        if ckpt_dir is not None:
            self.save_checkpoint(ckpt_dir, n_epochs, task_label, suffix="final")

        history["best_mse"] = best_mse
        history["best_ckpt_path"] = str(best_ckpt_path) if best_ckpt_path else None

        return history

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        ckpt_dir: Path,
        epoch: int,
        task_label: str = "task",
        suffix: str = "",
    ) -> Path:
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # "best" always overwrites a single file so there is exactly one best
        # checkpoint per run; the epoch is stored inside the dict.
        if suffix == "best":
            tag = f"{task_label}_best"
        else:
            tag = f"{task_label}_epoch{epoch}" + (f"_{suffix}" if suffix else "")
        path = ckpt_dir / f"{tag}.pt"
        torch.save(
            {
                "epoch": epoch,
                "task": task_label,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )
        print(f"  checkpoint → {path}")
        return path

    def load_checkpoint(self, path: Path) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt["epoch"]


# =============================================================================
# 4. Evaluation helpers (integration + MSE)
# =============================================================================


def integrate_from_context(
    model: FIMODE,
    ctx_traj: np.ndarray,  # (n_paths, T_ctx, D)
    ctx_times: np.ndarray,  # (T_ctx,)
    y0: np.ndarray,  # (D,)
    t_eval: np.ndarray,  # (L,)
    device: str = "cpu",
) -> np.ndarray:  # (L, D)
    """Encode context, integrate from y0, return trajectory of shape (L, D)."""
    model.eval()
    traj_t, time_t, mask_t = prepare_context_tensors(ctx_traj, ctx_times, device)

    @torch.no_grad()
    def fim_fn(t: float, y: np.ndarray) -> np.ndarray:
        D = y.shape[0]
        loc = torch.tensor(y, dtype=torch.float32, device=device).view(1, 1, D)
        out = model.model_forward(traj_t, time_t, loc, mask_t)
        return model.get_prediction_for_eval(out).squeeze().cpu().numpy()[:D]

    sol = solve_ivp(
        fim_fn,
        t_span=(float(t_eval[0]), float(t_eval[-1])),
        y0=np.asarray(y0, dtype=float),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-4,
        atol=1e-6,
    )
    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")
    return sol.y.T  # (L, D)


def make_vdp_eval_fn(data: Dict, device: str) -> Callable:
    """Eval on test window — integrates over full window, returns test-window MSE."""
    train_ts = data["train_ts"]
    ctx_traj = data["train_ys"][0][np.newaxis]  # (1, T_train, D)
    x0 = data["x0"]
    test_ts = data["test_ts"]
    test_ys = data["test_ys"]  # (1, T_test, D)
    t_full = np.sort(np.concatenate([train_ts, test_ts]))

    def eval_fn(model: FIMODE) -> float:
        pred = integrate_from_context(model, ctx_traj, train_ts, x0, t_full, device)
        idx_test = np.searchsorted(t_full, test_ts)
        return float(np.mean((pred[idx_test] - test_ys[0]) ** 2))

    return eval_fn


def make_fhn_eval_fn(data: Dict, device: str) -> Callable:
    """Eval at the 12 missing interpolation points — MSE at the missing quadrant.

    Context format: all 38 observed points as a single flat trajectory (no
    segment splitting).  This matches the full_trajectory training mode and
    is the correct format for evaluating the finetuned model.  Note that the
    zero-shot notebook evaluation uses a segmented/padded context instead —
    that preprocessing avoids a spurious large Δy/Δt transition at the gap for
    a model that has not been adapted to this data distribution.
    """
    train_ts = data["train_ts"]
    ctx_traj = data["train_ys"][0][np.newaxis]  # (1, 38, D)
    x0 = data["x0"]
    full_ts = data["full_ts"]
    interp_ts = data["interpolation_ts"]
    interp_ys = data["interpolation_ys"][0]  # (12, D)

    def eval_fn(model: FIMODE) -> float:
        pred = integrate_from_context(model, ctx_traj, train_ts, x0, full_ts, device)
        idx = np.searchsorted(full_ts, interp_ts)
        return float(np.mean((pred[idx] - interp_ys) ** 2))

    return eval_fn


# =============================================================================
# 5. Dataset loaders
# =============================================================================


def load_vdp_uniform(data_dir: Path) -> Dict:
    ref = np.load(data_dir / "vdp_uniform.npz")
    # raw test covers [0,14] with 100 pts; keep last 50 = [7,14]
    return {
        "train_ts": ref["train_ts"],
        "train_ys": ref["train_ys"],  # (1, T_train, D)
        "x0": ref["x0"],
        "test_ts": ref["test_ts"][-50:],
        "test_ys": ref["test_ys"][:, -50:],
        "label": "VDP-uniform",
    }


def load_vdp_nonuniform(data_dir: Path, seed: int = 0) -> Dict:
    ref = np.load(data_dir / "vdp_nonuniform.npz")
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(ref["test_ts"]), 50, replace=False))
    return {
        "train_ts": ref["train_ts"],
        "train_ys": ref["train_ys"],
        "x0": ref["x0"],
        "test_ts": ref["test_ts"][idx],
        "test_ys": ref["test_ys"][:, idx],
        "label": "VDP-nonuniform",
    }


def load_fhn(data_dir: Path) -> Dict:
    """Load FHN interpolation dataset.

    Context : 38 observed points on [0, 5] with a gap at ~[2.75, 3.88].
    Eval    : MSE at the 12 missing (interpolation) points inside the gap.
    """
    ref = np.load(data_dir / "fhn_interpolation.npz")
    return {
        "train_ts": ref["train_ts"],  # (38,)
        "train_ys": ref["train_ys"],  # (1, 38, 2)
        "x0": ref["x0"],  # (2,)
        "full_ts": ref["full_ts"],  # (50,)
        "full_ys": ref["full_ys"],  # (1, 50, 2)
        "interpolation_ts": ref["interpolation_ts"],  # (12,)
        "interpolation_ys": ref["interpolation_ys"],  # (1, 12, 2)
        "interpolation_mask": ref["interpolation_mask"],  # (50,) bool — True = missing
        "label": "FHN",
    }


def _load_mocap_pickle(pkl_path: Path):
    """Load MoCap pickle with the class shims needed for unpickling."""
    sys.path.insert(0, str(_HERE))
    from data_gen_mocap import Data, MocapDataset, Normalize

    import __main__

    __main__.Normalize = Normalize
    __main__.Data = Data
    __main__.MocapDataset = MocapDataset
    import pickle

    with open(pkl_path, "rb") as fh:
        return pickle.load(fh)


N_PCA_DIMS = 3  # FIM-ODE was trained on systems up to dimension 3


def _pca_to_50d(traj_norm_3d: np.ndarray, pca, pca_normalize) -> np.ndarray:
    """
    Convert predicted 3D normalized-PCA trajectory to 50D joint-angle space.

    The un-modelled PCA components (dims 3 and 4) are zeroed out before
    back-projection.  Input may be (T, 3) or (n, T, 3).
    """
    single = traj_norm_3d.ndim == 2
    if single:
        traj_norm_3d = traj_norm_3d[np.newaxis]  # (1, T, 3)
    n, T, _ = traj_norm_3d.shape
    pad = np.zeros((n, T, 5 - N_PCA_DIMS), dtype=traj_norm_3d.dtype)
    norm_5d = np.concatenate([traj_norm_3d, pad], axis=-1)  # (n, T, 5)
    pca_5d = pca_normalize.inverse(norm_5d)  # (n, T, 5)
    out_50d = pca.inverse_transform(pca_5d.reshape(-1, 5)).reshape(n, T, 50)
    return out_50d[0] if single else out_50d


def _gt_to_50d(traj_norm_5d: np.ndarray, pca, pca_normalize) -> np.ndarray:
    """Ground-truth 5D normalized-PCA → 50D, using all 5 components."""
    single = traj_norm_5d.ndim == 2
    if single:
        traj_norm_5d = traj_norm_5d[np.newaxis]
    n, T, _ = traj_norm_5d.shape
    pca_5d = pca_normalize.inverse(traj_norm_5d)
    out_50d = pca.inverse_transform(pca_5d.reshape(-1, 5)).reshape(n, T, 50)
    return out_50d[0] if single else out_50d


def load_mocap(data_dir: Path, subject: str = "09", variant: str = "short") -> Dict:
    """Load MoCap PCA dataset from pickle.

    Parameters
    ----------
    data_dir : path to the mocap base directory (parent of subject_XX directories)
    subject  : "09", "35", or "39"
    variant  : "short" or "long"

    Returns
    -------
    dict with keys:
        trn_ys   : (n_trn, T_trn, 3)  — first 3 PCA dims, normalized (FIM-ODE input)
        trn_ts   : (T_trn,)
        tst_ys   : (n_tst, T_tst, 3)  — first 3 PCA dims, normalized (FIM-ODE input)
        tst_ts   : (T_tst,)
        tst_ys_5d: (n_tst, T_tst, 5)  — full 5-dim normalized (for 50D back-projection)
        pca      : sklearn PCA object
        pca_norm : Normalize object
        label    : e.g. "mocap-09-short"
    """
    pkl_path = data_dir / f"subject_{subject}" / variant / "mocap_dataset.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"MoCap pickle not found: {pkl_path}")

    dataset = _load_mocap_pickle(pkl_path)

    trn_ys_5d = np.asarray(dataset.trn.ys, dtype=np.float32)  # (n_trn, T_trn, 5)
    trn_ts = np.asarray(dataset.trn.ts, dtype=np.float32)
    val_ys_5d = np.asarray(dataset.val.ys, dtype=np.float32)  # (n_val, T_val, 5)
    val_ts = np.asarray(dataset.val.ts, dtype=np.float32)
    tst_ys_5d = np.asarray(dataset.tst.ys, dtype=np.float32)  # (n_tst, T_tst, 5)
    tst_ts = np.asarray(dataset.tst.ts, dtype=np.float32)

    if trn_ts.ndim == 2:
        trn_ts = trn_ts[0]
    if val_ts.ndim == 2:
        val_ts = val_ts[0]
    if tst_ts.ndim == 2:
        tst_ts = tst_ts[0]

    return {
        "trn_ys": trn_ys_5d[:, :, :N_PCA_DIMS],  # (n_trn, T_trn, 3)
        "trn_ts": trn_ts,
        "val_ys": val_ys_5d[:, :, :N_PCA_DIMS],  # (n_val, T_val, 3)
        "val_ys_5d": val_ys_5d,  # (n_val, T_val, 5) for 50D eval
        "val_ts": val_ts,
        "tst_ys": tst_ys_5d[:, :, :N_PCA_DIMS],  # (n_tst, T_tst, 3)
        "tst_ys_5d": tst_ys_5d,  # (n_tst, T_tst, 5) for 50D eval
        "tst_ts": tst_ts,
        "pca": dataset.pca,
        "pca_norm": dataset.pca_normalize,
        "label": f"mocap-{subject}-{variant}",
    }


def make_mocap_eval_fn(data: Dict, device: str) -> Callable:
    """Eval on validation trajectories — used for checkpoint selection during finetuning.

    Encodes all training paths (first 3 PCA dims) as context, integrates each
    validation path from its own x_0, then back-projects to 50D joint-angle space.
    MSE is computed in 50D so it accounts for the variance lost by zeroing dims 3 and 4.

    Returns
    -------
    eval_fn : model → float  (mean 50D MSE over all val trajectories)
    """
    trn_ys = data["trn_ys"]  # (n_trn, T_trn, 3)
    trn_ts = data["trn_ts"]
    val_ys = data["val_ys"]  # (n_val, T_val, 3) — 3D IC for integration
    val_ys_5d = data["val_ys_5d"]  # (n_val, T_val, 5) — GT for 50D eval
    val_ts = data["val_ts"]
    pca = data["pca"]
    pca_norm = data["pca_norm"]

    def eval_fn(model: FIMODE) -> float:
        mses = []
        for i in range(len(val_ys)):
            y0 = val_ys[i, 0].astype(float)
            pred = integrate_from_context(model, trn_ys, trn_ts, y0, val_ts, device)  # (T_val, 3)
            pred_50d = _pca_to_50d(pred, pca, pca_norm)
            true_50d = _gt_to_50d(val_ys_5d[i], pca, pca_norm)
            mses.append(float(np.mean((pred_50d - true_50d) ** 2)))
        return float(np.mean(mses))

    return eval_fn


def make_mocap_test_eval_fn(data: Dict, device: str) -> Callable:
    """Eval on test trajectories — for final reporting only.

    Returns
    -------
    eval_fn : model → float  (mean 50D MSE over all tst trajectories)
    """
    trn_ys = data["trn_ys"]  # (n_trn, T_trn, 3)
    trn_ts = data["trn_ts"]
    tst_ys = data["tst_ys"]  # (n_tst, T_tst, 3)
    tst_ys_5d = data["tst_ys_5d"]  # (n_tst, T_tst, 5)
    tst_ts = data["tst_ts"]
    pca = data["pca"]
    pca_norm = data["pca_norm"]

    def eval_fn(model: FIMODE) -> float:
        mses = []
        for i in range(len(tst_ys)):
            y0 = tst_ys[i, 0].astype(float)
            pred = integrate_from_context(model, trn_ys, trn_ts, y0, tst_ts, device)  # (T_tst, 3)
            pred_50d = _pca_to_50d(pred, pca, pca_norm)
            true_50d = _gt_to_50d(tst_ys_5d[i], pca, pca_norm)
            mses.append(float(np.mean((pred_50d - true_50d) ** 2)))
        return float(np.mean(mses))

    return eval_fn


# =============================================================================
# 6. CLI entry point
# =============================================================================

TASKS = ["vdp-u", "vdp-nu", "fhn", "mocap"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Finetune FIM-ODE via trajectory reconstruction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=None, help="YAML config file; CLI flags override config values.")
    p.add_argument("--task", choices=TASKS, default="vdp-u")
    p.add_argument(
        "--data-dir", default=str(_ROOT / "data" / "ode" / "hedge_gp_odes_data"), help="Directory containing vdp_uniform.npz etc."
    )
    p.add_argument("--ckpt-dir", default=str(_ROOT / "results" / "ode" / "checkpoints" / "finetune"), help="Directory to save checkpoints.")
    p.add_argument("--run-dir", default=str(_ROOT / "results" / "ode" / "logs"), help="TensorBoard logdir root.")
    p.add_argument(
        "--mode",
        choices=["full_trajectory", "batch_trajectory", "consecutive"],
        default="full_trajectory",
        help="full_trajectory: single-shoot from x0 through entire "
        "training window. batch_trajectory: same but over all "
        "n_paths simultaneously (MoCap). consecutive: parallel "
        "1-step pairs.",
    )
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument(
        "--n-inner-steps",
        type=int,
        default=5,
        help="Sub-steps between consecutive observations (Euler sub-division, reduces discretisation error).",
    )
    p.add_argument("--freeze-encoder", action="store_true", help="Only update functional decoder (less forgetting).")
    p.add_argument("--integrator", choices=["euler", "improved_euler"], default="euler", help="ODE integrator for reconstruction.")
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--ckpt-every", type=int, default=100)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--device", default="cpu")
    p.add_argument("--local-ckpt", default=None, help="Path to local checkpoint directory (skips HF download).")
    p.add_argument(
        "--label",
        default=None,
        help="Custom run label used for checkpoint filenames and TensorBoard. "
        "Defaults to the task name (or mocap-{subject}-{variant}). "
        "Use this to distinguish runs with different hyperparameters.",
    )
    p.add_argument("--no-tb", action="store_true", help="Disable TensorBoard.")
    # MoCap-specific
    p.add_argument("--subject", default="09", choices=["09", "35", "39"], help="MoCap subject ID (only used when --task mocap).")
    p.add_argument("--variant", default="short", choices=["short", "long"], help="MoCap dataset variant (only used when --task mocap).")
    p.add_argument(
        "--mocap-dir", default=str(_ROOT / "data" / "mocap"), help="Base directory for MoCap pickle files (parent of subject_XX/)."
    )
    args = p.parse_args()

    # ── Merge YAML config (CLI overrides YAML) ────────────────────────────────
    if args.config is not None:
        import yaml

        with open(args.config) as fh:
            cfg = yaml.safe_load(fh)
        # Only apply YAML values where the CLI still has its default
        defaults = p.parse_args([])  # all-default namespace
        for key, val in cfg.items():
            cli_key = key.replace("-", "_")
            if getattr(args, cli_key, None) == getattr(defaults, cli_key, None):
                setattr(args, cli_key, val)

    return args


def main():
    args = parse_args()
    device = args.device
    data_dir = Path(args.data_dir)
    ckpt_dir = Path(args.ckpt_dir)
    run_dir = Path(args.run_dir)

    # ── Load model ───────────────────────────────────────────────────────────
    print("Loading FIM-ODE model …")
    if args.local_ckpt:
        model = load_fim_ode_local(Path(args.local_ckpt), device)
    else:
        model = load_fim_ode_hf(device)
    print(f"  {sum(p.numel() for p in model.parameters()):,} parameters")

    # ── Build run label (used for checkpoints and TensorBoard) ──────────────
    if args.label:
        base_label = args.label
    elif args.task == "mocap":
        base_label = f"mocap-{args.subject}-{args.variant}"
    else:
        base_label = args.task
    ts_str = time.strftime("%Y%m%d_%H%M%S")
    task_label = f"{base_label}_{ts_str}"

    # ── TensorBoard writer + config snapshot ────────────────────────────────
    log_dir = run_dir / task_label
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save config into the run directory so every run is self-contained
    import shutil

    import yaml

    if args.config is not None:
        shutil.copy(args.config, log_dir / "config.yaml")
    with open(log_dir / "args.yaml", "w") as fh:
        yaml.dump(vars(args), fh, default_flow_style=False, sort_keys=True)

    writer = None
    if not args.no_tb and _TB_AVAILABLE:
        writer = SummaryWriter(str(log_dir))
        print(f"  TensorBoard logdir : {log_dir}")
        print(f"  Monitor with       : tensorboard --logdir {run_dir}")
    elif not _TB_AVAILABLE:
        print("  [warn] torch.utils.tensorboard not available; skipping TB.")

    # ── Load dataset ─────────────────────────────────────────────────────────
    print(f"\nLoading dataset: {args.task} …")
    segmented = False

    if args.task == "vdp-u":
        data = load_vdp_uniform(data_dir)
        traj_np = data["train_ys"][0]  # (T, D)
        times_np = data["train_ts"]
        eval_fn = make_vdp_eval_fn(data, device)
        test_eval_fn = eval_fn

    elif args.task == "vdp-nu":
        data = load_vdp_nonuniform(data_dir)
        traj_np = data["train_ys"][0]
        times_np = data["train_ts"]
        eval_fn = make_vdp_eval_fn(data, device)
        test_eval_fn = eval_fn

    elif args.task == "fhn":
        data = load_fhn(data_dir)
        traj_np = data["train_ys"][0]  # (38, D)
        times_np = data["train_ts"]  # (38,)
        eval_fn = make_fhn_eval_fn(data, device)
        test_eval_fn = eval_fn

    elif args.task == "mocap":
        mocap_dir = Path(args.mocap_dir)
        data = load_mocap(mocap_dir, subject=args.subject, variant=args.variant)
        traj_np = data["trn_ys"]  # (n_trn, T_trn, 3)
        times_np = data["trn_ts"]  # (T_trn,)
        eval_fn = make_mocap_eval_fn(data, device)  # val set  (checkpoint selection)
        test_eval_fn = make_mocap_test_eval_fn(data, device)  # test set (final report)
        # Override mode to batch_trajectory when multiple paths are present
        if args.mode == "full_trajectory" and traj_np.ndim == 3:
            args.mode = "batch_trajectory"

    print(f"  context shape : {traj_np.shape}")
    print(f"  time range    : [{times_np.min():.2f}, {times_np.max():.2f}]")

    # ── Baseline eval ────────────────────────────────────────────────────────
    if eval_fn is not None:
        model.eval()
        with torch.no_grad():
            mse0 = eval_fn(model)
        print(f"\n  baseline MSE (pre-finetune): {mse0:.5e}")
        if writer:
            writer.add_scalar(f"{task_label}/eval_mse", mse0, 0)
        if test_eval_fn is not eval_fn:
            with torch.no_grad():
                mse0_test = test_eval_fn(model)
            print(f"  baseline test MSE (pre-finetune): {mse0_test:.5e}")
            if writer:
                writer.add_scalar(f"{task_label}/eval_test_mse", mse0_test, 0)

    # ── Finetuner ────────────────────────────────────────────────────────────
    finetuner = Finetuner(
        model=model,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        freeze_encoder=args.freeze_encoder,
        integrator=args.integrator,
        n_inner_steps=args.n_inner_steps,
    )

    mode_str = "segmented-consecutive" if segmented else args.mode
    print(f"\nFinetuning on {data['label']} for {args.epochs} epochs")
    print(f"  mode         : {mode_str}")
    print(f"  lr           : {args.lr}")
    print(f"  integrator   : {args.integrator}  (n_inner={args.n_inner_steps})")
    print(f"  freeze_enc   : {args.freeze_encoder}")

    history = finetuner.finetune(
        traj_np=traj_np,
        times_np=times_np,
        n_epochs=args.epochs,
        mask_np=None,
        mode=args.mode,
        eval_fn=eval_fn,
        eval_every=args.eval_every,
        writer=writer,
        ckpt_dir=ckpt_dir,
        ckpt_every=args.ckpt_every,
        log_every=args.log_every,
        task_label=task_label,
    )

    if writer:
        writer.close()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n── Finetuning complete ────────────────────────────────────────────")
    print(f"  final train loss : {history['train_loss'][-1]:.5e}")
    if history["eval_mse"]:
        best_mse = history["best_mse"]
        best_epoch = history["eval_epochs"][history["eval_mse"].index(best_mse)]
        print(f"  best val/train MSE : {best_mse:.5e}  (epoch {best_epoch})")
        if history.get("best_ckpt_path"):
            print(f"  best checkpoint    : {history['best_ckpt_path']}")
        model.eval()
        with torch.no_grad():
            mse_final = eval_fn(model)
        print(f"  final eval MSE: {mse_final:.5e}")
        if test_eval_fn is not eval_fn:
            with torch.no_grad():
                mse_final_test = test_eval_fn(model)
            print(f"  final test MSE: {mse_final_test:.5e}")
            if writer:
                writer.add_scalar(f"{task_label}/eval_test_mse", mse_final_test, history["eval_epochs"][-1])

    return history


if __name__ == "__main__":
    main()
