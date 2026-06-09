import numpy as np
import torch
from typing import Tuple
from pathlib import Path
from utils.h5 import get_ndarray_from_h5


class trajectory:
    """Trajectory data structure for storing time series data.
    
    Attributes:
        ts: Time points, shape (T,) where T is number of time points
        xs: State values, shape (T, D) where T is number of time points and D is dimension
        dim: State dimension D
    
    The class accepts xs in multiple formats:
        - (T, D): 2D array, stored as-is
        - (T,): 1D array, converted to (T, 1)
        - (1, T, D): 3D array, automatically squeezed to (T, D)
        - (1, 1, T, D): 4D array, automatically squeezed to (T, D)
    
    The class accepts ts in multiple formats:
        - (T,): 1D array, stored as-is
        - (1, T): 2D array, automatically squeezed to (T,)
        - (1, 1, T): 3D array, automatically squeezed to (T,)
        - (1, 1, T, 1): 4D array, automatically squeezed to (T,)
    """
    ts: np.ndarray      # shape (T,)
    xs: np.ndarray      # shape (T,D) where D=dim
    dim: int
    length: int

    def __init__(self, xs: np.ndarray, ts: np.ndarray):
        """Initialize trajectory.
        
        Args:
            xs: State values, shape (T, D), (T,), (1, T, D), or (1, 1, T, D)
            ts: Time points, shape (T,), (1, T), (1, 1, T), or (1, 1, T, 1)
        """
        # Convert to numpy arrays if needed
        xs = np.asarray(xs)
        ts = np.asarray(ts)
        
        # Normalize ts to 1-dimensional (similar to how xs is handled)
        # First remove trailing singleton dimensions
        while ts.ndim > 1 and ts.shape[-1] == 1:
            ts = ts.squeeze(-1)
        # Then remove leading singleton dimensions
        while ts.ndim > 1:
            ts = ts.squeeze(0)
        
        # Ensure ts is 1-dimensional after normalization
        if ts.ndim != 1:
            raise ValueError(f"ts must be 1-dimensional after squeezing, got shape {ts.shape}")
        
        # Handle different input shapes for xs
        if xs.ndim == 4:
            # Shape (1, 1, T, D) - squeeze first two dimensions
            assert xs.shape[0] == 1 and xs.shape[1] == 1, \
                f"xs with 4 dimensions must have shape (1, 1, T, D), got {xs.shape}"
            xs = xs.squeeze(0).squeeze(0)  # Results in (T, D)
        elif xs.ndim == 3:
            # Shape (1, T, D) - squeeze first dimension
            assert xs.shape[0] == 1, \
                f"xs with 3 dimensions must have shape (1, T, D), got {xs.shape}"
            xs = xs.squeeze(0)  # Results in (T, D)
        elif xs.ndim == 1:
            # Shape (T,) - add dimension to make (T, 1)
            xs = xs[:, np.newaxis]
        elif xs.ndim == 2:
            # Shape (T, D) - already correct
            pass
        else:
            raise ValueError(f"xs must have 1, 2, 3, or 4 dimensions, got {xs.ndim} dimensions with shape {xs.shape}")
        
        # Ensure xs is now 2-dimensional (T, D)
        assert xs.ndim == 2, f"After processing, xs must be 2-dimensional, got shape {xs.shape}"
        
        # Ensure time dimension matches
        T_ts = ts.shape[0]
        T_xs = xs.shape[0]
        assert T_ts == T_xs, f"Time dimension mismatch: ts has {T_ts} points, xs has {T_xs} points"
        
        self.ts = ts
        self.xs = xs
        self.length = xs.shape[0]
        self.dim = xs.shape[1]

    def _get_ts_torch(self) -> torch.Tensor:
        """Convert time points to torch tensor.
        
        Returns:
            torch.Tensor: Time points, shape (T,)
        """
        return torch.from_numpy(self.ts).float()
    
    def _get_xs_torch(self) -> torch.Tensor:
        """Convert state values to torch tensor.
        
        Returns:
            torch.Tensor: State values, shape (T, D)
        """
        return torch.from_numpy(self.xs).float()
    
    def __str__(self):
        return f"Trajectory of length {self.length} and dimension {self.dim}"


def trajectory_list_from_h5_files(path_to_xs: Path, path_to_ts: Path) -> list[trajectory]:
    """Load trajectories from h5 files and return as a list of trajectory objects.
    
    Handles various shapes:
        - (1, 1, T, D) -> single trajectory in list
        - (1, P, T, D) -> P trajectories in list
        - (B, P, T, D) -> B*P trajectories in list
    
    Args:
        path_to_xs: Path to h5 file containing state values
        path_to_ts: Path to h5 file containing time points
    
    Returns:
        list[trajectory]: List of trajectory objects
    """
    xs = get_ndarray_from_h5(path_to_xs)
    ts = get_ndarray_from_h5(path_to_ts)
    
    # Normalize ts to remove trailing singleton dimensions
    while ts.ndim > 1 and ts.shape[-1] == 1:
        ts = ts.squeeze(-1)
    
    # Handle different input shapes
    if xs.ndim == 4:
        # Shape (B, P, T, D) or (1, P, T, D) or (1, 1, T, D)
        B, P, T, D = xs.shape
        
        # ts should be (B, P, T) or (1, P, T) or (1, 1, T) after squeezing
        if ts.ndim == 3:
            # (B, P, T) or (1, P, T) or (1, 1, T)
            pass
        elif ts.ndim == 2:
            # (P, T) or (1, T) - broadcast to (1, P, T) or (1, 1, T)
            ts = ts[np.newaxis, :, :]
        elif ts.ndim == 1:
            # (T,) - broadcast to (1, 1, T)
            ts = ts[np.newaxis, np.newaxis, :]
        else:
            raise ValueError(f"Unexpected ts shape {ts.shape} for xs shape {xs.shape}")
        
        # Ensure ts has compatible shape
        if ts.shape[0] == 1 and B > 1:
            ts = np.broadcast_to(ts, (B, ts.shape[1], ts.shape[2]))
        if ts.shape[1] == 1 and P > 1:
            ts = np.broadcast_to(ts, (ts.shape[0], P, ts.shape[2]))
        
        # Iterate over batch and trajectory dimensions
        trajectory_list = []
        for b in range(B):
            for p in range(P):
                xs_traj = xs[b, p, :, :]  # (T, D)
                ts_traj = ts[b, p, :]  # (T,)
                trajectory_list.append(trajectory(xs=xs_traj, ts=ts_traj))
        
        return trajectory_list
    
    elif xs.ndim == 3:
        # Shape (1, T, D) or (P, T, D)
        if xs.shape[0] == 1:
            # Single trajectory: (1, T, D)
            xs_traj = xs.squeeze(0)  # (T, D)
            ts_traj = ts.squeeze(0) if ts.ndim > 1 else ts
            return [trajectory(xs=xs_traj, ts=ts_traj)]
        else:
            # Multiple trajectories: (P, T, D)
            P, T, D = xs.shape
            trajectory_list = []
            # ts might be (P, T) or (1, T) or (T,)
            if ts.ndim == 2 and ts.shape[0] == P:
                # (P, T)
                for p in range(P):
                    xs_traj = xs[p, :, :]  # (T, D)
                    ts_traj = ts[p, :]  # (T,)
                    trajectory_list.append(trajectory(xs=xs_traj, ts=ts_traj))
            else:
                # (1, T) or (T,) - use same time for all trajectories
                ts_traj = ts.squeeze(0) if ts.ndim > 1 else ts
                for p in range(P):
                    xs_traj = xs[p, :, :]  # (T, D)
                    trajectory_list.append(trajectory(xs=xs_traj, ts=ts_traj))
            return trajectory_list
    
    elif xs.ndim == 2:
        # Shape (T, D) - single trajectory
        ts_traj = ts.squeeze() if ts.ndim > 1 else ts
        return [trajectory(xs=xs, ts=ts_traj)]
    
    else:
        raise ValueError(f"Unexpected xs shape: {xs.shape}")



if __name__ == "__main__":
    """ Testing the data models... """
    
    """
        Testing class trajectory
    """
    # Test 1D input
    traj1 = trajectory(xs=np.linspace(0, 1, 100), ts=np.linspace(0, 1, 100))
    print(f"1D input: xs shape {traj1.xs.shape}, dim={traj1.dim}")
    
    # Test 2D input
    xs_2d = np.random.randn(100, 2)
    traj2 = trajectory(xs=xs_2d, ts=np.linspace(0, 1, 100))
    print(f"2D input: xs shape {traj2.xs.shape}, dim={traj2.dim}")
    
    # Test 3D input (1, T, D)
    xs_3d = np.random.randn(1, 100, 2)
    traj3 = trajectory(xs=xs_3d, ts=np.linspace(0, 1, 100))
    print(f"3D input: xs shape {traj3.xs.shape}, dim={traj3.dim}")
    
    # Test 4D input (1, 1, T, D)
    xs_4d = np.random.randn(1, 1, 100, 2)
    traj4 = trajectory(xs=xs_4d, ts=np.linspace(0, 1, 100))
    print(f"4D input: xs shape {traj4.xs.shape}, dim={traj4.dim}")


    """
        Testing ...
    """

