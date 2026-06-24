# we need:
# target and inferred vector fields from a couple of 2D target systems from ODEBench, ideally in a 1x3 block featuring:
# ground-truth, ODEFormer, FIM
# [This is for Part 1 of the experimental section]

from pathlib import Path
from plot_strogatz import create_vector_field_function
from typing import Tuple, Optional, List
import sympy as sp
import json

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42

import numpy as np
import torch
from scipy.integrate import solve_ivp

import sys

from utils.eval_models import OdeonEval, OdeFormerEval, PredictionModel

sys.path.append(str(Path(__file__).parent.parent.parent / "src"))
from fim.models.ode_trainer import DataCorruptionModel

from itertools import product

SEED = 1
FONT_SIZE = 45

if False:
    def plot_vector_field(
        vector_field_func: callable,
        x_range: Tuple[float, float] = (-5, 5),
        y_range: Tuple[float, float] = (-5, 5),
        grid_resolution: int = 20,
        title: str = "",
        ax: Optional[plt.Axes] = None,
        use_streamplot: bool = True,
        trajectories: Optional[List[np.ndarray]] = None,
        trajectory_colors: Optional[List[str]] = None,
        show_legend: bool = True,
    ) -> plt.Axes:
        """
        Plot a 2D vector field with optional trajectories.
        
        Args:
            vector_field_func: Function that takes (x, y) and returns (dx/dt, dy/dt)
            x_range: (x_min, x_max) for plotting
            y_range: (y_min, y_max) for plotting
            grid_resolution: Number of grid points per dimension
            title: Plot title
            ax: Optional axes to plot on
            use_streamplot: If True, use streamplot; otherwise use quiver
            trajectories: Optional list of trajectory arrays, each of shape (T, 2)
            trajectory_colors: Optional list of colors for trajectories
            
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # Create grid
        x = np.linspace(x_range[0], x_range[1], grid_resolution)
        y = np.linspace(y_range[0], y_range[1], grid_resolution)
        X, Y = np.meshgrid(x, y)
        
        # Evaluate vector field
        U, V = vector_field_func(X, Y)
        
        # Compute magnitude for coloring
        magnitude = np.sqrt(U**2 + V**2)
        
        if use_streamplot:
            # Use streamplot for smoother visualization
            # Note: streamplot doesn't support alpha directly, but we can adjust the colormap
            ax.streamplot(X, Y, U, V, color=magnitude, cmap='viridis', 
                        density=1.5, linewidth=1.5, arrowsize=1.5)
            # Create colorbar for streamplot
            sm = ScalarMappable(norm=Normalize(vmin=magnitude.min(), vmax=magnitude.max()), 
                            cmap='viridis')
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, label='Vector Field Magnitude')
            # Set custom ticks (e.g., at 0.5, 1.0, 1.5, 2.0, 2.5)
            # You can customize these values as needed
            #tick_min = magnitude.min()
            #tick_max = magnitude.max()
            #tick_step = (tick_max - tick_min) / 5  # 5 ticks total
            #cbar.set_ticks(np.arange(tick_min, tick_max + tick_step, tick_step))
        else:
            # Use quiver for arrow visualization
            quiver = ax.quiver(X, Y, U, V, magnitude, cmap='viridis', 
                            scale=None, angles='xy', scale_units='xy', 
                            width=0.003, alpha=0.7)
            plt.colorbar(quiver, ax=ax, label='Vector Field Magnitude')
        
        # Plot trajectories if provided
        if trajectories is not None:
            if trajectory_colors is None:
                # Default colors: orange for first, red for second, then cycle
                default_colors = ['#ff7f0e', '#d62728', '#2ca02c', '#1f77b4', '#9467bd', 
                                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                trajectory_colors = [default_colors[i % len(default_colors)] for i in range(len(trajectories))]
            
            for i, traj in enumerate(trajectories):
                if traj.shape[1] >= 2:
                    color = trajectory_colors[i] if i < len(trajectory_colors) else default_colors[i % len(default_colors)]
                    # Use light blue for the enclosing line (or use the provided color)
                    line_color = color  # Light blue, or use color if preferred
                    
                    # Plot thick colorful line that encloses the trajectory (drawn first, behind points)
                    #ax.plot(traj[:, 0], traj[:, 1], color=line_color, linewidth=27, 
                    #       alpha=0.3, zorder=999, solid_capstyle='round')
                    ax.plot(traj[:, 0], traj[:, 1], color=line_color, linewidth=10, 
                        alpha=1, zorder=1000, solid_capstyle='round')
                    
                    # Plot black circles at each observed point
                    ax.scatter(traj[:, 0], traj[:, 1], s=15, color='black', alpha=0.7, 
                            zorder=1002, edgecolors='lightgrey', linewidths=1)
                    
                    """
                    # Mark start point: light blue square with dark blue border, containing "0"
                    ax.scatter(traj[0, 0], traj[0, 1], s=500, marker='s', 
                            facecolor=line_color, edgecolors='darkblue', linewidths=3,
                            zorder=1003)
                    ax.text(traj[0, 0], traj[0, 1], f"{i}", fontsize=18, ha='center', 
                        va='center', color='white', fontweight='bold', zorder=1004)
                    
                    # Mark end point: concentric black circles
                    ax.scatter(traj[-1, 0], traj[-1, 1], s=200, color='black', 
                            marker='o', zorder=1003, edgecolors='black', linewidths=2)
                    ax.scatter(traj[-1, 0], traj[-1, 1], s=100, color='black', 
                            marker='o', zorder=1004)
                    """
            
            if len(trajectories) > 0 and show_legend:
                ax.legend(loc='best', fontsize=10, framealpha=0.9)
        
        #ax.set_xlabel('$x_0$', fontsize=14)
        #ax.set_ylabel('$x_1$', fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.grid(False)# (True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        return ax


    def plot_all(max_odes: int, save_path: Path):
        json_path = Path(__file__).parent / "data/strogatz_extended.json"
        # Load data
        with open(json_path, 'r') as f:
            ODEs = json.load(f)
        
        # Filter by dimension
        ODEs = [item for item in ODEs if item.get("dim") == 2]

        # Limit number of ODEs
        if max_odes is not None:
            ODEs = ODEs[:max_odes]

        # Set up PDF saving if requested
        pdf_pages = None
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            pdf_pages = PdfPages(save_path)
            print(f"Saving all plots to: {save_path}")

        # Plot each ODE
        figures = []
        for ode in ODEs:
            ode_id = ode.get("id")
            eq_description = ode.get("eq_description", f"ODE {ode_id}")
            #eq_str = ode.get("eq", "")
            substituted = ode.get("substituted", [])
            consts = ode.get("consts", [])
            
            print(f"\nPlotting ODE {ode_id}: {eq_description}")
            print(f"  Equation: {substituted}")

            # equations for u and v
            sub_eqs = substituted[0]
            if len(sub_eqs) == 2:
                # Parse substituted equations
                expressions = []
                symbols = [sp.symbols(f"x_{i}") for i in range(2)]
                for sub_eq in sub_eqs:
                    expr = sp.sympify(sub_eq)
                    expressions.append(expr)
                
                vector_field_func = create_vector_field_function(expressions, symbols)
            else:
                raise ValueError(f"Expected 2 substituted equations, got {len(sub_eqs)}")

            # Extract all trajectories and determine plot range
            x_range = (-5, 5)
            y_range = (-5, 5)
            trajectories = []
            solutions = ode.get("solutions", [[]])
            
            if solutions and len(solutions[0]) > 0:
                # Collect all trajectories
                all_x_values = []
                all_y_values = []
                
                for sol in solutions[0]:
                    if "y" in sol and sol.get("success", False):
                        traj = np.array(sol["y"]).T  # Shape: (T, D)
                        if traj.shape[1] >= 2:
                            trajectories.append(traj)
                            all_x_values.extend(traj[:, 0])
                            all_y_values.extend(traj[:, 1])
                
                # Determine plot range from all trajectories
                if all_x_values and all_y_values:
                    x_min, x_max = min(all_x_values), max(all_x_values)
                    y_min, y_max = min(all_y_values), max(all_y_values)
                    # Add padding
                    x_pad = (x_max - x_min) * 0.2 if (x_max - x_min) > 0 else 1.0
                    y_pad = (y_max - y_min) * 0.2 if (y_max - y_min) > 0 else 1.0
                    x_range = (x_min - x_pad, x_max + x_pad)
                    y_range = (y_min - y_pad, y_max + y_pad)
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 10))
            plot_vector_field(
                vector_field_func,
                x_range=x_range,
                y_range=y_range,
                #title=f"ODE {ode_id}: {eq_description}",
                title=f"ODE {ode_id}: Ground Truth",
                ax=ax,
                use_streamplot=True,
                trajectories=trajectories if trajectories else None,
                trajectory_colors=["#FFD700", "#FF6347"],
                show_legend=False,
            )
            
            plt.tight_layout()
            
            # Save to PDF or collect for display
            if pdf_pages is not None:
                pdf_pages.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                print(f"  Added to PDF")
            else:
                figures.append(fig)
                
        # Close PDF or show figures
        if pdf_pages is not None:
            pdf_pages.close()
            print(f"\nSaved {len(ODEs)} plots to {save_path}")
        elif figures:
            plt.show()
            # Close all figures after showing
            for fig in figures:
                plt.close(fig)



def evaluate_vector_field_on_grid(model: PredictionModel, grid_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate a model's predicted vector field on a grid of points.
    
    Args:
        model: Fitted PredictionModel (OdeonEval or OdeFormerEval)
        grid_points: Array of shape (N, 2) with (x, y) coordinates
        
    Returns:
        U, V: Arrays of shape matching grid_points with dx/dt and dy/dt
    """
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32)

    if hasattr(model, 'system'):
        vf = model.system(grid_tensor.unsqueeze(0)).squeeze(0)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
    U = vf[:, 0].detach().cpu().numpy()
    V = vf[:, 1].detach().cpu().numpy()

    #print(type(model))
    return U, V


def integrate_trajectory(vector_field_func: callable, y0: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    """
    Integrate a trajectory using scipy's solve_ivp.
    
    Args:
        vector_field_func: Function that takes (x, y) and returns (dx/dt, dy/dt) tuple
        y0: Initial condition, shape (2,)
        t_eval: Time points to evaluate at
        
    Returns:
        Trajectory array of shape (len(t_eval), 2)
    """
    def ode_func(t, y):
        dx_dt, dy_dt = vector_field_func(y[0], y[1])
        # Stack and flatten to ensure 1D array shape (2,)
        result = np.array([dx_dt, dy_dt]).flatten()
        return result
    
    sol = solve_ivp(ode_func, (t_eval[0], t_eval[-1]), y0, t_eval=t_eval, method='RK45')
    return sol.y.T  # Shape: (T, 2)


def plot_comparison_1x3(
    ode: dict,
    model1: PredictionModel,
    model2: PredictionModel,
    odeon_checkpoint_path: Optional[Path] = None,
    grid_resolution: int = 30,
    sigma: float = 0.,
    rho: float = 0.,
    max_num_points: int = 200,
    axes: Optional[plt.Axes] = None,
    figsize: Tuple[float, float] = (30,10),
    with_colorbar: bool = True,
) -> plt.Figure:
    """
    Create a 1x3 comparison plot: Ground Truth, ODEFormer, ODEON.
    
    Args:
        ode: ODE dictionary from JSON
        model1: Pre-loaded model
        model2: Pre-loaded model
        odeon_checkpoint_path: Path to ODEON checkpoint if model not provided
        grid_resolution: Resolution for vector field grid
        
    Returns:
        matplotlib Figure
    """
    ode_id = ode.get("id", "Unknown")
    eq_description = ode.get("eq_description", f"ODE {ode_id}")
    
    # Parse ground truth vector field
    substituted = ode.get("substituted", [])
    sub_eqs = substituted[0]
    if len(sub_eqs) != 2:
        raise ValueError(f"Expected 2 substituted equations, got {len(sub_eqs)}")
    
    expressions = []
    symbols = [sp.symbols(f"x_{i}") for i in range(2)]
    for sub_eq in sub_eqs:
        expr = sp.sympify(sub_eq)
        expressions.append(expr)
    
    gt_vector_field_func = create_vector_field_function(expressions, symbols)
    
    # Get first and second trajectories
    solutions = ode.get("solutions", [[]])
    if not solutions or not solutions[0]:
        raise ValueError("No solutions found in ODE data")
    
    first_traj = None
    first_times = None
    second_traj = None
    second_times = None
    
    traj_count = 0
    for sol in solutions[0]:
        if "y" in sol and sol.get("success", False):
            traj = np.array(sol["y"]).T  # Shape: (T, D)
            if traj.shape[1] >= 2:
                if traj_count == 0:
                    first_traj = traj
                    first_times = np.array(sol["t"])
                    traj_count += 1
                elif traj_count == 1:
                    second_traj = traj
                    second_times = np.array(sol["t"])
                    traj_count += 1
                    break
    
    if first_traj is None:
        raise ValueError("No valid 2D trajectory found")
    if second_traj is None:
        print("Warning: Only one trajectory found, using first trajectory for both")
        second_traj = first_traj
        second_times = first_times
    
    # regularly subsample the trajectory to 200 points
    idxs = torch.linspace(0, first_times.shape[0] - 1, max_num_points).long()
    first_traj = first_traj[idxs, :]
    second_traj = second_traj[idxs, :]
    first_times = first_times[idxs]
    second_times = second_times[idxs]
    
    # Determine plot range from both trajectories
    all_x_values = list(first_traj[:, 0]) + list(second_traj[:, 0])
    all_y_values = list(first_traj[:, 1]) + list(second_traj[:, 1])
    x_min, x_max = min(all_x_values), max(all_x_values)
    y_min, y_max = min(all_y_values), max(all_y_values)
    x_pad = (x_max - x_min) * 0.2 if (x_max - x_min) > 0 else 1.0
    y_pad = (y_max - y_min) * 0.2 if (y_max - y_min) > 0 else 1.0
    x_range = (x_min - x_pad, x_max + x_pad)
    y_range = (y_min - y_pad, y_max + y_pad)
    
    # Prepare trajectory data for models (need torch tensors)
    traj_tensor = torch.tensor(first_traj, dtype=torch.float32).unsqueeze(0)  # (1, T, 2)
    times_tensor = torch.tensor(first_times, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
    mask = torch.ones_like(times_tensor, dtype=torch.bool)
    
    apply_corruption = True
    # Apply corruption if requested
    if apply_corruption:
        print(f"Applying corruption: subsample_rho={rho}, multiplicative_noise_sigma={sigma}")
        
        # Reshape to (b, t, n, d) format expected by corruption functions
        # traj_tensor is (1, T, 2), need (1, 1, T, 2)
        traj_reshaped = traj_tensor.unsqueeze(1)  # (1, 1, T, 2)
        
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        # Apply multiplicative noise first
        traj_reshaped = DataCorruptionModel.add_multiplicative_noise(
            traj_reshaped,
            sigma_distribution="uniform",
            min_sigma=sigma,
            max_sigma=sigma
        )
        
        # Generate subsampling mask
        subsample_mask = DataCorruptionModel.generate_subsample_points_mask(
            traj_reshaped,
            min_ratio=rho,
            max_ratio=rho,
            p_random=1.0  # Use random subsampling
        )
        
        # Apply mask to trajectory and times
        traj_reshaped = traj_reshaped * subsample_mask
        mask = subsample_mask.squeeze(1)  # (1, T, 1) to match times_tensor shape

        #print(mask)
        #print(mask.shape)
        #print(mask.sum())
        
        # Reshape back to (1, T, 2)
        traj_tensor = traj_reshaped.squeeze(1)  # (1, T, 2)
        
        # Save corrupted trajectory and mask for plotting
        corrupted_traj_for_plot = traj_tensor.clone()  # (1, T, 2)
        corrupted_mask_for_plot = mask.clone()  # (1, T, 1)
        
        # Apply mask to times as well
        times_tensor = times_tensor * mask.float()
    else:
        corrupted_traj_for_plot = None
        corrupted_mask_for_plot = None
    
    # Fit models on first trajectory
    print("Fitting model1...")
    model1.fit(traj=traj_tensor.unsqueeze(0), times=times_tensor.unsqueeze(0), mask=mask.unsqueeze(0))

    print("Fitting model2...")
    model2.fit(traj=traj_tensor, times=times_tensor, mask=mask)
    
    # Create grid for vector field evaluation
    x = np.linspace(x_range[0], x_range[1], grid_resolution)
    y = np.linspace(y_range[0], y_range[1], grid_resolution)
    X, Y = np.meshgrid(x, y)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)  # (N, 2)
    
    # Evaluate vector fields
    print("Evaluating vector fields...")
    gt_U, gt_V = gt_vector_field_func(X, Y)
    
    # Create wrapper functions for model vector fields
    def odeformer_vf(x, y):
        """Wrapper for ODEFormer vector field that handles numpy arrays."""
        if isinstance(x, np.ndarray) and x.ndim > 0:
            # Meshgrid case
            points = np.stack([x.ravel(), y.ravel()], axis=1)
            U, V = evaluate_vector_field_on_grid(model2, points)
            return U.reshape(x.shape), V.reshape(x.shape)
        else:
            # Scalar case for integration
            points = np.array([[float(x), float(y)]])
            U, V = evaluate_vector_field_on_grid(model2, points)
            return U[0], V[0]
    
    def odeon_vf(x, y):
        """Wrapper for ODEON vector field that handles numpy arrays."""
        if isinstance(x, np.ndarray) and x.ndim > 0:
            # Meshgrid case
            points = np.stack([x.ravel(), y.ravel()], axis=1)
            U, V = evaluate_vector_field_on_grid(model1, points)
            return U.reshape(x.shape), V.reshape(x.shape)
        else:
            # Scalar case for integration
            points = np.array([[float(x), float(y)]])
            U, V = evaluate_vector_field_on_grid(model1, points)
            return U[0], V[0]
    
    of_U, of_V = odeformer_vf(X, Y)
    odeon_U, odeon_V = odeon_vf(X, Y)
    
    # Integrate predicted trajectories from both initial conditions
    print("Integrating trajectories...")
    y0_1 = first_traj[0].flatten()  # Initial condition 1
    y0_2 = second_traj[0].flatten()  # Initial condition 2
    t_eval_1 = first_times.flatten() if first_times.ndim > 1 else first_times
    t_eval_2 = second_times.flatten() if second_times.ndim > 1 else second_times
    
    # Ground truth trajectories (both true trajectories)
    gt_traj_1 = first_traj  # First true trajectory (original, uncorrupted)
    gt_traj_2 = second_traj  # Second true trajectory
    
    # Predicted trajectories from IC1
    of_traj_pred_1 = integrate_trajectory(odeformer_vf, y0_1, t_eval_1)
    odeon_traj_pred_1 = integrate_trajectory(odeon_vf, y0_1, t_eval_1)
    
    # Predicted trajectories from IC2
    of_traj_pred_2 = integrate_trajectory(odeformer_vf, y0_2, t_eval_2)
    odeon_traj_pred_2 = integrate_trajectory(odeon_vf, y0_2, t_eval_2)
    
    # Create 1x3 subplot with tighter spacing
    if axes == None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        # Extend subplot block right so colorbar can sit close; minimal gap
        if with_colorbar:
            plt.subplots_adjust(left=0.06, right=0.88, top=0.92, bottom=0.10, wspace=0.15)
        else:
            plt.subplots_adjust(left=0.06, right=0.94, top=0.92, bottom=0.10, wspace=0.15)

    # Compute shared xlim/ylim from all trajectories so all three subplot rectangles are identical size
    all_trajs = [gt_traj_1, gt_traj_2, of_traj_pred_1, of_traj_pred_2, odeon_traj_pred_1, odeon_traj_pred_2]
    all_x = np.concatenate([t[:, 0] for t in all_trajs])
    all_y = np.concatenate([t[:, 1] for t in all_trajs])
    x_min_global = float(np.min(all_x))
    x_max_global = float(np.max(all_x))
    y_min_global = float(np.min(all_y))
    y_max_global = float(np.max(all_y))
    pad_x = (x_max_global - x_min_global) * 0.1 if (x_max_global - x_min_global) > 0 else 0.1
    pad_y = (y_max_global - y_min_global) * 0.1 if (y_max_global - y_min_global) > 0 else 0.1
    shared_xlim = (x_min_global - pad_x, x_max_global + pad_x)
    shared_ylim = (y_min_global - pad_y, y_max_global + pad_y)

    # Compute shared magnitude range for consistent colorbar
    all_magnitudes = [
        np.sqrt(gt_U**2 + gt_V**2),
        np.sqrt(of_U**2 + of_V**2),
        np.sqrt(odeon_U**2 + odeon_V**2)
    ]
    vmin = min(mag.min() for mag in all_magnitudes)
    vmax = max(mag.max() for mag in all_magnitudes)
    
    # Helper function to plot one subplot with multiple trajectories
    def plot_subplot(ax, X_plot, Y_plot, U, V, trajs, title, vmin, vmax, trajectory_colors=None,
                     corrupted_traj=None, corrupted_mask=None, xlim=None, ylim=None):
        """
        Plot vector field with multiple trajectories.
        
        Args:
            trajs: List of trajectory arrays, each of shape (T, 2)
            trajectory_colors: List of colors for trajectories (default: ["#FFD700", "#FF6347"])
            corrupted_traj: Optional corrupted trajectory array of shape (T, 2) to plot as X markers
            corrupted_mask: Optional boolean mask array of shape (T, 1) or (T,) indicating which points to plot
            xlim: Optional (x_min, x_max) to force identical axis limits across subplots
            ylim: Optional (y_min, y_max) to force identical axis limits across subplots
        """
        if trajectory_colors is None:
            trajectory_colors = ["#2ca02c", "#FF6347"]  # Green for first, tomato for second
        
        mag = np.sqrt(U**2 + V**2)
        norm = Normalize(vmin=vmin, vmax=vmax)
        ax.streamplot(X_plot, Y_plot, U, V, color="#bbbbbb", #color=mag, cmap='viridis',
                    density=.9, linewidth=3., arrowsize=3., norm=norm)
        # Note: Colorbar is created separately for all plots
        
        # Plot corrupted observations if provided
        if corrupted_traj is not None and corrupted_mask is not None:
            # Convert to numpy if needed
            if isinstance(corrupted_traj, torch.Tensor):
                corrupted_traj = corrupted_traj.cpu().numpy()
            if isinstance(corrupted_mask, torch.Tensor):
                corrupted_mask = corrupted_mask.cpu().numpy()
            
            # Handle mask shape: (1, T, 1) -> (T,), or (T, 1) -> (T,)
            if corrupted_mask.ndim > 1:
                corrupted_mask = corrupted_mask.squeeze()
            
            # Handle trajectory shape: (1, T, 2) -> (T, 2)
            if corrupted_traj.ndim == 3:
                corrupted_traj = corrupted_traj.squeeze(0)
            
            # Get indices where mask is True (observations are present)
            mask_indices = corrupted_mask.astype(bool)
            
            # Plot corrupted observations as fat black X markers
            if np.any(mask_indices):
                ax.scatter(corrupted_traj[mask_indices, 0], corrupted_traj[mask_indices, 1],
                          s=120, marker='x', color='black', linewidths=4, alpha=0.8,
                          zorder=1001, label='Corrupted observations')
        
        # Plot each trajectory
        for i, traj in enumerate(trajs):
            color = trajectory_colors[i % len(trajectory_colors)]
            ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=8,
                    alpha=1, zorder=1000, solid_capstyle='round')
            ax.scatter(traj[:, 0], traj[:, 1], s=15, color='black', alpha=0.5,
                    zorder=1002, edgecolors='lightgrey', linewidths=1)
            
            # Add colored box with number at start point
            ax.scatter(traj[0, 0], traj[0, 1], s=1000, marker='s', 
                      facecolor=color, edgecolors='black', linewidths=3,
                      zorder=1003)
            ax.text(traj[0, 0], traj[0, 1], f"{i}", fontsize=28, ha='center', 
                   va='center', color='white', fontweight='bold', zorder=1004)
        
        ax.set_aspect('equal', adjustable='box')

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=.75*FONT_SIZE,
            width=2,
            length=6,
        )

        ax.set_title(title, fontsize=FONT_SIZE, fontweight='bold', pad=20)
        ax.grid(False)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_aspect('equal', adjustable='box')
    
    # Prepare corrupted trajectory for plotting (convert to numpy if needed)
    corrupted_traj_np = None
    corrupted_mask_np = None
    if corrupted_traj_for_plot is not None and corrupted_mask_for_plot is not None:
        corrupted_traj_np = corrupted_traj_for_plot.squeeze(0).cpu().numpy()  # (T, 2)
        corrupted_mask_np = corrupted_mask_for_plot.squeeze(0).cpu().numpy()  # (T, 1)
    
    # Plot all three subplots with both trajectories (shared xlim/ylim so rectangle size is identical)
    if rho != 0. or sigma != 0.:   # corrupted
        plot_subplot(axes[0], X, Y, gt_U, gt_V, [gt_traj_1, gt_traj_2], "Ground Truth", vmin, vmax,
                 corrupted_traj=corrupted_traj_np, corrupted_mask=corrupted_mask_np,
                 xlim=shared_xlim, ylim=shared_ylim)
    else:                          # uncorrupted
        plot_subplot(axes[0], X, Y, gt_U, gt_V, [gt_traj_1, gt_traj_2], "Ground Truth", vmin, vmax,
                 xlim=shared_xlim, ylim=shared_ylim)
    if rho != 0. or sigma != 0.:   # corrupted
        plot_subplot(axes[1], X, Y, of_U, of_V, [of_traj_pred_1, of_traj_pred_2], "ODEFormer", vmin, vmax,
                 corrupted_traj=corrupted_traj_np, corrupted_mask=corrupted_mask_np,
                 xlim=shared_xlim, ylim=shared_ylim)
    else:                          # uncorrupted
        plot_subplot(axes[1], X, Y, of_U, of_V, [of_traj_pred_1, of_traj_pred_2], "ODEFormer", vmin, vmax,
                 xlim=shared_xlim, ylim=shared_ylim)
    if rho != 0. or sigma != 0.:   # corrupted
        plot_subplot(axes[2], X, Y, odeon_U, odeon_V, [odeon_traj_pred_1, odeon_traj_pred_2], "FIM-ODE", vmin, vmax,
                 corrupted_traj=corrupted_traj_np, corrupted_mask=corrupted_mask_np,
                 xlim=shared_xlim, ylim=shared_ylim)
    else:                          # uncorrupted
        plot_subplot(axes[2], X, Y, odeon_U, odeon_V, [odeon_traj_pred_1, odeon_traj_pred_2], "FIM-ODE", vmin, vmax,
                 xlim=shared_xlim, ylim=shared_ylim)
    
    # Create a single shared colorbar in a separate axes outside the subplots
    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap='viridis')
    sm.set_array([])
    # Create a dedicated axes for the colorbar positioned to the right of all subplots
    # [left, bottom, width, height] in figure coordinates (0-1)
    # Position colorbar just right of subplots to minimize gap
    if with_colorbar:
        cbar_ax = fig.add_axes([0.905, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(sm, cax=cbar_ax)

        # label font size
        cbar.set_label("Vector Field Magnitude", fontsize=.75*FONT_SIZE, labelpad=20)

        # tick font size
        cbar.ax.tick_params(labelsize=.75*FONT_SIZE)

    # Add ODE identifier as figure title
    #fig.suptitle(f"ODE {ode_id}: {eq_description}, {max_num_points} points, subsampling ratio rho={rho}, sigma={sigma}", fontsize=18, fontweight='bold', y=0.98)
    
    # Layout fixed above via subplots_adjust; skip tight_layout to keep colorbar gap small
    return fig


def plot_comparison_1x3_fixed_points(
    ode: dict,
    model1: PredictionModel,
    model2: PredictionModel,
    fps_gt: List[Tuple[Tuple[float, ...], str]],
    fps_model1: List[Tuple[Tuple[float, ...], str]],
    fps_model2: List[Tuple[Tuple[float, ...], str]],
    odeon_checkpoint_path: Optional[Path] = None,
    grid_resolution: int = 30,
    sigma: float = 0.,
    rho: float = 0.,
    max_num_points: int = 200,
    figsize: Tuple[float, float] = (30, 10),
    with_colorbar: bool = True,
) -> plt.Figure:
    """Like plot_comparison_1x3 but annotates each subplot with fixed-point info.

    Parameters
    ----------
    fps_gt / fps_model1 / fps_model2
        Fixed points for Ground Truth, FIM-ODE (model1), and ODEFormer (model2) respectively.
        Annotated under their corresponding subplots (GT / ODEFormer / FIM-ODE order on canvas).
        Each list
        contains ``(coords, stability_label)`` tuples, e.g.::

            [((-0.07, 0.05), "unstable spiral"), ((1.0, 0.0), "stable node")]

        Pass ``[]`` when no fixed points were found.
    """
    fig = plot_comparison_1x3(
        ode=ode,
        model1=model1,
        model2=model2,
        odeon_checkpoint_path=odeon_checkpoint_path,
        grid_resolution=grid_resolution,
        sigma=sigma,
        rho=rho,
        max_num_points=max_num_points,
        axes=None,
        figsize=figsize,
        with_colorbar=with_colorbar,
    )

    # First 3 axes are always the subplots; the optional 4th is the colorbar.
    subplot_axes = fig.axes[:3]

    fp_fontsize = 0.8 * FONT_SIZE   # same as tick labels

    all_fps = [fps_gt, fps_model2, fps_model1]
    max_n_lines = max(max(len(fp_list), 1) for fp_list in all_fps)

    # Expand bottom margin first so axis positions are final before we read them.
    needed_bottom = 0.10 + max_n_lines * 0.07
    if with_colorbar:
        fig.subplots_adjust(left=0.06, right=0.88, top=0.92, bottom=needed_bottom, wspace=0.15)
    else:
        fig.subplots_adjust(left=0.06, right=0.94, top=0.92, bottom=needed_bottom, wspace=0.15)

    # Force matplotlib to finalise the layout (including set_aspect='equal' box
    # shrinkage) so ax.get_position() returns the true rendered axes height.
    fig.canvas.draw()

    # One text line in figure fraction (font size in points / figure height in points).
    line_h_fig = (fp_fontsize * 1.4) / (figsize[1] * 72)

    for ax, fp_list in zip(subplot_axes, all_fps):
        # Use the true post-layout axes height so the gap is consistent across
        # systems with different data aspect ratios.
        actual_h     = ax.get_position().height          # figure fraction, after equal-aspect shrink
        first_y      = -(line_h_fig * 1.1) / actual_h   # just below tick labels
        line_spacing = line_h_fig / actual_h

        if not fp_list:
            ax.text(
                0.0, first_y, "no fixed points found",
                transform=ax.transAxes,
                ha='left', va='top',
                fontsize=fp_fontsize, style='italic', color='gray',
                clip_on=False,
            )
        else:
            for i, (coords, label) in enumerate(fp_list):
                coord_str = '(' + ', '.join(f'{c:.2f}' for c in coords) + ')'
                text = f"FP: {coord_str},  {label}"
                ax.text(
                    0.0, first_y - i * line_spacing, text,
                    transform=ax.transAxes,
                    ha='left', va='top',
                    fontsize=fp_fontsize, color='black',
                    clip_on=False,
                )

    return fig


def plot_comparison_all(max_odes: int, save_path: Path, odeon_checkpoint_path: Optional[Path] = None, sigma_rho_list: List[Tuple[float, float]] = [(0.,0.0), (0.5,0.03), (0.9,0.05)]):
    """
    Create 1x3 comparison plots for multiple ODEs.
    
    Args:
        max_odes: Maximum number of ODEs to plot
        save_path: Path to save PDF
        odeon_checkpoint_path: Path to ODEON checkpoint directory
    """
    json_path = Path(__file__).parent / "data/strogatz_extended.json"
    with open(json_path, 'r') as f:
        ODEs = json.load(f)
    
    # Filter by dimension
    ODEs = [item for item in ODEs if item.get("dim") == 2]
    
    # Sort by ODE ID to ensure consistent ordering
    ODEs = sorted(ODEs, key=lambda x: x.get("id", 0))
    
    # Limit number of ODEs
    if max_odes is not None:
        ODEs = ODEs[:max_odes]
    
    # Set up PDF saving
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_pages = PdfPages(save_path)
    print(f"Saving all plots to: {save_path}")
    
    # Load models once
    device = "cuda" if torch.cuda.is_available() else "cpu"
    odeformer_model = OdeFormerEval(device=device)
    odeon_model = None if odeon_checkpoint_path is None else OdeonEval(odeon_checkpoint_path)
    
    # Plot each ODE
    successful_plots = 0
    failed_odes = []
    
    for ode in ODEs:
        ode_id = ode.get("id")
        eq_description = ode.get("eq_description", f"ODE {ode_id}")
        print(f"\nPlotting ODE {ode_id}: {eq_description}")
        
        try:
            fig = plot_comparison_1x3(
                ode=ode,
                model1=odeon_model,
                model2=odeformer_model,
                odeon_checkpoint_path=odeon_checkpoint_path,
                rho=0.,
                sigma=0.,
                max_num_points=512,
            )
            pdf_pages.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            successful_plots += 1
            print(f"  Added to PDF")
            for rho, sigma in sigma_rho_list:
                fig = plot_comparison_1x3(
                    ode=ode,
                    model1=odeon_model,
                    model2=odeformer_model,
                    odeon_checkpoint_path=odeon_checkpoint_path,
                    rho=rho,
                    sigma=sigma,
                    max_num_points=200,
                )
                pdf_pages.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                successful_plots += 1
                print(f"  Added to PDF")
        except Exception as e:
            print(f"  Error plotting ODE {ode_id}: {e}")
            import traceback
            traceback.print_exc()
            failed_odes.append((ode_id, eq_description, str(e)))
            continue
    
    pdf_pages.close()
    print(f"\nSaved {successful_plots}/{len(ODEs)} comparison plots to {save_path}")
    if failed_odes:
        print(f"\nFailed ODEs ({len(failed_odes)}):")
        for ode_id, desc, error in failed_odes:
            print(f"  ODE {ode_id}: {desc} - {error}")


def plots_for_paper(
    sigma_rho_list: List[Tuple[float, float]] = [(0.5, 0.03), (0.5, 0.03)],
) -> None:

    save_path=Path(__file__).parent / "odebench.pdf"
    odeon_checkpoint_path=Path("models") / "base_model" / "checkpoints"

    json_path = Path(__file__).parent / "data/strogatz_extended.json"
    with open(json_path, "r") as f:
        all_odes = json.load(f)

    # 2D ODEs only, indexed by id
    odes_2d = [item for item in all_odes if item.get("dim") == 2]
    ode_by_id = {item["id"]: item for item in odes_2d}

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    odeformer_model = OdeFormerEval(device=device)
    odeon_model = OdeonEval(odeon_checkpoint_path) if odeon_checkpoint_path else None

    # ------------------------------------------------------------
    # ODE 28
    # ode28 = ode_by_id[28]
    # rho = 0.5
    # sigma = 0.03
    # fig = plot_comparison_1x3(
    #     ode=ode28,
    #     model1=odeon_model,
    #     model2=odeformer_model,
    #     odeon_checkpoint_path=odeon_checkpoint_path,
    #     sigma=sigma,
    #     rho=rho,
    #     axes=None,
    #     figsize=(30,10),
    #     with_colorbar=False,
    # )
    # out_file = save_path.parent / f"{save_path.stem}_ode_28.pdf"
    # fig.savefig(out_file, bbox_inches="tight")
    # plt.close(fig)
    # print(f"  Saved {out_file}")

    # ------------------------------------------------------------
    # ODE 42
    # ode42 = ode_by_id[42]
    # rho = 0.5
    # sigma = 0.03
    # fig = plot_comparison_1x3(
    #     ode=ode42,
    #     model1=odeon_model,
    #     model2=odeformer_model,
    #     odeon_checkpoint_path=odeon_checkpoint_path,
    #     sigma=sigma,
    #     rho=rho,
    #     axes=None,
    #     figsize=(30,15),
    #     with_colorbar=False,
    # )
    # out_file = save_path.parent / f"{save_path.stem}_ode_42.pdf"
    # fig.savefig(out_file, bbox_inches="tight")
    # plt.close(fig)
    # print(f"  Saved {out_file}")

    # ODE 26
    ode26 = ode_by_id[26]
    rho = 0.5
    sigma = 0.03
    fig = plot_comparison_1x3(
        ode=ode26,
        model1=odeon_model,
        model2=odeformer_model,
        odeon_checkpoint_path=odeon_checkpoint_path,
        sigma=sigma,
        rho=rho,
        axes=None,
        figsize=(30,15),
        with_colorbar=False,
    )
    out_file = save_path.parent / f"{save_path.stem}_ode_26.pdf"
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_file}")


if __name__ == "__main__":
    plots_for_paper()
    quit()

    # For comparison plots (1x3 layout)
    # save_path = Path(__file__).parent / "plots_for_paper_comparison.pdf"
    # odeon_path = Path("models") / "base_model" / "checkpoints"
    # sigma_rho_list = [(0.,0.0), (0.5,0.03), (0.9,0.05)]
    # plot_comparison_all(max_odes=None, save_path=save_path, odeon_checkpoint_path=odeon_path, sigma_rho_list=sigma_rho_list)
    
    # For single plots (original)
    # save_path = Path(__file__).parent / "plots_for_paper.pdf"
    # plot_all(max_odes=2, save_path=save_path)