"""Plot 2D vector fields from ODEBench (Strogatz extended) dataset."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib.backends.backend_pdf import PdfPages


def parse_equation_string(eq_str: str, dim: int = 2) -> List[sp.Expr]:
    """
    Parse equation string into sympy expressions.

    Args:
        eq_str: Equation string with components separated by '|'
        dim: Dimension of the ODE system

    Returns:
        List of sympy expressions, one per dimension
    """
    components = eq_str.split("|")
    if len(components) != dim:
        raise ValueError(f"Expected {dim} components separated by '|', got {len(components)}")

    symbols = [sp.symbols(f"x_{i}") for i in range(dim)]
    expressions = []

    for comp in components:
        comp = comp.strip()
        # Parse the expression
        expr = sp.sympify(comp)
        expressions.append(expr)

    return expressions, symbols


def create_vector_field_function(
    expressions: List[sp.Expr], symbols: List[sp.Symbol], constants: Optional[Dict[str, float]] = None
) -> callable:
    """
    Create a callable vector field function from sympy expressions.

    Args:
        expressions: List of sympy expressions for each dimension
        symbols: List of sympy symbols (x_0, x_1, ...)
        constants: Optional dictionary mapping constant names to values

    Returns:
        Callable function that takes (x, y) and returns (dx/dt, dy/dt)
    """
    # Substitute constants if provided
    if constants:
        for expr in expressions:
            for const_name, const_value in constants.items():
                expr = expr.subs(sp.symbols(const_name), const_value)

    # Create lambdified functions
    funcs = []
    for expr in expressions:
        if expr.free_symbols:
            f = sp.lambdify(symbols, expr, modules="numpy")
        else:
            # Constant expression - need to capture val in closure
            val = float(expr)

            def make_const_func(const_val):
                def const_func(*args):
                    # Return array of same shape as first argument
                    if len(args) > 0:
                        return np.full_like(args[0], const_val)
                    else:
                        return const_val

                return const_func

            f = make_const_func(val)
        funcs.append(f)

    def vector_field(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate vector field at points (x, y)."""
        # Handle both scalar and array inputs
        x_was_scalar = np.isscalar(x)
        if x_was_scalar:
            x = np.array([x])
            y = np.array([y])

        # Ensure arrays are numpy arrays
        x = np.asarray(x)
        y = np.asarray(y)

        # Evaluate each component
        if len(funcs) > 0:
            dx_dt = funcs[0](x, y)
            # Handle scalar return values
            if np.isscalar(dx_dt):
                dx_dt = np.full_like(x, dx_dt)
        else:
            dx_dt = np.zeros_like(x)

        if len(funcs) > 1:
            dy_dt = funcs[1](x, y)
            # Handle scalar return values
            if np.isscalar(dy_dt):
                dy_dt = np.full_like(y, dy_dt)
        else:
            dy_dt = np.zeros_like(y)

        # Ensure same shape
        dx_dt = np.broadcast_to(dx_dt, x.shape)
        dy_dt = np.broadcast_to(dy_dt, y.shape)

        return dx_dt, dy_dt

    return vector_field


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
        ax.streamplot(X, Y, U, V, color=magnitude, cmap="viridis", density=1.5, linewidth=1.5, arrowsize=1.5)
    else:
        # Use quiver for arrow visualization
        quiver = ax.quiver(X, Y, U, V, magnitude, cmap="viridis", scale=None, angles="xy", scale_units="xy", width=0.003, alpha=0.7)
        plt.colorbar(quiver, ax=ax, label="Vector Field Magnitude")

    # Plot trajectories if provided
    if trajectories is not None:
        if trajectory_colors is None:
            # Default colors: orange for first, red for second, then cycle
            default_colors = ["#ff7f0e", "#d62728", "#2ca02c", "#1f77b4", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
            trajectory_colors = [default_colors[i % len(default_colors)] for i in range(len(trajectories))]

        for i, traj in enumerate(trajectories):
            if traj.shape[1] >= 2:
                color = trajectory_colors[i] if i < len(trajectory_colors) else default_colors[i % len(default_colors)]
                label = f"Trajectory {i + 1}" if len(trajectories) > 1 else "Trajectory"
                ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=2.5, label=label, alpha=0.9, zorder=5)
                # Mark start point
                ax.scatter(
                    traj[0, 0],
                    traj[0, 1],
                    color=color,
                    s=100,
                    marker="o",
                    edgecolors="black",
                    linewidths=2,
                    label=f"Start {i + 1}" if len(trajectories) > 1 else "Start",
                    zorder=6,
                )
                # Mark end point
                ax.scatter(
                    traj[-1, 0],
                    traj[-1, 1],
                    color=color,
                    s=100,
                    marker="s",
                    edgecolors="black",
                    linewidths=2,
                    label=f"End {i + 1}" if len(trajectories) > 1 else "End",
                    zorder=6,
                )

        if len(trajectories) > 0:
            ax.legend(loc="best", fontsize=10, framealpha=0.9)

    ax.set_xlabel("$x_0$", fontsize=14)
    ax.set_ylabel("$x_1$", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    return ax


def plot_strogatz(
    json_path: Optional[Path] = None,
    ode_ids: Optional[List[int]] = None,
    max_odes: Optional[int] = None,
    dim: int = 2,
    save_path: Optional[Path] = None,
    use_substituted: bool = True,
) -> None:
    """
    Plot 2D vector fields from Strogatz extended dataset.

    Args:
        json_path: Path to strogatz_extended.json. If None, uses default location.
        ode_ids: Optional list of ODE IDs to plot. If None, plots all 2D ODEs.
        max_odes: Maximum number of ODEs to plot (for testing)
        dim: Dimension to filter (default 2 for 2D vector fields)
        save_path: Path to save PDF file. If None, displays plots.
        use_substituted: If True, use substituted equations; otherwise parse and substitute manually.
    """
    # Default path
    if json_path is None:
        json_path = Path(__file__).parent.parent.parent.parent / "data" / "ode" / "odebench" / "strogatz_extended.json"

    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    # Load data
    with open(json_path, "r") as f:
        data = json.load(f)

    # Filter by dimension
    filtered_data = [item for item in data if item.get("dim") == dim]

    # Filter by ODE IDs if specified
    if ode_ids is not None:
        filtered_data = [item for item in filtered_data if item.get("id") in ode_ids]

    # Limit number of ODEs
    if max_odes is not None:
        filtered_data = filtered_data[:max_odes]

    print(f"Plotting {len(filtered_data)} {dim}D ODEs")

    # Set up PDF saving if requested
    pdf_pages = None
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        pdf_pages = PdfPages(save_path)
        print(f"Saving all plots to: {save_path}")

    # Plot each ODE
    figures = []
    for item in filtered_data:
        ode_id = item.get("id")
        eq_description = item.get("eq_description", f"ODE {ode_id}")
        eq_str = item.get("eq", "")
        substituted = item.get("substituted", [])
        consts = item.get("consts", [])

        print(f"\nPlotting ODE {ode_id}: {eq_description}")
        print(f"  Equation: {eq_str}")

        try:
            # Try to use substituted equations first (they have constants already substituted)
            if use_substituted and substituted and len(substituted) > 0:
                # substituted is a list of lists, one per solution
                # Use the first solution's substituted equations
                sub_eqs = substituted[0]
                if len(sub_eqs) == dim:
                    # Parse substituted equations
                    expressions = []
                    symbols = [sp.symbols(f"x_{i}") for i in range(dim)]
                    for sub_eq in sub_eqs:
                        expr = sp.sympify(sub_eq)
                        expressions.append(expr)

                    vector_field_func = create_vector_field_function(expressions, symbols)
                    print(f"  Using substituted equations: {sub_eqs}")
                else:
                    raise ValueError(f"Expected {dim} substituted equations, got {len(sub_eqs)}")
            else:
                # Parse original equation and substitute constants
                expressions, symbols = parse_equation_string(eq_str, dim)

                # Substitute constants if available
                if consts and len(consts) > 0:
                    # Use first set of constants
                    const_values = consts[0]
                    const_names = [f"c_{i}" for i in range(len(const_values))]
                    constants_dict = dict(zip(const_names, const_values))

                    # Substitute constants into expressions
                    for i, expr in enumerate(expressions):
                        for const_name, const_value in constants_dict.items():
                            expressions[i] = expressions[i].subs(sp.symbols(const_name), const_value)

                vector_field_func = create_vector_field_function(expressions, symbols)
                print("  Using original equation with constants substituted")

            # Extract all trajectories and determine plot range
            x_range = (-5, 5)
            y_range = (-5, 5)
            trajectories = []
            solutions = item.get("solutions", [[]])

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
                title=f"ODE {ode_id}: {eq_description}",
                ax=ax,
                use_streamplot=True,
                trajectories=trajectories if trajectories else None,
            )

            plt.tight_layout()

            # Save to PDF or collect for display
            if pdf_pages is not None:
                pdf_pages.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                print("  Added to PDF")
            else:
                figures.append(fig)

        except Exception as e:
            print(f"  Error plotting ODE {ode_id}: {e}")
            import traceback

            traceback.print_exc()
            if pdf_pages is None:
                # Close any partially created figure
                if "fig" in locals():
                    plt.close(fig)
            continue

    # Close PDF or show figures
    if pdf_pages is not None:
        pdf_pages.close()
        print(f"\nSaved {len(filtered_data)} plots to {save_path}")
    elif figures:
        plt.show()
        # Close all figures after showing
        for fig in figures:
            plt.close(fig)


if __name__ == "__main__":
    # Example usage
    plot_strogatz(
        max_odes=None,  # Plot first 5 2D ODEs
        save_path=Path(__file__).parent / "strogatz_vector_fields.pdf",
    )
