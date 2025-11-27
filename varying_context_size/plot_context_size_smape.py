import matplotlib
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.ticker import MaxNLocator


# --- USER CONFIGURATION ---
# Define the paths to the CSV files and their corresponding labels
DATA_FILES = {
    "Hawkes": "results/varying_context_size/const_base_exp_kernel_no_interactions_sin_exp_kernel_kernel_intensity_error_vs_context_paths.csv",
    "Sin Base Intensity": "results/varying_context_size/sin_exp_kernel_kernel_intensity_error_vs_context_paths.csv",
    "Poisson": "results/varying_context_size/const_poisson_kernel_intensity_error_vs_context_paths.csv",
}
# --- END OF USER CONFIGURATION ---


def load_data(filepath):
    """
    Load CSV file and extract context_size, smape_mean, and smape_std.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame with columns ['context_size', 'smape_mean', 'smape_std'], sorted by context_size.
    """
    try:
        df = pd.read_csv(filepath)
        # Extract only the columns we need
        data = df[["context_size", "smape_mean", "smape_std"]].copy()
        # Sort by context_size
        data = data.sort_values("context_size")
        return data
    except FileNotFoundError:
        print(f"Error: The file was not found at path: {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while loading {filepath}: {e}")
        return pd.DataFrame()


def create_plot(data_dict):
    """
    Generate and save a plot showing sMAPE vs context size with error bars.

    Args:
        data_dict (dict): Dictionary mapping labels to DataFrames with context_size, smape_mean, smape_std.
    """
    # Use a clean style without relying on seaborn; configure explicitly
    # Okabe–Ito colorblind-friendly palette (as requested)
    okabe_ito_colors = [
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#F0E442",
        "#0072B2",
        "#D55E00",
        "#CC79A7",
    ]
    plt.rcParams["axes.prop_cycle"] = cycler(color=okabe_ito_colors)

    # Set Computer Modern-like fonts without requiring LaTeX
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Computer Modern Roman",
                "CMU Serif",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "cm",
            "font.size": 16,
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "figure.titlesize": 22,
        }
    )

    # Create a single figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Plot data for each dataset with error bars
    for label, df in data_dict.items():
        if df.empty:
            continue
        x_vals = df["context_size"].values
        y_vals = df["smape_mean"].values
        y_err = df["smape_std"].values

        ax.errorbar(
            x_vals,
            y_vals,
            yerr=y_err,
            label=label,
            linewidth=2.0,
            marker="o",
            markersize=8,
            capsize=3,
            capthick=1.5,
            elinewidth=1.5,
        )

    # Set axis labels
    ax.set_xlabel("Context Size")
    ax.set_ylabel("sMAPE")

    # Remove grid lines and align aesthetics with intensity plotting script
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)
    ax.tick_params(axis="both", direction="out", width=0.5, length=2, pad=0.8)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune=None))
    # Set x-axis ticks at 100, 500, 1000, 1500, 2000
    ax.set_xticks([100, 500, 1000, 1500, 2000])
    ax.margins(x=0.05)

    # Add legend
    legend = ax.legend(
        loc="best",
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        borderpad=0.2,
        handletextpad=0.9,
        borderaxespad=0.2,
        labelspacing=0.35,
    )
    frame = legend.get_frame()
    frame.set_linewidth(0.6)
    frame.set_edgecolor("black")
    frame.set_facecolor("white")

    # Adjust layout
    plt.tight_layout()

    # Save as PDF
    output_filename = "context_size_smape_plot.pdf"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight", format="pdf")
    print(f"\nPlot saved as {output_filename}")
    plt.close()


def main():
    """Main function to orchestrate data loading and plotting."""
    data_dict = {}

    print("--- Loading Context Size Data ---")
    for label, filepath in DATA_FILES.items():
        print(f"Loading '{label}' from '{filepath}'...")
        df = load_data(filepath)
        if df.empty:
            print(f"  Warning: No data loaded from {filepath}. Skipping.")
            continue
        data_dict[label] = df
        print(f"  Loaded {len(df)} data points")

    if not data_dict:
        print("\nError: No data was successfully loaded. Please check file paths and format.")
        return

    print("\n--- Generating Plot ---")
    create_plot(data_dict)


if __name__ == "__main__":
    main()
