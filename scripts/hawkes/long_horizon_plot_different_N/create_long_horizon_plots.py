import os
import re

import matplotlib
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# 1.) Specifies the txt names with the tabulars as attached via a global variable
# --- USER CONFIGURATION ---
# Please update the dictionary below with the correct paths to your data files.
# The key is the horizon number (N), and the value is the path to the file.
HORIZON_FILES = {
    5: "scripts/hawkes/long_horizon_plot_different_N/N=5.txt",
    10: "scripts/hawkes/long_horizon_plot_different_N/N=10.txt",
    20: "scripts/hawkes/long_horizon_plot_different_N/N=20.txt",
}
# --- END OF USER CONFIGURATION ---


# 2.) Parses and loads the tabulars with all included models.
def parse_latex_table(filepath):
    """
    Parses a LaTeX tabular file to extract model performance metrics.
    This version is robust to handle different LaTeX commands for method names.

    Args:
        filepath (str): The path to the .txt file containing the LaTeX table.

    Returns:
        pd.DataFrame: A DataFrame with columns ['Method', 'RMSE_e', 'sMAPE_delta_t'].
    """
    data = []
    # Regex to capture content within \method{} or \textbf{}
    method_regex = re.compile(r"(?:\\method|\\textbf)\{(.+?)\}")
    # Regex to find the main numeric value of a metric
    value_regex = re.compile(r"\$?(\d+\.\d+)")

    try:
        with open(filepath, "r") as f:
            for line in f:
                # Remove LaTeX comments: anything after '%' is a comment
                if "%" in line:
                    line = line.split("%", 1)[0]
                line = line.strip()
                if not line:
                    continue

                if "&" not in line:
                    continue

                parts = line.split("&")
                if len(parts) < 6:
                    continue

                method_match = method_regex.search(parts[1])
                if not method_match:
                    continue

                # Clean up the method name (e.g., from '\FIMzeroshot' to 'FIMzeroshot')
                method = method_match.group(1).lstrip("\\")

                # Standardize common names if needed
                if method == "A-NHP":
                    method = "AttNHP"

                rmse_match = value_regex.search(parts[3])
                smape_match = value_regex.search(parts[5])

                if rmse_match and smape_match:
                    data.append({"Method": method, "RMSE_e": float(rmse_match.group(1)), "sMAPE_delta_t": float(smape_match.group(1))})
    except FileNotFoundError:
        print(f"Error: The file was not found at path: {filepath}")
        return pd.DataFrame()
    except Exception as e:
        print(f"An error occurred while parsing {filepath}: {e}")
        return pd.DataFrame()

    return pd.DataFrame(data)


# 3.) Creates plots for RMSE_e and sMAPE_delta_t side-by-side.
def create_plots(aggregated_data, horizons):
    """
    Generates and saves the side-by-side plots for RMSE_e and sMAPE.

    Args:
        aggregated_data (dict): A dictionary containing the processed data.
        horizons (list): A list of horizon values for the x-axis.
    """
    # Define styles for models. Your FIM models are given distinct styles.
    # Any model not in this dict will get a default style from matplotlib's color cycle.
    model_styles = {
        # Your models with custom, standout styles
        "FIMzeroshot": {"color": "#d62728", "marker": "*", "label": "FIM (zero-shot)", "linewidth": 2.5, "markersize": 12},
        "FIMfine": {"color": "#2ca02c", "marker": "D", "label": "FIM (fine-tuned)", "linewidth": 2.5, "markersize": 9},
        # Baseline models (optional, can be removed for default styling)
        "TCDDM": {"color": "#d3b5e5", "marker": "^"},
        "Dual-TPP": {"color": "#0077b6", "marker": "^"},
        "NHP": {"color": "#e01e8b", "marker": "^"},
        "AttNHP": {"color": "#6950a1", "marker": "^"},
        "HYPRO": {"color": "#e57c23", "marker": "^"},
        "CDiff": {"color": "#1b998b", "marker": "^"},
    }

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("seaborn-whitegrid")
    plt.rcParams.update(
        {"font.family": "serif", "font.size": 16, "axes.labelsize": 18, "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 12}
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot data for each model
    for method, data in aggregated_data.items():
        style = dict(model_styles.get(method, {}))
        label = style.pop("label", method)

        # Left Plot: RMSE_e
        x_vals = data.get("N", horizons)
        ax1.plot(x_vals, data["RMSE_e"], label=label, **style)

        # Right Plot: sMAPE
        ax2.plot(x_vals, data["sMAPE_delta_t"], label=label, **style)

    # --- Final Touches ---
    ax1.set_xlabel("Horizon N", fontweight="bold")
    ax1.set_ylabel(r"RMSE$_e$", fontweight="bold")
    ax1.set_xticks(horizons)
    ax1.legend()
    ax1.set_title("Event Type Prediction Error (RMSE)", fontsize=16)

    ax2.set_xlabel("Horizon N", fontweight="bold")
    ax2.set_ylabel("sMAPE", fontweight="bold")
    ax2.set_xticks(horizons)
    ax2.set_title("Time Prediction Error (sMAPE)", fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.suptitle("Model Performance vs. Prediction Horizon", fontsize=20, fontweight="bold")

    output_filename = "horizon_metrics_plot.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved as {output_filename}")
    plt.show()


def main():
    """Main function to orchestrate parsing and plotting."""
    aggregated_data = {}
    horizons = sorted(HORIZON_FILES.keys())

    print("--- Parsing Horizon Data ---")
    for horizon in horizons:
        filepath = HORIZON_FILES[horizon]
        print(f"Processing Horizon {horizon} from '{filepath}'...")

        df = parse_latex_table(filepath)
        if df.empty:
            print(f"  Warning: No data parsed from {filepath}. Skipping.")
            continue

        # Average metrics over all datasets for this horizon
        avg_metrics = df.groupby("Method").mean(numeric_only=True)

        for method, row in avg_metrics.iterrows():
            if method not in aggregated_data:
                aggregated_data[method] = {"RMSE_e": [], "sMAPE_delta_t": [], "N": []}

            aggregated_data[method]["RMSE_e"].append(row["RMSE_e"])
            aggregated_data[method]["sMAPE_delta_t"].append(row["sMAPE_delta_t"])
            aggregated_data[method]["N"].append(horizon)

    if not aggregated_data:
        print("\nError: No data was successfully parsed. Please check file paths and format.")
        return

    print("\n--- Generating Plots ---")
    create_plots(aggregated_data, horizons)


if __name__ == "__main__":
    # Create dummy files if the specified files don't exist, for demonstration.
    # This part will be skipped if your files are found.
    if not all(os.path.exists(f) for f in HORIZON_FILES.values()):
        print("Warning: One or more data files not found. Creating dummy files for demonstration.")
        print("Please update the HORIZON_FILES variable with the correct paths to your data.")

        dummy_template = r"""
\begin{tabular}{llrrrr}
Dataset & Method & OTD & \text{RMSE\textsubscript{e}} & sMAPE\textsubscript{$\Delta t$} \\
\midrule
\multirow{7}{*}{\datasetbf{Dummy}}
& \method{HYPRO}    & $0$ & ${rmse_hypro:.3f} {\pm 0}$ & ${smape_hypro:.3f} {\pm 0}$ \\
& \method{CDiff}    & $0$ & ${rmse_cdiff:.3f} {\pm 0}$ & ${smape_cdiff:.3f} {\pm 0}$ \\
& \textbf{\FIMzeroshot} & $0$ & ${rmse_fimzeroshot:.3f} {\pm 0}$ & ${smape_fimzeroshot:.3f} {\pm 0}$ \\
& \textbf{\FIMfine} & $0$ & ${rmse_fimfine:.3f} {\pm 0}$ & ${smape_fimfine:.3f} {\pm 0}$ \\
\end{tabular}
        """
        base_rmse = {"HYPRO": 0.9, "CDiff": 0.8, "FIMzeroshot": 0.7, "FIMfine": 0.5}
        base_smape = {"HYPRO": 108, "CDiff": 107, "FIMzeroshot": 100, "FIMfine": 95}

        for n, fpath in HORIZON_FILES.items():
            scale = 1 + (n / 10.0)
            vals = {}
            for m in base_rmse:
                vals[f"rmse_{m.lower()}"] = base_rmse[m] * (scale - 0.5 + np.random.rand() * 0.2)
                vals[f"smape_{m.lower()}"] = base_smape[m] * (1 + (n / 100) + np.random.rand() * 0.01)
            with open(fpath, "w") as f:
                f.write(dummy_template.format(**vals))

    main()
