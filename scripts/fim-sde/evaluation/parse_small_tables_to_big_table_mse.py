import os
import re

import numpy as np
from tabulate import tabulate

def parse_results(file_path):
    nested_dict = {}
    with open(file_path, 'r') as file:
        content = file.read()

    # Split content into sections for each dataset
    sections = content.split('Results for ')[1:]

    for section in sections:
        lines = section.strip().split('\n')
        dataset = lines[0].strip()

        # Find the start of the data table
        try:
            table_start = lines.index(next(line for line in lines if line.startswith('Model')))
        except StopIteration:
            continue

        # Extract data rows
        data_lines = lines[table_start+2:]
        models = {}
        for line in data_lines:
            if not line.strip():
                break  # End of current table
            parts = line.split()
            model = parts[0]
            mmd = float(parts[1])
            models[model] = mmd

        nested_dict[dataset] = models

    return nested_dict

def load_folder_path_results(folder_path, num_digits=3):
    # Find all files in folder
    file_paths = [f"{folder_path}/{file}" for file in os.listdir(folder_path) if file.endswith('.txt')]
    # Check if any files are empty, if yes remove them
    file_paths = [file for file in file_paths if os.path.getsize(file) > 0]
    results = []
    for file_path in file_paths:
        results.append(parse_results(file_path))
    mean_results = {}
    for dataset in results[0].keys():
        mean_results[dataset] = {}
        for model in results[0][dataset].keys():
            mmds = []
            for result in results:
                if model in result[dataset]:
                    mmds.append(result[dataset][model])
            mean = np.mean(mmds)
            std = np.std(mmds)
            
            # Round to three decimal places
            mean = round(mean, num_digits)
            std = round(std, num_digits)
            res_str = f"${mean} \pm {std}$"
            if len(mmds) < len(results):
                res_str += "*"*(len(results) - len(mmds)) # Stars denote missing values
            mean_results[dataset][model] = res_str
    return mean_results

# Example usage
if __name__ == "__main__":
    folder_paths = ['evaluations/mse/01311107/0.002', 'evaluations/mse/01311107/0.01', 'evaluations/mse/01311106/0.02']
    datasets = ['Double Well', 'Wang', 'Damped Linear', 'Damped Cubic', 'Duffing', 'Glycosis', 'Hopf']
    models = ["SparseGP", "BISDE", "FIM"]
    
    with open("evaluations/final_table_mse.txt", "w") as f:
        f.write("\\begin{tabular}{llllllllll}\n")
        f.write("$\\tau$ & Model & \\makecell{Double\\\\Well} & Wang & \\makecell{Damped\\\\Linear} & \\makecell{Damped\\\\Cubic} & Duffing & Glycolysis & Hopf\\\\\n")
        f.write("\hline\n")
        
        for folder_path in folder_paths:
            tau = folder_path.split('/')[-1]
            mean_results = load_folder_path_results(folder_path)
            for model in models:
                row = ""
                if model != "FIM":
                    row = "\\rowcolor{table_baselines}"
                row += f"{tau} & {model} & "
                for dataset in datasets:
                    if model in mean_results[dataset]:
                        row += f"{mean_results[dataset][model]} & "
                    else:
                        row += "N/A & "
                row = row[:-2] + "\\\\\n"
                f.write(row)
            f.write("\hline\n")
        f.write("\end{tabular}")
        