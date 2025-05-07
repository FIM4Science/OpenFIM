import json
from pprint import pprint

import numpy as np
import optree
import pandas as pd


bisde_json = "/Users/patrickseifner/sciebo/PhD/Notes/Projects/FIM-SDE/20250409_bisde_model_results_with_noise/20250505_bisde_results/20250505_bisde_results_all_systems_with_noise.json"
data_json = "/Users/patrickseifner/sciebo/PhD/Notes/Projects/FIM-SDE/20250409_bisde_model_results_with_noise/systems_observations_and_locations.json"

bisde_results: list[dict] = json.load(open(bisde_json, "r"))
bisde_results: list[dict] = [{k: np.array(v) if isinstance(v, list) else v for k, v in d.items()} for d in bisde_results]

all_data: list[dict] = json.load(open(data_json, "r"))
all_data: list[dict] = [{k: np.array(v) if isinstance(v, list) else v for k, v in d.items()} for d in all_data]


pprint(f"Result keys: {bisde_results[0].keys()}")
pprint(f"Shapes: {optree.tree_map(lambda x: x.shape if isinstance(x, np.ndarray) else x, bisde_results[0])}")
pprint(f"Length of list: {len(bisde_results)}")

# u = np.array([10, 5, 3.5])
# pprint(eval(bisde_results[0]["equations"]["Diffusion"]))


# Function to evaluate drift and diffusion
def evaluate_expression_1D(expression, u):
    """Evaluate the drift or diffusion expression for a given u."""
    try:
        return eval(expression)
    except Exception as e:
        print(f"Error evaluating expression '{expression}' for u={u}: {e}")
        return 0


metrics = []

for i, result in enumerate(bisde_results):
    synthetic_paths = result["synthetic_paths"]  # [5, 100, 500, D]
    drift = result["drift_at_locations"]  # [5, L, D]
    diffusion = result["diffusion_at_locations"]  # [5, L, D]
    D = synthetic_paths.shape[-1]

    # extract data where result was obtained from
    data_of_result = [d for d in all_data if (d["name"] == result["name"] and d["tau"] == result["tau"] and d["noise"] == result["noise"])]
    assert len(data_of_result) == 1
    data_of_result = data_of_result[0]

    locations = data_of_result["locations"]  # [5, L, D]

    for i in range(5):
        drift_of_exp = drift[i]
        diffusion_of_exp = diffusion[i]
        paths_of_exp = synthetic_paths[i]

        metrics.append({"metric": "Drift NaN at some location", "Percentage of Equations": np.isnan(drift_of_exp).any().astype(float)})
        metrics.append(
            {
                "metric": "Percentage of locations where drift NaN",
                "Percentage of Equations": np.isnan(drift_of_exp).mean().astype(float),
            }
        )
        metrics.append(
            {
                "metric": "Diffusion (under sqrt!) Nan at some location",
                "Percentage of Equations": np.isnan(diffusion_of_exp).any().astype(float),
            }
        )
        metrics.append(
            {
                "metric": "Path NaN at some point",
                "Percentage of Equations": np.isnan(paths_of_exp).any(axis=(-1, -2)).mean().astype(float),
            }
        )
        metrics.append(
            {
                "metric": "Diffusion (under sqrt!) negative at some location",
                "Percentage of Equations": (diffusion_of_exp < 0.0).any().astype(float),
            }
        )
        metrics.append(
            {
                "metric": "Path NaN and Diffusion (under sqrt!) negative at some location",
                "Percentage of Equations": (np.isnan(paths_of_exp).any() and (diffusion_of_exp < 0.0).any()).astype(float),
            }
        )

    # diffusion_equation = result["equations"]["Diffusion"]
    #
    # if np.isnan(diffusion).any():
    #     print(f"Diffusion {i} is Nan at locations: ", diffusion_equation)
    #     nan_locations = locations[np.isnan(diffusion).any(axis=-1)]
    #     print("At locations: ", nan_locations[:5])
    #     print("\n")
    #
    # # print((diffusion < 0.0).mean())
    #
    # # if D == 1:
    #

df = pd.DataFrame(metrics)
df = df.groupby("metric").mean()

pprint(df)
