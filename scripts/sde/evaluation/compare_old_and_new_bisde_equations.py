import itertools
import json
from functools import partial
from pathlib import Path
from pprint import pprint

import numpy as np
import optree
import pandas as pd


def get_statistics(results: dict, systems: list[str], exps_count: int, taus: list[float]):
    metrics = []

    for system, tau, exp in itertools.product(systems, taus, range(exps_count)):
        for eval_label, eval_results in results.items():
            all_results = eval_results["2D_results"]
            selected_results = [d for d in all_results if d["name"] == system and d["tau"] == tau]
            assert len(selected_results) == 1
            selected_results = selected_results[0]

            paths = np.array(selected_results["synthetic_paths"])[exp]
            drift = np.array(selected_results["drift_at_locations"])[exp]
            diffusion = np.array(selected_results["diffusion_at_locations"])[exp]

            metrics.append(
                {
                    "eval": eval_label,
                    "metric": "Drift NaN at some location",
                    "Percentage of Equations": np.isnan(drift).any().astype(float),
                }
            )
            metrics.append(
                {
                    "eval": eval_label,
                    "metric": "Percentage of locations where drift NaN",
                    "Percentage of Equations": np.isnan(drift).mean().astype(float),
                }
            )
            metrics.append(
                {
                    "eval": eval_label,
                    "metric": "Diffusion (under sqrt!) Nan at some location",
                    "Percentage of Equations": np.isnan(diffusion).any().astype(float),
                }
            )
            metrics.append(
                {
                    "eval": eval_label,
                    "metric": "Path NaN at some point",
                    "Percentage of Equations": np.isnan(paths).any(axis=(-1, -2)).mean().astype(float),
                }
            )
            metrics.append(
                {
                    "eval": eval_label,
                    "metric": "Diffusion (under sqrt!) negative at some location",
                    "Percentage of Equations": (diffusion < 0.0).any().astype(float),
                }
            )
            metrics.append(
                {
                    "eval": eval_label,
                    "metric": "Path NaN and Diffusion (under sqrt!) negative at some location",
                    "Percentage of Equations": (np.isnan(paths).any() and (diffusion < 0.0).any()).astype(float),
                }
            )

    df = pd.DataFrame(metrics)
    df = df.groupby(["eval", "metric"]).mean()
    df = df.unstack(0)

    pprint(df)


def print_vector_field_equations(results: dict, systems: list[str], exps_count: int, taus: list[float]):
    for system, tau, exp in itertools.product(systems, taus, range(exps_count)):
        print(f"{system=}, {tau=}, {exp=}\n")
        print("Drift=")
        for eval_label, eval_results in results.items():
            if (all_eqs := eval_results.get("2D_equations")) is not None:
                print(f"{eval_label=}\n")
                selected_exp_eqs = all_eqs[system][f"{tau}_{exp}"]

                drift_eq = [selected_exp_eqs[0]["Drift"], selected_exp_eqs[1]["Drift"]]
                print(drift_eq[0])
                print(drift_eq[1])
                print("\n")

        print("Diffusion=\n")
        for eval_label, eval_results in results.items():
            if (all_eqs := eval_results.get("2D_equations")) is not None:
                print(f"{eval_label=}")
                selected_exp_eqs = all_eqs[system][f"{tau}_{exp}"]

                diffusion_eq = [selected_exp_eqs[0]["Diffusion"], selected_exp_eqs[1]["Diffusion"]]
                print(diffusion_eq[0])
                print(diffusion_eq[1])
                print("\n")

        print("\n")


def estimate_function_library(results: dict):
    for eval_label, eval_results in results.items():
        for eq_dim_key in ["1D_equations", "2D_equations"]:
            if (equations := eval_results.get(eq_dim_key)) is not None:
                # get equations as list
                equations: list[str] = optree.tree_flatten(equations)[0]

                # split at + or - to get individual summands
                summands: list[list] = optree.tree_map(lambda x: x.replace("-", "+").split("+"), equations)
                summands = list(itertools.chain.from_iterable(summands))
                summands = [s for s in summands if s != ""]

                # removes factors from summands
                all_basis_vectors = [s.split("*", 1)[1] for s in summands if len(s.split("*", 1)) > 1]

                # remove duplicates
                all_basis_vectors = [b.replace(" ", "") for b in all_basis_vectors]
                basis_vectors = list(set(all_basis_vectors))
                basis_vectors.sort()

                print(f"Estimated function library for '{eq_dim_key}' of '{eval_label}':")
                print(basis_vectors)


if __name__ == "__main__":
    data_json = "/Users/patrickseifner/sciebo/PhD/Notes/Projects/FIM-SDE/20250409_bisde_model_results_with_noise/systems_observations_and_locations.json"

    jsons: dict = {
        # "Kosta (wrong parsing of functions)": {
        #     "1D_equations": Path(
        #         "/Users/patrickseifner/sciebo/PhD/Notes/Projects/FIM-SDE/20250409_bisde_model_results_with_noise/20250508_bisde_results_with_equations/all_results_1d.json"
        #     ),
        #     "2D_equations": Path(
        #         "/Users/patrickseifner/sciebo/PhD/Notes/Projects/FIM-SDE/20250409_bisde_model_results_with_noise/20250508_bisde_results_with_equations/all_results_2d.json"
        #     ),
        #     "2D_results": Path(
        #         "/Users/patrickseifner/sciebo/PhD/Notes/Projects/FIM-SDE/20250409_bisde_model_results_with_noise/20250508_bisde_results_with_equations/bisde_2d_results.json"
        #     ),
        # },
        # "Kosta (corrected)": {
        #     # "2D_equations": Path(""),
        #     "2D_results": Path(
        #         "/Users/patrickseifner/sciebo/PhD/Notes/Projects/FIM-SDE/20250409_bisde_model_results_with_noise/20250509_bisde_results_with_fixed_parsing_of_equations/bisde_2d_results.json"
        #     ),
        # },
        # "Kosta (multiple diffusion summands)": {
        #     "2D_results": Path(
        #         "/Users/patrickseifner/sciebo/PhD/Notes/Projects/FIM-SDE/20250409_bisde_model_results_with_noise/20250510_bisde_results_with_multiple_diffusion_summands/bisde_2d_results.json"
        #     ),
        # },
        "Kosta (multiple diffusion summands, with clipping)": {
            "2D_results": Path(
                "/Users/patrickseifner/sciebo/PhD/Notes/Projects/FIM-SDE/20250409_bisde_model_results_with_noise/20250511_bisde_results_with_multiple_diffusion_summands_with_diffusion_clipping/bisde_2d_clip_results.json"
            ),
        },
        "Cesar": {
            "1D_equations": Path(
                "/Users/patrickseifner/sciebo/PhD/Notes/Projects/FIM-SDE/20250409_bisde_model_results_with_noise/202501_bisde_equations_and_code_for_ICML_submission/results_2d_updated/results_1d.json"
            ),
            "2D_equations": Path(
                "/Users/patrickseifner/sciebo/PhD/Notes/Projects/FIM-SDE/20250409_bisde_model_results_with_noise/202501_bisde_equations_and_code_for_ICML_submission/results_2d_updated/results_2d_updated.json"
            ),
            "2D_results": Path(
                "/Users/patrickseifner/sciebo/PhD/Notes/Projects/FIM-SDE/20250409_bisde_model_results_with_noise/202501_bisde_equations_and_code_for_ICML_submission/bisde_experiments_friday_full.json"
            ),
        },
    }

    def _load_json(path: Path, systems: list, taus: list, exps_count: int):
        content = json.load(open(path, "r"))

        # experiments without noise
        valid_keys_0 = [f"{tau}_{exp}.csv" for tau in taus for exp in range(exps_count)]
        valid_keys_1 = [f"{tau}_0.0_{exp}.csv" for tau in taus for exp in range(exps_count)]
        valid_keys_2 = [f"{tau}_{exp}" for tau in taus for exp in range(exps_count)]
        valid_keys_3 = [f"{tau}_0.0_{exp}" for tau in taus for exp in range(exps_count)]
        valid_keys = valid_keys_0 + valid_keys_1 + valid_keys_2 + valid_keys_3

        # equations
        if isinstance(content, dict):
            # unify equation names
            content = {k.replace(" ", "_").lower(): v for k, v in content.items()}

            # select experiments without noise
            content = {
                system: {k.replace("_0.0", "").strip(".csv"): v for k, v in eqs.items() if k in valid_keys}
                for system, eqs in content.items()
            }

        # results
        if isinstance(content, list):
            # unify equation names
            content = [d | {"name": d["name"].replace(" ", "_").lower()} for d in content]

            # select experiments without noise
            content = [d for d in content if "noise" not in list(d.keys()) or d["noise"] == 0.0]

        return content

    systems = [
        "damped_linear",
        "damped_cubic",
        "duffing",
        "glycosis",
        "hopf",
        "syn_drift",
        "wang",
    ]
    exps_count = 5
    taus = [0.002, 0.01, 0.02]

    results = optree.tree_map(partial(_load_json, systems=systems, taus=taus, exps_count=exps_count), jsons)

    # print_vector_field_equations(results, systems, exps_count, taus)
    get_statistics(results, systems, exps_count, taus)
    # estimate_function_library(results)

    # all_data: list[dict] = json.load(open(data_json, "r"))
    # all_data: list[dict] = [{k: np.array(v) if isinstance(v, list) else v for k, v in d.items()} for d in all_data]
    #
    # pprint(f"Result keys: {bisde_results[0].keys()}")
    # pprint(f"Shapes: {optree.tree_map(lambda x: x.shape if isinstance(x, np.ndarray) else x, bisde_results[0])}")
    # pprint(f"Length of list: {len(bisde_results)}")
    #
    # # u = np.array([10, 5, 3.5])
    # # pprint(eval(bisde_results[0]["equations"]["Diffusion"]))
    #
    # # Function to evaluate drift and diffusion
    # def evaluate_expression_1D(expression, u):
    #     """Evaluate the drift or diffusion expression for a given u."""
    #     try:
    #         return eval(expression)
    #     except Exception as e:
    #         print(f"Error evaluating expression '{expression}' for u={u}: {e}")
    #         return 0
    #
    # metrics = []
    #
    # for i, result in enumerate(bisde_results):
    #     synthetic_paths = result["synthetic_paths"]  # [5, 100, 500, D]
    #     drift = result["drift_at_locations"]  # [5, L, D]
    #     diffusion = result["diffusion_at_locations"]  # [5, L, D]
    #     D = synthetic_paths.shape[-1]
    #
    #     # extract data where result was obtained from
    #     data_of_result = [
    #         d for d in all_data if (d["name"] == result["name"] and d["tau"] == result["tau"] and d["noise"] == result["noise"])
    #     ]
    #     assert len(data_of_result) == 1
    #     data_of_result = data_of_result[0]
    #
    #     locations = data_of_result["locations"]  # [5, L, D]
    #
    #     for i in range(5):
    #         drift_of_exp = drift[i]
    #         diffusion_of_exp = diffusion[i]
    #         paths_of_exp = synthetic_paths[i]
    #
    #         metrics.append({"metric": "Drift NaN at some location", "Percentage of Equations": np.isnan(drift_of_exp).any().astype(float)})
    #         metrics.append({
    #             "metric": "Percentage of locations where drift NaN",
    #             "Percentage of Equations": np.isnan(drift_of_exp).mean().astype(float),
    #         })
    #         metrics.append({
    #             "metric": "Diffusion (under sqrt!) Nan at some location",
    #             "Percentage of Equations": np.isnan(diffusion_of_exp).any().astype(float),
    #         })
    #         metrics.append({
    #             "metric": "Path NaN at some point",
    #             "Percentage of Equations": np.isnan(paths_of_exp).any(axis=(-1, -2)).mean().astype(float),
    #         })
    #         metrics.append({
    #             "metric": "Diffusion (under sqrt!) negative at some location",
    #             "Percentage of Equations": (diffusion_of_exp < 0.0).any().astype(float),
    #         })
    #         metrics.append({
    #             "metric": "Path NaN and Diffusion (under sqrt!) negative at some location",
    #             "Percentage of Equations": (np.isnan(paths_of_exp).any() and (diffusion_of_exp < 0.0).any()).astype(float),
    #         })
    #
    #     # diffusion_equation = result["equations"]["Diffusion"]
    #     #
    #     # if np.isnan(diffusion).any():
    #     #     print(f"Diffusion {i} is Nan at locations: ", diffusion_equation)
    #     #     nan_locations = locations[np.isnan(diffusion).any(axis=-1)]
    #     #     print("At locations: ", nan_locations[:5])
    #     #     print("\n")
    #     #
    #     # # print((diffusion < 0.0).mean())
    #     #
    #     # # if D == 1:
    #     #
    #
    # df = pd.DataFrame(metrics)
    # df = df.groupby("metric").mean()
    #
    # pprint(df)
