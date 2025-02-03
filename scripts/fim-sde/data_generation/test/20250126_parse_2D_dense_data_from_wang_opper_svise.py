import json
from copy import copy
from pathlib import Path


from fim import data_path
from fim.data_generation.sde.preprocess_utils import save_arrays_from_dict
from fim.utils.evaluation_sde import preprocess_gp_results, preprocess_system_data


if __name__ == "__main__":
    # set generation config
    data_json_path = Path("raw/SDE_2D_dense_trajectories_from_wang_opper_svise/sparse_gp_dense_observations.json")

    save_dir = Path("processed/test/20250126_2D_dense_data_from_wang_opper_svise")

    # prepare paths
    if not data_json_path.is_absolute():
        data_json_path = Path(data_path) / data_json_path

    if not save_dir.is_absolute():
        save_dir = Path(data_path) / save_dir

    all_data: list[dict] = json.load(open(data_json_path))

    # preprocess each system
    all_systems_data: dict = {
        all_data[i].get("name").replace(" ", "_"): preprocess_system_data(copy(all_data[i]), apply_diffusion_sqrt=True)
        for i in range(len(all_data))
    }

    for name, system_data in all_systems_data.items():
        save_arrays_from_dict(save_dir / "systems_data" / name, system_data)

    # results from gps
    all_results: dict = {all_data[i].get("name").replace(" ", "_"): preprocess_gp_results(all_data[i]) for i in range(len(all_data))}

    for name, system_data in all_results.items():
        save_arrays_from_dict(save_dir / "gp_results" / name, system_data)
