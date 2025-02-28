import csv

from fim.data.data_generation.dynamical_systems import Polynomials


params = {
    "name": "Polynomials",
    "data_bulk_name": "25k_train_dim_1_clean_chunk_0",
    "num_realizations": 500_000,
    "observed_dimension": None,
    "state_dim": 1,
    "redo": False,
    "enforce_positivity": "clip",
    "max_degree_drift": 3,
    "max_degree_diffusion": 2,
    "drift_params": {"uniform_degrees": True, "distribution": {"name": "normal", "std": 1}},
    "diffusion_params": {"uniform_degrees": True, "distribution": {"name": "normal", "std": 1.0}},
    "initial_state": {"distribution": "normal", "mean": 0.0, "std_dev": 1.0, "activation": None},
}
output_path = f"/home/cvejoski/Projects/FoundationModels/Wiener-Procs-FM/data/state_sde/expressions-pool/dimension_{params['state_dim']}/{params['state_dim']}d-polynomials-pool-diffusion.csv"
polynomial_system = Polynomials(params)
diffusions = polynomial_system.sample_diffusion_params(params["num_realizations"])
equations = polynomial_system.print_polynomials(diffusions, polynomial_system.max_degree_diffusion, for_export=True)

with open(output_path, "w", newline="") as file:
    writer = csv.writer(file, delimiter="\t")
    writer.writerows(equations)
