from pathlib import Path

import torch


PATH = "/cephfs_projects/foundation_models/data/torch_2M_ode_chebyshev_max_deg_100_rbf_gp_2_5_and_2_10_length_128_avg_min_8/"

OUT_PATH = "/home/cvejoski/Projects/FoundationModels/FIM/tests/resources/data/ode/torch_2M_ode_chebyshev_max_deg_100_rbf_gp_2_5_and_2_10_length_128_avg_min_8/"
OUT_PATH = Path(OUT_PATH)
OUT_PATH.mkdir(parents=True, exist_ok=True)
for file in Path(PATH).rglob("*.pt"):
    with open(file, "rb") as f:
        print(file)
        data = torch.load(f)
        B = 2  # Set the limit for the first dimension
        limited_data = {}
        for k, v in data.items():
            limited_data[k] = torch.clone(v[:B])
        # limited_data = torch.from_dlpack(asdlpack(limited_data))
        for k, v in limited_data.items():
            print(k, v.shape)
        torch.save(limited_data, str(OUT_PATH) + "/" + file.stem + ".pt")
        # with open(str(OUT_PATH) + "/" + file.name, "wb") as f_out:
        #     pickle.dump(limited_data, f_out)
