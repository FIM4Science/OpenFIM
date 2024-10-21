import pickle
from pathlib import Path

import torch
from dlpack import asdlpack


PATH = "/cephfs_projects/foundation_models/MJP/data/25k_hom_mjp_6_st_10s_1%_noise_rand_300-samples-per-intensity_with_initial_distribution/test/test/"

OUT_PATH = "/home/cvejoski/Projects/FoundationModels/FIM/tests/resources/data/mjp/25k_hom_mjp_6_st_10s_1_noise_rand_300-samples-per-intensity_with_initial_distribution/test/"
OUT_PATH = Path(OUT_PATH)
OUT_PATH.mkdir(parents=True, exist_ok=True)
for file in Path(PATH).rglob("*.pickle"):
    with open(file, "rb") as f:
        data = pickle.load(f)
        B = 2  # Set the limit for the first dimension
        limited_data = data[:B]
        limited_data = torch.from_dlpack(asdlpack(limited_data))
        torch.save(limited_data, str(OUT_PATH) + "/" + file.stem + ".pt")
        # with open(str(OUT_PATH) + "/" + file.name, "wb") as f_out:
        #     pickle.dump(limited_data, f_out)
