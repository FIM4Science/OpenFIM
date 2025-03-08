from pathlib import Path

import torch


PATH = "/cephfs_projects/foundation_models/hawkes/data/2k_3_st_hawkes_mixed_no_powerlaw_2000_paths_250_events/train/"

OUT_PATH = "/home/cvejoski/Projects/FoundationModels/FIM/data/interim/hawkes/1k_3_st_hawkes_mixed_no_powerlaw_300_paths_10_events/train"
OUT_PATH = Path(OUT_PATH)
OUT_PATH.mkdir(parents=True, exist_ok=True)
for file in Path(PATH).rglob("*.pt"):
    with open(file, "rb") as f:
        print(file)
        # data = pickle.load(f)
        data = torch.load(f)
        B = 1000  # Set the limit for the first dimension
        P = 300  # Set the limit for the second dimension
        # limited_data = {}
        # for k, v in data.items():
        #     limited_data[k] = torch.clone(v[:B])
        if data.dim() > 2:
            limited_data = torch.clone(data[:B, :P, :10])
        else:
            limited_data = torch.clone(data[:B, :P])
        # limited_data = data[:B]
        # limited_data = torch.from_dlpack(asdlpack(limited_data))
        # for k, v in limited_data.items():
        #     print(k, v.shape)
        torch.save(limited_data, str(OUT_PATH) + "/" + file.stem + ".pt")
        # with open(str(OUT_PATH) + "/" + file.name, "wb") as f_out:
        # pickle.dump(limited_data, f_out)
