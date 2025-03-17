from pathlib import Path

import torch


PATH = "/cephfs_projects/foundation_models/hawkes/data/2k_3_st_hawkes_mixed_no_powerlaw_2000_paths_250_events/test"

OUT_PATH = "/cephfs_projects/foundation_models/hawkes/data/2k_3_st_hawkes_mixed_no_powerlaw_2000_paths_250_events/test/"
OUT_PATH = Path(OUT_PATH)
in_file = Path(PATH) / "event_times.pt"


with open(in_file, "rb") as f:
    print(in_file)
    data = torch.load(f)
    delta_times = torch.diff(data, dim=-2)
    delta_times = torch.cat((torch.zeros_like(delta_times[..., :1, :]), delta_times), dim=-2)
    torch.save(delta_times, str(OUT_PATH) + "delta_times.pt")
