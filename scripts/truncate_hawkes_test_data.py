from pathlib import Path

import torch


def truncate_first_dim(obj, n=10):
    if isinstance(obj, torch.Tensor):
        if obj.ndim >= 1 and obj.size(0) > n:
            return obj[:n].contiguous()
        return obj
    if isinstance(obj, (list, tuple)):
        return type(obj)(truncate_first_dim(x, n) for x in obj)
    if isinstance(obj, dict):
        return {k: truncate_first_dim(v, n) for k, v in obj.items()}
    return obj


def process_dir(dir_path: Path, n: int = 10):
    print(f"Processing {dir_path}")
    files = [
        "base_intensity_functions.pt",
        "kernel_functions.pt",
        "event_times.pt",
        "event_types.pt",
        "seq_lengths.pt",
    ]
    for fname in files:
        f = dir_path / fname
        if not f.exists():
            print(f"  Skip missing {f}")
            continue
        print(f"  Loading {f}")
        obj = torch.load(f, map_location="cpu")
        new_obj = truncate_first_dim(obj, n)
        if new_obj is obj:
            print(f"  No change needed for {f}")
        else:
            torch.save(new_obj, f)
            print(f"  Truncated and saved {f}")


def main():
    root = Path(__file__).resolve().parents[1] / "tests" / "resources" / "data" / "hawkes"
    dirs = [root / "3D_hawkes_test_data", root / "10D_hawkes_test_data"]
    for d in dirs:
        process_dir(d, n=10)


if __name__ == "__main__":
    main()
