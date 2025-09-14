import argparse
import pathlib
from typing import Dict

import h5py
import torch


def save_h5(tensor: torch.Tensor, path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=tensor.cpu().numpy(), compression="gzip")


def convert_dir(pt_dir: pathlib.Path, mapping: Dict[str, str], overwrite: bool = False):
    for key, filename in mapping.items():
        pt_path = pt_dir / filename
        if not pt_path.exists():
            # tolerate missing optional files
            continue
        tensor = torch.load(pt_path, weights_only=True, map_location="cpu")
        h5_path = pt_path.with_suffix(".h5")
        if h5_path.exists() and not overwrite:
            continue
        save_h5(tensor, h5_path)


def main():
    parser = argparse.ArgumentParser(description="Convert Hawkes .pt data files to .h5 files per directory.")
    parser.add_argument("--dirs", nargs="+", required=True, help="One or more directories to convert")
    parser.add_argument(
        "--files",
        type=str,
        default="base_intensity_functions.pt,event_times.pt,event_types.pt,kernel_functions.pt,time_offsets.pt",
        help="Comma-separated list of expected .pt files (filenames). Missing ones are skipped.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .h5 files")
    args = parser.parse_args()

    filenames = args.files.split(",")
    mapping = {fn.replace(".pt", ""): fn for fn in filenames}

    for d in args.dirs:
        convert_dir(pathlib.Path(d), mapping, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
