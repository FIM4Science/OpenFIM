import argparse
import pathlib
from typing import Dict

import h5py
import torch


def save_h5(tensor: torch.Tensor, path: pathlib.Path, chunk_rows: int = 64, compression: str | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = tensor.cpu().numpy()
    chunks = None
    # row-chunking for fast slicing along first dim
    if arr.ndim >= 1 and arr.shape[0] > 0:
        chunks = (min(chunk_rows, arr.shape[0]),) + tuple(arr.shape[1:])
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=arr, chunks=chunks, compression=compression)


def convert_dir(pt_dir: pathlib.Path, mapping: Dict[str, str], overwrite: bool = False, compression: str | None = None):
    for key, filename in mapping.items():
        pt_path = pt_dir / filename
        if not pt_path.exists():
            # tolerate missing optional files
            continue
        tensor = torch.load(pt_path, weights_only=True, map_location="cpu")
        h5_path = pt_path.with_suffix(".h5")
        if h5_path.exists() and not overwrite:
            continue
        save_h5(tensor, h5_path, compression=compression)


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
    # Default to fast (no compression). Use --compression gzip to opt in.
    parser.add_argument(
        "--compression",
        choices=["none", "gzip"],
        default="none",
        help="Compression for h5 datasets. Default: none (fast).",
    )
    # Backward-compat flag; if provided, behaves like --compression none
    parser.add_argument("--fast", action="store_true", help="Deprecated: alias for --compression none")
    args = parser.parse_args()

    filenames = args.files.split(",")
    mapping = {fn.replace(".pt", ""): fn for fn in filenames}

    # Resolve compression preference: --fast overrides to none; otherwise use --compression
    compression = None if (args.fast or args.compression == "none") else "gzip"
    for d in args.dirs:
        convert_dir(pathlib.Path(d), mapping, overwrite=args.overwrite, compression=compression)


if __name__ == "__main__":
    main()
