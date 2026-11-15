"""
Check that the CDiff benchmark splits used for the FIM-PP multi-event results (Table 1) are in place.

The splits are the ones released by Zeng et al. (2024), "Interacting Diffusion Processes for Event Sequence
Forecasting" (https://github.com/networkslab/cdiff). Download them from the CDiff data folder
https://drive.google.com/drive/folders/1gT3fL5vJpLYPNtn9eGAbC7qrhmFgiyV0 and arrange them as

    data/external/CDiff_dataset/{taxi,taobao,stackoverflow,amazon,retweet}/{train,dev,test}.pkl

(StackOverflow is stored as "so/" in the CDiff folder; Amazon and Taobao are also in the CDiff GitHub repo under data/).

Usage:
    python scripts/hawkes/check_cdiff_data.py [--root data/external/CDiff_dataset]
"""

import argparse
import hashlib
import pickle
import sys
from pathlib import Path


# MD5 checksums of the files the paper numbers were computed on (the original CDiff release).
EXPECTED_MD5 = {
    "taxi": {
        "train": "7794824e3d9e299dc5ae0cde4468a808",
        "dev": "cb059f9bac3b5409be28dc0f2eb2af81",
        "test": "11d4729f0115890713096f60b2b59017",
    },
    "taobao": {
        "train": "fda2b7ae894a533aa6dd06895db03b56",
        "dev": "84aab6ce7712b8bdfb2f4fc295fc37ca",
        "test": "91d9b53bb0876c749f5ab7541f701a5c",
    },
    "stackoverflow": {
        "train": "d8cbeaa4cc3a3c9e70412d100096ee1b",
        "dev": "8559eea099f2a3eed971a1ca502fb1a7",
        "test": "3851a436627e76f12b7bd9e33518b8cb",
    },
    "amazon": {
        "train": "83555a36822774f63e68b79e9f8f8c7e",
        "dev": "b61a7db56cf9865fdb57571bc33cf85c",
        "test": "4dc1288df1a7241a6b6f592a2a9d2653",
    },
    "retweet": {
        "train": "bc4c1d1061f7fe54b0c6ded6a17a9ae9",
        "dev": "ea06059535ed18e02656ec85da11e4b8",
        "test": "8c633b624835b6f8cc38652133dda1ec",
    },
}

# Number of sequences per split and number of marks, for a quick sanity check of what was downloaded.
EXPECTED_COUNTS = {
    "taxi": (1400, 200, 400, 10),
    "taobao": (1300, 200, 500, 17),
    "stackoverflow": (1401, 401, 401, 22),
    "amazon": (5200, 300, 500, 16),
    "retweet": (6500, 731, 500, 3),
}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2] / "data" / "external" / "CDiff_dataset")
    args = ap.parse_args()

    ok = True
    for dataset, splits in EXPECTED_MD5.items():
        counts = []
        for split, expected in splits.items():
            path = args.root / dataset / f"{split}.pkl"
            if not path.exists():
                print(f"[MISSING]  {path}")
                ok = False
                continue
            with open(path, "rb") as f:
                d = pickle.load(f)
            counts.append(len(d[split]))
            if md5(path) != expected:
                print(f"[MISMATCH] {path}: md5 differs from the CDiff release used in the paper")
                ok = False
        if len(counts) == 3:
            n_train, n_dev, n_test, n_marks = EXPECTED_COUNTS[dataset]
            print(
                f"[{dataset}] sequences train/dev/test = {counts[0]}/{counts[1]}/{counts[2]} (expected {n_train}/{n_dev}/{n_test}), marks = {n_marks}"
            )
    print("All CDiff splits match the paper data." if ok else "Some CDiff splits are missing or differ, see above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
