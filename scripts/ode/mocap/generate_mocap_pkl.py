"""
Generate mocap_dataset.pkl files for all (subject, variant) combinations.

Reads the raw .npz files from scripts/ode/mocap/ and writes pickled
MocapDataset objects to data/mocap/subject_XX/<variant>/mocap_dataset.pkl.

Usage:
    python scripts/ode/mocap/generate_mocap_pkl.py

Variants:
    short — seqlen=50  (0.5 s context at dt=0.01)
    long  — seqlen=T   (full training trajectory)

PCA: 5 components, pca_normalize=True, data_normalize=False  (matches FIMODE training).
"""
import pickle
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent.parent

# data_gen_mocap must be importable for pickle to store the right class path
sys.path.insert(0, str(_HERE))
from data_gen_mocap import MocapDataset  # noqa: E402

NPZ_DIR  = _ROOT / "data" / "mocap"       # mocap09.npz, mocap35.npz, mocap39.npz
OUT_DIR  = _ROOT / "data" / "mocap"

# seqlen values from Table 2, supplementary material (arxiv:2106.10905).
# Columns: Subject | Task | Train length | Validation length | Test length
# Only train length is seqlen; val/tst always use their full lengths.
CONFIGS = [
    # (subject, variant, seqlen)
    ("09", "short",  50),
    ("09", "long",  100),
    ("35", "short",  50),
    ("35", "long",  250),
    ("39", "short", 100),
    ("39", "long",  250),
]


def main():
    for subject, variant, seqlen in CONFIGS:
        out_path = OUT_DIR / f"subject_{subject}" / variant / "mocap_dataset.pkl"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"subject {subject} / {variant:5s}  seqlen={seqlen:3d}  → {out_path.relative_to(_ROOT)}", end=" ... ", flush=True)

        dataset = MocapDataset(
            data_path=str(NPZ_DIR),
            subject=subject,
            dt=0.01,
            pca_components=5,
            seqlen=seqlen,
            data_normalize=False,
            pca_normalize=True,
        )

        with open(out_path, "wb") as fh:
            pickle.dump(dataset, fh)

        print(f"trn={dataset.trn.ys.shape}  val={dataset.val.ys.shape}  tst={dataset.tst.ys.shape}")

    print("\nDone. All pkl files written to data/mocap/")


if __name__ == "__main__":
    main()
