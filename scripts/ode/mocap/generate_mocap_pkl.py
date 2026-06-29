"""
Generate mocap_dataset.pkl and h5 files for all (subject, variant) combinations.

Reads the raw .npz files from data/ode/mocap/ and writes to
data/ode/mocap/subject_XX/<variant>/:
  - mocap_dataset.pkl       — pickled MocapDataset (train/val/test)
  - obs_values.h5['data']   — (1, n_traj, seqlen, 5)  train trajectories
  - obs_times.h5['data']    — (1, n_traj, seqlen, 1)  time grid repeated per traj
  - obs_mask.h5['data']     — (1, n_traj, seqlen, 1)  all-ones (fully observed)
  - locations.h5['data']    — (n_traj, 2, 5)           first two timesteps per traj

Usage:
    python scripts/ode/mocap/generate_mocap_pkl.py

PCA: 5 components, pca_normalize=True, data_normalize=False  (matches FIMODE training).
"""
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent.parent

# data_gen_mocap must be importable for pickle to store the right class path
sys.path.insert(0, str(_HERE))
from data_gen_mocap import MocapDataset  # noqa: E402

NPZ_DIR = _ROOT / "data" / "mocap"   # mocap09.npz, mocap35.npz, mocap39.npz
OUT_DIR = _ROOT / "data" / "mocap"

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


def _save_h5(path: Path, data: np.ndarray) -> None:
    with h5py.File(path, "w") as hf:
        hf.create_dataset("data", data=data)


def save_h5_files(dataset: MocapDataset, out_dir: Path) -> None:
    """Write obs_values/times/mask/locations h5 files for the train split."""
    trn_ys = dataset.trn.ys                          # (n_traj, seqlen, 5)
    n_traj = trn_ys.shape[0]

    obs_values = trn_ys[np.newaxis]                  # (1, n_traj, seqlen, 5)
    obs_times  = (dataset.trn.ts[np.newaxis, np.newaxis, :, np.newaxis]
                  .repeat(n_traj, axis=1))            # (1, n_traj, seqlen, 1)
    obs_mask   = np.ones((*obs_values.shape[:-1], 1),
                         dtype=bool)                  # (1, n_traj, seqlen, 1)
    locations  = trn_ys[:, :2, :]                    # (n_traj, 2, 5)

    _save_h5(out_dir / "obs_values.h5", obs_values)
    _save_h5(out_dir / "obs_times.h5",  obs_times)
    _save_h5(out_dir / "obs_mask.h5",   obs_mask)
    _save_h5(out_dir / "locations.h5",  locations)


def main():
    for subject, variant, seqlen in CONFIGS:
        out_dir  = OUT_DIR / f"subject_{subject}" / variant
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"subject {subject} / {variant:5s}  seqlen={seqlen:3d}  → {out_dir.relative_to(_ROOT)}", end=" ... ", flush=True)

        dataset = MocapDataset(
            data_path=str(NPZ_DIR),
            subject=subject,
            dt=0.01,
            pca_components=5,
            seqlen=seqlen,
            data_normalize=False,
            pca_normalize=True,
        )

        with open(out_dir / "mocap_dataset.pkl", "wb") as fh:
            pickle.dump(dataset, fh)

        save_h5_files(dataset, out_dir)

        print(f"trn={dataset.trn.ys.shape}  val={dataset.val.ys.shape}  tst={dataset.tst.ys.shape}")

    print("\nDone.")


if __name__ == "__main__":
    main()
