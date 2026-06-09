from pathlib import Path
import h5py
import pickle
from data_gen_mocap import Normalize, Data, MocapDataset


# 5D
base_path = Path("experiments/mocap")
for task in ["mocap09short", "mocap09long", "mocap35short", "mocap35long", "mocap39short", "mocap39long"]:
    print(task)
    for split in ["train", "test", "valid"]:
        data_path = base_path / task / "data" / split / "5d"
        print(split)

        for file_name in ["obs_values.h5", "obs_times.h5", "obs_mask.h5", "locations.h5"]:
            with h5py.File(data_path / file_name, "r") as f:
                print(file_name)
                print(f["data"][:].shape)

        print()

    # now for the .pkl (only exists in 5d directory)
    with open(data_path / "mocap_dataset.pkl", 'rb') as f:
        print("mocap_dataset.pkl")
        mocap_dataset: MocapDataset = pickle.load(f)

        print("trn.ys")
        print(mocap_dataset.trn.ys.shape)

        print("tst.ys")
        print(mocap_dataset.tst.ys.shape)

        print("val.ys")
        print(mocap_dataset.val.ys.shape)
    
    print()
    print()
