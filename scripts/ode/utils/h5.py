import h5py
from pathlib import Path


def get_shape_of_h5(path: str):
    """
    This function assumes the .h5 file has only one key; "data".
    This is the case for our training and inference .h5's.
    """

    with h5py.File(path, "r") as f:
        data = f["data"][:]
        _ = data.shape
    
    return _


def get_dtype_of_h5(path: str):
    """
    This function assumes the .h5 file has only one key; "data".
    This is the case for our training and inference .h5's.
    """

    with h5py.File(path, "r") as f:
        data = f["data"][:]
        _ = data.dtype
    
    return _


def get_type_of_h5(path: str):
    """
    This function assumes the .h5 file has only one key; "data".
    This is the case for our training and inference .h5's.
    """

    with h5py.File(path, "r") as f:
        data = f["data"][:]
        _ = type(data)
    
    return _


def parse_h5(path: str):
    """
    This function assumes the .h5 file has only one key; "data".
    This is the case for our training and inference .h5's.
    """

    with h5py.File(path, "r") as f:
        data = f["data"][:]
    
    return data


def print_h5(path: str):
    """
    This function assumes the .h5 file has only one key; "data".
    This is the case for our training and inference .h5's.
    """

    print(parse_h5(path))


def get_ndarray_from_h5(path_to_h5: Path):
    with h5py.File(path_to_h5, "r") as f:
        return f["data"][:]


if __name__ == "__main__":
    path = Path("experiments/vdp1/data_gpode/locations.h5")
    a = get_ndarray_from_h5(path)
    print(a.shape)