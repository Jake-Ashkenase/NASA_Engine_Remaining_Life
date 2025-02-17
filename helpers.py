import h5py

def load_from_hdf5(filename="data.h5"):
    with h5py.File(filename, "r") as f:
        X = f["X"][:]
        y = f["y"][:]
    return X, y