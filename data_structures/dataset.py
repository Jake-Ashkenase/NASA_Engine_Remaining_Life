import h5py
import torch
from torch.utils.data import Dataset


class RULDatasetNOT(Dataset):
    """
    Custom PyTorch Dataset for handling large HDF5 datasets efficiently.
    Loads data lazily to avoid memory issues.
    """

    def __init__(self, hdf5_file, transform=None, normalize=True):
        """
        Initializes the dataset.

        Parameters:
            hdf5_file (str): Path to the HDF5 file.
            transform (callable, optional): Optional transformation to apply.
            normalize (bool): Whether to normalize the data.
        """
        self.hdf5_file = hdf5_file
        self.transform = transform
        self.normalize = normalize

        # Open the file to read metadata but do not load data into memory
        with h5py.File(self.hdf5_file, "r", swmr=True, libver='latest') as f:
            self.data_size = f["X"].shape[0]  # Number of samples
            self.feature_shape = f["X"].shape[1:]  # (window_size, num_features)

            # Compute mean and std for normalization if required
            if self.normalize:
                self.mean = f["X"][:10000].mean(axis=(0, 1))  # Sampled mean
                self.std = f["X"][:10000].std(axis=(0, 1)) + 1e-8  # Sampled std (avoid div by zero)

    def __len__(self):
        """Returns total number of samples in the dataset."""
        return self.data_size

    def __getitem__(self, idx):
        """Loads and returns a sample (X, y) from the dataset lazily."""
        with h5py.File(self.hdf5_file, "r", swmr=True, libver='latest') as f:
            X = f["X"][idx]  # Shape: (window_size, num_features)
            y = f["y"][idx]  # Scalar target

        # Normalize data
        if self.normalize:
            X = (X - self.mean) / self.std

        # Apply any transformations (e.g., augmentations)
        if self.transform:
            X = self.transform(X)

        # Convert to PyTorch tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return X, y


class RULDatasetOld(Dataset):
    def __init__(self, hdf5_file, transform=None, normalize=True):
        self.hdf5_file = hdf5_file
        self.transform = transform
        self.normalize = normalize

        # Open file once, keep reference
        self.file = h5py.File(self.hdf5_file, "r", swmr=True)

        self.data_size = self.file["X"].shape[0]
        self.feature_shape = self.file["X"].shape[1:]

        if self.normalize:
            self.mean = self.file["X"][:10000].mean(axis=(0, 1))
            self.std = self.file["X"][:10000].std(axis=(0, 1)) + 1e-8

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        """Loads and returns a sample (X, y) lazily."""
        X = self.file["X"][idx]
        y = self.file["y"][idx]

        if self.normalize:
            X = (X - self.mean) / self.std

        if self.transform:
            X = self.transform(X)

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return X, y

    def __del__(self):
        """Ensures file is closed properly when dataset is deleted."""
        self.file.close()

class RULDataset(Dataset):
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file
        self.file = None  # Not opened yet

    def init_worker(self):
        self.file = h5py.File(self.hdf5_file, "r", swmr=True, libver='latest')
        self.X = self.file["X"]
        self.y = self.file["y"]
        # read shape, mean, std, etc.
        self.data_size = self.file["X"].shape[0]
        self.feature_shape = self.file["X"].shape[1:]

        if self.normalize:
            self.mean = self.file["X"][:10000].mean(axis=(0, 1))
            self.std = self.file["X"][:10000].std(axis=(0, 1)) + 1e-8

    def __getitem__(self, idx):
        X = self.file["X"][idx]
        y = self.file["y"][idx]

        if self.normalize:
            X = (X - self.mean) / self.std

        if self.transform:
            X = self.transform(X)

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return X, y

    def __del__(self):
        """Ensures file is closed properly when dataset is deleted."""
        self.file.close()


