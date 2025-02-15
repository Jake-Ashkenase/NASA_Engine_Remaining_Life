from torch.utils.data import Dataset, DataLoader
import h5py
import torch
import numpy as np


class RULDataset(Dataset):
    def __init__(self, X, y, dim="1d"):
        """
        Initialize the dataset by passing a DataFrame.
        Args:
            dataframe (pd.DataFrame): DataFrame with pixel data and labels.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.dim = dim

    def __len__(self):
        """
        Return the total number of samples in the dataset.
        """
        return len(self.y)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            (torch.Tensor, torch.Tensor): Tuple of features and label.
        """
        X_sample = self.X[idx]
        y_sample = self.y[idx]

        if self.dim == "2d":
            X_sample = X_sample.unsqueeze(0)

        return X_sample, y_sample


def create_train_test_dataloaders(X, y, test_size=0.2, batch_size=64, shuffle=True, dim="1d"):
    """
    Split the DataFrame into train and test sets and create DataLoaders.
    Args:
        dataframe (pd.DataFrame): DataFrame containing the data.
        test_size (float): Proportion of the data to be used as test data.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data.
    Returns:
        tuple: (train_loader, test_loader)
    """
    shuffle_idxs = np.random.permutation(len(y))
    X_shuffled = X[shuffle_idxs]
    y_shuffled = y[shuffle_idxs]

    split_idx = int(len(y) * test_size)
    X_test = X_shuffled[:split_idx]
    X_train = X_shuffled[split_idx:]
    y_test = y[:split_idx]
    y_train = y[:split_idx]

    # Create train and test datasets
    train_dataset = RULDataset(X_train, y_train, dim=dim)
    test_dataset = RULDataset(X_test, y_test, dim=dim)

    # Create DataLoaders for train and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def load_from_hdf5(filename="data.h5"):
    with h5py.File(filename, "r") as f:
        X = f["X"][:]
        y = f["y"][:]
    return X, y
