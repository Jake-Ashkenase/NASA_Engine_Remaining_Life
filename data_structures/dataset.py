from torch.utils.data import Dataset, DataLoader
import h5py
import torch
import numpy as np
from collections import defaultdict


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


def create_train_test_dataloaders(X, y, test_size=0.2, batch_size=64, shuffle=True, dim="1d",
                                  max_samples_per_class=None):
    """
    Split the DataFrame into train and test sets and create DataLoaders.
    Args:
        X (np.ndarray): Input features.
        y (np.ndarray): Target labels.
        test_size (float): Proportion of the data to be used as test data.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data.
        dim (str): Dimension type for the dataset.
        max_samples_per_class (int): Maximum samples to take for each unique y value.
    Returns:
        tuple: (train_loader, test_loader)
    """
    limited_X = X
    limited_y = y
    if max_samples_per_class is not None:
        # Collect indices for each unique y value
        class_samples = defaultdict(list)

        for idx in range(len(y)):
            class_samples[y[idx]].append(idx)  # Store the index of the sample

        # Create a list to hold the limited indices
        limited_indices = []

        for class_label, indices in class_samples.items():
            limited_indices.extend(indices[:max_samples_per_class])

        # Create a subset of the dataset with the limited indices
        limited_X = X[limited_indices]
        limited_y = y[limited_indices]

    # Shuffle the limited dataset if required
    if shuffle:
        shuffle_idxs = np.random.permutation(len(limited_y))
        limited_X = limited_X[shuffle_idxs]
        limited_y = limited_y[shuffle_idxs]

    # Split the limited dataset into train and test sets
    split_idx = int(len(limited_y) * (1 - test_size))
    X_train, X_test = limited_X[:split_idx], limited_X[split_idx:]
    y_train, y_test = limited_y[:split_idx], limited_y[split_idx:]

    # Create train and test datasets
    train_dataset = RULDataset(X_train, y_train, dim=dim)
    test_dataset = RULDataset(X_test, y_test, dim=dim)

    # Create DataLoaders for train and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
