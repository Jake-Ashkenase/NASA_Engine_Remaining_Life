from torch.utils.data import Dataset, DataLoader
import h5py
import torch
import numpy as np
from collections import defaultdict


class RULDatasetOld(Dataset):
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


class RULDataset(Dataset):
    def __init__(self, X, y, dim="1d", multitask=False):
        """
        Initialize the dataset.
        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Target labels. For multi-task mode, y must have shape [N,2]
                          (column 0: RUL, column 1: health label).
            dim (str): "1d" or "2d". For "2d", unsqueeze a channel dimension.
            multitask (bool): If True, the dataset is used for a multi-task model.
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.dim = dim
        self.multitask = multitask

        if self.multitask and self.y.ndim == 1:
            raise ValueError("For multitask mode, y must be two-dimensional (shape [N,2]).")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X_sample = self.X[idx]
        y_sample = self.y[idx]
        if self.dim == "2d":
            X_sample = X_sample.unsqueeze(0)
        #print(y_sample.shape)
        return X_sample, y_sample


def create_train_test_dataloaders_old(X, y, test_size=0.2, batch_size=64, shuffle=True, dim="1d",
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
        print("balancing classes")
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


def create_train_test_dataloaders_old_2(X, y, test_size=0.2, batch_size=64, shuffle=True, dim="1d",
                                  max_samples_per_class=None, max_samples=None):
    """
    Split the dataset into train and test sets and create DataLoaders.
    Args:
        X (np.ndarray): Input features.
        y (np.ndarray): Target labels.
        test_size (float): Proportion of data to use for testing.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.
        dim (str): "1d" for 1D CNN input, "2d" for 2D CNN input.
        max_samples_per_class (int): Maximum samples per class to balance data.
        max_samples (int): Maximum total samples to use from the dataset.

    Returns:
        tuple: (train_loader, test_loader)
    """
    X, y = np.array(X), np.array(y)  # Ensure input is NumPy array

    # Ensure class balancing if max_samples_per_class is set
    if max_samples_per_class is not None:
        print("Balancing classes...")
        class_samples = defaultdict(list)

        # Collect indices for each class
        for idx, label in enumerate(y):
            class_samples[label].append(idx)

        # Limit the number of samples per class
        limited_indices = []
        for class_label, indices in class_samples.items():
            limited_indices.extend(indices[:max_samples_per_class])

        # Subset data
        X, y = X[limited_indices], y[limited_indices]

    # Shuffle dataset
    if shuffle:
        shuffle_idxs = np.random.permutation(len(y))
        X, y = X[shuffle_idxs], y[shuffle_idxs]

    # Limit total samples if max_samples is set
    if max_samples is not None:
        X, y = X[:max_samples], y[:max_samples]

    # Split into train and test sets
    split_idx = int(len(y) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Create datasets
    train_dataset = RULDataset(X_train, y_train, dim=dim)
    test_dataset = RULDataset(X_test, y_test, dim=dim)

    # Verify sizes before creating DataLoaders
    print(f"Train set size: {len(train_dataset)}, Test set size: {len(test_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def create_train_test_dataloaders(X, y, test_size=0.2, batch_size=64, shuffle=True, dim="1d",
                                  max_samples_per_class=None, max_samples=None, multitask=False):
    """
    Split the dataset into train and test sets and create DataLoaders.
    Args:
        X (np.ndarray): Input features.
        y (np.ndarray): Target labels. For multi-task, y must have shape [N,2].
        test_size (float): Proportion of data for test set.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
        dim (str): "1d" for 1D CNN input, "2d" for 2D CNN input.
        max_samples_per_class (int): Maximum samples per unique label.
                                     For multi-task, uses column 1.
        max_samples (int): Maximum total samples to use.
        multitask (bool): If True, indicates that y is multi-dimensional.
    Returns:
        (train_loader, test_loader)
    """
    X = np.array(X)

    if multitask:
        y = np.array(y, ndmin=2)
    else:
        y = np.array(y)

    if max_samples_per_class is not None:
        print("Balancing classes...")
        class_samples = defaultdict(list)
        for idx, label in enumerate(y):
            if np.ndim(y) == 1 or (np.ndim(y) > 1 and not multitask):
                class_samples[label].append(idx)
            else:
                class_samples[label[1]].append(idx)
        limited_indices = []
        for class_label, indices in class_samples.items():
            limited_indices.extend(indices[:max_samples_per_class])
        X, y = X[limited_indices], y[limited_indices]

    if shuffle:
        shuffle_idxs = np.random.permutation(len(y))
        X, y = X[shuffle_idxs], y[shuffle_idxs]

    if max_samples is not None:
        X, y = X[:max_samples], y[:max_samples]

    split_idx = int(len(y) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    train_dataset = RULDataset(X_train, y_train, dim=dim, multitask=multitask)
    test_dataset = RULDataset(X_test, y_test, dim=dim, multitask=multitask)

    print(f"Train set size: {len(train_dataset)}, Test set size: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
