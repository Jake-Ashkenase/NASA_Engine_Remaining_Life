import torch.nn as nn
import torch
import numpy as np


class CNNRULClassifier(nn.Module):
    def __init__(self, num_features, num_classes, seq_length=50):
        """
        CNN model for classifying Remaining Useful Life (RUL) into discrete buckets.

        Parameters:
        - num_features: int, number of input features (sensor channels).
        - num_classes: int, number of RUL buckets for classification.
        - seq_length: int, length of the time-series sequence (default is 50).
        """
        super(CNNRULClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()

        # Compute flattened feature size dynamically based on input sequence length
        pooled_length = seq_length // 2 // 2 // 2  # Pooling reduces length by a factor of 2 each time
        self.fc1 = nn.Linear(128 * pooled_length, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def feature_extractor(self, x):
        """Extract intermediate feature embeddings before final classification layer."""
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = self.flatten(x)
        return torch.relu(self.fc1(x))  # Feature vector (batch_size, 64)
    
    def forward(self, x):
        """
        Forward pass for the CNN model.

        Parameters:
        - x: Tensor of shape (batch_size, num_features, sequence_length)

        Returns:
        - Logits for classification (before softmax)
        """
        x = self.feature_extractor(x)
        return self.fc2(x)

    def get_predict_and_true(self, test_loader, device):
        """
        Get predicted RUL and true RUL values for the test set.

        Parameters:
        - test_loader: DataLoader for the test set.
        - device: 'cuda' or 'cpu' based on availability.

        Returns:
        - Tuple of predicted RUL and true RUL values.
        """
        self.eval()
        predicted_rul = []
        actual_rul = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.permute(0, 2, 1)
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self(inputs).argmax(dim=1)
                predicted_rul.extend(outputs.cpu().numpy())
                actual_rul.extend(targets.cpu().numpy())
        return np.array(predicted_rul), np.array(actual_rul)
