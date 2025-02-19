import torch.nn as nn
import torch
import numpy as np


class HybridCNNClassifier(nn.Module):
    def __init__(self, num_features, seq_length, num_classes):
        """
        Hybrid CNN model for classifying Remaining Useful Life (RUL) into discrete buckets.
        This model combines a 1D CNN (for temporal patterns) and a 2D CNN (for feature interactions).

        Parameters:
        - num_features: int, number of input features (sensor channels).
        - seq_length: int, length of the time-series sequence.
        - num_classes: int, number of RUL buckets for classification.
        """
        super(HybridCNNClassifier, self).__init__()

        # 1D CNN Branch (Temporal Patterns)
        self.conv1d = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1)
        self.pool1d = nn.MaxPool1d(kernel_size=2)

        # 2D CNN Branch (Feature Interactions)
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1)
        self.pool2d = nn.MaxPool2d(kernel_size=(2, 2))

        # Compute Fully Connected Layer Sizes
        fc1d_input_size = 64 * (seq_length // 2)  # After 1D CNN
        fc2d_input_size = 32 * ((seq_length // 2) * (num_features // 2))  # After 2D CNN

        self.fc1 = nn.Linear(fc1d_input_size + fc2d_input_size, 128)
        self.fc2 = nn.Linear(128, num_classes)  # Classification output

    def forward(self, x):
        """
        Forward pass for the Hybrid CNN model.

        Parameters:
        - x: Tensor of shape (batch_size, seq_length, num_features)

        Returns:
        - Logits for classification (before softmax)
        """
        # 1D CNN Branch
        x1d = x.permute(0, 2, 1)  # Convert [batch, seq_length, features] → [batch, features, seq_length]
        x1d = self.conv1d(x1d)
        x1d = self.pool1d(torch.relu(x1d))
        x1d = torch.flatten(x1d, start_dim=1)

        # 2D CNN Branch
        x2d = x.unsqueeze(1)  # Convert [batch, seq_length, features] → [batch, 1, seq_length, features]
        x2d = self.conv2d(x2d)
        x2d = self.pool2d(torch.relu(x2d))
        x2d = torch.flatten(x2d, start_dim=1)

        # Combine both branches
        x_combined = torch.cat((x1d, x2d), dim=1)
        x_combined = torch.relu(self.fc1(x_combined))

        return self.fc2(x_combined)

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
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self(inputs).argmax(dim=1)
                predicted_rul.extend(outputs.cpu().numpy())
                actual_rul.extend(targets.cpu().numpy())
        return np.array(predicted_rul), np.array(actual_rul)

class ComplexHybridCNNClassifier(nn.Module):
    def __init__(self, num_features, seq_length, num_classes):
        super(ComplexHybridCNNClassifier, self).__init__()

        # 1d CNN Branch (Temporal Feature Extraction)
        self.conv1d_1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=7, padding=3)
        self.conv1d_2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, padding=3)
        self.conv1d_3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, padding=3)
        self.pool1d = nn.MaxPool1d(kernel_size=2)
        self.bn1d_1 = nn.BatchNorm1d(32)
        self.bn1d_2 = nn.BatchNorm1d(64)
        self.bn1d_3 = nn.BatchNorm1d(128)

        ## 2d CNN Branch (Feature Interactions)
        self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(7, 7), padding=3)
        self.conv2d_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(7, 7), padding=3)
        self.conv2d_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(7, 7), padding=3)
        self.pool2d = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn2d_1 = nn.BatchNorm2d(16)
        self.bn2d_2 = nn.BatchNorm2d(32)
        self.bn2d_3 = nn.BatchNorm2d(64)

        # Fully Connected Layers
        fc1d_input_size = 128 * (seq_length // 8)  # After three 1D CNN pooling layers
        fc2d_input_size = 64 * ((seq_length // 8) * (num_features // 8))  # After three 2D CNN pooling layers
        combined_fc_size = fc1d_input_size + fc2d_input_size

        self.fc1 = nn.Linear(combined_fc_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        # 1d CNN Branch
        x1d = x.permute(0, 2, 1)  # Convert [batch, seq_length, features] → [batch, features, seq_length]
        x1d = self.pool1d(torch.relu(self.bn1d_1(self.conv1d_1(x1d))))
        x1d = self.pool1d(torch.relu(self.bn1d_2(self.conv1d_2(x1d))))
        x1d = self.pool1d(torch.relu(self.bn1d_3(self.conv1d_3(x1d))))
        x1d = torch.flatten(x1d, start_dim=1)

        # 2d CNN Branch
        x2d = x.unsqueeze(1)  # Convert [batch, seq_length, features] → [batch, 1, seq_length, features]
        x2d = self.pool2d(torch.relu(self.bn2d_1(self.conv2d_1(x2d))))
        x2d = self.pool2d(torch.relu(self.bn2d_2(self.conv2d_2(x2d))))
        x2d = self.pool2d(torch.relu(self.bn2d_3(self.conv2d_3(x2d))))
        x2d = torch.flatten(x2d, start_dim=1)

        # Combine both branches
        x_combined = torch.cat((x1d, x2d), dim=1)

        # Fully Connected Layers
        x_combined = torch.relu(self.fc1(x_combined))
        x_combined = self.dropout(x_combined)
        x_combined = torch.relu(self.fc2(x_combined))
        x_combined = self.dropout(x_combined)

        return self.fc3(x_combined)

    def get_predict_and_true(self, data_loader, device):
        """
        Get the predicted and true values for a DataLoader.
        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        Returns:
            tuple: (predicted, true)
        """
        predicted = []
        true = []

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(device)
                outputs = self(inputs).squeeze().argmax(dim=1)
                predicted.extend(outputs.cpu().numpy())
                true.extend(targets.cpu().numpy())
        return np.array(predicted), np.array(true)

