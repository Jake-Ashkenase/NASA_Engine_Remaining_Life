import torch
import torch.nn as nn

class HybridCNNRegression(nn.Module):
    def __init__(self, num_features, seq_length):
        super(HybridCNNRegression, self).__init__()

        # 1d CNN Branch (Temporal Patterns)
        self.conv1d = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1)
        self.pool1d = nn.MaxPool1d(kernel_size=2)

        # 2d CNN Branch (Feature Interactions)
        self.conv2d = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1)
        self.pool2d = nn.MaxPool2d(kernel_size=(2, 2))

        # Compute Fully Connected Layer Sizes
        fc1d_input_size = 64 * (seq_length // 2)  # After 1D CNN
        fc2d_input_size = 32 * ((seq_length // 2) * (num_features // 2))  # After 2D CNN

        self.fc1 = nn.Linear(fc1d_input_size + fc2d_input_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # 1d CNN Branch
        x1d = x.permute(0, 2, 1)  # Convert [batch, seq_length, features] → [batch, features, seq_length]
        x1d = self.conv1d(x1d)
        x1d = self.pool1d(torch.relu(x1d))
        x1d = torch.flatten(x1d, start_dim=1)

        # 2d CNN Branch
        x2d = x.unsqueeze(1)  # Convert [batch, seq_length, features] → [batch, 1, seq_length, features]
        x2d = self.conv2d(x2d)
        x2d = self.pool2d(torch.relu(x2d))
        x2d = torch.flatten(x2d, start_dim=1)

        # Combine both branches
        x_combined = torch.cat((x1d, x2d), dim=1)

        x_combined = torch.relu(self.fc1(x_combined))
        return self.fc2(x_combined)

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
                outputs = self(inputs).squeeze()
                predicted.extend(outputs.cpu().numpy())
                true.extend(targets.cpu().numpy())
        return predicted, true

class OverfitHybridCNNRegression(nn.Module):
    def __init__(self, num_features, seq_length):
        super(OverfitHybridCNNRegression, self).__init__()

        # 1d CNN Branch (Temporal Feature Extraction)
        self.conv1d_1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1)
        self.conv1d_2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv1d_3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool1d = nn.MaxPool1d(kernel_size=2)
        self.bn1d_1 = nn.BatchNorm1d(32)
        self.bn1d_2 = nn.BatchNorm1d(64)
        self.bn1d_3 = nn.BatchNorm1d(128)

        # 2d CNN Branch (Feature Interactions)
        self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1)
        self.conv2d_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1)
        self.conv2d_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
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
        self.fc3 = nn.Linear(128, 1)

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
                outputs = self(inputs).squeeze()
                predicted.extend(outputs.cpu().numpy())
                true.extend(targets.cpu().numpy())
        return predicted, true


class ComplexHybridCNNRegression(nn.Module):
    def __init__(self, num_features, seq_length):
        """
        A simplified hybrid CNN for RUL regression that combines 1D and 2D CNN branches.
        This model is intended to be more complex than a basic hybrid but less prone to overfitting
        than a very deep complex model.

        Args:
            num_features (int): Number of sensor features.
            seq_length (int): Length of the input time series.
        """
        super(ComplexHybridCNNRegression, self).__init__()

        # 1D CNN Branch (Temporal Feature Extraction)
        self.conv1d_1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1)
        self.bn1d_1 = nn.BatchNorm1d(32)
        self.conv1d_2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn1d_2 = nn.BatchNorm1d(64)
        self.pool1d = nn.MaxPool1d(kernel_size=2)
        self.global_pool1d = nn.AdaptiveAvgPool1d(1)  # Output: [batch, 64, 1]

        # 2D CNN Branch (Feature Interaction Extraction)
        self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1)
        self.bn2d_1 = nn.BatchNorm2d(16)
        self.conv2d_2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1)
        self.bn2d_2 = nn.BatchNorm2d(32)
        self.pool2d = nn.MaxPool2d(kernel_size=(2, 2))
        self.global_pool2d = nn.AdaptiveAvgPool2d((1, 1))  # Output: [batch, 32, 1, 1]

        # Combined fully connected layers after global pooling:
        # 1D branch gives 64 features; 2D branch gives 32 features.
        combined_features = 64 + 32
        self.fc1 = nn.Linear(combined_features, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)  # Regression output

    def forward(self, x):
        # x: [batch, seq_length, num_features]
        # 1D Branch
        x1d = x.permute(0, 2, 1)  # [batch, num_features, seq_length]
        x1d = torch.relu(self.bn1d_1(self.conv1d_1(x1d)))
        x1d = self.pool1d(x1d)
        x1d = torch.relu(self.bn1d_2(self.conv1d_2(x1d)))
        x1d = self.pool1d(x1d)  # [batch, 64, seq_length//4]
        x1d = self.global_pool1d(x1d)  # [batch, 64, 1]
        x1d = torch.flatten(x1d, start_dim=1)  # [batch, 64]

        # 2D Branch
        x2d = x.unsqueeze(1)  # [batch, 1, seq_length, num_features]
        x2d = torch.relu(self.bn2d_1(self.conv2d_1(x2d)))
        x2d = self.pool2d(x2d)
        x2d = torch.relu(self.bn2d_2(self.conv2d_2(x2d)))
        x2d = self.pool2d(x2d)  # [batch, 32, seq_length//4, num_features//4]
        x2d = self.global_pool2d(x2d)  # [batch, 32, 1, 1]
        x2d = torch.flatten(x2d, start_dim=1)  # [batch, 32]

        # Combine branches
        x_combined = torch.cat((x1d, x2d), dim=1)  # [batch, 64+32 = 96]
        x_combined = torch.relu(self.fc1(x_combined))
        x_combined = self.dropout(x_combined)
        out = self.fc2(x_combined)
        return out

    def get_predict_and_true(self, data_loader, device):
        """
        Get predicted and true values for a DataLoader.
        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
            device (torch.device): Device for inference.
        Returns:
            tuple: (predicted, true)
        """
        predicted = []
        true = []
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(device)
                outputs = self(inputs).squeeze()
                predicted.extend(outputs.cpu().numpy())
                true.extend(targets.cpu().numpy())
        return predicted, true
