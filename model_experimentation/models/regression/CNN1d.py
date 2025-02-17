import torch.nn as nn
import torch

class CNNRULRegression(nn.Module):
    def __init__(self, num_features):
        super(CNNRULRegression, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * (50 // 2 // 2 // 2), 64)  # Adjust based on pooling
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

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
                inputs = inputs.permute(0, 2, 1)
                inputs = inputs.to(device)
                outputs = self(inputs).squeeze()
                predicted.extend(outputs.cpu().numpy())
                true.extend(targets.cpu().numpy())
        return predicted, true

