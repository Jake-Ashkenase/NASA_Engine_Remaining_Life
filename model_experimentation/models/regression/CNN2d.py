import torch.nn as nn
import torch


class CNNRUL2DRegression(nn.Module):
    def __init__(self, num_features):
        super(CNNRUL2DRegression, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1)

        self.pool = nn.MaxPool2d(kernel_size=(2, 2))

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * (50 // 2 // 2 // 2) * (num_features // 2 // 2 // 2), 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # Input shape: [batch, 1, 50, 44]
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)

        # Flatten and pass through FC layers
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        return torch.relu(self.fc2(x))

    def get_predict_and_true(self, data_loader, device):
        """
        Get the predicted and true values for a DataLoader.
        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
            device (str): Device to run the model on ('cuda' or 'cpu').
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
