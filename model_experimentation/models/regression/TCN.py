import torch.nn as nn
import torch.nn.functional as F
import torch

class TemporalBlock(nn.Module):
    """
    A single block in a Temporal Convolutional Network (TCN).
    - Uses dilated causal convolutions, batch normalization, and residual connections.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()

        padding = (kernel_size - 1) * dilation // 2  # Ensures output shape matches input

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)

        # Adjust residual connection if the number of channels changes
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)  # Match channel size if needed

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        return x + res  # Ensures shape matches

class TCNRegression(nn.Module):
    """
    Temporal Convolutional Network (TCN) for RUL Prediction.
    - Uses multiple TemporalBlocks with increasing dilation.
    - Outputs a single regression value per sequence.
    """
    def __init__(self, num_features, seq_length, num_channels=[64, 128, 256], kernel_size=3, dropout=0.2):
        super(TCNRegression, self).__init__()

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation = 2 ** i  # Exponentially increasing dilation rate
            in_channels = num_features if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, dilation, dropout))

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1] * seq_length, 1)  # Fully connected output layer

    def forward(self, x):
        """
        x shape: [batch, seq_length, num_features]
        Expected shape: [batch, num_features, seq_length]
        """
        x = x.permute(0, 2, 1)  # Convert [batch, seq_length, num_features] → [batch, num_features, seq_length]
        x = self.network(x)
        x = x.view(x.shape[0], -1)  # Flatten for FC layer
        return self.fc(x)

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
                inputs = inputs.to(self.device)
                outputs = self(inputs).squeeze()
                predicted.extend(outputs.cpu().numpy())
                true.extend(targets.cpu().numpy())
        return predicted, true
