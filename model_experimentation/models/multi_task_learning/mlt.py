import torch
import torch.nn as nn

class MultiTaskCNN(nn.Module):
    def __init__(self, num_features, seq_length):
        """
        CNN with shared feature extraction for multi-task learning.
        - First predicts health state (classification).
        - Then predicts RUL using different branches based on health state.
        """
        super(MultiTaskCNN, self).__init__()

        # Shared Feature Extraction (CNN Layers)
        self.conv1 = nn.Conv1d(num_features, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()

        # Compute output size after pooling
        pooled_length = seq_length // 2 // 2 // 2
        shared_output_size = 128 * pooled_length

        # Health State Classifier
        self.fc_health_1 = nn.Linear(shared_output_size, 64)
        self.fc_health_2 = nn.Linear(64, 1)  # Binary classification (0 or 1)

        # RUL Prediction Branch for Health State 0
        self.fc1_0 = nn.Linear(shared_output_size, 64)
        self.fc2_0 = nn.Linear(64, 1)  # Regression output for RUL

        # RUL Prediction Branch for Health State 1
        self.fc1_1 = nn.Linear(shared_output_size, 64)
        self.fc2_1 = nn.Linear(64, 1)  # Regression output for RUL

    def forward(self, x):
        """
        Forward pass:
        - Extracts features with shared CNN layers.
        - Predicts health state.
        - Selects RUL prediction branch based on predicted health state.
        """
        # Shared CNN Feature Extraction
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = self.flatten(x)

        # Predict Health State
        health_logits = torch.relu(self.fc_health_1(x))
        health_state = torch.sigmoid(self.fc_health_2(health_logits))  # Binary classification

        # RUL Prediction for Each Health State
        x0 = torch.relu(self.fc1_0(x))
        rul_0 = torch.relu(self.fc2_0(x0)) # RUL prediction for Health State 0

        x1 = torch.relu(self.fc1_1(x))
        rul_1 = torch.relu(self.fc2_1(x1))  # RUL prediction for Health State 1

        # Choose RUL branch based on predicted health state
        rul_prediction = torch.where(health_state >= 0.5, rul_1, rul_0)

        return health_state, rul_prediction

    def get_predict_and_true(self, data_loader, device):
        """
        Get predicted health state and RUL for a DataLoader.
        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
            device (str): Device to run the model on ('cuda' or 'cpu').
        Returns:
            tuple: (predicted_health, predicted_rul, true_health, true_rul)
        """
        predicted_health = []
        predicted_rul = []
        true_health = []
        true_rul = []

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(device)
                health_state, rul_prediction = self(inputs)

                # Assuming targets are structured as (health_state, rul)
                health, rul = targets
                predicted_health.extend(health_state.cpu().numpy())
                predicted_rul.extend(rul_prediction.cpu().numpy())
                true_health.extend(health.cpu().numpy())
                true_rul.extend(rul.cpu().numpy())

        return predicted_health, predicted_rul, true_health, true_rul