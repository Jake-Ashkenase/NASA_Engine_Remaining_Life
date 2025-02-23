import torch
import torch.nn as nn

class CNNRUL2DMultiTask(nn.Module):
    def __init__(self, num_features):
        super(CNNRUL2DMultiTask, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1)

        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()

        # After 3 pools, each dimension is /8 (since 2x2 pool thrice).
        # If input is [batch, 1, 50, num_features], height=50//8, width=num_features//8.
        in_height = 50 // 8
        in_width = num_features // 8
        self.fc1 = nn.Linear(128 * in_height * in_width, 64)

        # Final layer outputs 2 values:
        #   [..., 0] -> RUL (regression)
        #   [..., 1] -> health logit (binary classification)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)

        x = self.flatten(x)
        x = torch.relu(self.fc1(x))

        # Shape: [batch_size, 2]
        x = self.fc2(x)
        return x

    def get_predict_and_true(self, data_loader, device, classification=False):
        """
        Returns (predicted_rul, actual_rul) for plotting, ignoring the classification output.
        This keeps your existing plot_rul_predictions() function intact.
        """
        predicted = []
        actual = []

        self.eval()
        with torch.no_grad():
            for inputs, targets in data_loader:
                # targets shape: [batch_size, 2] => [RUL, Health]
                inputs = inputs.to(device)
                outputs = self(inputs)  # shape [batch_size, 2]

                if classification:
                    preds = outputs[:, 1]
                    actuals = targets[:, 1]
                else:
                    preds = outputs[:, 0]
                    actuals = targets[:, 0]

                predicted.extend(preds.cpu().numpy())
                actual.extend(actuals.cpu().numpy())

        return predicted, actual
