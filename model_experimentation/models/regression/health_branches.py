import torch
import torch.nn as nn
import torch.nn.functional as F


class Branch2DRegression(nn.Module):
    def __init__(self, num_features):
        super(Branch2DRegression, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.flatten = nn.Flatten()
        # After three 2x2 pools, height becomes 50//8 and width becomes num_features//8.
        self.fc1 = nn.Linear(128 * (50 // 8) * (num_features // 8), 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: [batch, 1, 50, num_features]
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

# The hybrid model that first classifies health then routes to the correct branch.
class HybridMultiBranchRUL(nn.Module):
    def __init__(self, pretrained_classifier, healthy_regressor, unhealthy_regressor):
        """
        Args:
            num_features (int): Number of sensor features (excluding the health column).
            pretrained_classifier (nn.Module): A pre-trained 1D CNN that classifies health state.
                It should accept input of shape [batch, 1, seq_length] (with seq_length=50) and output logits.
        """
        super(HybridMultiBranchRUL, self).__init__()
        self.pretrained_classifier = pretrained_classifier
        # Optionally freeze the classifier:
        for param in self.pretrained_classifier.parameters():
            param.requires_grad = False

        # Two separate regression branches:
        # self.branch_healthy = Branch2DRegression(num_features)
        # self.branch_unhealthy = Branch2DRegression(num_features)
        self.branch_healthy = healthy_regressor
        self.branch_unhealthy = unhealthy_regressor

    def forward(self, x):
        """
        x: Tensor of shape [batch, 1, 50, num_features+1]
           (last column in width dimension contains the health state)
        """
        # Split the input:
        # sensor_data: all columns except the last → shape: [batch, 1, 50, num_features]
        sensor_data = x[..., :-1]
        # health_data: last column → shape: [batch, 1, 50]
        health_data = x[..., -1]  # Remove unsqueeze; shape is now [batch, 1, 50]


        # Process health_data with the pre-trained classifier.
        # The classifier expects input shape [batch, 50, 44]
        health_logits = self.pretrained_classifier(sensor_data.squeeze(1).permute(0, 2, 1))
        # Assume binary classification: healthy = 1, unhealthy = 0.
        health_state = torch.argmax(health_logits, dim=1)  # shape: [batch]

        # Compute outputs from both regression branches.
        out_healthy = self.branch_healthy(sensor_data)    # [batch, 1]
        out_unhealthy = self.branch_unhealthy(sensor_data)  # [batch, 1]

        # Create mask from predicted health state: healthy (==1) → 1, unhealthy (==0) → 0.
        mask = health_state.float().unsqueeze(1)  # shape: [batch, 1]
        # Combine branch outputs.
        out = mask * out_healthy + (1 - mask) * out_unhealthy
        return out

    def get_predict_and_true(self, data_loader, device):
        """
        Returns predicted and true RUL for plotting.
        Assumes targets are shape [batch, 2] where column 0 is RUL.
        """
        predicted = []
        true = []
        self.eval()
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.to(device)
                outputs = self(inputs).squeeze()
                predicted.extend(outputs.cpu().numpy())
                true.extend(targets[:, 0].cpu().numpy())
        return predicted, true
