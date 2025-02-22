import torch
import torch.nn as nn


class AsymmetricHuberLoss(nn.Module):
    def __init__(self, delta=1.0, alpha=1.0, beta=5.0):  # Penalize overestimates 5x more
        super().__init__()
        self.delta = delta
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred, y_true):
        error = y_true - y_pred  # Positive if we underestimated, negative if overestimated
        abs_error = torch.abs(error)

        # Huber-like loss
        quadratic = 0.5 * (error ** 2)
        linear = self.delta * (abs_error - 0.5 * self.delta)

        loss = torch.where(abs_error < self.delta, quadratic, linear)

        # Apply asymmetric penalty
        loss = torch.where(error > 0, self.alpha * loss, self.beta * loss)  # Higher penalty for overestimates
        return loss.mean()


class MultiTaskLoss(nn.Module):
    def __init__(self, lambda_rul=1.0, lambda_health=1.0):
        """
        Multi-task loss function balancing health state classification & RUL regression.

        Parameters:
        - lambda_rul: Weight for RUL loss.
        - lambda_health: Weight for health state loss.
        """
        super(MultiTaskLoss, self).__init__()
        self.lambda_rul = lambda_rul
        self.lambda_health = lambda_health
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, health_pred, health_true, rul_pred, rul_true):
        """
        Computes the weighted multi-task loss.
        """
        loss_health = self.bce_loss(health_pred, health_true.float())
        loss_rul = self.mse_loss(rul_pred, rul_true)

        return self.lambda_health * loss_health + self.lambda_rul * loss_rul

