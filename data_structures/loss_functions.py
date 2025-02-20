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
