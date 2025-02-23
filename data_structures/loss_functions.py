import torch
import torch.nn as nn
import torch.nn.functional as F


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


class SmoothAdaptiveAsymmetricHuberLoss(nn.Module):
    def __init__(self, delta_base=5.0, overestimate_weight=2.0, underestimate_scale=0.05):
        """
        Smooth Adaptive Asymmetric Huber Loss:
        - Overestimations are penalized more at low RUL.
        - Underestimations are progressively penalized more as RUL increases.

        Parameters:
            delta_base (float): Base Huber delta.
            overestimate_weight (float): Overestimation penalty multiplier (high for low RUL).
            underestimate_scale (float): Controls how fast underestimation penalty increases with RUL.
        """
        super(SmoothAdaptiveAsymmetricHuberLoss, self).__init__()
        self.delta_base = delta_base
        self.overestimate_weight = overestimate_weight
        self.underestimate_scale = underestimate_scale

    def forward(self, predictions, targets):
        errors = predictions - targets  # Positive -> Overestimate, Negative -> Underestimate
        abs_errors = torch.abs(errors)

        # **Smooth weight functions** (instead of thresholds)
        weight_underestimate = 1 + self.underestimate_scale * targets  # Increases smoothly as RUL grows
        weight_overestimate = self.overestimate_weight * torch.exp(-targets / 20)  # Higher penalty for low RUL

        # Adaptive weighting
        weights = torch.ones_like(errors)
        weights = torch.where(errors < 0, weight_underestimate, weight_overestimate)  # Different penalties

        # Adaptive delta scaling (Optional)
        delta = self.delta_base * (1 + 0.01 * targets)  # Delta grows slightly with RUL

        # Standard Huber loss formula with smooth weighting
        huber_loss = torch.where(
            abs_errors <= delta,
            0.5 * (errors ** 2),
            delta * (abs_errors - 0.5 * delta)
        )

        return (weights * huber_loss).mean()


class AdaptiveFrequencyAsymmetricHuberLoss(nn.Module):
    def __init__(self, frequency_map, base_delta=5.0, min_weight=1.0, max_weight=3.0, overestimate_weight=2.0, underestimate_scale=0.05, overestimate_bias=0):
        """
        Adaptive Frequency-Weighted Asymmetric Huber Loss:
        - Overestimates are penalized more at low RUL.
        - Underestimates are penalized more at high RUL.
        - Rare RUL values contribute more to training via frequency weighting.

        Parameters:
            base_delta (float): Standard Huber delta.
            min_weight (float): Minimum frequency-based weight.
            max_weight (float): Maximum frequency-based weight.
            overestimate_weight (float): Multiplier for overestimation penalty (high at low RUL).
            underestimate_scale (float): Controls how fast underestimation penalty increases with RUL.
        """
        super(AdaptiveFrequencyAsymmetricHuberLoss, self).__init__()
        self.frequency_map = frequency_map
        self.base_delta = base_delta
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.overestimate_weight = overestimate_weight
        self.underestimate_scale = underestimate_scale
        self.overestimate_bias = overestimate_bias

    def forward(self, predictions, targets):
        errors = predictions - targets  # Positive -> Overestimate, Negative -> Underestimate
        abs_errors = torch.abs(errors)

        # Frequency-based weight (inverted: rare values get higher weights)
        weights_freq = torch.tensor([self.max_weight / (self.frequency_map.get(float(t), 1) + 1)
                                     for t in targets.cpu().numpy()], device=targets.device)
        weights_freq = torch.clamp(weights_freq, self.min_weight, self.max_weight)  # Keep within range

        # Asymmetric weight:
        weight_underestimate = 1 + self.underestimate_scale * targets  # More penalty for underestimates at high RUL
        weight_overestimate = self.overestimate_weight * (torch.exp(-targets / 10) + self.overestimate_bias)

        # Combined weighting
        weights_asym = torch.where(errors < 0, weight_underestimate, weight_overestimate)
        total_weight = weights_freq * weights_asym  # Apply both frequency-based and asymmetric penalties

        # Adaptive delta (Optional)
        delta = self.base_delta * (1 + 0.01 * targets)  # Slightly increase delta for higher RULs

        # Standard Huber loss formula with adaptive weighting
        huber_loss = torch.where(
            abs_errors <= delta,
            0.5 * (errors ** 2),
            delta * (abs_errors - 0.5 * delta)
        )

        return (total_weight * huber_loss).mean()


class MultiTaskCriterion(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        """
        alpha: weight for RUL regression loss
        beta:  weight for health classification loss
        """
        super(MultiTaskCriterion, self).__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()  # For binary classification using a single logit
        self.alpha = alpha
        self.beta = beta

    def forward(self, outputs, targets):
        """
        outputs: [batch_size, 2]
            - outputs[:,0] => RUL predictions
            - outputs[:,1] => health logits
        targets: [batch_size, 2]
            - targets[:,0] => RUL ground truth
            - targets[:,1] => health label (0 or 1)
        """
        # Separate the two tasks
        rul_pred = outputs[:, 0]
        health_logit = outputs[:, 1]

        rul_true = targets[:, 0]
        health_true = targets[:, 1].float()

        # Compute each loss
        # scale mse so it doesn't dominate bce
        loss_rul = self.mse(rul_pred, rul_true) / torch.var(rul_true)
        loss_health = self.bce(health_logit, health_true)

        # Weighted sum
        total_loss = self.alpha * loss_rul + self.beta * loss_health
        return total_loss


class FrequencyWeightedMSELoss(nn.Module):
    def __init__(self, frequency_map, max_weight=3.0, min_weight=1.0):
        """
        Frequency-weighted Mean Squared Error loss initialized with a frequency map.

        Args:
            frequency_map (dict): Dictionary mapping target values to their frequency counts.
        """
        super(FrequencyWeightedMSELoss, self).__init__()

        # # Convert frequency map to tensor
        # max_value = max(frequency_map.keys())
        # frequencies = torch.zeros(max_value + 1, dtype=torch.float32)
        # for value, freq in frequency_map.items():
        #     frequencies[value] = freq
        #
        # # Avoid division by zero
        # frequencies += 1e-6
        #
        # # Assign weights inversely proportional to frequencies
        # weights = 1.0 / frequencies
        #
        # # Normalize weights to have mean of 1
        # weights = weights / weights.mean()
        #
        # self.weights = weights
        self.frequency_map = frequency_map
        self.max_weight = max_weight
        self.min_weight = min_weight

    def forward(self, predictions, targets):
        """
        Calculate frequency-weighted MSE loss.

        Args:
            predictions (torch.Tensor): Model predictions.
            targets (torch.Tensor): True targets.

        Returns:
            torch.Tensor: Computed loss value.
        """
        weights_freq = torch.tensor([self.max_weight / (self.frequency_map.get(float(t), 1) + 1)
                                     for t in targets.cpu().numpy()], device=targets.device)
        weights_freq = torch.clamp(weights_freq, self.min_weight, self.max_weight)  # Keep within range

        mse = F.mse_loss(predictions, targets, reduction='none')
        weighted_mse = mse * weights_freq

        return weighted_mse.mean()


