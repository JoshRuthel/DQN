import torch
import torch.nn as nn

class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, predicted, target):
        error = predicted - target
        absolute_error = torch.abs(error)

        loss = torch.where(
            absolute_error < self.delta,
            0.5 * error**2,
            self.delta * (absolute_error - 0.5 * self.delta),
        )

        return loss.mean()