import torch
import torch.nn as nn

class WeightedSmoothL1Loss(nn.Module):
    def __init__(
        self,
        weights: list[int] = [2, 2, 4, 4, 1],
    ):  
        super().__init__()
        assert len(weights) == 5, "Weights vector should have length 5"

        self.register_buffer("weights", torch.Tensor(weights))
        self.loss_func = nn.SmoothL1Loss()

    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pred *= self.weights
        labels *= self.weights
        return self.loss_func(pred, labels)
    