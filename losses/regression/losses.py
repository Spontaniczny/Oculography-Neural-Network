import torch
import torch.nn as nn

class WeightedSmoothL1Loss(nn.Module):
    def __init__(
        self,
        weights: list[int] = [1, 1, 1, 1, 1],
        device: str = "cpu"
    ):  
        super().__init__()
        assert len(weights) == 5, "Weights vector should have length 5"

        self.weights = torch.Tensor(weights).to(device)
        self.loss_func = nn.SmoothL1Loss()

    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        
        pred *= self.weights
        labels *= self.weights
        return self.loss_func(pred, labels)
    