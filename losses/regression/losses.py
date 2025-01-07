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
    

class SineSmoothL1Loss(nn.Module):
    def __init__(
        self,
    ):  
        super().__init__()
        self.loss_func = nn.SmoothL1Loss()

    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pred[:, -1] = torch.sin(pred[:, -1])
        labels[:, -1] = torch.sin(labels[:, -1])
        return self.loss_func(pred, labels)
    

class SmoothL1LossWithArea(nn.Module):
    def __init__(
        self,
    ):  
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = len(pred)
        major_ax_p, minor_ax_p = pred[:, 2], pred[:, 3]
        major_ax_l, minor_ax_l = labels[:, 2], labels[:, 3]
        area_diff = torch.abs(major_ax_p * minor_ax_p - major_ax_l * minor_ax_l).sum() / batch_size
        return self.smooth_l1(pred, labels) + area_diff

