import torch.nn as nn
import torch


class DSCLoss(nn.Module):
    def __init__(
        self,
        logits: bool = True
    ):  
        
        super().__init__()
        self.sigmoid = nn.Sigmoid() if logits else lambda x: x

    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pred = self.sigmoid(pred)

        intersection = pred*labels
        sum_of_areas = pred.square().sum() + labels.square().sum()
        dsc = 2*intersection.sum() / sum_of_areas
        loss = 1 - dsc
        return loss
    

class IoULoss(nn.Module):
    def __init__(
        self,
        logits: bool = True
    ):  
        
        super().__init__()
        self.sigmoid = nn.Sigmoid() if logits else lambda x: x

    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pred = self.sigmoid(pred)

        intersection = torch.minimum(pred, labels)
        union = torch.maximum(pred, labels)

        iou = intersection.sum() / union.sum()
        loss = 1 - iou
        return loss