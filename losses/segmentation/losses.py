import torch.nn as nn
import torch
import torch.nn.functional as F


class DSCLoss(nn.Module):
    def __init__(
        self,
        logits: bool = True
    ):  
        
        super().__init__()
        self.sigmoid = nn.Sigmoid() if logits else lambda x: x
        self.eps = 1e-8

    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pred = self.sigmoid(pred)
        
        intersection = pred*labels
        sum_of_areas = pred.square().sum() + labels.square().sum()
        dsc = 2*intersection.sum() / (sum_of_areas + self.eps)
        loss = 1 - dsc

        return loss
    

class IoULoss(nn.Module):
    def __init__(
        self,
        logits: bool = True
    ):  
        
        super().__init__()
        self.sigmoid = nn.Sigmoid() if logits else lambda x: x
        self.eps = 1e-8

    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pred = self.sigmoid(pred)

        intersection = (pred * labels).sum()
        union = pred.sum() + labels.sum() - intersection

        iou = intersection / (union + self.eps)
        loss = 1 - iou
        return loss
    

class MCCLoss(nn.Module):
    def __init__(
        self,
        logits: bool = True
    ):  
        
        super().__init__()
        self.sigmoid = nn.Sigmoid() if logits else lambda x: x
        self.eps = 1e-8

    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pred = self.sigmoid(pred)

        tp = torch.sum(torch.mul(pred, labels))
        tn = torch.sum(torch.mul((1 - pred), (1 - labels)))
        fp = torch.sum(torch.mul(pred, (1 - labels)))
        fn = torch.sum(torch.mul((1 - pred), labels))

        numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
        denominator = torch.sqrt(
            torch.add(tp, fp)
            * torch.add(tp, fn)
            * torch.add(tn, fp)
            * torch.add(tn, fn)
        )


        mcc = torch.div(numerator.sum(), denominator.sum() + self.eps)
        return 1 - mcc
    

class FocalLoss(nn.Module):
    
    def __init__(
        self, 
        logits: bool = True,
        alpha: float = 0.25, 
        gamma: float = 2.0
    ):
        
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.sigmoid = nn.Sigmoid() if logits else lambda x: x


    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        
        pred = self.sigmoid(pred)
        ce_loss = F.binary_cross_entropy_with_logits(pred, labels, reduction='none')
        p_t = pred * labels + (1 - pred) * (1 - labels)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        return loss.mean()
    

class MAE(nn.Module):
    def __init__(
        self,
        logits: bool = True
    ):  
        
        super().__init__()
        self.sigmoid = nn.Sigmoid() if logits else lambda x: x
        self.l1 = nn.L1Loss()

    def forward(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pred = self.sigmoid(pred)
        return self.l1(pred, labels)

        
    