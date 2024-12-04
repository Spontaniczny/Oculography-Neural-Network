import torch
import torch.nn as nn
from losses.segmentation.losses import DSCLoss, IoULoss, MAE


def get_loss_function(loss_name: str) -> nn.Module:
    match loss_name:
        case "dice":
            return DSCLoss()
        case "iou":
            return IoULoss()
        case "bce":
            return nn.BCEWithLogitsLoss()
        case "mae":
            return MAE()
        case _:
            raise ValueError(f"Provided unknow loss function named {loss_name}")
        