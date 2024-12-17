import torch
import torch.nn as nn
import torch.optim
from typing import Callable
from losses.segmentation.losses import DSCLoss, IoULoss, MAE, MCCLoss, FocalLoss
from losses.regression.losses import WeightedSmoothL1Loss, SineSmoothL1Loss, GaussianLoss


def choose_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_loss_function(loss_name: str, net_type: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    if net_type == "segmentation":
        match loss_name:
            case "dice":
                return DSCLoss()
            case "iou":
                return IoULoss()
            case "bce":
                return nn.BCEWithLogitsLoss()
            case "mae":
                return MAE()
            case "mcc":
                return MCCLoss()
            case "focal":
                return FocalLoss()
            case _:
                raise ValueError(f"Provided loss function {loss_name} is not suitable for {net_type} net")
            
    elif net_type == "regression":
        match loss_name:
            case "smooth_l1":
                return nn.SmoothL1Loss()
            case "weighted_smooth_l1":
                return WeightedSmoothL1Loss()
            case "sin_smooth_l1":
                return SineSmoothL1Loss()
            case "gaussian":
                return GaussianLoss()
            case _:
                raise ValueError(f"Provided loss function {loss_name} is not suitable for {net_type} net")
    else:
        raise ValueError(f"Unknow type of net: {net_type}")
    

def get_core_optimizer(optimizer_name: str) -> torch.optim.Optimizer:
	return getattr(torch.optim, optimizer_name)
        