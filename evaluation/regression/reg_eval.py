import torch
from losses.segmentation import DSCLoss, MAE, IoULoss, MCCLoss
from losses.regression import WeightedSmoothL1Loss
from ellipse import Ellipse
from torch.utils.data import DataLoader
from models.regression import EllipseNet

def regression_evaluation_metrics(
        model: EllipseNet,
        test_dataset: DataLoader,
        device: str,
    ) -> dict[str, float]:

    number_of_batches = 0

    mask_metrics = ["mae", "dice", "iou", "mcc"]
    ellipse_loss = 0.0
    ellipse_loss_fn = WeightedSmoothL1Loss()

    metrics_values = {"mae": 0.0, "dice": 0.0, "iou": 0.0, "mcc": 0.0}
    metric_functions = {"mae": MAE(), "dice" : DSCLoss(), "iou" : IoULoss(), "mcc": MCCLoss}

    model = model.to(device)
    model = model.eval()

    with torch.no_grad():
        for images, params in test_dataset:
            images, params = images.to(device), params.to(device)
            outputs = model(images)

            ellipse_loss += ellipse_loss_fn(outputs, params).item()

            pred_batch, real_batch = [], []
            for ellipse_pred, ellipse_target in zip(outputs, params):
                ell_pred = Ellipse(*(ellipse_pred[:-1]*128), ellipse_pred[-1].item()*180, (128, 128))
                ell_real = Ellipse(*(ellipse_target[:-1]*128), ellipse_target[-1].item()*180, (128, 128))

                pred_batch.append(torch.Tensor(ell_pred.draw_ellipse()))
                real_batch.append(torch.Tensor(ell_real.draw_ellipse()))

            pred_batch = torch.stack(pred_batch).reshape(-1, 1, 128, 128)
            real_batch = torch.stack(real_batch).reshape(-1, 1, 128, 128)

            for metric in mask_metrics:
                batch_loss = metric_functions[metric](pred_batch, real_batch)
                metrics_values[metric] += batch_loss.item()
            
            number_of_batches += 1

    for metric in mask_metrics:
        metrics_values[metric] /= number_of_batches

    ellipse_loss /= number_of_batches

    return metrics_values | {"smooth_l1": ellipse_loss}

