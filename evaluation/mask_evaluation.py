import torch
import torch.nn as nn
from models import BaseNet


def final_evaluation(
    model: BaseNet,
    test_dataset: torch.utils.data.DataLoader,
    device: str
) -> dict[str, torch.Tensor]:
    model = model.to(device)
    model = model.eval()

    metric_names = [
        "iou",
        "mcc",
        "dice",
        "accuracy",
        "precision",
        "recall",
        "specificity",
    ]
        
    metrics = dict(zip(metric_names, [0]*len(metric_names)))
    counter = 0

    for test_batch, truth_mask in test_dataset:

        test_batch, truth_mask = test_batch.to(device), truth_mask.to(device)
        predicted_mask = model.predict_mask(test_batch).bool()
        truth_mask = truth_mask.bool()

        tp = (predicted_mask & truth_mask).sum()
        tn = (~predicted_mask & ~truth_mask).sum()

        fp = (predicted_mask & ~truth_mask).sum()
        fn = (~predicted_mask & truth_mask).sum()

        metrics["iou"] += tp / (tp + fn + fp + 1)
        metrics["mcc"] += (tp * tn - fp * fn) / (torch.sqrt((tp + fp)) * torch.sqrt(tp + fn) * torch.sqrt(tn + fp) * torch.sqrt(tn + fn) + 1)
        metrics["dice"] += (2*tp) / (2*tp + fn + fp + 1)
        metrics["accuracy"] += (tp + tn) / (tp + tn + fp + fn)
        metrics["precision"] += tp / (tp + fp + 1)
        metrics["recall"] += tp / (tp + fn + 1)
        metrics["specificity"] += tn / (tn + fp + 1)

        counter += 1

    for metric in metric_names:
        metrics[metric] = metrics[metric].item() /counter

    return metrics
