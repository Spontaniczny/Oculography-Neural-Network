import torch
import torch.nn as nn
from models import BaseNet
from ellipse import remove_noise
from typing import Tuple


def final_evaluation(
    model: BaseNet,
    test_dataset: torch.utils.data.DataLoader,
    device: str
) -> Tuple[dict[str, float], dict[str, float], dict[str, torch.Tensor]]:
    
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
        
    metrics_basic = dict(zip(metric_names, [0]*len(metric_names)))
    metrics_refined = dict(zip(metric_names, [0]*len(metric_names)))

    pupil_sizes_basic = []
    pupil_sizes_refined = []
    pupil_sizes_truth = []

    counter = 0

    for test_batch, truth_masks in test_dataset:

        test_batch, truth_masks = test_batch.to(device), truth_masks.to(device)

        truth_masks = model.draw_ellipse(truth_masks).to(device).bool()
        predicted_masks = model.predict_mask(test_batch).to(device).bool()

        
        for predicted_mask, truth_mask in zip(predicted_masks, truth_masks):
            
            pupil_sizes_truth.append(truth_mask.sum().item())
            for metrics, is_basic in [(metrics_basic, True), (metrics_refined, False)]:

                if is_basic:
                    pupil_sizes_basic.append(predicted_mask.sum().item())
                else:
                    pupil_sizes_refined.append(predicted_mask.sum().item())

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

                predicted_mask, _ = remove_noise(predicted_mask.to("cpu"))
                predicted_mask = predicted_mask.to(device).bool()

            counter += 1

    for metric in metric_names:
        metrics_basic[metric] = metrics_basic[metric].item() / counter
        metrics_refined[metric] = metrics_refined[metric].item() / counter

    pupil_sizes = {
        "pupil_sizes_basic": torch.Tensor(pupil_sizes_basic),
        "pupil_sizes_refined": torch.Tensor(pupil_sizes_refined),
        "pupil_sizes_truth": torch.Tensor(pupil_sizes_truth),

    }

    return metrics_basic, metrics_refined, pupil_sizes
