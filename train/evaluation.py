import torch
import torch.nn as nn
from .helper_functions import get_loss_function


def compute_loss_metrics(
        model: nn.Module,
        test_dataset: torch.utils.data.DataLoader,
        loss_metrics: list[str],
        device: str
    ) -> dict[str, float]:
    
    number_of_batches = 0

    metrics_values = {
        metric: 0.0 for metric in loss_metrics
    }

    metric_functions = {
        metric: get_loss_function(metric) for metric in loss_metrics
    }

    model = model.to(device)
    model = model.eval()

    with torch.no_grad():
        for images, labels in test_dataset:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            for metric in loss_metrics:
                batch_loss = metric_functions[metric](outputs, labels)
                metrics_values[metric] += batch_loss.item()
            
            number_of_batches += 1

    for metric in loss_metrics:
        metrics_values[metric] /= number_of_batches

    return metrics_values


def compute_binned_values_for_batch(
        pred: torch.Tensor, 
        labels: torch.Tensor,
        bins: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    
    pred, labels = pred.flatten(), labels.flatten()
    sorted_proba = pred.argsort()
    pred, labels = pred[sorted_proba], labels[sorted_proba]
    true_values, negative_values = pred[labels > 0.5], pred[labels <= 0.5]

    total_true, total_false = len(true_values), len(negative_values)

    hist_true, _ = torch.histogram(true_values, bins=bins)
    hist_false, _ = torch.histogram(negative_values, bins=bins)

    fn = torch.cumsum(hist_true, 0)
    tn = torch.cumsum(hist_false, 0)

    return tn, total_false - tn , fn, total_true - fn

def compute_auc(
        x_axis: torch.Tensor,  
        y_axis: torch.Tensor
    ) -> torch.Tensor:
    
    sum_of_bases = y_axis[:-1] + y_axis[1:]
    heights = x_axis[1:] - x_axis[:-1]
    total = sum_of_bases*heights / 2
    return total.sum()

def binary_metrics(
    model: nn.Module,
    test_dataset: torch.utils.data.DataLoader,
    device: str
) -> dict[str, torch.Tensor]:
    model = model.to(device)
    model = model.eval()

    bins = torch.linspace(0.0, 1.0, 101, device=device)

    tn_binned = torch.zeros_like(bins)
    fp_binned = torch.zeros_like(bins)
    fn_binned = torch.zeros_like(bins)
    tp_binned = torch.zeros_like(bins)

    with torch.no_grad():
        for images, labels in test_dataset:
            images, labels = images.to(device), labels.to(device)
            predictions = model.predict_proba(images)
            tn, fp, fn, tp = compute_binned_values_for_batch(predictions, labels, bins)

            tn_binned[1:] += tn
            fp_binned[1:] += fp
            fn_binned[1:] += fn
            tp_binned[1:] += tp

    eps = 1e-16
    precision = tp_binned / (tp_binned + fp_binned + eps)
    recall = tp_binned / (tp_binned + fn_binned + eps)

    precision[-1] = 1.0
    recall[0] = 1.0

    false_positive_rate = fp_binned / (fp_binned + tn_binned + eps)
    false_positive_rate[0] = 1.0

    f1_score = 2*precision*recall / (precision + recall)
    opt_threshold = bins[f1_score.argmax()]

    return {
        "opt_threshold": opt_threshold,
        "precision": precision,
        "recall": recall,
        "f1_binned": f1_score,
        "true_negatives_binned": tn_binned,
        "false_positives_binned": fp_binned,
        "false_negatives_binned": fn_binned,
        "true_positives_binned": tp_binned,
        "false_positive_rate": false_positive_rate,
        "auprc": compute_auc(precision, recall),
        "auroc": compute_auc(false_positive_rate.flip(0), recall.flip(0)), 
        "threshold": bins
    }

