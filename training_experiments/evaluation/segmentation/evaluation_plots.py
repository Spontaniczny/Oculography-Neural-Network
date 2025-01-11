import torch
import matplotlib.pyplot as plt
from io import BytesIO


def plot_precision_recall(
    metrics: dict[str, torch.Tensor],
) -> BytesIO:
    
    optimal_thresh_idx = metrics['f1_binned'].cpu().argmax()
    precision = metrics['precision'].cpu()
    recall = metrics['recall'].cpu()
    opt_x, opt_y = recall[optimal_thresh_idx], precision[optimal_thresh_idx]

    plt.plot(recall, precision, c='blue')
    plt.fill_between(recall, precision, color='darkblue', alpha=0.3, label=f"AUPRC = {metrics["auprc"].cpu():.3f}")
    plt.axvline(opt_x, color="green", linestyle="-.", alpha=0.5)
    plt.axhline(opt_y, color="green", linestyle="-.", alpha=0.5)

    plt.annotate(
        text=f"Opt threshold: {metrics["opt_threshold"]:.3f}\nF1 score: {metrics['f1_binned'].max().cpu():.3f}", 
        xy=(opt_x, opt_y), xytext=(-100, -60),
        textcoords='offset points', 
        arrowprops=dict(arrowstyle='->')
    )

    plt.title(f"Precision recall curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()

    return buffer


def plot_roc(
    metrics: dict[str, torch.Tensor],
) -> BytesIO:
    
    tpr = metrics["recall"].cpu().flip(0)
    fpr = metrics["false_positive_rate"].cpu().flip(0)

    plt.plot(fpr, tpr, label=f"auroc={metrics["auroc"].cpu():.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", c="black", alpha=0.5)

    plt.title(f"ROC")
    plt.xlabel("False Positive Ratio")
    plt.ylabel("True Positive Ratio")
    plt.legend(loc="lower right")
    plt.ylim(0, 1.05)
    plt.xlim(-0.01, 1)

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    return buffer
