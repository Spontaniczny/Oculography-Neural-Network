import argparse
from typing import Any
import uuid

def training_config(args: argparse.Namespace) -> dict[str, Any]:
    experiment_id = str(uuid.uuid4())
    nn_config = {
        "net_type": args.net_type,
        "backbone": args.backbone,
        "input_size": args.input_size,
        "training_data": args.dataset,
        "augment": args.augment,
        "experiment_id": experiment_id
    }
    return nn_config

def finetuning_config(args: argparse.Namespace) -> dict[str, Any]:
    experiment_id = f"{str(uuid.uuid4())}_finetuning"
    nn_config = {
        "base_net_config": args.net_config_file,
        "input_size": args.input_size,
        "training_data": args.dataset,
        "experiment_id": experiment_id
    }
    return nn_config


def add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to directory with training data"
    )

    losses = [
        "dice", 
        "iou", 
        "bce",
        "mcc",
        "focal",
        "smooth_l1", 
        "weighted_smooth_l1",
        "sin_smooth_l1",
        "gaussian"
    ]

    parser.add_argument(
        "--loss_type",
        type=str,
        default="dice",
        choices=losses,
        help="Loss functions used for training"
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        choices=list(range(3, 51)),
        metavar="[3-50]",
        help="Patience in early stopping"
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        choices=["AdamW", "Adam"]
    )

    parser.add_argument(
        "--max_epochs",
        type=int,
        default=50,
        choices=list(range(10, 201)),
        metavar="[10-200]",
        help="Maximum number of epochs during training"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        choices=[16, 32, 64, 128, 256],
        help="Batch size during data loading"
    )

    parser.add_argument(
        "--input_size",
        type=int,
        default=128,
        choices=[128, 256, 512],
        required=True,
        help="Size of input images (if smaller rescaling is applied)"
    )

def parse_finetuning_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--net_config_file",
        type=str,
        required=True
    )

    add_training_args(parser)
    args = parser.parse_args()
    return args


def parse_training_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    available_backbones = [
        "res_net_50",
        "res_net_34",
        "res_net_18",
        "xception",
        "mobile_net_small",
        "mobile_net_large"
    ]

    parser.add_argument(
        "--net_type",
        type=str,
        choices=["segmentation", "regression"],
        default="segmentation",
        required=True,
    )

    parser.add_argument(
        "--backbone", 
        type=str, choices=available_backbones, 
        nargs="?",
        default="res_net_50",
        help="Backbone for segmentation net"
    )

    parser.add_argument(
        "--augment",
        type=bool,
        default=False,
        help="Path to directory with training data"
    )

    add_training_args(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_training_args()
    print(args)