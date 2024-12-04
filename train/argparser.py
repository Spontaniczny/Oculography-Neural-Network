import argparse


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    available_backbones = [
        "res_net_50",
        "res_net_34",
        "res_net_18",
        "xception"
    ]

    parser.add_argument(
        "--backbone", 
        type=str, choices=available_backbones, 
        nargs="?",
        default="res_net_50",
        help="Backbone for segmentation net"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to directory with training data"
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
        "--max_epochs",
        type=int,
        default=50,
        choices=list(range(10, 201)),
        metavar="[10-200]",
        help="Maximum number of epochs during training"
    )

    parser.add_argument(
        "--loss_type",
        type=str,
        default="dice",
        choices=["dice", "iou", "bce"],
        help="Loss functions used for training"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        choices=[16, 32, 64],
        help="Batch size during data loading"
    )

    parser.add_argument(
        "--input_size",
        type=int,
        default=None,
        choices=[128, 256, 512],
        help="Size of input images (if smaller rescaling is applied)"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    print(args)