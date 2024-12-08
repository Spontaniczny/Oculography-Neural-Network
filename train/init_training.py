import torch
from models.segmentation import DeepLab
from models.regression import EllipseNet
from .training import train
from .helper_functions import get_loss_function, get_core_optimizer
import torch.optim

from data_utils import load_dataset, prepare_dataloaders
from .argparser import parse_arguments, neural_network_config

from evaluation.segmentation import compute_loss_metrics, binary_metrics, plot_precision_recall, plot_roc
from evaluation.regression import regression_evaluation_metrics

# import wandb
# import wandb
# from dotenv import load_dotenv
# import os
# from PIL import Image


def choose_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# def authenticate() -> bool:
#     if not load_dotenv():
#         raise ValueError("Could not find dotenv file")
    
#     wandb_key = os.environ.get("key")
#     if wandb_key is None:
#         raise ValueError("No wandb key Could not authenticate")
    
#     if not wandb.login(key=wandb_key, verify=True):
#         raise ValueError("Invalid authentication key")
    
#     return True


def main():
    # ds = load_dataset("../../datasets/rat_eye/^.*$", 128)
    # ds = prepare_segmentation_dataset(args.dataset, input_size=args.input_size)

    args = parse_arguments()

    # authenticate()
    # with wandb.init(project="first_experiment", config=nn_config) as run:


    # Preparing dataset and train, validation and test data_loaders
    ds = load_dataset(args.dataset, input_size=args.input_size, dataset_type=args.net_type)
    train_loader, val_loader, test_loader = prepare_dataloaders(
        ds, [0.6, 0.2, 0.2], batch_size=args.batch_size
    )

    # Preparing net and moving it to the correct device
    if args.net_type == "segmentation":
        net = DeepLab(backbone=args.backbone, input_size=args.input_size)
    else:
        net = EllipseNet(backbone=args.backbone, input_size=args.input_size)

    device = choose_device()
    net.to(device)

    print(f"Training {args.net_type} model on device: {device}")

    criterion = get_loss_function(args.loss_type, args.net_type)
    criterion = criterion.to(device)
    print(f"Minimizing {args.loss_type} loss")

    optimizer = get_core_optimizer(args.optimizer)(net.parameters())
    print(f"Using {args.optimizer} optimizer")

    nn_config = neural_network_config(args)
    print(f"Neural network config: {nn_config}")

    net = train(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=device
    )


    # Model evaluation

    if args.net_type == "segmentation":
        loss_metrics = compute_loss_metrics(net, test_loader, ["mae", "dice", "iou"], args.net_type, device)
        b_metrics = binary_metrics(net, test_loader, device)
        prc_curve = plot_precision_recall(b_metrics)
        roc_curve = plot_roc(b_metrics)
    else:
        metrics = regression_evaluation_metrics(net, test_loader, "cpu")
        print(metrics)

    net.save_model(nn_config)
    onnx_path = net.save_onnx(torch.randn(1, 1, args.input_size, args.input_size))

    # wandb.summary["loss_metrics"] = loss_metrics
    # wandb.summary["binary_metrics"] = b_metrics
    # wandb.summary.update()

    # wandb.log({
    #     "prc_curve": wandb.Image(Image.open(plot_precision_recall(b_metrics))),
    #     "roc_curve": wandb.Image(Image.open(plot_roc(b_metrics))),
    # })

    # # Saving neural network
    
    # wandb.save("saved_models/onnx_models/model.onnx")


if __name__ == "__main__":
    main()
    