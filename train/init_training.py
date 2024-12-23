from models.segmentation import DeepLab
from models.regression import EllipseNet
from .training import train
from .helper_functions import get_loss_function, get_core_optimizer, choose_device

from data_utils import load_dataset, prepare_dataloaders
from .argparser import parse_training_args, training_config

from evaluation.segmentation import compute_loss_metrics, binary_metrics, plot_precision_recall, plot_roc
from evaluation.regression import regression_evaluation_metrics
from evaluation.mask_evaluation import final_evaluation

from callbacks import TrainingLogger
from datetime import datetime


def main():

    # Parsing command line arguments
    args = parse_training_args()
    
    logger = TrainingLogger(
        project_name="rat_eye_final", 
        run_name=f"run_{datetime.now().strftime("%d-%m-%Y-%H:%M:%S")}", 
        config=vars(args) | {"finetuning" : "no"}
    )

    # Preparing dataset and train, validation and test data_loaders
    ds = load_dataset(args.dataset, input_size=args.input_size, dataset_type=args.net_type)
    train_loader, val_loader, test_loader = prepare_dataloaders(
        ds, [0.6, 0.2, 0.2], args.net_type, batch_size=args.batch_size, augment=args.augment
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

    nn_config = training_config(args)
    print(f"Neural network config: {nn_config}")

    net = train(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logger,
        criterion=criterion,
        optimizer=optimizer,
        max_epochs=args.max_epochs,
        patience=args.patience,
        device=device,
    )

    # Model evaluation
    print("Model evaluation")
    if args.net_type == "segmentation":
        loss_metrics = compute_loss_metrics(net, test_loader, ["mae", "dice", "iou", "mcc", "focal", "bce"], args.net_type, device)
        b_metrics = binary_metrics(net, test_loader, device)

        logger.save_scalar_metrics(loss_metrics, "loss_metrics")
        logger.save_metrics_table(b_metrics, "Binary_eval_metrics")

        prc_curve = plot_precision_recall(b_metrics)
        roc_curve = plot_roc(b_metrics)

        logger.save_fig("prc_curve", prc_curve)
        logger.save_fig("roc_curve", roc_curve)
    else:
        loss_metrics = regression_evaluation_metrics(net, test_loader, "cpu")
        logger.save_scalar_metrics(loss_metrics, "loss_metrics")

    final_metrics_basic, final_metrics_refined, pupil_sizes = final_evaluation(net, test_loader, device)
    logger.save_scalar_metrics(final_metrics_basic, "final_metrics_basic")
    logger.save_scalar_metrics(final_metrics_refined, "final_metrics_refined")

    logger.save_metrics_table(pupil_sizes, "Pupil Sizes")

    print("Saving net parameters and config")
    # Saving neural network
    net.save_model(nn_config)
    # logger.save_onnx_model(net, input_tensor=torch.randn(1, 1, args.input_size, args.input_size))

    print("Finished training and evaluation")


if __name__ == "__main__":
    main()
    