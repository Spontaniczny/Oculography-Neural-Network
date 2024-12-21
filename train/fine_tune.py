from inference import load_config_file, load_model
from .argparser import parse_finetuning_args, finetuning_config
from .training import train
from data_utils import load_dataset, prepare_dataloaders

from .helper_functions import get_loss_function, get_core_optimizer, choose_device

from evaluation.segmentation import compute_loss_metrics, binary_metrics, plot_precision_recall, plot_roc
from evaluation.regression import regression_evaluation_metrics
from evaluation import mask_evaluation

from callbacks import TrainingLogger
from datetime import datetime


def main():
    args = parse_finetuning_args()

    config_path = args.net_config_file
    config = load_config_file(config_path)

    net = load_model(config, config_path)
    ds = load_dataset(args.dataset, input_size=args.input_size, dataset_type=config["net_type"])

    train_loader, val_loader, test_loader = prepare_dataloaders(
        ds, [0.6, 0.2, 0.2], config["net_type"], batch_size=args.batch_size,
    )
    
    logger = TrainingLogger(
        project_name="rat_eye_final", 
        run_name=f"run_{datetime.now().strftime("%d-%m-%Y-%H:%M:%S")}_finetuning", 
        config=vars(args) | {"finetuning" : "yes"}
    )

    device = choose_device()
    net.freeze_backbone()

    net.to(device)

    print(f"Finetuning {config["net_type"]} model on device: {device}")

    criterion = get_loss_function(args.loss_type, config["net_type"])
    criterion = criterion.to(device)
    print(f"Minimizing {args.loss_type} loss")

    optimizer = get_core_optimizer(args.optimizer)(net.get_trainable_params())
    print(f"Using {args.optimizer} optimizer")

    nn_config = finetuning_config(args)
    nn_config |= {"net_type": config["net_type"], "input_size": config["input_size"], "backbone": config["backbone"]}
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

    print("Model evaluation")
    if config["net_type"] == "segmentation":
        loss_metrics = compute_loss_metrics(net, test_loader, ["mae", "dice", "iou", "mcc"], config["net_type"], device)
        b_metrics = binary_metrics(net, test_loader, device)

        logger.save_scalar_metrics(loss_metrics, "loss_metrics")
        logger.save_metrics_table(b_metrics, "Binary prediction eval metrics")

        prc_curve = plot_precision_recall(b_metrics)
        roc_curve = plot_roc(b_metrics)

        logger.save_fig("prc_curve", prc_curve)
        logger.save_fig("roc_curve", roc_curve)
    else:
        loss_metrics = regression_evaluation_metrics(net, test_loader, "cpu")
        logger.save_scalar_metrics(loss_metrics, "loss_metrics")

    final_metrics = mask_evaluation(net, test_loader, device)
    logger.save_scalar_metrics(final_metrics, "final_metrics")

    print("Saving net parameters and config")
    # Saving neural network
    net.save_model(nn_config, finetuning=True)
    print("Finished finetuning")


if __name__ == "__main__":
    main()