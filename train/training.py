import torch
import torch.nn as nn
from copy import deepcopy
from typing import Callable
from torch.utils.data import DataLoader
from models.segmentation.deeplabv3 import DeepLab
from data_utils import load_dataset
from torch.utils.data import DataLoader, Dataset, random_split
from .argparser import parse_arguments, neural_network_config
from .helper_functions import get_loss_function
from .evaluation import compute_loss_metrics, binary_metrics
from .evaluation_plots import plot_precision_recall, plot_roc
import wandb
import wandb
from dotenv import load_dotenv
import os
from PIL import Image


def choose_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def authenticate() -> bool:
    if not load_dotenv():
        raise ValueError("Could not find dotenv file")
    
    wandb_key = os.environ.get("key")
    if wandb_key is None:
        raise ValueError("No wandb key Could not authenticate")
    
    if not wandb.login(key=wandb_key, verify=True):
        raise ValueError("Invalid authentication key")
    
    return True


def prepare_dataloaders(
        dataset: Dataset,
        split_ratio: list[float],
        batch_size: int = 16, 
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:

    train_set, val_set, test_set = random_split(dataset, split_ratio)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    return train_loader, val_loader, test_loader


def validate_model(
        model: nn.Module,
        val_dataset: torch.utils.data.DataLoader,
        loss_fn: Callable[[torch.Tensor], float],
        device: str
    ) -> float:
    
    total_loss = 0.0
    number_of_batches = 0

    model = model.eval()
    with torch.no_grad():
        for images, labels in val_dataset:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            batch_loss = loss_fn(outputs, labels)
            total_loss += batch_loss.item()
            number_of_batches += 1

    average_loss = total_loss / number_of_batches
    return average_loss


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_epochs: int = 50,
    patience: int = 5, # parameter for early stopping
    loss_function: str = "dice",
    device: str = "cpu"
):
    
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters())
    criterion = get_loss_function(loss_function)

    wandb.watch(model, criterion, log="all", log_freq=10)

    best_model, best_loss = model, float("inf")
    steps_without_improvement = 0
    train_losses, val_losses = [], []


    for epoch in range(max_epochs):
        epoch_loss = 0.0
        number_of_batches = 0
        print(f"Epoch {epoch}")

        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            
            batch_loss = criterion(outputs, labels)
            batch_loss.backward()

            optimizer.step()
            optimizer.zero_grad()
        
            epoch_loss += batch_loss.item()
            number_of_batches += 1

            if (i + 1) % 10 == 0:
                wandb.log({
                    "batch": number_of_batches,
                    "Current_average_loss": epoch_loss / number_of_batches
                })
        
        train_loss = epoch_loss / number_of_batches
        train_losses.append(train_loss)

        val_loss = validate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        wandb.log({
            "Train average loss": train_loss,
            "Validation average loss": val_loss
        })

        if val_loss < best_loss:
            steps_without_improvement = 0
            best_loss = val_loss
            best_model = deepcopy(model)
        else:
            steps_without_improvement += 1
            if steps_without_improvement > patience:
                wandb.log({
                    "Early stopping": epoch,
                })

    return best_model


def main():
    # ds = load_dataset("../../datasets/rat_eye/^.*$", 128)
    # ds = prepare_segmentation_dataset(args.dataset, input_size=args.input_size)

    args = parse_arguments()

    authenticate()
    nn_config = neural_network_config(args)

    with wandb.init(project="first_experiment", config=nn_config) as run:

        ds = load_dataset(args.dataset, input_size=args.input_size)
        train_loader, val_loader, test_loader = prepare_dataloaders(
            ds, [0.6, 0.2, 0.2], batch_size=args.batch_size
        )
        device = choose_device()

        print(f"Training model on device: {device}")
        net = DeepLab(**nn_config)
        net.to(device)

        net = train(
            model=net,
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=args.max_epochs,
            loss_function=args.loss_type,
            patience=args.patience,
            device=device
        )

        loss_metrics = compute_loss_metrics(net, test_loader, ["mae", "dice", "iou"], device)
        b_metrics = binary_metrics(net, test_loader, device)

        wandb.summary["loss_metrics"] = loss_metrics
        wandb.summary["binary_metrics"] = b_metrics
        # wandb.summary.update()

        wandb.log({
            "prc_curve": wandb.Image(Image.open(plot_precision_recall(b_metrics))),
            "roc_curve": wandb.Image(Image.open(plot_roc(b_metrics))),
        })

        # Saving neural network
        torch.onnx.export(net, torch.randn(1, 1, args.input_size, args.input_size).to(device), "model.onnx")
        wandb.save("model.onnx")


if __name__ == "__main__":
    main()
    