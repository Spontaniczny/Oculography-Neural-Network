import torch
import torch.nn as nn
from typing import Callable
from torch.utils.data import DataLoader

from training_experiments.models.segmentation import DeepLab
from training_experiments.train.helper_functions import get_loss_function, get_core_optimizer, choose_device
from training_experiments.data_utils import load_dataset, prepare_dataloaders

import uuid
import argparse


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    available_backbones = [
        "res_net_34",
        "res_net_18",
        "xception",
        "mobile_net_small",
        "mobile_net_large",
    ]

    parser.add_argument(
        "--backbone",
        type=str,
        choices=available_backbones,
        default="mobile_net_small",
        required=True,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to directory with training data"
    )

    return parser


@torch.no_grad()
def validate_model(
        model: DeepLab,
        val_dataset: torch.utils.data.DataLoader,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: str,
    ) -> float:
    
    total_loss = 0.0
    number_of_batches = 0

    model = model.eval()
    for images, labels in val_dataset:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        batch_loss = criterion(outputs, labels)
        total_loss += batch_loss.item()
        number_of_batches += 1

    average_loss = total_loss / number_of_batches
    return average_loss

def train(
    model: DeepLab,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    max_epochs: int = 100,
    patience: int = 10,
    device: str = "cpu",
):

    model.to(device)

    best_loss = float("inf")
    steps_without_improvement = 0

    for epoch in range(max_epochs):
        epoch_loss = 0.0
        number_of_batches = 0
        print(f"Epoch {epoch + 1}")

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
        
        train_loss = epoch_loss / number_of_batches
        val_loss = validate_model(model, val_loader, criterion, device)

        print(f"Train loss: {train_loss}")
        print(f"Val loss: {val_loss}")

        if val_loss < best_loss:
            steps_without_improvement = 0
            best_loss = val_loss
            model.cache_current_weights()
        else:
            steps_without_improvement += 1
            if steps_without_improvement > patience:
                print("Early stopping")
                model.load_best_weights()
                break
    
    return model


def main():

    parser = create_argparser()
    args = parser.parse_args()
    backbone = args.backbone
    dataset_path = args.dataset


    device = choose_device()
    net = DeepLab(backbone, 256)
    net = net.to(device)

    criterion = get_loss_function("dice", "segmentation")
    criterion = criterion.to(device)

    optimizer = get_core_optimizer("AdamW")(net.parameters())
    
    dataset = load_dataset(dataset_path, 256, "segmentation")
    train_loader, val_loader, test_loader = prepare_dataloaders(
        dataset=dataset, 
        split_ratio=[0.6, 0.2, 0.2], 
        dataset_type="segmenation", 
        batch_size=16, 
        augment=True
    )

    net = train(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )   

    experiment_id = str(uuid.uuid4())

    nn_config = {
        "net_type": "segmentation",
        "input_size": 256,
        "training_data": dataset_path,
        "backbone": backbone,
        "augment": True,
        "experiment_id": str(experiment_id)
    }

    net.save_model(nn_config, "./training_inference")


if __name__ == "__main__":
    main()
    