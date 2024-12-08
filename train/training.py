import torch
import torch.nn as nn
from copy import deepcopy
from typing import Callable
from torch.utils.data import DataLoader
from .helper_functions import get_loss_function

def validate_model(
        model: nn.Module,
        val_dataset: torch.utils.data.DataLoader,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: str
    ) -> float:
    
    total_loss = 0.0
    number_of_batches = 0

    model = model.eval()
    with torch.no_grad():
        for images, labels in val_dataset:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            batch_loss = criterion(outputs, labels)
            total_loss += batch_loss.item()
            number_of_batches += 1

    average_loss = total_loss / number_of_batches
    return average_loss

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    max_epochs: int = 50,
    patience: int = 5, # parameter for early stopping
    device: str = "cpu",
):
    
    model.to(device)

    # wandb.watch(model, criterion, log="all")

    best_model, best_loss = model, float("inf")
    steps_without_improvement = 0
    train_losses, val_losses = [], []


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

            if (i + 1) % 10 == 0:
                print(f"Batch: {i + 1}")
                # wandb.log({
                #     "batch": number_of_batches,
                #     "Current_average_loss": epoch_loss / number_of_batches
                # })
        
        train_loss = epoch_loss / number_of_batches
        train_losses.append(train_loss)

        val_loss = validate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # wandb.log({
        #     "Train average loss": train_loss,
        #     "Validation average loss": val_loss
        # })

        print("Train average loss", train_loss)
        print("Validation average loss", val_loss)
        if val_loss < best_loss:
            steps_without_improvement = 0
            best_loss = val_loss
            best_model = deepcopy(model)
        else:
            steps_without_improvement += 1
            if steps_without_improvement > patience:
                print("Early stopping")
                # wandb.log({
                #     "Early stopping": epoch,
                # })

    return best_model
