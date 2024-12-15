import torch
import torch.nn as nn
from typing import Callable
from torch.utils.data import DataLoader
from models import BaseNet
from callbacks import TrainingLogger

def validate_model(
        model: BaseNet,
        val_dataset: torch.utils.data.DataLoader,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        device: str,
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
    model: BaseNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    logger: TrainingLogger,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    max_epochs: int = 50,
    patience: int = 5, # parameter for early stopping
    device: str = "cpu",
):

    model.to(device)
    # logger.model_watch(model, criterion, log_step=100)

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

            if (i + 1) % 10 == 0:
                logger.on_batch_end(i + 1, epoch_loss)
        
        train_loss = epoch_loss / number_of_batches
        val_loss = validate_model(model, val_loader, criterion, device)
        logger.on_epoch_end(epoch, train_loss, val_loss)

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
    
    logger.on_training_end()
    return model
