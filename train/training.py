import torch
from torch import optim
import torch.nn as nn
from copy import deepcopy
from typing import Callable
from losses.segmentation.losses import DSCLoss, IoULoss
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data_utils.segmentation import prepare_segmentation_dataset, SegmentationDataset
from torch.utils.data import DataLoader, Dataset, random_split
import torch
from models.segmentation.deeplabv3 import DeepLab


def choose_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"




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


def plot_training_stats(train_loss: list[float], val_loss: list[float]):
    epochs = len(train_loss)
    plt.xticks(range(epochs))

    plt.plot(train_loss, color='red', label="Training loss")
    plt.plot(val_loss, color='blue', label="Validation loss")

    plt.xlabel("Epoch number")
    plt.ylabel("Average batch loss")

    plt.legend()
    plt.show()


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_epochs: int = 200,
    patience: int = 3, # parameter for early stopping
    learning_rate: float = 1e-3,
    device: str = "cpu"
):
    
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = DSCLoss()

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
                print(f"Batch {i + 1}")
        
        train_loss = epoch_loss / number_of_batches
        train_losses.append(train_loss)
        print(f"Train batch loss: {train_loss:.6f}")

        val_loss = validate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        print(f"Validation average loss: {val_loss:.6f}")

        if val_loss < best_loss:
            steps_without_improvement = 0
            best_loss = val_loss
            best_model = deepcopy(model)
        else:
            steps_without_improvement += 1
            if steps_without_improvement > patience:
                print("Early stopping")
                break

    plot_training_stats(train_losses, val_losses)
    return best_model


if __name__ == "__main__":
    ds = prepare_segmentation_dataset("datasets/ratEye/01", input_size=128)
    train_loader, val_loader, test_loader = prepare_dataloaders(ds, [0.6, 0.2, 0.2], batch_size=16)

    device = choose_device()
    net = DeepLab()
    net = train(net, train_loader, val_loader, 200, patience=25, device=device)