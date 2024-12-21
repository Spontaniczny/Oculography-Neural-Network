import wandb
import torch
import torch.nn as nn
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv
import os
from PIL import Image
import matplotlib.pyplot as plt
import wandb.integration

def authenticate_wandb() -> bool:
    if not load_dotenv():
        raise ValueError("Could not find dotenv file")
    
    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key is None:
        raise ValueError(
            '''
            Environmental variable WANDB_API_KEY needs to be set in order 
            to authenticate.
            '''
        )
    
    if not wandb.login(verify=True):
        raise ValueError("Invalid authentication key")
    
    return True


class TrainingLogger:

    def __init__(self, project_name: str, run_name: str, config: dict):
        self.project_name = project_name
        self.run_name = run_name
        self.config = config

        authenticate_wandb()
        wandb.init(project=project_name, name=run_name, config=config)

        self.training_losses: list[float] = []
        self.validation_losses: list[float] = []

    def model_watch(self, model: nn.Module, criterion: nn.Module, log_step: int):
        wandb.watch(model, criterion, log="all", log_freq=log_step)

    def on_batch_end(self, batch: int, current_loss: float):
        batch_average_loss = current_loss / batch
        print(f"Batch: {batch}")
        print(f"Current batch loss: {batch_average_loss}")


    def on_epoch_end(self, epoch: int, train_loss: float, val_loss: float):
        print(f"End of epoch {epoch + 1}")
        print(f"Training loss: {train_loss}")
        print(f"Validation loss: {val_loss}")

        wandb.log({
            "Epoch": epoch,
            "Training loss": train_loss,
            "Validation loss": val_loss,
        })

        self.training_losses.append(train_loss)
        self.validation_losses.append(val_loss)


    def save_onnx_model(self, model: nn.Module, input_tensor: torch.Tensor):
        model.to("cpu")
        saving_path = "tmp/model.onnx"
        torch.onnx.export(model, input_tensor.to("cpu"), saving_path)
        wandb.save(saving_path)

    def on_training_end(self):
        epochs = len(self.training_losses)
        plt.xticks(range(epochs))
        plt.plot(self.training_losses, color='red', label="Training loss")
        plt.plot(self.validation_losses, color='blue', label="Validation loss")
        plt.xlabel("Epoch number")
        plt.ylabel("Average batch loss")
        plt.legend()

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        self.save_fig(fig_name="Train_val_loss_plot", buffer=buffer)
        plt.close()


    def save_fig(self, fig_name: str, buffer: BytesIO):
        wandb.summary[fig_name] = wandb.Image(Image.open(buffer))

    def save_scalar_metrics(self, metrics: dict[str, float], metrics_name: str):
        wandb.summary[metrics_name] = metrics

    def save_metrics_table(self, metrics: dict[str, torch.Tensor], metrics_name: str):
        for key in metrics.keys():
            metrics[key] = metrics[key].to("cpu")

        df = pd.DataFrame.from_dict(metrics)
        table = wandb.Table(dataframe=df)
        wandb.summary[metrics_name] = table


if __name__ == "__main__":
    pass