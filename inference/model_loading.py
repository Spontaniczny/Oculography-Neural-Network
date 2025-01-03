import json
import os
import torch
from typing import Any, Optional
from torch.utils.data import DataLoader
from data_utils import load_dataset
from models.segmentation import DeepLab, U_NET
from models.regression import EllipseNet
from models import BaseNet


def load_config_file(config_path: str) -> dict[str, Any]:
    with open(config_path, "r") as f:
        config = json.loads(f.read())

    return config

def load_model(config: dict[str, Any], config_path: str) -> BaseNet:
    if config["net_type"] == "segmentation":
        if config["backbone"] == "u_net":
            model = U_NET(
                input_size=config["input_size"],
                upsampling_method=config["upsampling"]
            )
        else:
            model = DeepLab(
                backbone=config["backbone"],
                input_size=config["input_size"]
            )
    else:
        model = EllipseNet(
            backbone=config["backbone"],
            input_size=config["input_size"]
        )

    state_dict_path = "/".join(os.path.normpath(config_path).split("/")[:-1])
    state_dict_path += f"/{config["experiment_id"]}.pt"
    model.load_state_dict(torch.load(state_dict_path, weights_only=True))

    return model


def load_dataloader(config: dict[str, Any], dataset: Optional[str] = None, batch_size: int = 128) -> DataLoader:
    if dataset is None:
        dataset = config["training_data"]
    
    ds = load_dataset(dataset, config["input_size"], config["net_type"])
    data_loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return data_loader