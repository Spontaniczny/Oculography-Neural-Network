import json
import os
import torch
import torchvision.transforms.v2 as v2
from typing import Any, Optional
from torch.utils.data import DataLoader
from data_utils import load_dataset
from models.segmentation import DeepLab
from models.regression import EllipseNet
from models import BaseNet
from PIL import Image
import numpy as np


def load_config_file(config_path: str) -> dict[str, Any]:
    with open(config_path, "r") as f:
        config = json.loads(f.read())

    return config

def load_model(config: dict[str, Any], config_path: str) -> BaseNet:
    if config["net_type"] == "segmentation":
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


def load_dataloader(config: dict[str, Any], dataset: Optional[str] = None) -> DataLoader:
    if dataset is None:
        dataset = config["training_data"]
    
    ds = load_dataset(dataset, config["input_size"], config["net_type"])
    data_loader = DataLoader(ds, batch_size=16)
    return data_loader


def apply_mask_on_original(original_rgba: Image.Image, mask: np.ndarray) -> Image.Image:
    if mask.dtype != np.bool:
        mask = mask > 0.5

    black_image = np.zeros((*mask.shape, 4), np.uint8)
    black_image[mask] = np.array([0, 255, 0, 64])
    green_mask = Image.fromarray(black_image, "RGBA")
    annotated = Image.alpha_composite(original_rgba, green_mask)
    return annotated


def save_results(input_batch: torch.Tensor, result_masks: torch.Tensor, save_folder: str):
    saving_path = f"results/{save_folder}"
    if not os.path.exists(saving_path):
        os.mkdir(saving_path)
    
    resize_bilinear = v2.Resize((512, 512), interpolation=v2.InterpolationMode.BILINEAR)
    resize_nearest = v2.Resize((512, 512), interpolation=v2.InterpolationMode.NEAREST)
    to_pil = v2.ToPILImage()

    for i, (input_image, result_mask) in enumerate(zip(input_batch, result_masks)):
        in_im: Image.Image = to_pil(resize_bilinear(input_image))
        in_im.save(f"{saving_path}/frame_{i:05d}.png", format="png")

        res_im: Image.Image = to_pil(resize_nearest(result_mask))
        res_im.save(f"{saving_path}/mask_{i:05d}.png", format="png")

        in_im = to_pil(input_image).convert("RGBA")
        annotation_image = apply_mask_on_original(in_im, result_mask.numpy().squeeze())
        annotation_image.resize((512, 512)).save(f"{saving_path}/annotated_{i:05d}.png", format="png")



def main():
    config_path = "saved_models/deeplab/15:12:2024-20:34:55.json"
    config = load_config_file(config_path)
    model = load_model(config, config_path)
    dataset_path = "datasets/rat_eye/40"

    data_loader = load_dataloader(config, dataset_path)
    batch, _ = next(iter(data_loader))

    model.eval()
    with torch.no_grad():
        out = model.predict_mask(batch)
    
    # print(out)
    # print(out.shape)
    save_results(batch, out, "12")


if __name__ == "__main__":
    main()
