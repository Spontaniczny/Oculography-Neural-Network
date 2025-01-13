import torch
import torch.nn as nn
from typing import Callable
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import os

from training_experiments.models.segmentation import DeepLab
from training_experiments.train.helper_functions import choose_device

from PIL import Image
from tqdm import tqdm
from training_experiments.data_utils.datasets import InferenceDataset
from torch.utils.data import DataLoader
from training_experiments.inference import load_config_file, load_model
import argparse

import matplotlib.pyplot as plt
import pandas as pd
from .video_reader import BatchVideoReader


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to directory with data to annotate"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to configuration of a model that will be used to label data"
    )

    return parser


def save_results_with_plot(saving_path: str, pupil_sizes: list[int]):
    size_over_time = pd.DataFrame.from_dict({
        "pupil_size": pupil_sizes
    })
    size_over_time.to_csv(f"{saving_path}/pupil_size.csv")

    plt.plot(pupil_sizes)
    plt.xlabel("Time")
    plt.ylabel("Relative pupil size")
    plt.title("Change of pupil size over time")
    plt.show()


def infer_dataset(
        model: DeepLab, 
        loader: DataLoader,
        save_folder: str, 
        remove_artifacts: bool,
        device: str,
    ) -> None:
    
    saving_path = f"training_inference/results/{save_folder}"
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    model.eval()
    frame_count = 0
    pupil_sizes = []
    
    with tqdm(total=len(loader.dataset), desc="Frames annotated", unit="steps") as pbar:
        for batch in loader:
            out = model.predict_mask(batch.to(device))

            for i, (input_image, result_mask) in enumerate(zip(batch, out)):
                pupil_sizes.append(int(result_mask.sum().to("cpu").item()))
                frame_count += 1
                pbar.update(1)

    save_results_with_plot(saving_path, pupil_sizes)


def infer_video(
        model: DeepLab,
        video_reader: BatchVideoReader,
        save_folder: str, 
        remove_artifacts: bool,
        device: str,
):
    saving_path = f"training_inference/results/{save_folder}"
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    

    model.eval()
    pupil_sizes = []
    
    with tqdm(total=len(video_reader), desc="Frames annotated", unit="steps") as pbar:
        while video_reader.video_open:
            frames = video_reader.get_next()
            out = model.predict_mask(frames)
            pupil_sizes.append(
                out.sum(axis=(1, 2, 3)).to("cpu")
            )
            pbar.update(len(frames))


def main():
    parser = create_argparser()
    args = parser.parse_args()
    model_config_path = args.model
    dataset_path = args.dataset

    device = choose_device()
    config = load_config_file(model_config_path)
    model = load_model(config, model_config_path).to(device)

    if os.path.isdir(dataset_path):
        ds = InferenceDataset(dataset_path)
        data_loader = DataLoader(
            ds, 
            batch_size=128,
            num_workers=6,
            prefetch_factor=6,
        )
        infer_dataset(
            model=model,
            loader=data_loader,
            save_folder="10",
            remove_artifacts=True,
            device=device
        )
    else:
        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((256, 256), v2.InterpolationMode.BILINEAR),
        ])

        video_reader = BatchVideoReader(
            video_path=dataset_path, 
            batch_size=32,
            image_transform=transform,
            device=device
        )

        infer_video(
            model=model,
            video_reader=video_reader,
            save_folder="10",
            remove_artifacts=True,
            device=device
        )
        

if __name__ == "__main__":
    main()
    