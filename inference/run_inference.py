import os
import torch
import torchvision.transforms.v2 as v2
from PIL import Image
import numpy as np
from .model_loading import load_config_file, load_model
from data_utils.datasets import InferenceDataset
from torch.utils.data import DataLoader
import argparse
from models import BaseNet
from train.helper_functions import choose_device
from ellipse import find_ellipse


def apply_mask_on_original(original_rgba: Image.Image, mask: np.ndarray) -> Image.Image:
    if mask.dtype != np.bool:
        mask = mask > 0.5

    black_image = np.zeros((*mask.shape, 4), np.uint8)
    black_image[mask] = np.array([0, 255, 0, 64])
    green_mask = Image.fromarray(black_image, "RGBA")
    annotated = Image.alpha_composite(original_rgba, green_mask)
    return annotated

def infer_dataset(
        model: BaseNet, 
        loader: DataLoader,
        save_folder: str, 
        save_width: int, 
        save_height: int,
        remove_artifacts: bool,
        device: str,
        max_images: int
    ) -> None:
    
    saving_path = f"results/{save_folder}"
    if not os.path.exists(saving_path):
        os.mkdir(saving_path)
    
    resize_bilinear = v2.Resize((save_width, save_height), interpolation=v2.InterpolationMode.BILINEAR)
    resize_nearest = v2.Resize((save_width, save_height), interpolation=v2.InterpolationMode.NEAREST)
    to_pil = v2.ToPILImage()

    model.eval()

    frame_count = 0
    for batch in loader:
        
        out = model.predict_mask(batch.to(device))
        out = out.to("cpu")
        batch = batch.to("cpu")

        for i, (input_image, result_mask) in enumerate(zip(batch, out)):
            print(i)
            if remove_artifacts:
                mask, ellipse = find_ellipse(result_mask)
                result_mask = mask.float()
                # print(mask.dtype, result_mask.dtype)

            in_im: Image.Image = to_pil(resize_bilinear(input_image))
            in_im.save(f"{saving_path}/frame_{frame_count:05d}.png", format="png")

            res_im: Image.Image = to_pil(resize_nearest(result_mask))
            res_im.save(f"{saving_path}/mask_{frame_count:05d}.png", format="png")

            in_im = to_pil(input_image).convert("RGBA")
            annotation_image = apply_mask_on_original(in_im, result_mask.numpy().squeeze())
            annotation_image.resize((save_width, save_height)).save(f"{saving_path}/annotated_{frame_count:05d}.png", format="png")

            frame_count += 1
            if frame_count >= max_images >= 0:
                return


def argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--save_width",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--save_height",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--save_folder",
        type=str,
        default="1",
    )

    parser.add_argument(
        "--remove_artifacts",
        type=bool,
        default=False
    )

    parser.add_argument(
        "--max_images",
        type=int,
        default=-1
    )

    return parser


def main():
    
    parser = argparser()
    args = parser.parse_args()

    device = choose_device()

    config = load_config_file(args.model_config)
    model = load_model(config, args.model_config).to(device)

    ds = InferenceDataset(args.dataset_path)
    data_loader = DataLoader(ds, batch_size=32, shuffle=True)

    infer_dataset(
        model, 
        data_loader, 
        save_folder=args.save_folder,
        save_width=args.save_height, 
        save_height=args.save_height,
        remove_artifacts=args.remove_artifacts,
        device=device,
        max_images=args.max_images
    )


if __name__ == "__main__":
    main()
