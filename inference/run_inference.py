import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as v2

from PIL import Image
from tqdm import tqdm
from data_utils.datasets import InferenceDataset
from torch.utils.data import DataLoader
from models import BaseNet
from train.helper_functions import choose_device
from ellipse import remove_noise
from .model_loading import load_config_file, load_model
from .argparser import argparser



def apply_mask_on_original(original_rgba: Image.Image, mask: np.ndarray) -> Image.Image:
    if mask.dtype != np.bool:
        mask = mask > 0.5

    black_image = np.zeros((*mask.shape, 4), np.uint8)
    black_image[mask] = np.array([0, 255, 0, 64])
    green_mask = Image.fromarray(black_image, "RGBA")
    annotated = Image.alpha_composite(original_rgba, green_mask)
    return annotated


def save_results_with_plot(save_folder: str, pupil_sizes: int):
    size_over_time = pd.DataFrame.from_dict({
        "pupil_size": pupil_sizes
    })
    size_over_time.to_csv(f"results/{save_folder}/pupil_size.csv")

    plt.plot(pupil_sizes)
    plt.xlabel("Time")
    plt.ylabel("Relative pupil size")
    plt.title("Change of pupil size over time")
    plt.show()

    

def infer_dataset(
        model: BaseNet, 
        loader: DataLoader,
        save_folder: str, 
        save_width: int, 
        save_height: int,
        remove_artifacts: bool,
        device: str,
        save_step: int
    ) -> None:
    
    saving_path = f"results/{save_folder}"
    if not os.path.exists(saving_path):
        os.mkdir(saving_path)
    
    resize_bilinear = v2.Resize((save_width, save_height), interpolation=v2.InterpolationMode.BILINEAR)
    resize_nearest = v2.Resize((save_width, save_height), interpolation=v2.InterpolationMode.NEAREST)
    to_pil = v2.ToPILImage()

    model.eval()
    frame_count = 0

    pupil_sizes = []
    
    with tqdm(total=len(loader.dataset), desc="Frames annotated", unit="steps") as pbar:
        for batch in loader:
            out = model.predict_mask(batch.to(device))
            out = out.to("cpu")
            batch = batch.to("cpu")

            for i, (input_image, result_mask) in enumerate(zip(batch, out)):
                if remove_artifacts:
                    result_mask, ellipse = remove_noise(result_mask, find_ellipse=True)
                    result_mask = result_mask.mask.float()
                    # print(mask.dtype, result_mask.dtype)

                if save_step > 0 and frame_count % save_step == 0:
                    in_im: Image.Image = to_pil(resize_bilinear(input_image))
                    in_im.save(f"{saving_path}/frame_{frame_count:05d}.png", format="png")

                    res_im: Image.Image = to_pil(resize_nearest(result_mask))
                    res_im.save(f"{saving_path}/mask_{frame_count:05d}.png", format="png")

                    in_im = to_pil(input_image).convert("RGBA")
                    annotation_image = apply_mask_on_original(in_im, result_mask.numpy().squeeze())
                    annotation_image.resize((save_width, save_height)).save(f"{saving_path}/annotated_{frame_count:05d}.png", format="png")

                pupil_sizes.append(result_mask.sum())
                frame_count += 1
                pbar.update(1)

    save_results_with_plot(save_folder, pupil_sizes)

def main():
    
    parser = argparser()
    args = parser.parse_args()

    device = choose_device()

    config = load_config_file(args.model_config)
    model = load_model(config, args.model_config).to(device)

    ds = InferenceDataset(args.dataset_path)

    data_loader = DataLoader(
        ds, 
        batch_size=128,
        num_workers=6,
        prefetch_factor=6,
    )

    infer_dataset(
        model, 
        data_loader, 
        save_folder=args.save_folder,
        save_width=args.save_height, 
        save_height=args.save_height,
        remove_artifacts=args.remove_artifacts,
        device=device,
        save_step=args.save_step
    )


if __name__ == "__main__":
    main()
