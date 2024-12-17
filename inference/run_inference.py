import os
import torch
import torchvision.transforms.v2 as v2
from PIL import Image
import numpy as np
from .model_loading import load_dataloader, load_config_file, load_model


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
    config_path = "saved_models/ellipsenet/17:12:2024-14:42:43_finetuning.json"
    config = load_config_file(config_path)

    model = load_model(config, config_path)
    dataset_path = "datasets/rat_eye/01|02"
    data_loader = load_dataloader(config, dataset_path)

    batch, _ = next(iter(data_loader))

    model.eval()
    with torch.no_grad():
        out = model.predict_mask(batch)
    
    save_results(batch, out, "3")


if __name__ == "__main__":
    main()
