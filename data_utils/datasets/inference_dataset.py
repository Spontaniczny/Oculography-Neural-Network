import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode


class InferenceDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        net_input_size: int = 256
    ) -> None:
        super().__init__()

        self.transform_in = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((net_input_size, net_input_size), InterpolationMode.BILINEAR),
        ])
        self.image_paths = self.get_images_in_dir(image_dir)

    def get_images_in_dir(self, dir_path: str) -> list[str]:
        file_names = os.listdir(dir_path)

        images = filter(
            lambda filename: filename.lower().endswith(('.jpg', '.png')), 
            file_names
        )
        image_paths = list(map(lambda image: f"{dir_path}/{image}", images))
        
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        input_img = self.transform_in(Image.open(img_path))
        return input_img[0].unsqueeze(0)