import pandas as pd
import torch
from PIL import Image
from typing import Optional
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
from .image_transforms import CorrectFormat


class SegmentationDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        net_input_size: Optional[int] = None
    ) -> None:
        super().__init__()

        self._data = metadata
        self._original_image_width = int(metadata['w'].mean())
        self._original_image_height = int(metadata['h'].mean())

        if net_input_size is None:
            net_input_size = min(self._original_image_width, self._original_image_height)

        if net_input_size % 16:
            net_input_size -= net_input_size % 16

        self.net_input_size = net_input_size

        self.transform_in = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((net_input_size, net_input_size), InterpolationMode.BILINEAR),
            CorrectFormat()
        ]) 

        self.transform_out = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((net_input_size, net_input_size), InterpolationMode.NEAREST),
            CorrectFormat()
        ])    

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        img_data = self._data.iloc[idx]
        input_img = self.transform_in(Image.open(img_data.frame))
        output_img = self.transform_out(Image.open(img_data.annotation))
        return input_img, output_img
