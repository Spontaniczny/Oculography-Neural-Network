import pandas as pd
import numpy as np
import torch
from PIL import Image
from typing import Optional
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode


class RegressionDataset(Dataset):
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
        ]) 
    

    def __len__(self):
        return len(self._data)
    
    def ellipse_params_to_tensor(self, img_data: pd.Series) -> torch.Tensor:
        original_w = img_data.w
        original_h = img_data.h

        x_center = img_data.x_center / original_w
        y_center = img_data.y_center / original_h

        major = (img_data.major_axis / original_w)
        minor = (img_data.minor_axis / original_h)
        rotate_angle = img_data.rotate_angle / 180

        params = torch.Tensor([x_center, y_center, major, minor, rotate_angle])
        return params

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_data = self._data.iloc[idx]
        input_img = self.transform_in(Image.open(img_data.frame))
        return input_img, self.ellipse_params_to_tensor(img_data)
    
