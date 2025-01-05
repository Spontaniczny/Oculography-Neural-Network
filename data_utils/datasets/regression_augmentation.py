from random import randrange, uniform
from torch.utils.data import Dataset
from .regression_dataset import RegressionDataset
from torchvision.transforms import v2
from abc import ABC, abstractmethod
import torch


class RegressionTransform(ABC):

    @abstractmethod
    def __call__(self, input_image: torch.Tensor, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass 


class RegressionHorizontalFlip(RegressionTransform):

    def __init__(self):
        super().__init__()

    def __call__(self, input_image: torch.Tensor, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_flipped = torch.flip(input_image, (-1, ))
        flip_params = torch.clone(params)
        flip_params[1] = 1 - params[1]
        flip_params[4] = ((params[4]*180 - 90)*(-1) + 90) / 180
        return image_flipped, flip_params
    

class RegressionResizedCrop(RegressionTransform):
    def __init__(
            self,
            scale: tuple[float, float] = (0.1, 0.3),
        ):
        
        super().__init__()
        self.scale = scale
    
    def draw_crop_parameters(self) -> tuple[float, float, float, float]:
        left = uniform(*self.scale)
        top = uniform(*self.scale)
        height = uniform(1 - self.scale[1], 1.0 - top)
        width = uniform(1 - self.scale[1], 1.0 - left)
        return top, left, height, width

    def __call__(self, input_image: torch.Tensor, ellipse_params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        crop_params = self.draw_crop_parameters()
        input_cropped = self.crop_and_resize_image(input_image, crop_params)
        params_cropped = self.crop_and_resize_params(ellipse_params, crop_params)
        return input_cropped, params_cropped
    

    def crop_and_resize_image(
            self, 
            tensor: torch.Tensor,
            params: tuple[float, float, float, float],
        ) -> torch.Tensor:
        top, left, h, w = params
        _, height, width = tensor.shape

        cropped = v2.functional.resized_crop(
            tensor, 
            int(top*height), 
            int(left*width), 
            int(h*height), 
            int(w*width), 
            [height, width],
            v2.InterpolationMode.BILINEAR
        )
        return cropped
    

    def crop_and_resize_params(
            self, 
            ellipse_params: torch.Tensor,
            params: tuple[float, float, float, float],
        ) -> torch.Tensor:

        top, left, h, w = params
        cropped_params = torch.clone(ellipse_params)
        cropped_params[0] = (ellipse_params[0] - top) / h
        cropped_params[1] = (ellipse_params[1] - left) / w
        cropped_params[2] /= h
        cropped_params[3] /= w
        return cropped_params
    

class RegressionAugmented(Dataset):

    def __init__(
            self, 
            base_dataset: RegressionDataset,
        ) -> None:
        super().__init__()
        
        self.base_dataset = base_dataset
        self.transforms = [
            RegressionHorizontalFlip(),
            RegressionResizedCrop()
        ]
        self.count_transforms = len(self.transforms)


    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        input_img, ellipse_params = self.base_dataset[idx]
        if (choice := randrange(self.count_transforms*2)) < self.count_transforms:
            return self.transforms[choice](input_img, ellipse_params)
        return input_img, ellipse_params
    
