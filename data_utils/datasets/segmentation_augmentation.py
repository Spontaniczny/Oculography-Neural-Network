from random import randrange, uniform
from torch.utils.data import Dataset
from torchvision.transforms import v2
from .segmentation_dataset import SegmentationDataset
from abc import ABC, abstractmethod
import torch
from torchvision.io import decode_image



class SegmentationTransform(ABC):

    @abstractmethod
    def __call__(self, input_image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass 


class SegmentationHorizontalFlip(SegmentationTransform):

    def __init__(self):
        super().__init__()

    def __call__(self, input_image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.flip(input_image, (-1, )), torch.flip(mask, (-1, ))
    

class SegmentationResizedCrop(SegmentationTransform):
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

    def __call__(self, input_image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        params = self.draw_crop_parameters()
        input_cropped = self.crop_and_resize(input_image, params, v2.InterpolationMode.BILINEAR)
        mask_cropped = self.crop_and_resize(mask, params, v2.InterpolationMode.NEAREST)
        return input_cropped, mask_cropped
    

    def crop_and_resize(
            self, 
            tensor: torch.Tensor,
            params: tuple[float, float, float, float],
            interpolation: v2.InterpolationMode
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
            interpolation
        )
        return cropped
    

class SegmentationAugmented(Dataset):

    def __init__(
            self, 
            base_dataset: SegmentationDataset,
            
        ) -> None:
        super().__init__()
        
        self.base_dataset = base_dataset
        self.transforms = [
            SegmentationHorizontalFlip(),
            SegmentationResizedCrop()
        ]
        self.count_transforms = len(self.transforms)


    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        input_img, mask = self.base_dataset[idx]
        if (choice := randrange(self.count_transforms*2)) < self.count_transforms:
            return self.transforms[choice](input_img, mask)
        return input_img, mask
    
