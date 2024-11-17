from random import randrange, uniform
from scipy.stats import halfcauchy
from torch.utils.data import Dataset
from torchvision.transforms import v2
from .segmentaion_dataset import SegmentationDataset
from abc import ABC, abstractmethod
import torch
from torchvision.io import decode_image



class Transform(ABC):

    @abstractmethod
    def __call__(self, input_image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pass 


class HorizontalFlip(Transform):

    def __init__(self):
        super().__init__()

    def __call__(self, input_image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.flip(input_image, (-1, )), torch.flip(mask, (-1, ))
    

class ResizedCrop(Transform):
    def __init__(
            self,
            scale: tuple[float, float] = (0.04, 0.2),
            ratio: tuple[float, float] = (0.75, 1.3333333333333333),
            input_interpolation: v2.InterpolationMode = v2.InterpolationMode.BILINEAR,
            mask_interpolation: v2.InterpolationMode = v2.InterpolationMode.NEAREST
        ):
        
        super().__init__()
        self.scale = scale
        self.ratio = ratio
        self.input_interpolation = input_interpolation
        self.mask_interpolation = mask_interpolation
    
    def draw_crop_parameters(self) -> tuple[float, float, float, float]:
        left = uniform(*self.scale)
        top = uniform(*self.scale)
        width = uniform(self.scale[1], 1.0 - left)
        r_ratio = uniform(*self.ratio)
        height = max(1.0 - top, width / r_ratio)

        return top, left, width, height

    def __call__(self, input_image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        params = self.draw_crop_parameters()
        mask_cropped = self.crop_and_resize(mask, params, self.input_interpolation)
        input_cropped = self.crop_and_resize(input_image, params, self.mask_interpolation)
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
    

class AugmentedDataset(Dataset):

    def __init__(self, base_dataset: SegmentationDataset) -> None:
        super().__init__()
        
        self.base_dataset = base_dataset
        self.input_size = base_dataset.input_image_size
        self.transforms = [
            HorizontalFlip(),
            # ResizedCrop()
        ]

        self.count_transforms = len(self.transforms)


    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        input_img, mask = self.base_dataset[idx]
        if (choice := randrange(self.count_transforms*2)) < self.count_transforms:
            return self.transforms[choice](input_img, mask)
        return input_img, mask
    
    
if __name__ == "__main__":
    image = decode_image("../datasets/mouse/frames/2P-fc2_save_20200806_1_0002.jpg")
    mask = decode_image("../datasets/mouse/annotations/2P-fc2_save_20200806_1_0002.png")

    flip = ResizedCrop()
    image, mask = flip(image, mask)
    
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(image.permute(1, 2, 0), cmap="grey")
    ax[1].imshow(mask.permute(1, 2, 0), cmap="grey") 
    ax[0].axis("off")
    ax[1].axis("off")
    plt.show()
