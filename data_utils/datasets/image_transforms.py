import torch
import torch.nn as nn

def transform_image(img_tensor: torch.Tensor) -> torch.Tensor:
    """
    Transform image to (channel, width, height) format where channel = 1
    """
    dim = len(img_tensor.shape)
    if dim == 2:
        return img_tensor.unsqueeze(0)
    elif dim == 3 or dim == 4:
        return img_tensor[0].unsqueeze(0)
    else:
        raise ValueError("Provided images have incorrect format")
    

class CorrectFormat(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return transform_image(img)