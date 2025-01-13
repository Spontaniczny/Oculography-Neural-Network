import cv2
from PIL import Image
import torch
import numpy as np
from torchvision.transforms import v2


class BatchVideoReader:

    def __init__(
            self, 
            video_path: str, 
            batch_size: int, 
            image_transform: v2.Compose,
            device: str
        ):

        self.video_path = video_path
        self.video_capture = cv2.VideoCapture(self.video_path)
        self.batch_size = batch_size
        self.transforms = image_transform
        self.device = device
        self.video_open = True
    
    def get_next(self) -> torch.Tensor:
        
        frames = []
        for _ in range(self.batch_size):
            ret, frame = self.video_capture.read()
            if not ret:
                self.video_capture.release()
                self.video_open = False
                break

            img = Image.fromarray(frame[:, :, 0])
            transformed_img = self.transforms(img).to(self.device)
            frames.append(transformed_img)

        return torch.stack(frames)
    
    def __len__(self) -> int:
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))