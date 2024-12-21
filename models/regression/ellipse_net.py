import torch
import torch.nn as nn
from ..backbones import init_backbone
from .. import BaseNet
from ellipse import Ellipse
from typing import Optional

class EllipseNet(BaseNet):
    def __init__(self, 
        backbone: str,
        input_size: int,
        channel_reduction: int = 64
    ):

        super().__init__()

        self.input_size = input_size
        self.backbone = init_backbone(backbone)


        self.channel_reduce = nn.Sequential(
            nn.Conv2d(
                in_channels=self.backbone.output_channels, 
                out_channels=channel_reduction,
                kernel_size=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2)

        )

        backbone_out_size = (input_size // self.backbone.output_stride) ** 2 // 4

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Linear(channel_reduction*backbone_out_size // 2, 1024),
            self.dropout,
            self.relu,
            nn.Linear(1024, 512),
            self.relu,
            nn.Linear(512, 5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.channel_reduce(x)
        x = self.mlp(x)
        return x
    
    @torch.inference_mode()
    def predict_mask(self, batch: torch.Tensor, threshold: Optional[float] = 0.5) -> torch.Tensor:
        params = self(batch)
        B, C, w, h = batch.shape
        ellipses = []
        for ellipse_params in params:
            ellipse = Ellipse(*(ellipse_params[:-1]*w), ellipse_params[-1].item()*180, (w, h))
            ellipses.append(torch.Tensor(ellipse.draw_ellipse()))
        
        return torch.stack(ellipses).reshape(-1, 1, w, h)
    