import torch
import torch.nn as nn
from ..backbones import init_backbone

class EllipseNet(nn.Module):
    def __init__(self, 
        
        input_size: int,
        backbone: str = "res_net_18",
        channel_reduction: int = 64
    ):

        super().__init__()

        self.input_size = input_size
        self.backbone = init_backbone(backbone)

        self.conv = nn.Conv2d(
            in_channels=self.backbone.output_channels, 
            out_channels=channel_reduction,
            kernel_size=1
        )

        backbone_out_size = (input_size // self.backbone.output_stride) ** 2

        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(channel_reduction*backbone_out_size, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 5)
        

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.conv(x)
        x = self.flatten(x)

        x = self.linear1(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x
    