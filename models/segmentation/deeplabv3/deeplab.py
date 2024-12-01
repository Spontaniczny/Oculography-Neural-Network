import torch
import torch.nn.functional as F
import torch.nn as nn
from ...backbones import init_backbone


class ASPPConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        
    def forward(self, x):
        return self.model(x)


class ASPPPooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, height, width = x.shape
        x = self.model(x)
        return F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)

class ASPP(nn.Module):
    
    def __init__(self, in_channels: int, atrous_rates: list[int], out_channels: int = 256) -> None:
        super().__init__()

        basic_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU()
        )

        self.convs = nn.ModuleList(
            [basic_conv] + \
            [ASPPConv(in_channels, out_channels, rate) for rate in atrous_rates] + \
            [ASPPPooling(in_channels, out_channels)]
        )

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        project = self.project(res)
        
        return project
    

class DeepLab(nn.Module):

    def __init__(
            self, 
            backbone: str = "res_net_50"
        ):
        super().__init__()
        self.backbone = init_backbone(backbone)
        self.segmentation = ASPP(self.backbone.output_channels, [2, 4, 8])
        self.final = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 1, 1),
        )

    def forward(self, x):
        B, C, heigth, width = x.shape
        x = self.backbone(x)
        x = self.segmentation(x)
        x = self.final(x)
        return F.interpolate(x, size=(heigth, width), mode="bilinear", align_corners=False)
    
