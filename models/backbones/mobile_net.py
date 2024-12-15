import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import Backbone


class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                groups=in_channels, bias=False, padding=1, stride=stride)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    

class SEBlock(nn.Sequential):
    def __init__(self, in_channels: int, inter_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, inter_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(inter_channels, in_channels, kernel_size=1),
            nn.Hardsigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.layers(x)
        return x * y


class InvertedResidual(nn.Module):
    
    def __init__(
            self, 
            in_dim: int, 
            hidden_dim: int, 
            out_dim: int, 
            kernel: int, 
            stride: int,
            use_se: bool = True,
            inter_channels: int = 32,
            activation_name: str = "hardswish"
        ):
        super().__init__()

        assert activation_name in ["hardswish", "relu"], "Wrong activation"

        self.activation = nn.ReLU(inplace=True) if activation_name == "relu" else nn.Hardswish()
        self.skip = (stride == 1) and (in_dim == out_dim)

        modules = []
        if in_dim != hidden_dim:
            modules.extend([
                nn.Conv2d(in_dim, hidden_dim, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                self.activation
            ])
        
        modules.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel, stride=stride, padding=(kernel - 1) // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            self.activation
        ])


        if use_se:
            modules.append(SEBlock(hidden_dim, inter_channels=inter_channels))

        modules.extend([
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_dim),
        ])

        self.model = nn.Sequential(
            *modules
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x)
        x = x + y if self.skip else y
        return x


class MobileNetSmall(Backbone):
    def __init__(self):
        
        super().__init__()
        
        self.first_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish()
        )

        self.inverted_residuals = nn.Sequential(
            InvertedResidual(16, 16, 16, kernel=3, stride=2, use_se=True, inter_channels=8),
            InvertedResidual(16, 72, 24, kernel=3, stride=2, use_se=False, activation_name="relu"),
            InvertedResidual(24, 88, 24, kernel=3, stride=1, use_se=False, activation_name="relu"),
            InvertedResidual(24, 96, 40, kernel=5, stride=1, use_se=True, inter_channels=24),
            InvertedResidual(40, 240, 40, kernel=5, stride=1, use_se=True, inter_channels=64),
            InvertedResidual(40, 240, 40, kernel=5, stride=1, use_se=True, inter_channels=64),
            InvertedResidual(40, 120, 48, kernel=5, stride=1, use_se=True, inter_channels=32),
            InvertedResidual(48, 144, 48, kernel=5, stride=1, use_se=True, inter_channels=40),
            InvertedResidual(48, 288, 96, kernel=5, stride=1, use_se=True, inter_channels=72),
            InvertedResidual(96, 576, 96, kernel=5, stride=1, use_se=True, inter_channels=144),
            InvertedResidual(96, 576, 96, kernel=5, stride=1, use_se=True, inter_channels=144),
        )

        self.last_block = nn.Sequential(
            nn.Conv2d(96, 576, kernel_size=1, bias=False),
            nn.BatchNorm2d(576),
            nn.Hardswish()
        )

        self._output_channels = 576
        self._output_stride = 8
        

    def forward(self, x):
        x = self.first_block(x)
        x = self.inverted_residuals(x)
        x = self.last_block(x)
        return x
    
    @property
    def output_channels(self):
        return self._output_channels
    
    @property
    def output_stride(self):
        return self._output_stride



class MobileNetLarge(Backbone):
    def __init__(self):
        
        super().__init__()
        
        self.first_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish()
        )

        self.inverted_residuals = nn.Sequential(
            InvertedResidual(16, 16, 16, kernel=3, stride=1, use_se=False, activation_name="relu"),
            InvertedResidual(16, 64, 24, kernel=3, stride=2, use_se=False, activation_name="relu"),
            InvertedResidual(24, 72, 24, kernel=3, stride=1, use_se=False, activation_name="relu"),
            InvertedResidual(24, 72, 40, kernel=5, stride=1, use_se=True, inter_channels=24, activation_name="relu"),
            InvertedResidual(40, 120, 40, kernel=5, stride=1, use_se=True, inter_channels=32, activation_name="relu"),
            InvertedResidual(40, 120, 40, kernel=5, stride=1, use_se=True, inter_channels=32, activation_name="relu"),
            InvertedResidual(40, 240, 80, kernel=3, stride=2, use_se=False),
            InvertedResidual(80, 200, 80, kernel=3, stride=1, use_se=False),
            InvertedResidual(80, 184, 80, kernel=3, stride=1, use_se=False),
            InvertedResidual(80, 184, 80, kernel=3, stride=1, use_se=False),
            InvertedResidual(80, 480, 112, kernel=3, stride=1, use_se=True, inter_channels=120),
            InvertedResidual(112, 672, 112, kernel=3, stride=1, use_se=True, inter_channels=168),
            InvertedResidual(112, 672, 160, kernel=5, stride=1, use_se=True, inter_channels=168),
            InvertedResidual(160, 960, 160, kernel=5, stride=1, use_se=True, inter_channels=240),
            InvertedResidual(160, 960, 160, kernel=5, stride=1, use_se=True, inter_channels=240),
        )
        
        self.last_block = nn.Sequential(
            nn.Conv2d(160, 960, kernel_size=1, bias=False),
            nn.BatchNorm2d(960),
            nn.Hardswish()
        )

        self._output_channels = 960
        self._output_stride = 8
        

    def forward(self, x):
        x = self.first_block(x)
        x = self.inverted_residuals(x)
        x = self.last_block(x)
        return x
    
    @property
    def output_channels(self):
        return self._output_channels
    
    @property
    def output_stride(self):
        return self._output_stride
    

def create_mobile_net_small() -> MobileNetSmall:
    mobile_net_small = MobileNetSmall()
    return mobile_net_small


def create_mobile_net_large() -> MobileNetLarge:
    mobile_net_small = MobileNetLarge()
    return mobile_net_small

