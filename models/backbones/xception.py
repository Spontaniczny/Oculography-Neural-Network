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


class Block(nn.Sequential):

    def __init__(self, in_channels, out_channels, last_stride: int = 1, downsampling=False):
        super().__init__()

        self.sep_conv1 = SeparableConv2d(in_channels, out_channels, kernel_size=3)
        self.sep_conv2 = SeparableConv2d(out_channels, out_channels, kernel_size=3)
        self.sep_conv3 = SeparableConv2d(out_channels, out_channels, kernel_size=3, stride=last_stride)
        
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.batch_norm3 = nn.BatchNorm2d(out_channels)

        if downsampling:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=last_stride)
        else:
            self.skip = lambda x: x

        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        res = x
        x = self.relu(self.batch_norm1(self.sep_conv1(x)))
        x = self.relu(self.batch_norm2(self.sep_conv2(x)))
        x = self.relu(self.batch_norm3(self.sep_conv3(x)))

        x = x + self.skip(res)
        return x

class AlignedXception(Backbone):
    def __init__(
            self, 
            output_stride: int = 8,
            middle_blocks: int = 16
        ):
        super().__init__()

        if output_stride == 4:
            strides = (1, 1, 1)
        elif output_stride == 8:
            strides = (2, 1, 1)
        elif output_stride == 16:
            strides = (2, 2, 2)
        else:
            raise NotImplementedError()

        # Entry flow

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.entry_block1 = Block(64, 128, last_stride=strides[0], downsampling=True)
        self.entry_block2 = Block(128, 256, last_stride=strides[1], downsampling=True)
        self.entry_block3 = Block(256, 256, last_stride=strides[2], downsampling=True)

        # Middle flow

        self.middle_flow = nn.ModuleList([
            Block(256, 256)
            for _ in range(middle_blocks)
        ])

        # Exit flow
        self.exit_block = Block(256, 256, last_stride=2, downsampling=True)

        self.exit_conv1 = SeparableConv2d(256, 512, 3)
        self.bn_exit_1 = nn.BatchNorm2d(512)

        self.exit_conv2 = SeparableConv2d(512, 512, 3)
        self.bn_exit_2 = nn.BatchNorm2d(512)

        self.exit_conv3 = SeparableConv2d(512, 1024, 3)
        self.bn_exit_3 = nn.BatchNorm2d(1024)

        self._output_channels = 1024
        self._output_stride = output_stride

    def forward(self, x):
        # Entry flow
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        x = self.entry_block1(x)
        x = self.entry_block2(x)
        x = self.entry_block3(x)

        # Middle flow
        for block in self.middle_flow:
            x = block(x)

        # Exit flow

        x = self.exit_block(x)
        x = self.relu(self.bn_exit_1(self.exit_conv1(x)))
        x = self.relu(self.bn_exit_2(self.exit_conv2(x)))
        x = self.relu(self.bn_exit_3(self.exit_conv3(x)))

        return x
    
    @property
    def output_channels(self):
        return self._output_channels
    
    @property
    def output_stride(self):
        return self._output_stride
    
def create_xception() -> Backbone:
    return AlignedXception(middle_blocks=8)
