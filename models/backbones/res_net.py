import torch.nn as nn
from dataclasses import dataclass

@dataclass(frozen=True)
class BlockDescr:
    in_channels: int
    inter_channels: int
    out_channels: int
    inter_blocks_count: int
    stride: int = 1
    init_dilation: int = 1
    inter_dilation: int = 1
    use_downsampling: bool = True


class ResNetBlock(nn.Module):
    def __init__(
            self, 
            params: BlockDescr,
        ):

        super().__init__()

        self.relu = nn.ReLU(inplace=True)

        if params.use_downsampling:
            self.residual_block = self.create_residual_block(
                params.in_channels, 
                params.out_channels, 
                params.stride
            )
        else:
            self.residual_block = lambda x: x

        self.initial_block = self.create_initial_block(
            params.in_channels, 
            params.inter_channels, 
            params.out_channels, 
            params.stride, 
            params.init_dilation
        )

        self.intermediate_blocks = nn.ModuleList([
            self.create_intermediate_block(
                params.inter_channels, 
                params.out_channels, 
                params.inter_dilation
            )
            for _ in range(params.inter_blocks_count)
        ])

    def create_initial_block(self, in_channels: int, inter_channels: int, out_channels: int, stride: int, dilation: int):
        initial_block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            self.relu,
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False),
            nn.BatchNorm2d(inter_channels),
            self.relu,
            nn.Conv2d(inter_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        return initial_block
    
    def create_residual_block(self, in_channels: int, out_channels: int, stride: int):
        residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        return residual
    
    def create_intermediate_block(self, inter_channels: int, in_channels: int, dilation: int):
        intermediate_block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            self.relu,
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(inter_channels),
            self.relu,
            nn.Conv2d(inter_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )

        return intermediate_block

    def forward(self, x):
        x = self.residual_block(x) + self.initial_block(x)
        x = self.relu(x)

        for intermediate_block in self.intermediate_blocks:
            x = intermediate_block(x) + x
            x = self.relu(x)
        return x


class ResNet(nn.Sequential):
    def __init__(self, block_params: list[BlockDescr]):

        first = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        modules = []
        modules.append(first)
        for params in block_params:
            modules.append(ResNetBlock(params))

        super().__init__(*modules)
    

    def forward(self, x):
        for mod in self:
            x = mod(x)
        return x
    

def create_res_net_50() -> ResNet:

    blocks = [
        BlockDescr(64, 64, 256, 2, 1),
        BlockDescr(256, 128, 512, 3, 2),
        BlockDescr(512, 256, 1024, 5, inter_dilation=2),
        BlockDescr(1024, 512, 2048, 2, 1, 2, 4)
    ]

    res_net50 = ResNet(blocks)
    return res_net50


def create_res_net_18() -> ResNet:
    blocks = [
        BlockDescr(64, 64, 64, 1, 1, use_downsampling=False),
        BlockDescr(64, 128, 128, 1, 2),
        BlockDescr(128, 256, 256, 1, 2),
        BlockDescr(256, 512, 512, 1, 1),
    ]

    res_net18 = ResNet(blocks)
    return res_net18


def create_res_net_34() -> ResNet:
    blocks = [
        BlockDescr(64, 64, 64, 2, 1, use_downsampling=False),
        BlockDescr(64, 128, 128, 3, 2),
        BlockDescr(128, 256, 256, 5, 2),
        BlockDescr(256, 512, 512, 2, 1),
    ]
    res_net34 = ResNet(blocks)
    return res_net34
