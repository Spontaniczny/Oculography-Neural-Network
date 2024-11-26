import torch.nn as nn
import torch
from torch.nn.functional import sigmoid


def _convolutions(in_channels: int, out_channels: int, padding: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, (3, 3), padding=padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, (3, 3), padding=padding),
        nn.ReLU(),
    )

class U_NET(nn.Module):
    
    def __init__(
            self, 
            depth: int = 3,
            start_dim_channel_dim: int = 16,
            upsampling_method: str = "conv_transposed"
        ) -> None:
        
        super().__init__()

        assert upsampling_method in ["conv_transposed", "nearest", "bilinear"], "Passed incorrect upsampling method"
        assert depth <= 6, "depth should be an integer in the range [2, 5]"

        channel_dim = [start_dim_channel_dim* pow(2, i)  for i in range(depth - 1)]
        self.downsample = nn.ModuleList(
            [_convolutions(1, start_dim_channel_dim)] + \
            [_convolutions(dim, dim*2) for dim in channel_dim]
        )

        self.pooling_layer = nn.MaxPool2d(2)
        
        self.upsample = self.create_upsampler(upsampling_method, channel_dim)
    
        self.decoders = nn.ModuleList([
            _convolutions(dim*2, dim)
            for dim in channel_dim[::-1]
        ])
        
        self.final_convolution = nn.Conv2d(start_dim_channel_dim, 1, (1, 1))
        
        self.batch_norm2d = nn.BatchNorm2d(1)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # Downsampling path
        x = self.batch_norm2d(x)

        residual_blocks = []
        for conv_block in self.downsample[:-1]:
            x = conv_block(x)
            residual_blocks.append(x)
            x = self.pooling_layer(x)

        x = self.downsample[-1](x)

        # Upsampling path

        for upsampler, residual, decoder in zip(
            self.upsample,
            residual_blocks[::-1],
            self.decoders
        ):
            x = upsampler(x)
            x = torch.concat((residual, x), 1)
            x = decoder(x)
        
        x = self.final_convolution(x)
        return x
    

    def create_upsampler(self, method: str, channel_dim: list[int]) -> nn.ModuleList:
        if method == "conv_transposed":
            upsamplers = nn.ModuleList([
                nn.ConvTranspose2d(dim*2, dim, (2, 2), stride=2)
                for dim in channel_dim[::-1]
            ])
        else:
            if method == "nearest":
                upsample = nn.UpsamplingNearest2d(scale_factor=2)
            else:
                upsample = nn.UpsamplingBilinear2d(scale_factor=2)
            
            upsamplers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(dim*2, dim, (1, 1), stride=1),
                    upsample
                )
                for dim in channel_dim[::-1]
            ])
        
        return upsamplers

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return sigmoid(self(x))
    

    def predict_binary(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return self.predict_proba(x) > threshold
    