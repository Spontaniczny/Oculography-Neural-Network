import torch.nn as nn
import torch
from torch.nn.functional import sigmoid


def convolutions(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, (3, 3)),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, (3, 3)),
        nn.ReLU(),
    )

class Crop(nn.Module):
    def __init__(self, residual_size: int, expanded_size: int) -> None:
        super().__init__()
        diff = residual_size - expanded_size
        self.offset_x = diff // 2
        self.offset_y = diff // 2 + diff % 2

    def forward(self, x: torch.tensor) -> torch.tensor:
        return x[:, :, self.offset_x:-self.offset_y, self.offset_x:-self.offset_y]

class U_NET(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()

        self.encoders = nn.ModuleList([
            convolutions(1, 64),
            convolutions(64, 128),
            convolutions(128, 256),
            convolutions(256, 512)
        ])

        self.pooling_layer = nn.MaxPool2d(2)

        self.cropping_layers = nn.ModuleList([
            Crop(76, 68),
            Crop(161, 128),
            Crop(330, 248),
        ])
        
        self.expanders = nn.ModuleList([
            nn.ConvTranspose2d(512, 256, (2, 2), stride=2),
            nn.ConvTranspose2d(256, 128, (2, 2), stride=2),
            nn.ConvTranspose2d(128, 64, (2, 2), stride=2)
        ])

        self.decoders = nn.ModuleList([
            convolutions(512, 256),
            convolutions(256, 128),
            convolutions(128, 64)
        ])
        
        self.final_convolution = nn.Conv2d(64, 1, (1, 1))
    
    
    def forward(self, x: torch.tensor) -> torch.tensor:
        
        # Contracting path

        intermediate_blocks = []
        for encoder_layer in self.encoders[:-1]:
            encoder_block = encoder_layer(x)
            # Saving intermediate results for residual connections
            intermediate_blocks.append(encoder_block)
            x = self.pooling_layer(encoder_block)
        
        x = self.encoders[-1](x)
        
        # Expanding path

        for residual, expander_layer, decoder, crop in zip(
            reversed(intermediate_blocks),
            self.expanders,
            self.decoders,
            self.cropping_layers
        ):
            expanded = expander_layer(x)
            block_cropped = crop(residual)
            concat = torch.concat((block_cropped, expanded), 1)
            x = decoder(concat)
        
        final = self.final_convolution(x)
        return final
    

    def predict_proba(self, x):
        return sigmoid(self(x))
    

    def predict_binary(self, x, threshold: float = 0.5):
        return self.predict_proba(x) > threshold
    
    
