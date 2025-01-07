import torch.nn as nn
import torch
from abc import ABC, abstractmethod

class Backbone(ABC, nn.Module):
    
    @abstractmethod
    def output_channels(self):
        ...

    @abstractmethod
    def output_stride(self):
        ...
