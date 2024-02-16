import torch 
import torch.nn as nn
from .double_conv import double_conv

class down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dimensions = 256):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), double_conv(in_channels= in_channels, out_channels= in_channels, residual= True), double_conv(in_channels=in_channels, out_channels= out_channels)
        )

        self.embed_layer = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(emb_dimensions, out_channels)
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.embed_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
    
