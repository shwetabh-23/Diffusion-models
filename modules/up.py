import torch 
import torch.nn as nn
from .double_conv import double_conv
class up(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim = 256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode= 'bilinear', align_corners= True)
        self.conv = nn.Sequential(
            double_conv(in_channels, in_channels, residual= True),
            double_conv(in_channels, out_channels, in_channels // 2)
        )

        self.embed_layer = nn.Sequential(
            nn.SiLU(), nn.Linear(embed_dim, out_channels)
        )
    
    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim = 1)
        x = self.conv(x)
        embed = self.embed_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])

        return x + embed
    
