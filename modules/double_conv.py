import torch 
import torch.nn as nn
from torch.nn import functional as F

class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual = None):
        super().__init__()

   
        self.residual = residual

        if not mid_channels:
            mid_channels = out_channels
        #breakpoint()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels= in_channels, out_channels= mid_channels, kernel_size= 3, padding= 1, bias= False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(in_channels= mid_channels, out_channels= out_channels, padding=1, kernel_size= 3, bias= False),
            nn.GroupNorm(1, out_channels)
        )

    def forward(self, x):
        #breakpoint()
        if self.residual:
            #breakpoint()
            return F.gelu(x + self.double_conv(x))
        
        else:
            return self.double_conv(x)
        