import torch 
import torch.nn as nn

from .up import up
from .down import down
from .double_conv import double_conv
from .self_attention import self_attention
from memory_profiler import profile


class Unet(nn.Module):
    def __init__(self, c_in = 3, c_out = 3, time_dim = 256):
        super().__init__()

        self.time_dim = time_dim
        #breakpoint()
        self.inc = double_conv(c_in, 64)
        self.down1 = down(64, 128)
        
        self.sa1 = self_attention(128, 32)

        self.down2 = down(128, 256)
        self.sa2 = self_attention(256, 16)

        self.down3 = down(256, 256)
        self.sa3 = self_attention(256, 8)

        self.bot1 = double_conv(256, 512)
        self.bot2 = double_conv(512, 512)
        self.bot3 = double_conv(512, 256)

        self.up1 = up(512, 128)
        self.sa4 = self_attention(128, 16)
        
        self.up2 = up(256, 64)
        self.sa5 = self_attention(64, 32)
        self.up3 = up(128, 64)
        self.sa6 = self_attention(64, 64)

        self.out_c = nn.Conv2d(64, c_out, kernel_size= 1)

    def positional_enc(self, t, channels):
        t = t.to('cuda')
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float().to('cuda') / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)

        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim= -1)
        return pos_enc
   # @profile
    def forward(self, x, t):    
        t = t.unsqueeze(-1).type(torch.float)
        t  = self.positional_enc(t, channels= self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)

        output = self.out_c(x)

        return output
    
if __name__ == '__main__':
    x = torch.randn(3, 3, 64, 64)
    t = x.new_tensor([500] * x.shape[0]).long()

    unet = Unet(device='cpu')
    print(unet(x, t).shape)
    breakpoint()
