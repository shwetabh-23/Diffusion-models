import torch
import torch.nn as nn

class self_attention(nn.Module):
    def __init__(self, channels, size):
        super(self_attention, self).__init__()

        self.channels = channels
        self.size = size

        self.mha = nn.MultiheadAttention(embed_dim= channels, num_heads= 4, batch_first= True)
        self.ln = nn.LayerNorm(normalized_shape= [channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]), nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels)
        )
        
    def forward(self, x):
        #breakpoint()
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attn_value, _ = self.mha(x_ln, x_ln, x_ln)
        attn_value = attn_value + x
        attn_value = self.ff_self(attn_value) + attn_value

        return attn_value.swapaxes(1, 2).view(-1, self.channels, self.size, self.size)
    
