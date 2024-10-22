import torch
from torch import nn as nn
from mmseg.models.builder import MODELS
from timm.layers import DropPath
from typing import List

class LayerNorm(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
    def forward(self,x):
         # x : [batch channel height width]
        x = x.permute(0,2,3,1)
        x = self.norm(x)
        # x : [batch height width channel]
        x = x.permute(0,3,1,2)
        return x
    
class Downsample(nn.Module):
    def __init__(self,input_channels,out_channels):
        super().__init__()
        self.norm = LayerNorm(input_channels)
        self.conv = nn.Conv2d(input_channels,out_channels,kernel_size=2,stride=2)
    def forward(self,x:torch.Tensor):
        x = self.norm(x)
        x = self.conv(x)
        return x

@MODELS.register_module()
class AdapterConvNeXtBlock(nn.Module):
    def __init__(self,embed_dim,drop_prob=0.2):
        super().__init__()
        self.conv = nn.Conv2d(embed_dim,embed_dim,7,1,3,groups=embed_dim)
        self.norm = LayerNorm(embed_dim)
        self.feature = nn.Sequential(
            nn.Conv2d(embed_dim,embed_dim * 4,1),
            nn.GELU(),
            nn.Conv2d(embed_dim * 4,embed_dim,1)
        )
        self.scale = nn.Parameter(data=torch.ones(embed_dim))
        self.drop_path = DropPath(drop_prob)
    def forward(self,x:torch.Tensor,h:int=512 // 16,w:int=512 // 16):
        B = x.shape[0]
        cls,feature = torch.split(x,[1,x.shape[1] - 1],dim=1)
        feature = feature.permute(0, 2, 1).reshape(B, -1, h , w).contiguous()
        out = self.conv(feature)
        out = self.norm(out)
        out = self.feature(out)

        out = out.permute(0,2,3,1)
        out = self.scale * out
        out = out.permute(0,3,1,2)
        out = self.drop_path(out)

        feature = feature.reshape(B, -1, feature.shape[1]).permute(0, 2, 1)

        return torch.cat((cls,feature),dim=1)

if __name__ == "__main__":
    inp = torch.randn((2, 1025, 1024))
    model = AdapterConvNeXtBlock(1024)
    out = model(inp)
    print(out.shape)