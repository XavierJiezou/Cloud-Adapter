import torch
from torch import nn as nn
from mmseg.models.builder import MODELS
from timm.layers import DropPath,trunc_normal_
from typing import List
from timm.layers import create_act_layer

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
    def __init__(
        self,
        embed_dim, 
        rank_type="low", # low or high 
        rank_scale=4, # 1, 2, 4, 8 
        alpha = 1,  # 1, 2, 4, 8 or nn.Parameter(data=torch.ones(embed_dim))
        act_layer = "silu", # nn.GELU or nn.SiLU
        has_conv = True,
        has_proj = True,
        drop_prob=0, 
    ):
        super().__init__()
        
        self.has_conv = has_conv
        self.has_proj = has_proj
        
        if self.has_conv:
            self.conv = nn.Sequential(
                LayerNorm(embed_dim),
                nn.Conv2d(embed_dim, embed_dim, 7, 1, 3, groups=embed_dim),
            )
                
        if self.has_proj:
            if rank_type == "low":
                rank_dim = embed_dim // rank_scale
            elif rank_type == "high":
                rank_dim = embed_dim * rank_scale
            else:
                raise ValueError("rank_type must be low or high")
            
            self.proj = nn.Sequential(
                LayerNorm(embed_dim),
                nn.Conv2d(embed_dim, rank_dim, 1),
                create_act_layer(act_layer),
                nn.Conv2d(rank_dim, embed_dim, 1)
            )
            
            self.alpha = alpha
            
        self.drop_path = DropPath(drop_prob)

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
        
    def forward(self,x:torch.Tensor,h:int=256 // 16,w:int=256 // 16):
        B = x.shape[0]
        cls,feature = torch.split(x,[1,x.shape[1] - 1],dim=1)
        
        feature = feature.permute(0, 2, 1).reshape(B, -1, h , w).contiguous()
        
        res = feature
        
        if self.has_conv:
            feature = self.conv(feature)
        
        if self.has_proj:
            feature = self.alpha * self.proj(feature)
        
        feature = self.drop_path(feature)

        feature = res + feature

        feature = feature.reshape(B, -1, feature.shape[1])

        

        return torch.cat((cls,feature),dim=1)

if __name__ == "__main__":
    inp = torch.randn((2, 1025, 256))
    model = AdapterConvNeXtBlock(
        embed_dim=1024, 
        rank_type="high", # low or high 
        rank_scale=4, # 1, 2, 4, 8 
        alpha = 1,  # 1, 2, 4, 8 or nn.Parameter(data=torch.ones(embed_dim))
        act_layer = "silu", # nn.GELU or nn.SiLU
        has_conv = True,
        has_proj = True,
        drop_prob=0, 
    )
    out = model(inp)
    print(model)
    assert out.shape == (2, 1025, 256)

    conv_params = 0
    proj_params = 0
    for name, param in model.named_parameters():
        if "conv" in name:
            conv_params += param.numel()
        if "proj" in name:
            proj_params += param.numel()
    print(f"conv_params: {conv_params/1e6:.2f}M")
    print(f"proj_params: {proj_params/1e6:.2f}M")
    print(f"total_params: {(conv_params + proj_params)/1e6:.2f}M")