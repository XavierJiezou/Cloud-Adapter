from mmseg.models.builder import MODELS
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
from operator import mul
from torch import Tensor
from thop import profile,clever_format


@MODELS.register_module()
class MyReinsToken(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dims: int,
        patch_size: int,
        query_dims: int = 256,
        token_length: int = 100,
        rank=8, 
        alpha=16,
        activate=None,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.activate = activate
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.query_dims = query_dims
        self.token_length = token_length
        self.create_model()


    def create_model(self):
        # 独立MLP
        self.tokens = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.embed_dims])
        )
        self.inverse_token = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.embed_dims])
        )
        
        # 参数初始化
        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1) + self.embed_dims
            )
        )
        nn.init.uniform_(self.tokens.data, -val, val)
        nn.init.uniform_(self.inverse_token.data, -val, val)
        
        # 参数初始化
        std_dev = 1/torch.sqrt(torch.tensor(self.rank).float())
        
        # 共享MLP
        self.C = torch.nn.Parameter(torch.randn(self.embed_dims,self.rank * self.embed_dims)*std_dev)
        self.D = torch.nn.Parameter(torch.zeros(self.rank * self.embed_dims, self.embed_dims))

    def get_tokens(self, layer: int) -> Tensor:
        return self.tokens[layer]

    def forward(
        self, feats: Tensor, layer: int, batch_first=False, has_cls_token=True
    ) -> Tensor:
        if batch_first:
            feats = feats.permute(1, 0, 2) # 1025 B emd_dim
        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=0)  # feature: 1024 B emd_dim
            
            
        # 独立mlp计算
        delta_feat = self.alpha * (feats @ self.tokens[layer] @ self.inverse_token[layer])
        # 共享mlp计算
        delta_feat = self.alpha*(delta_feat@self.C@self.D)
        # 残差连接
        feats = feats + delta_feat
        
        
        
        if has_cls_token:
            feats = torch.cat([cls_token, feats], dim=0)
        if batch_first:
            feats = feats.permute(1, 0, 2)
            
            
        return feats


if __name__ == "__main__":

    features = torch.randn((2,1025,1024)) # B N C
    layer = 1
    rein = MyReinsToken(
        token_length=50,
        embed_dims=1024,
        num_layers=24,
        patch_size=16,
        query_dims=100
    )
    # print(rein)
    output = rein(features,1,True)
    print(output.shape)
    params = sum(p.numel() for p in rein.parameters() if p.requires_grad)
    print(f"Backbone trainable parameters: {params / 1e6:.2f}M")