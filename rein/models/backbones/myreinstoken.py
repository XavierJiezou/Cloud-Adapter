from mmseg.models.builder import MODELS
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
from operator import mul
from torch import Tensor
from thop import profile,clever_format

class TokenLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        rank=8,
        alpha=16,
    ) -> None:
        super().__init__()
        std_dev = 1/torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank)*std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
    
    def forward(self, x):
        return self.alpha*(x@self.A@self.B)


@MODELS.register_module()
class MyReinsToken(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dims: int,
        patch_size: int,
        query_dims: int = 256,
        token_length: int = 100,
        use_softmax: bool = True,
        link_token_to_query: bool = True,
        scale_init: float = 0.001,
        zero_mlp_delta_f: bool = False,
        mlp_scale=4,
        rank=8, 
        alpha=16,
        activate=None,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.activate = activate
        self.mlp_scale = mlp_scale
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.query_dims = query_dims
        self.token_length = token_length
        self.link_token_to_query = link_token_to_query
        self.scale_init = scale_init
        self.use_softmax = use_softmax
        self.zero_mlp_delta_f = zero_mlp_delta_f
        self.scale = 1.0
        self.create_model()


    def create_model(self):
        std_dev = 1/torch.sqrt(torch.tensor(self.rank).float())

        self.tokens = nn.Parameter(
            torch.empty([self.num_layers, self.embed_dims,self.token_length])
        )
        self.inverse_token = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.embed_dims])
        )
        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1) + self.embed_dims
            )
        )
        nn.init.uniform_(self.tokens.data, -val, val)
        nn.init.uniform_(self.inverse_token.data, -val, val)

        # self.tokens = nn.Parameter(
        #     torch.randn(self.num_layers,self.embed_dims, self.rank)*std_dev
        # )
        
        self.C = torch.nn.Parameter(torch.randn(self.embed_dims,self.rank * self.embed_dims)*std_dev)
        self.D = torch.nn.Parameter(torch.zeros(self.rank * self.embed_dims, self.embed_dims))
        
    def return_auto(self, feats):
        return feats

    def get_tokens(self, layer: int) -> Tensor:
        return self.tokens[layer]

    def forward(
        self, feats: Tensor, layer: int, batch_first=False, has_cls_token=True
    ) -> Tensor:
        # B 1025 emd_dim
        if batch_first:
            feats = feats.permute(1, 0, 2) # 1025 B emd_dim
        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=0)  # feature: 1024 B emd_dim
            
            
            
            
        # tokens = self.get_tokens(layer) # length * emd_dim
        # delta_feat = self.forward_delta_feat(
        #     feats,
        #     tokens,
        #     layer,
        # )
        delta_feat = self.alpha * (feats @ self.tokens[layer] @ self.inverse_token[layer])
        delta_feat = self.alpha*(delta_feat@self.C@self.D)
        # delta_feat = self.shared_mlp(delta_feat)
        delta_feat = delta_feat * self.scale
        feats = feats + delta_feat
        
        
        
        
        if has_cls_token:
            feats = torch.cat([cls_token, feats], dim=0)
        if batch_first:
            feats = feats.permute(1, 0, 2)
        return feats

    # def forward_delta_feat(self, feats: Tensor, tokens: Tensor, layers: int) -> Tensor:
    #     attn = torch.einsum("nbc,mc->nbm", feats, tokens)

    #     return torch.einsum("nbm,mc->nbc", attn, self.inverse_token[layers])




if __name__ == "__main__":

    features = torch.randn((2,1025,1024))
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