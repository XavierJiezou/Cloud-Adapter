from mmseg.models.builder import MODELS
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
from operator import mul
from timm.models.layers import trunc_normal_
from torch import Tensor

@MODELS.register_module()
class MyReins(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dims: int,
        patch_size: int,
        query_dims: int = 256,
        token_length: int = 100,
        mlp_scale=8,
    ) -> None:
        super().__init__()
        self.mlp_scale = mlp_scale
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.query_dims = query_dims
        self.token_length = token_length
        self.create_model()
        self.init_weights()


    def create_model(self):
        self.learnable_tokens = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.embed_dims])
        ) 
        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1) + self.embed_dims
            )
        )
        ### added by zxc
        if self.is_depend:
            hidden_num = self.embed_dims//self.mlp_scale
            if self.is_conv:
                self.depend_mlp = nn.ModuleList([
                   nn.Conv2d(self.embed_dims, self.embed_dims, 7, 1, 3, groups=self.embed_dims)  for _ in range(self.num_layers)])
            else:
                if self.high_high:
                    hidden_num = self.embed_dims * self.mlp_scale
                self.depend_mlp = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.embed_dims, hidden_num),
                        nn.Linear(hidden_num, self.embed_dims),
                ) for i in range(self.num_layers)])
        else:
            self.depend_mlp = nn.Identity()



        
        if self.is_share:
            hidden_num = self.embed_dims*self.mlp_scale
            if self.low_low:
                hidden_num = self.embed_dims//self.mlp_scale
            self.shared_mlp = nn.Sequential(
                nn.Linear(self.embed_dims, hidden_num),
                activate(),
                nn.Linear(hidden_num, self.embed_dims),
            )
        else:
            self.shared_mlp = nn.Identity()
        ### added by zxc

        # nn.init.uniform_(self.learnable_tokens.data, -val, val)
        # nn.init.kaiming_uniform_(self.mlp_delta_f.weight, a=math.sqrt(5))
        # nn.init.kaiming_uniform_(self.mlp_token2feat.weight, a=math.sqrt(5))
        self.transform = nn.Linear(self.embed_dims, self.query_dims)
        self.merge = nn.Linear(self.query_dims * 3, self.query_dims)
        self.scale = 1.0
        if self.zero_mlp_delta_f:
            # del self.scale
            self.scale = 1.0
            # nn.init.zeros_(self.mlp_delta_f.weight)
            # nn.init.zeros_(self.mlp_delta_f.bias)
        

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.Conv2d):
                trunc_normal_(m.weight, std=.02)
                nn.init.constant_(m.bias, 0)


    def forward(
        self, feats: Tensor, layer: int, batch_first=False, has_cls_token=True
    ) -> Tensor:
        # B 1025 emd_dim
        if batch_first:
            feats = feats.permute(1, 0, 2) # 1025 B emd_dim
        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=0)  # feature: 1024 B emd_dim

        
        if self.is_depend:
            tokens = self.depend_mlp[layer]
        else:
            tokens = self.depend_mlp
        delta_feat = self.forward_delta_feat(
            feats,
            tokens,
            layer,
        )
        delta_feat = self.shared_mlp(delta_feat)

        feats = feats + delta_feat
        if has_cls_token:
            feats = torch.cat([cls_token, feats], dim=0)
        if batch_first:
            feats = feats.permute(1, 0, 2)
        return feats

    def forward_delta_feat(self, feats: Tensor, tokens: Tensor, layers: int) -> Tensor:
        return tokens(feats)




if __name__ == "__main__":

    features = torch.randn((2,1025,1024))
    layer = 1
    rein = MyReins(
        token_length=100,
        embed_dims=1024,
        num_layers=24,
        patch_size=16,
        query_dims=100,
        is_conv=True
    )

    output = rein(features,1,True)
    print(output.shape)