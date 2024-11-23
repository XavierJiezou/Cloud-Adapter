from mmseg.models.builder import MODELS
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import reduce
from operator import mul
from torch import Tensor


@MODELS.register_module()
class MyReinsTokenMlp(nn.Module):
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
        activate=None,
        is_depend=True,
        is_share=True,
        high_high = False,
        low_low=False
    ) -> None:
        super().__init__()
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
        self.is_depend = is_depend
        self.is_share = is_share
        self.high_high = high_high
        self.low_low = low_low
        self.create_model()
        self.init_weights()


    def create_model(self):
        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1) + self.embed_dims
            )
        )
        activate = nn.Identity
        if self.activate == "silu":
            activate = nn.SiLU
        ### added by zxc
        if self.is_depend:
            hidden_num = self.embed_dims//self.mlp_scale
            if self.high_high:
                hidden_num = self.embed_dims * self.mlp_scale
            self.depend_mlp = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.embed_dims])
        ) 
            self.inverse_mlp = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.embed_dims])
        )
            nn.init.uniform_(self.depend_mlp.data, -val, val)
            nn.init.uniform_(self.inverse_mlp.data, -val, val)
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

    def return_auto(self, feats):
        return feats

    def forward(
        self, feats: Tensor, layer: int, batch_first=False, has_cls_token=True
    ) -> Tensor:
        # B 1025 emd_dim
        if batch_first:
            feats = feats.permute(1, 0, 2) # 1025 B emd_dim
        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=0)  # feature: 1024 B emd_dim
        # tokens = self.get_tokens(layer) # length * emd_dim
        delta_feat = self.forward_delta_feat(
            feats,
            layer,
        )
        delta_feat = self.shared_mlp(delta_feat)
        delta_feat = delta_feat * self.scale
        feats = feats + delta_feat
        if has_cls_token:
            feats = torch.cat([cls_token, feats], dim=0)
        if batch_first:
            feats = feats.permute(1, 0, 2)
        return feats

    def forward_delta_feat(self, feats: Tensor, layers: int) -> Tensor:
        feat = torch.einsum("nbc,mc->nbm", feats, self.depend_mlp[layers])
        return torch.einsum("nbm,mc->nbc", feat, self.inverse_mlp[layers])




if __name__ == "__main__":

    features = torch.randn((2,1025,1024))
    layer = 1
    rein = MyReinsTokenMlp(
        token_length=100,
        embed_dims=1024,
        num_layers=24,
        patch_size=16,
        query_dims=100
    )

    output = rein(features,1,True)
    print(output.shape)