from mmseg.models.builder import BACKBONES, MODELS
from torch import nn as nn
from .cnnadapter import CNNAdapter
from .dino_v2 import DinoVisionTransformer
from .utils import set_requires_grad, set_train
import torch
import torch.nn.functional as F

@BACKBONES.register_module()
class CNNAdapterDinoVisionTransformer(DinoVisionTransformer):
    def __init__(
        self,
        cnnadapter_config=None,
        has_cat = False,
        cross_attention_count=-1,
        num_layers=24,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cnnadapter: CNNAdapter = MODELS.build(cnnadapter_config)
        self.has_cat = has_cat
        self.cross_attention_count = cross_attention_count
        self.num_layers = num_layers

    def is_cross_attention(self,idx:int):
        if self.cross_attention_count == -1:
            return True
        
        if idx % (self.num_layers // self.cross_attention_count) == 0:
            return True
        
        return False
        


    def forward_features(self, x, masks=None):
        B, _, h, w = x.shape
        cache = self.cnnadapter.cnn(x) # 得到多尺度特征
        H, W = h // self.patch_size, w // self.patch_size
        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        cur_cross_idx = 0
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if self.is_cross_attention(idx):
                x = self.cnnadapter.forward(
                    x,
                    cur_cross_idx,
                    batch_first=True,
                    has_cls_token=True,
                    cache=cache,
                )
                cur_cross_idx += 1
            if idx in self.out_indices:
                outs.append(
                    x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
                )
        return outs, cache

    def forward(self, *args, **kwargs):
        ret, cache = self.forward_features(*args, **kwargs)
        if isinstance(ret[0], torch.Tensor):
            ret[0] = F.interpolate(
                ret[0], scale_factor=4, mode="bilinear", align_corners=False
            )
            ret[1] = F.interpolate(
                ret[1], scale_factor=2, mode="bilinear", align_corners=False
            )
            ret[3] = F.interpolate(
                ret[3], scale_factor=0.5, mode="bilinear", align_corners=False
            )
            if self.has_cat:
                ret[0] = torch.cat((ret[0], cache[0]), dim=1)
                ret[1] = torch.cat((ret[1], cache[1]), dim=1)
                ret[2] = torch.cat((ret[2], cache[2]), dim=1)
                ret[3] = torch.cat((ret[3], cache[3]), dim=1)
            # ret[0] = torch.cat(ret[0], cache[0], dim=1) # bs 1024 128 128, bs 256 128 128
        else:
            ret[0][0] = F.interpolate(
                ret[0][0], scale_factor=4, mode="bilinear", align_corners=False
            )
            ret[0][1] = F.interpolate(
                ret[0][1], scale_factor=2, mode="bilinear", align_corners=False
            )
            ret[0][3] = F.interpolate(
                ret[0][3], scale_factor=0.5, mode="bilinear", align_corners=False
            )
            if self.has_cat:
                ret[0][0] = torch.cat((ret[0][0], cache[0]), dim=1)
                ret[0][1] = torch.cat((ret[0][1], cache[1]), dim=1)
                ret[0][2] = torch.cat((ret[0][2], cache[2]), dim=1)
                ret[0][3] = torch.cat((ret[0][3], cache[3]), dim=1)
        return ret

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["cnnadapter"])
        set_train(self, ["cnnadapter"])

    def state_dict(self, destination, prefix, keep_vars):
        state = super().state_dict(destination, prefix, keep_vars)
        keys = [k for k in state.keys() if "cnnadapter" not in k]
        for key in keys:
            state.pop(key)
            if key in destination:
                destination.pop(key)
        return state

if __name__ == "__main__":
    model = CNNAdapterDinoVisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        img_size=512,
        ffn_layer="mlp",
        init_values=1e-05,
        block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        lcnnadapter_config=dict(
            type="CNNAdapter",
            emd_dim=1024,
            num_layers=24,
            cache_dim=256,
        ),
    )

    # 得到多尺度特征这里，有两个:pmaa,convnext
    # 接下来，对于每个Transformer Block处理后的x ,有两种处理方法，一种是来自pmaa，一种是convnext，然后进行组合，搭配