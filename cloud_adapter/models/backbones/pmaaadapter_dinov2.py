from mmseg.models.builder import BACKBONES, MODELS
from torch import nn as nn
from .pmaaadapter import PMAAAdapter
from .dino_v2 import DinoVisionTransformer
from .utils import set_requires_grad, set_train
import torch
import torch.nn.functional as F

@BACKBONES.register_module()
class PMAAAdapterDinoVisionTransformer(DinoVisionTransformer):
    def __init__(
        self,
        pmaa_adapter_config=None,
        has_cat = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pmaa_adapter: PMAAAdapter = MODELS.build(pmaa_adapter_config)
        self.has_cat = has_cat
        


    def forward_features(self, x, masks=None):
        B, _, h, w = x.shape
        cache = self.pmaa_adapter.pmaa(x) # 得到多尺度特征
        H, W = h // self.patch_size, w // self.patch_size
        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x = self.pmaa_adapter.forward(
                x,
                idx,
                batch_first=True,
                has_cls_token=True,
                cache=cache,
            )
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
        set_requires_grad(self, ["pmaa_adapter"])
        set_train(self, ["pmaa_adapter"])

    def state_dict(self, destination, prefix, keep_vars):
        state = super().state_dict(destination, prefix, keep_vars)
        keys = [k for k in state.keys() if "pmaa_adapter" not in k]
        for key in keys:
            state.pop(key)
            if key in destination:
                destination.pop(key)
        return state