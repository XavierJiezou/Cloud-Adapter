from mmseg.models.builder import BACKBONES, MODELS
from torch import nn as nn
from .cloud_adapter import CloudAdapter
from .dino_v2 import DinoVisionTransformer
from .utils import set_requires_grad, set_train
import torch
import torch.nn.functional as F


@BACKBONES.register_module()
class CloudAdapterDinoVisionTransformer(DinoVisionTransformer):
    def __init__(
        self,
        cloud_adapter_config=None,
        has_cat=False,
        # [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, ],
        adapter_index=[0, 6, 12, 18],  # Transformer Block 的索引
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.cloud_adapter: CloudAdapter = MODELS.build(cloud_adapter_config)
        self.has_cat = has_cat
        self.adapter_index = adapter_index

    def forward_features(self, x, masks=None):
        B, _, h, w = x.shape
        cache = self.cloud_adapter.cnn(x)  # 得到多尺度特征或者单个特征
        H, W = h // self.patch_size, w // self.patch_size
        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        cur_idx = 0  # 交互模块的索引
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if idx in self.adapter_index:
                x = self.cloud_adapter.forward(
                    x,
                    cur_idx,
                    batch_first=True,
                    has_cls_token=True,
                    cache=cache,
                )
                cur_idx += 1
            if idx in self.out_indices:
                outs.append(
                    x[:, 1:, :].permute(0, 2, 1).reshape(
                        B, -1, H, W).contiguous()
                )
        return outs, cache

    def process_cache(self,ret,cache):
        cache = F.interpolate(
            cache,size=(ret.shape[-2],ret.shape[-1]),mode="bilinear",align_corners=False)
        return cache

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
                if isinstance(cache,tuple) or isinstance(cache,list):
                    ret[0] = torch.cat((ret[0], cache[0]), dim=1)
                    ret[1] = torch.cat((ret[1], cache[1]), dim=1)
                    ret[2] = torch.cat((ret[2], cache[2]), dim=1)
                    ret[3] = torch.cat((ret[3], cache[3]), dim=1)
                else:
                    ret[0] = torch.cat((ret[0], self.process_cache(ret[0],cache)), dim=1)
                    ret[1] = torch.cat((ret[1], self.process_cache(ret[1],cache)), dim=1)
                    ret[2] = torch.cat((ret[2], self.process_cache(ret[2],cache)), dim=1)
                    ret[3] = torch.cat((ret[3], self.process_cache(ret[3],cache)), dim=1)
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
                if isinstance(cache,tuple) or isinstance(cache,list):
                    ret[0][0] = torch.cat((ret[0][0], cache[0]), dim=1)
                    ret[0][1] = torch.cat((ret[0][1], cache[1]), dim=1)
                    ret[0][2] = torch.cat((ret[0][2], cache[2]), dim=1)
                    ret[0][3] = torch.cat((ret[0][3], cache[3]), dim=1)
                else:
                    ret[0][0] = torch.cat((ret[0][0], self.process_cache(ret[0][0],cache)), dim=1)
                    ret[0][1] = torch.cat((ret[0][1], self.process_cache(ret[0][1],cache)), dim=1)
                    ret[0][2] = torch.cat((ret[0][2], self.process_cache(ret[0][2],cache)), dim=1)
                    ret[0][3] = torch.cat((ret[0][3], self.process_cache(ret[0][3],cache)), dim=1)
        return ret

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["cloud_adapter"])
        set_train(self, ["cloud_adapter"])

    def state_dict(self, destination, prefix, keep_vars):
        state = super().state_dict(destination, prefix, keep_vars)
        keys = [k for k in state.keys() if "cloud_adapter" not in k]
        for key in keys:
            state.pop(key)
            if key in destination:
                destination.pop(key)
        return state