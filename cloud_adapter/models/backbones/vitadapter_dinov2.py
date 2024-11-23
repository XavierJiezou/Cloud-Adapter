# Copyright (c) Shanghai AI Lab. All rights reserved.
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import BACKBONES
# from ops.modules import MSDeformAttn

from cloud_adapter.models.backbones import DinoVisionTransformer
from mmcv.ops import MultiScaleDeformableAttention as MSDeformAttn
from timm.models.layers import DropPath, trunc_normal_
from torch.nn.init import normal_
from .utils import set_requires_grad, set_train
from cloud_adapter.models.backbones.adapter_modules import SpatialPriorModule, InteractionBlock, deform_inputs

_logger = logging.getLogger(__name__)


@BACKBONES.register_module()
class ViTAdapter(DinoVisionTransformer):
    def __init__(self, pretrain_size=224,conv_inplane=64, n_points=4, deform_num_heads=6,
                 init_values=0., interaction_indexes=None, with_cffn=True, cffn_ratio=0.25,drop_rate=0.,
                 deform_ratio=1.0, add_vit_feature=True, use_extra_extractor=True, **kwargs):

        super().__init__(**kwargs)

        # self.num_classes = 80
        self.cls_token = None
        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        embed_dim = self.embed_dim
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane,
                                      embed_dim=embed_dim)
        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                             init_values=init_values, drop_path=self.drop_path_rate,
                             norm_layer=self.norm_layer, with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                             extra_extractor=((True if i == len(interaction_indexes) - 1 else False) and use_extra_extractor))
            for i in range(len(interaction_indexes))
        ])
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.adapter_norm1 = nn.SyncBatchNorm(embed_dim)
        self.adapter_norm2 = nn.SyncBatchNorm(embed_dim)
        self.adapter_norm3 = nn.SyncBatchNorm(embed_dim)
        self.adapter_norm4 = nn.SyncBatchNorm(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["level_embed","pos_drop","spm","interactions","up","adapter_norm1","adapter_norm2","adapter_norm3","adapter_norm4"])
        set_train(self, ["level_embed","pos_drop","spm","interactions","up","adapter_norm1","adapter_norm2","adapter_norm3","adapter_norm4"])

    # def state_dict(self, destination, prefix, keep_vars):
    #     state = super().state_dict(destination, prefix, keep_vars)
    #     keys = [k for k in state.keys() if "loracacheadapter" not in k]
    #     for key in keys:
    #         state.pop(key)
    #         if key in destination:
    #             destination.pop(key)
    #     return state

    def _init_deform_weights(self, m):
        pass
        # if isinstance(m, MSDeformAttn):
        #     m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        B, _, h, w = x.shape
        H,W = h//self.patch_size, w//self.patch_size
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        # print(c1.shape) # [2, 1024, 128, 128])
        # print(c2.shape) # [2, 4096, 1024]
        # print(c3.shape) # [2, 1024, 1024]
        # print(c4.shape) # [2, 256, 1024]
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        x = self.patch_embed(x)
        bs, n, dim = x.shape
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        x = self.pos_drop(x + pos_embed)
        # print(x.shape) # [2, 1024, 1024]

        # Interaction
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.blocks[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, H, W)

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x3 = x.transpose(1, 2).view(bs, dim, H, W).contiguous()
            x1 = F.interpolate(x3, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x3, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.adapter_norm1(c1)
        f2 = self.adapter_norm2(c2)
        f3 = self.adapter_norm3(c3)
        f4 = self.adapter_norm4(c4)
        return [f1, f2, f3, f4]

if __name__ == "__main__":
    device = "cuda:2"
    model = ViTAdapter(
        pretrain_size=512,
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

        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
        deform_num_heads=16,
    ).to(device)
    inp = torch.randn((2,3,512,512)).to(device)
    features = model(inp)
    for feature in features:
        print(feature.shape)