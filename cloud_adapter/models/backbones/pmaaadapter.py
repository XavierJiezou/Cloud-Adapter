import torch
from torch import nn
from einops import rearrange
from torch import nn, einsum
from einops import rearrange
from mmseg.models.builder import MODELS
import math
import torch
from torch import nn as nn
from mmseg.models.builder import MODELS
from timm.layers import DropPath,trunc_normal_
from typing import List
from timm.layers import create_act_layer
from functools import partial
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.layers import CondConv2d, get_condconv_initializer, create_conv2d, DropPath, get_norm_act_layer


def num_groups(group_size, channels):
    if not group_size:
        return 1
    else:
        assert channels % group_size == 0
        return channels // group_size


def _init_weight_goog(m, n='', fix_group_fanout=True):
    if isinstance(m, CondConv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        if fix_group_fanout:
            fan_out //= m.groups
        init_weight_fn = get_condconv_initializer(
            lambda w: nn.init.normal_(w, 0, math.sqrt(2.0 / fan_out)), m.num_experts, m.weight_shape)
        init_weight_fn(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        if fix_group_fanout:
            fan_out //= m.groups
        nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        fan_out = m.weight.size(0)
        fan_in = 0
        if 'routing_fn' in n:
            fan_in = m.weight.size(1)
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        nn.init.uniform_(m.weight, -init_range, init_range)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class DepthwiseSeparableConv(nn.Module):
    def __init__(
            self, in_chs, out_chs, dw_kernel_size=3, stride=1, dilation=1, group_size=1, pad_type='',
            noskip=False, pw_kernel_size=1, pw_act=False, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d,
            se_layer=None, drop_path_rate=0.):
        super(DepthwiseSeparableConv, self).__init__()
        norm_act_layer = get_norm_act_layer(norm_layer)
        groups = num_groups(group_size, in_chs)
        self.has_skip = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_pw_act = pw_act

        self.conv_dw = create_conv2d(
            in_chs, in_chs, dw_kernel_size, stride=stride, dilation=dilation, padding=pad_type, groups=groups)
        self.bn1 = norm_act_layer(in_chs, inplace=True)

        self.se = se_layer(
            in_chs, act_layer=act_layer) if se_layer else nn.Identity()

        self.conv_pw = create_conv2d(
            in_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = norm_act_layer(
            out_chs, inplace=True, apply_act=self.has_pw_act)
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':
            return dict(module='conv_pw', hook_type='forward_pre', num_chs=self.conv_pw.in_channels)
        else:
            return dict(module='', hook_type='', num_chs=self.conv_pw.out_channels)

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.se(x)
        x = self.conv_pw(x)
        x = self.bn2(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x




class PMAAConvBlock(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=256, depth=4, norm=nn.BatchNorm2d, act=nn.ReLU, return_multi_feats=False):
        super().__init__()
        self.depth = depth
        self.return_multi_feats=return_multi_feats
        
        self.proj_1x1 = DepthwiseSeparableConv(in_channels, hidden_channels, dw_kernel_size=1, norm_layer=norm, act_layer=act)
        
        self.spp_dw = nn.ModuleList()
        
        self.spp_dw.append(
            DepthwiseSeparableConv(hidden_channels, hidden_channels, dw_kernel_size=3, stride=1, group_size=hidden_channels, pad_type="same")
        )

        for _ in range(self.depth):
            self.spp_dw.append(
                DepthwiseSeparableConv(
                    hidden_channels, hidden_channels, dw_kernel_size=3, stride=2, group_size=hidden_channels
                )
            )

        self._init_weights()

    def forward(self, x):
        B, C, H, W = x.shape
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]

        for k in range(1, self.depth+1):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        
        if self.return_multi_feats:
            return output
        else:
            global_f = torch.zeros(output[-1].shape, requires_grad=True, device=output1.device)
            for fea in output:
                global_f = global_f + F.adaptive_avg_pool2d(
                    fea, output_size=output[-1].shape[-2:]
                )
            return global_f

    def _init_weights(self):
        init_fn = _init_weight_goog
        for n, m in self.named_modules():
            init_fn(m, n)


class PMAA(nn.Module):
    def __init__(self, hidden_channels=256, depth=4, norm=nn.BatchNorm2d, act=nn.ReLU, return_multi_feats=True) -> None:
        super(PMAA, self).__init__()
        self.net= PMAAConvBlock(3, hidden_channels, depth=depth, norm=norm, act=act, return_multi_feats=return_multi_feats)
 

    def forward(self, x):
        return self.net(x)


class InteractiveModule(nn.Module):
    def __init__(self, emd_dim=1024, context_dim=64, kernel: int = 1, norm=nn.BatchNorm2d, local_groups=32, global_groups=2):
        super().__init__()
        self.local_embedding = nn.Sequential(
            nn.Conv2d(emd_dim, emd_dim, kernel, groups=local_groups,
                      padding=int((kernel - 1) / 2), bias=False),
            norm(emd_dim)
        )
        self.global_embedding = nn.Sequential(
            nn.Conv2d(context_dim, emd_dim, kernel, groups=global_groups,
                      padding=int((kernel - 1) / 2), bias=False),
            norm(emd_dim)
        )
        self.global_act = nn.Sequential(
            nn.Conv2d(context_dim, emd_dim, kernel, groups=global_groups,
                      padding=int((kernel - 1) / 2), bias=False),
            norm(emd_dim)
        )
        self.act = nn.Sigmoid()
        self._init_weights()
    
    def _init_weights(self):
        init_fn = _init_weight_goog
        for n, m in self.named_modules():
            init_fn(m, n)

    def forward(self, x, cache, layer):

        N, B, C = x.shape
        H=W=int(math.sqrt(N))
        # reshape x -> B, C, H, W
        x = x.permute(1, 2, 0).reshape(B, C, H, W)
        local_feat = self.local_embedding(x)

        global_act = self.global_act(cache)
        sig_act = F.interpolate(self.act(global_act), size=(H, W))

        global_feat = self.global_embedding(cache)
        global_feat = F.interpolate(global_feat, size=(H, W))

        out = local_feat * sig_act + global_feat
        
        return out.permute(2, 3, 0, 1).reshape(N, B, C)

@MODELS.register_module()
class PMAAAdapter(nn.Module):
    def __init__(self, num_layers, emd_dim=1024, context_dim=64, local_groups=32, global_groups=2):
        super().__init__()
        self.pmaa = PMAA(context_dim)
        self.net = nn.ModuleList(
            InteractiveModule(emd_dim, context_dim, local_groups=local_groups, global_groups=global_groups)
            for _ in range(num_layers)
        )
        self.init_weight()

    def init_weight(self):
        for m in self.net.modules(): 
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        
    def forward(self, feats, layer, batch_first=True, has_cls_token=True, cache=None):
        if batch_first:
            feats = feats.permute(1, 0, 2)
        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=0)
            
        feats = self.net[layer](feats, cache, layer)
        
        if has_cls_token:
            feats = torch.cat([cls_token, feats], dim=0)
        if batch_first:
            feats = feats.permute(1, 0, 2)
        return feats

if __name__ == "__main__":

    x = torch.randn((1, 1025, 1024))
    model = PMAAAdapter(24, 1024, 64, local_groups=32, global_groups=2) # 
    cache = model.pmaa(torch.randn((1, 3, 512, 512)))
    # print(cache.shape)
    for feature in cache:
        print(feature.shape)

    exit(0)
    output = model(x, 0, cache=cache)
    

    # output, cache = model(x, 1, cache=cache)
    # print(output.shape, cache.shape)
    
    # compute params (Mb) of the total model
    params= sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total params: {params:.2f} Mb")
    
    # # compute params (Mb) of the pmaa model
    params= sum(p.numel() for p in model.pmaa.parameters()) / 1e6
    print(f"PMAA params: {params:.2f} Mb")
    
    # compute macs (Gflps) and params (Mb) of the total model
    # from thop import profile
    # macs, params = profile(model, inputs=((x, 0, True, True, cache)), verbose=False)
    # print(f"Total macs: {macs / 1e9:.2f} Gflps, params: {params / 1e6:.2f} Mb")
    
    # # compute macs (Gflps) and params (Mb) of the pmaa model
    # macs, params = profile(model.pmaa, inputs=(torch.randn(1, 3, 512, 512),), verbose=False)
    # print(f"PMAA macs: {macs / 1e9:.2f} Gflps, params: {params / 1e6:.2f} Mb")

    # 
