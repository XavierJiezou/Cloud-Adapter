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
from timm.layers import DropPath, trunc_normal_
from typing import List
from timm.layers import create_act_layer
from functools import partial
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.layers import CondConv2d, get_condconv_initializer, create_conv2d, DropPath, get_norm_act_layer


class LoRaMLP(nn.Module):
    def __init__(self, in_dim, out_dim, rank_dim=8):
        super().__init__()
        self.loramlp = nn.Sequential(
            nn.Linear(in_dim, rank_dim, bias=False),
            nn.Linear(rank_dim, out_dim, bias=False),
        )

    def forward(self, x):
        return self.loramlp(x)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, rank_dim=None):
        super().__init__()
        inner_dim = dim_head * heads  # 512
        context_dim = query_dim if context_dim is None else context_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        if not rank_dim:
            self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
            self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

            self.to_out = nn.Linear(inner_dim, query_dim, bias=False)
        else:
            self.to_q = LoRaMLP(query_dim, inner_dim, rank_dim=rank_dim)
            self.to_k = LoRaMLP(context_dim, inner_dim, rank_dim=rank_dim)
            self.to_v = LoRaMLP(context_dim, inner_dim, rank_dim=rank_dim)

            self.to_out = LoRaMLP(inner_dim, query_dim, rank_dim=rank_dim)

    def forward(self, x, context):
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        return self.to_out(out)


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
    def __init__(self, in_channels=3, hidden_channels=256, depth=4, norm=nn.BatchNorm2d, act=nn.ReLU, return_multi_feats=False, return_last_feature=True, has_stem=True, has_block=True):
        super().__init__()
        self.return_last_feature = return_last_feature
        self.depth = depth
        self.has_stem = has_stem
        self.return_multi_feats = return_multi_feats

        self.proj_1x1 = DepthwiseSeparableConv(
            in_channels, hidden_channels, dw_kernel_size=1, norm_layer=norm, act_layer=act)

        self.spp_dw = nn.ModuleList()

        if has_stem:
            self.spp_dw.append(
                DepthwiseSeparableConv(hidden_channels, hidden_channels, dw_kernel_size=3,
                                       stride=1, group_size=hidden_channels, pad_type="same")
            )
        else:
            self.spp_dw.append(nn.Identity())

        if has_block:
            for _ in range(self.depth):
                self.spp_dw.append(
                    DepthwiseSeparableConv(
                        hidden_channels, hidden_channels, dw_kernel_size=3, stride=2, group_size=hidden_channels
                    )
                )
        else:
            for _ in range(self.depth):
                self.spp_dw.append(
                    nn.MaxPool2d(kernel_size=2, stride=2)
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
            return output[1:]
        else:
            if self.return_last_feature:
                return output[-1]
            global_f = torch.zeros(
                output[-1].shape, requires_grad=True, device=output1.device)
            for fea in output:
                global_f = global_f + F.adaptive_avg_pool2d(
                    fea, output_size=output[-1].shape[-2:]
                )
            return global_f

    def _init_weights(self):
        init_fn = _init_weight_goog
        for n, m in self.named_modules():
            init_fn(m, n)


class ConvnextInteractiveModule(nn.Module):
    def __init__(self, emd_dim=1024, context_dim=256, rank_dim=None):
        super().__init__()
        self.attn = CrossAttention(emd_dim, context_dim, rank_dim=rank_dim)

    def forward(self, x, cache, index):
        # x: 1024 2 1024
        if isinstance(cache, list) or isinstance(cache, tuple):
            # len(cache) 4 cache[4]-23
            # 0-5->0 6-11 -> 1 12-17->2 18-23->3
            cache = cache[index]
        cache = F.interpolate(
            cache, (int(math.sqrt(x.shape[0])), int(math.sqrt(x.shape[0]))), mode="bilinear", align_corners=False
        )
        cache = cache.flatten(2)  # B C N
        cache = cache.permute(2, 0, 1)  # N B C

        # Reshape: batch first
        x = x.permute(1, 0, 2)  # B N C
        cache = cache.permute(1, 0, 2)  # B N C
        return (x + self.attn(x, cache)).permute(1, 0, 2)


class PMAAInteractiveModule(nn.Module):
    def __init__(self,
                 emd_dim=1024,
                 context_dim=64,
                 kernel: int = 1,
                 norm=nn.BatchNorm2d,
                 local_groups=32,
                 global_groups=2,
                 return_multi_feats=False,
                 ):
        super().__init__()
        self.return_multi_feats = return_multi_feats
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

    def forward(self, x, cache, index):
        if isinstance(cache, list) or isinstance(cache, tuple):
            cache = cache[index]
        N, B, C = x.shape
        H = W = int(math.sqrt(N))
        # reshape x -> B, C, H, W
        x = x.permute(1, 2, 0).reshape(B, C, H, W)
        local_feat = self.local_embedding(x)  # 32
        global_act = self.global_act(cache)
        sig_act = F.interpolate(self.act(global_act), size=(H, W))  # 32

        global_feat = self.global_embedding(cache)
        global_feat = F.interpolate(global_feat, size=(H, W))  # 32

        out = local_feat * sig_act + global_feat

        return out.permute(2, 3, 0, 1).reshape(N, B, C)


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7,
                                padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        # pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3],
                 return_multi_feats=False,
                 return_last_feature=True
                 ):
        super().__init__()
        self.return_last_feature = return_last_feature
        self.return_multi_feats = return_multi_feats

        # stem and 3 intermediate downsampling conv layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=2, stride=2),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stages = nn.ModuleList()
        dp_rates = [x.item()
                    for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            # logger = get_root_logger()
            # load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.return_multi_feats:
            return tuple(outs)
        if self.return_last_feature:
            return outs[-1]
        global_f = torch.zeros(
            outs[-1].shape, requires_grad=True, device=outs[-1].device)
        for fea in outs:
            global_f = global_f + F.adaptive_avg_pool2d(
                fea, output_size=outs[-1].shape[-2:]
            )
        return global_f

    def forward(self, x):
        x = self.forward_features(x)
        return x


class NoAdaptingModule(nn.Identity):
    def __init__(self):
        super().__init__()

    def forward(self, x, cache, layer):
        return x


@MODELS.register_module()
class CloudAdapter(nn.Module):
    def __init__(self,
                 cnn_type="convnext",  # convnext or mobilenet
                 int_type="convnext",  # cross_attention or
                 # 共同的参数 start
                 emd_dim=1024,
                 num_layers=24,

                 # 先判断是否返回多特征，之后再判断是否进行特征融合
                 return_multi_feats=True,
                 return_last_feature=False,

                 # 共同的参数 end

                 # pmaa 提取单个特征 or 多尺寸特征 start
                 hidden_channels=256,
                 depth=4,
                 norm=nn.BatchNorm2d,
                 act=nn.ReLU,
                 # pmaa 提取单个特征 or 多尺寸特征 end

                 # pmaa net start
                 local_groups=1,
                 global_groups=1,
                 # pmaa net end

                 # convnext 提取单个特征 or 多尺寸特征 start
                 context_dim=256,
                 rank_dim=None,
                 # convnext 提取单个特征 or 多尺寸特征 end,
                 has_stem=True,
                 has_block=True,
                 ):
        super().__init__()
        self.cnn = nn.Identity()
        self.net = nn.Identity()
        if cnn_type == "pmaa":
            self.cnn = PMAAConvBlock(
                hidden_channels=hidden_channels,
                depth=depth,
                norm=norm,
                act=act,
                return_multi_feats=return_multi_feats,
                return_last_feature=return_last_feature,
                has_stem=has_stem,
                has_block=has_block
            )
        elif cnn_type == "convnext":
            self.cnn = ConvNeXt(depths=[1]*4,
                                dims=[context_dim]*4,
                                return_multi_feats=return_multi_feats,
                                return_last_feature=return_last_feature
                                )

        else:
            raise ValueError(
                f"cnn_type must in ['convnext','pmaa'],but got {cnn_type}")

        if int_type == "convnext":
            self.net = nn.ModuleList(
                ConvnextInteractiveModule(emd_dim, context_dim, rank_dim)
                for _ in range(num_layers)
            )
        elif int_type == "pmaa":
            self.net = nn.ModuleList(
                PMAAInteractiveModule(
                    emd_dim, context_dim, local_groups=local_groups, global_groups=global_groups)
                for _ in range(num_layers)
            )

        elif int_type == "no_adapting":
            self.net = nn.ModuleList(
                NoAdaptingModule() for _ in range(num_layers)
            )
        else:
            raise ValueError(
                f"int_type must in ['convnext','pmaa'],but got {int_type}")

    def forward(self, feats, layer, batch_first=True, has_cls_token=True, cache=None):
        if batch_first:
            feats = feats.permute(1, 0, 2)  # 1025 2 1024
        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=0)
        # 24 // 1
        # feat: 1024 2 1024
        feats = self.net[layer].forward(
            feats, cache, layer//(len(self.net) // 4))

        if has_cls_token:
            feats = torch.cat([cls_token, feats], dim=0)
        if batch_first:
            feats = feats.permute(1, 0, 2)
        return feats


