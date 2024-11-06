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
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

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
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
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

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x



class InductionBias(nn.Module):
    def __init__(self, in_chans=3, dim=16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dim, kernel_size=4, stride=4),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, kernel_size=4, stride=4),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )
    
    def forward(self, x): # input.shape=(bs, 3, 512, 512) output.shape=([bs, 1025, 16])
        x = self.stem(x) # x.shape=(bs, 16, 32, 32)
        x = self.proj(x) # x.shape=(bs, 16, 32, 32)
        x = x.flatten(2) # x.shape=(bs, 1024, 16)
        return x.permute(2, 0, 1)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, bias):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=bias),
        )
        
    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Linear(inner_dim, query_dim, bias=False)
        

    def forward(self, x, context):
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        
        return self.to_out(out)


class InteractiveModule(nn.Module):
    def __init__(self, emd_dim=1024, context_dim=256):
        super().__init__()
        self.attn = CrossAttention(emd_dim, context_dim)

    def forward(self, x, cache, layer):
        cache = cache[layer//6]
        cache = F.interpolate(
            cache, (int(math.sqrt(x.shape[0])), int(math.sqrt(x.shape[0]))), mode="bilinear", align_corners=False
        )
        cache = cache.flatten(2) # B C N
        cache = cache.permute(2, 0, 1) # N B C
        
        # Reshape: batch first
        x = x.permute(1, 0, 2) # B N C
        cache = cache.permute(1, 0, 2) # B N C
        return (x + self.attn(x, cache)).permute(1, 0, 2)

@MODELS.register_module()
class CNNAdapter(nn.Module):
    def __init__(self, num_layers, emd_dim=1024, context_dim=256):
        super().__init__()
        self.cnn = ConvNeXt(depths=[1]*4, dims=[context_dim]*4)
        self.net = nn.ModuleList(
            InteractiveModule(emd_dim, context_dim)
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
    x = torch.randn((2, 1025, 1024))

    convnext = ConvNeXt(in_chans=3,depths=[1]*4, dims=[64]*4)

    out = convnext(torch.randn((2,3,512,512)))

    model = CNNAdapter(4, 1024, 128)
    cache = model.cnn(torch.randn((2, 3, 512, 512)))

    for feature in cache:
        print(feature.shape)

    output = model(x, 0, cache=cache)
    print(output.shape)

    # output, cache = model(x, 1, cache=cache)
    # print(output.shape, cache.shape)
    
    # compute params
    total_params = 0 
    for param in model.parameters(): 
        total_params +=param.numel()
    print(f"Total parameters in the model: {total_params/1e6:.2f}MB")

        
    
    cnn = model.cnn
    inp = torch.randn(2, 3, 512, 512)
    out = cnn(inp)
    total_params = 0 
    for param in cnn.parameters(): 
        total_params +=param.numel()
    print(f"Total parameters in the convnext: {total_params/1e6:.2f}MB") # 0.23MB


