import torch
from torch import nn
from einops import rearrange
from torch import nn, einsum
from einops import rearrange
from mmseg.models.builder import MODELS
import math


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


class LoRAMLP(nn.Module):
    def __init__(self, in_dim, rank_dim, out_dim, bias):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, rank_dim, bias=bias),
            nn.LayerNorm(rank_dim),  # 添加LayerNorm
            nn.Linear(rank_dim, out_dim, bias=bias),
            nn.LayerNorm(out_dim)    # 添加LayerNorm
        )
        
    def forward(self, x):
        return self.net(x)


class LoRACrossAttention(nn.Module):
    def __init__(self, query_dim, rank_dim=8, context_dim=None, heads=2, dim_head=8):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = LoRAMLP(query_dim, rank_dim, inner_dim, bias=False)
        self.to_k = LoRAMLP(context_dim, rank_dim, inner_dim, bias=False)
        self.to_v = LoRAMLP(context_dim, rank_dim, inner_dim, bias=False)

        self.to_out = LoRAMLP(inner_dim, rank_dim, query_dim, bias=False)
        

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


class LoRACacheModule(nn.Module):
    def __init__(self, emd_dim=1024, rank_dim=8, cache_dim=16):
        super().__init__()
        self.main = LoRAMLP(emd_dim, rank_dim, emd_dim, bias=True)
        self.last = LoRAMLP(emd_dim, rank_dim, cache_dim, bias=True)
        self.fuse = LoRAMLP(2*cache_dim, rank_dim, cache_dim, bias=True)
        self.attn = LoRACrossAttention(emd_dim, rank_dim, cache_dim)

    def forward(self, x, cache=None):
        x_main = self.main(x)
        x_last = self.last(x)

        if cache is not None:
            cache = self.fuse(torch.cat([x_last, cache], dim=-1)) 

            attn_output = self.attn(x_main.permute(1, 0, 2), context=cache.permute(1, 0, 2))
            
            x_main = x_main + attn_output.permute(1, 0, 2)
            
        if cache is None:
            cache = x_last
            
        return x_main, cache

@MODELS.register_module()
class LoRACacheAdapter(nn.Module):
    def __init__(self, num_layers, emd_dim=1024, rank_dim=16, cache_dim=256,has_cnn=True):
        super().__init__()
        self.rank_dim = rank_dim
        self.cnn = nn.Identity()
        if has_cnn:
            self.cnn = InductionBias(3, cache_dim)
        self.net = nn.ModuleList(
            LoRACacheModule(emd_dim, rank_dim, cache_dim) 
            for i in range(num_layers)
        )
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        
    def forward(self, feats, layer, batch_first=True, has_cls_token=True, cache=None):
        if batch_first:
            feats = feats.permute(1, 0, 2)
        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=0)
            
        feats, cache = self.net[layer](feats, cache)
        
        if has_cls_token:
            feats = torch.cat([cls_token, feats], dim=0)
        if batch_first:
            feats = feats.permute(1, 0, 2)
        return feats, cache

if __name__ == "__main__":
    x = torch.randn((2, 1025, 1024))
    model = LoRACacheAdapter(24)
    output, cache = model(x, 0)
    print(output.shape, cache.shape)

    # output, cache = model(x, 1, cache=cache)
    # print(output.shape, cache.shape)
    
    # compute params
    total_params = 0 
    for param in model.parameters(): 
        total_params +=param.numel()
    print(f"Total parameters in the model: {total_params/1e6:.2f}MB")
        
    
    cnn = InductionBias()
    inp = torch.randn(2, 3, 512, 512)
    out = cnn(inp)
    total_params = 0 
    for param in cnn.parameters(): 
        total_params +=param.numel()
    print(f"Total parameters in the model: {total_params/1e6:.2f}MB") # 0.23MB


