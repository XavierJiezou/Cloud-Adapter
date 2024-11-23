from mmseg.models.builder import BACKBONES, MODELS
from torch import nn as nn
from .convnext_adapter import AdapterConvNeXtBlock
from .dino_v2 import DinoVisionTransformer
from .utils import set_requires_grad, set_train

@BACKBONES.register_module()
class ConvnextDinoVisionTransformer(DinoVisionTransformer):
    def __init__(
        self,
        convnext_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # self.convnext: AdapterConvNeXtBlock = MODELS.build(convnext_config)
        self.convnext_list = nn.ModuleList([MODELS.build(convnext_config) for _ in range(self.n_blocks-1)])
    
    def forward_features(self, x, masks=None):
        B, _, h, w = x.shape
        H, W = h // self.patch_size, w // self.patch_size
        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if idx != len(self.blocks) - 1:
                x = self.convnext_list[idx].forward(
                    x,
                    h=H,
                    w=W,
                )
            if idx in self.out_indices:
                outs.append(
                    x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
                )
        return outs

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["convnext"])
        set_train(self, ["convnext"])

    def state_dict(self, destination, prefix, keep_vars):
        state = super().state_dict(destination, prefix, keep_vars)
        keys = [k for k in state.keys() if "convnext" not in k]
        for key in keys:
            state.pop(key)
            if key in destination:
                destination.pop(key)
        return state