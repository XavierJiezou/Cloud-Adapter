from mmseg.models.builder import BACKBONES, MODELS
from torch import nn as nn
from .loracacheadapter import LoRACacheAdapter
from .dino_v2 import DinoVisionTransformer
from .utils import set_requires_grad, set_train


@BACKBONES.register_module()
class LoRACacheAdapterDinoVisionTransformer(DinoVisionTransformer):
    def __init__(
        self,
        loracacheadapter_config=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.loracacheadapter: LoRACacheAdapter = MODELS.build(loracacheadapter_config)

    def forward_features(self, x, masks=None):
        B, _, h, w = x.shape
        first_cache = self.loracacheadapter.cnn(x)
        if isinstance(self.loracacheadapter.cnn,nn.Identity):
            first_cache = None
        H, W = h // self.patch_size, w // self.patch_size
        x = self.prepare_tokens_with_masks(x, masks)
        outs = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            x, cache = self.loracacheadapter.forward(
                x,
                idx,
                batch_first=True,
                has_cls_token=True,
                cache=first_cache if idx == 0 else cache,
            )
            if idx in self.out_indices:
                outs.append(
                    x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
                )
        return outs

    def train(self, mode: bool = True):
        if not mode:
            return super().train(mode)
        set_requires_grad(self, ["loracacheadapter"])
        set_train(self, ["loracacheadapter"])

    def state_dict(self, destination, prefix, keep_vars):
        state = super().state_dict(destination, prefix, keep_vars)
        keys = [k for k in state.keys() if "loracacheadapter" not in k]
        for key in keys:
            state.pop(key)
            if key in destination:
                destination.pop(key)
        return state

if __name__ == "__main__":
    model = LoRACacheAdapterDinoVisionTransformer(
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
        loracacheadapter_config=dict(
            type="LoRACacheAdapter",
            emd_dim=1024,
            num_layers=24,
            rank_dim=8,
            cache_dim=16,
        ),
    )
