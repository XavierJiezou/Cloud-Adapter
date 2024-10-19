from mmseg.apis import init_model
import torch
import rein.models

config_file = 'work_dirs/head_dinov2_mask2former_hrc_whu/head_dinov2_mask2former_hrc_whu.py'
# checkpoint_file = 'work_dirs/full_dinov2_mask2former_hrc_whu/best_mIoU_iter_900.pth'

device = 'cuda:0'

# 通过配置文件和模型权重文件构建模型
model = init_model(config_file, device=device)

inp = torch.randn((2,3,256,256),device=device)
backbone = model.backbone
head = model.decode_head

# 打印backbone的参数数量（以百万为单位）
backbone_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
print(f"Backbone trainable parameters: {backbone_params / 1e6:.2f}M")

# 打印head的参数数量（以百万为单位）
head_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
print(f"Head trainable parameters: {head_params / 1e6:.2f}M")