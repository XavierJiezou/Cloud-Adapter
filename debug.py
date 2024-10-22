from mmseg.apis import init_model
import torch
import rein.models

config_file = 'configs/dinov2/head_dinov2_l_mask2former_gf1.py'
# checkpoint_file = 'work_dirs/full_dinov2_mask2former_hrc_whu/best_mIoU_iter_900.pth'

device = 'cuda:0'

# 通过配置文件和模型权重文件构建模型
model = init_model(config_file, device=device)

inp = torch.randn((2,3,512,512),device=device)

backbone = model.backbone
head = model.decode_head

