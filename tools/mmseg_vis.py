from mmseg.apis import MMSegInferencer, init_model, inference_model, show_result_pyplot
import sys
import numpy as np
import argparse
from rich.progress import track
from PIL import Image
import torch
from glob import glob
from typing import List, Tuple
import torchvision
import os
import os.path as osp
os.chdir(osp.abspath(osp.dirname(osp.dirname(__file__))))

sys.path.append(os.curdir)


def get_args() -> Tuple[str, str, str, str, int]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, help='Image file')
    parser.add_argument('--config', type=str, help='config path')
    parser.add_argument('--checkpoint', type=str, help='checkpoint path')
    parser.add_argument('--device', type=str, help='cpu/cuda:0', default="cpu")
    args = parser.parse_args()
    return args.dataset_name, args.config, args.checkpoint, args.device


def get_image_sub_path(dataset_name: str) -> str:
    if dataset_name in ["cloudsen12_high_l1c", "cloudsen12_high_l2a", "l8_biome", "hrc_whu"]:
        return "test"
    return "val"


def get_image_list(dataset_name: str) -> List[str]:
    # data/cloudsen12_high_l1c/img_dir/test/0.png
    image_sub_dir = get_image_sub_path(dataset_name)
    image_list = glob(os.path.join("data", dataset_name,
                      "img_dir", image_sub_dir, "*"))
    return image_list


def get_classes(dataset_name: str) -> int:
    if dataset_name in ["cloudsen12_high_l1c", "cloudsen12_high_l2a"]:
        return ["clear", "thick cloud", "thin cloud", "cloud shadow"]
    if dataset_name == "l8_biome":
        return ["Clear", "Cloud Shadow", "Thin Cloud", "Cloud"]
    if dataset_name in ["gf12ms_whu_gf1", "gf12ms_whu_gf2", "hrc_whu"]:
        return ['clear sky', 'cloud']
    raise Exception("dataset_name not supported")


def get_palette(dataset_name: str) -> List[Tuple[int, int, int]]:
    if dataset_name in ["cloudsen12_high_l1c", "cloudsen12_high_l2a"]:
        return [79, 253, 199, 77, 2, 115, 251, 255, 41, 221, 53, 223]
    if dataset_name == "l8_biome":
        return [79, 253, 199, 221, 53, 223, 251, 255, 41, 77, 2, 115]
    if dataset_name in ["gf12ms_whu_gf1", "gf12ms_whu_gf2", "hrc_whu"]:
        return [79, 253, 199, 77, 2, 115]
    raise Exception("dataset_name not supported")


def give_colors_to_mask(mask: np.ndarray, colors=None,save_path:str=None) -> np.ndarray:
    """将mask转换为彩色

    
    """
    # 使用pillow 的p 模式将Mask进行上色

    im = Image.fromarray(mask.astype(np.uint8)).convert("P")
    

    im.putpalette(colors)
    im.save(save_path)


def inference():
    import cloud_adapter
    import cloud_adapter.models
    dataset_name, config, checkpoint, device = get_args()
    model = init_model(config, checkpoint, device)
    img_list = get_image_list(dataset_name)
    colors = get_palette(dataset_name)
    os.makedirs(os.path.join("visualization", dataset_name,"cloud-adapter"), exist_ok=True)
    os.makedirs(os.path.join("visualization", dataset_name,"label"), exist_ok=True)
    os.makedirs(os.path.join("visualization", dataset_name,"input"), exist_ok=True)
    for img_path in track(img_list,total=len(img_list)):

        result = inference_model(model, img_path)
        ann_path = img_path.replace("img_dir", "ann_dir")
        gt = np.array(Image.open(ann_path))
        img = np.array(Image.open(img_path).convert("RGB"))
        pred_sem_seg: torch.Tensor = result.pred_sem_seg.data
        pred_mask = pred_sem_seg.cpu().squeeze().numpy()
        
        filename = osp.basename(img_path).split(".")[0] + ".png"
        give_colors_to_mask(pred_mask, colors,os.path.join("visualization", dataset_name,"cloud-adapter",filename))

        give_colors_to_mask(gt,colors,os.path.join("visualization", dataset_name,"label",filename))
        Image.fromarray(img).save(os.path.join("visualization", dataset_name,"input",filename))
        
        
if __name__ == '__main__':
    # examlr usage:python tools/mmseg_vis.py --dataset_name hrc_whu --config work_dirs/ours_adapter_pmaa_convnext_lora_16_adapter_all_hrc_whu/ours_adapter_pmaa_convnext_lora_16_adapter_all_hrc_whu.py --checkpoint work_dirs/ours_adapter_pmaa_convnext_lora_16_adapter_all_hrc_whu/full_weight.pth --device cuda:3
    # main()
    inference()
