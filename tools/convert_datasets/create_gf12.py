import os
from PIL import Image
import numpy as np
from dataset.gf12ms_whu import GF12MSWHU
import albumentations
from tqdm import tqdm
import sys
import argparse


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--root",
        type=str,
        help="gf12数据集路径",
        default="/data/zouxuechao/cloudseg/gf12ms_whu",
    )
    parse.add_argument(
        "--save_path", type=str, help="数据集保存路径", default="/data/zouxuechao/mmseg"
    )
    args = parse.parse_args()
    return args.root, args.save_path


def imgRGB2P(ann, dst_path):
    # 定义调色板的索引
    bin_colormap_reverse = np.array([[0, 0, 0], [255, 255, 255]]).astype(
        np.uint8
    )  # 会按照值的循序进行索引
    # 转化为p模式
    img_p = ann.convert("P")
    img_p.putpalette(bin_colormap_reverse)
    img_p.save(dst_path)


def get_dataset(root,save_path,phase, serial):
    all_transform = albumentations.PadIfNeeded(
        min_height=256, min_width=256, p=1, always_apply=True
    )
    dataset = GF12MSWHU(
        root=root, phase=phase, serial=serial, all_transform=all_transform
    )
    child_root = "gf12ms_whu_gf1" if serial == "gf1" else "gf12ms_whu_gf2"
    os.makedirs(os.path.join(save_path,child_root, "img_dir", phase), exist_ok=True)
    os.makedirs(os.path.join(save_path,child_root, "ann_dir", phase), exist_ok=True)

    for data in tqdm(
        dataset, total=len(dataset), desc=f"{serial}-{phase} processing..."
    ):
        img_path = data["img_path"]
        filename = img_path.split(os.path.sep)[-1]

        filename = filename.replace("tiff", "png")

        ann = data["ann"].astype(np.uint8)

        img = data["img"]

        img = (img * 255).astype(np.uint8)

        img = Image.fromarray(img)
        ann = Image.fromarray(ann)

        img.save(os.path.join(save_path,child_root, "img_dir", phase, filename))
        imgRGB2P(ann, os.path.join(save_path,child_root, "ann_dir", phase, filename))


if __name__ == "__main__":
    # use example:PYTHONPATH=$PYTHONPATH:. python tools/convert_datasets/create_gf12.py --root /data/zouxuechao/cloudseg/gf12ms_whu --save_path /data/zouxuechao/mmseg
    root,save_path = get_args()
    for phase in ["train","val"]:
        for serial in ["gf1","gf2"]:
            get_dataset(root,save_path,phase,serial)

    # get_dataset("train",serial="gf1")
    # get_dataset("val",serial="gf1")

    # get_dataset("train",serial="gf2")
    # get_dataset("val",serial="gf2")
