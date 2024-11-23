import os
from PIL import Image
import numpy as np
from dataset.l8_biome_crop import L8BiomeCrop
from tqdm import tqdm
import sys
import argparse


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--root",
        type=str,
        help="l8_biome数据集路径",
        default="/home/zouxuechao/zs/rein/data/l8_biome_crop",
    )
    parse.add_argument(
        "--save_path", type=str, help="数据集保存路径", default="/data/zouxuechao/mmseg"
    )
    args = parse.parse_args()
    return args.root, args.save_path



def get_dataset(root,save_path,phase):
    dataset = L8BiomeCrop(
        root=root, phase=phase
    )
    child_root = "l8_biome"
    os.makedirs(os.path.join(save_path,child_root, "img_dir", phase), exist_ok=True)
    os.makedirs(os.path.join(save_path,child_root, "ann_dir", phase), exist_ok=True)

    for data in tqdm(
        dataset, total=len(dataset), desc=f"l8_biome-{phase} processing..."
    ):
        img_path = data["img_path"]
        filename = img_path.split(os.path.sep)[-1]

        filename = filename.split(".")[0]

        filename = filename + ".png"

        ann = data["ann"].astype(np.uint8)

        img = data["img"]

        img = (img * 255).astype(np.uint8)

        img = Image.fromarray(img)
        ann = Image.fromarray(ann)

        img.save(os.path.join(save_path,child_root, "img_dir", phase, filename))
        ann.save(os.path.join(save_path,child_root, "ann_dir", phase, filename))

if __name__ == "__main__":
    # use example: PYTHONPATH=$PYTHONPATH:. python tools/convert_datasets/create_l8.py --root /home/zouxuechao/zs/rein/data/l8_biome_crop --save_path /data/zouxuechao/mmseg
    root,save_path = get_args()
    for phase in ["train","val","test"]:
        get_dataset(root,save_path,phase)
