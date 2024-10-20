import os
from PIL import Image
import numpy as np
from dataset.cloudsen12_high import CloudSEN12High
from tqdm import tqdm
import sys
import argparse


def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--root",
        type=str,
        help="cloudseghigh数据集路径",
        default="/data/zouxuechao/cloudseg/cloudsen12_high",
    )
    parse.add_argument(
        "--save_path", type=str, help="数据集保存路径", default="/data/zouxuechao/mmseg"
    )
    args = parse.parse_args()
    return args.root, args.save_path



def get_dataset(root,save_path,phase,level):
    dataset = CloudSEN12High(
        root=root, phase=phase,level=level
    )
    child_root = "cloudsen12_high_l1c" if level == "l1c" else "cloudsen12_high_l2a"
    os.makedirs(os.path.join(save_path,child_root, "img_dir", phase), exist_ok=True)
    os.makedirs(os.path.join(save_path,child_root, "ann_dir", phase), exist_ok=True)
    index = 0
    for data in tqdm(
        dataset, total=len(dataset), desc=f"cloudsen12_high-{level}-{phase} processing..."
    ):

        filename = f"{index}.png"

        ann = data["ann"].astype(np.uint8)

        img = data["img"]

        img = (img * 255).astype(np.uint8)

        img = Image.fromarray(img)
        ann = Image.fromarray(ann)

        img.save(os.path.join(save_path,child_root, "img_dir", phase, filename))
        ann.save(os.path.join(save_path,child_root, "ann_dir", phase, filename))

        index += 1

if __name__ == "__main__":
    # use example: PYTHONPATH=$PYTHONPATH:. python tools/convert_datasets/create_l1c_l2a.py --root /data/zouxuechao/cloudseg/cloudsen12_high --save_path /data/zouxuechao/mmseg
    root,save_path = get_args()
    for phase in ["train","val","test"]:
        for level in ["l1c","l2a"]:
            get_dataset(root,save_path,phase,level)
