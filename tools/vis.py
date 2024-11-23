import numpy as np
from PIL import Image

img_root = "data/cloudsen12_high_l2a/img_dir/test"
ann_root = "data/cloudsen12_high_l2a/ann_dir/test"
# palette for 4 classes segmentation mask
palette = [
    [79, 253, 199], 
    [77, 2, 115], 
    [251, 255, 41],
    [221, 53, 223],
]
for i in range(1, 100):
    img = Image.open(f"{img_root}/{i}.png")
    ann = Image.open(f"{ann_root}/{i}.png")
    ann = np.array(ann)
    segmap = np.zeros((ann.shape[0], ann.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        segmap[ann == label] = color
    segmap = Image.fromarray(segmap)
    segmap.save(f"vis_img/{i}_vis.png")
    img.save(f"vis_ann/{i}_vis.png")