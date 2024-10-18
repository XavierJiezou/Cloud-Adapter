import os
from glob import glob
from typing import Literal, List

import albumentations
import numpy as np
import tifffile as tf
from PIL import Image
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import albumentations.pytorch
from src.utils.stretch import gaussian_stretch


class GF12MSWHU(Dataset):
    METAINFO = dict(
        classes=("clear", "cloud"),
        palette=((0, 0, 0), (255, 255, 255)),
        img_size=(250, 250),
        ann_size=(250, 250),
    )

    def __init__(
            self,
            root: str = "data/gf12ms_whu",
            phase: Literal["train", "val", "test"] = "train",
            serial: Literal["gf1", "gf2", "all"] = "all",
            bands: List[str] = ["B3", "B2", "B1"], # only B1, B2, B3, B4 are available, B4 is nir, and B3, B2, B1 are rgb
            all_transform: albumentations.Compose = None,
            img_transform: albumentations.Compose = None,
            ann_transform: albumentations.Compose = None,
    ) -> None:
        super().__init__()
        self.image_paths, self.mask_paths = self.__load_data(root, phase,serial)
        self.bands = bands
        self.all_transform = all_transform
        self.img_transform = img_transform
        self.ann_transform = ann_transform

    def __load_data(self, root:str, phase:str, serial:str):
        if phase == "train":
            filename = "TrainBlock250"
        elif phase == "val" or phase == "test":
            filename = "TestBlock250"
        else:
            raise ValueError(
                "phase must be one of 'train','val','test', but got {}".format(phase)
            )
            
        if serial == "all":
            serial = "**"
        elif serial == "gf1":
            serial = "GF1MS-WHU"
        elif serial == "gf2":
            serial = "GF2MS-WHU"
        else:
            raise ValueError("serial must be one of 'gf1','gf2','all', but got {}".format(serial))
        
        mask_paths = glob(os.path.join(root, serial, filename, "*.tif"))
        image_paths = [
            filename.replace("_Mask", "").replace("tif", "tiff")
            for filename in mask_paths
        ]
        return image_paths, mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = tf.imread(image_path).transpose(1, 2, 0) # (C, H, W) -> (H, W, C)
        
        # bands
        if len(self.bands)>4:
            raise ValueError("The number of bands must be less than 4")
        else:
            tmp = np.zeros((image.shape[0], image.shape[1], len(self.bands)), dtype=np.float32)
            for i, band in enumerate(self.bands):
                if band == "B1":
                    tmp[:,:,i] = image[:,:,0]
                elif band == "B2":
                    tmp[:,:,i] = image[:,:,1]
                elif band == "B3":
                    tmp[:,:,i] = image[:,:,2]
                elif band == "B4":
                    tmp[:,:,i] = image[:,:,3]
                else:
                    raise ValueError("The band must be one of 'B1','B2','B3','B4', but got {}".format(band))
            image = tmp
            
        image = (image - image.min()) / (image.max() - image.min() + 1e-6) # normalize
        
        mask = np.array(Image.open(mask_path)) # (H, W)

        if self.all_transform:
            transformed = self.all_transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        if self.img_transform:
            image = self.img_transform(image=image)["image"]
        if self.ann_transform:
            mask = self.ann_transform(image=mask)["image"]

        return {"img": image, "ann": np.int64(mask), "img_path": image_path}


def show_gf12ms_whu():
    all_transform = albumentations.Compose([
            albumentations.OneOf([
                albumentations.HorizontalFlip(p=0.5),
                albumentations.VerticalFlip(p=0.5),
                albumentations.RandomRotate90(p=0.5),
                albumentations.Transpose(p=0.5),
            ], p=1),
        ])
    
    img_transform = albumentations.Compose([
        albumentations.pytorch.ToTensorV2(),
    ])
    
    gf1_train_dataset = GF12MSWHU(phase="train", serial="gf1", all_transform=all_transform, img_transform=img_transform)
    
    gf2_train_dataset = GF12MSWHU(phase="train", serial="gf2", all_transform=all_transform, img_transform=img_transform)
    
    for gf1, gf2 in zip(gf1_train_dataset, gf2_train_dataset):
        plt.figure(figsize=(12, 12))
    
        plt.subplot(2, 2, 1)
        img = gf1["img"].permute(1, 2, 0)
        img = (img*255).numpy().astype(np.uint8)
        plt.imshow(gaussian_stretch(img))
        plt.title("GF1_img")
        plt.axis("off")
        
        plt.subplot(2, 2, 2)
        plt.imshow(gf1["ann"])
        plt.title("GF1_ann")
        plt.axis("off")
        
        plt.subplot(2, 2, 3)
        img = gf2["img"].permute(1, 2, 0)
        img = (img*255).numpy().astype(np.uint8)
        plt.imshow(gaussian_stretch(img))
        plt.title("GF2_img")
        plt.axis("off")
        
        plt.subplot(2, 2, 4)
        plt.imshow(gf2["ann"])
        plt.title("GF2_ann")
        plt.axis("off")
        
        plt.savefig("gf12ms_whu.png", bbox_inches="tight", pad_inches=0)
    

if __name__ == "__main__":
    show_gf12ms_whu()
