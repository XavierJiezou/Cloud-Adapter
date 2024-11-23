import os
import albumentations
import albumentations.pytorch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import List
import tifffile as tf


class L8BiomeCrop(Dataset):
    METAINFO = dict(
        classes = ("Clear", "Cloud Shadow", "Thin Cloud", "Cloud"),
        palette=(
            (0, 0, 0),
            (85, 85, 85),
            (170, 170, 170),
            (255, 255, 255),
        ),
        img_size=(512, 512),
        ann_size=(512, 512),
    )

    def __init__(
        self, 
        root: str = "data/l8_biome_crop",
        bands: List[str] = ["B4", "B3", "B2"],
        phase: str = "train",
        all_transform: albumentations.Compose = None,
        img_transform: albumentations.Compose = None,
        ann_transform: albumentations.Compose = None
    ):
        self.root = root
        self.bands = bands
        self.phase = phase
        self.all_transform = all_transform
        self.img_transform = img_transform
        self.ann_transform = ann_transform
        
        self.data = self.load_data()
        


    def load_data(self):
        data = []
        maps = {
            "barren": "Barren",
            "forest": "Forest",
            "grass": "Grass/Crops",
            "shrubland": "Shrubland",
            "snow":"Snow/Ice",
            "urban": "Urban",
            "water": "Water",
            "wetlands":"Wetlands",
        }
        split_file = os.path.join(self.root, f'{self.phase}.txt')
        with open(split_file, 'r') as f:
            for line in f:
                image_file = line.strip()
                img_path = os.path.join(self.root, 'img_dir', image_file)
                ann_path = os.path.join(self.root, 'ann_dir', image_file)
                lac_type = image_file.split('_')[0]
                lac_type = maps[lac_type]
                data.append((img_path, ann_path, lac_type))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, mask_path, lac_type = self.data[idx]
        
        image = tf.imread(image_path) # H, W, C
        
        if len(self.bands)>11:
            raise ValueError("The number of bands must be less than 11")
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
                elif band == "B5":
                    tmp[:,:,i] = image[:,:,4]
                elif band == "B6":
                    tmp[:,:,i] = image[:,:,5]
                elif band == "B7":
                    tmp[:,:,i] = image[:,:,6]
                elif band == "B8":
                    tmp[:,:,i] = image[:,:,7]
                elif band == "B9":
                    tmp[:,:,i] = image[:,:,8]
                elif band == "B10":
                    tmp[:,:,i] = image[:,:,9]
                elif band == "B11":
                    tmp[:,:,i] = image[:,:,10]
                else:
                    raise ValueError("The band must be one of 'B1','B2','B3','B4', 'B5','B6', 'B7', 'B8', 'B9', 'B10', 'B11', but got {}".format(band))
            image = tmp
            
        image = (image - image.min()) / (image.max() - image.min() + 1e-6) # normalize
        mask = tf.imread(mask_path) # (H, W)

        if self.all_transform:
            transformed = self.all_transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        if self.img_transform:
            image = self.img_transform(image=image)["image"]
        if self.ann_transform:
            mask = self.ann_transform(image=mask)["image"]

        return {
            'img': image,
            'ann': np.int64(mask),
            'img_path': image_path,
            'ann_path': mask_path,
            'lac_type': lac_type,
        }
        

def show_l8_biome_crop():
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
    
    dataset = L8BiomeCrop(
        all_transform=all_transform,
        img_transform=img_transform,
    )
    
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    for sample in dataset:
        print(sample['img'].shape, sample['ann'].shape) 
        axs[0].imshow(sample['img'].permute(1, 2, 0)*2.5)
        axs[0].set_title('Image')
        color_map = np.array(dataset.METAINFO['palette'])
        color_ann = color_map[sample['ann']]
        axs[1].imshow(color_ann)
        axs[1].set_title('Annotation')
        plt.suptitle(f'Land Cover Type: {sample["lac_type"]}')
        plt.savefig('l8_biome_crop.png', bbox_inches="tight")
        # break


if __name__ == '__main__':
    show_l8_biome_crop()
