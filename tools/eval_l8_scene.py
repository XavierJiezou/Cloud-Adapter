import argparse
from glob import glob
import os
import numpy as np
from PIL import Image
from mmeval import MeanIoU
import torch
from mmseg.apis import init_model, inference_model
from rich.progress import track


def parse_args():
    default_config = "work_dirs/ours_adapter_pmaa_convnext_lora_16_adapter_all_l8_load_head_40w/ours_adapter_pmaa_convnext_lora_16_adapter_all_l8_load_head_40w.py"
    default_weight = "work_dirs/ours_adapter_pmaa_convnext_lora_16_adapter_all_l8_load_head_40w/full_weight.pth"
    parser = argparse.ArgumentParser(
        description="MMSeg test (and eval) a model")
    parser.add_argument(
        "--config", help="Path to the training configuration file.", default=default_config)
    parser.add_argument(
        "--checkpoint", help="Path to the checkpoint file for both the REIN and head models.", default=default_weight)
    parser.add_argument(
        "--img_dir", help="Path to the directory containing images to be processed.", default="data/l8_biome")
    parser.add_argument("--device", default="cuda:1")

    args = parser.parse_args()
    return args.config, args.checkpoint, args.img_dir, args.device


def get_img_list(img_dir: str):
    image_list = glob(os.path.join(img_dir, "img_dir", "test", "*"))
    assert len(image_list) > 0, f"{img_dir} is empty"
    return image_list


def main():
    import cloud_adapter
    import cloud_adapter.models

    config, checkpoint, img_dir, device = parse_args()
    model = init_model(config, checkpoint, device)
    image_list = get_img_list(img_dir)

    scenes_cls = [
        "Grass/Crops",
        "Urban",
        "Wetlands",
        "Snow/Ice",
        "Barren",
        "Forest",
        "Shrubland",
        "Water",
    ]
    scene_mapping = {
        "grass":"Grass/Crops",
        "urban":"Urban",
        "wetlands":"Wetlands",
        "forest":"Forest",
        "shrubland":"Shrubland",
        "snow":"Snow/Ice",
        "barren":"Barren",
        "water":"Water"
    }
    scene_metrics = {scene: {} for scene in scenes_cls}
    miou = MeanIoU(num_classes=4)
    for image_path in track(image_list, total=len(image_list)):
        ann_path = image_path.replace("img_dir", "ann_dir")
        gt = np.array(Image.open(ann_path))
        gt = gt[np.newaxis]
        result = inference_model(model, image_path)
        pred_sem_seg: np.ndarray = result.pred_sem_seg.data.cpu().numpy()
        result_iou = miou(pred_sem_seg, gt)
        scene = os.path.basename(image_path).split("_")[0]
        scene = scene_mapping[scene]
        if "mIoU" not in scene_metrics[scene]:
            scene_metrics[scene]["mIoU"] = []
        scene_metrics[scene]["mIoU"].append(result_iou['mIoU'])
        
        if "aAcc" not in scene_metrics[scene]:
            scene_metrics[scene]["aAcc"] = []
        scene_metrics[scene]["aAcc"].append(result_iou['aAcc'])

        if "mAcc" not in scene_metrics[scene]:
            scene_metrics[scene]["mAcc"] = []
        scene_metrics[scene]["mAcc"].append(result_iou['mAcc'])

        if "mDice" not in scene_metrics[scene]:
            scene_metrics[scene]["mDice"] = []
        scene_metrics[scene]["mDice"].append(result_iou['mDice'])
    
    # 计算平均指标
    for scene in scenes_cls:
        scene_metrics[scene]["mIoU"] = sum(scene_metrics[scene]["mIoU"]) / len(scene_metrics[scene]["mIoU"])
        scene_metrics[scene]["aAcc"] = sum(scene_metrics[scene]["aAcc"]) / len(scene_metrics[scene]["aAcc"])
        scene_metrics[scene]["mAcc"] = sum(scene_metrics[scene]["mAcc"]) / len(scene_metrics[scene]["mAcc"])
        scene_metrics[scene]["mDice"] = sum(scene_metrics[scene]["mDice"]) / len(scene_metrics[scene]["mDice"])

    # 将结果保存为json文件
    import json
    with open("results.json", "w") as f:
        json.dump(scene_metrics, f,indent=4)
        
        


if __name__ == "__main__":
    main()
