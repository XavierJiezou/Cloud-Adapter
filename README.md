<div align="center">

![logo](https://github.com/user-attachments/assets/f9351412-d54a-4ac6-9344-d412fe3b3581)

# [TGRS 2025] Cloud-Adapter

Adapting Vision Foundation Models for Robust Cloud Segmentation in Remote Sensing Images

[![arXiv Paper](https://img.shields.io/badge/arXiv-2411.13127-B31B1B)](https://arxiv.org/abs/2411.13127)
[![Project Page](https://img.shields.io/badge/Project%20Page-Cloud--Adapter-blue)](https://xavierjiezou.github.io/Cloud-Adapter/)
[![HugginngFace Models](https://img.shields.io/badge/🤗HugginngFace-Models-orange)](https://huggingface.co/XavierJiezou/cloud-adapter-models)
[![HugginngFace Datasets](https://img.shields.io/badge/🤗HugginngFace-Datasets-orange)](https://huggingface.co/datasets/XavierJiezou/cloud-adapter-datasets)
<!--[![Overleaf](https://img.shields.io/badge/Overleaf-Open-green?logo=Overleaf&style=flat)](https://www.overleaf.com/project/6695fd4634d7fee5d0b838e5)-->

<!--Love the project? Please consider [donating](https://paypal.me/xavierjiezou?country.x=C2&locale.x=zh_XC) to help it improve!-->

![framework](https://github.com/user-attachments/assets/30920919-5894-468f-93e0-880b4822a5c1)

</div>

<!--This repository serves as the official implementation of the paper **"Adapting Vision Foundation Models for Robust Cloud Segmentation in Remote Sensing Images"**. It provides a comprehensive pipeline for semantic segmentation, including data preprocessing, model training, evaluation, and deployment, specifically tailored for cloud segmentation tasks in remote sensing imagery.-->

---

## 📢 Latest Updates

- **8/2025:** Our paper was accepted by **IEEE Transactions on Geoscience and Remote Sensing (TGRS)**.


## Installation  

1. Clone the Repository  

```bash  
git clone https://github.com/XavierJiezou/Cloud-Adapter.git
cd Cloud-Adapter  
```  

2. Install Dependencies  

You can either set up the environment manually or use our pre-configured environment for convenience:  

- Option 1: Manual Installation  

Ensure you are using Python 3.8 or higher, then install the required dependencies:  

```bash  
pip install -r requirements.txt  
```  

- Option 2: Use Pre-configured Environment  

We provide a pre-configured environment (`envs`) hosted on Hugging Face. You can download it directly from [Hugging Face](https://huggingface.co/XavierJiezou/cloud-adapter-models). Follow the instructions on the page to set up and activate the environment.  

## Prepare Data  

We have open-sourced all datasets used in the paper, which are hosted on [Hugging Face Datasets](https://huggingface.co/datasets/XavierJiezou/cloud-adapter-datasets). Please follow the instructions on the dataset page to download the data.  

After downloading, organize the dataset as follows:  

```  
Cloud-Adapter
├── ...
├── data
│   ├── cloudsen12_high_l1c
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   │   ├── test
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   │   ├── test
│   ├── cloudsen12_high_l2a
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   │   ├── test
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   │   ├── test
│   ├── gf12ms_whu_gf1
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   │   ├── test
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   │   ├── test
│   ├── gf12ms_whu_gf2
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   │   ├── test
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   │   ├── test
│   ├── hrc_whu
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   │   ├── test
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   │   ├── test
├── ...
```   

## Training

### Step 1: Download and Convert Weights

1. Download pretrained weights of vision foundation models

You can download the pretrained weights from the [DINOv2 official repository](https://github.com/facebookresearch/dinov2). 

Once downloading, you can convert the weights using the following command:

```bash
python tools/convert_models/convert_dinov2.py weight_path save_path --height image_height --width image_width
```

This command allows you to specify the desired image height and width for your use case.

You can also download the pretrained weights from [SAM official repository](https://github.com/facebookresearch/segment-anything). 

After downloading, use the following command to convert the weights:

```bash
python tools/convert_models/convert_sam.py weight_path save_path --height image_height --width image_width
```

### Step 2: Modify the Configuration File

After converting the backbone network weights, make sure to correctly specify the path to the configuration file within your config settings.

For example: 

```python
# configs/_base_/models/cloud_adapter_dinov2.py
model = dict(
    backbone=dict(
        type="CloudAdapterDinoVisionTransformer",
        init_cfg=dict(
            type="Pretrained",
            checkpoint="checkpoints/dinov2_converted.pth", # you can set weight path here
        ),
    ),
   
)
```

Update the `configs` directory with your training configuration, or use one of the provided example configurations. You can customize the backbone, dataset paths, and hyperparameters in the configuration file (e.g., `configs/adapter/cloud_adapter_pmaa_convnext_lora_16_adapter_all.py`).  

### Step 3: Start Training  

Use the following command to begin training:  

```bash  
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/adapter/cloud_adapter_pmaa_convnext_lora_16_adapter_all.py
```  

### Step 4: Resume or Fine-tune  

To resume training from a checkpoint or fine-tune using pretrained weights, run:  

```bash  
python tools/train.py configs/adapter/cloud_adapter_pmaa_convnext_lora_16_adapter_all.py --resume-from path/to/checkpoint.pth  
```

### Step 5: Generate Complete Weights

To optimize disk usage and accelerate training, the saved weights include only the adapter and head components.To synthesize the full weights, use the following command:
```python
python tools/generate_full_weights.py --segmentor_save_path full_weight_path backbone_path --backbone backbone_path --head adapter_and_head_weight_path
```
Make sure to provide the appropriate paths for the backbone and the adapter/head weights.

## Evaluation

All model weights used in the paper have been open-sourced and are available on [Hugging Face Models](https://huggingface.co/XavierJiezou/cloud-adapter-models).

Use the following command to evaluate the trained model:  

```bash  
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/adapter/cloud_adapter_pmaa_convnext_lora_16_adapter_all.py path/to/checkpoint.pth  
```  

### Special Evaluation: L8_Biome Dataset  

If you want to evaluate the model’s performance on different scenes of the **L8_Biome** dataset, you can run the following script:

```bash  
python tools/eval_l8_scene.py --config configs/to/path.py --checkpoint path/to/checkpoint.pth --img_dir data/l8_biome
```  

This will automatically evaluate the model across various scenes of the **L8_Biome** dataset, providing detailed performance metrics for each scene.  


## Reproducing Paper Comparisons  

If you would like to reproduce the other models and comparisons presented in the paper, please refer to our other repository: [CloudSeg](https://github.com/XavierJiezou/cloudseg). This repository contains the implementation and weights of the other models used for comparison in the study.

## Visualization

We have published the pre-trained model's visualization results of various datasets on Hugging Face at [Hugging Face](https://huggingface.co/XavierJiezou/cloud-adapter-models/tree/vis). If you prefer not to run the code, you can directly visit the repository to download the visualization results. 

## Gradio Demo  

We have created a **Gradio** demo to showcase the model's functionality. If you'd like to try it out, follow these steps:

1. Navigate to the `hugging_face` directory:

```bash  
cd hugging_face  
```

2. Run the demo:

```bash  
python app.py  
```

This will start the Gradio interface, where you can upload remote sensing images and visualize the model's segmentation results in real-time.

## Troubleshooting  

- If you encounter a `file not found` error, it is likely that the model weights have not been downloaded. Please visit [Hugging Face Models](https://huggingface.co/XavierJiezou/cloud-adapter-models) to download the pretrained model weights.

- **GPU Requirements**: To run the model on a GPU, you will need at least **16GB** of GPU memory.  

- **Running on CPU**: If you prefer to run the demo on CPU instead of GPU, set the following environment variable before running the demo:

```bash  
export CUDA_VISIBLE_DEVICES=-1  
```

## Citation

If you use our code or models in your research, please cite with:

```latex
@misc{cloud-adapter,
      title={Adapting Vision Foundation Models for Robust Cloud Segmentation in Remote Sensing Images}, 
      author={Xuechao Zou and Shun Zhang and Kai Li and Shiying Wang and Junliang Xing and Lei Jin and Congyan Lang and Pin Tao},
      year={2024},
      eprint={2411.13127},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.13127}, 
}
```

## Acknowledgments

[![Readme Card](https://github-readme-stats.vercel.app/api/pin/?username=open-mmlab&repo=mmsegmentation)](https://github.com/open-mmlab/mmsegmentation)

[![Readme Card](https://github-readme-stats.vercel.app/api/pin/?username=w1oves&repo=Rein)](https://github.com/w1oves/Rein)
