<div align="center">

![logo](https://github.com/user-attachments/assets/f9351412-d54a-4ac6-9344-d412fe3b3581)

# Cloud-Adapter

Cloud Segmentation for Remote Sensing Images.

[![python](https://img.shields.io/badge/-Python_3.9_%7C_3.10_%7C_3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![demo](https://img.shields.io/badge/🤗HugginngFace-Spaces-orange)](https://huggingface.co/spaces/caixiaoshun/cloudseg)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/XavierJiezou/cloudseg#license)
[![Template](https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray)](https://github.com/ashleve/lightning-hydra-template)

<!--Love the project? Please consider [donating](https://paypal.me/xavierjiezou?country.x=C2&locale.x=zh_XC) to help it improve!-->

</div>

# Introduction  

This repository serves as the official implementation of the paper **"Adapting Vision Foundation Models for Robust Cloud Segmentation in Remote Sensing Images"**. It provides a comprehensive pipeline for semantic segmentation, including data preprocessing, model training, evaluation, and deployment, specifically tailored for cloud segmentation tasks in remote sensing imagery.  

---


## Quick Start  

### 1. Clone the Repository  

```bash  
git clone https://github.com/XavierJiezou/Cloud-Adapter.git
cd Cloud-Adapter  
```  

### 2. Install Dependencies  

You can either set up the environment manually or use our pre-configured environment for convenience:  

#### Option 1: Manual Installation  

Ensure you are using Python 3.8 or higher, then install the required dependencies:  

```bash  
pip install -r requirements.txt  
```  

#### Option 2: Use Pre-configured Environment  

We provide a pre-configured environment (`envs`) hosted on Hugging Face. You can download it directly from [Hugging Face](https://huggingface.co/XavierJiezou/cloud-adapter-models). Follow the instructions on the page to set up and activate the environment.  

---

### 3. Prepare Data  

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

### 4. Model Weights  

All model weights used in the paper have been open-sourced and are available on [Hugging Face Models](https://huggingface.co/XavierJiezou/cloud-adapter-models). You can download the pretrained models and directly integrate them into your pipeline.  

To use a pretrained model, specify the path to the downloaded weights in your configuration file or command-line arguments.  

---

### 5. Train the Model  

We utilize the [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) framework for training. Please ensure you have the MMSegmentation library installed and the configuration file properly set up.  

#### Step 1: Modify the Configuration File  

Update the `configs` directory with your training configuration, or use one of the provided example configurations. You can customize the backbone, dataset paths, and hyperparameters in the configuration file (e.g., `configs/adapter/cloud_adapter_pmaa_convnext_lora_16_adapter_all.py`).  

#### Step 2: Start Training  

Use the following command to begin training:  

```bash  
CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/adapter/cloud_adapter_pmaa_convnext_lora_16_adapter_all.py
```  

#### Step 3: Resume or Fine-tune  

To resume training from a checkpoint or fine-tune using pretrained weights, run:  

```bash  
python tools/train.py configs/adapter/cloud_adapter_pmaa_convnext_lora_16_adapter_all.py --resume-from path/to/checkpoint.pth  
```  

### 6. Evaluate the Model  

Use the following command to evaluate the trained model:  

```bash  
CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/adapter/cloud_adapter_pmaa_convnext_lora_16_adapter_all.py path/to/checkpoint.pth  
```  

#### Special Evaluation: L8_Biome Dataset  

If you want to evaluate the model’s performance on different scenes of the **L8_Biome** dataset, you can run the following script:

```bash  
python tools/eval_l8_scene.py --config configs/to/path.py --checkpoint path/to/checkpoint.pth --img_dir data/l8_biome
```  

This will automatically evaluate the model across various scenes of the **L8_Biome** dataset, providing detailed performance metrics for each scene.  


#### Reproducing Paper Comparisons  

If you would like to reproduce the other models and comparisons presented in the paper, please refer to our other repository: [CloudSeg](https://github.com/XavierJiezou/cloudseg). This repository contains the implementation and weights of the other models used for comparison in the study.

## Visualization

We have published the pre-trained model's visualization results of various datasets on Hugging Face at [Hugging Face](https://huggingface.co/XavierJiezou/cloud-adapter-models/tree/vis). If you prefer not to run the code, you can directly visit the repository to download the visualization results. 

### 7. Gradio Demo  

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

#### Troubleshooting  

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

[![Readme Card](https://github-readme-stats.vercel.app/api/pin/?username=open-mmlab&repo=mmsegmentation)]([https://github.com/python-poetry/poetry](https://github.com/open-mmlab/mmsegmentation))

[![Readme Card](https://github-readme-stats.vercel.app/api/pin/?username=w1oves&repo=Rein)](https://github.com/w1oves/Rein)
