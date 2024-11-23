from mmseg.apis import init_model
from typing import List
from glob import glob
from cloud_adapter.cloud_adapter_dinov2 import CloudAdapterDinoVisionTransformer
import numpy as np
from PIL import Image
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
import gradio as gr
import torch
import os


class CloudAdapterGradio:
    def __init__(self, config_path=None, checkpoint_path=None, device="cpu", example_inputs=None, num_classes=2, palette=None):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model: EncoderDecoder = init_model(
            self.config_path, self.checkpoint_path, device=self.device)
        self.model.eval()
        self.example_inputs = example_inputs
        self.img_size = 256 if num_classes == 2 else 512
        self.palette = palette
        self.legend = self.html_legend(num_classes=num_classes)

        self.create_ui()

    def html_legend(self, num_classes=2):
        if num_classes == 2:
            return """
        <div style="margin-top: 10px; text-align: left; display: flex; align-items: center; gap: 20px;justify-content: center;">
            <div style="display: flex; align-items: center;">
                <div style="width: 20px; height: 20px; background-color: rgb(79, 253, 199); margin-right: 10px; "></div>
                <span>Clear</span>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="width: 20px; height: 20px; background-color: rgb(77, 2, 115); margin-right: 10px; "></div>
                <span>Cloud</span>
            </div>
        </div>
        """
        return """
        <div style="margin-top: 10px; text-align: left; display: flex; align-items: center; gap: 20px;justify-content: center;">
            <div style="display: flex; align-items: center;">
                <div style="width: 20px; height: 20px; background-color: rgb(79, 253, 199); margin-right: 10px; "></div>
                <span>Clear Sky</span>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="width: 20px; height: 20px; background-color: rgb(77, 2, 115); margin-right: 10px; "></div>
                <span>Thick Cloud</span>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="width: 20px; height: 20px; background-color: rgb(251, 255, 41); margin-right: 10px; "></div>
                <span>Thin Cloud</span>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="width: 20px; height: 20px; background-color: rgb(221, 53, 223); margin-right: 10px; "></div>
                <span>Cloud Shadow</span>
            </div>
        </div>
"""

    def create_ui(self):
        with gr.Row():
            # 左侧：输入图片和按钮
            with gr.Column(scale=1):  # 左侧列
                in_image = gr.Image(
                    label='Input Image',
                    sources='upload',
                    elem_classes='input_image',
                    interactive=True,
                    type="pil",
                )
                with gr.Row():
                    run_button = gr.Button(
                        'Run',
                        variant="primary",
                    )
                # 示例输入列表
                gr.Examples(
                    examples=self.example_inputs,
                    inputs=in_image,
                    label="Example Inputs"
                )

            # 右侧：输出图片
            with gr.Column(scale=1):  # 右侧列
                with gr.Column():
                    # 输出图片
                    out_image = gr.Image(
                        label='Output Image',
                        elem_classes='output_image',
                        interactive=False
                    )
                    # 图例
                    legend = gr.HTML(
                        value=self.legend,
                        elem_classes="output_legend",
                    )

        # 按钮点击逻辑：触发图像转换
        run_button.click(
            self.inference,
            inputs=in_image,
            outputs=out_image,
        )

    @torch.no_grad()
    def inference(self, image: Image.Image) -> Image.Image:
        return self.cloud_adapter_forward(image)

    @torch.no_grad()
    def cloud_adapter_forward(self, image: Image.Image) -> Image.Image:
        """
        Cloud Adapter Inference
        """
        ori_size = image.size
        image = image.resize((self.img_size, self.img_size),
                     resample=Image.Resampling.BILINEAR)
        image = np.array(image)
        # print(image.shape)
        image = (image - np.min(image)) / (np.max(image)-np.min(image))

        image = torch.from_numpy(image).unsqueeze(0).to(self.device)
        image = image.permute(0, 3, 1, 2).float()

        outs = self.model.predict(image)
        pred_mask = outs[0].pred_sem_seg.data.cpu().numpy().astype(np.uint8)

        im = Image.fromarray(pred_mask[0]).convert("P")
        im.putpalette(self.palette)

        del image
        del outs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return im.resize(ori_size, resample=Image.Resampling.BILINEAR)


def get_palette(dataset_name: str) -> List[int]:
    if dataset_name in ["cloudsen12_high_l1c", "cloudsen12_high_l2a"]:
        return [79, 253, 199, 77, 2, 115, 251, 255, 41, 221, 53, 223]
    if dataset_name == "l8_biome":
        return [79, 253, 199, 221, 53, 223, 251, 255, 41, 77, 2, 115]
    if dataset_name in ["gf12ms_whu_gf1", "gf12ms_whu_gf2", "hrc_whu"]:
        return [79, 253, 199, 77, 2, 115]
    raise Exception("dataset_name not supported")


if __name__ == '__main__':
    title = 'Cloud Segmentation for Remote Sensing Images'
    custom_css = """
h1 {
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 20px;
}
"""
    hrc_whu_examples = glob("example_inputs/hrc_whu/*")
    gf1_examples = glob("example_inputs/gf1/*")
    gf2_examples = glob("example_inputs/gf2/*")
    l1c_examples = glob("example_inputs/l1c/*")
    l2a_examples = glob("example_inputs/l2a/*")
    l8_examples = glob("example_inputs/l8/*")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    with gr.Blocks(analytics_enabled=False, title=title,css=custom_css) as demo:
        gr.Markdown(f'# {title}')
        with gr.Tabs():
            with gr.TabItem('Google Earth'):
                CloudAdapterGradio(
                    config_path="cloud-adapter-configs/binary_classes_256x256.py",
                    checkpoint_path="checkpoints/cloud-adapter/hrc_whu_full_weight.pth",
                    device=device,
                    example_inputs=hrc_whu_examples,
                    num_classes=2,
                    palette=get_palette("hrc_whu"),
                )
            with gr.TabItem('Gaofen-1'):
                CloudAdapterGradio(
                    config_path="cloud-adapter-configs/binary_classes_256x256.py",
                    checkpoint_path="checkpoints/cloud-adapter/gf1_full_weight.pth",
                    device=device,
                    example_inputs=gf1_examples,
                    num_classes=2,
                    palette=get_palette("gf12ms_whu_gf1"),
                )
            with gr.TabItem('Gaofen-2'):
                CloudAdapterGradio(
                    config_path="cloud-adapter-configs/binary_classes_256x256.py",
                    checkpoint_path="checkpoints/cloud-adapter/gf2_full_weight.pth",
                    device=device,
                    example_inputs=gf2_examples,
                    num_classes=2,
                    palette=get_palette("gf12ms_whu_gf2"),
                )

            with gr.TabItem('Sentinel-2 (L1C)'):
                CloudAdapterGradio(
                    config_path="cloud-adapter-configs/multi_classes_512x512.py",
                    checkpoint_path="checkpoints/cloud-adapter/l1c_full_weight.pth",
                    device=device,
                    example_inputs=l1c_examples,
                    num_classes=4,
                    palette=get_palette("cloudsen12_high_l1c"),
                )
            with gr.TabItem('Sentinel-2 (L2A)'):
                CloudAdapterGradio(
                    config_path="cloud-adapter-configs/multi_classes_512x512.py",
                    checkpoint_path="checkpoints/cloud-adapter/l2a_full_weight.pth",
                    device=device,
                    example_inputs=l2a_examples,
                    num_classes=4,
                    palette=get_palette("cloudsen12_high_l2a"),
                )
            with gr.TabItem('Landsat-8'):
                CloudAdapterGradio(
                    config_path="cloud-adapter-configs/multi_classes_512x512.py",
                    checkpoint_path="checkpoints/cloud-adapter/l8_full_weight.pth",
                    device=device,
                    example_inputs=l8_examples,
                    num_classes=4,
                    palette=get_palette("l8_biome"),
                )

    demo.launch(share=True, debug=True)
