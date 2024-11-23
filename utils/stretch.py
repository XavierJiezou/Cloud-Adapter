# -*- encoding: utf-8 -*-
'''
@File    :   stretch.py
@Time    :   2024/08/11 01:35:36
@Author  :   XavierJiezou 
@Version :   1.0
@Contact :   xuechaozou@foxmail.com
@Citation:   https://www.nv5geospatialsoftware.com/docs/BackgroundStretchTypes.html
'''

import numpy as np
import matplotlib.pyplot as plt


def linear_stretch(image):
    min_val = np.min(image)
    max_val = np.max(image)
    stretched = (image - min_val) * 255.0 / (max_val - min_val)
    return np.clip(stretched, 0, 255).astype(np.uint8)

def linear_percent_stretch(image, percent=2):
    low, high = np.percentile(image, (percent, 100 - percent))
    stretched = (image - low) * 255.0 / (high - low)
    return np.clip(stretched, 0, 255).astype(np.uint8)

def equalization_stretch(image):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]
    stretched = np.interp(image.flatten(), bins[:-1], cdf_normalized)
    return stretched.reshape(image.shape).astype(np.uint8)

def gaussian_stretch(image):
    mean = np.mean(image)
    std_dev = np.std(image)
    stretched = 127 + (image - mean) * (128.0 / (3 * std_dev + 1e-6)) # 1e-6 to avoid division by zero
    return np.clip(stretched, 0, 255).astype(np.uint8)

def square_root_stretch(image):
    stretched = np.sqrt(image) * np.sqrt(255.0 / np.max(image))
    return np.clip(stretched, 0, 255).astype(np.uint8)

def logarithmic_stretch(image):
    stretched = np.log1p(image) * (255.0 / np.log1p(np.max(image)))
    return np.clip(stretched, 0, 255).astype(np.uint8)

def optimized_linear_stretch(image, min_percent=0.025, max_percent=0.99, min_adjust_percent=0.1, max_adjust_percent=0.5):
    cdf, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = cdf.cumsum()
    cdf_normalized = cdf / cdf[-1]
    
    min_value = np.searchsorted(cdf_normalized, min_percent)
    max_value = np.searchsorted(cdf_normalized, max_percent)
    
    a = bins[min_value]
    b = bins[max_value]
    
    c = a - min_adjust_percent * (b - a)
    d = b + max_adjust_percent * (b - a)
    
    stretched = (image - c) * 255.0 / (d - c)
    return np.clip(stretched, 0, 255).astype(np.uint8)

def main(image_path):
    # 加载一个示例图像
    import tifffile as tiff
    image = tiff.imread(image_path)
    # image = (image - image.min()) / (image.max() - image.min())
    image = (image/65535.* 255.).astype(np.uint8)
    
    # 应用不同的拉伸算法
    linear_image = linear_stretch(image)
    percent_image = linear_percent_stretch(image)
    equalized_image = equalization_stretch(image)
    gaussian_image = gaussian_stretch(image)
    sqrt_image = square_root_stretch(image)
    log_image = logarithmic_stretch(image)
    optimized_image = optimized_linear_stretch(image)
    
    # 展示结果
    fig, axs = plt.subplots(2, 4, figsize=(12, 6))
    axs[0, 0].imshow(image, cmap='gray')
    axs[0, 0].set_title("Original Image")
    axs[0, 1].imshow(linear_image, cmap='gray')
    axs[0, 1].set_title("Linear Stretch")
    axs[0, 2].imshow(percent_image, cmap='gray')
    axs[0, 2].set_title("Linear Percent Stretch")
    axs[0, 3].imshow(equalized_image, cmap='gray')
    axs[0, 3].set_title("Equalization Stretch")
    axs[1, 0].imshow(gaussian_image, cmap='gray')
    axs[1, 0].set_title("Gaussian Stretch")
    axs[1, 1].imshow(sqrt_image, cmap='gray')
    axs[1, 1].set_title("Square Root Stretch")
    axs[1, 2].imshow(log_image, cmap='gray')
    axs[1, 2].set_title("Logarithmic Stretch")
    axs[1, 3].imshow(optimized_image, cmap='gray')
    axs[1, 3].set_title("Optimized Linear Stretch")
    
    for ax in axs.flat:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("images/s2_rgb_int16_stretch.jpg", bbox_inches="tight", pad_inches=0)
    plt.show()

if __name__ == "__main__":
    main("images/s2_rgb_int16.tif")
