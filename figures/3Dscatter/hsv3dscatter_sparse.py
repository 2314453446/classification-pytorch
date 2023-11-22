import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random

import getmask

# 读取Nemo图像
image_path = r'D:\Learning_software\classification-pytorch\05-26_00095_P0034280.png'
image = cv2.imread(image_path)

def hsv3dscatter_with_reduced_density(image):
    # 将图像转换为HSV颜色空间
    nemo_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 获取图像的尺寸
    height, width, _ = image.shape

    # 创建3D散点图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 提取HSV通道数据
    hue = nemo_hsv[:, :, 0]
    saturation = nemo_hsv[:, :, 1]
    value = nemo_hsv[:, :, 2]

    # 减小点的密度：随机采样一部分像素
    sample_ratio = 0.01  # 采样比例，可以根据需要调整
    total_pixels = height * width
    num_samples = int(total_pixels * sample_ratio)
    sample_indices = random.sample(range(total_pixels), num_samples)

    # 从采样的像素中获取数据
    hue_samples = hue.reshape(-1)[sample_indices]
    saturation_samples = saturation.reshape(-1)[sample_indices]
    value_samples = value.reshape(-1)[sample_indices]

    # 创建HSV散点图
    ax.scatter(hue_samples, saturation_samples, value_samples, c=nemo_hsv.reshape(-1, 3) / [179.0, 255.0, 255.0], marker='o', s=0.1)

    ax.set_xlabel('Hue (H)')
    ax.set_ylabel('Saturation (S)')
    ax.set_zlabel('Value (V)')

    plt.show()

def rgb3dscatter_with_reduced_density(image, image_path):
    mask = getmask.get_mask(image_path)
    masked_image = image.copy()
    masked_image[mask == 0] = [0, 0, 0]

    # 获取图像的尺寸
    height, width, _ = masked_image.shape

    # 创建3D散点图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 提取RGB通道数据
    r = masked_image[:, :, 2]
    g = masked_image[:, :, 1]
    b = masked_image[:, :, 0]

    # 减小点的密度：随机采样一部分像素
    sample_ratio = 0.01  # 采样比例，可以根据需要调整
    total_pixels = height * width
    num_samples = int(total_pixels * sample_ratio)
    sample_indices = random.sample(range(total_pixels), num_samples)

    # 从采样的像素中获取数据
    r_samples = r.reshape(-1)[sample_indices]
    g_samples = g.reshape(-1)[sample_indices]
    b_samples = b.reshape(-1)[sample_indices]

    # 创建RGB 3D散点图
    ax.scatter(r_samples, g_samples, b_samples, c='red', marker='o', s=0.1)

    ax.set_xlabel('Red (R)')
    ax.set_ylabel('Green (G)')
    ax.set_zlabel('Blue (B)')

    plt.show()

if __name__ == "__main__":
    # 调用函数以显示HSV 3D散点图，具有减小的点密度
    hsv3dscatter_with_reduced_density(image)
    # 调用函数以显示RGB 3D散点图，具有减小的点密度
    rgb3dscatter_with_reduced_density(image, image_path)
