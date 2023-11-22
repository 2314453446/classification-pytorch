import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
import getmask

# 读取Nemo图像
image_path = r'D:\Learning_software\classification-pytorch\05-26_00095_P0034280.png'
image = cv2.imread(image_path)

def org_hsv3dscatter(image):

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

    # 创建HSV散点图
    ax.scatter(hue.reshape(-1), saturation.reshape(-1), value.reshape(-1), c=nemo_hsv.reshape(-1, 3) / [179.0, 255.0, 255.0], marker='o',s=1)

    ax.set_xlabel('Hue (H)')
    ax.set_ylabel('Saturation (S)')
    ax.set_zlabel('Value (V)')

    plt.show()

def filtered_hsv3dcatter(image,image_path):
    mask = getmask.get_mask(image_path)
    masked_image = image.copy()
    masked_image[mask == 0] = [0,0,0]

    # 将新图像转换为HSV颜色空间
    masked_hsv = cv2.cvtColor(masked_image, cv2.COLOR_BGR2HSV)

    # 获取图像的尺寸
    height, width, _ = masked_image.shape

    # 创建3D散点图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 提取HSV通道数据
    hue = masked_hsv[:, :, 0]
    saturation = masked_hsv[:, :, 1]
    value = masked_hsv[:, :, 2]

    # 创建HSV 3D散点图
    colors = np.stack((hue.reshape(-1) / 179.0, saturation.reshape(-1) / 255.0, value.reshape(-1) / 255.0), axis=-1)
    ax.scatter(hue.reshape(-1), saturation.reshape(-1), value.reshape(-1), c=colors, marker='o', s=1)

    ax.set_xlabel('Hue (H)')
    ax.set_ylabel('Saturation (S)')
    ax.set_zlabel('Value (V)')

    plt.show()
    cv2.imwrite('./masked_image.png', masked_image)


def org_rgb3dscatter(image):
    # 获取图像的尺寸
    height, width, _ = image.shape

    # 创建3D散点图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 提取RGB通道数据
    r = image[:, :, 2]
    g = image[:, :, 1]
    b = image[:, :, 0]

    # 创建原始RGB 3D散点图
    colors = np.stack((r.reshape(-1) / 255.0, g.reshape(-1) / 255.0, b.reshape(-1) / 255.0), axis=-1)
    ax.scatter(r.reshape(-1), g.reshape(-1), b.reshape(-1), c=colors, marker='o',s=0.01)

    ax.set_xlabel('Red (R)')
    ax.set_ylabel('Green (G)')
    ax.set_zlabel('Blue (B)')

    plt.show()


def rgb3dscatter(image, image_path):
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

    # 创建RGB 3D散点图
    colors = np.stack((r.reshape(-1) / 255.0, g.reshape(-1) / 255.0, b.reshape(-1) / 255.0), axis=-1)
    ax.scatter(r.reshape(-1), g.reshape(-1), b.reshape(-1), c=colors, marker='o', s=0.01)

    ax.set_xlabel('Red (R)')
    ax.set_ylabel('Green (G)')
    ax.set_zlabel('Blue (B)')

    plt.show()
    cv2.imwrite('./masked_image.png', masked_image)



if __name__=="__main__":
    # 调用函数以显示原始hsv 3D散点图
    org_hsv3dscatter(image)
    # 调用函数以显示hsv 3D散点图
    filtered_hsv3dcatter(image, image_path)
    # 调用函数以显示原始RGB 3D散点图
    org_rgb3dscatter(image)
    # 调用函数以显示RGB 3D散点图
    rgb3dscatter(image, image_path)