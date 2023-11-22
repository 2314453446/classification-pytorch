#请用终端运行此代码， 内嵌控制台无法获取鼠标位置信息

import matplotlib.pyplot as plt
import os
from PIL import Image
import csv

# 路径设置
image_folder = r"D:\Learning_software\classification-pytorch\figures\gravitypoint\images"  # 将这里改为你的图片文件夹路径
output_file = r"D:\Learning_software\classification-pytorch\figures\gravitypoint\output.csv"

# 初始化一个变量来标记是否已经接收到点击事件
clicked = False

# 定义一个函数来处理鼠标点击事件
def onclick(event):
    global clicked
    # 获取图片尺寸
    img_width, img_height = img.size

    # 计算点击坐标的比例
    x_ratio = event.xdata / img_width
    y_ratio = event.ydata / img_height

    # 保存点击坐标的比例和当前图片名
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([current_image, x_ratio, y_ratio])

    # 设置已接收到点击事件
    clicked = True

    # 关闭当前的图片窗口，以便打开下一张图片
    plt.close()

# 准备CSV文件的标题
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Image", "X_Ratio", "Y_Ratio"])  # 写入标题行

# 读取文件夹中的所有图片文件
images = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

for image in images:
    current_image = image
    img_path = os.path.join(image_folder, image)
    img = Image.open(img_path)

    # 显示图片并等待点击事件
    fig, ax = plt.subplots()
    ax.imshow(img)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    # 等待直到接收到点击
    while not clicked:
        plt.pause(0.1)  # 短暂暂停，避免阻塞

    # 重置点击标志为下一张图片
    clicked = False
