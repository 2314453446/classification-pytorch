import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw
import csv

# 路径设置
image_folder = r"D:\Learning_software\classification-pytorch\figures\gravitypointv2\images"  # 原始图片文件夹路径
output_folder = r"D:\Learning_software\classification-pytorch\figures\gravitypointv2\GT_vis"  # 修改后图片保存路径
output_file = r"D:\Learning_software\classification-pytorch\figures\gravitypointv2\output.csv"

# 如果输出文件夹不存在，则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 定义一个函数来处理鼠标点击事件
def onclick(event):
    global img, clicked

    # 计算点击坐标的比例
    x_ratio = event.xdata / img.width
    y_ratio = event.ydata / img.height

    # 将点绘制到图片上
    draw = ImageDraw.Draw(img)
    offset = 10  # Increase the offset for a larger point
    draw.ellipse([(event.xdata - offset, event.ydata - offset), (event.xdata + offset, event.ydata + offset)],
                 fill='red', outline='red')

    # 保存修改后的图片
    modified_image_path = os.path.join(output_folder, current_image)
    img.save(modified_image_path)

    # 保存点击坐标的比例和当前图片名
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([current_image, x_ratio, y_ratio])

    # 设置已接收到点击事件
    clicked = True

    # 关闭当前的图片窗口
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

    # 初始化点击事件检测变量
    clicked = False

    # 显示图片并等待点击事件
    fig, ax = plt.subplots()
    ax.imshow(img)
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    # 等待直到接收到点击
    while not clicked:
        plt.pause(0.1)  # 短暂暂停，避免阻塞
