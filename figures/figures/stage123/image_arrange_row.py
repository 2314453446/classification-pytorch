import os
from PIL import Image
import matplotlib.pyplot as plt

# 设置文件夹路径
folder1 = r'D:\Learning_software\classification-pytorch\figures\figures\stage123\label'
folder2 = r'D:\Learning_software\classification-pytorch\figures\figures\stage123\fack_predict'
folder3 = r'D:\Learning_software\classification-pytorch\figures\figures\stage123\predict'



# 索引值列表，索引从0开始
indices = [2,3]  # 示例索引值，可以根据需要更改

# 图片间的间隔（像素）
spacing = 10

# 获取指定索引处的文件名
def get_image_path(folder, index):
    try:
        return os.path.join(folder, sorted(os.listdir(folder))[index])
    except IndexError:
        print(f"在文件夹 {folder} 中没有找到索引为 {index} 的图片")
        return None

# 存储每行拼接后的图片
row_images = []

# 为每个索引创建一行图片
for index in indices:
    images = []

    # 对于每个文件夹，找到对应的图片
    for folder in [folder1, folder2, folder3]:
        image_path = get_image_path(folder, index)
        if image_path and os.path.exists(image_path):
            images.append(Image.open(image_path))

    # 检查是否找到了所有图片
    if len(images) == 3:
        # 计算行图像的总宽度和最大高度
        total_width = sum(image.width for image in images) + (len(images) - 1) * spacing
        max_height = max(image.height for image in images)

        # 创建一行的新图像
        row_image = Image.new('RGB', (total_width, max_height), (255, 255, 255))

        # 水平拼接图片
        x_offset = 0
        for image in images:
            row_image.paste(image, (x_offset, 0))
            x_offset += image.width + spacing

        # 添加到行图片列表
        row_images.append(row_image)

# 计算最终图像的总高度
total_height = sum(image.height for image in row_images) + (len(row_images) - 1) * spacing

# 创建最终图像
final_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))

# 垂直拼接行图片
y_offset = 0
for image in row_images:
    final_image.paste(image, (0, y_offset))
    y_offset += image.height + spacing

# 保存最终图像
final_save_path = 'combined_2_3_stage1.jpg'
final_image.save(final_save_path)
print(f"图片已保存到 {final_save_path}")