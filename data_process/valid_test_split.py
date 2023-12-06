import os
import random
import shutil

# 示例目录路径
source_directory = r"D:\masterPROJECT\laser_weeding\weed_dataset\PhenoBench_classficationv3\row_image\val\images"
destination_directory = r"D:\masterPROJECT\laser_weeding\weed_dataset\PhenoBench_classficationv3\row_image\test\images"

# 确保目标目录存在
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# 获取源目录下的所有图片文件
all_images = [file for file in os.listdir(source_directory) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

# 随机选择一半的图片
selected_images = random.sample(all_images, len(all_images) // 2)

# 将选中的图片移动到目标目录
for image in selected_images:
    shutil.move(os.path.join(source_directory, image), destination_directory)

selected_images # 显示被选中的图片列表
