# 从文件夹A中获取图片名列表，从文件夹B中选择所有与文件夹A下同名的文件，并且移动到C文件夹下
import os
import shutil

# 定义文件夹路径
folder_a = r"D:\masterPROJECT\laser_weeding\weed_dataset\PhenoBench_classficationv3\row_image\test\images"
folder_b = r"D:\masterPROJECT\laser_weeding\weed_dataset\PhenoBench_classficationv3\row_image\val\semantics"
folder_c = r"D:\masterPROJECT\laser_weeding\weed_dataset\PhenoBench_classficationv3\row_image\test\semantics"

# 确保目标目录存在
if not os.path.exists(folder_c):
    os.makedirs(folder_c)

# 从文件夹A获取所有图片的文件名（不包括路径）
image_names = set([file for file in os.listdir(folder_a) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))])

# 在文件夹B中查找与文件夹A中同名的文件，并移动到文件夹C
for file in os.listdir(folder_b):
    if file in image_names:
        shutil.move(os.path.join(folder_b, file), folder_c)
