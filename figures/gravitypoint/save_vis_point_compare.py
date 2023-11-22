import pandas as pd
from PIL import Image, ImageDraw
import os

def draw_points_on_image(image_path, points, output_folder):
    with Image.open(image_path) as img:
        draw = ImageDraw.Draw(img)
        for point, color in points:
            # Desired diameter for the points
            diameter = 20  # Change this value to increase or decrease the size of the points

            # Calculate the actual coordinates of the point
            x, y = point[0] * img.width, point[1] * img.height

            # Calculate the offset based on the desired diameter
            offset = diameter / 2

            # Draw the ellipse with the new diameter
            draw.ellipse([(x - offset, y - offset), (x + offset, y + offset)], fill=color)

        # 保存到新文件夹
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        img.save(output_path)

# CSV文件路径、图片文件夹路径和输出文件夹路径
csv_files = [r'D:\Learning_software\classification-pytorch\figures\gravitypoint\output.csv',
             r'D:\Learning_software\classification-pytorch\figures\gravitypoint\box_point.csv',
             r'D:\Learning_software\classification-pytorch\figures\gravitypoint\predict_output.csv']
images_folder = r'D:\Learning_software\classification-pytorch\figures\gravitypoint\images'
output_folder = r'D:\Learning_software\classification-pytorch\figures\gravitypoint\images_GT_box_pre'
colors = ['red', 'yellow', 'blue']  # 为每个CSV文件指定不同的颜色

# 收集所有图像上的点
points_per_image = {}  # key: image_path, value: list of (point, color) tuples

for csv_file, color in zip(csv_files, colors):
    df = pd.read_csv(csv_file)
    for index, row in df.iterrows():
        image_path = os.path.join(images_folder, row['Image'])
        point = (row['X_Ratio'], row['Y_Ratio'])
        if image_path not in points_per_image:
            points_per_image[image_path] = []
        points_per_image[image_path].append((point, color))

# 绘制点并保存图像
for image_path, points in points_per_image.items():
    draw_points_on_image(image_path, points, output_folder)
