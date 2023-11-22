from PIL import Image, ImageDraw
import numpy as np
import os
import csv
import getmask

def calculate_mask_centroid(img_array, mask_value):
    # 找到mask值对应的像素
    mask_pixels = np.where(img_array == mask_value)

    # 计算重心
    if len(mask_pixels[0]) == 0 or len(mask_pixels[1]) == 0:
        return None  # 如果没有找到mask值，返回None
    centroid_y = np.mean(mask_pixels[0]) / img_array.shape[0]  # y坐标的比例
    centroid_x = np.mean(mask_pixels[1]) / img_array.shape[1]  # x坐标的比例

    return centroid_x, centroid_y

def overlay_mask_on_image(original_img, mask_img_array, mask_opacity):
    # 将mask转换为透明度蒙版
    mask = Image.fromarray(mask_img_array).convert("L")
    mask = mask.point(lambda x: x * mask_opacity)

    # 将原始图像转换为RGBA
    original_img_rgba = original_img.convert("RGBA")

    # 将蒙版转换为RGBA
    mask_rgba = Image.new("RGBA", original_img_rgba.size)
    mask_rgba.putalpha(mask)

    # 合成图像
    combined_img = Image.alpha_composite(original_img_rgba, mask_rgba)
    return combined_img

def process_folder(folder_path, mask_value, output_csv, output_folder, mask_opacity):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image File', 'Centroid X Ratio', 'Centroid Y Ratio'])

        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, file_name)

                # 获取掩码图像并计算重心
                mask_img_array = getmask.get_single_predict(image_path)
                centroid_ratio = calculate_mask_centroid(mask_img_array, mask_value)

                if centroid_ratio:
                    writer.writerow([file_name, centroid_ratio[0], centroid_ratio[1]])

                    # 打开原始图像
                    original_img = Image.open(image_path)

                    # 在原始图像上叠加掩码
                    img_with_mask = overlay_mask_on_image(original_img, mask_img_array, mask_opacity)

                    # 在重心位置绘制点
                    draw = ImageDraw.Draw(img_with_mask)
                    # New desired radius for the points
                    radius = 10  # Change this value to increase or decrease the size of the points

                    # Calculate the new bounding box with the updated radius
                    x = centroid_ratio[0] * original_img.width
                    y = centroid_ratio[1] * original_img.height
                    draw.ellipse([(x - radius, y - radius), (x + radius, y + radius)], fill='blue', outline='blue')

                    # 保存修改后的图像
                    output_image_path = os.path.join(output_folder, file_name)
                    img_with_mask.save(output_image_path)

# 文件夹路径、Mask值、输出CSV文件的路径和输出图片的文件夹路径
folder_path = r'D:\Learning_software\classification-pytorch\figures\gravitypoint\images'
mask_value = 255  # Mask的值
output_csv = r'D:\Learning_software\classification-pytorch\figures\gravitypoint\predict_output.csv'
output_folder = r'D:\Learning_software\classification-pytorch\figures\gravitypoint\images_predict'  # 新的输出文件夹
mask_opacity = 0.5  # Mask透明度（0到1之间）

# 处理文件夹并保存到CSV和新文件夹
process_folder(folder_path, mask_value, output_csv, output_folder, mask_opacity)
