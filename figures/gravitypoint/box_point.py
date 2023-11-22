from PIL import Image, ImageDraw
import os
import csv

def draw_box_and_center(img):
    draw = ImageDraw.Draw(img)

    # 画边界框
    bbox = img.getbbox()
    draw.rectangle(bbox, outline='yellow', width=5)

    point_size = 10  # The radius of the point, adjust this to change the size

    # Calculate and draw the center point with the new size
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    draw.ellipse([(center_x - point_size, center_y - point_size),
                  (center_x + point_size, center_y + point_size)],
                 fill='yellow', outline='yellow')

    return img, (center_x / img.width, center_y / img.height)

def process_folder(folder_path, output_csv, output_images_folder):
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    if not os.path.exists(output_images_folder):
        os.makedirs(output_images_folder)

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image File', 'Center X Ratio', 'Center Y Ratio'])

        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, file_name)
                img = Image.open(image_path)

                # 画边界框和中心点
                processed_img, center_ratio = draw_box_and_center(img)

                # 保存修改后的图像
                processed_img.save(os.path.join(output_images_folder, file_name))

                # 写入CSV文件
                writer.writerow([file_name, center_ratio[0], center_ratio[1]])

# 文件夹路径、输出CSV文件的路径和输出图片的文件夹路径
folder_path = r'D:\Learning_software\classification-pytorch\figures\gravitypoint\images'
output_csv = r'D:\Learning_software\classification-pytorch\figures\gravitypoint\box_point.csv'
output_images_folder = r'D:\Learning_software\classification-pytorch\figures\gravitypoint\images_box'  # 新的输出图片文件夹路径

# 处理文件夹并保存到CSV和新文件夹
process_folder(folder_path, output_csv, output_images_folder)
