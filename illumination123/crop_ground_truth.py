import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_mask_contours(labeled_img, threshold=(0, 0)):
    (threshold_min, threshold_max) = threshold

    # 创建掩码时明确指定数据类型
    mask = np.zeros_like(labeled_img, dtype=np.uint8)
    mask[(labeled_img >= threshold_min) & (labeled_img <= threshold_max)] = 255

    # 找到边界
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def crop_contours_area(img, contours):
    box_list = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        box_list.append(img[y:y + h, x:x + w])
    return box_list


def read_images_in_directory(directory_path):
    image_files = []

    # 检查目录是否存在
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return image_files

    # 遍历目录中的所有文件
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        # 检查文件是否为图像文件
        if os.path.isfile(file_path) and filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
            image_files.append(file_path)

    return image_files


def make_dataset(input_labedir, input_imgdir, outputdir):
    weed_save_path = outputdir + "/" + "weed"
    crop_save_path = outputdir + "/" + "crop"
    image_files = read_images_in_directory(input_imgdir)
    label_files = read_images_in_directory(input_labedir)
    if image_files:
        for image_file in image_files:
            image_name = os.path.basename(image_file)
            label_file_path = os.path.join(input_labedir, os.path.basename(image_file))
            labeled_image = cv2.imread(label_file_path, cv2.IMREAD_UNCHANGED)
            rgb_image = cv2.imread(image_file)
            # imshow(labeled_image)
            print(image_file)
            print(label_file_path)
            crop_contours = get_mask_contours(labeled_image, (1, 1))
            weed_contours = get_mask_contours(labeled_image, (2, 2))
            crop_box_list = crop_contours_area(rgb_image, crop_contours)
            weed_box_list = crop_contours_area(rgb_image, weed_contours)
            if crop_contours:
                os.makedirs(crop_save_path, exist_ok=True)
                for i, contour in enumerate(crop_contours):
                    x, y, w, h = cv2.boundingRect(contour)
                    crop_croped_img = rgb_image[y:y + h, x:x + w]
                    cv2.imwrite(os.path.join(crop_save_path, f'{image_name[:-4]}_crop{i}.png'), crop_croped_img)
            if weed_contours:
                os.makedirs(weed_save_path, exist_ok=True)
                for i, contour in enumerate(weed_contours):
                    x, y, w, h = cv2.boundingRect(contour)
                    weed_croped_img = rgb_image[y:y + h, x:x + w]
                    cv2.imwrite(os.path.join(weed_save_path, f'{image_name[:-4]}_weed{i}.png'), weed_croped_img)
                    # print()
            print("1")


input_labedir = r'D:\Learning_software\datasets\classfication-pytorch\illumination123\0605\labels'
input_imgdir = r'D:\Learning_software\datasets\classfication-pytorch\illumination123\0605\images'
outputdir = r'D:\Learning_software\datasets\classfication-pytorch\illumination123\0605\valid_0605'
make_dataset(input_labedir, input_imgdir, outputdir)