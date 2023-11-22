import cv2
from PIL import Image
import getmask
from classification import Classification
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import imageio

# Initialize the Classification
classification = Classification()

def imshow(showimage, title='title'):
    cv2.imshow(title, showimage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 原始lable 有1234四个类别，修改后仅保存1 2两个类别，分别代表crop和weed
def label2relabel(image_path):
    image = Image.open(image_path)
    image_np = np.array(image)

    # 修改像素值
    image_np[np.isin(image_np, [1, 3])] = 1
    image_np[np.isin(image_np, [2, 4])] = 2
    image = image_np.astype(np.uint16)
    return image

# 重写label,自动获取目录下的所有文件，修改成指定文件后，保存到新目录
def process_and_save_images(source_folder, destination_folder):
    # 确保目标文件夹存在，如果不存在则创建
    os.makedirs(destination_folder, exist_ok=True)

    # 遍历源文件夹中的所有文件
    for filename in os.listdir(source_folder):
        if filename.endswith('.png'):  # 确保处理PNG文件
            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(destination_folder, filename)

            # 应用label2relabel函数
            relabel_img = label2relabel(source_path)

            # 保存重写后的图像
            Image.fromarray(relabel_img).save(destination_path)
            print(f"Processed and saved: {destination_path}")

def get_contour_gravity_point(contour,image):
    # Calculate the moment of contour
    M = cv2.moments(contour)

    # Calculate the centroid of the contour
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        print(f'Centroid of the contour is at: ({cx}, {cy})')

        # Draw centroid on the image (optional)
        cv2.circle(image, (cx, cy), 10, (255, 0, 0), -1)
    return image

# 获取预测结果存成深度图 作物值为1 ，杂草为2
def get_predict_result(overlay): #overlay为可视化的mask
    # 修改像素值
    overlay[np.all(overlay == [0, 0, 255], axis=-1)] = [0, 0, 2]
    overlay[np.all(overlay == [0, 255, 0], axis=-1)] = [0, 0, 1]

    # 提取蓝色通道作为单通道图像
    single_channel_image = overlay[:, :, 2]

    # 转换为十六位深度
    image_16bit = np.uint16(single_channel_image)
    return image_16bit
    # 保存图像
    # output_path = 'your_output_path.png'  # 指定输出路径
    # cv2.imwrite(output_path, image_16bit)

# 计算各个类别重合率
def calculate_class_overlap(y_true, y_pred, n_classes):
    # 将掩码展平为一维数组
    y_true_flattened = y_true.flatten()
    y_pred_flattened = y_pred.flatten()

    # 计算混淆矩阵
    cm = confusion_matrix(y_true_flattened, y_pred_flattened, labels=np.arange(n_classes))

    # 初始化一个字典来存储每个类别的重合率
    overlap_per_class = {}

    # 遍历每个类别
    for i in range(n_classes):
        # 计算每个类别的重合率
        class_overlap = cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0
        overlap_per_class[i] = class_overlap

    return overlap_per_class

# 加载两个文件夹下的同名文件，并计算重合率，
def load_images_and_calculate_overlap(folder_true, folder_pred, n_classes, output_file):
    # 获取两个文件夹中的文件名
    true_image_files = set(os.listdir(folder_true))
    pred_image_files = set(os.listdir(folder_pred))

    # 找出两个文件夹中都存在的同名文件
    common_files = true_image_files.intersection(pred_image_files)

    # 对每个共有的文件
    for file_name in common_files:
        # 完整的文件路径
        y_true_path = os.path.join(folder_true, file_name)
        y_pred_path = os.path.join(folder_pred, file_name)

        # 读取图像
        y_true_image = Image.open(y_true_path)
        y_pred_image = Image.open(y_pred_path)

        # 转换为NumPy数组
        y_true = np.array(y_true_image)
        y_pred = np.array(y_pred_image)

        # 计算重合率
        class_overlap = calculate_class_overlap(y_true, y_pred, n_classes)

        # 以追加模式打开文件并写入结果
        with open(output_file, 'a') as file:
            file.write(f"{file_name}: {class_overlap}\n")

        print(f"Overlap for {file_name}: {class_overlap}")


if __name__=="__main__":

    # Define the input and output directories
    input_directory = r"D:\Learning_software\datasets\classfication-pytorch\val\images"  # 输入图片目录
    vis_output_directory = r"D:\Learning_software\datasets\classfication-pytorch\fake_figure\fack_vis_output"  # 可视化输出目录
    predict2label_dir =r"D:\Learning_software\datasets\classfication-pytorch\fake_figure\fack_predict2label" #预测结果转换成2类别png输出目录

    # 加载label文件夹下的所有文件，relabel后仅保存两个类别
    source_folder = r'D:\Learning_software\datasets\classfication-pytorch\val\semantics'  # 源label文件夹路径
    destination_folder = r'D:\Learning_software\datasets\classfication-pytorch\label2relabel'  # 目标文件夹路重新2类别划分输出目录

    # 计算mask iou路径
    folder_true = 'D:\Learning_software\datasets\classfication-pytorch\label2relabel'  # 真实标签图像文件夹路径
    folder_pred = r'D:\Learning_software\datasets\classfication-pytorch\fake_figure\fack_predict2label'  # 预测标签图像文件夹路径
    output_overlap_file = r'D:\Learning_software\datasets\classfication-pytorch\fake_figure\overlap_resutl.txt'
    n_classes =3

    class_colors = {
        "weed": (0, 0, 255),
        "crop": (0, 255, 0),
        # ... 其他类别
    }
    # Check if output directory exists, if not, create it
    if not os.path.exists(vis_output_directory):
        os.makedirs(vis_output_directory)



    # List all files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith(".png"):  # assuming you are reading .png files
            image_path = os.path.join(input_directory, filename)
            image = cv2.imread(image_path)
            input_image = image.copy()
            mask_image = getmask.get_mask(image_path)
            contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            overlay = np.zeros_like(input_image, dtype=np.uint8)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # Crop the target
                target = input_image[y:y + h, x:x + w]
                # Uncomment below if resizing is needed
                # target = cv2.resize(target,(224,224))
                target = Image.fromarray(cv2.cvtColor(target, cv2.COLOR_BGR2RGB))
                class_name = classification.detect_image2(target)
                print(class_name)

                # 画图，画出mask小于面积小于阈值的轮廓，如果轮廓面积小于30且类别为 crop,将它改为 weed类别
                #
                if class_name=="crop":
                    contour_area = cv2.contourArea(contour)
                    if contour_area<200:
                        class_name = "weed"
                if class_name=="weed":
                    contour_area = cv2.contourArea(contour)
                    if contour_area<100:
                        class_name = "crop"

                # 画出质心点
                if class_name == "weed":
                    input_image = get_contour_gravity_point(contour, input_image)

                cv2.rectangle(input_image, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(input_image, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                # 画出 mask分割图层
                # 选择类别对应的颜色
                color = class_colors.get(class_name, (255, 255, 255))  # 默认为白色
                # 在透明层上填充轮廓区域
                cv2.drawContours(overlay, [contour], -1, color, thickness=cv2.FILLED)

            # 定义透明度（alpha），在 0 和 1 之间
            alpha = 0.5
            # 将透明层叠加到原始图像上
            cv2.addWeighted(overlay, alpha, input_image, 1 - alpha, 0, input_image)
            # 写入overlay 文件，并转换数据格式与标签对齐weed值为2，crop值为1
            predict_path = os.path.join(predict2label_dir, filename)
            overlay_16bit = get_predict_result(overlay)
            cv2.imwrite(predict_path, overlay_16bit)

            # Save the processed image to the output directory
            vis_output_path = os.path.join(vis_output_directory, filename)
            cv2.imwrite(vis_output_path, input_image)

    # 重新label 文件 仅保留 crop:1 weed:2 两个类别，0代表背景
    process_and_save_images(source_folder, destination_folder)

    # 计算重合率
    load_images_and_calculate_overlap(folder_true, folder_pred, n_classes, output_overlap_file)