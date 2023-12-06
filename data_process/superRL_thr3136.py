import cv2
from cv2 import dnn_superres
import os

def superresolve_images(input_dir, output_dir, algorithm, scale, model_path):
    # 创建超分辨率模型
    sr = dnn_superres.DnnSuperResImpl_create()

    # 读取模型
    sr.readModel(model_path)
    # 设定算法和放大比例
    sr.setModel(algorithm, scale)

    # 遍历指定目录下的所有图像文件
    for filename in os.listdir(input_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(input_dir, filename)

            # 载入图像
            img = cv2.imread(img_path)
            if img is None:
                print("Couldn't load image: " + img_path)
                continue

            # 计算图像像素总数
            img_pixels = img.shape[0] * img.shape[1]

            # 根据像素数量决定是否进行超分辨率处理
            if img_pixels < 3136:
                if algorithm == "bilinear":
                    img_new = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                elif algorithm == "bicubic":
                    img_new = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                elif algorithm == "edsr" or algorithm == "fsrcnn":
                    # 放大图像
                    img_new = sr.upsample(img)
                else:
                    print("Algorithm not recognized")
                    continue
            else:
                img_new = img

            # 如果放大失败
            if img_new is None:
                print("Processing failed for " + img_path)
                continue

            # 保存图像到输出目录
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, img_new)
            print(f"Image saved to {output_path}")

if __name__ == '__main__':
    input_dir = r"D:\masterPROJECT\laser_weeding\weed_dataset\PhenoBench_classficationv2\PhenoBench_classfication_Partial_SR\test\crop"
    output_dir = r"D:\masterPROJECT\laser_weeding\weed_dataset\PhenoBench_classficationv2\PhenoBench_classfication_Partial_SR\test\PSRcrop"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    algorithm = "fsrcnn"
    scale = 4
    model_path = r"D:\Learning_software\classification-pytorch\data_process\FSRCNN_x4.pb"

    superresolve_images(input_dir, output_dir, algorithm, scale, model_path)
