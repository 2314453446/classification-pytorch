# import cv2
# from cv2 import dnn_superres
# def main():
#     img_path = r"D:\Learning_software\opencv\PhoneBench_CV\05-15_00029_P0030852_weed0.png"
#     # 可选择算法，bilinear, bicubic, edsr, fsrcnn
#     # algorithm = "bilinear"
#     algorithm = "fsrcnn"
#     # 放大比例，可输入值2，3，4
#     scale = 4
#     # 模型路径
#     path = r"D:\Learning_software\opencv\PhoneBench_CV\FSRCNN_x4.pb"
#
#     # 载入图像
#     img = cv2.imread(img_path)
#     # 如果输入的图像为空
#     if img is None:
#         print("Couldn't load image: " + str(img_path))
#         return
#
#     # 创建模型
#     sr = dnn_superres.DnnSuperResImpl_create()
#
#     if algorithm == "bilinear":
#         img_new = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
#     elif algorithm == "bicubic":
#         img_new = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
#     elif algorithm == "edsr" or algorithm == "fsrcnn":
#         # 读取模型
#         sr.readModel(path)
#         #  设定算法和放大比例
#         sr.setModel(algorithm, scale)
#         # 放大图像
#         img_new = sr.upsample(img)
#     else:
#         print("Algorithm not recognized")
#
#     # 如果失败
#     if img_new is None:
#         print("Upsampling failed")
#
#     print("Upsampling succeeded. \n")
#
#     # 展示图片
#     cv2.namedWindow("Initial Image", cv2.WINDOW_AUTOSIZE)
#     # 初始化图片
#     cv2.imshow("Initial Image", img_new)
#     if img_new is not None:
#         # 保存图像到指定路径
#         output_path = r'D:\Learning_software\opencv\PhoneBench_CV/output_image.jpg'
#         cv2.imwrite(output_path, img_new)
#         print(f"Image saved to {output_path}")
#     else:
#         print("Failed to load the input image.")
#     cv2.waitKey(0)
#
#
# if __name__ == '__main__':
#     main()


# https://www.guyuehome.com/35416
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

            # 进行超分辨率处理
            if algorithm == "bilinear":
                img_new = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            elif algorithm == "bicubic":
                img_new = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            elif algorithm == "edsr" or algorithm == "fsrcnn":
                # 读取模型
                sr.readModel(model_path)
                #  设定算法和放大比例
                sr.setModel(algorithm, scale)
                # 放大图像
                img_new = sr.upsample(img)
            else:
                print("Algorithm not recognized")

            # 如果失败
            if img_new is None:
                print("Upsampling failed for " + img_path)
                continue

            # 保存图像到输出目录
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, img_new)
            print(f"Image saved to {output_path}")


if __name__ == '__main__':
    input_dir = r"D:\masterPROJECT\laser_weeding\weed_dataset\PhenoBench_classficationv2\with_small_objects\test\weed"
    output_dir = r'D:\masterPROJECT\laser_weeding\weed_dataset\PhenoBench_classficationv2\with_small_objects\test\SRweed'
    if not os.path.exists(output_dir):
        # 如果目录不存在，则创建它
        os.makedirs(output_dir)
    # 可选择算法，bilinear, bicubic, edsr, fsrcnn
    algorithm = "fsrcnn"
    # 放大比例，可输入值2，3，4
    scale = 4
    model_path = r"D:\Learning_software\classification-pytorch\data_process\FSRCNN_x4.pb"

    superresolve_images(input_dir, output_dir, algorithm, scale, model_path)
