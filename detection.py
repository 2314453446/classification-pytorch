import cv2
from PIL import Image
import  getmask
from classification import Classification
import numpy as np
from sklearn.metrics import confusion_matrix
import imageio

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


def get_contour_gravity_point(contour,image):
    # Calculate the moment of contour
    M = cv2.moments(contour)

    # Calculate the centroid of the contour
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        print(f'Centroid of the contour is at: ({cx}, {cy})')

        # Draw centroid on the image (optional)
        cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)
    return image

# 获取预测结果存成深度图 作物值为1 ，杂草为2
def get_predict_result(overlay): #overlay为可视化的mask
    # 修改像素值
    overlay[np.all(overlay == [255, 0, 0], axis=-1)] = [0, 0, 2]
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


classfication = Classification()

image_path = "00115_1_1.jpg"
image = cv2.imread(image_path)
input_image = image.copy()
mask_image = getmask.get_mask(image_path)
contours,_ = cv2.findContours(mask_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

overlay = np.zeros_like(input_image,dtype=np.uint8)
class_colors = {
    "weed": (255, 0, 0),
    "crop": (0, 255, 0),
    # ... 其他类别
}

for contour in contours:
    x,y,w,h = cv2.boundingRect(contour)

    # 裁剪目标
    target = input_image[y:y + h, x:x + w]
    # target = cv2.resize(target,(224,224))
    target = Image.fromarray(cv2.cvtColor(target,cv2.COLOR_BGR2RGB))
    class_name = classfication.detect_image2(target)
    print(class_name)

    #画出质心点
    input_image = get_contour_gravity_point(contour,input_image)

    cv2.rectangle(input_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(input_image, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    #画出 mask分割图层
    # 选择类别对应的颜色
    color = class_colors.get(class_name, (255, 255, 255))  # 默认为白色
    # 在透明层上填充轮廓区域
    cv2.drawContours(overlay, [contour], -1, color, thickness=cv2.FILLED)

# 定义透明度（alpha），在 0 和 1 之间
alpha = 0.5
# 将透明层叠加到原始图像上
cv2.addWeighted(overlay, alpha, input_image, 1 - alpha, 0, input_image)
cv2.imwrite("output_img.jpg",input_image)

# 写入overlay 文件，并转换数据格式与标签对齐weed值为2，crop值为1
predict_path = "maskiou/predict/00115_1_1.jpg"
overlay_16bit=get_predict_result(overlay)
cv2.imwrite(predict_path, overlay_16bit)

#重写label文件，仅保存两个类别---------------------------------
label_image = "maskiou/label/00115_1_1.jpg"
relabel_img = label2relabel(label_image)
relabel_path = 'maskiou/relabel/00115_1_1.jpg'
Image.fromarray(relabel_img).save(relabel_path)

# 计算各类别匹配的重合率---------------------------------
# 读取十六位深度图像作为示例
y_true_path = 'maskiou/relabel/00115_1_1.jpg'
y_pred_path = 'maskiou/predict/00115_1_1.jpg'
y_true_image = Image.open(y_true_path)
y_pred_image = Image.open(y_pred_path)
y_true = np.array(y_true_image)
y_pred = np.array(y_pred_image)
# 确保设置了正确的类别数
n_classes = 3  # 替换为你的类别数量
# 计算每个类别的重合率
class_overlap = calculate_class_overlap(y_true, y_pred, n_classes)
print(class_overlap)

# imshow(overlay)
# 可视化输出结果
cv2.imwrite("output_img.jpg",input_image)