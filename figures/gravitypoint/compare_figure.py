import pandas as pd
import matplotlib.pyplot as plt

# 读取三个CSV文件
df1 = pd.read_csv(r'D:\Learning_software\classification-pytorch\figures\gravitypoint\output.csv')  # 替换为第一个CSV文件的实际路径
df2 = pd.read_csv(r'D:\Learning_software\classification-pytorch\figures\gravitypoint\box_point.csv')  # 替换为第二个CSV文件的实际路径
df3 = pd.read_csv(r'D:\Learning_software\classification-pytorch\figures\gravitypoint\predict_output.csv')  # 替换为第三个CSV文件的实际路径

# 绘制X坐标比例的对比图
plt.figure(figsize=(12, 6))
plt.plot(df1['X_Ratio'], label='Ground_Truth')
plt.plot(df2['X_Ratio'], label='Box_Center_Point')
plt.plot(df3['X_Ratio'], label='Mask_Gravity_Point')
plt.xlabel('Sample Index')
plt.ylabel('X Coordinate Ratio')
plt.title('Comparison of Fitting Accuracy for X-Coordinate Center Points')
plt.legend()
plt.show()

# 绘制Y坐标比例的对比图
plt.figure(figsize=(12, 6))
plt.plot(df1['Y_Ratio'], label='Ground_Truth')
plt.plot(df2['Y_Ratio'], label='Box_Center_Point')
plt.plot(df3['Y_Ratio'], label='Mask_Gravity_Point')
plt.xlabel('Sample Index')
plt.ylabel('Y Coordinate Ratio')
plt.title('Comparison of Fitting Accuracy for Y-Coordinate Center Points')
plt.legend()
plt.show()
