import pandas as pd

# 载入第一个文件（ground truth）
file_path_output = r'D:\Learning_software\classification-pytorch\figures\gravitypointv2\output.csv'
output_df = pd.read_csv(file_path_output)

# 载入第二个文件（用于比较）
file_path_box_point = r'D:\Learning_software\classification-pytorch\figures\gravitypointv2\box_point.csv'
box_point_df = pd.read_csv(file_path_box_point)

# 将两个数据框合并在一起，基于 'Image' 列
merged_df = pd.merge(output_df, box_point_df, on='Image', suffixes=('_ground_truth', '_comparison'))
# 获取merged_df的总行数
total_rows = merged_df.shape[0]

# 计算 X 和 Y 比率的绝对百分比误差
merged_df['X_Error'] = abs((merged_df['X_Ratio_ground_truth'] - merged_df['X_Ratio_comparison']) ) * 100
merged_df['Y_Error'] = abs((merged_df['Y_Ratio_ground_truth'] - merged_df['Y_Ratio_comparison']) ) * 100

# 计算平均误差率
average_x_error = merged_df['X_Error'].mean()
average_y_error = merged_df['Y_Error'].mean()

# 输出平均误差率
print("Average X Error Rate:", average_x_error)
print("Average Y Error Rate:", average_y_error)
