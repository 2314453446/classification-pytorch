import os
import shutil


def find_and_copy_same_files(source_folder, target_folders, destination_folders):
    # 确保目标文件夹和目的地文件夹数量相同
    if len(target_folders) != len(destination_folders):
        print("目标文件夹和目的地文件夹的数量不匹配")
        return

    # 遍历源文件夹中的每个文件
    for file_name in os.listdir(source_folder):
        source_file_path = os.path.join(source_folder, file_name)

        # 如果是文件，则在目标文件夹中查找同名文件并复制
        if os.path.isfile(source_file_path):
            for target_folder, destination_folder in zip(target_folders, destination_folders):
                target_file_path = os.path.join(target_folder, file_name)

                # 如果在目标文件夹中找到了同名文件
                if os.path.exists(target_file_path):
                    # 确保目的地文件夹存在
                    if not os.path.exists(destination_folder):
                        os.makedirs(destination_folder)

                    # 复制文件
                    destination_file_path = os.path.join(destination_folder, file_name)
                    shutil.copy2(target_file_path, destination_file_path)
                    print(f"文件 {file_name} 已从 {target_folder} 复制到 {destination_folder}")


# 从B C 文件夹中找到与A 文件夹下的同名文件 ，分别存储到D ，E 文件夹
source_folder = r'D:\Learning_software\classification-pytorch\figures\figures\stage123\label' # A
target_folders = [r'D:\Learning_software\datasets\classfication-pytorch\fake_figure\fack_vis_output',
                  r'D:\Learning_software\datasets\classfication-pytorch\vis_output']
destination_folders = [r'D:\Learning_software\classification-pytorch\figures\figures\stage123\fack_predict', # D
                       r'D:\Learning_software\classification-pytorch\figures\figures\stage123\predict'] #E
find_and_copy_same_files(source_folder, target_folders, destination_folders)
