import os
import shutil
import random
import math

def calculate_size_ranges(folder, num_ranges):
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    file_sizes = [os.path.getsize(os.path.join(folder, f)) for f in files]
    
    min_size = min(file_sizes)
    max_size = max(file_sizes)
    step = (max_size - min_size) / num_ranges

    size_ranges = [(min_size + i * step, min_size + (i + 1) * step) for i in range(num_ranges)]
    size_ranges[-1] = (size_ranges[-1][0], float('inf'))  # Adjust the upper bound of the last range to infinity

    return size_ranges

def distribute_files_by_size(original_folder, train_folder, validation_folder, test_folder, num_ranges, train_percentage, validation_percentage, test_percentage):
    size_ranges = calculate_size_ranges(original_folder, num_ranges)

    # Get the list of files
    files = [f for f in os.listdir(original_folder) if os.path.isfile(os.path.join(original_folder, f))]
    random.shuffle(files)

    # Calculate counts for each set
    total_count = len(files)
    train_count = math.ceil(total_count * train_percentage)
    validation_count = math.ceil(total_count * validation_percentage)
    test_count = total_count - train_count - validation_count

    # Function to move files to the specified folder
    def move_files_to_folder(file_list, dest_folder):
        for file in file_list:
            file_path = os.path.join(original_folder, file)
            try:
                shutil.move(file_path, os.path.join(dest_folder, file))
            except FileNotFoundError as e:
                print(f"Error moving file: {e}")
                print(f"File path: {file_path}")
                print(f"Destination folder: {os.path.join(dest_folder, file)}")

    # Move files to train folder
    move_files_to_folder(files[:train_count], train_folder)
    # Move files to validation folder
    move_files_to_folder(files[train_count:train_count + validation_count], validation_folder)
    # Move files to test folder
    move_files_to_folder(files[train_count + validation_count:], test_folder)

original_folder = r"D:\masterPROJECT\laser_weeding\weed_dataset\PhenoBench_classficationv2\test_folder\row\weed"
train_folder = r"D:\masterPROJECT\laser_weeding\weed_dataset\PhenoBench_classficationv2\PhenoBench_classfication_PSR_with_smallobj_thr200\train\weed"
validation_folder = r"D:\masterPROJECT\laser_weeding\weed_dataset\PhenoBench_classficationv2\PhenoBench_classfication_PSR_with_smallobj_thr200\valid\weed"
test_folder = r"D:\masterPROJECT\laser_weeding\weed_dataset\PhenoBench_classficationv2\PhenoBench_classfication_PSR_with_smallobj_thr200\test\weed"

train_percentage = 0.7
validation_percentage = 0.2
test_percentage = 0.1

num_ranges = 3  # Adjust the number of size ranges as needed

distribute_files_by_size(original_folder, train_folder, validation_folder, test_folder, num_ranges, train_percentage, validation_percentage, test_percentage)
