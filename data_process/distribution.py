import os
from collections import Counter

def calculate_size_distribution(folder):
    file_sizes = [os.path.getsize(os.path.join(folder, file)) for file in os.listdir(folder) if os.path.isfile(os.path.join(folder, file))]
    size_distribution = Counter(file_sizes)
    return size_distribution

def calculate_percentage_distribution(distribution, total_files):
    percentage_distribution = {size: (count / total_files) * 100 for size, count in distribution.items()}
    return percentage_distribution

# Set the path to your folder
folder_path = r"D:\masterPROJECT\laser_weeding\weed_dataset\PhenoBench_classficationv2\with_small_objects\train\crop"

# Define your size ranges (modify as needed)
small_size_range = (0, 1000)
medium_size_range = (1001, 5000)
big_size_range = (5001, float('inf'))

# Get the size distribution
distribution = calculate_size_distribution(folder_path)

# Calculate the total number of files
total_files = sum(distribution.values())

# Calculate the percentage distribution within each size range
small_files = sum(count for size, count in distribution.items() if small_size_range[0] <= size <= small_size_range[1])
medium_files = sum(count for size, count in distribution.items() if medium_size_range[0] <= size <= medium_size_range[1])
big_files = sum(count for size, count in distribution.items() if big_size_range[0] <= size <= big_size_range[1])

small_percentage = (small_files / total_files) * 100
medium_percentage = (medium_files / total_files) * 100
big_percentage = (big_files / total_files) * 100

# Display the percentage distribution within each size range
print("Percentage Distribution Within Each Size Range:")
print(f"Small Files: {small_percentage:.2f}%")
print(f"Medium Files: {medium_percentage:.2f}%")
print(f"Big Files: {big_percentage:.2f}%")