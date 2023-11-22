import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Path to the CSV file with filtered Class 1 accuracies >= 0.9
filtered_csv_class_1_path = r'D:\Learning_software\classification-pytorch\figures\mask_accuracy\overlap_result_75_90.csv'

# Reading the CSV file into a DataFrame
df_filtered_class_1 = pd.read_csv(filtered_csv_class_1_path)

# Determine the window size for the moving average (smoothing), with a minimum of 2
window_size = max(2, int(len(df_filtered_class_1) * 0.1))

# Calculate the moving averages for each class to smooth the data
smoothed_accuracy_class_0 = df_filtered_class_1['Accuracy Class 0'].rolling(window=window_size, min_periods=1).mean()
smoothed_accuracy_class_1 = df_filtered_class_1['Accuracy Class 1'].rolling(window=window_size, min_periods=1).mean()
smoothed_accuracy_class_2 = df_filtered_class_1['Accuracy Class 2'].rolling(window=window_size, min_periods=1).mean()
print(type(smoothed_accuracy_class_1))

# Create a line plot with different colors for each class
plt.figure(figsize=(10, 6))

# Plotting each class with a distinct color
plt.plot(smoothed_accuracy_class_0[5:], label='Background IOU', marker='o', color='blue')
plt.plot(smoothed_accuracy_class_1[5:], label='Crop IOU', marker='s', color='green')
plt.plot(smoothed_accuracy_class_2[5:], label='Weed IOU', marker='^', color='red')

# Adding labels and a title to the plot
plt.xlabel('Sample Index')
plt.ylabel('Smoothed Mask IOU')
plt.title('Smoothed IOU by Class')

# Adding a legend to the plot
plt.legend()

# Adding grid lines to the plot for better readability
plt.grid(True)

# Saving the figure to a file
smoothed_figure_path = './smoothed_accuracy_by_class.png'
plt.savefig(smoothed_figure_path, dpi=300)

# Closing the plot to prevent display in the current environment
plt.close()

# Output the path where the figure is saved
smoothed_figure_path
