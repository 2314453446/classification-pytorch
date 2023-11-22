from PIL import Image
import numpy as np
import os

# Define paths
depth_dir = r'D:\Learning_software\classification-pytorch\figures\mask_accuracy\label2relabel'
color_dir = r'D:\Learning_software\classification-pytorch\figures\mask_accuracy\val\images'
output_dir = r'D:\Learning_software\classification-pytorch\figures\mask_accuracy\vis_lab'

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define colors for each class in RGBA format (with 0.5 opacity for the last value)
colors = {
    0: (0, 0, 0, 0),          # Background, fully transparent
    1: (0, 255, 0, 128),      # Crop, green with 0.5 transparency
    2: (255, 0, 0, 128)       # Weed, red with 0.5 transparency
}

# Get the list of depth image filenames
depth_image_files = [f for f in os.listdir(depth_dir) if f.endswith('.png')]

# Process each depth image
for depth_image_file in depth_image_files:
    depth_image_path = os.path.join(depth_dir, depth_image_file)
    color_image_path = os.path.join(color_dir, depth_image_file)  # Assuming same filename for color image

    # Open the depth image and create masks
    depth_image = Image.open(depth_image_path)
    depth_array = np.array(depth_image)

    # Open the color image
    color_image = Image.open(color_image_path).convert("RGBA")

    # Create an RGBA image for overlay
    overlay = Image.new("RGBA", color_image.size, (0, 0, 0, 0))
    overlay_array = np.array(overlay)

    # Apply colors to each class in the overlay
    for class_value, color in colors.items():
        class_mask = (depth_array == class_value)
        overlay_array[class_mask] = color

    # Convert the overlay array back to an image
    overlay_image = Image.fromarray(overlay_array)

    # Composite the overlay onto the color image
    combined_image = Image.alpha_composite(color_image, overlay_image)

    # Save the combined image to the output directory
    combined_image.save(os.path.join(output_dir, depth_image_file))
