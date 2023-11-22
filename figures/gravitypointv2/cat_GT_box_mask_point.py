import os
from PIL import Image

# Replace these paths with the actual paths to your directories
directories = [r'D:\Learning_software\classification-pytorch\figures\gravitypointv2\images',
               r'D:\Learning_software\classification-pytorch\figures\gravitypointv2\images_predict',
               r'D:\Learning_software\classification-pytorch\figures\gravitypointv2\images_box',
               r'D:\Learning_software\classification-pytorch\figures\gravitypointv2\GT_vis',
               r'D:\Learning_software\classification-pytorch\figures\gravitypointv2\images_GT_box_pre']
output_directory = r'D:\Learning_software\classification-pytorch\figures\gravitypointv2\images_cat'

# Ensure the output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Get a list of image filenames from the first directory
image_filenames = [f for f in os.listdir(directories[0]) if os.path.isfile(os.path.join(directories[0], f))]

# Iterate over the image filenames
for filename in image_filenames:
    # Read and store all images from the directories
    images = [Image.open(os.path.join(directory, filename)) for directory in directories]

    # Calculate the total width and the maximum height
    total_width = sum(image.width for image in images)
    max_height = max(image.height for image in images)

    # Create a new blank image with the total width and maximum height
    new_image = Image.new('RGB', (total_width, max_height))

    # Paste images next to each other
    x_offset = 0
    for image in images:
        new_image.paste(image, (x_offset, 0))
        x_offset += image.width

    # Save the new image in the output directory
    new_image.save(os.path.join(output_directory, filename))
