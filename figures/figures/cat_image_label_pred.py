from PIL import Image

# Replace the following paths with the paths of your images
image_paths = [r"D:\Learning_software\classification-pytorch\figures\figures\label_pred\label\05-26_00179_P0034121.png",
               r"D:\Learning_software\classification-pytorch\figures\figures\label_pred\label\05-15_00062_P0030859.png",
               r"D:\Learning_software\classification-pytorch\figures\figures\label_pred\label\06-05_00046_P0037822.png",
               
               r"D:\Learning_software\classification-pytorch\figures\figures\label_pred\predict\05-26_00179_P0034121.png",
               r"D:\Learning_software\classification-pytorch\figures\figures\label_pred\predict\05-15_00062_P0030859.png",
               r"D:\Learning_software\classification-pytorch\figures\figures\label_pred\predict\06-05_00046_P0037822.png"]

images = [Image.open(image) for image in image_paths]

# Assuming all images are the same size
width, height = images[0].size

# Create a new image with a width of 3 images and a height of 2 images
combined_image = Image.new('RGB', (3 * width, 2 * height))

# Paste each image into the combined image
for i, image in enumerate(images):
    combined_image.paste(image, (width * (i % 3), height * (i // 3)))

# Save the combined image
combined_image.save(r"D:\Learning_software\classification-pytorch\figures\figures\combined_image.jpg")
combined_image.show()
