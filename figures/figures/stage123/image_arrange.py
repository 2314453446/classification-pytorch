from PIL import Image
import os

def combine_images_from_folders(folders, indices, grid_size=(3, 3), gap=10, output_path="./combined_image1.jpg"):
    """
    Combine images from multiple folders into a single image.

    :param folders: List of folders containing images.
    :param indices: List of tuples with indices to pick from each folder.
    :param grid_size: Tuple representing the grid size for combining images.
    :param gap: The gap in pixels between images in the grid.
    :param output_path: Path to save the combined image.
    """
    if len(folders) != len(indices):
        print("Folders and indices length mismatch.")
        return

    # Load images based on the provided indices
    images = []
    for folder, idx in zip(folders, indices):
        all_files = [f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        selected_files = [all_files[i] for i in idx if i < len(all_files)]
        images.extend([Image.open(os.path.join(folder, file)) for file in selected_files])

    # Check if enough images are loaded
    if len(images) != grid_size[0] * grid_size[1]:
        print(f"Expected {grid_size[0] * grid_size[1]} images, but got {len(images)}")
        return

    # Calculate dimensions for the combined image
    width, height = images[0].size
    combined_width = grid_size[0] * width + (grid_size[0] - 1) * gap
    combined_height = grid_size[1] * height + (grid_size[1] - 1) * gap
    combined_image = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))

    # Place images on the grid
    for i, image in enumerate(images):
        row = i // grid_size[0]
        col = i % grid_size[0]
        x = col * (width + gap)
        y = row * (height + gap)
        combined_image.paste(image, (x, y))

    # Save the combined image
    combined_image.save(output_path)
    print(f"Combined image saved at {output_path}")

# Example usage
folders = [r"D:\Learning_software\classification-pytorch\figures\figures\stage123\label",
           r"D:\Learning_software\classification-pytorch\figures\figures\stage123\predict",
           r"D:\Learning_software\classification-pytorch\figures\figures\stage123\fack_predict"]  # Replace with actual folder paths
indices = [(1,0,9),(1,0,9),(1,0,9)]  # Indices of images to be taken from each folder
combine_images_from_folders(folders, indices)
