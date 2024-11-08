import os
from PIL import Image

def find_smallest_image_size(directory):
    smallest_size = None
    smallest_image = None

    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            filepath = os.path.join(directory, filename)
            with Image.open(filepath) as img:
                size = img.size
                if smallest_size is None or size < smallest_size:
                    smallest_size = size
                    smallest_image = filename

    if smallest_size:
        print(f"The smallest image is {smallest_image} with size {smallest_size}")
    else:
        print("No images found in the directory.")

# Example usage
directory_path = './dataset/2'
find_smallest_image_size(directory_path)
