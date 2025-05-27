import os
import h5py
import numpy as np
from PIL import Image

def save_images_to_h5py(folder_path, h5py_file_path, image_size=(128, 128)):
    """
    Save images from a folder into an HDF5 file.

    Parameters:
    - folder_path (str): Path to the folder containing images.
    - h5py_file_path (str): Path to save the HDF5 file.
    - image_size (tuple): Desired size to resize images (width, height).
    """
    # Get a list of image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    print(f"Found {len(image_files)} images in folder: {folder_path}")

    # Create an HDF5 file
    with h5py.File(h5py_file_path, 'w') as h5f:
        # Create datasets for images and file names
        images_dataset = h5f.create_dataset(
            'images',
            shape=(len(image_files), *image_size, 3),  # Assuming RGB images
            dtype=np.uint8
        )
        filenames_dataset = h5f.create_dataset(
            'filenames',
            shape=(len(image_files),),
            dtype=h5py.string_dtype(encoding='utf-8')
        )

        for i, image_file in enumerate(image_files):
            image_path = os.path.join(folder_path, image_file)

            try:
                # Open and resize the image
                with Image.open(image_path) as img:
                    img = img.convert('RGB')  # Ensure RGB format
                    img = img.resize(image_size, Image.ANTIALIAS)
                    images_dataset[i] = np.asarray(img)  # Convert to numpy array and store
                    filenames_dataset[i] = image_file  # Store the filename
                print(f"Processed {i + 1}/{len(image_files)}: {image_file}")

            except Exception as e:
                print(f"Error processing {image_file}: {e}")

    print(f"Images saved to HDF5 file: {h5py_file_path}")

# Example usage
# if __name__ == "__main__":
#     folder_path = "counts_5"  # Replace with your folder path
#     h5py_file_path = "output_images.h5"  # Replace with your desired HDF5 file name
#     save_images_to_h5py(folder_path, h5py_file_path)