import rasterio
import tensorflow as tf
import numpy as np
import os
import glob

def preprocess_groundtruth(folder_path, date_pattern):
    """
    Preprocess the ground truth raster by selecting the file with the specified date,
    setting all non-zero values to 1, and converting the result to a TensorFlow tensor.

    Args:
        folder_path (str): Path to the folder containing ground truth raster files.
        date_pattern (str): The date pattern (e.g., '2022-01-01') to identify the file.

    Returns:
        tf.Tensor: Processed ground truth data as a TensorFlow tensor.
    """
    # Use glob to find the file with the specified date in the name
    file_pattern = os.path.join(folder_path, f"*{date_pattern}*.tif")
    matching_files = glob.glob(file_pattern)
    
    if len(matching_files) == 0:
        raise FileNotFoundError(f"No file found with the date pattern '{date_pattern}' in '{folder_path}'.")
    elif len(matching_files) > 1:
        raise ValueError(f"Multiple files found with the date pattern '{date_pattern}': {matching_files}. Please ensure only one file matches.")
    
    groundtruth_path = matching_files[0]  # Get the first (and only) matching file

    # Open the ground truth raster using rasterio
    with rasterio.open(groundtruth_path) as src:
        groundtruth_data = src.read(1)  # Read the first band (assuming single-band groundtruth)

    # Set all non-zero values to 1
    groundtruth_data[groundtruth_data != 0] = 1

    # Convert to float32 and then to a TensorFlow tensor
    groundtruth_tensor = tf.convert_to_tensor(groundtruth_data.astype(np.float32), dtype=tf.float32)

    return groundtruth_tensor

# #load the groundtruth data,make it binary and put it in a tensor
groundtruth_path = "data/preprocessed/groundtruth/00N_080W"
date_pattern = "2022-06-01"
groundtruth_tensor = preprocess_groundtruth(groundtruth_path, date_pattern)
print(groundtruth_tensor.shape)

# Ensure groundtruth tensor has the correct shape: (batch_size, height, width, 1)
groundtruth_tensor = tf.expand_dims(groundtruth_tensor, axis=0)  # Add batch dimension
print(groundtruth_tensor.shape)  # This should output (1, 2500, 2500, 1)

