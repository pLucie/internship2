import rasterio
import tensorflow as tf
import numpy as np
import os
import glob

def preprocess_groundtruth(folder_path, date_pattern, tile_id):
    """
    Preprocess the ground truth raster by selecting the file with the specified date and tile ID,
    setting all non-zero values to 1, and converting the result to a TensorFlow tensor.

    Args:
        folder_path (str): Path to the folder containing ground truth raster files (without the tile ID).
        date_pattern (str): The date pattern (e.g., '2022-01-01') to identify the file.
        tile_id (str): The tile ID to specify the subdirectory or file naming convention (e.g., '00N_080W').

    Returns:
        tf.Tensor: Processed ground truth data as a TensorFlow tensor.
    """
    # Construct the full folder path using the tile ID
    full_folder_path = os.path.join(folder_path, tile_id)

    # Use glob to find the file with the specified date in the name
    file_pattern = os.path.join(full_folder_path, f"*{date_pattern}*.tif")
    matching_files = glob.glob(file_pattern)
    
    if len(matching_files) == 0:
        raise FileNotFoundError(f"No file found with the date pattern '{date_pattern}' in '{full_folder_path}'.")
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
