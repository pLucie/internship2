import os
import glob
import numpy as np
import rasterio
import tensorflow as tf

def stack_tifs_by_date(folder_path, date_pattern, tile_id, desired_suffixes=None, normalization="min-max"):
    """
    Stacks and normalizes all TIFF files with a specific date pattern, tile ID, and desired suffixes from a folder,
    and directly converts them to a TensorFlow tensor with channel-last ordering.

    Args:
        folder_path (str): Path to the folder containing the TIFF files (without the tile ID).
        date_pattern (str): The date pattern (e.g., '2023-01-01') to filter the files.
        tile_id (str): The tile ID to specify the subdirectory or file naming convention (e.g., '00N_080W').
        desired_suffixes (list of str, optional): A list of desired suffixes (e.g., ['confidence', 'temperature']).
            If None, all files matching the date pattern will be included.
        normalization (str): Normalization method ('min-max' or 'z-score').

    Returns:
        tuple: A tuple containing:
            - tf.Tensor: A 4D tensor with shape (H, W, C) where C is the number of layers (TIFF files).
            - list: List of file names corresponding to each layer.
    """
    # Construct the full folder path using the tile ID
    full_folder_path = os.path.join(folder_path, tile_id)

    # Use glob to get all TIFF files with the specific date pattern
    file_pattern = os.path.join(full_folder_path, f"*{date_pattern}*.tif")
    tif_files = glob.glob(file_pattern)

    # Filter files by desired suffixes if provided
    if desired_suffixes is not None:
        tif_files = [
            tif for tif in tif_files
            if any(tif.endswith(f"_{suffix}.tif") for suffix in desired_suffixes)
        ]

    # Initialize a list to store the data arrays and file names
    rasters = []
    file_names = []

    # Loop over the found files and read them
    for tif_file in tif_files:
        with rasterio.open(tif_file) as src:
            # Read the data from the TIFF file (assuming it's a single band raster)
            raster_data = src.read(1)  # Read the first band (adjust if you have multiple bands)

            # Apply normalization
            if normalization == "min-max":
                min_val = np.nanmin(raster_data)
                max_val = np.nanmax(raster_data)
                # Avoid division by zero
                if max_val > min_val:
                    raster_data = (raster_data - min_val) / (max_val - min_val)
                else:
                    raster_data = np.zeros_like(raster_data)  # Handle edge case of constant raster
            elif normalization == "z-score":
                mean_val = np.nanmean(raster_data)
                std_val = np.nanstd(raster_data)
                # Avoid division by zero
                if std_val > 0:
                    raster_data = (raster_data - mean_val) / std_val
                else:
                    raster_data = np.zeros_like(raster_data)  # Handle edge case of constant raster

            rasters.append(raster_data)
            file_names.append(os.path.basename(tif_file))  # Keep only the file name

    # Stack the rasters into a 3D numpy array (stacking along a new axis)
    if rasters:
        stacked_rasters = np.stack(rasters, axis=-1)

        # Convert the stacked raster to a TensorFlow tensor with channel-last ordering (H, W, C)
        input_tensor = tf.convert_to_tensor(stacked_rasters, dtype=tf.float32)
        
        # Add a batch dimension to the tensor (1, H, W, C)
        input_tensor = tf.expand_dims(input_tensor, axis=0)
        
    else:
        # Handle case where no files are found
        input_tensor = tf.constant([], dtype=tf.float32)

    return input_tensor, file_names


def stack_tifs_by_date2(folder_path, date_pattern, desired_suffixes=None, normalization="min-max"):
    """
    Stacks and normalizes all TIFF files with a specific date pattern and desired suffixes from a folder,
    and directly converts them to a TensorFlow tensor with channel-last ordering.

    Args:
        folder_path (str): Path to the folder containing the TIFF files.
        date_pattern (str): The date pattern (e.g., '2023-01-01') to filter the files.
        desired_suffixes (list of str, optional): A list of desired suffixes (e.g., ['aridityannual', 'closenesstoroads']).
            If None, all files matching the date pattern will be included.
        normalization (str): Normalization method ('min-max' or 'z-score').

    Returns:
        tuple: A tuple containing:
            - tf.Tensor: A 4D tensor with shape (H, W, C) where C is the number of layers (TIFF files).
            - list: List of file names corresponding to each layer.
    """
    # Use glob to get all TIFF files with the specific date pattern
    file_pattern = os.path.join(folder_path, f"*{date_pattern}*.tif")
    tif_files = glob.glob(file_pattern)

    # Filter files by desired suffixes if provided
    if desired_suffixes is not None:
        tif_files = [
            tif for tif in tif_files
            if any(tif.endswith(f"_{suffix}.tif") for suffix in desired_suffixes)
        ]

    # Initialize a list to store the data arrays and file names
    rasters = []
    file_names = []

    # Loop over the found files and read them
    for tif_file in tif_files:
        with rasterio.open(tif_file) as src:
            # Read the data from the TIFF file (assuming it's a single band raster)
            raster_data = src.read(1)  # Read the first band (adjust if you have multiple bands)

            # Apply normalization
            if normalization == "min-max":
                min_val = np.nanmin(raster_data)
                max_val = np.nanmax(raster_data)
                # Avoid division by zero
                if max_val > min_val:
                    raster_data = (raster_data - min_val) / (max_val - min_val)
                else:
                    raster_data = np.zeros_like(raster_data)  # Handle edge case of constant raster
            elif normalization == "z-score":
                mean_val = np.nanmean(raster_data)
                std_val = np.nanstd(raster_data)
                # Avoid division by zero
                if std_val > 0:
                    raster_data = (raster_data - mean_val) / std_val
                else:
                    raster_data = np.zeros_like(raster_data)  # Handle edge case of constant raster

            rasters.append(raster_data)
            file_names.append(os.path.basename(tif_file))  # Keep only the file name

    # Stack the rasters into a 3D numpy array (stacking along a new axis)
    if rasters:
        stacked_rasters = np.stack(rasters, axis=-1)

        # Convert the stacked raster to a TensorFlow tensor with channel-last ordering (H, W, C)
        input_tensor = tf.convert_to_tensor(stacked_rasters, dtype=tf.float32)
    else:
        # Handle case where no files are found
        input_tensor = tf.constant([], dtype=tf.float32)

    return input_tensor, file_names
  

