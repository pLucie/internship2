import rasterio
import numpy as np
from rasterio.transform import from_origin

def save_as_tiff(array, output_path, reference_image_path, index=None):
    """
    Save a NumPy array or a list of arrays as GeoTIFF files.
    """
    # Load reference image metadata
    with rasterio.open(reference_image_path) as src:
        transform = src.transform
        crs = src.crs

    if isinstance(array, list):
        # Save each array in the list separately
        for i, arr in enumerate(array):
            output_file = f"{output_path.rstrip('.tif')}_layer_{i}.tif"
            save_as_tiff(arr, output_file, reference_image_path, index=i)
    elif isinstance(array, np.ndarray):
        # If the array is 4D with shape (batch_size, height, width, channels), save each channel separately
        if len(array.shape) == 4:
            batch_size, height, width, channels = array.shape
            for i in range(channels):
                feature_map_channel = array[0, :, :, i]  # Extract the i-th channel (batch_size=1)
                output_file = f"{output_path.rstrip('.tif')}_feature_{i}.tif"
                with rasterio.open(
                        output_file,
                        'w',
                        driver='GTiff',
                        count=1,
                        dtype=feature_map_channel.dtype.name,
                        width=width,
                        height=height,
                        crs=crs,
                        transform=transform,
                ) as dst:
                    dst.write(feature_map_channel, 1)
        else:
            # Handle other shapes (e.g., 2D array)
            with rasterio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    count=1,
                    dtype=array.dtype.name,
                    width=array.shape[1],
                    height=array.shape[0],
                    crs=crs,
                    transform=transform,
            ) as dst:
                dst.write(array, 1)
    else:
        raise TypeError("Input array must be a NumPy array or a list of NumPy arrays.")
