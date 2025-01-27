import tensorflow as tf
from stack_tifs_by_date_tensorflow import stack_tifs_by_date

# Function to define the input path and datasets
def prepare_input_data(folder_path, date_pattern, datasets):
    """
    Prepares input data by stacking TIFs based on the date pattern and desired datasets.

    Parameters:
        folder_path (str): Path to the data folder.
        date_pattern (str): Date pattern to filter files (e.g., "YYYY-MM-DD").
        datasets (list): List of dataset suffixes to include.

    Returns:
        tf.Tensor: Input image tensor with a batch dimension.
        list: File names corresponding to the stacked datasets.
    """
    # Stack TIFs based on date and dataset suffixes
    input_image, file_names = stack_tifs_by_date(folder_path, date_pattern, desired_suffixes=datasets)
    
    # Add batch dimension to the input image
    input_image = tf.expand_dims(input_image, axis=0)
    
    return input_image, file_names
