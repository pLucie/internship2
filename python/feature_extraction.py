import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
from preprocess_groundtruth_tensorflow import preprocess_groundtruth
from stack_tifs_by_date_tensorflow import stack_tifs_by_date2
from save_as import save_as_tiff

def process_with_model(model_path, output_dir, input_folder_path, groundtruth_path, dates_to_process, tile_id, monthly_datasets):
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    # Create output directory for this model
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    print(f"\nLoading model: {model_path}")

    # Load the model
    model = load_model(model_path, compile=False)
    print("Model loaded successfully!")

    # Loop through each date and process it
    for date_pattern in dates_to_process:
        print(f"\nProcessing date: {date_pattern}")

        # Fix: Add tile_id as an argument
        input_image, _ = stack_tifs_by_date2(input_folder_path, date_pattern, desired_suffixes=monthly_datasets)

        # Ensure input tensor has the correct shape
        input_image = tf.expand_dims(input_image, axis=0)
        print(f"Input tensor shape: {input_image.shape}")

        # Predict feature maps
        feature_map = model.predict(input_image)

        # Convert feature map to numpy array
        feature_map = np.array(feature_map)

        # Define output file path inside the model's folder
        output_path = os.path.join(model_output_dir, f"{date_pattern}_feature_map_output.tif")

        # Save the feature map as a TIFF file
        save_as_tiff(feature_map, output_path, groundtruth_path)
        print(f"Feature map saved: {output_path}")

    # Clear session to free memory
    del model
    tf.keras.backend.clear_session()
    print(f"Model {model_path} unloaded.\n")







