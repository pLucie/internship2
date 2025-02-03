import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from save_as import save_as_tiff


# def extract_and_save_features(models_folder, input_image, output_folder, groundtruth_image_path):
#     # Ensure the output folder exists
#     os.makedirs(output_folder, exist_ok=True)
#
#     # Get all model files in the models folder, excluding weight-only files
#     model_files = [f for f in os.listdir(models_folder) if f.endswith('.h5') and 'weights' not in f]
#     print(model_files)
#
#     for model_file in model_files:
#         print(f"Processing model: {model_file}")
#
#         # Create a subfolder named after the model (without the .h5 extension)
#         model_name = os.path.splitext(model_file)[0]
#         model_output_folder = os.path.join(output_folder, model_name)
#         os.makedirs(model_output_folder, exist_ok=True)
#
#         # Define the expected output file path for the feature map
#         feature_map_output_path = os.path.join(model_output_folder, f'{model_name}_feature_map.tif')
#
#         # Check if the feature map already exists
#         if os.path.exists(feature_map_output_path):
#             print(f"Feature map for {model_file} already exists, skipping...")
#             continue
#
#         # Load the model
#         model_path = os.path.join(models_folder, model_file)
#         model = load_model(model_path, compile=False)
#         print(f"Model {model_file} loaded successfully!")
#
#         # Extract features using the model
#         feature_map = model.predict(input_image)
#
#         # Check if feature_map is a list and extract the first element
#         if isinstance(feature_map, list):
#             feature_map = feature_map[0]
#
#         # Ensure feature_map is a NumPy array
#         if not isinstance(feature_map, np.ndarray):
#             feature_map = np.array(feature_map)
#
#         # Save the feature map as a TIFF file
#         save_as_tiff(feature_map, feature_map_output_path, groundtruth_image_path)
#         print(f"Feature map for {model_file} saved successfully in {model_output_folder}!")

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# def extract_and_save_features(models_folder, input_image, output_folder, groundtruth_image_path):
#     os.makedirs(output_folder, exist_ok=True)
#
#     model_files = [f for f in os.listdir(models_folder) if f.endswith('.h5') and 'weights' not in f]
#     print(f"Found {len(model_files)} models: {model_files}")
#
#     input_tensor = tf.convert_to_tensor(input_image, dtype=tf.float32)  # Convert input to tensor once
#
#     for model_file in model_files:
#         model_name = os.path.splitext(model_file)[0]
#         model_output_folder = os.path.join(output_folder, model_name)
#         os.makedirs(model_output_folder, exist_ok=True)
#
#         feature_map_output_path = os.path.join(model_output_folder, f'{model_name}_feature_map.tif')
#
#         if os.path.exists(feature_map_output_path):
#             print(f"Feature map for {model_file} already exists, skipping...")
#             continue
#
#         print(f"Loading model: {model_file}")
#         model_path = os.path.join(models_folder, model_file)
#         model = load_model(model_path, compile=False)
#         print(f"Model {model_file} loaded successfully!")
#
#         # Optimize inference using TensorFlow graph execution
#         model_fn = tf.function(model)
#
#         print(f"Extracting features for {model_file}...")
#         feature_map = model_fn(input_tensor, training=False)  # Faster inference
#
#         if isinstance(feature_map, list):
#             feature_map = feature_map[0]
#
#         feature_map = np.array(feature_map)  # Ensure NumPy array
#
#         save_as_tiff(feature_map, feature_map_output_path, groundtruth_image_path)
#         print(f"Feature map for {model_file} saved successfully!")

# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model


def extract_and_save_features(models_folder, input_image, output_folder, groundtruth_image_path):
    os.makedirs(output_folder, exist_ok=True)

    model_files = [f for f in os.listdir(models_folder) if f.endswith('.h5') and 'weights' not in f]
    print(f"Found {len(model_files)} models: {model_files}")

    input_tensor = tf.convert_to_tensor(input_image, dtype=tf.float32)  # Convert input to tensor once

    for model_file in model_files:
        model_name = os.path.splitext(model_file)[0]
        model_output_folder = os.path.join(output_folder, model_name)
        os.makedirs(model_output_folder, exist_ok=True)

        # Construct the dynamic feature map filename pattern
        feature_map_exists = False
        for i in range(input_image.shape[0]):  # Assuming each "feature" corresponds to a dimension of input_image
            feature_map_output_path = os.path.join(model_output_folder, f'{model_name}_feature_map_feature_{i}.tif')
            if os.path.exists(feature_map_output_path):
                print(f"Feature map for {model_name}_feature_{i} already exists at {feature_map_output_path}, skipping...")
                feature_map_exists = True
                break

        if feature_map_exists:
            continue

        print(f"Loading model: {model_file}")
        model_path = os.path.join(models_folder, model_file)
        model = load_model(model_path, compile=False)
        print(f"Model {model_file} loaded successfully!")

        # Optimize inference using TensorFlow graph execution
        model_fn = tf.function(model)

        print(f"Extracting features for {model_name}...")
        feature_map = model_fn(input_tensor, training=False)  # Faster inference

        if isinstance(feature_map, list):
            feature_map = feature_map[0]

        feature_map = np.array(feature_map)  # Ensure NumPy array

        # Save the feature maps for each feature in the model's output
        for i in range(feature_map.shape[-1]):  # Iterate through the features
            feature_map_output_path = os.path.join(model_output_folder, f'{model_name}_feature_map_feature_{i}.tif')
            save_as_tiff(feature_map[..., i], feature_map_output_path, groundtruth_image_path)
            print(f"Feature map for {model_name}_feature_{i} saved successfully at {feature_map_output_path}!")




