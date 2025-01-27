#this function will exctract the features from a tile.
#a pre trained model is loaded.

import os
import numpy as np
from keras.models import load_model
from save_as import save_as_tiff

def extract_and_save_features(models_folder, input_image, output_folder, groundtruth_image_path):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all model files in the models folder, excluding weight-only files
    model_files = [f for f in os.listdir(models_folder) if f.endswith('.h5') and 'weights' not in f]
    print(model_files)
    
    for model_file in model_files:
        print(f"Processing model: {model_file}")
        
        # Load the model
        model_path = os.path.join(models_folder, model_file)
        model = load_model(model_path, compile=False)
        print(f"Model {model_file} loaded successfully!")
        
        # Extract features using the model
        feature_map = model.predict(input_image)
        
        # Check if feature_map is a list and extract the first element
        if isinstance(feature_map, list):
            feature_map = feature_map[0]
        
        # Ensure feature_map is a NumPy array
        if not isinstance(feature_map, np.ndarray):
            feature_map = np.array(feature_map)
        
        # Create a subfolder named after the model (without the .h5 extension)
        model_name = os.path.splitext(model_file)[0]
        model_output_folder = os.path.join(output_folder, model_name)
        os.makedirs(model_output_folder, exist_ok=True)
        
        # Save the feature map as a TIFF file
        feature_map_output_path = os.path.join(model_output_folder, f'{model_name}_feature_map.tif')
        save_as_tiff(feature_map, feature_map_output_path, groundtruth_image_path)
        print(f"Feature map for {model_file} saved successfully in {model_output_folder}!")

        

