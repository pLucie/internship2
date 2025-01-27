# Load Python libraries

# Standard library imports
import os
import glob

# Third-party library imports
import numpy as np
import rasterio
from rasterio.transform import from_origin
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras import Model, Input
from keras.models import load_model

#load user defined functions
from stack_tifs_by_date_tensorflow import stack_tifs_by_date
from preprocess_groundtruth_tensorflow import preprocess_groundtruth
from save_as import save_as_tiff
from extract_features import extract_and_save_features
from resunet import ResUNet
from prepare_input_data import prepare_input_data
from add_batch_dimension import add_batch_dimension

#run the functions
if __name__ == "__main__":
  
    # Global constants
    TILE_ID = "00N_080W"
    DATA_FOLDER_PATH_INPUT = "data/preprocessed/input"
    DATE_PATTERN_INPUT = "2022-01-01"
    DATA_FOLDER_PATH_GROUNDTRUTH = "data/preprocessed/groundtruth"
    DATE_PATTERN_GROUNDTRUTH = "2022-06-01"
    MONTHLY_DATASETS = [
    "confidence", "lastmonth", "lastsixmonths", "lastthreemonths",
    "patchdensity", "precipitation", "previoussameseason",
    "smoothedsixmonths", "smoothedtotal", "temperature", "timesinceloss",
    "nightlights", "totallossalerts"
    ]
  
    # Call the main processing logic
    input_image, file_names = stack_tifs_by_date(DATA_FOLDER_PATH_INPUT, DATE_PATTERN_INPUT, TILE_ID, MONTHLY_DATASETS)
    input_image = add_batch_dimension(input_image)
    groundtruth_image = add_batch_dimension(preprocess_groundtruth(DATA_FOLDER_PATH_GROUNDTRUTH, DATE_PATTERN_GROUNDTRUTH, TILE_ID))



# def weighted_binary_crossentropy(y_true, y_pred, weight_1=2.5):
#     """
#     Weighted binary crossentropy loss function.
# 
#     Parameters:
#     - y_true: Ground truth values.
#     - y_pred: Predicted values.
#     - weight_1: Weight for the positive class (1).
# 
#     Returns:
#     - Weighted binary crossentropy loss.
#     """
#     epsilon = K.epsilon()
#     y_pred = K.clip(y_pred, epsilon, 1 - epsilon)
#     # Compute binary crossentropy
#     bce = - (y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
#     # Apply weights
#     weighted_bce = bce * (y_true * weight_1 + (1 - y_true))
#     return K.mean(weighted_bce)

  
#input_layer = Input(shape=(2500, 2500, 13))  # Define the input layer
#resunet_features = ResUNet(input_layer)

# Feature extractor output: 5 feature maps
#feature_output = Conv2D(5, (1, 1), activation='linear', name='feature_output')(resunet_features)

# Ground truth classification output: 1 binary map
#classification_output = Conv2D(1, (1, 1), activation='sigmoid', name='classification_output')(resunet_features)

# Define the model
#model = Model(inputs=input_layer, outputs=[feature_output, classification_output])

# model.compile(optimizer='adam',
#               loss={'feature_output': None,  # No loss for feature output
#                     'classification_output':
#                         lambda y_true, y_pred: weighted_binary_crossentropy(y_true, y_pred, weight_1=2.0)},
#               loss_weights={'feature_output': 0.0,  # Do not focus on feature extraction
#                             'classification_output': 1.0})
# 

