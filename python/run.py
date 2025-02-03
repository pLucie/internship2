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

# load user defined functions
from stack_tifs_by_date_tensorflow import stack_tifs_by_date
from preprocess_groundtruth_tensorflow import preprocess_groundtruth
from save_as import save_as_tiff
from extract_features import extract_and_save_features
from resunet import ResUNet
from prepare_input_data import prepare_input_data
from add_batch_dimension import add_batch_dimension
from weighted_loss_function import weighted_binary_crossentropy
from F05_loss_function import weighted_f05_loss
from utils import get_completed_runs, log_completed_runs, load_completed_runs_from_log
from train_and_save_models import train_and_save_models
from resunet import ResUNet
from extract_features import extract_and_save_features

# run the functions
if __name__ == "__main__":

    # Ensure paths are handled correctly
    BASE_DIR = os.path.abspath("C:/internship2/")  # Set this to your project root
    TILE_ID = "00N_080W"
    DATA_FOLDER_PATH_INPUT = os.path.join(BASE_DIR, "data", "preprocessed", "input")
    DATA_FOLDER_PATH_GROUNDTRUTH = os.path.join(BASE_DIR, "data", "preprocessed", "groundtruth")
    GROUNDTRUTH_IMAGE_PATH = os.path.join(DATA_FOLDER_PATH_GROUNDTRUTH, TILE_ID,
                                          f"{TILE_ID}_2022-06-01_groundtruth6m.tif")
    OUTPUT_DIR_IMAGES = os.path.join(BASE_DIR, "output")
  
    # GLOBAL VARIABLES
    DATE_PATTERN_INPUT = "2022-01-01"
    DATE_PATTERN_GROUNDTRUTH = "2022-06-01"
    MONTHLY_DATASETS = [
        "confidence", "lastmonth", "lastsixmonths", "lastthreemonths",
        "patchdensity", "precipitation", "previoussameseason",
        "smoothedsixmonths", "smoothedtotal", "temperature", "timesinceloss",
        "nightlights", "totallossalerts"]
    
    # PARAMETERS FOR TRAINING
    EPOCHS_LIST = [10, 30]
    LOSS_FUNCTIONS = { 
      "weighted_binary_crossentropy": weighted_binary_crossentropy,
      "weighted_f05_loss": weighted_f05_loss}
    GROUNDTRUTH_WEIGHTS = [1.0, 2.0, 5.0]
    OUTPUT_DIR_MODELS = "C:/models/automated_training/"

    # Call the main processing logic for defining the input image and the groundtruth image for single image training
    input_image, file_names = stack_tifs_by_date(DATA_FOLDER_PATH_INPUT, DATE_PATTERN_INPUT, TILE_ID, MONTHLY_DATASETS)
    print(input_image.shape)
    groundtruth_image = add_batch_dimension(preprocess_groundtruth(DATA_FOLDER_PATH_GROUNDTRUTH, DATE_PATTERN_GROUNDTRUTH, TILE_ID))
    print(groundtruth_image.shape)
    
    # Completed runs
    completed_runs = get_completed_runs(OUTPUT_DIR_MODELS)
    log_file_path = os.path.join(OUTPUT_DIR_MODELS, "completed_runs.log")
    log_completed_runs(log_file_path, completed_runs)

    # Train models
    train_and_save_models(OUTPUT_DIR_MODELS, EPOCHS_LIST, LOSS_FUNCTIONS, GROUNDTRUTH_WEIGHTS, input_image, groundtruth_image)

    # define the parameters for feature extraction (change the input image to a year later)
    DATE_PATTERN_INPUT = "2023-01-01"   # use a later date to extract the features from and test the outputs of this.
    input_image = stack_tifs_by_date(DATA_FOLDER_PATH_INPUT, DATE_PATTERN_INPUT, TILE_ID, MONTHLY_DATASETS)

    # load models, extract features and save them.
    extract_and_save_features(OUTPUT_DIR_MODELS, input_image, OUTPUT_DIR_IMAGES, GROUNDTRUTH_IMAGE_PATH)
