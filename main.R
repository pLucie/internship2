#this file will call all the function that you will need to run
#the code for this internship.

#the following function will download and load all the required packages for this project:
#the function that you load in is called package_check and check if the packages are already downloaded
#and otherwise downloads and loads them.
source("r/load_packages.R")

#the following part will get you the tile data for the requested country add save this data
#in a created folder called data in you current directory.
source("r/create_data_folder.R")

#setup a virtual environment to run python scripts:
source("r/create_virtual_environment.r")

#all python files will be called from within the main python file to hopefully speed up processing time.
source_python("python/run.py")

#next step is to use the forest foresight package to make predictions with the extracted features combined with the original features.
#make sure to make the prediction for after
# after running run.py you will get all the models and different outputs for each model settings. we will now start testing which setting yields the best results.
# first make a function to rename the current files to the forest foresight structure. which is the following:{TILE_ID}_{DATE}_{FEATURE}.tif
source("r/rename_tif_files.R")
