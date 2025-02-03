# This function will rename the model outputs to the desired format, which is: {TILE_ID}_{DATE}_{FEATURE}.tif.

rename_tiff_files <- function(base_dir = "C:/internship2/output", 
                              tile_id = "00N_080W", 
                              date = "2023-01-01") {
  # Load required package
  if (!requireNamespace("stringr", quietly = TRUE)) {
    install.packages("stringr")
  }
  library(stringr)
  
  # Get all folders in the base directory
  folders <- list.dirs(base_dir, recursive = FALSE)
  
  for (folder in folders) {
    # List all TIFF files in the folder
    tiff_files <- list.files(folder, pattern = "\\.tif$", full.names = TRUE)
    
    for (file in tiff_files) {
      # Extract feature name from original filename
      feature_match <- str_match(basename(file), "feature_map_feature_(\\d+)")
      
      if (!is.na(feature_match[2])) {
        feature <- paste0("feature_", feature_match[2])
        
        # Construct new filename
        new_filename <- paste0(tile_id, "_", date, "_", feature, ".tif")
        new_filepath <- file.path(folder, new_filename)
        
        # Rename the file
        file.rename(file, new_filepath)
        
        message("Renamed: ", file, " -> ", new_filepath)
      } else {
        warning("Skipping file (feature not found): ", file)
      }
    }
  }
}

# Run the function
rename_tiff_files()
