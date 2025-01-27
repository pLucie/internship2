#creates the function to make the data folder in the current directory
create_data_folder <- function() {
  # Define the folder path
  folder_path <- file.path(getwd(), "data")

  # Check if the folder already exists
  if (!dir.exists(folder_path)) {
    # Create the folder
    dir.create(folder_path)
    message("Folder 'data' created in the current working directory.")
  } else {
    message("Folder 'data' already exists in the current working directory.")
  }
}

#call the function
create_data_folder()
