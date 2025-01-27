# Path to virtual environment in current directory
env_path <- file.path(getwd(), "venv")

# Create the virtual environment if it doesn't exist
if (!virtualenv_exists(env_path)) {
  virtualenv_create(envname = env_path)
  virtualenv_install(envname = env_path, packages = c("numpy", "tensorflow", "rasterio"))
}

# Use the virtual environment
use_virtualenv(env_path, required = TRUE)

# Path to the virtual environment's Python executable
env_path <- file.path(getwd(), "venv/Scripts/python.exe")

# Set the Python interpreter for reticulate
use_python(env_path, required = TRUE)
