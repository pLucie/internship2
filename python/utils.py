import os

def get_completed_runs(output_dir):
    """
    Extracts completed run IDs from model filenames in the output directory.
    """
    completed_runs = set()
    for file_name in os.listdir(output_dir):
        if file_name.endswith(".h5") and "resunet" in file_name:
            run_id = file_name.replace("resunet_", "").replace(".h5", "")
            completed_runs.add(run_id)
    return completed_runs


def log_completed_runs(log_file_path, completed_runs):
    """
    Writes the completed run IDs to a log file.
    """
    with open(log_file_path, "w") as log_file:
        for run_id in completed_runs:
            log_file.write(run_id + "\n")
    print(f"Completed runs logged in: {log_file_path}")


def load_completed_runs_from_log(log_file_path):
    """
    Loads completed runs from an existing log file if it exists.
    """
    if os.path.exists(log_file_path):
        with open(log_file_path, "r") as log_file:
            return set(log_file.read().splitlines())
    return set()
