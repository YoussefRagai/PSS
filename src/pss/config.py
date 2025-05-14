import os
from pathlib import Path

# Get the directory where the executable is located
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent

# Define paths relative to the executable location
DATA_PATH = str(BASE_DIR / "master_seasons.msgpack")
ANALYSIS_DATA_DIR = str(BASE_DIR / "analysis")

def get_data_paths():
    """Get the paths for data files."""
    return {
        'data_path': DATA_PATH,
        'analysis_dir': ANALYSIS_DATA_DIR
    } 