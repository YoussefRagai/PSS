import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths - using environment variables with fallback to hardcoded paths
DATA_PATH = os.getenv('DATA_PATH', "/Users/youssefragai/Desktop/Master/UI/master_seasons.msgpack")
ANALYSIS_DATA_DIR = os.getenv('ANALYSIS_DATA_DIR', "/Users/youssefragai/Desktop/Master/UI/analysis")

def get_data_paths():
    """Get the data paths."""
    return {
        'master_seasons': DATA_PATH,
        'analysis_dir': ANALYSIS_DATA_DIR
    } 