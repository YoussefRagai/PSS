import os
from pathlib import Path

# Base paths - using hardcoded paths like in msgpack_test.py
DATA_PATH = "/Users/youssefragai/Desktop/Master/UI/master_seasons.msgpack"
ANALYSIS_DATA_DIR = "/Users/youssefragai/Desktop/Master/UI/analysis"

def get_data_paths():
    """Get the data paths."""
    return {
        'master_seasons': DATA_PATH,
        'analysis_dir': ANALYSIS_DATA_DIR
    } 