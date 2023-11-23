"""Utility functions for loading datasets."""

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..config import DATASETS_FOLDER_PATH


def load_data_with_caching(filename: str, load_function: Callable):
    """Loads from local file if it exists, otherwise calls an external loading function.

    Args:
        filename: Name of the file where the data will be cached in the datasets folder
            specified in the `SOMAP_DATASETS_FOLDER_PATH` environment variable. Should
            include the file extension.
        load_function: Function that loads the data and returns it as a Numpy array.
    """
    filepath: Path = DATASETS_FOLDER_PATH / filename

    try:
        data = np.load(filepath)
        logging.info(f"Loading data from existing file at `{filepath}`.")

    except FileNotFoundError:
        logging.info(f"Data file not found at `{filepath}`. Invoking load function...")
        data = load_function()
        if not DATASETS_FOLDER_PATH.exists():
            DATASETS_FOLDER_PATH.mkdir(exist_ok=True)
        np.save(filepath, data)
        logging.info(f"Saving data to file at `{filepath}` to speed up next loadings.")

    except IOError as e:
        logging.error(f"Error reading file at `{filepath}`: {e}")
        raise

    return data
