"""Configrations."""

import os
from pathlib import Path


# Plot backend
_ENV_VAR_PLOT_BACKEND = "SOMAP_PLOT_BACKEND"
_SUPPORTED_BACKENDS = ["altair", "array2image"]
_DEFAULT_PLOT_BACKEND = "altair"

# Datasets folder path
_ENV_VAR_DATASETS_FOLDER_PATH = "SOMAP_DATASETS_FOLDER_PATH"
_DEFAULT_DATASETS_FOLDER_PATH = Path.cwd() / "somap_datasets"
DATASETS_FOLDER_PATH = Path(
    os.getenv(_ENV_VAR_DATASETS_FOLDER_PATH, _DEFAULT_DATASETS_FOLDER_PATH)
)
