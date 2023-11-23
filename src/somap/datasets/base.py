"""Base class for SOM datasets."""

from dataclasses import dataclass

import numpy as np


@dataclass
class SomDataset:
    """Dataset structure."""

    data: np.ndarray
    nb: int  # Number of data points
    shape: tuple  # Shape of each data point
