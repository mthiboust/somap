"""MNIST datasets."""

import numpy as np
from datasets import load_dataset

from .base import SomDataset
from .utils import load_data_with_caching


class MNIST(SomDataset):
    """Classic MNIST.

    60k grayscale 28x28 images of digits from 0 to 9.
    """

    def __init__(self):
        """Loads the classic MNIST dataset."""

        def _load_func():
            dataset = load_dataset("mnist")
            return np.asarray(dataset["train"]["image"]) / 255.0

        self.data = load_data_with_caching("mnist.npy", _load_func)
        self.nb = self.data.shape[0]
        self.shape = self.data.shape[1:]


class Digits(SomDataset):
    """Kind of mini-MNIST.

    ~2k grayscale 8x8 images of digits from 0 to 9.
    """

    def __init__(self):
        """Loads the mini-MNIST dataset from scikit-learn."""

        def _load_func():
            dataset = load_dataset("sklearn-docs/digits", header=None)
            return np.asarray(dataset["train"].data).T[:, :64].reshape(-1, 8, 8) / 16.0

        self.data = load_data_with_caching("digits.npy", _load_func)
        self.nb = self.data.shape[0]
        self.shape = self.data.shape[1:]
