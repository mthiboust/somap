"""Import the selected plot backend.

The plot backend can be specified via the `SOMAP_PLOT_BACKEND` 
environment varialbe. For example: `SOMAP_PLOT_BACKEND=altair`.
"""

import importlib
import logging
import os

from ..config import _DEFAULT_PLOT_BACKEND, _ENV_VAR_PLOT_BACKEND, _SUPPORTED_BACKENDS


def set_plot_backend(name):
    """Sets the plot backend environment variable."""
    os.environ[_ENV_VAR_PLOT_BACKEND] = name


def import_backend(name=None):
    """Returns the selected plot backend module."""
    PLOT_BACKEND = os.getenv(_ENV_VAR_PLOT_BACKEND, _DEFAULT_PLOT_BACKEND)

    if PLOT_BACKEND not in _SUPPORTED_BACKENDS:
        logging.error(
            f"The detected plot backend `{PLOT_BACKEND}` is not supported. "
            f"Verify if the `SOMAP_PLOT_BACKEND` environment variable is set correctly."
            f" Supported backends are `{'`, `'.join(_SUPPORTED_BACKENDS)}`. "
            f"Falling back to the default backend `{_DEFAULT_PLOT_BACKEND}`."
        )
        PLOT_BACKEND = _DEFAULT_PLOT_BACKEND

    plot_backend = PLOT_BACKEND if name is None else name
    try:
        backend = importlib.import_module(f".{plot_backend}", __package__)
        logging.info(f"Using the `{plot_backend}` plot backend.")
        return backend

    except ImportError as e:
        logging.fatal(f"Failed to import the `{plot_backend}` plot backend. ")
        raise e
