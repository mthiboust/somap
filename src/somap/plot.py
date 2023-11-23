"""Plot functions for somap.Som objects."""

from typing import IO

from .core import AbstractSom
from .plot_backends import import_backend


def plot(som: AbstractSom, *args, **kwargs):
    """Returns a graph/plot object of the SOM.

    Extra args depend on the chosen backend.
    See `.plot_backends.{name_of_backend}` for details.

    Args:
        som: Self-Organizing Map as a Som object.
        args: Positional arguments passed to the `SomPlot` backend.
        kwargs: Keyword arguments passed to the `SomPlot` backend.
    """
    backend = import_backend()
    som_plot = backend.SomPlot(som, *args, **kwargs)
    return som_plot.plot()


def save_plot(som: AbstractSom, filename: str | IO, *args, **kwargs):
    """Saves a graph/plot image of the SOM.

    Args:
        som: Self-Organizing Map as a Som object.
        filename: File in which to write the chart.
        args: Positional arguments passed to the `SomPlot` backend.
        kwargs: Keyword arguments passed to the `SomPlot` backend.
    """
    backend = import_backend()
    som_plot = backend.SomPlot(som, *args, **kwargs)
    som_plot.save(filename)
