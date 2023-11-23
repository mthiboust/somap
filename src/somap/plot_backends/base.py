"""Abstract base class for plot backends."""

from abc import ABC, abstractmethod


class AbstractSomPlot(ABC):
    """SomPlot."""

    @abstractmethod
    def __init__(self, som, *, show_prototypes=True, show_activity=False, **kwargs):
        """Creates a SomPlot.

        Args:
            som: Self-Organizing Map as a Som object.
            args: Positional arguments passed to the `SomPlot` backend.
            show_prototypes: Show prototypes of each node as an image.
            show_activity: Show activity value of each node as a color.
            kwargs: Keyword arguments passed to the `SomPlot` backend.
        """
        raise NotImplementedError

    @abstractmethod
    def plot(self):
        """Returns the plot chart."""
        raise NotImplementedError

    @abstractmethod
    def save(self):
        """Saves the plot as an image."""
        raise NotImplementedError
