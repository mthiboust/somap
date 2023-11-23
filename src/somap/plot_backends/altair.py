"""Altair plot backends."""


from typing import IO

import altair as alt
import numpy as np
import pandas as pd
from array2image import array_to_image

from ..core import AbstractSom
from .base import AbstractSomPlot
from .utils import image_to_data_url


def _encode_image(array, **kwargs):
    return image_to_data_url(array_to_image(array, **kwargs))


def _encode_plot_data(values, images, **kwargs) -> pd.DataFrame:
    """Transforms data into a Pandas dataframe usable by an Altair chart.

    All arguments should be 2D array-like (python list, numpy array or jax array).
    The first 2 dimensions of each argument should match.

    Args:
        values: 2D array (a scalar for each 2D elements).
        images: 4D array (a 2D image for each 2D elements).
        kwargs: Keyword arguments of `somap.plot_backends.utils.array_to_image()`.

    Returns:
        The encoded plot data.
    """
    values = np.asarray(values)
    images = np.asarray(images)

    dim_x, dim_y = values.shape

    # Create a Dataframe with 'x', 'y' columns
    r, c = np.indices(values.shape)
    df = pd.DataFrame(np.c_[r.ravel(), c.ravel()], columns=(["x", "y"]))

    # Add 'value' and 'image' columns
    df["value"] = values.T.ravel("F")
    df["image"] = [
        _encode_image(images[i][j], **kwargs)
        for i in range(dim_x)
        for j in range(dim_y)
    ]  # 'image' is an `altair` keyword for tooltip images

    return df


def plot(
    data: pd.DataFrame,
    shape: tuple[int, int],
    topography: str,
    *,
    bin_size_scale: float = 1.0,
    w_scale: float = 1.0,
    h_scale: float = 1.0,
    values: bool = True,
    images: bool = True,
    tooltip: bool = True,
    color_scale: alt.Scale = alt.Scale(scheme="lighttealblue"),
    color_legend: alt.Legend | None = None,
    title: str | None = None,
) -> alt.LayerChart:
    """Return an Altair chart containing shape[0] * shape[1] hexagons.

    Args:
        data: Pandas dataframe with the following columns:
            x: Horizontal coordinate in the hex grid.
            y: Vertical coordonates in the hex grid.
            values: one scalar value per hexagon (optional).
            images: one encoded image per hexagon (optional).
            image: one encoded image per hexagon (for tooltip) (optional).
            infos: one text value per hexagon(optional).
        shape: column and line numbers of the hex grid.
        topography: Topography of the 2D map, either 'square' or 'hex'.
        bin_size_scale: size of the hexagons.
        w_scale: horizontal scaling factor (sometimes needed to go around rendering
            bugs of external libs).
        h_scale: vertical scaling factor (sometimes needed to go around rendering bugs
            of external libs).
        values: if True, fill the hexagons with a color corresponding to data['values'].
        images: if True, show the image of all hexagon with an image corresponding to
            data['images'].
        tooltip: if True, show an interactive tooltip given detailed values of the
            selected hexagon
        color_scale: color scale used if color is True. Continous scale from light blue
            to dark blue by default.
        color_legend:arr color legend used if color is True. Dynamic legend by default.
            (set to 'None' to hide the legend).
        title: Title of the plot.

    Returns:
        The altair chart.
    """
    if values and "value" not in data.columns:
        raise ValueError(
            "Chart data must contain the 'values' field to fill colors. \
            If you don't want to fill colors, use the `color=False` argument"
        )

    if images and "image" not in data.columns:
        raise ValueError(
            "Chart data must contain the 'images' field to show images. \
            If you don't want to show images, use the `images=False` argument"
        )

    x_nb, y_nb = shape

    if topography == "hex":
        marker_shape = "M0,-2.3094L2,-1.1547 2,1.1547 0,2.3094 -2,1.1547 -2,-1.1547Z"
        bin_size = int(22 * bin_size_scale)
        w_scale = 2 * w_scale
        h_scale = 1.6 * h_scale
    elif topography == "square":
        marker_shape = "M 1 -1 L 1 1 L -1 1 L -1 -1 Z"
        bin_size = int(40 * bin_size_scale)
        w_scale = 0.9 * w_scale
        h_scale = 0.9 * h_scale
    else:
        raise ValueError(
            f"Topography must be either 'square' or 'hex' (not {topography})"
        )

    _axis = alt.Axis(title="", labels=False, grid=False, tickOpacity=0, domainOpacity=0)

    chart = (
        alt.Chart(data)
        .encode(
            x=alt.X("x:Q", axis=_axis),
            y=alt.Y("y:Q", axis=_axis),
            stroke=alt.value("black"),
            strokeWidth=alt.value(0.2),
        )
        .properties(width=w_scale * bin_size * x_nb, height=h_scale * bin_size * y_nb)
    )

    if values:
        chart = chart.encode(
            fill=alt.Color("value:Q", scale=color_scale, legend=color_legend)
        )

    if images:
        chart = chart.encode(url="image")

    if tooltip:
        chart = chart.encode(
            tooltip=["x", "y"]
            + ["image"] * ("image" in data.columns)
            + ["value:Q"] * ("value" in data.columns)
        )

    if topography == "hex":
        chart = chart.encode(x=alt.X("xPos:Q", axis=_axis))
        chart = chart.transform_calculate(
            xPos="(datum.x + 1 / 2 * ((datum.y + 1) % 2))"
        )

    if title is not None:
        chart = chart.properties(title=title)

    c = alt.layer(
        chart.mark_point(size=bin_size**2, shape=marker_shape),
        chart.mark_image(width=25, height=25),
    ).configure_view(strokeWidth=0)

    return c


class SomPlot(AbstractSomPlot):
    """SomPlot."""

    def __init__(
        self,
        som: AbstractSom,
        *,
        show_prototypes=True,
        show_activity=False,
        **kwargs,
    ):
        """Creates a SomPlot.

        Args:
            som: Self-Organizing Map as a Som object.
            args: Positional arguments passed to the `SomPlot` backend.r.ravel()
            show_prototypes: Show prototypes of each node as an image.
            show_activity: Show activity value of each node as a color.
            kwargs: Keyword arguments passed to the `SomPlot` backend.
        """
        _images = som.w_bu.reshape(som.shape + som.input_shape)
        _values = som.i_act_nb

        kwargs1 = {k[4:]: v for k, v in kwargs.items() if k.startswith("img_")}
        kwargs2 = {k[4:]: v for k, v in kwargs.items() if k.startswith("plt_")}

        chart_data = _encode_plot_data(images=_images, values=_values, **kwargs1)
        self.chart = plot(
            chart_data,
            som.shape,
            som.topography,
            images=show_prototypes,
            values=show_activity,
            **kwargs2,
        )

    def plot(self):
        """Returns the plot chart."""
        return self.chart

    def save(self, filename: str | IO, *args, **kwargs):
        """Saves the plot as an image."""
        self.chart.save(filename)
