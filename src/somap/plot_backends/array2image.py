"""Array2image plot backends."""

from pathlib import Path

from array2image import array_to_image
from PIL import Image, ImageDraw, ImageFont

from ..core import AbstractSom
from .base import AbstractSomPlot


current_file_path = Path(__file__).resolve().parent
FONT_PATH = current_file_path / "resources/Carlito-Regular.ttf"


def add_text_border(image, text, border_height=50, font_size=20, font_path=FONT_PATH):
    """Adds a header with a text to an image."""
    # Original image size
    width, height = image.size

    # New image size (with border)
    new_height = height + border_height
    new_image = Image.new("RGB", (width, new_height), "white")

    # Paste the original image onto the new image
    new_image.paste(image, (0, border_height))

    # Draw the text
    draw = ImageDraw.Draw(new_image)
    font = ImageFont.truetype(str(font_path), 14)

    draw.text((0, 0), text, font=font, fill="black")

    return new_image


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
        if som.topography != "square":
            raise ValueError(
                f"The 'array2image' plot backend only support the "
                f"'square' topography, not the '{som.topography}'."
            )

        data = som.w_bu.reshape(som.shape + som.input_shape)
        self.image = array_to_image(data, **kwargs)
        self.image = add_text_border(self.image, f"{som.params}")

    def plot(self):
        """Returns the plot chart."""
        return self.image

    def save(self, filename: str, *args, **kwargs):
        """Saves the plot as an image."""
        self.image.save(filename, *args, **kwargs)
