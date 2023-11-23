"""Utility functions for plotting SOMs."""

import base64
from io import BytesIO

from PIL import Image


def image_to_data_url(img: Image, img_format: str = "PNG") -> str:
    """Converts a PIL image to an image encoded as a Data URL.

    Web-based plotting libraries relying on Javascript often need images encoded as a
    Data URLs.

    Args:
        img: PIL image.
        img_format: Format of the image. Default to 'PNG'.

    Returns:ata_ur
        A Data URL string of the base64-encoded image binary content.
    """
    buffered = BytesIO()
    img.save(buffered, format=img_format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8").replace("\n", "")
    data_url = f"data:image/{img_format.lower()};base64,{img_str}"

    return data_url
