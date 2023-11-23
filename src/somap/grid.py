"""Grid topography functions."""

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Integer


def meshgrid(
    shape: tuple[int, int],
    center: Integer[Array, "2"],
    topography: str = "hex",
    borderless: bool = True,
) -> tuple[Float[Array, "x y"], Float[Array, "x y"]]:
    """Return a 2D meshgrid of a given shape centered at a given (x,y) coordinate.

    When using hexagonal topography, you can choose shape[0] ~ sin(60°) * shape[1]
    for a square-like size.

    You can see the grid arrangement in matplotlib:
        ```python
        from matplotlib import pyplot as plt
        import jax.numpy as jnp
        import somap as smp

        xv, yv = smp.grid.meshgrid((11, 13), jnp.array([5, 6]))
        fig, ax = plt.subplots()
        ax.scatter(xv, yv)
        plt.axvline(x = 0)
        plt.axhline(y = 0)
        plt.gca().set_aspect('equal', adjustable='box')
        ```

    Note:
        The use of 'numpy' operators instead of 'jax.numpy' operators prevent this
        computation to be completely performed on an accelerator (e.g. GPU).
        But I don't know how to translate 'np.arange(size_x)' into a jit'able
        'jax.numpy' operation ('jnp.arange(size_x)' won't work because of the
        size-dependency of the result).

    Args:
        shape: Shape of the grid.
        center: Position of the (0,0) coordinate.
        topography: Type of topography: 'hex' for hexagonal lattice, 'square' for
            square lattice.
        borderless: If True, the grid topography is toroidal. It means that the left
        (resp. top) border is linked to the right (resp. bottom) one.

    Returns:
        Two 2D arrays representing the x-coordinates and y-coordinates of the meshgrid.
    """
    size_x, size_y = shape
    center_x, center_y = center[0], center[1]

    if borderless:
        roll_offset_x, roll_offset_y = (center_x, center_y)
        center_x, center_y = (size_x // 2, size_y // 2)
    else:
        roll_offset_x, roll_offset_y = (0, 0)

    xv, yv = np.meshgrid(
        np.arange(size_x, dtype=np.float32),
        np.arange(size_y, dtype=np.float32),
        sparse=False,
        indexing="ij",
    )

    if topography == "square":
        xx, yy = (jnp.array(xv) - center_x, jnp.array(yv) - center_y)

    elif topography == "hex":
        ratio = (np.sqrt(3) / 2).astype(np.float32)
        xv = xv / ratio
        xv[:, ::2] += 1 - ratio / 2

        xx, yy = (
            jnp.array(xv)
            - (center_x + 1 / 2 * ((center_y + roll_offset_y + 1) % 2)) / ratio,
            jnp.array(yv) - center_y,
        )

    else:
        raise ValueError(
            f"Topography must be either 'square' or 'hex' (not {topography})"
        )

    if borderless:
        xx = jnp.roll(xx, roll_offset_x - size_x // 2, axis=0)
        yy = jnp.roll(yy, roll_offset_y - size_y // 2, axis=1)

    return xx, yy


def distance_map(
    shape: tuple[int, int],
    winner: Integer[Array, "2"],
    topography: str = "hex",
    borderless: bool = True,
) -> Float[Array, "x y"]:
    """Return the euclidian distance of each grid coordinate from the winner position.

    Results are normalized so that the max distance is around 1.

    Args:
        shape: Shape of the grid.
        winner: Coordinates of the winner.
        topography: Type of topography: 'hex' for hexagonal lattice, 'square' for
            square lattice.
        borderless: If True, the grid topography is toroidal. It means that the left
            (resp. top) border is linked to the right (resp. bottom) one.

    Returns:
        A 2D array.
    """
    xx, yy = meshgrid(shape, winner, topography=topography, borderless=borderless)
    zz = jnp.sqrt(xx**2 + yy**2)

    d_y = shape[1] - 1
    if topography == "hex":
        d_x = (shape[0] - 1) * (jnp.sqrt(3) / 2)
    else:
        d_x = shape[0] - 1

    max_distance = jnp.sqrt(d_x**2 + d_y**2)

    if borderless:
        max_distance /= 2

    return zz / max_distance
