"""Catalog of metrics functions.

Used to compute different kind of SOM data-independent errors.

The classical quantization and topographic errors are data-dependent.
They are directly computed in the SOM algorithm.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .distance import dist_l2
from .grid import distance_map
from .neighborhood import nbh_static_kohonen


def intrinsic_topographic_error(
    model,
    f_dist: Callable = dist_l2,
    sigma: float = 1,
    topography=True,
    borderless=True,
):
    """Evaluates how similar each node prototype are with their neighbor prototypes.

    All the info is intrinsic to the model: it does not depend on a given input.
    A small "intrinsic topographic error" means that all neighbouring nodes are similar.

    Args:
        model: The current Adaptative Map and its state.
        f_dist: the distance function used.
        sigma: the radius of the distance function of the static kohonen map.
        topography: Topography of the 2D map, either 'square' or 'hex'.
        borderless: if True, the borders of the map loop, else the map is finite.

    Returns:
        All the errors computed.
    """
    w_bu = model["w_bu"]
    shape = (w_bu.shape[0], w_bu.shape[1])

    # Vectorize over the 2D grid both time for a cartesian product
    f_dist = jax.vmap(jax.vmap(f_dist, in_axes=(0, None)), in_axes=(0, None))
    f_dist = jax.vmap(jax.vmap(f_dist, in_axes=(None, 0)), in_axes=(None, 0))

    all_distances = f_dist(w_bu, w_bu)

    # Generate a list of all XY coordinates
    x, y = np.meshgrid(
        np.arange(shape[0], dtype=np.int16),
        np.arange(shape[1], dtype=np.int16),
        sparse=False,
        indexing="ij",
    )

    xx = x[:, :, np.newaxis]
    yy = y[:, :, np.newaxis]

    coords = np.concatenate((xx, yy), axis=2)  # coords.shape = (shape[0], shape[1], 2)

    # Vectorize the neighbourhood functions over all XY coordinates
    @jax.vmap
    @jax.vmap
    def v_nbh_euclidian_dist(winner_coords):
        return distance_map(
            shape, winner_coords, topography=topography, borderless=borderless
        )

    @jax.vmap
    @jax.vmap
    def v_neighbourhood_static_kohonen(hexgrid_distance_from_winner):
        return nbh_static_kohonen(hexgrid_distance_from_winner, sigma)

    # Compute the neighbourhood values
    nbh_euclidian_distances = v_nbh_euclidian_dist(coords)
    nbh_values = v_neighbourhood_static_kohonen(nbh_euclidian_distances)

    # Compute the topographic error of all XY coordinates
    errors = jnp.sum(all_distances * nbh_values, axis=(2, 3))

    return errors
