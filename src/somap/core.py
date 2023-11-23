"""Abstract base classes for defining SOMs."""

from abc import abstractstaticmethod
from collections.abc import Callable
from typing import TypedDict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Integer, PRNGKeyArray

from .grid import distance_map
from .utils import filter_scan


class InputData(TypedDict):
    """Structure of the input data.

    Note:
        Classical SOMs only have the 'bu_v' bottum-up input value.
        Other inputs allow to create more complex SOMs receiving top-down and lateral
        inputs with a mask.

    Note bis:
        TypedDict instead of dataclass to facilitate future modifications.
    """

    bu_v: Float[Array, " x"]  # Bottom-up input value
    bu_m: Float[Array, " x"] | None  # Bottom-up input mask
    td_v: Float[Array, "2"] | None  # Top-down input value
    td_m: Float[Array, "2"] | None  # Top-down input mask
    lat_v: Float[Array, " y"] | None  # Lateral input value
    lat_m: Float[Array, " y"] | None  # Lateral input mask


class AbstractSomParams(eqx.Module):
    """Abstract base class for SOM parameters."""


class SomAlgo(eqx.Module):
    """Generic SOM functions."""

    f_dist: Callable
    f_nbh: Callable
    f_lr: Callable
    f_update: Callable


class AbstractSom(eqx.Module):
    """Abstract base class for SOM models."""

    shape: tuple
    topography: str
    borderless: bool
    input_shape: tuple
    params: AbstractSomParams
    metrics: bool = True
    debug: bool = False
    algo: SomAlgo = eqx.field(init=False)
    in_size: int = eqx.field(init=False)
    t: Integer[Array, ""] = eqx.field(init=False)
    w_bu: Float[Array, "x y ..."] = eqx.field(init=False)  # same as "prototype weights"
    i_act_nb: Integer[Array, "x y"] = eqx.field(init=False)
    winner: Integer[Array, "2"] = eqx.field(init=False)

    def __init__(
        self,
        shape,
        topography,
        borderless,
        input_shape,
        params,
        metrics=True,
        debug=False,
        key: PRNGKeyArray = jax.random.PRNGKey(0),
    ):
        """Creates a SOM models.

        Args:
            shape: Shape of the 2D map.
            topography: Topography of the 2D map. Either 'square' for a square grid
                or 'hex' for hexagonal grid.
            borderless: Toroidal topography if True, meaning that the top (resp. left)
                border meets the bottom (resp. right) border.
            input_shape: Shape of the input data.
            params: Parameters of the SOM (depends on the SOM flavor).
            metrics: If True, returns quantization and topographic errors as auxilary
                data.
            debug: If True, returns debug data as auxilary data.
            key: JAX random key used during map initialization.
        """
        self.shape = shape
        self.topography = topography
        self.borderless = borderless
        self.input_shape = input_shape
        self.params = params
        self.metrics = metrics
        self.debug = debug

        self.in_size = int(np.prod(self.input_shape))
        self.t = jnp.array(0, dtype=jnp.int32)
        self.w_bu = jax.random.uniform(
            key, (self.shape + self.input_shape), dtype=jnp.float32
        )
        self.i_act_nb = jnp.zeros(self.shape, dtype=jnp.int32)
        self.algo = self.generate_algo(params)
        self.winner = jnp.zeros((2,), dtype=jnp.int32)

    @abstractstaticmethod
    def generate_algo(params: AbstractSomParams) -> SomAlgo:
        """Converts specific SOM parameters into generic SOM functions."""
        raise NotImplementedError

    def __call__(self, input: InputData):
        """Makes a single iteration.

        Args:
            input: Data array for the given SOM models.

        Returns:
            A tuple with the new SOM model and the auxilary data.
        """
        input_bu_v = input["bu_v"].reshape(self.in_size)
        w_bu = self.w_bu.reshape((self.shape) + (self.in_size,))

        # Compute distances over the 2D grid
        f_dist = jax.vmap(
            jax.vmap(self.algo.f_dist, in_axes=(0, None)), in_axes=(0, None)
        )
        dist = f_dist(w_bu, input_bu_v)

        # Find the coordinates of the node with the minimal distance
        x, y = jnp.unravel_index(jnp.argmin(dist), dist.shape)
        winner = jnp.array([x, y])

        if self.metrics:
            # Find the second winner (replace the winner by a high value,
            # then compute the `argmin` which is faster than a whole `argsort`)
            x2, y2 = jnp.unravel_index(
                jnp.argmin(dist.at[x, y].set(dist.max())), dist.shape
            )

        # Compute the neighbourhood 2D grid values
        d = distance_map(self.shape, winner, self.topography, self.borderless)
        nbh = self.algo.f_nbh(d, self.t, dist[x, y])

        # Compute the learning rate for the 2D grid
        lr = self.algo.f_lr(self.t, dist)

        # Update model parameters
        new_self = self.bulk_set(
            {
                "w_bu": self.algo.f_update(lr, nbh, input_bu_v, w_bu).reshape(
                    (self.shape + self.input_shape)
                ),
                "i_act_nb": self.i_act_nb.at[x, y].set((1 + self.i_act_nb[x, y])),
                "t": self.t + 1,
                "winner": winner,
            }
        )

        aux = {}
        if self.metrics:
            aux["metrics"] = {
                "quantization_error": dist[x, y],
                "topographic_error": d[x2, y2],
            }
        if self.debug:
            aux["debug"] = {
                "dist": dist,
                "nbh": nbh,
                "lr": lr,
            }

        return new_self, aux

    def set(self, attribute: str, value):
        """Sets an attribute to a specific value.

        Args:
            attribute: name of the attribute.
            value: new value of the attribute.

        Returns:
            A new instance of the updated object.
        """
        return eqx.tree_at(lambda s: s.__getattribute__(attribute), self, value)

    def bulk_set(self, attr_dict):
        """Sets multiples attributes at once.

        Args:
            attr_dict: dictionary where keys are attribute names and values are
                attributes values to be set.

        Returns:
            A new instance of the updated object.
        """

        def _f(module):
            return [module.__getattribute__(key) for key in attr_dict.keys()]

        return eqx.tree_at(_f, self, attr_dict.values())


@eqx.filter_jit
def make_step(model: AbstractSom, input: InputData):
    """Makes a single iteration.

    Args:
        model: SOM model.
        input: Data array for the given SOM models.

    Returns:
        A tuple with the new SOM model and the auxilary data.
    """
    input = jax.tree_map(lambda x: jnp.asarray(x), input)
    return model(input)


@eqx.filter_jit
def make_steps(model: AbstractSom, inputs):
    """Makes multiple iterations at once.

    Uses the `jax.lax.scan()` function to optimize computations.

    Args:
        model: SOM model.
        inputs: Batch data array for the given SOM models.

    Returns:
        A tuple with the new SOM model and the auxilary data.
    """
    inputs = jax.tree_map(lambda x: jnp.asarray(x), inputs)
    return filter_scan(make_step, model, inputs)
