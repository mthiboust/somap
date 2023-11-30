"""Catalog of update functions."""

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from .utils import experimental_warning


class AbstractUpdate(eqx.Module):
    """Ensures that all update functions have the same signatures."""

    @abstractmethod
    def __call__(self, lr, nbh, input_bu, w_bu):
        """SOM Update function.

        Args:
            lr: Learning rate.
            nbh: Neighborhood.
            input_bu: Data input.
            w_bu: Prototype weights.

        Returns:
            The updated prototype weights.
        """
        pass


class SomUpdate(AbstractUpdate):
    """Generic update function."""

    def __call__(
        self,
        lr: Float[Array, "..."],
        nbh: Float[Array, "..."],
        input_bu: Float[Array, "..."],
        w_bu: Float[Array, "..."],
    ) -> Float[Array, "..."]:
        """Updates the prototype weights."""
        out = w_bu + (lr * nbh)[:, :, jnp.newaxis] * (input_bu - w_bu)
        return jnp.clip(out, 0, 1.0)


@experimental_warning
class CyclicSomUpdate(AbstractUpdate):
    """Cyclic update functions."""

    def __call__(self, lr, nbh, input_bu, w_bu):
        """Updates the prototype weights where 0 is the same as 1."""
        # convert [0:1] float range into [0:255] integer range
        input_bu_uint8 = (input_bu * 255).astype(jnp.uint8)
        w_bu_uint8 = (w_bu * 255).astype(jnp.uint8)

        # interpret the diff as an unsigned int8
        # diff is between -1 and 1
        diff = (input_bu_uint8 - w_bu_uint8).astype(jnp.int8) / 128.0

        # out may escape the [0:1] interval, so it needs a modulo 1
        out = (lr * nbh)[:, :, jnp.newaxis] * diff + w_bu
        out = out % 1

        return out
