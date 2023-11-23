"""Catalog of learning rate functions."""

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer


class AbstractLr(eqx.Module):
    """Ensures that all learning rate functions have the same signatures."""

    @abstractmethod
    def __call__(
        self, t: Integer[Array, ""], distances: Float[Array, "x y"]
    ) -> Float[Array, ""] | Float[Array, "x y"]:
        """SOM learning rate function.

        Args:
            t: Current iteration.
            distances: Distances between the prototype weights and the input data.

        Returns:
            A scalar ar 2D array containing the learning rate.
        """
        pass


class ConstantLr(AbstractLr):
    """Basic SOM learning rate function."""

    alpha: float | Float[Array, "..."] = 0.01

    def __call__(self, _, __) -> Float[Array, ""]:
        """Returns the static SOM learning rate."""
        return jnp.array(self.alpha)


class KsomLr(AbstractLr):
    """Kohonen SOM learning rate function."""

    t_f: int | Integer[Array, "..."] = 100000
    alpha_i: float | Float[Array, "..."] = 0.01
    alpha_f: float | Float[Array, "..."] = 0.001

    def __call__(self, t: Integer[Array, ""], _) -> Float[Array, ""]:
        """Returns the Kohonen SOM learning rate."""
        return self.alpha_i * (self.alpha_f / self.alpha_i) ** (t / self.t_f)


class DsomLr(AbstractLr):
    """DSOM learning rate function."""

    alpha: float | Float[Array, "..."] = 0.001

    def __call__(self, _, distances: Float[Array, "x y"]) -> Float[Array, "x y"]:
        """Returns the DSOM learning rate."""
        return self.alpha * distances
