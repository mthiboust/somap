"""Catalog of distance functions.

`dist` means distance
`wdist` means weighted distance

All distance functions are defined for comparison between 2 1D vectors.
"""


from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from .utils import experimental_warning


class AbstractDist(eqx.Module):
    """Ensures that all distance functions have the same signatures."""

    @abstractmethod
    def __call__(
        self, a: Float[Array, " n"], b: Float[Array, " n"]
    ) -> Float[Array, ""]:
        """SOM distance function.

        Args:
            a: 1D array.
            b: 1D array.

        Returns:
            The distance as a scalar.
        """
        pass


class EuclidianDist(AbstractDist):
    """Euclidian distance function."""

    def __call__(
        self, a: Float[Array, " n"], b: Float[Array, " n"]
    ) -> Float[Array, ""]:
        """Computes the euclidian norm."""
        return jnp.linalg.norm(a - b).mean()


@experimental_warning
class CyclicEuclidianDist(AbstractDist):
    """Cyclic euclidian distance function."""

    def __call__(
        self, a: Float[Array, " n"], b: Float[Array, " n"]
    ) -> Float[Array, ""]:
        """Returns a cyclic euclidian distance where 0 and 1 are equals.

        More optimized than the `arccos(cos())` based cyclic distance.
        Takes advantage of data type's limits.
        """
        a = (a * 255).astype(jnp.uint8)
        b = (b * 255).astype(jnp.uint8)
        diff = (a - b).astype(jnp.int8)
        return (jnp.linalg.norm(diff) / 128.0).mean()


@experimental_warning
def wdist_l2(
    a: Float[Array, " n"], b: Float[Array, " n"], w: Float[Array, " n"]
) -> Float[Array, ""]:
    """Computes a weighted version of the euclidian norm."""
    q = a - b
    return jnp.sqrt((w * q * q).sum())


@experimental_warning
def dist_cim(v1, v2, sigma):
    """The Correntropy Induced Metric (CIM) distance.

    It is based on correntropy which is a generalized correlation.
    It computes the distance between two vectors without suffering from the curse of
    dimensionality like the Euclidian distance thanks to a kernel function.
    The chosen kernel is a Gaussian function (parameterised by 'sigma').
    For large sigma, the CIM distance tends to the L2 metric.
    For very small sigma, the CIM distance tends to the L0 norm.

    Args:
        v1: first vector to be compared
        v2: second vector to be compared
        sigma: kernel bandwidth

    Returns:
        A scalar value between 0 and 1 corresponding to the similarity between v1 and v2

    Raises:
        ValueError: incompatible shapes of v1 and v2
    """
    if v1.ndim != 1 or v2.ndim != 1 or v1.shape != v2.shape:
        raise ValueError("v1 and v2 should be 1-D vectors of the same length")

    x = v1 - v2

    kernel = jnp.exp(-(x**2) / (2 * sigma**2))
    correntropy = jnp.mean(kernel)

    return jnp.sqrt(1 - correntropy)
