"""Catalog of neighborhood functions.

Neighborhood functions are defined as `equinox.Module` parametrized functions 
"""

from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer


class AbstractNbh(eqx.Module):
    """Ensures that all neighborhood functions have the same signatures."""

    @abstractmethod
    def __call__(
        self,
        distance_map: Float[Array, "x y"],
        t: Integer[Array, ""],
        quantization_error: Float[Array, ""],
    ) -> Float[Array, "x y"]:
        """SOM Neighborhood function.

        Args:
            distance_map: Distance of each grid elements from the winning element.
            t: Current iteration.
            quantization_error: The computed difference between the winner prototype
                and the input.

        Returns:
            The neighborhood distance.
        """
        pass


class GaussianNbh(AbstractNbh):
    """Exponentially decreasing neighborhood function."""

    sigma: float | Float[Array, "..."] = 0.1

    def __call__(self, distance_map: Float[Array, "x y"], t, __) -> Float[Array, "x y"]:
        """Return the Kohonen time-independent neighboring value of each element.

        Args:
            self: Module's parameters
                self.sigma: Neighbourhood distance.
            distance_map: Distance of each element from the winner element.
            t: Not used
            __: Not used

        Returns:
            The kohonen neighborhood distance.
        """
        # Surprisingly, `(t+1)/(t+1)` speeds up the jitted function on CPU with JAX.
        sigma = self.sigma * (t + 1) / (t + 1)
        return jnp.exp(-(distance_map**2) / (2 * sigma**2))


class KsomNbh(AbstractNbh):
    """Kohonen neighborhood function."""

    t_f: int | Integer[Array, "..."] = 100000
    sigma_i: float | Float[Array, "..."] = 1.0
    sigma_f: float | Float[Array, "..."] = 0.01

    def __call__(
        self, distance_map: Float[Array, "x y"], t: Integer[Array, ""], _
    ) -> Float[Array, "x y"]:
        """Returns the Kohonen neighboring value of each element.

        Args:
            self: Module's parameters
                self.t_f: Aimed iteration.
                self.sigma_i: Current neighborhood distance.
                self.sigma_f: Aimed neighborhood distance.
            distance_map: Distance of each grid elements from the winning element.
            t: Current iteration.
            _: Not used

        Returns:
            The kohonen neighborhood distance.
        """
        sigma = self.sigma_i * (self.sigma_f / self.sigma_i) ** (t / self.t_f)
        return jnp.exp(-(distance_map**2) / (2 * sigma**2))


class DsomNbh(AbstractNbh):
    """Dynamic Kohonen neighborhood function."""

    plasticity: float | Float[Array, "..."] = 0.1

    def __call__(
        self, distance_map: Float[Array, "x y"], _, quantization_error: Float[Array, ""]
    ) -> Float[Array, "x y"]:
        """Computes the Dynamic SOM neighboring value of each grid element.

        See:
            Nicolas P. Rougier, Yann Boniface. Dynamic Self-Organising Map.
            Neurocomputing, Elsevier, 2011, 74 (11), pp.1840-1847.
            ff10.1016/j.neucom.2010.06.034ff. ffinria-00495827

        Args:
            self:
                self.plasticity: Dynamic value to compute the neighbourhood distance.
            distance_map: Distance of each element from the winner element.
            _: Not used
            quantization_error: The computed difference between the winner prototype
                and the input.

        Returns:
            The neighborhood distance, as calculated in the article.
        """
        return jnp.exp(
            -(distance_map**2) / (self.plasticity**2 * quantization_error**2)
        )


class MexicanHatNbh(AbstractNbh):
    """Mexican Hat neighborhood function."""

    sigma: float | Float[Array, "..."] = 0.1

    def __call__(self, distance_map: Float[Array, "x y"], _, __) -> Float[Array, "x y"]:
        """Computes the Mexican Hat neighboring value of each grid element.

        Args:
            self:
                self.sigma: Scale factor for the spread of the neighborhood.
            distance_map: Distance of each element from the winner element.
            _: Not used
            __: Not used

        Returns:
            The Mexican Hat neighborhood distance.
        """
        r2_norm = distance_map**2 / self.sigma**2
        return (1 - 0.5 * r2_norm) * jnp.exp(-r2_norm / 2)
