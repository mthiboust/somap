"""Catalog of different flavors of SOMs.

To add a new SOMs, define the 2 following classes:
```python
class MySomParams(AbstractSomParams):
    pass
    
class MySom(AbstractSom):
    pass
```
"""

from jaxtyping import Array, Float, Integer

from .core import AbstractSom, AbstractSomParams, SomAlgo
from .distance import EuclidianDist
from .learning_rate import ConstantLr, DsomLr, KsomLr
from .neighborhood import DsomNbh, GaussianNbh, KsomNbh
from .update import SomUpdate


class SomParams(AbstractSomParams, SomAlgo):
    """Generic SOM parameters.

    Same as the `SomAlgo` class but with the `AbstractSomParams` parent.
    """

    pass


class Som(AbstractSom):
    """Generic SOM.

    Distance, neighborhood, learning rate and update functions are directly specified.
    This is the more flexible way of defining a SOM.
    """

    @staticmethod
    def generate_algo(p: SomParams) -> SomAlgo:
        """Identity function returning generic SOM functions."""
        return SomAlgo(
            f_dist=p.f_dist,
            f_nbh=p.f_nbh,
            f_lr=p.f_lr,
            f_update=p.f_update,
        )


class KsomParams(AbstractSomParams):
    """Kohonen SOM parameters."""

    t_f: int | Integer[Array, "..."]
    sigma_i: float | Float[Array, "..."]
    sigma_f: float | Float[Array, "..."]
    alpha_i: float | Float[Array, "..."]
    alpha_f: float | Float[Array, "..."]


class Ksom(AbstractSom):
    """Kohonen SOM."""

    @staticmethod
    def generate_algo(p: KsomParams) -> SomAlgo:
        """Converts Kohonen SOM parameters into generic SOM functions."""
        return SomAlgo(
            f_dist=EuclidianDist(),
            f_nbh=KsomNbh(t_f=p.t_f, sigma_i=p.sigma_i, sigma_f=p.sigma_f),
            f_lr=KsomLr(t_f=p.t_f, alpha_i=p.alpha_i, alpha_f=p.alpha_f),
            f_update=SomUpdate(),
        )


class StaticKsomParams(AbstractSomParams):
    """Time-independant Kohonen SOM parameters."""

    sigma: float | Float[Array, "..."]
    alpha: float | Float[Array, "..."]


class StaticKsom(AbstractSom):
    """Time-independant Kohonen SOM."""

    @staticmethod
    def generate_algo(p: StaticKsomParams) -> SomAlgo:
        """Converts Static Kohonen SOM parameters into generic SOM functions."""
        return SomAlgo(
            f_dist=EuclidianDist(),
            f_nbh=GaussianNbh(sigma=p.sigma),
            f_lr=ConstantLr(alpha=p.alpha),
            f_update=SomUpdate(),
        )


class DsomParams(AbstractSomParams):
    """Dynamic SOM parameters."""

    plasticity: float | Float[Array, "..."]
    alpha: float | Float[Array, "..."]


class Dsom(AbstractSom):
    """Dynamic SOM."""

    @staticmethod
    def generate_algo(p: DsomParams) -> SomAlgo:
        """Converts Dynamic SOM parameters into generic SOM functions."""
        return SomAlgo(
            f_dist=EuclidianDist(),
            f_nbh=DsomNbh(plasticity=p.plasticity),
            f_lr=DsomLr(alpha=p.alpha),
            f_update=SomUpdate(),
        )
