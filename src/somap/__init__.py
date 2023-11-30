"""Somap public resources."""

# ruff: noqa: E402

from jaxtyping import install_import_hook


# Enables shape and dtype runtime typeckecking on arrays
hook = install_import_hook("somap", "beartype.beartype")

# Classic imports
from . import datasets
from .core import AbstractSom, AbstractSomParams, make_step, make_steps, SomAlgo
from .distance import EuclidianDist, wdist_l2
from .grid import distance_map
from .learning_rate import AbstractLr, ConstantLr, DsomLr, KsomLr
from .neighborhood import AbstractNbh, DsomNbh, GaussianNbh, KsomNbh, MexicanHatNbh
from .plot import plot, save_plot
from .plot_backends import set_plot_backend
from .serialisation import load, save
from .som import (
    Dsom,
    DsomParams,
    Ksom,
    KsomParams,
    Som,
    SomParams,
    StaticKsom,
    StaticKsomParams,
)
from .update import SomUpdate


__all__ = [
    "datasets",
    "AbstractSom",
    "AbstractSomParams",
    "make_step",
    "make_steps",
    "SomAlgo",
    "EuclidianDist",
    "wdist_l2",
    "distance_map",
    "AbstractLr",
    "ConstantLr",
    "DsomLr",
    "KsomLr",
    "AbstractNbh",
    "DsomNbh",
    "GaussianNbh",
    "KsomNbh",
    "MexicanHatNbh",
    "plot",
    "save_plot",
    "set_plot_backend",
    "load",
    "save",
    "Dsom",
    "DsomParams",
    "Ksom",
    "KsomParams",
    "Som",
    "SomParams",
    "StaticKsomParams",
    "StaticKsom",
    "SomUpdate",
]

# Cleans the runtime typechecking stuff
hook.uninstall()
del hook, install_import_hook
