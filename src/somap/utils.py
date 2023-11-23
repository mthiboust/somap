"""Utility functions."""

import functools
import inspect
import logging
from collections.abc import Callable

import equinox as eqx
from jax import lax


def filter_scan(f: Callable, init, xs, *args, **kwargs):
    """Same as lax.scan, but allows to have eqx.Module in carry.

    Found at https://github.com/knyazer/equinox/blob/68f677e3153969c51a0646633eba0a2fec5b3ad2/examples/proximal_policy_optimization_brax.ipynb
    """
    init_dynamic_carry, static_carry = eqx.partition(init, eqx.is_array)

    def to_scan(dynamic_carry, x):
        carry = eqx.combine(dynamic_carry, static_carry)
        new_carry, out = f(carry, x)
        dynamic_new_carry, _ = eqx.partition(new_carry, eqx.is_array)
        return dynamic_new_carry, out

    out_carry, out_ys = lax.scan(to_scan, init_dynamic_carry, xs, *args, **kwargs)
    return eqx.combine(out_carry, static_carry), out_ys


def experimental_warning(entity):
    """Decorator that logs a warning when class is instantiated or function is called.

    Args:
        entity (class or function): The class or function to be decorated.

    Returns:
        class or function: The decorated class or function with added warning
            functionality.
    """
    if inspect.isclass(entity):
        # If the entity is a class, wrap it
        orig_init = entity.__init__

        @functools.wraps(orig_init)
        def new_init(self, *args, **kwargs):
            logging.warning(f"{entity.__name__} is still an experimental module.")
            orig_init(self, *args, **kwargs)

        entity.__init__ = new_init
        return entity

    elif callable(entity):
        # If the entity is a callable (like a function), wrap it
        @functools.wraps(entity)
        def new_func(*args, **kwargs):
            logging.warning(f"{entity.__name__} is still an experimental function.")
            return entity(*args, **kwargs)

        return new_func

    else:
        raise TypeError("Decorator can only be used on classes or functions")
