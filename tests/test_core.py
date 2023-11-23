"""Tests the different SOMs."""

# ruff: noqa: D103

import jax
import jax.numpy as jnp
import somap as smp

from tests import inputs_data


jax.config.update("jax_platform_name", "cpu")

shape = (11, 13)
topography = "hex"
borderless = False
input_shape = (8, 8)


def test_make_step():
    params = smp.StaticKsomParams(sigma=0.1, alpha=0.01)
    model = smp.StaticKsom(shape, topography, borderless, input_shape, params)
    model2 = smp.StaticKsom(shape, topography, borderless, input_shape, params)

    assert jnp.allclose(model.w_bu, model2.w_bu)

    # Iterates over the toy dataset
    for i in range(0, inputs_data["bu_v"].shape[0]):
        input_data = jax.tree_map(lambda x: x[i], inputs_data)
        model, aux = model(input_data)
        model2, aux = smp.make_step(model2, input_data)

    assert jnp.allclose(model.w_bu, model2.w_bu)


def test_make_steps():
    params = smp.StaticKsomParams(sigma=0.1, alpha=0.01)
    model = smp.StaticKsom(shape, topography, borderless, input_shape, params)
    model2 = model

    # Iterates over the toy dataset
    for i in range(0, inputs_data["bu_v"].shape[0]):
        input_data = jax.tree_map(lambda x: x[i], inputs_data)
        model, aux = model(input_data)

    model2, aux = smp.make_steps(model2, inputs_data)

    assert jnp.allclose(model.w_bu, model2.w_bu)
