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


def test_som():
    params = smp.SomParams(
        f_dist=smp.EuclidianDist(),
        f_nbh=smp.GaussianNbh(sigma=0.1),
        f_lr=smp.ConstantLr(alpha=0.01),
        f_update=smp.SomUpdate(),
    )
    model = smp.Som(shape, topography, borderless, input_shape, params)

    # Iterates over the toy dataset
    for i in range(0, inputs_data["bu_v"].shape[0]):
        input_data = jax.tree_map(lambda x: x[i], inputs_data)
        model, aux = model(input_data)

    assert jnp.min(model.w_bu) >= 0 and jnp.max(model.w_bu) <= 1


def test_static_ksom():
    params = smp.StaticKsomParams(sigma=0.1, alpha=0.01)
    model = smp.StaticKsom(shape, topography, borderless, input_shape, params)

    # Iterates over the toy dataset
    for i in range(0, inputs_data["bu_v"].shape[0]):
        input_data = jax.tree_map(lambda x: x[i], inputs_data)
        model, aux = model(input_data)

    assert jnp.min(model.w_bu) >= 0 and jnp.max(model.w_bu) <= 1


def test_ksom():
    params = smp.KsomParams(
        t_f=60000, sigma_i=1.0, sigma_f=0.01, alpha_i=0.1, alpha_f=0.001
    )
    model = smp.Ksom(shape, topography, borderless, input_shape, params)

    # Iterates over the toy dataset
    for i in range(0, inputs_data["bu_v"].shape[0]):
        input_data = jax.tree_map(lambda x: x[i], inputs_data)
        model, aux = model(input_data)

    assert jnp.min(model.w_bu) >= 0 and jnp.max(model.w_bu) <= 1


def test_dsom():
    params = smp.DsomParams(alpha=0.001, plasticity=0.02)
    model = smp.Dsom(shape, topography, borderless, input_shape, params)

    # Iterates over the toy dataset
    for i in range(0, inputs_data["bu_v"].shape[0]):
        input_data = jax.tree_map(lambda x: x[i], inputs_data)
        model, aux = model(input_data)

    assert jnp.min(model.w_bu) >= 0 and jnp.max(model.w_bu) <= 1
