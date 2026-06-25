"""Composable agent building blocks (the parts you wire into a policy).

Raw-JAX and functional so blocks compose freely and stay jit/vmap-safe. Start
minimal: this seeds the module with one foundational block (an MLP). Add encoders,
belief nets, attention, aggregators, low-level controllers here as experiments
need them; graduate proven blocks in with a test.
"""
from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp


def mlp_init(key: jax.Array, sizes: Sequence[int]):
    """Glorot-init params for an MLP with the given layer sizes."""
    params = []
    keys = jax.random.split(key, len(sizes) - 1)
    for k, d_in, d_out in zip(keys, sizes[:-1], sizes[1:]):
        scale = jnp.sqrt(2.0 / (d_in + d_out))
        w = jax.random.normal(k, (d_in, d_out)) * scale
        b = jnp.zeros((d_out,))
        params.append((w, b))
    return params


def mlp_apply(params, x: jax.Array, activation=jax.nn.relu) -> jax.Array:
    """Apply the MLP; activation on every layer except the last."""
    for i, (w, b) in enumerate(params):
        x = x @ w + b
        if i < len(params) - 1:
            x = activation(x)
    return x
