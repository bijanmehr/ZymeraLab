"""Trainers + shared training utilities.

Independent trainers (ppo / es / supervised) will live here as flat functions
over shared utils, with NO forced common interface — added when an experiment
needs them. P1 seeds the shared-utils side with ``evaluate`` (the doctrine: eval
= sample policy, multi-seed via vmap). Experiments WRITE their learning stack by
importing these + ``zymera.nets``.
"""
from __future__ import annotations

from typing import Callable, Dict

import jax
import jax.numpy as jnp

from .rollout import rollout


def evaluate(env, policy: Callable, n_steps: int, n_episodes: int,
             key: jax.Array) -> Dict[str, float]:
    """Roll ``policy`` in ``env`` over ``n_episodes`` seeds (vmap, no python loop);
    report summed-reward statistics. Eval samples the policy — never argmax."""
    keys = jax.random.split(key, n_episodes)
    trajs = jax.vmap(lambda k: rollout(env, policy, n_steps, k))(keys)
    ep_returns = trajs["reward"].sum(axis=(1, 2))          # (n_episodes,)
    return {
        "return_mean": float(jnp.mean(ep_returns)),
        "return_std": float(jnp.std(ep_returns)),
        "n_episodes": int(n_episodes),
    }
