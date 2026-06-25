"""
``rollout(env, policy, n_steps, key)`` — the scan rollout primitive.

JIT-pure, ``vmap``-friendly. The key protocol is FROZEN (identical to
zymera v0, so trajectories are reproducible across the migration):
``reset_key, scan_key = split(key)``; per step
``k, action_key, step_key = split(k, 3)``.

Memory profile is opt-in:

* ``keep="lean"`` (default) — the stacked trajectory drops the ``channel``
  and ``mission`` slots from ``World`` (the comm ring buffer is the first
  thing to blow device memory on multi-seed runs). The live scan carry is
  always the full state; only the *recorded* snapshot is filtered.
* ``keep="all"`` — record everything (use for viz / re-simulation).
* ``collect=("reward_terms", ...)`` — additionally stack the named ``info``
  keys (leading ``T`` axis, no initial entry).
"""

from typing import Callable, Dict, Sequence

import jax
import jax.numpy as jnp


def _filter_state(state, keep: str):
    if keep == "all":
        return state
    if keep == "lean":
        return state.replace(channel=(), mission=())
    raise ValueError(f"keep must be 'lean' or 'all', got {keep!r}")


def rollout(
    env,
    policy: Callable,
    n_steps: int,
    key: jax.Array,
    *,
    keep: str = "lean",
    collect: Sequence[str] = (),
) -> Dict:
    """Roll ``env`` forward ``n_steps`` ticks under ``policy``.

    Returns a dict with keys:

    * ``"world"``  — stacked World pytree, leaves ``(T+1, ...)`` (filtered per ``keep``)
    * ``"obs"``    — ``(T+1, N, ...)``
    * ``"action"`` / ``"reward"`` / ``"done"`` — ``(T, N)``
    * ``"info"``   — only when ``collect`` is non-empty: ``{k: (T, ...)}``

    Vmap over the ``key`` arg for free seed-parallelism::

        trajs = jax.vmap(lambda k: rollout(env, policy, 100, k))(jax.random.split(key, 32))
    """
    collect = tuple(collect)
    reset_key, scan_key = jax.random.split(key)
    obs0, state0 = env.reset(reset_key)

    def body(carry, _):
        state, obs, k = carry
        k, action_key, step_key = jax.random.split(k, 3)
        action = policy(obs, action_key)
        obs_next, state_next, reward, done, info = env.step(state, action, step_key)
        per_step = {
            "world":  _filter_state(state_next, keep),
            "obs":    obs_next,
            "action": action,
            "reward": reward,
            "done":   done,
        }
        if collect:
            per_step["info"] = {c: info[c] for c in collect}
        return (state_next, obs_next, k), per_step

    _, stacked = jax.lax.scan(body, (state0, obs0, scan_key), xs=None, length=n_steps)

    # Prepend the initial snapshot: state-side leaves get a (T+1, ...) axis.
    init_state = jax.tree_util.tree_map(
        lambda x: x[None, ...], _filter_state(state0, keep)
    )
    full_state = jax.tree_util.tree_map(
        lambda init, rest: jnp.concatenate([init, rest], axis=0),
        init_state, stacked["world"],
    )
    out = {
        "world":  full_state,
        "obs":    jnp.concatenate([obs0[None, ...], stacked["obs"]], axis=0),
        "action": stacked["action"],
        "reward": stacked["reward"],
        "done":   stacked["done"],
    }
    if collect:
        out["info"] = stacked["info"]
    return out


def random_policy(obs: jax.Array, key: jax.Array) -> jax.Array:
    """Uniform random actions — ``(obs, key) -> (N,) int32``."""
    from .env import N_ACTIONS

    return jax.random.randint(key, (obs.shape[0],), 0, N_ACTIONS).astype(jnp.int32)
