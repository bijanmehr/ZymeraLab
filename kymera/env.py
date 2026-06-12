"""
Env contract, state schema, and registry — the spine of kymera.

This module owns the FROZEN contracts everything else conforms to
(see docs/specs/2026-06-11-kymera-design.md):

* :class:`ActionId` / ``ACTION_DELTAS`` — the movement vocabulary.
* :class:`Body` / :class:`World` — the state pytree every component reads.
* :class:`Env` — gym-style base: ``reset(key)`` / ``step(state, action, key)``.
* :class:`GridEnv` — the orchestrator over the five components
  (worldgen / dynamics / comms / obs / missions).
* ``make`` / ``make_from`` / ``register_env`` / ``list_envs`` — registry.

Key protocol (FROZEN — parity and reproducibility ride on it):

* ``reset(key)``: ``wkey, skey = split(key)`` → terrain(wkey), spawn(skey);
  group assignment gets ``fold_in(key, 1)``; mission init gets ``fold_in(key, 2)``.
* ``step(state, action, key)``: ``k_chan, k_mis = split(key)``.
"""

from enum import IntEnum
from typing import Any, Callable, Dict, Tuple

import chex
import jax
import jax.numpy as jnp

# =============================================================================
# Actions
# =============================================================================


class ActionId(IntEnum):
    """Movement vocabulary on the square grid.

    Integer values are stable — extend by appending so existing
    checkpoints keep their meaning.
    """

    STAY = 0
    NORTH = 1
    EAST = 2
    SOUTH = 3
    WEST = 4


N_ACTIONS = len(ActionId)

_DELTA_BY_ACTION = {
    ActionId.STAY:  (0,  0),
    ActionId.NORTH: (-1, 0),
    ActionId.EAST:  (0,  1),
    ActionId.SOUTH: (1,  0),
    ActionId.WEST:  (0, -1),
}
assert set(_DELTA_BY_ACTION) == set(ActionId)

ACTION_DELTAS = jnp.array(
    [_DELTA_BY_ACTION[a] for a in ActionId], dtype=jnp.int32
)


# =============================================================================
# State pytrees
# =============================================================================


@chex.dataclass(frozen=True)
class Body:
    """Per-agent physical state. SoA — every field is shape ``(N, ...)``."""

    position: chex.Array        # (N, 2) int32 — (row, col)
    energy:   chex.Array        # (N,)  float32 — zeros until the energy roadmap lands


@chex.dataclass(frozen=True)
class World:
    """The simulated world state. Immutable JAX pytree.

    Field semantics (the committed contract components and user code read):

    * ``explored``   — (H, W) int32 per-cell visit counts (heatmaps, redundancy).
    * ``seen_by``    — (N, H, W) bool, each agent's OWN covered/sensed cells.
      The team-coverage metric reads ``covered = seen_by.any(0)``.
    * ``comm_graph`` — (N, N) bool, edges that DELIVERED this step (realized,
      post-dropout). Potential topology lives in :class:`kymera.metrics.StepCtx`.
    * ``channel``    — channel-owned pytree (ring buffers, beliefs); ``()`` when
      the env has no channel.
    * ``mission``    — mission-owned pytree (waypoints, NPC positions); ``()``
      by default. Structure must be identical between reset and every step.
    * ``group``      — (N,) int32 group ids, assigned at reset (red-within-blue).
    """

    body:       Body
    explored:   chex.Array      # (H, W) int32
    seen_by:    chex.Array      # (N, H, W) bool
    wall:       chex.Array      # (H, W) bool
    comm_graph: chex.Array      # (N, N) bool — delivered edges
    step_count: chex.Array      # () int32
    channel:    Any
    mission:    Any
    group:      chex.Array      # (N,) int32

    # ---- shape helpers ------------------------------------------------------

    @property
    def grid_h(self) -> int:
        return self.explored.shape[0]

    @property
    def grid_w(self) -> int:
        return self.explored.shape[1]

    @property
    def n_agents(self) -> int:
        return self.body.position.shape[0]

    @property
    def visited(self) -> chex.Array:
        """(H, W) bool — any agent has stepped here."""
        return self.explored > 0

    @property
    def covered(self) -> chex.Array:
        """(H, W) bool — covered by any agent's footprint. THE coverage source."""
        return self.seen_by.any(0)


# =============================================================================
# Env base
# =============================================================================


class Env:
    """Gym-style base.

    Contract::

        obs, state = env.reset(key)
        obs, state, reward, done, info = env.step(state, action, key)

    ``state`` is a :class:`World` (or compatible pytree); ``action`` is
    ``(N,) int32``; ``reward``/``done`` are ``(N,)``. ``info`` has a fixed
    keyset per env configuration (scan-stackable).
    """

    n_agents: int
    n_actions: int = N_ACTIONS

    def reset(self, key: jax.Array) -> Tuple[jax.Array, World]:
        raise NotImplementedError

    def step(
        self, state: World, action: jax.Array, key: jax.Array,
    ) -> Tuple[jax.Array, World, jax.Array, jax.Array, Dict[str, Any]]:
        raise NotImplementedError


# =============================================================================
# Registry
# =============================================================================

_REGISTRY: Dict[str, Callable[..., Env]] = {}


def register_env(name: str, factory: Callable[..., Env]) -> None:
    """Register ``factory(**kwargs) -> Env`` under ``name``."""
    if name in _REGISTRY:
        raise ValueError(f"env name already registered: {name!r}")
    _REGISTRY[name] = factory


def list_envs() -> list:
    return sorted(_REGISTRY)


def make(name: str, **kwargs) -> Env:
    """Construct a registered env (recipe) by name.

    The returned env remembers ``(name, kwargs)`` so ``env.spec()`` /
    ``env.replace(...)`` / ``make_from`` round-trip.
    """
    if name not in _REGISTRY:
        raise ValueError(f"unknown env: {name!r}; available: {list_envs()}")
    env = _REGISTRY[name](**kwargs)
    env._recipe = (name, dict(kwargs))
    return env


def make_from(spec: Dict[str, Any]) -> Env:
    """Rebuild an env from ``env.spec()`` output: ``{"recipe": name, **kwargs}``."""
    spec = dict(spec)
    name = spec.pop("recipe")
    return make(name, **spec)
