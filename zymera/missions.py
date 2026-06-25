"""
Missions — reward as data, plus group routing for the adversarial roadmap.

A :class:`Mission` is a frozen composition of :class:`RewardTerm`\\ s: the env
asks it for ``reward`` (Σ weighted terms + the UNWEIGHTED per-term dict that
feeds ``info["reward_terms"]``), ``done``, mission-owned ``metrics``, and
drawable ``annotations``. :class:`GroupedMission` routes per-group missions
over a shared :class:`~zymera.metrics.StepCtx` so k-of-N red agents can score
a different objective than the blue team (design spec §3.1).

JAX contract (the static-object rule):

* Missions and terms are frozen, hashable, closure-captured trace-time
  constants. Term SET is static; weights may become traced later — never
  gate on a weight's value (doctrine #3).
* ``init_state`` / ``update`` must keep an IDENTICAL pytree structure between
  reset and every step (doctrine #6). The default mission state is ``()``.
* ``done`` gating on ``max_steps`` is Python-time (static config), never on
  traced data.

Sign convention: term functions return UNSIGNED magnitudes; penalties get
their sign from the :class:`RewardTerm` weight (v0 subtracted
``w_coll * collisions`` — here the default collision term carries ``-4.0``).
"""

from dataclasses import dataclass, replace
from typing import (
    Any, Callable, Dict, Optional, Protocol, Tuple, Union, runtime_checkable,
)

import chex
import jax
import jax.numpy as jnp

from .metrics import coverage_fraction

# =============================================================================
# Reward terms
# =============================================================================

#: ``(prev_world, world, action, ctx) -> (N,) float32`` — UNWEIGHTED magnitude.
TermFn = Callable[[Any, Any, chex.Array, Any], chex.Array]


@dataclass(frozen=True)
class RewardTerm:
    """One named reward component: ``weight · fn(prev, world, action, ctx)``.

    ``requires`` names the :class:`~zymera.metrics.StepCtx` fields the term
    reads — the env unions these across terms/obs at ``__init__`` so only the
    requested machinery compiles. ``fn`` must return the UNWEIGHTED ``(N,)``
    value; the weighted sum happens in :meth:`Mission.reward`.
    """

    name:     str
    weight:   float
    fn:       TermFn
    requires: frozenset = frozenset()

    def __post_init__(self):
        object.__setattr__(self, "requires", frozenset(self.requires))


# =============================================================================
# Annotation primitives (viz data contract, spec §4)
# =============================================================================
# Plain frozen dataclasses with no JAX dependency: missions stay drawable
# without zymera.viz knowing their internals.


@dataclass(frozen=True)
class Point:
    """A marked cell — VIP, intruder, rally point. ``pos`` is ``(2,)`` (row, col)."""

    pos: Any
    tag: str = ""


@dataclass(frozen=True)
class Path:
    """An ordered cell sequence — patrol route, planned path. ``cells`` is ``(K, 2)``."""

    cells: Any
    tag: str = ""


@dataclass(frozen=True)
class Region:
    """A cell mask — jammed zone, goal area. ``mask`` is ``(H, W)`` bool."""

    mask: Any
    tag: str = ""


Annotation = Union[Point, Path, Region]


# =============================================================================
# Mission
# =============================================================================


@dataclass(frozen=True)
class Mission:
    """A reward-term bundle with the full mission protocol surface.

    Defaults: ``()`` mission state, identity ``update``, timeout-only ``done``
    (all-False when ``max_steps`` is None), sum-of-weighted-terms reward
    returning the unweighted per-term dict.
    """

    terms:     Tuple[RewardTerm, ...] = ()
    max_steps: Optional[int] = None

    def __post_init__(self):
        object.__setattr__(self, "terms", tuple(self.terms))
        names = [t.name for t in self.terms]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(
                f"duplicate reward-term name(s) {dupes}: term names must be "
                "unique within a Mission (they key info['reward_terms'])"
            )
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1 or None, got {self.max_steps}")

    # ---- static composition --------------------------------------------------

    @property
    def requires(self) -> frozenset:
        """Union of the terms' StepCtx requirements."""
        out = frozenset()
        for t in self.terms:
            out |= t.requires
        return out

    # ---- mission protocol ------------------------------------------------------

    def init_state(self, key: jax.Array, world) -> Any:
        """Mission-owned pytree at reset. Default: ``()`` (uses no key)."""
        del key, world
        return ()

    def update(self, prev, world, ctx, mstate, key: jax.Array) -> Any:
        """Advance mission-owned state (scripted NPCs, waypoints). Default:
        identity — MUST preserve pytree structure (doctrine #6)."""
        del prev, world, ctx, key
        return mstate

    def done(self, world, ctx, mstate) -> chex.Array:
        """(N,) bool. All-False, or ``step_count >= max_steps`` broadcast when
        ``max_steps`` is set (Python-time gate on static config)."""
        del ctx, mstate
        n = world.n_agents
        if self.max_steps is None:
            return jnp.zeros((n,), dtype=jnp.bool_)
        return jnp.broadcast_to(world.step_count >= self.max_steps, (n,))

    def reward(
        self, prev, world, action, ctx, mstate,
    ) -> Tuple[chex.Array, Dict[str, chex.Array]]:
        """``(Σ wᵢ·termᵢ  (N,) f32,  {name: UNWEIGHTED (N,) f32})``.

        The unweighted dict feeds ``info["reward_terms"]`` so analysis can
        re-weight post-hoc without re-running.
        """
        del mstate
        n = world.n_agents
        unweighted = {
            t.name: t.fn(prev, world, action, ctx).astype(jnp.float32)
            for t in self.terms
        }
        total = jnp.zeros((n,), jnp.float32)
        for t in self.terms:
            total = total + t.weight * unweighted[t.name]
        return total.astype(jnp.float32), unweighted

    def metrics(self, world, ctx, mstate) -> Dict[str, chex.Array]:
        """Mission-owned success metrics: team coverage fraction, plus the
        giant-component fraction when the ctx carries ``reach`` (Python-time
        gate — ctx structure is a function of env config only)."""
        del mstate
        out = {"coverage": coverage_fraction(world.covered)}
        if ctx is not None and ctx.reach is not None:
            out["giant_fraction"] = (
                ctx.reach.sum(-1).max() / world.n_agents
            ).astype(jnp.float32)
        return out

    def annotations(self, world, mstate) -> Tuple[Annotation, ...]:
        """Drawable primitives for zymera.viz. Default: none."""
        del world, mstate
        return ()


# =============================================================================
# Group assignment
# =============================================================================


@runtime_checkable
class Assignment(Protocol):
    """Reset-time group-id assignment: ``assign(key, n_agents) -> (N,) int32``."""

    def assign(self, key: jax.Array, n_agents: int) -> chex.Array:
        ...


@dataclass(frozen=True)
class FixedAssignment:
    """Deterministic group ids. ``groups=None`` → everyone in group 0.
    Ignores its key."""

    groups: Optional[Tuple[int, ...]] = None

    def __post_init__(self):
        if self.groups is not None:
            object.__setattr__(self, "groups", tuple(int(g) for g in self.groups))

    def assign(self, key: jax.Array, n_agents: int) -> chex.Array:
        del key
        if self.groups is None:
            return jnp.zeros((n_agents,), jnp.int32)
        if len(self.groups) != n_agents:
            raise ValueError(
                f"FixedAssignment has {len(self.groups)} group ids "
                f"but the env has {n_agents} agents"
            )
        return jnp.asarray(self.groups, jnp.int32)


@dataclass(frozen=True)
class RandomKofN:
    """A random k-subset of agents gets ``group`` (default 1); the rest stay 0.

    Membership re-randomizes per reset WITHOUT retracing — the draw is pure
    JAX on the reset key (the red-within-blue graft, spec §3.1).
    """

    k:     int
    group: int = 1

    def __post_init__(self):
        if self.k < 0:
            raise ValueError(f"k must be >= 0, got {self.k}")

    def assign(self, key: jax.Array, n_agents: int) -> chex.Array:
        if self.k > n_agents:
            raise ValueError(f"RandomKofN k={self.k} exceeds n_agents={n_agents}")
        perm = jax.random.permutation(key, n_agents)
        return (
            jnp.zeros((n_agents,), jnp.int32).at[perm[: self.k]].set(self.group)
        )


# =============================================================================
# GroupedMission
# =============================================================================


@dataclass(frozen=True)
class GroupedMission:
    """Per-group objectives over a shared world: ``missions[g]`` scores the
    agents with ``world.group == g``.

    Fixed-shape routing: every group-mission's reward/done is computed over
    ALL N agents, then where-selected by group id — no dynamic shapes, no
    retrace when :class:`RandomKofN` re-rolls membership. Per-term/metric
    names are namespaced ``g{i}/<name>``. Agents whose group id has no
    mission (out of range) score 0 and never finish — keep ids in range.

    NOTE (judge-panel mandate): the StepCtx is SHARED across groups —
    coverage counts every agent's footprint and connectivity reads the union
    graph. Per-group ctx semantics (e.g. blue-only coverage) must be decided
    explicitly per term when red training lands.
    """

    assignment: Assignment
    missions:   Tuple[Mission, ...]

    def __post_init__(self):
        object.__setattr__(self, "missions", tuple(self.missions))
        if not self.missions:
            raise ValueError("GroupedMission needs at least one mission")

    # ---- static composition --------------------------------------------------

    @property
    def requires(self) -> frozenset:
        """Union of all group-missions' StepCtx requirements."""
        out = frozenset()
        for m in self.missions:
            out |= m.requires
        return out

    @property
    def terms(self) -> Tuple[RewardTerm, ...]:
        """All groups' terms, names namespaced ``g{i}/<name>`` (protocol
        compatibility; uniqueness holds when each sub-mission's does)."""
        return tuple(
            replace(t, name=f"g{g}/{t.name}")
            for g, m in enumerate(self.missions)
            for t in m.terms
        )

    # ---- mission protocol ------------------------------------------------------

    def init_state(self, key: jax.Array, world) -> Tuple[Any, ...]:
        keys = jax.random.split(key, len(self.missions))
        return tuple(
            m.init_state(k, world) for m, k in zip(self.missions, keys)
        )

    def update(self, prev, world, ctx, mstate, key: jax.Array) -> Tuple[Any, ...]:
        keys = jax.random.split(key, len(self.missions))
        return tuple(
            m.update(prev, world, ctx, ms, k)
            for m, ms, k in zip(self.missions, mstate, keys)
        )

    def done(self, world, ctx, mstate) -> chex.Array:
        """(N,) bool — each agent reports its OWN group's mission done."""
        out = jnp.zeros((world.n_agents,), dtype=jnp.bool_)
        for g, (m, ms) in enumerate(zip(self.missions, mstate)):
            out = jnp.where(world.group == g, m.done(world, ctx, ms), out)
        return out

    def reward(
        self, prev, world, action, ctx, mstate,
    ) -> Tuple[chex.Array, Dict[str, chex.Array]]:
        """Each group-mission's total is computed over ALL N (fixed shape),
        then routed by ``where(group == g, ...)`` and summed. The per-term
        dict keeps the UNMASKED unweighted values under ``g{i}/<name>`` —
        mask post-hoc with ``world.group`` in analysis."""
        total = jnp.zeros((world.n_agents,), jnp.float32)
        unweighted: Dict[str, chex.Array] = {}
        for g, (m, ms) in enumerate(zip(self.missions, mstate)):
            r_g, terms_g = m.reward(prev, world, action, ctx, ms)
            total = total + jnp.where(world.group == g, r_g, 0.0)
            for name, val in terms_g.items():
                unweighted[f"g{g}/{name}"] = val
        return total.astype(jnp.float32), unweighted

    def metrics(self, world, ctx, mstate) -> Dict[str, chex.Array]:
        out: Dict[str, chex.Array] = {}
        for g, (m, ms) in enumerate(zip(self.missions, mstate)):
            for name, val in m.metrics(world, ctx, ms).items():
                out[f"g{g}/{name}"] = val
        return out

    def annotations(self, world, mstate) -> Tuple[Annotation, ...]:
        out: Tuple[Annotation, ...] = ()
        for m, ms in zip(self.missions, mstate):
            out = out + tuple(m.annotations(world, ms))
        return out
