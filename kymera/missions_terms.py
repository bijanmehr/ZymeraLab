"""
The proven reward-term zoo, ported from zymera v0's reward block
(``examples/comm_coverage.py`` step) and ``examples/cbf.py``.

Every term is ``(prev_world, world, action, ctx) -> (N,) float32`` returning
an UNSIGNED, UNWEIGHTED magnitude — penalties get their sign from the
:class:`~kymera.missions.RewardTerm` weight, exactly mirroring v0's
``reward = w_cov·newly + w_conn·conn − w_overlap·overlap − w_coll·coll``
(so :data:`DEFAULT_TERMS` carries ``-4.0`` on the collision term).

Parameterized terms are factories returning a term function. Each function
(including factory outputs) carries a ``.requires`` frozenset attribute
naming the :class:`~kymera.metrics.StepCtx` fields it reads — pass it
through to ``RewardTerm(requires=...)`` so the env derives those fields.
The CBF terms compute from positions directly (``requires = ∅``): their λ₂
eigendecomposition only compiles into envs that actually use them.
"""

from typing import Callable

import chex
import jax
import jax.numpy as jnp

from . import metrics
from .missions import RewardTerm

# =============================================================================
# requires-tagging helper
# =============================================================================


def _requires(*names: str) -> Callable:
    """Attach ``fn.requires = frozenset(names)`` (decorator)."""
    req = frozenset(names)

    def deco(fn):
        fn.requires = req
        return fn

    return deco


# =============================================================================
# Coverage / connectivity / collision (v0 comm_coverage.py:320-338)
# =============================================================================


@_requires("newly_covered")
def new_coverage(prev, world, action, ctx) -> chex.Array:
    """(N,) — cells of my footprint this step the TEAM had not covered before
    (v0 ``newly``)."""
    del prev, world, action
    return ctx.newly_covered


@_requires("reach")
def reach_fraction(prev, world, action, ctx) -> chex.Array:
    """(N,) — fraction of OTHER agents reachable through the potential comm
    graph (v0 per-agent ``connectivity``, the ``conn_cap=None`` branch)."""
    del prev, action
    n = world.n_agents
    return ((ctx.reach.sum(-1) - 1) / max(n - 1, 1)).astype(jnp.float32)


def capped_giant(cap: int) -> Callable:
    """Factory — (N,) shared ``min(giant, cap) / cap`` where ``giant`` is the
    largest-component size (v0 ``conn_cap`` branch: cap=N−1 ⇒ "3 connected +
    1 roamer" scores like a full clump → no clump incentive)."""
    if cap < 1:
        raise ValueError(f"cap must be >= 1, got {cap}")

    @_requires("reach")
    def fn(prev, world, action, ctx) -> chex.Array:
        del prev, action
        giant = ctx.reach.sum(-1).max()
        val = (jnp.minimum(giant, cap) / cap).astype(jnp.float32)
        return jnp.broadcast_to(val, (world.n_agents,))

    return fn


@_requires("collisions")
def collision_count(prev, world, action, ctx) -> chex.Array:
    """(N,) — number of OTHER agents sharing my cell. Unsigned magnitude:
    weight it negatively (v0 subtracted ``w_coll·collisions``)."""
    del prev, world, action
    return ctx.collisions


@_requires("overlap")
def same_step_overlap(prev, world, action, ctx) -> chex.Array:
    """(N,) — cells of my footprint this step ANOTHER agent also covers now
    (v0 anti-redundancy; weight negatively)."""
    del prev, world, action
    return ctx.overlap


# =============================================================================
# Local connectivity (partial-info; v0 comm_coverage.py:353-362)
# =============================================================================


def cohesion_leash(leash: float, comm_r: int) -> Callable:
    """Factory — (N,) ``max(nearest-teammate-dist − leash, 0)``, with the
    nearest distance clamped at ``comm_r`` (an agent can't measure a teammate
    it can't sense). Soft tether; weight negatively."""

    @_requires("dist")
    def fn(prev, world, action, ctx) -> chex.Array:
        del prev, action
        off = ~jnp.eye(world.n_agents, dtype=bool)
        d_off = jnp.where(off, ctx.dist, jnp.inf)
        nn = jnp.minimum(d_off.min(-1), float(comm_r))
        return jnp.maximum(nn - leash, 0.0).astype(jnp.float32)

    return fn


def degree_floor(floor: float, comm_r: int) -> Callable:
    """Factory — (N,) ``max(floor − in-range-neighbour-count, 0)``. Purely
    local anti-isolation signal (no λ₂, no global graph); weight negatively."""

    @_requires("dist")
    def fn(prev, world, action, ctx) -> chex.Array:
        del prev, action
        off = ~jnp.eye(world.n_agents, dtype=bool)
        num_nb = ((ctx.dist <= comm_r) & off).sum(-1).astype(jnp.float32)
        return jnp.maximum(floor - num_nb, 0.0).astype(jnp.float32)

    return fn


# =============================================================================
# PBRS — potential-based reward shaping (v0 comm_coverage.py:364-376)
# =============================================================================


def phi_nearest_frontier(world) -> chex.Array:
    """Φ1 (N,) — NEGATED Chebyshev distance to the nearest uncovered free
    cell (frontier-seeking). Negated so larger Φ = closer to fresh ground."""
    uncovered = ~world.covered & ~world.wall
    return -metrics.dist_to_frontier(
        world.body.position, uncovered, world.grid_h, world.grid_w
    )


def phi_field_mean(world) -> chex.Array:
    """Φ3 () — NEGATED mean over uncovered free cells of the distance to the
    nearest agent (team blanket/territory; shared scalar)."""
    uncovered = ~world.covered & ~world.wall
    return -metrics.field_mean_dist(
        world.body.position, uncovered, world.grid_h, world.grid_w
    )


def pbrs(phi: Callable, gamma: float) -> Callable:
    """Factory — PBRS combinator ``F = γ·Φ(world) − Φ(prev)``, policy-invariant
    by construction. ``phi(world)`` returns ``(N,)`` or a shared scalar
    (broadcast to ``(N,)``). With the NEGATED-distance potentials above this
    matches v0's ``w·(γ·(−p1) − (−p0))`` exactly; weight positively."""

    @_requires()
    def fn(prev, world, action, ctx) -> chex.Array:
        del action, ctx
        f = gamma * phi(world) - phi(prev)
        return jnp.broadcast_to(f, (world.n_agents,)).astype(jnp.float32)

    return fn


# =============================================================================
# Soft-CBF barrier penalties (ported from v0 examples/cbf.py)
# =============================================================================
# Discrete-time CBF: a transition x_k → x_{k+1} is safe if
#     h(x_{k+1}) ≥ (1 − α)·h(x_k),    0 < α ≤ 1
# so the violation residual is relu((1−α)·h_prev − h_next). The barrier is a
# THRESHOLD (λ₂ ≥ eps), so a just-connected stretched chain satisfies it —
# no clump incentive. Both terms compute from positions directly
# (requires = ∅); weight negatively.


def _cheby_f32(pos: chex.Array) -> chex.Array:
    """(N, N) float32 Chebyshev distances (v0 cbf._cheby)."""
    return jnp.max(jnp.abs(pos[:, None, :] - pos[None, :, :]), -1).astype(
        jnp.float32
    )


def _soft_weights(pos: chex.Array, comm_r: int, sharp: float) -> chex.Array:
    """(N, N) smooth edge weights ≈1 within comm_r, ≈0 beyond; zero diagonal."""
    w = jax.nn.sigmoid(sharp * (comm_r - _cheby_f32(pos)))
    return w * (1.0 - jnp.eye(pos.shape[0]))


def _lambda2(pos: chex.Array, comm_r: int, sharp: float) -> chex.Array:
    """Fiedler value of the weighted Laplacian. λ₂ > 0 ⟺ connected."""
    w = _soft_weights(pos, comm_r, sharp)
    lap = jnp.diag(w.sum(-1)) - w
    return jnp.linalg.eigvalsh(lap)[1]            # eigvalsh: ascending order


def _coll_barriers(pos: chex.Array, d_min: float) -> chex.Array:
    """(N, N) per-pair collision barrier h_ij = d_ij − d_min; self-pairs
    pushed to a large value so they never register a violation."""
    return (_cheby_f32(pos) - d_min) + jnp.eye(pos.shape[0]) * 1e3


def _cbf_residual(h_prev, h_next, alpha: float) -> chex.Array:
    """Discrete-time CBF violation residual: relu((1−α)·h_prev − h_next)."""
    return jax.nn.relu((1.0 - alpha) * h_prev - h_next)


def cbf_conn(alpha: float, eps: float, sharp: float, comm_r: int) -> Callable:
    """Factory — (N,) shared connectivity-barrier violation residual on
    ``h = λ₂ − eps``, divided by N as v0 does (shared team penalty)."""

    @_requires()
    def fn(prev, world, action, ctx) -> chex.Array:
        del action, ctx
        n = world.n_agents
        hp = _lambda2(prev.body.position, comm_r, sharp) - eps
        hn = _lambda2(world.body.position, comm_r, sharp) - eps
        res = _cbf_residual(hp, hn, alpha) / n
        return jnp.broadcast_to(res, (n,)).astype(jnp.float32)

    return fn


def cbf_coll(alpha: float, dmin: float) -> Callable:
    """Factory — (N,) per-agent sum of pairwise collision-barrier violation
    residuals on ``h_ij = d_ij − dmin``."""

    @_requires()
    def fn(prev, world, action, ctx) -> chex.Array:
        del action, ctx
        hp = _coll_barriers(prev.body.position, dmin)
        hn = _coll_barriers(world.body.position, dmin)
        return _cbf_residual(hp, hn, alpha).sum(-1).astype(jnp.float32)

    return fn


# =============================================================================
# Default term set (the v0 comm-coverage reward: w_cov/w_conn/w_coll = 1/2/4)
# =============================================================================

DEFAULT_TERMS = (
    RewardTerm("coverage",     1.0, new_coverage,    requires=new_coverage.requires),
    RewardTerm("connectivity", 2.0, reach_fraction,  requires=reach_fraction.requires),
    RewardTerm("collision",   -4.0, collision_count, requires=collision_count.requires),
)
