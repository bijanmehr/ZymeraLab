"""
Canonical metrics + :class:`StepCtx` — compute once, read everywhere.

This module replaces the metric blocks zymera v0 triplicated across the env
reward, the trainer cores, and graded eval: pure-JAX geometry / graph helpers
(:func:`pairwise_dist`, :func:`adjacency`, :func:`reach`, ...) plus the
per-step derived-quantity cache :class:`StepCtx` produced by :func:`derive`.

JAX contract (see docs/specs/2026-06-11-zymera-design.md §3.1):

* Every function is jit/vmap/``lax.scan``-safe with fixed shapes.
* :class:`StepCtx` structure is a function of the ``requires`` set ONLY —
  un-requested fields stay Python ``None`` at trace time, so un-requested
  machinery never compiles and the pytree structure never depends on data.
* Doctrine #4: ``ctx.adj`` / ``ctx.reach`` read POTENTIAL topology;
  realized (post-dropout) edges arrive separately as ``ctx.delivered``.
  Carrying both prevents a silent re-baseline when comms become stochastic.
"""

import math
from typing import Optional

import chex
import jax.numpy as jnp

# =============================================================================
# Geometry
# =============================================================================


def cheby_footprint(pos: chex.Array, h: int, w: int, r: int) -> chex.Array:
    """(N, H, W) bool — cells within Chebyshev radius ``r`` of each agent."""
    rows = jnp.arange(h)[None, :, None]
    cols = jnp.arange(w)[None, None, :]
    pr = pos[:, 0][:, None, None]
    pc = pos[:, 1][:, None, None]
    return jnp.maximum(jnp.abs(rows - pr), jnp.abs(cols - pc)) <= r


def pairwise_dist(pos: chex.Array) -> chex.Array:
    """(N, N) int32 — Chebyshev distances between agents."""
    return jnp.max(jnp.abs(pos[:, None, :] - pos[None, :, :]), axis=-1).astype(
        jnp.int32
    )


# =============================================================================
# Graph
# =============================================================================


def adjacency(pos: chex.Array, radius: int) -> chex.Array:
    """(N, N) bool — Chebyshev disk graph (symmetric, diagonal True)."""
    return pairwise_dist(pos) <= radius


def reach(adj: chex.Array) -> chex.Array:
    """(N, N) bool transitive closure — ``reach[i, j]`` = j reachable from i.

    Log2 matrix squaring: the loop count depends only on the STATIC agent
    count, so this unrolls at trace time (no traced-value branching).
    """
    n = adj.shape[0]                       # static (number of agents)
    out = adj
    for _ in range(math.ceil(math.log2(max(n, 2))) + 1):
        out = out | (jnp.matmul(out.astype(jnp.int32), out.astype(jnp.int32)) > 0)
    return out


def connected(adj: chex.Array) -> chex.Array:
    """Scalar bool — is the graph connected?"""
    return reach(adj).all()


def giant_component(adj: chex.Array) -> chex.Array:
    """Scalar int32 — size of the largest connected component."""
    return reach(adj).sum(-1).max().astype(jnp.int32)


# =============================================================================
# Occupancy / coverage
# =============================================================================


def collisions(pos: chex.Array) -> chex.Array:
    """(N,) float32 — number of OTHER agents co-located with each agent."""
    d = pairwise_dist(pos)
    off = ~jnp.eye(pos.shape[0], dtype=bool)
    return ((d == 0) & off).sum(-1).astype(jnp.float32)


def coverage_fraction(covered: chex.Array) -> chex.Array:
    """Scalar float32 — fraction of ALL cells covered (v0 used sum / H·W,
    walls included in the denominator; kept for metric continuity)."""
    return covered.astype(jnp.float32).mean()


def redundancy(seen_by: chex.Array, covered: chex.Array) -> chex.Array:
    """Scalar float32 — Σ per-agent covered cells / team covered cells.

    1.0 = perfectly disjoint coverage; N = everyone covered the same cells.
    """
    return (seen_by.sum() / jnp.maximum(covered.sum(), 1)).astype(jnp.float32)


def dist_to_frontier(
    pos: chex.Array, uncovered: chex.Array, h: int, w: int
) -> chex.Array:
    """(N,) float32 — Chebyshev distance from each agent to its nearest
    UNCOVERED cell (0 if nothing uncovered). Φ1 = −this; absolute cells →
    size-invariant."""
    rows = jnp.arange(h)[None, :, None]
    cols = jnp.arange(w)[None, None, :]
    ay = pos[:, 0][:, None, None]
    ax = pos[:, 1][:, None, None]
    d = jnp.maximum(jnp.abs(rows - ay), jnp.abs(cols - ax))        # (N, h, w)
    d_unc = jnp.where(uncovered[None], d, h + w + 1)
    md = d_unc.reshape(pos.shape[0], -1).min(-1)
    return jnp.where(uncovered.any(), md, 0.0).astype(jnp.float32)


def field_mean_dist(
    pos: chex.Array, uncovered: chex.Array, h: int, w: int
) -> chex.Array:
    """Scalar float32 — MEAN over uncovered cells of the Chebyshev distance to
    the nearest agent (0 if nothing uncovered). Φ3 = −this; mean (not sum) →
    size-invariant."""
    rows = jnp.arange(h)[:, None, None]
    cols = jnp.arange(w)[None, :, None]
    ay = pos[:, 0][None, None, :]
    ax = pos[:, 1][None, None, :]
    nearest = jnp.maximum(jnp.abs(rows - ay), jnp.abs(cols - ax)).min(-1)   # (h, w)
    nunc = uncovered.sum()
    mean_d = jnp.where(uncovered, nearest, 0.0).sum() / jnp.maximum(nunc, 1)
    return jnp.where(nunc > 0, mean_d, 0.0).astype(jnp.float32)


# =============================================================================
# StepCtx — per-step derived quantities
# =============================================================================


@chex.dataclass(frozen=True)
class StepCtx:
    """Per-step derived quantities, computed ONCE by :func:`derive` and read
    by reward terms, obs builders, and mission metrics.

    Every field is either a fixed-shape array or Python ``None`` — which one
    is decided at trace time by the ``requires`` set, never by data, so the
    pytree structure is stable under vmap/scan for a given env configuration.

    * ``adj`` / ``reach`` are POTENTIAL-topology quantities (doctrine #4);
      ``delivered`` is the realized (post-dropout) edge set from the channel.
    * ``newly_covered`` / ``overlap`` reproduce v0's coverage reward inputs:
      per-agent counts of fresh footprint cells and same-step footprint
      overlap with other agents.
    """

    dist:          Optional[chex.Array] = None   # (N, N) int32 Chebyshev
    adj:           Optional[chex.Array] = None   # (N, N) bool potential topology
    delivered:     Optional[chex.Array] = None   # (N, N) bool realized edges
    reach:         Optional[chex.Array] = None   # (N, N) bool closure of adj
    collisions:    Optional[chex.Array] = None   # (N,) f32 co-located others
    blocked:       Optional[chex.Array] = None   # (N,) bool reverted moves
    newly_covered: Optional[chex.Array] = None   # (N,) f32 fresh covered cells
    overlap:       Optional[chex.Array] = None   # (N,) f32 same-step overlap


DERIVABLE = frozenset(
    {"dist", "adj", "delivered", "reach", "collisions", "blocked",
     "newly_covered", "overlap"}
)


def derive(
    prev,
    world,
    requires: frozenset,
    *,
    topology=None,
    cover_r: int = 0,
    delivered: Optional[chex.Array] = None,
    blocked: Optional[chex.Array] = None,
) -> StepCtx:
    """Compute the requested :class:`StepCtx` fields for the ``prev → world``
    transition. Gating is Python-time on the static ``requires`` set.

    * ``dist`` / ``collisions`` come from ``world.body.position``.
    * ``adj`` (and ``reach``, its closure) come from ``topology.adjacency(world)``
      — requesting either without a topology is a configuration error.
    * ``newly_covered[i]`` = #cells in ``cheby_footprint(pos, cover_r) & ~wall``
      not yet in ``prev.covered`` (v0's ``newly`` reward input).
    * ``overlap[i]`` = #cells of i's footprint this step that another agent's
      footprint also covers this step (v0 anti-redundancy).
    * ``delivered`` / ``blocked`` pass through the keyword arguments (produced
      by the channel and the dynamics respectively).
    """
    unknown = frozenset(requires) - DERIVABLE
    if unknown:
        raise ValueError(
            f"unknown StepCtx requirement(s) {sorted(unknown)}; "
            f"derivable: {sorted(DERIVABLE)}"
        )

    pos = world.body.position
    n = world.n_agents

    # -- graph quantities (potential topology) --------------------------------
    adj_val = None
    if "adj" in requires or "reach" in requires:
        if topology is None:
            raise ValueError(
                "StepCtx requires 'adj'/'reach' but no topology was supplied"
            )
        adj_val = topology.adjacency(world)

    # -- coverage footprint quantities -----------------------------------------
    newly_val = overlap_val = None
    if "newly_covered" in requires or "overlap" in requires:
        fp = cheby_footprint(pos, world.grid_h, world.grid_w, cover_r)
        fp = fp & ~world.wall[None]
        if "newly_covered" in requires:
            newly_val = (
                (fp & ~prev.covered[None]).reshape(n, -1).sum(-1)
                .astype(jnp.float32)
            )
        if "overlap" in requires:
            seen_count = fp.sum(0)                                    # (H, W)
            others = (seen_count[None] - fp.astype(jnp.int32)) > 0    # (N, H, W)
            overlap_val = (
                (fp & others).reshape(n, -1).sum(-1).astype(jnp.float32)
            )

    # -- pass-throughs (channel / dynamics outputs) ----------------------------
    if "delivered" in requires and delivered is None:
        raise ValueError(
            "StepCtx requires 'delivered' but no delivered edges were supplied"
        )
    if "blocked" in requires and blocked is None:
        raise ValueError(
            "StepCtx requires 'blocked' but no blocked mask was supplied"
        )

    return StepCtx(
        dist=pairwise_dist(pos) if "dist" in requires else None,
        adj=adj_val if "adj" in requires else None,
        delivered=delivered if "delivered" in requires else None,
        reach=reach(adj_val) if "reach" in requires else None,
        collisions=collisions(pos) if "collisions" in requires else None,
        blocked=blocked if "blocked" in requires else None,
        newly_covered=newly_val,
        overlap=overlap_val,
    )
