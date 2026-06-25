"""
Comms — potential topology + delivery channels.

Two layers, deliberately separated (see docs/specs/2026-06-11-zymera-design.md):

* :class:`Topology` — *potential* adjacency: who COULD talk this step.
  Reward terms and connectivity metrics read this unless they explicitly
  opt into delivered edges.
* Channels (:class:`GossipChannel`, :class:`NullChannel`) — *realized*
  delivery: which edges actually carried payload (post-dropout). The
  delivered adjacency is what ``World.comm_graph`` records.

v1 scope (committed): payloads are OR-reducible ``(N, H, W)`` bool grids
only. A payload type zoo is the fork-drift failure mode this design exists
to prevent — extend the payload, don't fork the channel.

``GossipChannel(delay=1, dropout=0.0)`` reproduces zymera v0's gossip
bit-for-bit (``examples/comm_coverage.py`` step): delivery is each
neighbour's PREVIOUS cumulative shared map over NEW-position adjacency —
the diagonal self-loop keeps ``shared`` monotone, and the 1-step delay is
what makes information flood multi-hop over time.

Key protocol: ``deliver`` consumes its ``key`` ONLY when ``dropout > 0``
(trace-time gate), so the no-dropout path is deterministic given state —
exactly like v0, which had no channel randomness at all.
"""

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Tuple, runtime_checkable

import chex
import jax
import jax.numpy as jnp

# =============================================================================
# Potential topology
# =============================================================================


@runtime_checkable
class Topology(Protocol):
    """Who COULD talk this step. ``adjacency(world) -> (N, N) bool``,
    symmetric, diagonal True."""

    def adjacency(self, world) -> chex.Array:
        ...


@dataclass(frozen=True)
class DiskTopology:
    """Agents within ``radius`` of each other (in ``metric``) are adjacent.

    Matches v0 ``_adjacency``: Chebyshev distance, symmetric, diagonal True
    (an agent always "hears" itself — that self-loop is what keeps the
    gossip belief monotone).
    """

    radius: int
    metric: str = "chebyshev"

    def __post_init__(self):
        if self.metric != "chebyshev":
            raise NotImplementedError(
                f"DiskTopology metric {self.metric!r} not implemented; "
                "only 'chebyshev' is supported for now."
            )

    def adjacency(self, world) -> chex.Array:
        pos = world.body.position                       # (N, 2) int32
        dist = jnp.max(jnp.abs(pos[:, None, :] - pos[None, :, :]), axis=-1)
        return dist <= self.radius                      # (N, N) bool, diag True


# =============================================================================
# Channels
# =============================================================================


@chex.dataclass(frozen=True)
class ChannelState:
    """Gossip channel state. Lives at ``World.channel``."""

    shared: chex.Array          # (N, H, W) bool — post-gossip belief per agent
    buffer: chex.Array          # (delay, N, H, W) bool ring buffer; [0] is next out


@dataclass(frozen=True)
class GossipChannel:
    """Delayed, lossy flooding of bool grids over a potential topology.

    Per ``deliver``: every agent broadcasts its ``delay``-steps-old shared
    map to current neighbours; each edge (off-diagonal) survives an
    independent symmetric Bernoulli(1 - dropout) draw; the diagonal always
    delivers. ``delay=1, dropout=0.0`` ≡ v0 gossip exactly.

    ``bandwidth`` (top-k most-recent cells) needs int32 step-stamp maps the
    payload doesn't carry yet — reserved, must be ``None`` for now.
    """

    topology: Topology
    delay: int = 1
    dropout: float = 0.0
    bandwidth: Optional[int] = None

    def __post_init__(self):
        if self.delay < 1:
            raise ValueError(f"delay must be >= 1, got {self.delay}")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")
        if self.bandwidth is not None:
            raise NotImplementedError(
                "GossipChannel bandwidth is reserved but not implemented: "
                "fixed-shape top-k needs int32 step-stamped explored maps "
                "(stamps updated on receipt too), which the v1 bool-grid "
                "payload does not carry. Pass bandwidth=None."
            )

    # ---- channel API --------------------------------------------------------

    def init(self, world, outbox0: chex.Array) -> ChannelState:
        """Reset-time state: belief = own outbox; ring buffer pre-filled
        with it (so the first ``delay`` payloads are the reset snapshot)."""
        del world
        buffer = jnp.broadcast_to(
            outbox0[None], (self.delay,) + outbox0.shape
        )
        return ChannelState(shared=outbox0, buffer=buffer)

    def deliver(
        self, world, outbox: chex.Array, st: ChannelState, key: jax.Array,
    ) -> Tuple[chex.Array, ChannelState, chex.Array]:
        """One gossip tick.

        Returns ``(incoming, st', delivered_adj)``:

        * ``incoming`` — (N, H, W) bool, OR of delivering neighbours'
          ``delay``-old shared maps (self included via the diagonal).
        * ``st'`` — ``shared' = outbox | incoming``; buffer rolled.
        * ``delivered_adj`` — (N, N) bool realized edges (→ ``comm_graph``).
        """
        adj = self.topology.adjacency(world)            # (N, N) bool potential
        n = adj.shape[0]
        eye = jnp.eye(n, dtype=bool)

        # Per-EDGE symmetric dropout, off-diagonal only; diag always True.
        # Trace-time gate: the dropout==0 path consumes no randomness.
        if self.dropout > 0.0:
            u = jax.random.uniform(key, (n, n))
            keep = jnp.triu(u < (1.0 - self.dropout), k=1)   # upper tri only
            success = (adj & (keep | keep.T)) | eye          # mirror edge fate
        else:
            success = adj | eye

        payload = st.buffer[0]                          # (N, H, W) delay-old shared
        incoming = (success[:, :, None, None] & payload[None]).any(axis=1)
        shared = outbox | incoming                      # monotone via diag self-loop
        buffer = jnp.concatenate([st.buffer[1:], shared[None]], axis=0)
        return incoming, ChannelState(shared=shared, buffer=buffer), success


@dataclass(frozen=True)
class NullChannel:
    """No comms: every agent hears only itself. For non-comm missions."""

    def init(self, world, outbox0: chex.Array) -> Any:
        del world, outbox0
        return ()

    def deliver(
        self, world, outbox: chex.Array, st: Any, key: jax.Array,
    ) -> Tuple[chex.Array, Any, chex.Array]:
        del st, key
        n = world.body.position.shape[0]
        return outbox, (), jnp.eye(n, dtype=bool)
