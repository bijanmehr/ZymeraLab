"""
ObsBuilder + the named obs-channel registry.

Observations are composed from *named channels* — small pure functions
``fn(world, ctx) -> (N, H, W) | (H, W)`` looked up in :data:`CHANNEL_FNS`.
A new observation idea is a new channel registered from an experiment file
(:func:`register_channel`), never an edit to the simulator.

Builders (both frozen, hashable, closure-captured static objects):

* :class:`VectorObs` — v0 ``World._sensing``: ``(N, 3)`` float32 of own
  ``(row, col)`` + global visited fraction.
* :class:`GridObs`  — stacked per-agent planes ``(N, C, H, W)`` float32,
  plus an optional centralized ``(Cg, H, W)`` critic view (CTDE).

Doctrine: per-agent belief channels read the POST-GOSSIP belief at
``world.channel.shared`` (duck-typed attribute — any channel pytree with a
``.shared`` field works). Ground truth (``world.covered``, ``world.wall``)
is reserved for the *central* critic view; leaking it into ``agent_obs``
is how a "decentralized" policy quietly stops being one.

Parity: ``GridObs(("known", "own_pos", "known_walls", "neighbors",
"local_frontier"))`` reproduces v0 ``CommCoverageEnv._obs`` bit-for-bit;
the default ``central`` tuple reproduces v0 ``global_state``.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Protocol, Tuple, runtime_checkable

import chex
import jax.numpy as jnp

# =============================================================================
# Grid helpers (self-contained — obs must not depend on zymera.metrics)
# =============================================================================


def _onehot_pos(pos: chex.Array, h: int, w: int) -> chex.Array:
    """(N, H, W) bool — a single True at each agent's cell."""
    n = pos.shape[0]
    grid = jnp.zeros((n, h, w), dtype=jnp.bool_)
    return grid.at[jnp.arange(n), pos[:, 0], pos[:, 1]].set(True)


def _cheby_window(pos: chex.Array, h: int, w: int, r: int) -> chex.Array:
    """(N, H, W) bool — cells within Chebyshev radius ``r`` of each agent."""
    rows = jnp.arange(h)[None, :, None]
    cols = jnp.arange(w)[None, None, :]
    pr = pos[:, 0][:, None, None]
    pc = pos[:, 1][:, None, None]
    return jnp.maximum(jnp.abs(rows - pr), jnp.abs(cols - pc)) <= r


# =============================================================================
# Named channels
# =============================================================================
#
# Signature: fn(world, ctx) -> (N, H, W) per-agent planes or (H, W) team
# plane (broadcast per-agent when used in ``agent_obs``). ``ctx`` is the
# StepCtx (duck-typed); only channels that declare a dependency may read it
# ("neighbors" reads ``ctx.adj`` — potential topology, per doctrine #4).


def _known(world, ctx) -> chex.Array:
    """(N, H, W) — each agent's post-gossip belief map."""
    del ctx
    return world.channel.shared.astype(jnp.float32)


def _own_pos(world, ctx) -> chex.Array:
    """(N, H, W) — one-hot of each agent's own cell."""
    del ctx
    h, w = world.wall.shape
    return _onehot_pos(world.body.position, h, w).astype(jnp.float32)


def _known_walls(world, ctx) -> chex.Array:
    """(N, H, W) — wall cells the agent has actually learned about."""
    del ctx
    return (world.wall[None] & world.channel.shared).astype(jnp.float32)


def _neighbors(world, ctx) -> chex.Array:
    """(N, H, W) — one-hots of in-range teammates (potential adjacency,
    self excluded). Exactly v0: ``(adj_offdiag & onehot).any(1)``."""
    pos = world.body.position
    n = pos.shape[0]
    h, w = world.wall.shape
    adj = ctx.adj & ~jnp.eye(n, dtype=bool)
    onehot = _onehot_pos(pos, h, w)
    return (adj[:, :, None, None] & onehot[None, :, :, :]).any(axis=1).astype(jnp.float32)


def _local_frontier(world, ctx, sense_r: int = 1) -> chex.Array:
    """(N, H, W) — cells within ``sense_r`` of the agent it does NOT yet
    know: a direct "which way is fresh ground nearby" signal. ``sense_r``
    is bound by the owning :class:`GridObs` instance."""
    del ctx
    h, w = world.wall.shape
    window = _cheby_window(world.body.position, h, w, sense_r)
    return ((~world.channel.shared) & window).astype(jnp.float32)


def _occ_frontier(world, ctx) -> chex.Array:
    """(N, H, W) — the occupancy frontier of each agent's belief: known-FREE
    cells that border an UNKNOWN cell (a 4-neighbour not yet in the belief).

    Unlike :func:`_local_frontier` (egocentric "unknown within ``sense_r`` of
    me"), this is the *global* edge of the explored region — the set of true
    exploration targets (Yamauchi). It stays informative under ``sense_free``
    occupancy, where the egocentric window is already all-sensed so
    ``local_frontier`` collapses to empty. Free-ness and adjacency are read from
    the post-gossip belief (``shared`` ∩ ``~wall`` — the wall-ness of a *known*
    cell is itself known), never from unknown ground truth. Off-grid neighbours
    count as not-unknown, so the field edge is not a frontier — natural
    containment within the mission field."""
    del ctx
    shared = world.channel.shared                 # (N,H,W) known cells
    free = shared & ~world.wall[None]             # known-free
    unknown = ~shared
    nbr_unknown = (
        jnp.pad(unknown[:, 1:, :],  ((0, 0), (0, 1), (0, 0)))    # neighbour below
        | jnp.pad(unknown[:, :-1, :], ((0, 0), (1, 0), (0, 0)))  # neighbour above
        | jnp.pad(unknown[:, :, 1:],  ((0, 0), (0, 0), (0, 1)))  # neighbour right
        | jnp.pad(unknown[:, :, :-1], ((0, 0), (0, 0), (1, 0)))  # neighbour left
    )
    return (free & nbr_unknown).astype(jnp.float32)


def _team_explored(world, ctx) -> chex.Array:
    """(H, W) — ground-truth team coverage (``world.covered``)."""
    del ctx
    return world.covered.astype(jnp.float32)


def _all_pos(world, ctx) -> chex.Array:
    """(H, W) — any agent here."""
    del ctx
    h, w = world.wall.shape
    return _onehot_pos(world.body.position, h, w).any(0).astype(jnp.float32)


def _walls(world, ctx) -> chex.Array:
    """(H, W) — ground-truth obstacle mask."""
    del ctx
    return world.wall.astype(jnp.float32)


def _boundary(world, ctx) -> chex.Array:
    """(H, W) — the field boundary: the outer ring of grid cells.

    The LPAC backbone global-average-pools the CNN map (for scale-invariance),
    which strips absolute position — so agents are field-extent-blind and cannot
    tell a true map-edge from merely-unexplored interior. This static ring
    re-grounds "here is the edge of the world": stacked with ``own_pos`` /
    ``known`` the CNN reads near-boundary-ness locally, so a frontier policy can
    treat an edge frontier as a dead end and stay contained in the field. The
    ring is the edge at *any* grid size, so it is scale-invariant and transfers
    under a small→large warm-start (the regime where field-extent matters most).
    A team plane (same for every agent); broadcast per-agent in ``agent_obs``."""
    del ctx
    h, w = world.wall.shape
    ring = jnp.zeros((h, w), dtype=jnp.bool_)
    ring = ring.at[0, :].set(True).at[h - 1, :].set(True)
    ring = ring.at[:, 0].set(True).at[:, w - 1].set(True)
    return ring.astype(jnp.float32)


CHANNEL_FNS: Dict[str, Callable] = {
    "known":          _known,
    "own_pos":        _own_pos,
    "known_walls":    _known_walls,
    "neighbors":      _neighbors,
    "local_frontier": _local_frontier,
    "occ_frontier":   _occ_frontier,
    "team_explored":  _team_explored,
    "all_pos":        _all_pos,
    "walls":          _walls,
    "boundary":       _boundary,
}


def register_channel(name: str, fn: Callable) -> None:
    """Register a custom obs channel ``fn(world, ctx) -> (N, H, W) | (H, W)``.

    Experiment files add their channel ideas here; proven ones graduate
    into the table above with a test.
    """
    if name in CHANNEL_FNS:
        raise ValueError(f"obs channel already registered: {name!r}")
    CHANNEL_FNS[name] = fn


# =============================================================================
# ObsBuilder protocol
# =============================================================================


@runtime_checkable
class ObsBuilder(Protocol):
    """Composable observation builder (duck-typed; see module docstring).

    ``requires`` declares the StepCtx fields ``agent_obs``/``central_obs``
    read — it joins the env's union at ``__init__`` so unrequested context
    machinery never compiles.
    """

    requires: frozenset

    @property
    def obs_channels(self) -> int:
        """Per-agent channel count C (or feature dim D for vector obs)."""
        ...

    @property
    def central_channels(self) -> Optional[int]:
        """Centralized channel count Cg, or None when there is no critic view."""
        ...

    def agent_obs(self, world, ctx) -> chex.Array:
        """(N, C, H, W) — or (N, D) for vector builders."""
        ...

    def central_obs(self, world, ctx) -> Optional[chex.Array]:
        """(Cg, H, W) centralized critic view, or None."""
        ...


# =============================================================================
# Builders
# =============================================================================


@dataclass(frozen=True)
class VectorObs:
    """v0 ``World._sensing`` verbatim: ``(N, 3)`` float32 per-agent obs of
    own ``(row, col)`` + visited fraction (mean of ``world.visited`` over
    ALL cells, walls included — v0 parity)."""

    @property
    def requires(self) -> frozenset:
        return frozenset()

    @property
    def obs_channels(self) -> int:
        return 3

    @property
    def central_channels(self) -> Optional[int]:
        return None

    def agent_obs(self, world, ctx=None) -> chex.Array:
        del ctx
        n = world.body.position.shape[0]
        pos = world.body.position.astype(jnp.float32)
        coverage = jnp.mean(world.visited.astype(jnp.float32))
        coverage_col = jnp.broadcast_to(coverage, (n, 1))
        return jnp.concatenate([pos, coverage_col], axis=-1)

    def central_obs(self, world, ctx=None) -> None:
        del world, ctx
        return None


_DEFAULT_CENTRAL = ("team_explored", "all_pos", "walls")    # = v0 global_state


@dataclass(frozen=True)
class GridObs:
    """Stacked named channels: per-agent ``(N, C, H, W)`` float32 planes,
    plus an optional centralized ``(Cg, H, W)`` critic view.

    * ``channels`` — per-agent plane names, stacked in order. ``(H, W)``
      team planes broadcast to every agent.
    * ``sense_r``  — Chebyshev radius bound into the ``"local_frontier"``
      channel (the one channel parameterized by the builder).
    * ``central``  — team-plane names for ``central_obs``; ``None`` means
      no critic view. Each must produce an ``(H, W)`` plane.

    Channel names are validated at construction (Python time) — fail
    before the trace, not inside it.
    """

    channels: Tuple[str, ...]
    sense_r: int = 1
    central: Optional[Tuple[str, ...]] = _DEFAULT_CENTRAL

    def __post_init__(self):
        # Coerce to tuples so the component stays hashable (static-object rule).
        object.__setattr__(self, "channels", tuple(self.channels))
        if self.central is not None:
            object.__setattr__(self, "central", tuple(self.central))
        if self.sense_r < 0:
            raise ValueError(f"sense_r must be >= 0, got {self.sense_r}")
        names = self.channels + (self.central if self.central is not None else ())
        for name in names:
            if name not in CHANNEL_FNS:
                raise ValueError(
                    f"unknown obs channel {name!r}; available: "
                    f"{sorted(CHANNEL_FNS)} (add custom ones via "
                    "zymera.obs.register_channel)"
                )

    # ---- ObsBuilder protocol ------------------------------------------------

    @property
    def requires(self) -> frozenset:
        return frozenset({"adj"}) if "neighbors" in self.channels else frozenset()

    @property
    def obs_channels(self) -> int:
        return len(self.channels)

    @property
    def central_channels(self) -> Optional[int]:
        return None if self.central is None else len(self.central)

    def agent_obs(self, world, ctx) -> chex.Array:
        n = world.body.position.shape[0]
        h, w = world.wall.shape
        planes = []
        for name in self.channels:                       # static — unrolled at trace
            plane = self._channel(name, world, ctx)
            if plane.ndim == 2:                          # (H, W) team plane → per-agent
                plane = jnp.broadcast_to(plane, (n, h, w))
            planes.append(plane)
        return jnp.stack(planes, axis=1)                 # (N, C, H, W)

    def central_obs(self, world, ctx) -> Optional[chex.Array]:
        if self.central is None:                         # Python-time gate (static config)
            return None
        planes = []
        for name in self.central:
            plane = self._channel(name, world, ctx)
            if plane.ndim != 2:
                raise ValueError(
                    f"central channel {name!r} must be an (H, W) team plane, "
                    f"got shape {plane.shape}"
                )
            planes.append(plane)
        return jnp.stack(planes, axis=0)                 # (Cg, H, W)

    # ---- internals ------------------------------------------------------------

    def _channel(self, name: str, world, ctx) -> chex.Array:
        fn = CHANNEL_FNS[name]
        if name == "local_frontier":                     # the builder-parameterized one
            return fn(world, ctx, sense_r=self.sense_r)
        return fn(world, ctx)
