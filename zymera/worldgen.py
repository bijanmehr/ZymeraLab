"""
WorldGen — Terrain and Spawn components (the un-baked halves of ``reset``).

Two small protocols, each a family of frozen, hashable, trace-time-static
dataclasses (the static-object rule, spec §3.0):

* :class:`Terrain` — ``walls(key, h, w) -> (H, W) bool`` obstacle mask.
  Implementations: :class:`OpenTerrain`, :class:`RandomWalls`,
  :class:`MapFile`, :class:`Rooms`.
* :class:`Spawn` — ``positions(key, wall, n_agents) -> (N, 2) int32``
  distinct free cells. Implementations: :class:`ScatterSpawn`,
  :class:`ClusterSpawn`, :class:`FixedSpawn`.

Parity lineage (bit-for-bit, gated by ``tests/test_worldgen.py``):
``RandomWalls`` ports v0 ``zymera.env._random_wall``; ``ScatterSpawn`` ports
the v0 ``World.initial`` weighted-choice spawn; ``ClusterSpawn`` ports
``examples/comm_coverage.py::CommCoverageEnv._cluster_spawn`` verbatim —
including its internal ``akey, ckey = split(key)``, the uniform tie-break
noise, and the ``lax.top_k`` overflow behaviour.

All methods are pure JAX (jit/vmap/scan-safe); ``h``/``w``/``n_agents`` are
static Python ints, only ``key``/``wall`` are traced.
"""

from dataclasses import dataclass
from typing import Protocol, Tuple

import chex
import jax
import jax.numpy as jnp

# =============================================================================
# Protocols
# =============================================================================


class Terrain(Protocol):
    """Obstacle-mask generator. ``h``/``w`` are static; ``key`` may be ignored."""

    def walls(self, key: jax.Array, h: int, w: int) -> chex.Array:
        """Return an ``(H, W)`` bool mask — True where a wall blocks the cell."""
        ...


class Spawn(Protocol):
    """Initial-position generator over a terrain's free cells."""

    def positions(self, key: jax.Array, wall: chex.Array, n_agents: int) -> chex.Array:
        """Return ``(N, 2)`` int32 ``(row, col)`` — distinct, free cells."""
        ...


def _flat_to_rc(flat: chex.Array, w: int) -> chex.Array:
    """Flat cell indices ``(N,)`` → ``(N, 2)`` int32 ``(row, col)``."""
    return jnp.stack([flat // w, flat % w], axis=-1).astype(jnp.int32)


# =============================================================================
# Terrain implementations
# =============================================================================


@dataclass(frozen=True)
class OpenTerrain:
    """No obstacles — the all-False mask. Ignores the key."""

    def walls(self, key: jax.Array, h: int, w: int) -> chex.Array:
        del key
        return jnp.zeros((h, w), dtype=jnp.bool_)


@dataclass(frozen=True)
class RandomWalls:
    """``n_obstacles`` uniformly random wall cells, fresh each reset key.

    v0 ``_random_wall`` verbatim: flat indices drawn by ``jax.random.choice``
    without replacement, then scattered into the mask. ``n_obstacles`` is
    clamped to the cell count at trace time.
    """

    n_obstacles: int

    def __post_init__(self):
        if self.n_obstacles < 0:
            raise ValueError(f"n_obstacles must be >= 0, got {self.n_obstacles}")

    def walls(self, key: jax.Array, h: int, w: int) -> chex.Array:
        n = min(self.n_obstacles, h * w)
        idx = jax.random.choice(key, h * w, shape=(n,), replace=False)
        flat = jnp.zeros((h * w,), dtype=jnp.bool_).at[idx].set(True)
        return flat.reshape(h, w)


@dataclass(frozen=True)
class MapFile:
    """A fixed map, stored as nested tuples so the component stays hashable
    (spec §3.0 — components that conceptually hold arrays store tuples and
    materialize inside methods). Ignores the key.

    Build via :meth:`from_string` (``'#'`` = wall, ``'.'`` = free,
    whitespace-only rows skipped) or :meth:`load` (same format, from a file).
    """

    cells: Tuple[Tuple[bool, ...], ...]    # row-major; True = wall

    def __post_init__(self):
        if not self.cells or not self.cells[0]:
            raise ValueError("MapFile needs at least one non-empty row")
        widths = {len(row) for row in self.cells}
        if len(widths) != 1:
            raise ValueError(f"MapFile rows must be equal length, got widths {sorted(widths)}")

    @classmethod
    def from_string(cls, s: str) -> "MapFile":
        """Parse a map drawing: ``'#'`` = wall, ``'.'`` = free; blank rows skipped."""
        rows = []
        for line in s.splitlines():
            line = line.strip()
            if not line:
                continue
            bad = set(line) - {"#", "."}
            if bad:
                raise ValueError(f"MapFile: unknown chars {sorted(bad)} (use '#' and '.')")
            rows.append(tuple(c == "#" for c in line))
        return cls(cells=tuple(rows))

    @classmethod
    def load(cls, path) -> "MapFile":
        """Read :meth:`from_string` format from ``path``."""
        with open(path, "r") as f:
            return cls.from_string(f.read())

    @property
    def grid_h(self) -> int:
        return len(self.cells)

    @property
    def grid_w(self) -> int:
        return len(self.cells[0])

    def walls(self, key: jax.Array, h: int, w: int) -> chex.Array:
        del key
        if (h, w) != (self.grid_h, self.grid_w):
            raise ValueError(
                f"MapFile is {self.grid_h}x{self.grid_w} but env asked for {h}x{w}"
            )
        return jnp.array(self.cells, dtype=jnp.bool_)


@dataclass(frozen=True)
class Rooms:
    """``rooms`` equal-width rooms separated by full-height vertical walls,
    one ``door_w``-tall door per wall at a key-driven row.

    Deliberately simple (spec §3.1): wall ``i`` (of ``rooms - 1``) sits at
    column ``(i + 1) * w // rooms``; the door's top row is drawn uniformly
    from ``[0, h - door_w]`` per wall from the reset key. ``rooms=1`` is the
    open grid. Doors keep the free cells connected, so any spawn works.
    """

    rooms: int = 2
    door_w: int = 1

    def __post_init__(self):
        if self.rooms < 1:
            raise ValueError(f"rooms must be >= 1, got {self.rooms}")
        if self.door_w < 1:
            raise ValueError(f"door_w must be >= 1, got {self.door_w}")

    def walls(self, key: jax.Array, h: int, w: int) -> chex.Array:
        n_walls = self.rooms - 1
        if n_walls == 0:
            return jnp.zeros((h, w), dtype=jnp.bool_)
        if self.rooms > w:
            raise ValueError(f"cannot fit {self.rooms} rooms in width {w}")
        cols = jnp.array([(i + 1) * w // self.rooms for i in range(n_walls)])
        # One door per wall: top row uniform in [0, h - door_w] (whole column
        # opens when door_w >= h). Mask arithmetic — no traced-value branching.
        door_top = jax.random.randint(key, (n_walls,), 0, max(h - self.door_w, 0) + 1)
        rows = jnp.arange(h)
        door = (rows[None, :] >= door_top[:, None]) & (
            rows[None, :] < door_top[:, None] + self.door_w
        )                                                       # (n_walls, H)
        return jnp.zeros((h, w), dtype=jnp.bool_).at[:, cols].set(~door.T)


# =============================================================================
# Spawn implementations
# =============================================================================


@dataclass(frozen=True)
class ScatterSpawn:
    """Distinct free cells anywhere on the grid.

    v0 ``World.initial`` verbatim: flat indices sampled without replacement,
    weighted to free cells (wall cells get probability zero).
    """

    def positions(self, key: jax.Array, wall: chex.Array, n_agents: int) -> chex.Array:
        h, w = wall.shape
        free = (~wall).reshape(-1).astype(jnp.float32)
        probs = free / jnp.sum(free)
        flat = jax.random.choice(key, h * w, shape=(n_agents,), replace=False, p=probs)
        return _flat_to_rc(flat, w)


@dataclass(frozen=True)
class ClusterSpawn:
    """N distinct free cells clustered within ``radius`` of a random interior
    anchor.

    v0 ``_cluster_spawn`` verbatim: ``akey, ckey = split(key)``; anchor drawn
    from free ∩ interior cells; every cell scored
    block-and-free (noise + 1) ≫ free (noise) ≫ wall (−1) with uniform
    tie-break noise; ``lax.top_k`` takes the N best. Guarantees N distinct
    cells and degrades gracefully when obstacles crowd the patch (overflow
    spills to the highest-noise free cells anywhere on the grid).
    """

    radius: int

    def __post_init__(self):
        if self.radius < 0:
            raise ValueError(f"radius must be >= 0, got {self.radius}")

    def positions(self, key: jax.Array, wall: chex.Array, n_agents: int) -> chex.Array:
        h, w = wall.shape
        r = self.radius
        akey, ckey = jax.random.split(key)
        free = ~wall
        interior = jnp.zeros((h, w), bool).at[r:h - r, r:w - r].set(True)
        ap = (free & interior).reshape(-1).astype(jnp.float32)
        anchor = jax.random.choice(akey, h * w, p=ap / jnp.sum(ap))
        ar, ac = anchor // w, anchor % w
        rows = jnp.arange(h)[:, None]
        cols = jnp.arange(w)[None, :]
        block = (jnp.maximum(jnp.abs(rows - ar), jnp.abs(cols - ac)) <= r) & free
        noise = jax.random.uniform(ckey, (h * w,))
        score = jnp.where(block.reshape(-1), noise + 1.0,
                          jnp.where(free.reshape(-1), noise, -1.0))
        flat = jax.lax.top_k(score, n_agents)[1].astype(jnp.int32)
        return _flat_to_rc(flat, w)


@dataclass(frozen=True)
class FixedSpawn:
    """Spawn at explicit ``(row, col)`` cells, in order. Ignores the key.

    Cells must be distinct; the first ``n_agents`` of them are used (so one
    component can serve a ladder of team sizes). The caller is responsible
    for the cells being free on the paired terrain — checked by tests, not
    at trace time (the wall array is traced).
    """

    cells: Tuple[Tuple[int, int], ...]

    def __post_init__(self):
        if len(set(self.cells)) != len(self.cells):
            raise ValueError(f"FixedSpawn cells must be distinct, got {self.cells}")

    def positions(self, key: jax.Array, wall: chex.Array, n_agents: int) -> chex.Array:
        del key, wall
        if n_agents > len(self.cells):
            raise ValueError(
                f"FixedSpawn has {len(self.cells)} cells but {n_agents} agents asked"
            )
        return jnp.array(self.cells[:n_agents], dtype=jnp.int32)
