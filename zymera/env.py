"""
Env contract, state schema, and registry — the spine of zymera.

This module owns the FROZEN contracts everything else conforms to
(see docs/specs/2026-06-11-zymera-design.md):

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
      post-dropout). Potential topology lives in :class:`zymera.metrics.StepCtx`.
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


# =============================================================================
# GridEnv — the orchestrator over the five components
# =============================================================================

# Component imports sit BELOW the core definitions so modules that import
# names from zymera.env (e.g. dynamics -> ACTION_DELTAS) resolve them during
# package initialization.
from . import metrics as _metrics                                  # noqa: E402
from .comms import DiskTopology, GossipChannel, NullChannel        # noqa: E402
from .dynamics import GridDynamics, NoCollision, SequentialClaim   # noqa: E402
from .missions import Mission, RewardTerm                          # noqa: E402
from .obs import GridObs, VectorObs                                # noqa: E402
from .worldgen import ClusterSpawn, OpenTerrain, RandomWalls, ScatterSpawn  # noqa: E402

# Obs channels that read the post-gossip belief (world.channel.shared) and
# therefore need a payload-carrying channel, not NullChannel.
_BELIEF_OBS_CHANNELS = frozenset({"known", "known_walls", "local_frontier"})


class GridEnv(Env):
    """Square-grid env composed from the five swappable components.

    ::

        env = GridEnv(grid_h=16, grid_w=16, n_agents=4,
                      spawn=ClusterSpawn(2),
                      dynamics=GridDynamics(collision=SequentialClaim()),
                      channel=GossipChannel(DiskTopology(5)),
                      obs=GridObs(("known", "own_pos", "known_walls",
                                   "neighbors", "local_frontier")),
                      mission=Mission(terms=DEFAULT_TERMS))

    One step runs: dynamics -> visit counts -> cover footprint -> channel
    delivery (``comm_graph`` := delivered edges) -> ``metrics.derive`` (once)
    -> mission update/reward/done -> obs. Per-term UNWEIGHTED rewards land in
    ``info["reward_terms"]``; mission success metrics in ``info["metrics"]``.

    ``cover_r`` is the coverage footprint radius (core grid physics): the
    cells within Chebyshev ``cover_r`` of an agent count as covered by it
    (``World.seen_by``). ``cover_r=0`` covers only the agent's own cell.
    """

    def __init__(self, *, grid_h: int = 8, grid_w: int = 8, n_agents: int = 1,
                 cover_r: int = 0, terrain=None, spawn=None, dynamics=None,
                 channel=None, obs=None, mission=None):
        self.grid_h, self.grid_w = int(grid_h), int(grid_w)
        self.n_agents = int(n_agents)
        self.cover_r = int(cover_r)
        self.terrain = terrain if terrain is not None else OpenTerrain()
        self.spawn = spawn if spawn is not None else ScatterSpawn()
        self.dynamics = dynamics if dynamics is not None else GridDynamics()
        self.channel = channel if channel is not None else NullChannel()
        self.obs = obs if obs is not None else VectorObs()
        self.mission = mission if mission is not None else Mission(terms=())
        # Topology comes from the channel; group assignment from the mission.
        self._topology = getattr(self.channel, "topology", None)
        self._assignment = getattr(self.mission, "assignment", None)
        self.requires = frozenset(self.obs.requires) | frozenset(self.mission.requires)
        self._validate()

    # --- construction-time validation (fail at Python time, helpfully) ------

    def _validate(self) -> None:
        if ({"adj", "reach"} & self.requires) and self._topology is None:
            raise ValueError(
                "obs/mission require the comm topology "
                f"({sorted({'adj', 'reach'} & self.requires)}) but the channel "
                f"{type(self.channel).__name__} has none — use e.g. "
                "GossipChannel(DiskTopology(radius))."
            )
        obs_channels = frozenset(getattr(self.obs, "channels", ()))
        if (obs_channels & _BELIEF_OBS_CHANNELS) and isinstance(self.channel, NullChannel):
            raise ValueError(
                f"obs channels {sorted(obs_channels & _BELIEF_OBS_CHANNELS)} read "
                "the post-gossip belief (world.channel.shared) — NullChannel "
                "carries none. Use a GossipChannel."
            )
        if self.cover_r < 0:
            raise ValueError(f"cover_r must be >= 0, got {self.cover_r}")

    # --- helpers -------------------------------------------------------------

    def _ctx(self, prev: World, world: World, *, delivered, blocked) -> "_metrics.StepCtx":
        return _metrics.derive(
            prev, world, self.requires, topology=self._topology,
            cover_r=self.cover_r, delivered=delivered, blocked=blocked,
        )

    def _footprint(self, pos: jax.Array, wall: jax.Array) -> jax.Array:
        fp = _metrics.cheby_footprint(pos, self.grid_h, self.grid_w, self.cover_r)
        return fp & ~wall[None]

    # --- gym API (key protocol FROZEN — see module docstring) ----------------

    def reset(self, key: jax.Array) -> Tuple[jax.Array, World]:
        wkey, skey = jax.random.split(key)
        wall = jnp.asarray(self.terrain.walls(wkey, self.grid_h, self.grid_w), dtype=jnp.bool_)
        pos = self.spawn.positions(skey, wall, self.n_agents)
        if self._assignment is not None:
            group = self._assignment.assign(jax.random.fold_in(key, 1), self.n_agents)
        else:
            group = jnp.zeros((self.n_agents,), dtype=jnp.int32)
        r, c = pos[:, 0], pos[:, 1]
        body = Body(position=pos, energy=jnp.zeros((self.n_agents,), jnp.float32))
        explored = jnp.zeros((self.grid_h, self.grid_w), jnp.int32).at[r, c].add(1)
        seen0 = self._footprint(pos, wall)
        world = World(
            body=body, explored=explored, seen_by=seen0, wall=wall,
            comm_graph=jnp.zeros((self.n_agents, self.n_agents), dtype=jnp.bool_),
            step_count=jnp.zeros((), jnp.int32), channel=(), mission=(), group=group,
        )
        world = world.replace(channel=self.channel.init(world, seen0))
        world = world.replace(mission=self.mission.init_state(jax.random.fold_in(key, 2), world))
        ctx = self._ctx(world, world, delivered=world.comm_graph,
                        blocked=jnp.zeros((self.n_agents,), dtype=jnp.bool_))
        return self.obs.agent_obs(world, ctx), world

    def step(
        self, state: World, action: jax.Array, key: jax.Array,
    ) -> Tuple[jax.Array, World, jax.Array, jax.Array, Dict[str, Any]]:
        k_chan, k_mis = jax.random.split(key)
        body1, blocked = self.dynamics.step(state, action)
        r, c = body1.position[:, 0], body1.position[:, 1]
        explored1 = state.explored.at[r, c].add(1)
        seen1 = state.seen_by | self._footprint(body1.position, state.wall)
        w = state.replace(body=body1, explored=explored1, seen_by=seen1,
                          step_count=state.step_count + 1)
        _incoming, chan1, delivered = self.channel.deliver(w, seen1, state.channel, k_chan)
        w = w.replace(channel=chan1, comm_graph=delivered)
        ctx = self._ctx(state, w, delivered=delivered, blocked=blocked)
        mstate1 = self.mission.update(state, w, ctx, state.mission, k_mis)
        world1 = w.replace(mission=mstate1)
        reward, terms = self.mission.reward(state, world1, action, ctx, mstate1)
        done = self.mission.done(world1, ctx, mstate1)
        obs = self.obs.agent_obs(world1, ctx)
        info = {
            "explored":     world1.explored,
            "step_count":   world1.step_count,
            "seen_by":      world1.seen_by,
            "comm_graph":   delivered,
            "reward_terms": terms,
            "metrics":      self.mission.metrics(world1, ctx, mstate1),
        }
        return obs, world1, reward, done, info

    # --- conveniences ---------------------------------------------------------

    def action_mask(self, state: World) -> jax.Array:
        """(N, A) bool physical validity — delegates to dynamics."""
        return self.dynamics.action_mask(state)

    def central_obs(self, state: World) -> jax.Array:
        """Centralized critic view — delegates to the obs builder."""
        return self.obs.central_obs(state, None)

    def annotations(self, state: World):
        """Mission overlay primitives for viz."""
        return self.mission.annotations(state, state.mission)

    def __repr__(self) -> str:
        terms = ", ".join(f"{t.name}×{t.weight:g}" for t in self.mission.terms)
        return (
            f"GridEnv({self.grid_h}×{self.grid_w}, n_agents={self.n_agents}, "
            f"cover_r={self.cover_r},\n"
            f"  terrain={self.terrain!r}, spawn={self.spawn!r},\n"
            f"  dynamics={self.dynamics!r},\n"
            f"  channel={self.channel!r},\n"
            f"  obs={self.obs!r},\n"
            f"  mission=[{terms}])"
        )


# --- Env-level spec()/replace() for recipe-built envs -------------------------

def _env_spec(self) -> Dict[str, Any]:
    """``{"recipe": name, **kwargs}`` — round-trips through :func:`make_from`."""
    if getattr(self, "_recipe", None) is None:
        raise ValueError(
            "spec()/replace() need a recipe-built env (zymera.make(...)); "
            "this env was constructed directly."
        )
    name, kw = self._recipe
    return {"recipe": name, **kw}


def _env_replace(self, **overrides) -> Env:
    """Rebuild this recipe env with some kwargs overridden."""
    _env_spec(self)  # validates this is a recipe-built env
    name, kw = self._recipe
    return make(name, **{**kw, **overrides})


Env.spec = _env_spec
Env.replace = _env_replace


# =============================================================================
# Recipes
# =============================================================================

# Term shorthand: ("name", weight) or ("name", weight, {params}) tuples are
# resolved against zymera.missions_terms; RewardTerm objects pass through.
_PLAIN_TERMS = {
    "coverage":     "new_coverage",
    "connectivity": "reach_fraction",
    "collision":    "collision_count",
    "overlap":      "same_step_overlap",
}


def _resolve_terms(terms, comm_r: int) -> tuple:
    from . import missions_terms as mt

    resolved = []
    for t in terms:
        if isinstance(t, RewardTerm):
            resolved.append(t)
            continue
        name, weight, *rest = t
        params = dict(rest[0]) if rest else {}
        if name in _PLAIN_TERMS:
            fn = getattr(mt, _PLAIN_TERMS[name])
        elif name == "capped_giant":
            fn = mt.capped_giant(**params)
        elif name == "cohesion":
            fn = mt.cohesion_leash(**{"leash": 4.0, "comm_r": comm_r, **params})
        elif name == "degree":
            fn = mt.degree_floor(**{"floor": 1.0, "comm_r": comm_r, **params})
        elif name == "pbrs_frontier":
            fn = mt.pbrs(mt.phi_nearest_frontier, params.get("gamma", 0.99))
        elif name == "pbrs_field":
            fn = mt.pbrs(mt.phi_field_mean, params.get("gamma", 0.99))
        elif name == "cbf_conn":
            fn = mt.cbf_conn(**{"alpha": 0.5, "eps": 0.1, "sharp": 2.0,
                                "comm_r": comm_r, **params})
        elif name == "cbf_coll":
            fn = mt.cbf_coll(**{"alpha": 0.5, "dmin": 1.0, **params})
        else:
            raise ValueError(
                f"unknown term shorthand {name!r}; known: "
                f"{sorted(_PLAIN_TERMS)} + ['capped_giant', 'cohesion', 'degree', "
                "'pbrs_frontier', 'pbrs_field', 'cbf_conn', 'cbf_coll'] "
                "(or pass a zymera.RewardTerm directly)"
            )
        resolved.append(RewardTerm(name=name, weight=float(weight), fn=fn,
                                   requires=getattr(fn, "requires", frozenset())))
    return tuple(resolved)


def _empty_recipe(grid_h: int = 8, grid_w: int = 8, n_agents: int = 1,
                  n_obstacles: int = 0, max_steps=None) -> GridEnv:
    """Open grid, vector obs, no reward — the hello-world env."""
    return GridEnv(
        grid_h=grid_h, grid_w=grid_w, n_agents=n_agents, cover_r=0,
        terrain=RandomWalls(n_obstacles) if n_obstacles else OpenTerrain(),
        spawn=ScatterSpawn(), obs=VectorObs(),
        mission=Mission(terms=(), max_steps=max_steps),
    )


def _comm_coverage_recipe(grid: int = 16, n_agents: int = 4, comm_r: int = 5,
                          cover_r: int = 0, sense_r: int = 1, n_obstacles: int = 0,
                          spawn_radius=2, collision: str = "none",
                          delay: int = 1, dropout: float = 0.0, bandwidth=None,
                          max_steps=None, terms=None) -> GridEnv:
    """Cooperative coverage with range-limited gossip (the zymera-v0 task)."""
    from . import missions_terms as mt

    if collision not in ("none", "block"):
        raise ValueError(f'collision must be "none" or "block", got {collision!r} '
                         "(collision-free SAMPLING is the trainer's mask_collisions flag)")
    term_tuple = mt.DEFAULT_TERMS if terms is None else _resolve_terms(terms, comm_r)
    return GridEnv(
        grid_h=grid, grid_w=grid, n_agents=n_agents, cover_r=cover_r,
        terrain=RandomWalls(n_obstacles) if n_obstacles else OpenTerrain(),
        spawn=ClusterSpawn(spawn_radius) if spawn_radius is not None else ScatterSpawn(),
        dynamics=GridDynamics(collision=SequentialClaim() if collision == "block"
                              else NoCollision()),
        channel=GossipChannel(DiskTopology(comm_r), delay=delay, dropout=dropout,
                              bandwidth=bandwidth),
        obs=GridObs(channels=("known", "own_pos", "known_walls", "neighbors",
                              "local_frontier"),
                    sense_r=sense_r, central=("team_explored", "all_pos", "walls")),
        mission=Mission(terms=term_tuple, max_steps=max_steps),
    )


register_env("empty", _empty_recipe)
register_env("comm-coverage", _comm_coverage_recipe)
