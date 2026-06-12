"""kymera.obs — parity vs zymera v0 goldens + builder behavior.

Parity gates (the bit-exact contract):

* GridObs("known","own_pos","known_walls","neighbors","local_frontier")
  == golden ``commcov.npz`` obs (v0 ``CommCoverageEnv._obs``), EXACT.
* VectorObs == golden ``empty.npz`` obs (v0 ``World._sensing``), atol 1e-6.
* GridObs default central == v0 ``global_state`` semantics.

Worlds are built directly from the golden arrays — no env / metrics / comms
imports (those modules are built concurrently). ``world.channel`` is a tiny
one-field stand-in pytree exposing ``.shared`` (obs only duck-types it);
``ctx`` likewise only needs ``.adj``.
"""

from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kymera.env import Body, World
from kymera.obs import CHANNEL_FNS, GridObs, VectorObs, register_channel

GOLDEN = Path(__file__).parent / "golden"

V0_CHANNELS = ("known", "own_pos", "known_walls", "neighbors", "local_frontier")


# ---- minimal pytree stand-ins ------------------------------------------------


@chex.dataclass(frozen=True)
class _Chan:
    """Channel stand-in — obs only duck-types ``.shared``."""

    shared: chex.Array          # (N, H, W) bool


@chex.dataclass(frozen=True)
class _Ctx:
    """StepCtx stand-in — the 'neighbors' channel only reads ``.adj``."""

    adj: chex.Array             # (N, N) bool


def _adjacency(pos, r):
    """Chebyshev disk adjacency, diag True (inline — kymera.metrics may not
    exist yet while modules build in parallel)."""
    d = jnp.max(jnp.abs(pos[:, None, :] - pos[None, :, :]), axis=-1)
    return d <= r


# ---- golden loaders ------------------------------------------------------------


@pytest.fixture(scope="module")
def commcov():
    return np.load(GOLDEN / "commcov.npz")


@pytest.fixture(scope="module")
def empty():
    return np.load(GOLDEN / "empty.npz")


def _commcov_world(g, t):
    """World at trajectory step t of the commcov golden rollout."""
    h, w = g["wall_k0"].shape
    n = g["positions"].shape[1]
    return World(
        body=Body(
            position=jnp.asarray(g["positions"][t], jnp.int32),
            energy=jnp.zeros((n,), jnp.float32),
        ),
        explored=jnp.zeros((h, w), jnp.int32),           # unused by these channels
        seen_by=jnp.asarray(g["explored_by"][t]),
        wall=jnp.asarray(g["wall_k0"]),
        comm_graph=jnp.zeros((n, n), jnp.bool_),
        step_count=jnp.asarray(t, jnp.int32),
        channel=_Chan(shared=jnp.asarray(g["shared"][t])),
        mission=(),
        group=jnp.zeros((n,), jnp.int32),
    )


def _empty_world(g, t):
    """World at trajectory step t of the empty-env golden rollout."""
    h, w = g["explored"].shape[1:]
    n = g["positions"].shape[1]
    return World(
        body=Body(
            position=jnp.asarray(g["positions"][t], jnp.int32),
            energy=jnp.zeros((n,), jnp.float32),
        ),
        explored=jnp.asarray(g["explored"][t], jnp.int32),   # visit counts → .visited
        seen_by=jnp.zeros((n, h, w), jnp.bool_),
        wall=jnp.zeros((h, w), jnp.bool_),
        comm_graph=jnp.zeros((n, n), jnp.bool_),
        step_count=jnp.asarray(t, jnp.int32),
        channel=(),                                          # VectorObs never reads it
        mission=(),
        group=jnp.zeros((n,), jnp.int32),
    )


# =============================================================================
# Parity gate 1 — GridObs vs v0 CommCoverageEnv._obs (EXACT)
# =============================================================================


@pytest.mark.parametrize("t", [0, 1, 35, 70])
def test_grid_obs_matches_golden(commcov, t):
    comm_r = int(commcov["cfg"][4])
    sense_r = int(commcov["cfg"][5])
    builder = GridObs(V0_CHANNELS, sense_r=sense_r)
    world = _commcov_world(commcov, t)
    ctx = _Ctx(adj=_adjacency(world.body.position, comm_r))
    got = np.asarray(builder.agent_obs(world, ctx))
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, commcov["obs"][t])


def test_grid_obs_jit_matches_eager(commcov):
    """Pure-JAX discipline: the builder closes over static config only."""
    comm_r = int(commcov["cfg"][4])
    builder = GridObs(V0_CHANNELS, sense_r=1)
    world = _commcov_world(commcov, 35)
    ctx = _Ctx(adj=_adjacency(world.body.position, comm_r))
    eager = builder.agent_obs(world, ctx)
    jitted = jax.jit(builder.agent_obs)(world, ctx)
    np.testing.assert_array_equal(np.asarray(jitted), np.asarray(eager))


# =============================================================================
# Parity gate 2 — VectorObs vs v0 World._sensing
# =============================================================================


@pytest.mark.parametrize("t", [0, 10, 20])
def test_vector_obs_matches_golden(empty, t):
    builder = VectorObs()
    world = _empty_world(empty, t)
    got = np.asarray(builder.agent_obs(world, None))
    assert got.shape == (4, 3) and got.dtype == np.float32
    np.testing.assert_allclose(got, empty["obs"][t], atol=1e-6)


def test_vector_obs_no_central(empty):
    builder = VectorObs()
    assert builder.requires == frozenset()
    assert builder.obs_channels == 3
    assert builder.central_channels is None
    assert builder.central_obs(_empty_world(empty, 0), None) is None


# =============================================================================
# Parity gate 3 — central_obs vs v0 global_state
# =============================================================================


def test_central_obs_matches_v0_global_state(commcov):
    t = 35
    builder = GridObs(V0_CHANNELS, sense_r=1)        # default central triple
    world = _commcov_world(commcov, t)               # seen_by := golden explored_by[t]
    cen = np.asarray(builder.central_obs(world, None))

    h, w = commcov["wall_k0"].shape
    assert cen.shape == (3, h, w) and cen.dtype == np.float32
    assert builder.central_channels == 3

    # team_explored = union of per-agent own knowledge (== golden team map)
    team = commcov["explored_by"][t].any(0)
    np.testing.assert_array_equal(cen[0], team.astype(np.float32))
    np.testing.assert_array_equal(cen[0], commcov["team_explored"][t].astype(np.float32))
    # all_pos = any-agent one-hot
    allp = np.zeros((h, w), np.float32)
    allp[commcov["positions"][t, :, 0], commcov["positions"][t, :, 1]] = 1.0
    np.testing.assert_array_equal(cen[1], allp)
    # walls (all-False in this golden config)
    np.testing.assert_array_equal(cen[2], commcov["wall_k0"].astype(np.float32))


# =============================================================================
# Builder behavior
# =============================================================================


def test_requires_gating():
    assert GridObs(V0_CHANNELS).requires == frozenset({"adj"})
    assert GridObs(("known", "own_pos")).requires == frozenset()


def test_channel_counts_and_no_central():
    builder = GridObs(("known", "own_pos"), central=None)
    assert builder.obs_channels == 2
    assert builder.central_channels is None
    assert builder.central_obs(None, None) is None    # Python-time gate, no world touch


def test_unknown_channel_rejected_at_construction():
    with pytest.raises(ValueError, match="unknown obs channel 'fog_of_war'"):
        GridObs(("known", "fog_of_war"))
    with pytest.raises(ValueError, match="unknown obs channel"):
        GridObs(("known",), central=("not_a_channel",))


def test_team_plane_broadcasts_per_agent(commcov):
    world = _commcov_world(commcov, 35)
    builder = GridObs(("walls", "team_explored", "all_pos"), central=None)
    out = np.asarray(builder.agent_obs(world, None))
    n = world.n_agents
    assert out.shape == (n, 3, 16, 16)
    for i in range(1, n):                             # team planes identical per agent
        np.testing.assert_array_equal(out[i], out[0])
    np.testing.assert_array_equal(
        out[0, 1], commcov["explored_by"][35].any(0).astype(np.float32)
    )


def test_register_channel_duplicate_raises():
    with pytest.raises(ValueError, match="already registered"):
        register_channel("known", lambda world, ctx: None)


def test_register_custom_channel_usable(commcov):
    name = "test_obs_custom_ones"
    register_channel(name, lambda world, ctx: jnp.ones_like(world.wall, jnp.float32))
    try:
        world = _commcov_world(commcov, 0)
        out = np.asarray(GridObs((name, "own_pos"), central=None).agent_obs(world, None))
        assert out.shape == (4, 2, 16, 16)
        assert (out[:, 0] == 1.0).all()
    finally:
        CHANNEL_FNS.pop(name, None)                   # keep the registry clean for other tests


def test_grid_obs_is_hashable_static_object():
    """Static-object rule: builders are trace-time constants."""
    a = GridObs(V0_CHANNELS, sense_r=1)
    b = GridObs(list(V0_CHANNELS), sense_r=1)         # list coerces to tuple
    assert hash(a) == hash(b) and a == b
