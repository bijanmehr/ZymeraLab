"""worldgen tests — v0 spawn/wall bit-parity (the Task 4 gate) + invariants.

Parity goldens (``tests/golden/``) were dumped read-only from zymera v0:
``reset(PRNGKey(s))`` splits ``wkey, skey``; spawns consume ``skey``.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kymera import worldgen

GOLDEN = Path(__file__).parent / "golden"


def _skey(seed: int) -> jax.Array:
    """The spawn key v0 reset(PRNGKey(seed)) hands to the spawn logic."""
    _wkey, skey = jax.random.split(jax.random.PRNGKey(seed))
    return skey


def _wkey(seed: int) -> jax.Array:
    wkey, _skey = jax.random.split(jax.random.PRNGKey(seed))
    return wkey


# =============================================================================
# Parity gates (bit-for-bit vs zymera v0 goldens)
# =============================================================================


class TestSpawnParity:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_cluster_spawn_matches_commcov_golden(self, seed):
        golden = np.load(GOLDEN / "commcov.npz")
        wall = jnp.zeros((16, 16), dtype=jnp.bool_)
        pos = worldgen.ClusterSpawn(radius=2).positions(_skey(seed), wall, 4)
        np.testing.assert_array_equal(np.asarray(pos), golden[f"pos0_k{seed}"])

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_scatter_spawn_matches_empty_golden(self, seed):
        golden = np.load(GOLDEN / "empty.npz")
        wall = jnp.zeros((8, 8), dtype=jnp.bool_)
        pos = worldgen.ScatterSpawn().positions(_skey(seed), wall, 4)
        np.testing.assert_array_equal(np.asarray(pos), golden[f"pos0_k{seed}"])

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_commcov_golden_walls_are_open(self, seed):
        # The golden config has n_obstacles=0 — OpenTerrain reproduces it.
        golden = np.load(GOLDEN / "commcov.npz")
        wall = worldgen.OpenTerrain().walls(_wkey(seed), 16, 16)
        np.testing.assert_array_equal(np.asarray(wall), golden[f"wall_k{seed}"])


# =============================================================================
# Terrain implementations
# =============================================================================


class TestOpenTerrain:
    def test_all_false_and_key_independent(self):
        t = worldgen.OpenTerrain()
        w0 = t.walls(jax.random.PRNGKey(0), 6, 9)
        w1 = t.walls(jax.random.PRNGKey(99), 6, 9)
        assert w0.shape == (6, 9) and w0.dtype == jnp.bool_
        assert not bool(w0.any())
        np.testing.assert_array_equal(np.asarray(w0), np.asarray(w1))


class TestRandomWalls:
    def test_exact_count_and_determinism(self, key):
        t = worldgen.RandomWalls(10)
        w0 = t.walls(key, 16, 16)
        w1 = t.walls(key, 16, 16)
        w2 = t.walls(jax.random.PRNGKey(7), 16, 16)
        assert int(w0.sum()) == 10
        np.testing.assert_array_equal(np.asarray(w0), np.asarray(w1))
        assert not np.array_equal(np.asarray(w0), np.asarray(w2))  # key-driven

    def test_zero_obstacles(self, key):
        assert not bool(worldgen.RandomWalls(0).walls(key, 8, 8).any())

    def test_count_clamped_to_grid(self, key):
        w = worldgen.RandomWalls(1000).walls(key, 4, 4)
        assert int(w.sum()) == 16

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            worldgen.RandomWalls(-1)


class TestMapFile:
    MAP = """
        ####
        #..#
        #..#
        ####
    """

    def test_from_string(self, key):
        t = worldgen.MapFile.from_string(self.MAP)
        w = t.walls(key, 4, 4)
        expected = np.ones((4, 4), bool)
        expected[1:3, 1:3] = False
        np.testing.assert_array_equal(np.asarray(w), expected)

    def test_load_roundtrip(self, key, tmp_path):
        p = tmp_path / "map.txt"
        p.write_text(self.MAP)
        t = worldgen.MapFile.load(p)
        assert t == worldgen.MapFile.from_string(self.MAP)

    def test_size_mismatch_raises(self, key):
        t = worldgen.MapFile.from_string(self.MAP)
        with pytest.raises(ValueError, match="4x4"):
            t.walls(key, 8, 8)

    def test_bad_chars_raise(self):
        with pytest.raises(ValueError, match="unknown chars"):
            worldgen.MapFile.from_string("#x#\n...")

    def test_ragged_rows_raise(self):
        with pytest.raises(ValueError, match="equal length"):
            worldgen.MapFile.from_string("##\n###")

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            worldgen.MapFile.from_string("   \n  ")


class TestRooms:
    def test_single_room_is_open(self, key):
        assert not bool(worldgen.Rooms(rooms=1).walls(key, 8, 8).any())

    def test_partition_columns_and_doors(self, key):
        h, w, rooms, door_w = 10, 12, 3, 2
        wall = np.asarray(worldgen.Rooms(rooms=rooms, door_w=door_w).walls(key, h, w))
        wall_cols = [(i + 1) * w // rooms for i in range(rooms - 1)]   # [4, 8]
        for c in range(w):
            if c in wall_cols:
                # Full-height wall minus exactly one door_w-tall door.
                assert wall[:, c].sum() == h - door_w
                free_rows = np.flatnonzero(~wall[:, c])
                assert np.array_equal(free_rows, np.arange(free_rows[0], free_rows[0] + door_w))
            else:
                assert not wall[:, c].any()

    def test_door_row_is_key_driven_and_deterministic(self):
        t = worldgen.Rooms(rooms=2, door_w=1)
        w0 = np.asarray(t.walls(jax.random.PRNGKey(0), 16, 16))
        w0b = np.asarray(t.walls(jax.random.PRNGKey(0), 16, 16))
        np.testing.assert_array_equal(w0, w0b)
        # Door row varies across keys (16 possible rows; 8 keys ⇒ collision-proof check)
        doors = {
            int(np.flatnonzero(~np.asarray(t.walls(jax.random.PRNGKey(s), 16, 16))[:, 8])[0])
            for s in range(8)
        }
        assert len(doors) > 1

    def test_giant_door_opens_whole_wall(self, key):
        wall = worldgen.Rooms(rooms=2, door_w=100).walls(key, 8, 8)
        assert not bool(wall.any())

    def test_validation(self):
        with pytest.raises(ValueError):
            worldgen.Rooms(rooms=0)
        with pytest.raises(ValueError):
            worldgen.Rooms(door_w=0)
        with pytest.raises(ValueError, match="rooms"):
            worldgen.Rooms(rooms=9).walls(jax.random.PRNGKey(0), 8, 4)


# =============================================================================
# Spawn invariants across all terrains
# =============================================================================

TERRAINS = [
    worldgen.OpenTerrain(),
    worldgen.RandomWalls(10),
    worldgen.Rooms(rooms=2, door_w=3),
    worldgen.MapFile.from_string(
        "................\n" * 4
        + "......####......\n" * 2
        + "................\n" * 10
    ),
]


def _spawn_invariants(pos, wall, n_agents):
    pos = np.asarray(pos)
    wall = np.asarray(wall)
    assert pos.shape == (n_agents, 2) and pos.dtype == np.int32
    assert (pos >= 0).all()
    assert (pos[:, 0] < wall.shape[0]).all() and (pos[:, 1] < wall.shape[1]).all()
    assert len({tuple(p) for p in pos}) == n_agents          # distinct
    assert not wall[pos[:, 0], pos[:, 1]].any()              # free cells only


@pytest.mark.parametrize("terrain", TERRAINS, ids=lambda t: type(t).__name__)
@pytest.mark.parametrize("spawn", [worldgen.ScatterSpawn(), worldgen.ClusterSpawn(2)],
                         ids=lambda s: type(s).__name__)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_spawns_distinct_and_free(terrain, spawn, seed):
    key = jax.random.PRNGKey(seed)
    wkey, skey = jax.random.split(key)
    wall = terrain.walls(wkey, 16, 16)
    pos = spawn.positions(skey, wall, 5)
    _spawn_invariants(pos, wall, 5)


class TestFixedSpawn:
    def test_returns_cells_in_order_ignores_key(self):
        s = worldgen.FixedSpawn(cells=((0, 0), (3, 4), (7, 7)))
        wall = jnp.zeros((8, 8), dtype=jnp.bool_)
        p0 = s.positions(jax.random.PRNGKey(0), wall, 3)
        p1 = s.positions(jax.random.PRNGKey(9), wall, 3)
        np.testing.assert_array_equal(np.asarray(p0), [[0, 0], [3, 4], [7, 7]])
        np.testing.assert_array_equal(np.asarray(p0), np.asarray(p1))
        _spawn_invariants(p0, wall, 3)

    def test_prefix_for_smaller_team(self, key):
        s = worldgen.FixedSpawn(cells=((0, 0), (3, 4), (7, 7)))
        p = s.positions(key, jnp.zeros((8, 8), bool), 2)
        np.testing.assert_array_equal(np.asarray(p), [[0, 0], [3, 4]])

    def test_too_many_agents_raises(self, key):
        s = worldgen.FixedSpawn(cells=((0, 0),))
        with pytest.raises(ValueError, match="1 cells"):
            s.positions(key, jnp.zeros((8, 8), bool), 2)

    def test_duplicate_cells_raise(self):
        with pytest.raises(ValueError, match="distinct"):
            worldgen.FixedSpawn(cells=((1, 1), (1, 1)))


# =============================================================================
# JAX discipline: jit-able, hashable trace-time constants
# =============================================================================


@pytest.mark.parametrize("terrain", TERRAINS + [worldgen.RandomWalls(0)],
                         ids=lambda t: type(t).__name__)
def test_terrain_jittable(terrain, key):
    eager = terrain.walls(key, 16, 16)
    jitted = jax.jit(lambda k: terrain.walls(k, 16, 16))(key)
    np.testing.assert_array_equal(np.asarray(eager), np.asarray(jitted))


@pytest.mark.parametrize(
    "spawn",
    [worldgen.ScatterSpawn(), worldgen.ClusterSpawn(2),
     worldgen.FixedSpawn(cells=((0, 0), (1, 1), (2, 2), (3, 3)))],
    ids=lambda s: type(s).__name__,
)
def test_spawn_jittable(spawn, key):
    wall = worldgen.RandomWalls(10).walls(jax.random.PRNGKey(3), 16, 16)
    eager = spawn.positions(key, wall, 4)
    jitted = jax.jit(lambda k, w: spawn.positions(k, w, 4))(key, wall)
    np.testing.assert_array_equal(np.asarray(eager), np.asarray(jitted))


def test_components_hashable_and_eq():
    # Static-object rule: components close over jit — they must hash and compare.
    assert hash(worldgen.OpenTerrain()) == hash(worldgen.OpenTerrain())
    assert worldgen.RandomWalls(10) == worldgen.RandomWalls(10)
    assert worldgen.RandomWalls(10) != worldgen.RandomWalls(11)
    assert hash(worldgen.ClusterSpawn(2)) == hash(worldgen.ClusterSpawn(2))
    assert worldgen.MapFile.from_string("..\n..") == worldgen.MapFile.from_string("..\n..")
    {worldgen.Rooms(2, 1), worldgen.FixedSpawn(cells=((0, 0),)), worldgen.ScatterSpawn()}
