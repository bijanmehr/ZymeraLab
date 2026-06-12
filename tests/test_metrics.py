"""kymera.metrics — hand-computed cases, golden-replay parity, StepCtx contract."""

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from kymera import metrics
from kymera.env import Body, World
from kymera.metrics import StepCtx, derive

GOLDEN = Path(__file__).parent / "golden"


@pytest.fixture(scope="module")
def commcov():
    return np.load(GOLDEN / "commcov.npz")


# -- helpers -------------------------------------------------------------------


@dataclass(frozen=True)
class _DiskTopo:
    """Minimal Topology stand-in (avoids depending on kymera.comms)."""

    radius: int

    def adjacency(self, world):
        return metrics.adjacency(world.body.position, self.radius)


def make_world(pos, h=16, w=16, seen_by=None, wall=None):
    pos = jnp.asarray(pos, jnp.int32)
    n = pos.shape[0]
    if wall is None:
        wall = jnp.zeros((h, w), bool)
    if seen_by is None:
        seen_by = jnp.zeros((n, h, w), bool)
    return World(
        body=Body(position=pos, energy=jnp.zeros((n,), jnp.float32)),
        explored=jnp.zeros((h, w), jnp.int32),
        seen_by=jnp.asarray(seen_by, bool),
        wall=jnp.asarray(wall, bool),
        comm_graph=jnp.zeros((n, n), bool),
        step_count=jnp.zeros((), jnp.int32),
        channel=(),
        mission=(),
        group=jnp.zeros((n,), jnp.int32),
    )


def closure_bfs(adj):
    """Reference transitive closure via Python BFS (numpy, test-only)."""
    adj = np.asarray(adj)
    n = adj.shape[0]
    out = np.zeros_like(adj, dtype=bool)
    for s in range(n):
        seen, stack = {s}, [s]
        while stack:
            u = stack.pop()
            for v in range(n):
                if adj[u, v] and v not in seen:
                    seen.add(v)
                    stack.append(v)
        for v in seen:
            out[s, v] = True
    return out


# =============================================================================
# Hand-computed 3-agent cases
# =============================================================================


class TestGraphHandCases:
    def test_pairwise_dist(self):
        pos = jnp.array([[0, 0], [0, 2], [5, 5]], jnp.int32)
        d = metrics.pairwise_dist(pos)
        expected = np.array([[0, 2, 5], [2, 0, 5], [5, 5, 0]], np.int32)
        assert d.dtype == jnp.int32
        np.testing.assert_array_equal(np.asarray(d), expected)

    def test_adjacency_symmetric_diag_true(self):
        pos = jnp.array([[0, 0], [0, 2], [5, 5]], jnp.int32)
        adj = metrics.adjacency(pos, 2)
        expected = np.array(
            [[True, True, False], [True, True, False], [False, False, True]]
        )
        np.testing.assert_array_equal(np.asarray(adj), expected)
        assert bool((adj == adj.T).all())
        assert bool(jnp.diag(adj).all())

    def test_reach_split_graph(self):
        # {0,1} connected, 2 isolated
        pos = jnp.array([[0, 0], [0, 2], [5, 5]], jnp.int32)
        r = metrics.reach(metrics.adjacency(pos, 2))
        expected = np.array(
            [[True, True, False], [True, True, False], [False, False, True]]
        )
        np.testing.assert_array_equal(np.asarray(r), expected)
        assert not bool(metrics.connected(metrics.adjacency(pos, 2)))
        assert int(metrics.giant_component(metrics.adjacency(pos, 2))) == 2

    def test_reach_chain_is_transitive(self):
        # 0-1 and 1-2 in range, 0-2 not: closure must still link 0 and 2.
        pos = jnp.array([[0, 0], [0, 2], [0, 4]], jnp.int32)
        adj = metrics.adjacency(pos, 2)
        assert not bool(adj[0, 2])
        r = metrics.reach(adj)
        assert bool(r.all())
        assert bool(metrics.connected(adj))
        assert int(metrics.giant_component(adj)) == 3

    def test_collisions(self):
        pos = jnp.array([[1, 1], [1, 1], [3, 3]], jnp.int32)
        c = metrics.collisions(pos)
        assert c.dtype == jnp.float32
        np.testing.assert_array_equal(np.asarray(c), [1.0, 1.0, 0.0])
        np.testing.assert_array_equal(
            np.asarray(metrics.collisions(jnp.array([[0, 0], [0, 1], [5, 5]]))),
            [0.0, 0.0, 0.0],
        )


class TestCoverageHandCases:
    def test_cheby_footprint_corner(self):
        fp = metrics.cheby_footprint(jnp.array([[0, 0]], jnp.int32), 4, 4, 1)
        assert fp.shape == (1, 4, 4)
        expected = np.zeros((4, 4), bool)
        expected[:2, :2] = True
        np.testing.assert_array_equal(np.asarray(fp[0]), expected)

    def test_coverage_fraction_all_cells(self):
        covered = jnp.zeros((4, 4), bool).at[0, :3].set(True).at[1, 1:3].set(True)
        assert float(metrics.coverage_fraction(covered)) == pytest.approx(5 / 16)

    def test_redundancy(self):
        seen = np.zeros((2, 4, 4), bool)
        seen[0, 0, :3] = True              # agent 0: 3 cells
        seen[1, 0, 2] = True               # agent 1: 2 cells, one shared
        seen[1, 1, 1] = True
        seen = jnp.asarray(seen)
        covered = seen.any(0)              # union: 4 cells
        assert float(metrics.redundancy(seen, covered)) == pytest.approx(5 / 4)

    def test_dist_to_frontier(self):
        unc = jnp.zeros((4, 4), bool).at[3, 3].set(True)
        pos = jnp.array([[0, 0], [3, 2]], jnp.int32)
        d = metrics.dist_to_frontier(pos, unc, 4, 4)
        np.testing.assert_allclose(np.asarray(d), [3.0, 1.0])
        # nothing uncovered -> zeros
        none = metrics.dist_to_frontier(pos, jnp.zeros((4, 4), bool), 4, 4)
        np.testing.assert_allclose(np.asarray(none), [0.0, 0.0])

    def test_field_mean_dist(self):
        pos = jnp.array([[0, 0], [3, 2]], jnp.int32)
        unc = jnp.zeros((4, 4), bool).at[3, 3].set(True)
        assert float(metrics.field_mean_dist(pos, unc, 4, 4)) == pytest.approx(1.0)
        # two uncovered cells: (3,3) -> 1 (agent 1), (0,3) -> 3 (either) => mean 2
        unc2 = unc.at[0, 3].set(True)
        assert float(metrics.field_mean_dist(pos, unc2, 4, 4)) == pytest.approx(2.0)
        assert float(
            metrics.field_mean_dist(pos, jnp.zeros((4, 4), bool), 4, 4)
        ) == pytest.approx(0.0)


# =============================================================================
# Golden parity (v0 comm-coverage trajectory)
# =============================================================================


class TestGoldenParity:
    @pytest.mark.parametrize("t", [0, 10, 35, 55, 70])
    def test_reach_matches_bfs_closure(self, commcov, t):
        comm_r = int(commcov["cfg"][4])
        pos = jnp.asarray(commcov["positions"][t], jnp.int32)
        adj = metrics.adjacency(pos, comm_r)
        # reference adjacency computed independently in numpy
        p = np.asarray(pos)
        d = np.abs(p[:, None, :] - p[None, :, :]).max(-1)
        np.testing.assert_array_equal(np.asarray(adj), d <= comm_r)
        np.testing.assert_array_equal(
            np.asarray(metrics.reach(adj)), closure_bfs(np.asarray(adj))
        )

    @pytest.mark.parametrize("t", [0, 5, 23, 50, 68])
    def test_newly_covered_replays_v0_newly(self, commcov, t):
        # vis_r = 0 in the golden config: footprint == the agent's own cell,
        # so v0's `newly` = onehot(pos[t+1]) & ~team_explored[t], per agent.
        team = commcov["team_explored"][t]
        assert (commcov["explored_by"][t].any(0) == team).all()  # consistency

        prev = make_world(commcov["positions"][t], seen_by=commcov["explored_by"][t])
        world = make_world(commcov["positions"][t + 1])
        ctx = derive(prev, world, frozenset({"newly_covered"}), cover_r=0)

        n = prev.n_agents
        expected = np.zeros(n, np.float32)
        for i in range(n):
            r, c = commcov["positions"][t + 1][i]
            expected[i] = 0.0 if team[r, c] else 1.0
        np.testing.assert_allclose(np.asarray(ctx.newly_covered), expected)
        # only the requested field is populated
        for name in ("dist", "adj", "delivered", "reach", "collisions",
                     "blocked", "overlap"):
            assert getattr(ctx, name) is None


# =============================================================================
# StepCtx / derive contract
# =============================================================================

REQ = frozenset({"dist", "adj", "reach", "collisions", "newly_covered", "overlap"})


class TestStepCtx:
    def test_overlap_hand_case(self):
        prev = make_world([[0, 0], [0, 0], [2, 2]], h=8, w=8)
        world = make_world([[0, 0], [0, 0], [2, 2]], h=8, w=8)
        # cover_r=0: agents 0 and 1 share a cell; agent 2 alone.
        ctx0 = derive(prev, world, frozenset({"overlap"}), cover_r=0)
        np.testing.assert_allclose(np.asarray(ctx0.overlap), [1.0, 1.0, 0.0])
        # cover_r=1: 0/1 footprints identical (4 corner cells); 2 meets them at (1,1).
        ctx1 = derive(prev, world, frozenset({"overlap"}), cover_r=1)
        np.testing.assert_allclose(np.asarray(ctx1.overlap), [4.0, 4.0, 1.0])

    def test_newly_covered_masks_walls(self):
        wall = jnp.zeros((8, 8), bool).at[0, 1].set(True)
        prev = make_world([[0, 0]], h=8, w=8, wall=wall)
        world = make_world([[0, 0]], h=8, w=8, wall=wall)
        # cover_r=1 footprint = 2x2 corner block minus the wall cell -> 3 fresh
        ctx = derive(prev, world, frozenset({"newly_covered"}), cover_r=1)
        np.testing.assert_allclose(np.asarray(ctx.newly_covered), [3.0])

    def test_passthrough_delivered_blocked(self):
        prev = make_world([[0, 0], [1, 1]], h=4, w=4)
        world = make_world([[0, 1], [1, 1]], h=4, w=4)
        eye = jnp.eye(2, dtype=bool)
        blk = jnp.array([False, True])
        ctx = derive(prev, world, frozenset({"delivered", "blocked"}),
                     delivered=eye, blocked=blk)
        np.testing.assert_array_equal(np.asarray(ctx.delivered), np.eye(2, dtype=bool))
        np.testing.assert_array_equal(np.asarray(ctx.blocked), [False, True])

    def test_python_time_errors(self):
        prev = make_world([[0, 0]], h=4, w=4)
        world = make_world([[0, 1]], h=4, w=4)
        with pytest.raises(ValueError, match="unknown StepCtx requirement"):
            derive(prev, world, frozenset({"lambda2"}))
        with pytest.raises(ValueError, match="topology"):
            derive(prev, world, frozenset({"adj"}))
        with pytest.raises(ValueError, match="delivered"):
            derive(prev, world, frozenset({"delivered"}))
        with pytest.raises(ValueError, match="blocked"):
            derive(prev, world, frozenset({"blocked"}))

    def test_structure_depends_only_on_requires(self):
        topo = _DiskTopo(2)
        seen = np.zeros((3, 8, 8), bool)
        seen[0, 0, 0] = True
        w1 = (make_world([[0, 0], [0, 1], [4, 4]], h=8, w=8),
              make_world([[0, 1], [0, 1], [4, 5]], h=8, w=8))
        w2 = (make_world([[7, 7], [3, 3], [0, 0]], h=8, w=8, seen_by=seen),
              make_world([[7, 6], [3, 4], [0, 1]], h=8, w=8, seen_by=seen))
        ctx1 = derive(w1[0], w1[1], REQ, topology=topo, cover_r=0)
        ctx2 = derive(w2[0], w2[1], REQ, topology=topo, cover_r=0)
        s1 = jax.tree_util.tree_structure(ctx1)
        s2 = jax.tree_util.tree_structure(ctx2)
        assert s1 == s2
        # a different requires set yields a DIFFERENT structure (None vs leaf)
        ctx3 = derive(w1[0], w1[1], frozenset({"dist"}))
        assert jax.tree_util.tree_structure(ctx3) != s1

    def test_derive_is_jitable(self):
        topo = _DiskTopo(2)
        prev = make_world([[0, 0], [0, 2], [5, 5]], h=8, w=8)
        world = make_world([[0, 1], [0, 2], [5, 5]], h=8, w=8)

        fn = jax.jit(
            lambda p, w: derive(p, w, REQ, topology=topo, cover_r=0)
        )
        ctx_jit = fn(prev, world)
        ctx_eager = derive(prev, world, REQ, topology=topo, cover_r=0)
        for name in ("dist", "adj", "reach", "collisions",
                     "newly_covered", "overlap"):
            np.testing.assert_array_equal(
                np.asarray(getattr(ctx_jit, name)),
                np.asarray(getattr(ctx_eager, name)),
            )
        # spot-check semantics through the jitted path
        np.testing.assert_array_equal(
            np.asarray(ctx_jit.adj),
            np.asarray(metrics.adjacency(world.body.position, 2)),
        )
        np.testing.assert_allclose(np.asarray(ctx_jit.newly_covered), [1.0, 1.0, 1.0])

    def test_ctx_is_scan_stackable(self):
        # StepCtx with a fixed requires set must vmap/stack cleanly.
        topo = _DiskTopo(2)
        prev = make_world([[0, 0], [0, 2], [5, 5]], h=8, w=8)
        worlds = [
            make_world([[0, 1], [0, 2], [5, 5]], h=8, w=8),
            make_world([[1, 0], [0, 3], [5, 6]], h=8, w=8),
        ]
        ctxs = [derive(prev, w, REQ, topology=topo) for w in worlds]
        stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *ctxs)
        assert isinstance(stacked, StepCtx)
        assert stacked.dist.shape == (2, 3, 3)
        assert stacked.newly_covered.shape == (2, 3)
