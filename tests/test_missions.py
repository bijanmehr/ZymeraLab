"""zymera.missions / missions_terms — golden reward parity, PBRS telescoping,
group routing, assignment, construction-time validation."""

import dataclasses
from pathlib import Path as _Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from zymera import metrics
from zymera.env import Body, World
from zymera.metrics import StepCtx, derive
from zymera.missions import (
    Annotation, Assignment, FixedAssignment, GroupedMission, Mission, Path,
    Point, RandomKofN, Region, RewardTerm,
)
from zymera.missions_terms import (
    DEFAULT_TERMS, capped_giant, cbf_coll, cbf_conn, cohesion_leash,
    collision_count, degree_floor, new_coverage, pbrs, phi_field_mean,
    phi_nearest_frontier, reach_fraction, same_step_overlap,
)

GOLDEN = _Path(__file__).parent / "golden"

try:                                   # comms is a concurrent sibling module —
    from zymera.comms import DiskTopology as _Topo      # use it if importable,
except ImportError:                    # else a stand-in with the same adjacency
    @dataclasses.dataclass(frozen=True)
    class _Topo:
        radius: int

        def adjacency(self, world):
            return metrics.adjacency(world.body.position, self.radius)


@pytest.fixture(scope="module")
def commcov():
    return np.load(GOLDEN / "commcov.npz")


# -- helpers -------------------------------------------------------------------


def make_world(pos, h=16, w=16, seen_by=None, wall=None, step_count=0, group=None):
    pos = jnp.asarray(pos, jnp.int32)
    n = pos.shape[0]
    if wall is None:
        wall = jnp.zeros((h, w), bool)
    if seen_by is None:
        seen_by = jnp.zeros((n, h, w), bool)
    if group is None:
        group = jnp.zeros((n,), jnp.int32)
    return World(
        body=Body(position=pos, energy=jnp.zeros((n,), jnp.float32)),
        explored=jnp.zeros((h, w), jnp.int32),
        seen_by=jnp.asarray(seen_by, bool),
        wall=jnp.asarray(wall, bool),
        comm_graph=jnp.zeros((n, n), bool),
        step_count=jnp.asarray(step_count, jnp.int32),
        channel=(),
        mission=(),
        group=jnp.asarray(group, jnp.int32),
    )


def np_footprint(pos, h, w, r):
    """Reference (N, H, W) Chebyshev footprint, pure numpy."""
    rows = np.arange(h)[None, :, None]
    cols = np.arange(w)[None, None, :]
    pr = pos[:, 0][:, None, None]
    pc = pos[:, 1][:, None, None]
    return np.maximum(np.abs(rows - pr), np.abs(cols - pc)) <= r


def np_closure(adj):
    """Reference transitive closure via BFS, pure numpy."""
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


REQ = frozenset({"newly_covered", "reach", "collisions", "dist", "overlap"})


def golden_transition(commcov, t):
    """(prev, world, action, ctx) for the golden t -> t+1 transition."""
    comm_r = int(commcov["cfg"][4])
    cover_r = int(commcov["cfg"][3])           # vis_r in v0 terms
    prev = make_world(commcov["positions"][t], seen_by=commcov["explored_by"][t],
                      step_count=t)
    world = make_world(commcov["positions"][t + 1],
                       seen_by=commcov["explored_by"][t + 1], step_count=t + 1)
    action = jnp.asarray(commcov["actions"][t], jnp.int32)
    ctx = derive(prev, world, REQ, topology=_Topo(comm_r), cover_r=cover_r)
    return prev, world, action, ctx


# =============================================================================
# Golden term parity — THE reward parity pre-gate (plan Task 8)
# =============================================================================


class TestGoldenTermParity:
    @pytest.mark.parametrize("t", [0, 5, 35])
    def test_new_coverage_matches_v0_newly(self, commcov, t):
        prev, world, action, ctx = golden_transition(commcov, t)
        h, w, _, vis_r = (int(x) for x in commcov["cfg"][:4])
        pos1 = np.asarray(commcov["positions"][t + 1])
        vis = np_footprint(pos1, h, w, vis_r) & ~np.asarray(commcov["wall_k0"])
        expected = (vis & ~commcov["team_explored"][t][None]).reshape(
            pos1.shape[0], -1).sum(-1).astype(np.float32)
        got = new_coverage(prev, world, action, ctx)
        np.testing.assert_allclose(np.asarray(got), expected)

    @pytest.mark.parametrize("t", [0, 5, 35])
    def test_reach_fraction_matches_v0_connectivity(self, commcov, t):
        prev, world, action, ctx = golden_transition(commcov, t)
        comm_r = int(commcov["cfg"][4])
        pos1 = np.asarray(commcov["positions"][t + 1])
        n = pos1.shape[0]
        d = np.abs(pos1[:, None, :] - pos1[None, :, :]).max(-1)
        reach = np_closure(d <= comm_r)
        expected = ((reach.sum(-1) - 1) / max(n - 1, 1)).astype(np.float32)
        got = reach_fraction(prev, world, action, ctx)
        np.testing.assert_allclose(np.asarray(got), expected)

    @pytest.mark.parametrize("t", [0, 5, 35])
    def test_collision_count_matches_v0(self, commcov, t):
        prev, world, action, ctx = golden_transition(commcov, t)
        pos1 = np.asarray(commcov["positions"][t + 1])
        d = np.abs(pos1[:, None, :] - pos1[None, :, :]).max(-1)
        expected = ((d == 0) & ~np.eye(pos1.shape[0], dtype=bool)).sum(-1)
        got = collision_count(prev, world, action, ctx)
        np.testing.assert_allclose(np.asarray(got), expected.astype(np.float32))

    @pytest.mark.parametrize("t", [0, 5, 35])
    def test_default_terms_total_matches_golden_reward(self, commcov, t):
        # 1.0·new_coverage + 2.0·reach_fraction − 4.0·collision_count
        # must reproduce the v0 per-step reward — the parity pre-gate.
        prev, world, action, ctx = golden_transition(commcov, t)
        mission = Mission(terms=DEFAULT_TERMS)
        total, unweighted = mission.reward(prev, world, action, ctx, ())
        np.testing.assert_allclose(
            np.asarray(total), commcov["reward"][t], atol=1e-5
        )
        assert set(unweighted) == {"coverage", "connectivity", "collision"}
        recombined = (1.0 * unweighted["coverage"]
                      + 2.0 * unweighted["connectivity"]
                      - 4.0 * unweighted["collision"])
        np.testing.assert_allclose(np.asarray(recombined), np.asarray(total),
                                   atol=1e-6)

    def test_default_terms_total_full_trajectory(self, commcov):
        # Every step, not just spot checks — cheap at 16×16/N=4.
        mission = Mission(terms=DEFAULT_TERMS)
        for t in range(int(commcov["cfg"][7])):
            prev, world, action, ctx = golden_transition(commcov, t)
            total, _ = mission.reward(prev, world, action, ctx, ())
            np.testing.assert_allclose(
                np.asarray(total), commcov["reward"][t], atol=1e-5,
                err_msg=f"reward mismatch at t={t}",
            )


# =============================================================================
# Term zoo — hand cases
# =============================================================================


class TestTermZoo:
    def test_requires_attributes(self):
        assert new_coverage.requires == frozenset({"newly_covered"})
        assert reach_fraction.requires == frozenset({"reach"})
        assert capped_giant(3).requires == frozenset({"reach"})
        assert collision_count.requires == frozenset({"collisions"})
        assert same_step_overlap.requires == frozenset({"overlap"})
        assert cohesion_leash(4.0, 5).requires == frozenset({"dist"})
        assert degree_floor(1.0, 5).requires == frozenset({"dist"})
        assert pbrs(phi_nearest_frontier, 0.99).requires == frozenset()
        assert cbf_conn(0.5, 0.1, 2.0, 5).requires == frozenset()
        assert cbf_coll(0.5, 1.0).requires == frozenset()

    def test_same_step_overlap_reads_ctx(self):
        prev = make_world([[0, 0], [0, 0], [2, 2]], h=8, w=8)
        world = make_world([[0, 0], [0, 0], [2, 2]], h=8, w=8)
        ctx = derive(prev, world, frozenset({"overlap"}), cover_r=0)
        got = same_step_overlap(prev, world, None, ctx)
        np.testing.assert_allclose(np.asarray(got), [1.0, 1.0, 0.0])

    def test_capped_giant(self):
        # chain of 3 connected (r=2) + 1 isolated -> giant = 3
        prev = make_world([[0, 0], [0, 2], [0, 4], [9, 9]], h=16, w=16)
        world = make_world([[0, 0], [0, 2], [0, 4], [9, 9]], h=16, w=16)
        ctx = derive(prev, world, frozenset({"reach"}), topology=_Topo(2))
        np.testing.assert_allclose(
            np.asarray(capped_giant(3)(prev, world, None, ctx)), np.ones(4)
        )
        np.testing.assert_allclose(
            np.asarray(capped_giant(4)(prev, world, None, ctx)),
            np.full(4, 0.75),
        )
        with pytest.raises(ValueError, match="cap"):
            capped_giant(0)

    def test_cohesion_leash(self):
        world = make_world([[0, 0], [0, 3], [0, 4]], h=8, w=8)
        ctx = StepCtx(dist=metrics.pairwise_dist(world.body.position))
        # nearest-teammate dists: [3, 1, 1]; leash 1 -> [2, 0, 0]
        got = cohesion_leash(1.0, 5)(world, world, None, ctx)
        np.testing.assert_allclose(np.asarray(got), [2.0, 0.0, 0.0])

    def test_cohesion_leash_clamps_at_comm_r(self):
        world = make_world([[0, 0], [0, 9]], h=16, w=16)
        ctx = StepCtx(dist=metrics.pairwise_dist(world.body.position))
        # nearest = 9, clamped to comm_r=5; leash 4 -> penalty 1 (not 5)
        got = cohesion_leash(4.0, 5)(world, world, None, ctx)
        np.testing.assert_allclose(np.asarray(got), [1.0, 1.0])

    def test_degree_floor(self):
        world = make_world([[0, 0], [0, 1], [9, 9]], h=16, w=16)
        ctx = StepCtx(dist=metrics.pairwise_dist(world.body.position))
        # in-range neighbours (r=2): [1, 1, 0]; floor 1 -> [0, 0, 1]
        got = degree_floor(1.0, 2)(world, world, None, ctx)
        np.testing.assert_allclose(np.asarray(got), [0.0, 0.0, 1.0])


# =============================================================================
# PBRS — telescoping + v0 sign convention
# =============================================================================


class TestPBRS:
    @pytest.mark.parametrize("phi", [phi_nearest_frontier, phi_field_mean])
    def test_telescopes_at_gamma_one(self, commcov, phi):
        # gamma=1: Σ_t [Φ(w_{t+1}) − Φ(w_t)] == Φ(w_T) − Φ(w_0).
        worlds = [
            make_world(commcov["positions"][t], seen_by=commcov["explored_by"][t])
            for t in range(13)
        ]
        fn = pbrs(phi, gamma=1.0)
        total = sum(
            np.asarray(fn(worlds[t], worlds[t + 1], None, None))
            for t in range(12)
        )
        expected = np.broadcast_to(
            np.asarray(phi(worlds[-1]) - phi(worlds[0])), total.shape
        )
        np.testing.assert_allclose(total, expected, atol=1e-4)

    def test_single_step_matches_v0_form(self, commcov):
        # v0 adds w·(γ·(−p1) − (−p0)); phi == −dist so term == γ·phi1 − phi0.
        t, g = 3, 0.99
        prev = make_world(commcov["positions"][t],
                          seen_by=commcov["explored_by"][t])
        world = make_world(commcov["positions"][t + 1],
                           seen_by=commcov["explored_by"][t + 1])
        h, w = int(commcov["cfg"][0]), int(commcov["cfg"][1])
        unc0 = ~commcov["team_explored"][t] & ~np.asarray(commcov["wall_k0"])
        unc1 = ~commcov["team_explored"][t + 1] & ~np.asarray(commcov["wall_k0"])
        p0 = metrics.dist_to_frontier(prev.body.position, jnp.asarray(unc0), h, w)
        p1 = metrics.dist_to_frontier(world.body.position, jnp.asarray(unc1), h, w)
        expected = g * (-np.asarray(p1)) - (-np.asarray(p0))
        got = pbrs(phi_nearest_frontier, g)(prev, world, None, None)
        np.testing.assert_allclose(np.asarray(got), expected, atol=1e-6)

    def test_scalar_phi_broadcasts(self, commcov):
        prev = make_world(commcov["positions"][0],
                          seen_by=commcov["explored_by"][0])
        world = make_world(commcov["positions"][1],
                           seen_by=commcov["explored_by"][1])
        got = pbrs(phi_field_mean, 0.99)(prev, world, None, None)
        assert got.shape == (4,)
        assert got.dtype == jnp.float32
        assert len(set(np.asarray(got).tolist())) == 1   # shared team value


# =============================================================================
# CBF terms — ported math, hand-verified
# =============================================================================


class TestCBF:
    def test_cbf_coll_hand_case(self):
        # pair distance 2 -> 1 with dmin=1: h 1 -> 0; residual relu(.5·1−0)=.5
        prev = make_world([[0, 0], [0, 2]], h=8, w=8)
        world = make_world([[0, 0], [0, 1]], h=8, w=8)
        got = cbf_coll(alpha=0.5, dmin=1.0)(prev, world, None, None)
        np.testing.assert_allclose(np.asarray(got), [0.5, 0.5], atol=1e-6)

    def test_cbf_coll_safe_transition_is_zero(self):
        prev = make_world([[0, 0], [0, 4]], h=8, w=8)
        world = make_world([[0, 0], [0, 5]], h=8, w=8)   # moving APART
        got = cbf_coll(alpha=0.5, dmin=1.0)(prev, world, None, None)
        np.testing.assert_allclose(np.asarray(got), [0.0, 0.0])

    def test_cbf_conn_two_agent_analytic(self):
        # 2-node weighted graph: L eigenvalues {0, 2w} -> λ₂ = 2·sigmoid(sharp·(r−d)).
        alpha, eps, sharp, comm_r = 0.5, 0.1, 2.0, 5
        prev = make_world([[0, 0], [0, 2]], h=16, w=16)    # d=2 (connected)
        world = make_world([[0, 0], [0, 8]], h=16, w=16)   # d=8 (broken)
        w0 = 1.0 / (1.0 + np.exp(-sharp * (comm_r - 2)))
        w1 = 1.0 / (1.0 + np.exp(-sharp * (comm_r - 8)))
        hp, hn = 2 * w0 - eps, 2 * w1 - eps
        expected = max((1 - alpha) * hp - hn, 0.0) / 2     # shared, /N
        got = cbf_conn(alpha, eps, sharp, comm_r)(prev, world, None, None)
        np.testing.assert_allclose(np.asarray(got), np.full(2, expected),
                                   rtol=1e-5)

    def test_cbf_conn_stable_chain_is_zero(self):
        # A connected chain that doesn't move incurs ~0 (threshold property).
        world = make_world([[0, 0], [0, 4], [0, 8]], h=16, w=16)
        got = cbf_conn(0.5, 0.1, 2.0, 5)(world, world, None, None)
        np.testing.assert_allclose(np.asarray(got), np.zeros(3), atol=1e-6)


# =============================================================================
# Mission protocol
# =============================================================================


class TestMission:
    def test_duplicate_term_name_raises(self):
        with pytest.raises(ValueError, match="duplicate"):
            Mission(terms=(
                RewardTerm("cov", 1.0, new_coverage, new_coverage.requires),
                RewardTerm("cov", 2.0, collision_count, collision_count.requires),
            ))

    def test_bad_max_steps_raises(self):
        with pytest.raises(ValueError, match="max_steps"):
            Mission(terms=(), max_steps=0)

    def test_requires_is_union_of_terms(self):
        m = Mission(terms=DEFAULT_TERMS)
        assert m.requires == frozenset({"newly_covered", "reach", "collisions"})
        assert Mission(terms=()).requires == frozenset()

    def test_init_state_update_identity(self, key):
        world = make_world([[0, 0], [1, 1]], h=8, w=8)
        m = Mission(terms=DEFAULT_TERMS)
        ms = m.init_state(key, world)
        assert ms == ()
        assert m.update(world, world, None, ms, key) == ()

    def test_done_max_steps(self):
        m = Mission(terms=(), max_steps=5)
        ctx = None
        for sc, expect in [(3, False), (5, True), (7, True)]:
            d = m.done(make_world([[0, 0], [1, 1]], step_count=sc), ctx, ())
            assert d.shape == (2,) and d.dtype == jnp.bool_
            assert bool(d.all()) is expect and bool(d.any()) is expect
        # max_steps=None -> never done
        forever = Mission(terms=())
        d = forever.done(make_world([[0, 0], [1, 1]], step_count=10_000), ctx, ())
        assert not bool(d.any())

    def test_empty_terms_reward(self):
        world = make_world([[0, 0], [1, 1]], h=8, w=8)
        total, terms = Mission(terms=()).reward(world, world, None, None, ())
        np.testing.assert_allclose(np.asarray(total), [0.0, 0.0])
        assert terms == {}

    def test_metrics_coverage_and_conditional_giant(self):
        seen = np.zeros((2, 8, 8), bool)
        seen[0, 0, :4] = True
        world = make_world([[0, 0], [0, 2]], h=8, w=8, seen_by=seen)
        m = Mission(terms=())
        no_reach = m.metrics(world, StepCtx(), ())
        assert set(no_reach) == {"coverage"}
        assert float(no_reach["coverage"]) == pytest.approx(4 / 64)
        ctx = derive(world, world, frozenset({"reach"}), topology=_Topo(2))
        with_reach = m.metrics(world, ctx, ())
        assert set(with_reach) == {"coverage", "giant_fraction"}
        assert float(with_reach["giant_fraction"]) == pytest.approx(1.0)

    def test_annotations_default_empty(self):
        world = make_world([[0, 0]], h=4, w=4)
        assert Mission(terms=()).annotations(world, ()) == ()

    def test_annotation_primitives_are_frozen(self):
        p = Point((1, 2), tag="vip")
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.tag = "x"
        assert Path(((0, 0), (0, 1)), tag="route").tag == "route"
        assert Region(np.zeros((4, 4), bool), tag="jam").tag == "jam"


# =============================================================================
# Assignment
# =============================================================================


class TestAssignment:
    def test_fixed_default_all_zero(self, key):
        ids = FixedAssignment().assign(key, 4)
        assert ids.dtype == jnp.int32
        np.testing.assert_array_equal(np.asarray(ids), np.zeros(4, np.int32))

    def test_fixed_groups(self, key):
        ids = FixedAssignment((0, 0, 1, 1)).assign(key, 4)
        np.testing.assert_array_equal(np.asarray(ids), [0, 0, 1, 1])
        with pytest.raises(ValueError, match="4 agents"):
            FixedAssignment((0, 1)).assign(key, 4)

    def test_random_k_of_n_exact_count(self):
        a = RandomKofN(2)
        for s in range(20):
            ids = a.assign(jax.random.PRNGKey(s), 4)
            assert ids.shape == (4,) and ids.dtype == jnp.int32
            assert int((ids == 1).sum()) == 2
            assert int((ids == 0).sum()) == 2

    def test_random_k_of_n_varies_across_keys(self):
        a = RandomKofN(2)
        seen = {
            tuple(np.asarray(a.assign(jax.random.PRNGKey(s), 4)).tolist())
            for s in range(20)
        }
        assert len(seen) > 1

    def test_random_k_of_n_custom_group_and_bounds(self, key):
        ids = RandomKofN(1, group=3).assign(key, 4)
        assert int((ids == 3).sum()) == 1
        with pytest.raises(ValueError, match="exceeds"):
            RandomKofN(5).assign(key, 4)
        with pytest.raises(ValueError, match="k must be"):
            RandomKofN(-1)

    def test_protocol_duck_typing(self):
        assert isinstance(FixedAssignment(), Assignment)
        assert isinstance(RandomKofN(1), Assignment)


# =============================================================================
# GroupedMission
# =============================================================================


def _const_term(name, value):
    def fn(prev, world, action, ctx):
        return jnp.full((world.n_agents,), value, jnp.float32)

    return RewardTerm(name, 1.0, fn)


class TestGroupedMission:
    def _gm(self, max_steps=(None, None)):
        return GroupedMission(
            assignment=FixedAssignment((0, 0, 1, 1)),
            missions=(
                Mission(terms=(_const_term("one", 1.0),), max_steps=max_steps[0]),
                Mission(terms=(_const_term("two", 2.0),), max_steps=max_steps[1]),
            ),
        )

    def test_empty_missions_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            GroupedMission(assignment=FixedAssignment(), missions=())

    def test_reward_routes_by_group(self, key):
        gm = self._gm()
        world = make_world([[0, 0], [1, 1], [2, 2], [3, 3]], h=8, w=8,
                           group=(0, 0, 1, 1))
        ms = gm.init_state(key, world)
        assert ms == ((), ())
        total, terms = gm.reward(world, world, None, StepCtx(), ms)
        np.testing.assert_allclose(np.asarray(total), [1.0, 1.0, 2.0, 2.0])
        assert set(terms) == {"g0/one", "g1/two"}
        # per-term values stay UNMASKED (full N, unweighted) for post-hoc analysis
        np.testing.assert_allclose(np.asarray(terms["g0/one"]), np.ones(4))
        np.testing.assert_allclose(np.asarray(terms["g1/two"]), np.full(4, 2.0))

    def test_done_routes_own_groups_mission(self, key):
        gm = self._gm(max_steps=(5, None))
        world = make_world([[0, 0], [1, 1], [2, 2], [3, 3]], h=8, w=8,
                           group=(0, 0, 1, 1), step_count=7)
        ms = gm.init_state(key, world)
        d = gm.done(world, StepCtx(), ms)
        np.testing.assert_array_equal(np.asarray(d), [True, True, False, False])
        early = make_world([[0, 0], [1, 1], [2, 2], [3, 3]], h=8, w=8,
                           group=(0, 0, 1, 1), step_count=3)
        assert not bool(gm.done(early, StepCtx(), ms).any())

    def test_requires_union_and_namespaced_terms(self):
        gm = GroupedMission(
            assignment=FixedAssignment((0, 0, 1, 1)),
            missions=(
                Mission(terms=(RewardTerm("cov", 1.0, new_coverage,
                                          new_coverage.requires),)),
                Mission(terms=(RewardTerm("coll", -1.0, collision_count,
                                          collision_count.requires),)),
            ),
        )
        assert gm.requires == frozenset({"newly_covered", "collisions"})
        assert [t.name for t in gm.terms] == ["g0/cov", "g1/coll"]

    def test_metrics_namespaced(self, key):
        gm = self._gm()
        world = make_world([[0, 0], [1, 1], [2, 2], [3, 3]], h=8, w=8,
                           group=(0, 0, 1, 1))
        ms = gm.init_state(key, world)
        out = gm.metrics(world, StepCtx(), ms)
        assert set(out) == {"g0/coverage", "g1/coverage"}

    def test_reward_jits_with_random_assignment(self, key):
        # RandomKofN re-rolls membership per reset without retracing:
        # routing is pure where() on world.group.
        gm = self._gm()

        @jax.jit
        def total_for(group):
            world = make_world([[0, 0], [1, 1], [2, 2], [3, 3]], h=8, w=8,
                               group=group)
            t, _ = gm.reward(world, world, None, StepCtx(), ((), ()))
            return t

        g1 = RandomKofN(2).assign(jax.random.PRNGKey(3), 4)
        out = np.asarray(total_for(g1))
        expected = np.where(np.asarray(g1) == 0, 1.0, 2.0)
        np.testing.assert_allclose(out, expected)
        g2 = RandomKofN(2).assign(jax.random.PRNGKey(9), 4)
        np.testing.assert_allclose(
            np.asarray(total_for(g2)), np.where(np.asarray(g2) == 0, 1.0, 2.0)
        )
