"""Occupancy belief + field boundary + a real exploration frontier.

The SLAM fix (``sense_walls``) folds *walls* into the gossip belief. ``sense_free``
goes the whole way: it folds the **full sensed region** (free *and* wall within the
sense radius) into the belief, so the shared map becomes a true free/occupied/unknown
**occupancy** grid — not just a coverage trail. That upgrade unlocks two things the
position-blind LPAC policy was missing:

* ``occ_frontier`` — the global edge of the explored region (known-free cells that
  border an unknown cell, à la Yamauchi). The egocentric ``local_frontier`` *collapses
  to empty* under occupancy (its window is already all-sensed), so this is the channel
  that actually tells the agent where to explore.
* ``boundary`` — the field-edge ring, re-grounding the mission-field extent that the
  global-average-pool throws away.

All three are opt-in; with every flag off the recipe is byte-for-byte v0 (covered by the
golden-file parity gate). Coverage state (``seen_by``, free-only) is never touched.
"""
import jax
import numpy as np

import zymera
from zymera.obs import _boundary, _local_frontier, _occ_frontier  # package-internal


def _rollout_worlds(env, steps=40, seed=0):
    traj = zymera.rollout(env, zymera.random_policy, n_steps=steps,
                          key=jax.random.PRNGKey(seed), keep="all")
    return traj["world"]


def _last_world(env, steps=12, seed=0):
    """A single (unstacked) world late in an episode, for the (N,H,W) channels."""
    world = _rollout_worlds(env, steps=steps, seed=seed)
    return jax.tree_util.tree_map(lambda x: x[-1], world)


def _belief_sum(world):
    return int(np.asarray(world.channel.shared).sum())


_OPEN = dict(grid=12, n_agents=3, comm_r=5, n_obstacles=0, sense_r=2)
_ROOMS = dict(grid=14, n_agents=3, comm_r=5, n_obstacles=20, sense_r=2)


class TestSenseFreeOccupancy:
    def test_belief_grows_beyond_coverage(self):
        """sense_free folds the full sensed region in, so the belief is strictly
        larger than the wall-blind coverage-only belief at the same seed."""
        off = _rollout_worlds(zymera.make("comm-coverage", **_OPEN))
        on = _rollout_worlds(zymera.make("comm-coverage", sense_free=True, **_OPEN))
        assert _belief_sum(on) > _belief_sum(off)

    def test_coverage_state_unchanged(self):
        """Occupancy sensing must not touch the (free-only) coverage state."""
        off = _rollout_worlds(zymera.make("comm-coverage", **_OPEN))
        on = _rollout_worlds(zymera.make("comm-coverage", sense_free=True, **_OPEN))
        assert np.array_equal(np.asarray(off.seen_by), np.asarray(on.seen_by))

    def test_walls_enter_belief(self):
        """On an obstacle map sense_free lights up known_walls (occupancy map)."""
        on = _rollout_worlds(zymera.make("comm-coverage", sense_free=True, **_ROOMS))
        wall = np.asarray(on.wall)                  # (T,H,W)
        shared = np.asarray(on.channel.shared)       # (T,N,H,W)
        assert int((wall[:, None] & shared).sum()) > 0


class TestOccFrontier:
    def test_frontier_is_known_free_edge(self):
        """Every occ_frontier cell is known-free AND borders an unknown cell —
        the channel matches the occupancy-frontier definition exactly."""
        w = _last_world(zymera.make("comm-coverage", sense_free=True, **_ROOMS))
        of = np.asarray(_occ_frontier(w, None)).astype(bool)        # (N,H,W)
        shared = np.asarray(w.channel.shared)
        wall = np.asarray(w.wall)
        free = shared & ~wall[None]                                  # known-free
        unknown = ~shared
        nbr = np.zeros_like(unknown)
        nbr[:, :-1, :] |= unknown[:, 1:, :]
        nbr[:, 1:, :] |= unknown[:, :-1, :]
        nbr[:, :, :-1] |= unknown[:, :, 1:]
        nbr[:, :, 1:] |= unknown[:, :, :-1]
        assert np.array_equal(of, free & nbr)
        assert np.all(of <= free)                                   # never wall/unknown

    def test_frontier_nonempty_but_strict_subset(self):
        """There is a frontier to chase, and it is an edge — not the whole region."""
        w = _last_world(zymera.make("comm-coverage", sense_free=True, **_OPEN))
        of = np.asarray(_occ_frontier(w, None)).astype(bool)
        free = np.asarray(w.channel.shared) & ~np.asarray(w.wall)[None]
        assert 0 < of.sum() < free.sum()

    def test_local_frontier_collapses_occ_survives(self):
        """Under occupancy the egocentric window is all-sensed, so local_frontier
        goes empty — and occ_frontier is the channel that stays informative."""
        env = zymera.make("comm-coverage", sense_free=True, **_OPEN)
        w = _last_world(env)
        lf = np.asarray(_local_frontier(w, None, sense_r=_OPEN["sense_r"]))
        of = np.asarray(_occ_frontier(w, None))
        assert lf.sum() == 0
        assert of.sum() > 0


class TestBoundaryChannel:
    def test_boundary_is_edge_ring(self):
        w = _last_world(zymera.make("comm-coverage", boundary=True, **_OPEN))
        b = np.asarray(_boundary(w, None)).astype(bool)            # (H,W)
        g = _OPEN["grid"]
        expect = np.zeros((g, g), dtype=bool)
        expect[0, :] = expect[-1, :] = True
        expect[:, 0] = expect[:, -1] = True
        assert np.array_equal(b, expect)


class TestObsWiring:
    def test_default_recipe_has_five_channels(self):
        """Flags off → the v0 channel stack, unchanged."""
        o0, _ = zymera.make("comm-coverage", **_OPEN).reset(jax.random.PRNGKey(0))
        assert o0.shape[1] == 5                                     # (N, C, H, W)

    def test_sense_free_adds_occ_frontier_plane(self):
        o0, _ = zymera.make("comm-coverage", **_OPEN).reset(jax.random.PRNGKey(0))
        o1, _ = zymera.make("comm-coverage", sense_free=True, **_OPEN
                            ).reset(jax.random.PRNGKey(0))
        assert o1.shape[1] == o0.shape[1] + 1

    def test_boundary_adds_one_plane(self):
        o0, _ = zymera.make("comm-coverage", **_OPEN).reset(jax.random.PRNGKey(0))
        o1, _ = zymera.make("comm-coverage", boundary=True, **_OPEN
                            ).reset(jax.random.PRNGKey(0))
        assert o1.shape[1] == o0.shape[1] + 1
