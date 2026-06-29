"""Wall perception (SLAM-style occupancy).

By default the gossip belief carries only *free* covered cells, so walls are
invisible to the decentralized policy: ``known_walls`` is structurally empty
and walls masquerade as ``local_frontier`` (the agent is lured into them).
With ``sense_walls=True`` the env folds sensed walls into the gossip outbox,
so the team's shared belief becomes a true occupancy map — ``known_walls``
lights up and sensed walls drop out of the frontier — while the *coverage*
state (``seen_by``, free-only) is untouched.
"""
import jax
import numpy as np

import zymera


def _rollout_worlds(env, steps=40, seed=0):
    traj = zymera.rollout(env, zymera.random_policy, n_steps=steps,
                          key=jax.random.PRNGKey(seed), keep="all")
    return traj["world"]


def _cheby_window(pos, h, w, r):
    rr = np.arange(h)[None, :, None]
    cc = np.arange(w)[None, None, :]
    return np.maximum(np.abs(rr - pos[:, 0][:, None, None]),
                      np.abs(cc - pos[:, 1][:, None, None])) <= r


def _known_walls_sum(world):
    """obs._known_walls == wall & channel.shared, summed over the episode."""
    wall = np.asarray(world.wall)                  # (T,H,W)
    shared = np.asarray(world.channel.shared)       # (T,N,H,W)
    return int((wall[:, None] & shared).sum())


def _walls_as_frontier_sum(world, sense_r=1):
    """obs._local_frontier == (~shared) & window; count wall cells flagged."""
    wall = np.asarray(world.wall)
    shared = np.asarray(world.channel.shared)
    pos = np.asarray(world.body.position)           # (T,N,2)
    t, n = pos.shape[0], pos.shape[1]
    h, w = wall.shape[1], wall.shape[2]
    win = np.stack([_cheby_window(pos[k], h, w, sense_r) for k in range(t)])
    frontier = (~shared) & win
    return int((frontier & wall[:, None]).sum())


_KW = dict(grid=10, n_agents=3, comm_r=5, n_obstacles=14, sense_r=1)


class TestWallSensing:
    def test_default_is_wall_blind(self):
        """Parity behaviour: walls never enter the belief; they read as frontier."""
        world = _rollout_worlds(zymera.make("comm-coverage", **_KW))
        assert _known_walls_sum(world) == 0
        assert _walls_as_frontier_sum(world) > 0      # the bug, by default

    def test_enabled_walls_are_sensed(self):
        world = _rollout_worlds(zymera.make("comm-coverage", sense_walls=True, **_KW))
        assert _known_walls_sum(world) > 0            # agents now perceive walls

    def test_enabled_walls_not_frontier(self):
        """Walls sensed within the frontier window are excluded from frontier."""
        world = _rollout_worlds(zymera.make("comm-coverage", sense_walls=True, **_KW))
        # wall_sense_r couples to sense_r (=1), so every wall in the frontier
        # window has been sensed this step -> none can read as fresh frontier.
        assert _walls_as_frontier_sum(world) == 0

    def test_coverage_state_unchanged(self):
        """Folding walls into the gossip belief must not touch coverage."""
        off = _rollout_worlds(zymera.make("comm-coverage", **_KW))
        on = _rollout_worlds(zymera.make("comm-coverage", sense_walls=True, **_KW))
        assert np.array_equal(np.asarray(off.seen_by), np.asarray(on.seen_by))
