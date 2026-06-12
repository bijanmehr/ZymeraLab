"""THE full parity gate: kymera reproduces zymera v0 trajectories exactly.

Goldens were dumped read-only from the live v0 install (see
docs/plans/2026-06-12-kymera-implementation.md Task 1). Same PRNGKey + the
same random policy must reproduce walls, spawns, actions, positions, gossip
shared maps, observations exactly, and per-step rewards within fp tolerance.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import kymera

GOLD = Path(__file__).parent / "golden"


@pytest.fixture(scope="module")
def commcov():
    return np.load(GOLD / "commcov.npz")


@pytest.fixture(scope="module")
def empty():
    return np.load(GOLD / "empty.npz")


def _commcov_env():
    return kymera.make("comm-coverage", grid=16, n_agents=4, comm_r=5,
                       cover_r=0, sense_r=1, spawn_radius=2)


def test_spawn_parity(commcov):
    env = _commcov_env()
    for s in (0, 1, 2):
        _, state = env.reset(jax.random.PRNGKey(s))
        np.testing.assert_array_equal(np.asarray(state.body.position),
                                      commcov[f"pos0_k{s}"])
        np.testing.assert_array_equal(np.asarray(state.wall), commcov[f"wall_k{s}"])


def test_commcov_full_rollout_parity(commcov):
    env = _commcov_env()
    traj = kymera.rollout(env, kymera.random_policy, 70, jax.random.PRNGKey(0),
                          keep="all")
    # Same action stream => the frozen key protocol is intact.
    np.testing.assert_array_equal(np.asarray(traj["action"]), commcov["actions"])
    # Same physics.
    np.testing.assert_array_equal(np.asarray(traj["world"].body.position),
                                  commcov["positions"])
    # Same own-knowledge maps (v0 explored_by == kymera seen_by).
    np.testing.assert_array_equal(np.asarray(traj["world"].seen_by),
                                  commcov["explored_by"])
    # Same gossip (v0 shared == kymera channel.shared) — bit-for-bit.
    np.testing.assert_array_equal(np.asarray(traj["world"].channel.shared),
                                  commcov["shared"])
    # Same observations.
    np.testing.assert_array_equal(np.asarray(traj["obs"]), commcov["obs"])
    # Same rewards (term-sum vs v0 monolith) within fp tolerance.
    np.testing.assert_allclose(np.asarray(traj["reward"]), commcov["reward"],
                               atol=1e-5)


def test_commcov_coverage_metric_matches(commcov):
    env = _commcov_env()
    traj = kymera.rollout(env, kymera.random_policy, 70, jax.random.PRNGKey(0),
                          keep="all")
    covered = np.asarray(traj["world"].seen_by.any(1))          # (T+1, H, W)
    np.testing.assert_array_equal(covered, commcov["team_explored"])


def test_empty_rollout_parity(empty):
    env = kymera.make("empty", grid_h=8, grid_w=8, n_agents=4)
    for s in (0, 1, 2):
        obs0, state = env.reset(jax.random.PRNGKey(s))
        np.testing.assert_array_equal(np.asarray(state.body.position),
                                      empty[f"pos0_k{s}"])
        np.testing.assert_allclose(np.asarray(obs0), empty[f"obs0_k{s}"], atol=1e-6)
    traj = kymera.rollout(env, kymera.random_policy, 20, jax.random.PRNGKey(0))
    np.testing.assert_array_equal(np.asarray(traj["action"]), empty["actions"])
    np.testing.assert_array_equal(np.asarray(traj["world"].body.position),
                                  empty["positions"])
    np.testing.assert_array_equal(np.asarray(traj["world"].explored),
                                  empty["explored"])
    np.testing.assert_allclose(np.asarray(traj["obs"]), empty["obs"], atol=1e-6)


def test_gossip_superset_invariant(commcov):
    """shared must always be a superset of own knowledge (v0 selftest)."""
    env = _commcov_env()
    traj = kymera.rollout(env, kymera.random_policy, 30, jax.random.PRNGKey(7),
                          keep="all")
    w = traj["world"]
    assert bool((w.channel.shared >= w.seen_by).all())
