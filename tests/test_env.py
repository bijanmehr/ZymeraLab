"""GridEnv orchestrator: construction, validation, doctrine, conveniences."""

import jax
import jax.numpy as jnp
import pytest

import kymera
from kymera.comms import DiskTopology, GossipChannel, NullChannel
from kymera.missions import FixedAssignment, GroupedMission, Mission, RewardTerm


def _tiny(**kw):
    return kymera.make("comm-coverage", grid=6, n_agents=3, comm_r=2,
                       spawn_radius=1, **kw)


# ---- registry / construction -------------------------------------------------


def test_recipes_registered():
    assert {"empty", "comm-coverage"} <= set(kymera.list_envs())


def test_unknown_recipe_raises():
    with pytest.raises(ValueError, match="unknown env"):
        kymera.make("nope")


def test_belief_obs_requires_channel():
    from kymera.obs import GridObs
    with pytest.raises(ValueError, match="post-gossip belief"):
        kymera.GridEnv(grid_h=6, grid_w=6, n_agents=2,
                       obs=GridObs(("known", "own_pos")), channel=NullChannel())


def test_neighbors_obs_requires_topology():
    from kymera.obs import GridObs
    with pytest.raises(ValueError, match="topology"):
        kymera.GridEnv(grid_h=6, grid_w=6, n_agents=2,
                       obs=GridObs(("own_pos", "neighbors")), channel=NullChannel())


def test_bad_collision_string_raises():
    with pytest.raises(ValueError, match="collision"):
        kymera.make("comm-coverage", collision="masked")


def test_unknown_term_shorthand_raises():
    with pytest.raises(ValueError, match="unknown term"):
        kymera.make("comm-coverage", terms=[("nonsense", 1.0)])


# ---- spec / replace / repr -----------------------------------------------------


def test_spec_roundtrip(key):
    env = kymera.make("comm-coverage", grid=8, n_agents=3, comm_r=2, spawn_radius=1)
    env2 = kymera.make_from(env.spec())
    assert env.spec() == env2.spec()
    o1, s1 = env.reset(key)
    o2, s2 = env2.reset(key)
    assert jnp.array_equal(o1, o2)
    assert jnp.array_equal(s1.body.position, s2.body.position)


def test_replace_overrides():
    env = _tiny()
    env2 = env.replace(n_agents=4)
    assert env2.n_agents == 4 and env.n_agents == 3
    assert env2.spec()["n_agents"] == 4


def test_direct_construction_has_no_spec():
    env = kymera.GridEnv(grid_h=6, grid_w=6, n_agents=2)
    with pytest.raises(ValueError, match="recipe-built"):
        env.spec()


def test_repr_shows_composition():
    r = repr(_tiny())
    assert "GossipChannel" in r and "coverage" in r and "6×6" in r


# ---- step mechanics / doctrine -------------------------------------------------


def test_reset_step_shapes(key):
    env = _tiny()
    obs, state = env.reset(key)
    assert obs.shape == (3, 5, 6, 6)
    action = jnp.zeros((3,), jnp.int32)
    obs1, s1, reward, done, info = env.step(state, action, key)
    assert obs1.shape == (3, 5, 6, 6)
    assert reward.shape == (3,) and done.shape == (3,)
    assert set(info) == {"explored", "step_count", "seen_by", "comm_graph",
                         "reward_terms", "metrics"}
    assert set(info["reward_terms"]) == {"coverage", "connectivity", "collision"}


def test_mission_structure_stable_for_all_registered_envs(key):
    """Doctrine #6: state pytree structure identical between reset and step."""
    for name in kymera.list_envs():
        env = kymera.make(name) if name != "comm-coverage" else _tiny()
        _, s0 = env.reset(key)
        _, s1, *_ = env.step(s0, jnp.zeros((env.n_agents,), jnp.int32), key)
        assert (jax.tree_util.tree_structure(s0)
                == jax.tree_util.tree_structure(s1)), name


def test_rollout_jit_vmap(key):
    env = _tiny()
    f = jax.jit(lambda k: kymera.rollout(env, kymera.random_policy, 5, k))
    traj = f(key)
    assert traj["obs"].shape == (6, 3, 5, 6, 6)
    batch = jax.vmap(lambda k: kymera.rollout(env, kymera.random_policy, 5, k))(
        jax.random.split(key, 4))
    assert batch["reward"].shape == (4, 5, 3)


def test_rollout_lean_drops_channel(key):
    env = _tiny()
    lean = kymera.rollout(env, kymera.random_policy, 4, key)
    full = kymera.rollout(env, kymera.random_policy, 4, key, keep="all")
    assert lean["world"].channel == ()
    assert full["world"].channel.shared.shape == (5, 3, 6, 6)


def test_rollout_collect_reward_terms(key):
    traj = kymera.rollout(_tiny(), kymera.random_policy, 4, key,
                          collect=("reward_terms",))
    assert traj["info"]["reward_terms"]["coverage"].shape == (4, 3)


def test_block_collision_never_shares_cell(key):
    env = _tiny(collision="block")
    traj = kymera.rollout(env, kymera.random_policy, 15, key, keep="all")
    pos = traj["world"].body.position            # (T+1, N, 2)
    d = jnp.abs(pos[:, :, None, :] - pos[:, None, :, :]).max(-1)
    off = ~jnp.eye(3, dtype=bool)
    assert not bool(((d == 0) & off).any())


def test_action_mask_stay_always_valid(key):
    env = _tiny()
    _, state = env.reset(key)
    mask = env.action_mask(state)
    assert mask.shape == (3, 5)
    assert bool(mask[:, 0].all())


def test_central_obs(key):
    env = _tiny()
    _, state = env.reset(key)
    assert env.central_obs(state).shape == (3, 6, 6)


def test_max_steps_done(key):
    env = kymera.make("comm-coverage", grid=6, n_agents=3, comm_r=2,
                      spawn_radius=1, max_steps=2)
    _, s = env.reset(key)
    a = jnp.zeros((3,), jnp.int32)
    _, s, _, d1, _ = env.step(s, a, key)
    _, s, _, d2, _ = env.step(s, a, key)
    assert not bool(d1.any()) and bool(d2.all())


# ---- groups -------------------------------------------------------------------


def test_grouped_mission_routes_reward(key):
    from kymera import missions_terms as mt

    blue = Mission(terms=(RewardTerm("cov", 1.0, mt.new_coverage,
                                     mt.new_coverage.requires),))
    red = Mission(terms=(RewardTerm("coll", 1.0, mt.collision_count,
                                    mt.collision_count.requires),))
    env = kymera.GridEnv(
        grid_h=6, grid_w=6, n_agents=4,
        channel=GossipChannel(DiskTopology(2)),
        mission=GroupedMission(assignment=FixedAssignment((0, 0, 1, 1)),
                               missions=(blue, red)),
    )
    _, state = env.reset(key)
    assert jnp.array_equal(state.group, jnp.array([0, 0, 1, 1]))
    _, s1, reward, done, info = env.step(state, jnp.zeros((4,), jnp.int32), key)
    assert reward.shape == (4,) and done.shape == (4,)
    assert {"g0/cov", "g1/coll"} <= set(info["reward_terms"])
