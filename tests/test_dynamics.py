"""Tests for zymera.dynamics — movement parity vs v0 goldens + claim-order properties."""

import functools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from zymera.dynamics import (
    GridDynamics,
    NoCollision,
    SequentialClaim,
    sequential_masked_logp_ent,
    sequential_masked_sample,
)
from zymera.env import ActionId, Body, N_ACTIONS, World

GOLDEN = Path(__file__).parent / "golden"


def _world(pos, wall):
    """Minimal World pytree — only the fields dynamics reads matter."""
    pos = jnp.asarray(pos, jnp.int32)
    wall = jnp.asarray(wall, jnp.bool_)
    n = pos.shape[0]
    h, w = wall.shape
    return World(
        body=Body(position=pos, energy=jnp.zeros((n,), jnp.float32)),
        explored=jnp.zeros((h, w), jnp.int32),
        seen_by=jnp.zeros((n, h, w), jnp.bool_),
        wall=wall,
        comm_graph=jnp.eye(n, dtype=jnp.bool_),
        step_count=jnp.int32(0),
        channel=(),
        mission=(),
        group=jnp.zeros((n,), jnp.int32),
    )


# =============================================================================
# Golden parity — replay v0 trajectories step by step
# =============================================================================


@pytest.mark.parametrize(
    "npz_name, wall_key",
    [("empty", None), ("commcov", "wall_k0")],
)
def test_movement_parity_golden(npz_name, wall_key):
    """From positions[t] + actions[t], step must reproduce positions[t+1]."""
    data = np.load(GOLDEN / f"{npz_name}.npz")
    positions = data["positions"]                      # (T+1, N, 2)
    actions = data["actions"]                          # (T, N)
    if wall_key is None:
        wall = np.zeros((8, 8), dtype=bool)            # empty-v0 golden: open 8x8
    else:
        wall = data[wall_key]
        assert not wall.any()                          # golden cfg has no obstacles
    wall = jnp.asarray(wall)
    dyn = GridDynamics(NoCollision())

    def step_pos(pos, act):
        body, blocked = dyn.step(_world(pos, wall), act)
        return body.position, blocked

    got, blocked = jax.jit(jax.vmap(step_pos))(
        jnp.asarray(positions[:-1]), jnp.asarray(actions)
    )
    np.testing.assert_array_equal(np.asarray(got), positions[1:])
    assert not np.asarray(blocked).any()               # NoCollision never blocks


# =============================================================================
# targets / action_mask unit semantics
# =============================================================================


def test_targets_clip_and_wall_revert():
    # 3x3 grid, wall at (0,1); agent 0 in the corner, agent 1 center.
    wall = np.zeros((3, 3), dtype=bool)
    wall[0, 1] = True
    world = _world([[0, 0], [1, 1]], wall)
    tg = np.asarray(GridDynamics().targets(world))

    assert tg.dtype == np.int32 and tg.shape == (2, N_ACTIONS, 2)
    # Agent 0 at (0,0): NORTH clips, WEST clips, EAST hits the wall -> revert.
    assert (tg[0, ActionId.STAY] == [0, 0]).all()
    assert (tg[0, ActionId.NORTH] == [0, 0]).all()
    assert (tg[0, ActionId.WEST] == [0, 0]).all()
    assert (tg[0, ActionId.EAST] == [0, 0]).all()
    assert (tg[0, ActionId.SOUTH] == [1, 0]).all()
    # Agent 1 at (1,1): NORTH hits the wall -> revert; others free.
    assert (tg[1, ActionId.NORTH] == [1, 1]).all()
    assert (tg[1, ActionId.EAST] == [1, 2]).all()
    assert (tg[1, ActionId.SOUTH] == [2, 1]).all()
    assert (tg[1, ActionId.WEST] == [1, 0]).all()


def test_action_mask_physical_validity():
    wall = np.zeros((3, 3), dtype=bool)
    wall[0, 1] = True
    world = _world([[0, 0], [1, 1]], wall)
    mask = np.asarray(GridDynamics().action_mask(world))

    assert mask.shape == (2, N_ACTIONS) and mask.dtype == bool
    assert mask[:, ActionId.STAY].all()                # STAY always valid
    # Agent 0: only SOUTH moves (NORTH/WEST clip, EAST walls).
    assert not mask[0, ActionId.NORTH]
    assert not mask[0, ActionId.WEST]
    assert not mask[0, ActionId.EAST]
    assert mask[0, ActionId.SOUTH]
    # Agent 1: NORTH walls; the rest are free.
    assert not mask[1, ActionId.NORTH]
    assert mask[1, ActionId.EAST] and mask[1, ActionId.SOUTH] and mask[1, ActionId.WEST]


# =============================================================================
# Collision rules
# =============================================================================


def test_no_collision_passthrough():
    old = jnp.asarray([[0, 0], [2, 2]], jnp.int32)
    proposed = jnp.asarray([[1, 1], [1, 1]], jnp.int32)   # deliberate overlap
    new_pos, blocked = NoCollision().resolve(old, proposed)
    np.testing.assert_array_equal(np.asarray(new_pos), np.asarray(proposed))
    assert blocked.shape == (2,) and not np.asarray(blocked).any()


def test_sequential_claim_lower_index_wins():
    """Agents 0 and 1 contest one cell: 0 wins, 1 reverts + blocked."""
    old = jnp.asarray([[0, 0], [0, 2], [2, 2]], jnp.int32)
    proposed = jnp.asarray([[0, 1], [0, 1], [2, 2]], jnp.int32)
    new_pos, blocked = SequentialClaim().resolve(old, proposed)
    np.testing.assert_array_equal(
        np.asarray(new_pos), [[0, 1], [0, 2], [2, 2]]
    )
    np.testing.assert_array_equal(np.asarray(blocked), [False, True, False])


def test_sequential_claim_order_pinned():
    """Order 0..N-1: you may take a LOWER index's vacated cell, never a
    higher index's about-to-be-vacated cell."""
    # Agent 0 vacates (0,0) -> (0,1); agent 1 enters (0,0): allowed.
    old = jnp.asarray([[0, 0], [1, 0]], jnp.int32)
    proposed = jnp.asarray([[0, 1], [0, 0]], jnp.int32)
    new_pos, blocked = SequentialClaim().resolve(old, proposed)
    np.testing.assert_array_equal(np.asarray(new_pos), [[0, 1], [0, 0]])
    assert not np.asarray(blocked).any()

    # Agent 0 targets agent 1's cell while 1 moves away: 0 is blocked
    # (1 hasn't committed yet), then 1 moves freely.
    old = jnp.asarray([[0, 0], [1, 0]], jnp.int32)
    proposed = jnp.asarray([[1, 0], [2, 0]], jnp.int32)
    new_pos, blocked = SequentialClaim().resolve(old, proposed)
    np.testing.assert_array_equal(np.asarray(new_pos), [[0, 0], [2, 0]])
    np.testing.assert_array_equal(np.asarray(blocked), [True, False])


def test_sequential_claim_swap_blocked():
    """A position swap is fully blocked — both agents revert."""
    old = jnp.asarray([[0, 0], [0, 1]], jnp.int32)
    proposed = jnp.asarray([[0, 1], [0, 0]], jnp.int32)
    new_pos, blocked = SequentialClaim().resolve(old, proposed)
    np.testing.assert_array_equal(np.asarray(new_pos), np.asarray(old))
    assert np.asarray(blocked).all()


# =============================================================================
# Masked sampling
# =============================================================================


def test_masked_logp_blocks_occupied_cell():
    """An action targeting an occupied cell gets ~zero probability."""
    # 2x1 grid: agent 0 at (0,0), agent 1 at (1,0). NORTH for agent 1 targets
    # agent 0's cell -> masked.
    world = _world([[0, 0], [1, 0]], np.zeros((2, 1), dtype=bool))
    dyn = GridDynamics(SequentialClaim())
    tg = dyn.targets(world)
    logits = jnp.zeros((2, N_ACTIONS), jnp.float32)
    actions = jnp.asarray([ActionId.STAY, ActionId.NORTH], jnp.int32)
    logp, ent = sequential_masked_logp_ent(logits, tg, world.body.position, actions)
    assert float(logp[1]) < -20.0                      # masked -> ~ -1e9 shifted
    assert float(logp[0]) > -20.0
    assert ent.shape == () and float(ent) >= 0.0


# =============================================================================
# Property test — masked sampling never proposes a revertable move
# =============================================================================

_SHAPES = [(4, 6, 6), (5, 7, 7), (6, 8, 8), (4, 5, 8), (6, 8, 5)]


@functools.lru_cache(maxsize=None)
def _checker(n, h, w):
    """One jitted check fn per shape so 200 configs compile only 5 times."""
    dyn = GridDynamics(SequentialClaim())

    @jax.jit
    def run(pos, wall, logits, key):
        world = _world(pos, wall)
        tg = dyn.targets(world)
        actions, logp = sequential_masked_sample(logits, tg, pos, key)
        body, blocked = dyn.step(world, actions)
        logp2, ent = sequential_masked_logp_ent(logits, tg, pos, actions)
        mask = dyn.action_mask(world)
        return actions, logp, logp2, ent, blocked, body.position, mask

    return run


def test_masked_sampling_never_blocked_property():
    """200 random configs: SequentialClaim never reverts a masked-sampled
    joint action; final positions stay distinct; logp matches at replay."""
    rng = np.random.default_rng(1234)
    for cfg in range(200):
        n, h, w = _SHAPES[cfg % len(_SHAPES)]
        wall = rng.random((h, w)) < 0.2
        free = np.argwhere(~wall)
        if len(free) < n:                              # pathological draw: clear walls
            wall[:] = False
            free = np.argwhere(~wall)
        pos = free[rng.choice(len(free), size=n, replace=False)].astype(np.int32)
        logits = rng.normal(size=(n, N_ACTIONS)).astype(np.float32)

        actions, logp, logp2, ent, blocked, new_pos, mask = _checker(n, h, w)(
            jnp.asarray(pos), jnp.asarray(wall), jnp.asarray(logits),
            jax.random.PRNGKey(cfg),
        )
        actions, blocked, new_pos = map(np.asarray, (actions, blocked, new_pos))

        assert not blocked.any(), f"cfg {cfg}: masked sample was reverted"
        assert len({tuple(p) for p in new_pos}) == n, f"cfg {cfg}: collision"
        assert ((actions >= 0) & (actions < N_ACTIONS)).all()
        np.testing.assert_allclose(
            np.asarray(logp2), np.asarray(logp), rtol=1e-5, atol=1e-6,
            err_msg=f"cfg {cfg}: replay logp != sample logp",
        )
        assert np.isfinite(float(ent)) and float(ent) >= 0.0
        assert np.asarray(mask)[:, ActionId.STAY].all()
