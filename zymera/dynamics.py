"""
Movement, collision resolution, action masks, and masked sampling.

This module kills v0's three-place collision split: the env's ``_move`` /
``_resolve_collisions`` (``examples/comm_coverage.py``) and the trainer's
``_targets`` / ``_masked_sample`` / ``_masked_logp_ent``
(``examples/_frontier_core.py``) all collapse onto ONE source of truth —
:meth:`GridDynamics.targets`. The trainer's collision-free sampler consumes
the same per-action target table the env steps with, so the two can never
drift apart again.

Semantics (ported bit-for-bit from v0):

* A move into the boundary is clipped in-bounds; a move into a wall cell is
  reverted (the agent stays put). With an open grid the wall check is a no-op.
* :class:`SequentialClaim` commits moves one agent at a time inside a
  ``lax.scan`` (doctrine: scan-over-agents for sequential conflict
  resolution); an agent whose target cell is already taken reverts to its old
  cell and is reported ``blocked`` — v0's ``hard_collision`` silently erased
  the attempt, so the collision reward term lost its learning signal.

**Pinned invariant (tested, forever):** agents claim in order ``0..N-1``.
Lower index wins a contested cell; an agent may enter a cell a LOWER-indexed
agent vacated this step, but not one a higher-indexed agent is about to
vacate. The masked samplers below walk the same order, so a masked-sampled
joint action is never reverted by :class:`SequentialClaim`.
"""

from dataclasses import dataclass
from typing import Protocol, Tuple

import chex
import jax
import jax.numpy as jnp

from .env import ACTION_DELTAS, ActionId, Body, N_ACTIONS

# Logit value for masked-out actions — large enough that softmax mass is ~0,
# small enough not to overflow float32 (same constant as v0 _frontier_core).
_MASKED_LOGIT = -1e9


# =============================================================================
# Collision rules
# =============================================================================


class CollisionRule(Protocol):
    """Resolves simultaneous move proposals into committed positions."""

    def resolve(
        self, old_pos: chex.Array, proposed: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        """``(old_pos (N,2), proposed (N,2)) -> (new_pos (N,2), blocked (N,) bool)``."""
        ...


@dataclass(frozen=True)
class NoCollision:
    """Pass-through — agents may share a cell; nothing is ever blocked."""

    def resolve(
        self, old_pos: chex.Array, proposed: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        n = proposed.shape[0]
        return proposed, jnp.zeros((n,), dtype=jnp.bool_)


@dataclass(frozen=True)
class SequentialClaim:
    """HARD collision avoidance — v0 ``_resolve_collisions`` plus ``blocked``.

    Commits moves one agent at a time (``lax.scan`` over agents ``0..N-1``,
    each commit visible to the next); an agent whose target cell is already
    taken by another agent reverts to its old cell and is flagged ``blocked``.
    Guarantees no two agents ever share a cell (given distinct ``old_pos``).
    """

    def resolve(
        self, old_pos: chex.Array, proposed: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        n = old_pos.shape[0]

        def body(finalized, i):
            ti = proposed[i]
            same = jnp.all(finalized == ti[None, :], axis=-1)   # cells equal to my target
            same = same.at[i].set(False)                        # ignore self
            taken = same.any()
            new_i = jnp.where(taken, old_pos[i], ti)            # taken → stay
            return finalized.at[i].set(new_i), taken

        finalized, blocked = jax.lax.scan(body, old_pos, jnp.arange(n))
        return finalized, blocked


# =============================================================================
# Grid dynamics
# =============================================================================


@dataclass(frozen=True)
class GridDynamics:
    """Square-grid movement: boundary clip + wall revert + a collision rule.

    Frozen / hashable — close over it in jitted code (the static-object rule).
    """

    collision: CollisionRule = NoCollision()

    def targets(self, world) -> chex.Array:
        """(N, A, 2) int32 — the committed cell for every agent × action.

        Boundary-clipped then wall-reverted, exactly as v0
        ``_frontier_core._targets`` / ``comm_coverage._move``. THE single
        source of truth: ``step`` gathers from it, the trainer's
        collision-free sampler masks against it.
        """
        pos = world.body.position
        wall = world.wall
        h, w = wall.shape
        t = pos[:, None, :] + ACTION_DELTAS[None, :, :]          # (N, A, 2)
        r = jnp.clip(t[..., 0], 0, h - 1)
        c = jnp.clip(t[..., 1], 0, w - 1)
        hit = wall[r, c]                                         # (N, A) bool
        r = jnp.where(hit, pos[:, None, 0], r)
        c = jnp.where(hit, pos[:, None, 1], c)
        return jnp.stack([r, c], axis=-1).astype(jnp.int32)

    def action_mask(self, world) -> chex.Array:
        """(N, A) bool — physical validity per action.

        True iff the action's target differs from the agent's current cell
        (i.e. the move is in-bounds and not into a wall) OR the action is
        STAY. STAY is always valid.
        """
        tg = self.targets(world)
        moved = (tg != world.body.position[:, None, :]).any(-1)  # (N, A)
        is_stay = jnp.arange(N_ACTIONS) == ActionId.STAY
        return moved | is_stay[None, :]

    def step(self, world, action: chex.Array) -> Tuple[Body, chex.Array]:
        """Apply ``action`` (N,) int32 → ``(Body', blocked (N,) bool)``.

        Gathers each agent's target from :meth:`targets`, then resolves
        conflicts through the configured :class:`CollisionRule`.
        """
        tg = self.targets(world)
        n = action.shape[0]
        proposed = tg[jnp.arange(n), action]                     # (N, 2)
        new_pos, blocked = self.collision.resolve(world.body.position, proposed)
        return world.body.replace(position=new_pos), blocked


# =============================================================================
# Sequential masked sampling (collision-free action sampling)
# =============================================================================


def sequential_masked_sample(
    logits: chex.Array,
    targets: chex.Array,
    init_pos: chex.Array,
    key: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """Sample collision-free actions agent-by-agent → ``(actions, masked logp)``.

    Port of v0 ``_frontier_core._masked_sample`` consuming a precomputed
    ``targets`` table (from :meth:`GridDynamics.targets`). Walks agents in
    the pinned order ``0..N-1``: an action is masked iff its target cell is
    already claimed (lower index → its committed target; higher index → its
    current cell). STAY is always valid — nobody can claim your own cell
    before your turn. By construction the sampled joint action is never
    reverted by :class:`SequentialClaim.resolve`.

    Args:
        logits:   (N, A) float — unmasked policy logits.
        targets:  (N, A, 2) int32 — per agent × action committed cells.
        init_pos: (N, 2) int32 — current positions (the claim-scan seed).
        key:      PRNG key.

    Returns:
        ``(actions (N,) int32, logp (N,) float32)`` — masked log-probs of the
        sampled actions.
    """
    n, _ = logits.shape

    def body(carry, i):
        committed, k = carry
        ti = targets[i]                                              # (A, 2)
        eq = jnp.all(ti[:, None, :] == committed[None, :, :], -1)    # (A, N)
        valid = ~(eq.at[:, i].set(False)).any(-1)                    # (A,)
        ml = jnp.where(valid, logits[i], _MASKED_LOGIT)
        k, sk = jax.random.split(k)
        a = jax.random.categorical(sk, ml)
        return (committed.at[i].set(ti[a]), k), (a, jax.nn.log_softmax(ml)[a])

    (_, _), (actions, logps) = jax.lax.scan(body, (init_pos, key), jnp.arange(n))
    return actions.astype(jnp.int32), logps


def sequential_masked_logp_ent(
    logits: chex.Array,
    targets: chex.Array,
    init_pos: chex.Array,
    actions: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    """Recompute masked log-probs + entropy for stored actions.

    Port of v0 ``_frontier_core._masked_logp_ent`` consuming a precomputed
    ``targets`` table. Replays the same ``0..N-1`` claim order as
    :func:`sequential_masked_sample`, so the returned ``logp`` equals the
    sample-time value for the same ``(logits, targets, init_pos, actions)``.

    Returns:
        ``(logp (N,) float32, entropy () float32)`` — entropy is the mean
        over agents of each masked distribution's entropy.
    """
    n, _ = logits.shape

    def body(committed, i):
        ti = targets[i]
        eq = jnp.all(ti[:, None, :] == committed[None, :, :], -1)
        valid = ~(eq.at[:, i].set(False)).any(-1)
        ml = jnp.where(valid, logits[i], _MASKED_LOGIT)
        lp = jax.nn.log_softmax(ml)
        ent = -(jnp.where(valid, jnp.exp(lp) * lp, 0.0)).sum()
        return committed.at[i].set(ti[actions[i]]), (lp[actions[i]], ent)

    _, (logps, ents) = jax.lax.scan(body, init_pos, jnp.arange(n))
    return logps, ents.mean()
