"""Tests for zymera.comms — topology + gossip/null channels.

The gate is the parity test: replaying the golden v0 trajectory through
``GossipChannel(DiskTopology(5), delay=1, dropout=0.0)`` must reproduce
the v0 ``shared`` maps bit-for-bit at every step.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from zymera.comms import (
    ChannelState,
    DiskTopology,
    GossipChannel,
    NullChannel,
    Topology,
)
from zymera.env import Body, World

GOLDEN = Path(__file__).parent / "golden" / "commcov.npz"


# =============================================================================
# Helpers
# =============================================================================


def make_world(pos, h, w):
    """Minimal full World — channels only read body.position."""
    pos = jnp.asarray(pos, jnp.int32)
    n = pos.shape[0]
    return World(
        body=Body(position=pos, energy=jnp.zeros((n,), jnp.float32)),
        explored=jnp.zeros((h, w), jnp.int32),
        seen_by=jnp.zeros((n, h, w), dtype=bool),
        wall=jnp.zeros((h, w), dtype=bool),
        comm_graph=jnp.eye(n, dtype=bool),
        step_count=jnp.zeros((), jnp.int32),
        channel=(),
        mission=(),
        group=jnp.zeros((n,), jnp.int32),
    )


def relay_outbox(n=3, h=1, w=8):
    """Agent 0 holds one unique bit at cell (0, 0); others know nothing."""
    out = jnp.zeros((n, h, w), dtype=bool)
    return out.at[0, 0, 0].set(True)


# =============================================================================
# DiskTopology
# =============================================================================


class TestDiskTopology:
    def test_chebyshev_symmetric_diag_true(self):
        world = make_world([[0, 0], [0, 3], [3, 3], [9, 9]], 16, 16)
        adj = DiskTopology(3).adjacency(world)
        assert adj.shape == (4, 4) and adj.dtype == jnp.bool_
        assert bool(jnp.diag(adj).all())
        np.testing.assert_array_equal(np.asarray(adj), np.asarray(adj).T)
        # Chebyshev: (0,0)-(3,3) is distance 3 -> adjacent; (0,0)-(9,9) is not.
        assert bool(adj[0, 1]) and bool(adj[0, 2]) and not bool(adj[0, 3])

    def test_satisfies_protocol(self):
        assert isinstance(DiskTopology(2), Topology)

    def test_unknown_metric_rejected_at_construction(self):
        with pytest.raises(NotImplementedError, match="chebyshev"):
            DiskTopology(2, metric="euclidean")


# =============================================================================
# GossipChannel — construction
# =============================================================================


class TestGossipConstruction:
    def test_bad_delay_rejected(self):
        with pytest.raises(ValueError, match="delay"):
            GossipChannel(DiskTopology(2), delay=0)

    def test_bad_dropout_rejected(self):
        with pytest.raises(ValueError, match="dropout"):
            GossipChannel(DiskTopology(2), dropout=1.5)

    def test_bandwidth_reserved(self):
        with pytest.raises(NotImplementedError, match="bandwidth"):
            GossipChannel(DiskTopology(2), bandwidth=8)

    def test_init_shapes(self):
        world = make_world([[0, 0], [0, 1], [0, 2]], 1, 8)
        out0 = relay_outbox()
        st = GossipChannel(DiskTopology(1), delay=3).init(world, out0)
        assert st.shared.shape == (3, 1, 8)
        assert st.buffer.shape == (3, 3, 1, 8)
        np.testing.assert_array_equal(np.asarray(st.shared), np.asarray(out0))
        for d in range(3):
            np.testing.assert_array_equal(
                np.asarray(st.buffer[d]), np.asarray(out0)
            )


# =============================================================================
# GossipChannel — v0 parity gate
# =============================================================================


class TestGossipParity:
    def test_golden_shared_bit_for_bit(self):
        """Replay golden positions/outboxes; shared must match all 70 steps."""
        g = np.load(GOLDEN)
        h, w = int(g["cfg"][0]), int(g["cfg"][1])
        comm_r, t_steps = int(g["cfg"][4]), int(g["cfg"][7])
        positions = jnp.asarray(g["positions"])      # (T+1, N, 2)
        explored_by = jnp.asarray(g["explored_by"])  # (T+1, N, H, W) outboxes
        shared = np.asarray(g["shared"])             # (T+1, N, H, W) target

        chan = GossipChannel(DiskTopology(comm_r), delay=1, dropout=0.0)
        st = chan.init(make_world(positions[0], h, w), explored_by[0])
        np.testing.assert_array_equal(np.asarray(st.shared), shared[0])

        key = jax.random.PRNGKey(0)  # unused on the dropout=0 path

        @jax.jit
        def step(pos, outbox, st):
            return chan.deliver(make_world(pos, h, w), outbox, st, key)

        for t in range(t_steps):
            incoming, st, dadj = step(positions[t + 1], explored_by[t + 1], st)
            np.testing.assert_array_equal(
                np.asarray(st.shared), shared[t + 1],
                err_msg=f"shared mismatch at step {t + 1}",
            )
            # delivered == potential when dropout == 0
            np.testing.assert_array_equal(
                np.asarray(dadj),
                np.asarray(
                    DiskTopology(comm_r).adjacency(
                        make_world(positions[t + 1], h, w)
                    )
                ),
            )

    def test_dropout_zero_ignores_key(self):
        """No-dropout path consumes no randomness: any key, same answer."""
        g = np.load(GOLDEN)
        chan = GossipChannel(DiskTopology(5), delay=1, dropout=0.0)
        world = make_world(g["positions"][1], 16, 16)
        out = jnp.asarray(g["explored_by"][1])
        st = chan.init(make_world(g["positions"][0], 16, 16),
                       jnp.asarray(g["explored_by"][0]))
        inc_a, st_a, adj_a = chan.deliver(world, out, st, jax.random.PRNGKey(0))
        inc_b, st_b, adj_b = chan.deliver(world, out, st, jax.random.PRNGKey(42))
        np.testing.assert_array_equal(np.asarray(st_a.shared),
                                      np.asarray(st_b.shared))
        np.testing.assert_array_equal(np.asarray(adj_a), np.asarray(adj_b))


# =============================================================================
# GossipChannel — delay semantics
# =============================================================================


class TestDelay:
    """3-agent relay chain 0—1—2 (radius 1, agent 2 out of 0's range).

    Agent 0's unique bit reaches agent 1 at step 1 either way (the init
    buffer is pre-filled with the reset outbox), but the RELAYED hop to
    agent 2 lands at step 2 for delay=1 vs step 3 for delay=2.
    """

    def _run(self, delay, n_steps=4):
        world = make_world([[0, 0], [0, 1], [0, 2]], 1, 8)
        out = relay_outbox()
        chan = GossipChannel(DiskTopology(1), delay=delay, dropout=0.0)
        st = chan.init(world, out)
        arrival = {}
        for t in range(1, n_steps + 1):
            _, st, _ = chan.deliver(world, out, st, jax.random.PRNGKey(t))
            for i in range(3):
                if i not in arrival and bool(st.shared[i, 0, 0]):
                    arrival[i] = t
        return arrival

    def test_relay_chain_delay_1_vs_2(self):
        a1 = self._run(delay=1)
        a2 = self._run(delay=2)
        assert a1 == {0: 1, 1: 1, 2: 2}
        assert a2 == {0: 1, 1: 1, 2: 3}
        assert a2[2] == a1[2] + 1  # relayed hop is one step later


# =============================================================================
# GossipChannel — dropout
# =============================================================================


class TestDropout:
    def test_dropout_one_self_only(self):
        """dropout=1.0: delivered_adj == eye; incoming == own old payload."""
        world = make_world([[0, 0], [0, 1], [0, 2]], 1, 8)
        out = relay_outbox()
        chan = GossipChannel(DiskTopology(1), delay=1, dropout=1.0)
        st = chan.init(world, out)
        incoming, st2, dadj = chan.deliver(
            world, out, st, jax.random.PRNGKey(7)
        )
        np.testing.assert_array_equal(
            np.asarray(dadj), np.eye(3, dtype=bool)
        )
        np.testing.assert_array_equal(np.asarray(incoming), np.asarray(out))
        np.testing.assert_array_equal(np.asarray(st2.shared), np.asarray(out))

    def test_dropout_half_jits_symmetric_diag(self):
        world = make_world([[0, 0], [0, 1], [0, 2], [0, 3]], 1, 8)
        out = jnp.zeros((4, 1, 8), dtype=bool).at[0, 0, 0].set(True)
        chan = GossipChannel(DiskTopology(1), delay=1, dropout=0.5)
        st = chan.init(world, out)
        deliver = jax.jit(lambda k: chan.deliver(world, out, st, k))
        potential = np.asarray(DiskTopology(1).adjacency(world))
        for s in range(8):
            incoming, st2, dadj = deliver(jax.random.PRNGKey(s))
            d = np.asarray(dadj)
            np.testing.assert_array_equal(d, d.T)        # symmetric edge fate
            assert d.diagonal().all()                    # diag always True
            assert not (d & ~potential).any()            # delivered ⊆ potential
            # gossip superset invariant: belief never loses bits
            assert bool((st2.shared | ~st.shared).all())
            assert bool((st2.shared | ~out).all())

    def test_dropout_actually_drops_some_edges(self):
        """Across keys, dropout=0.5 must realize BOTH kept and dropped edges."""
        world = make_world([[0, 0], [0, 1], [0, 2], [0, 3]], 1, 8)
        out = jnp.zeros((4, 1, 8), dtype=bool)
        chan = GossipChannel(DiskTopology(1), delay=1, dropout=0.5)
        st = chan.init(world, out)
        potential = np.asarray(DiskTopology(1).adjacency(world))
        offdiag = potential & ~np.eye(4, dtype=bool)
        seen_drop, seen_keep = False, False
        for s in range(32):
            _, _, dadj = chan.deliver(world, out, st, jax.random.PRNGKey(s))
            d = np.asarray(dadj) & offdiag
            seen_drop |= bool((offdiag & ~d).any())
            seen_keep |= bool(d.any())
        assert seen_drop and seen_keep


# =============================================================================
# NullChannel
# =============================================================================


class TestNullChannel:
    def test_passthrough(self):
        world = make_world([[0, 0], [0, 1], [5, 5]], 8, 8)
        out = jnp.zeros((3, 8, 8), dtype=bool).at[1, 2, 3].set(True)
        chan = NullChannel()
        st = chan.init(world, out)
        assert st == ()
        incoming, st2, dadj = chan.deliver(
            world, out, st, jax.random.PRNGKey(0)
        )
        np.testing.assert_array_equal(np.asarray(incoming), np.asarray(out))
        assert st2 == ()
        np.testing.assert_array_equal(
            np.asarray(dadj), np.eye(3, dtype=bool)
        )


# =============================================================================
# jit / vmap safety
# =============================================================================


class TestTransforms:
    def test_deliver_vmaps_over_keys(self):
        world = make_world([[0, 0], [0, 1], [0, 2]], 1, 8)
        out = relay_outbox()
        chan = GossipChannel(DiskTopology(1), delay=2, dropout=0.5)
        st = chan.init(world, out)
        keys = jax.random.split(jax.random.PRNGKey(0), 5)
        incoming, st2, dadj = jax.vmap(
            lambda k: chan.deliver(world, out, st, k)
        )(keys)
        assert incoming.shape == (5, 3, 1, 8)
        assert st2.shared.shape == (5, 3, 1, 8)
        assert st2.buffer.shape == (5, 2, 3, 1, 8)
        assert dadj.shape == (5, 3, 3)

    def test_channel_state_is_scan_carry(self):
        """Buffer structure is fixed-shape: ChannelState works as a scan carry."""
        world = make_world([[0, 0], [0, 1], [0, 2]], 1, 8)
        out = relay_outbox()
        chan = GossipChannel(DiskTopology(1), delay=2, dropout=0.0)
        st0 = chan.init(world, out)

        def body(st, _):
            _, st, dadj = chan.deliver(world, out, st, jax.random.PRNGKey(0))
            return st, dadj

        st_t, dadjs = jax.lax.scan(body, st0, xs=None, length=6)
        assert st_t.buffer.shape == st0.buffer.shape
        assert dadjs.shape == (6, 3, 3)
        assert bool(st_t.shared[2, 0, 0])  # bit flooded to the far agent
