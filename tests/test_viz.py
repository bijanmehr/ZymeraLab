"""Viz smoke tests (Agg backend via conftest)."""

import jax
import pytest

import zymera
from zymera import viz


@pytest.fixture(scope="module")
def traj():
    env = zymera.make("comm-coverage", grid=6, n_agents=3, comm_r=2, spawn_radius=1)
    return zymera.rollout(env, zymera.random_policy, 5, jax.random.PRNGKey(0),
                          keep="all", collect=("reward_terms",))


def test_render_gif_from_stacked(tmp_path, traj):
    p = viz.render_gif(traj["world"], str(tmp_path / "ep.gif"), comm_radius=2)
    assert (tmp_path / "ep.gif").stat().st_size > 1000
    assert p.endswith(".gif")


def test_render_gif_from_list(tmp_path, key):
    env = zymera.make("empty", grid_h=5, grid_w=5, n_agents=2)
    _, s = env.reset(key)
    worlds = [s]
    for _ in range(3):
        _, s, *_ = env.step(s, jax.numpy.zeros((2,), jax.numpy.int32), key)
        worlds.append(s)
    viz.render_gif(worlds, str(tmp_path / "list.gif"))
    assert (tmp_path / "list.gif").exists()


def test_make_report(tmp_path, traj):
    env = zymera.make("comm-coverage", grid=6, n_agents=3, comm_r=2, spawn_radius=1)
    p = viz.make_report(traj, str(tmp_path / "r.html"), env=env, title="t")
    html = open(p).read()
    assert "base64" in html and "Reward terms" in html and "GridEnv" in html


def test_headless_core_unaffected():
    import subprocess, sys
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys, zymera; sys.exit(1 if 'matplotlib' in sys.modules else 0)"],
        capture_output=True)
    assert out.returncode == 0
