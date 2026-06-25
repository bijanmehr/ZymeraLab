# Zymera Lab — P1 Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the lean lab structure — `zymera.sensor` + `zymera.nets` + `zymera.train` seeds, a separate `zymera_experiments/` folder, and docs rewritten to the new design — with the existing 216 tests still green.

**Architecture:** The `zymera` package gains three flat modules next to the simulator: `sensor.py` (host visibility), `nets.py` (composable agent building blocks), `train.py` (trainers + shared utils). Experiments live OUTSIDE the package in `Project.Zymera/zymera_experiments/`, importing `zymera`. Build clean — each module starts with one genuinely-reusable element and grows when an experiment needs more.

**Tech Stack:** JAX, chex, numpy (raw JAX for the P1 seeds; equinox/optax deferred to P2). pytest.

Work on branch `zymera-lab-consolidation`. Run all commands from `Project.Zymera/`. The venv is `zymera_lab/.venv`; pytest entry: `zymera_lab/.venv/bin/python -m pytest zymera_lab/tests -q`.

---

### Task 1: `zymera.sensor` — host radius-visibility sensor

**Files:**
- Create: `zymera_lab/zymera/sensor.py`
- Test: `zymera_lab/tests/test_sensor.py`

- [ ] **Step 1: Write the failing test**

```python
# zymera_lab/tests/test_sensor.py
import numpy as np
from zymera import sensor

def test_chebyshev_radius_and_walls():
    wall = np.zeros((5, 5), dtype=bool)
    wall[0, 0] = True                      # a wall cell is never visible
    vis = sensor.visible_cells((2, 2), wall, radius=1)   # pos = (row, col)
    assert vis.shape == (5, 5)
    assert vis[2, 2] and vis[1, 1] and vis[3, 3]         # within Chebyshev radius 1
    assert not vis[0, 0]                                 # wall
    assert not vis[4, 4]                                 # outside radius

def test_radius_zero_is_just_own_cell():
    wall = np.zeros((3, 3), dtype=bool)
    vis = sensor.visible_cells((1, 1), wall, radius=0)
    assert vis.sum() == 1 and vis[1, 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `zymera_lab/.venv/bin/python -m pytest zymera_lab/tests/test_sensor.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'zymera.sensor'`

- [ ] **Step 3: Write minimal implementation**

```python
# zymera_lab/zymera/sensor.py
"""Host-side visibility sensor (a numpy analysis tool, not on the jit path).

v1 is non-occluded: a cell is visible from ``pos`` iff it is within ``radius``
under the chosen metric and is not a wall. Occlusion (ray-casting) is a P2
extension when an experiment needs it. Positions are ``(row, col)``.
"""
from __future__ import annotations

import numpy as np


def visible_cells(pos, wall, radius: int, metric: str = "chebyshev"):
    """Return a ``(H, W)`` bool mask of cells visible from ``pos``."""
    wall = np.asarray(wall, dtype=bool)
    h, w = wall.shape
    rr, cc = np.mgrid[0:h, 0:w]
    pr, pc = int(pos[0]), int(pos[1])
    if metric == "chebyshev":
        dist = np.maximum(np.abs(rr - pr), np.abs(cc - pc))
    elif metric == "manhattan":
        dist = np.abs(rr - pr) + np.abs(cc - pc)
    else:
        raise ValueError(f"unknown metric {metric!r}")
    return (dist <= radius) & (~wall)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `zymera_lab/.venv/bin/python -m pytest zymera_lab/tests/test_sensor.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git -C zymera_lab add zymera/sensor.py tests/test_sensor.py
git -C zymera_lab commit -m "feat(zymera): host radius-visibility sensor (P1)"
```

---

### Task 2: `zymera.nets` — first building block (raw-JAX MLP)

**Files:**
- Create: `zymera_lab/zymera/nets.py`
- Test: `zymera_lab/tests/test_nets.py`

- [ ] **Step 1: Write the failing test**

```python
# zymera_lab/tests/test_nets.py
import jax, jax.numpy as jnp
from zymera import nets

def test_mlp_shapes_and_determinism():
    key = jax.random.PRNGKey(0)
    params = nets.mlp_init(key, sizes=(4, 8, 2))
    x = jnp.ones((3, 4))                       # batch of 3, in-dim 4
    y = nets.mlp_apply(params, x)
    assert y.shape == (3, 2)
    y2 = nets.mlp_apply(params, x)
    assert jnp.allclose(y, y2)                 # pure function

def test_mlp_jittable():
    key = jax.random.PRNGKey(1)
    params = nets.mlp_init(key, sizes=(2, 4, 1))
    f = jax.jit(lambda p, x: nets.mlp_apply(p, x))
    out = f(params, jnp.zeros((1, 2)))
    assert out.shape == (1, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `zymera_lab/.venv/bin/python -m pytest zymera_lab/tests/test_nets.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'zymera.nets'`

- [ ] **Step 3: Write minimal implementation**

```python
# zymera_lab/zymera/nets.py
"""Composable agent building blocks (the parts you wire into a policy).

Raw-JAX and functional so blocks compose freely and stay jit/vmap-safe. Start
minimal: this seeds the module with one foundational block (an MLP). Add encoders,
belief nets, attention, aggregators, low-level controllers here as experiments
need them; graduate proven blocks in with a test.
"""
from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp


def mlp_init(key: jax.Array, sizes: Sequence[int]):
    """Glorot-init params for an MLP with the given layer sizes."""
    params = []
    keys = jax.random.split(key, len(sizes) - 1)
    for k, d_in, d_out in zip(keys, sizes[:-1], sizes[1:]):
        scale = jnp.sqrt(2.0 / (d_in + d_out))
        w = jax.random.normal(k, (d_in, d_out)) * scale
        b = jnp.zeros((d_out,))
        params.append((w, b))
    return params


def mlp_apply(params, x: jax.Array, activation=jax.nn.relu) -> jax.Array:
    """Apply the MLP; activation on every layer except the last."""
    for i, (w, b) in enumerate(params):
        x = x @ w + b
        if i < len(params) - 1:
            x = activation(x)
    return x
```

- [ ] **Step 4: Run test to verify it passes**

Run: `zymera_lab/.venv/bin/python -m pytest zymera_lab/tests/test_nets.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git -C zymera_lab add zymera/nets.py tests/test_nets.py
git -C zymera_lab commit -m "feat(zymera): nets.py seed — raw-JAX MLP building block (P1)"
```

---

### Task 3: `zymera.train` — first shared util (`evaluate`)

**Files:**
- Create: `zymera_lab/zymera/train.py`
- Test: `zymera_lab/tests/test_train.py`

- [ ] **Step 1: Write the failing test**

```python
# zymera_lab/tests/test_train.py
import jax
import zymera
from zymera import train

def test_evaluate_random_policy_returns_metrics():
    env = zymera.make("comm-coverage", grid=8, n_agents=2)
    report = train.evaluate(env, zymera.random_policy, n_steps=16,
                            n_episodes=4, key=jax.random.PRNGKey(0))
    assert set(report).issuperset({"return_mean", "return_std", "n_episodes"})
    assert report["n_episodes"] == 4
    assert report["return_mean"] == report["return_mean"]   # not NaN
```

- [ ] **Step 2: Run test to verify it fails**

Run: `zymera_lab/.venv/bin/python -m pytest zymera_lab/tests/test_train.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'zymera.train'`

- [ ] **Step 3: Write minimal implementation**

```python
# zymera_lab/zymera/train.py
"""Trainers + shared training utilities.

Independent trainers (ppo / es / supervised) will live here as flat functions
over shared utils, with NO forced common interface — added when an experiment
needs them. P1 seeds the shared-utils side with ``evaluate`` (the doctrine: eval
= sample policy, multi-seed via vmap). Experiments WRITE their learning stack by
importing these + ``zymera.nets``.
"""
from __future__ import annotations

from typing import Callable, Dict

import jax
import jax.numpy as jnp

from .rollout import rollout


def evaluate(env, policy: Callable, n_steps: int, n_episodes: int,
             key: jax.Array) -> Dict[str, float]:
    """Roll ``policy`` in ``env`` over ``n_episodes`` seeds (vmap, no python loop);
    report summed-reward statistics. Eval samples the policy — never argmax."""
    keys = jax.random.split(key, n_episodes)
    trajs = jax.vmap(lambda k: rollout(env, policy, n_steps, k))(keys)
    ep_returns = trajs["reward"].sum(axis=(1, 2))          # (n_episodes,)
    return {
        "return_mean": float(jnp.mean(ep_returns)),
        "return_std": float(jnp.std(ep_returns)),
        "n_episodes": int(n_episodes),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `zymera_lab/.venv/bin/python -m pytest zymera_lab/tests/test_train.py -q`
Expected: PASS (1 passed). If `reward` axes differ, inspect `rollout` output keys
(`zymera/rollout.py`) — `reward` is stacked `(T, N)` per episode, so summing axes
`(1, 2)` over the vmapped `(n_episodes, T, N)` array is correct.

- [ ] **Step 5: Commit**

```bash
git -C zymera_lab add zymera/train.py tests/test_train.py
git -C zymera_lab commit -m "feat(zymera): train.py seed — evaluate() shared util (P1)"
```

---

### Task 4: `zymera_experiments/` — the separate experiments folder

**Files:**
- Create: `zymera_experiments/README.md`
- Create: `zymera_experiments/00_random_rollout.py`

- [ ] **Step 1: Create the folder README**

```markdown
# zymera_experiments

Experiments live HERE, outside the `zymera_lab/` library. Each file imports
`zymera` (the simulator + `nets` + `train`) and writes its own learning stack —
wire blocks into a policy, pick/compose a trainer, define the mission, run.

**Dependency rule (one-way):** `zymera_experiments → zymera`. Never edit the
library to make an experiment work; if a block or trainer proves out across >1
experiment, graduate it into `zymera/nets.py` / `zymera/train.py` with a test.

Run an experiment against the lab's venv, e.g.:

    ../zymera_lab/.venv/bin/python 00_random_rollout.py
```

- [ ] **Step 2: Create the example experiment**

```python
# zymera_experiments/00_random_rollout.py
"""Smallest possible experiment: a random policy on comm-coverage, rendered.

Demonstrates the import pattern and the one-way dependency on the lab. Not part
of the lab's test suite."""
import jax

import zymera
from zymera import train


def main() -> None:
    env = zymera.make("comm-coverage", grid=12, n_agents=4)
    report = train.evaluate(env, zymera.random_policy, n_steps=40,
                            n_episodes=8, key=jax.random.PRNGKey(0))
    print("random-policy eval:", report)

    traj = zymera.rollout(env, zymera.random_policy, 40, jax.random.PRNGKey(1),
                          keep="all")
    zymera.viz.render_gif(traj, "random_rollout.gif")
    print("wrote random_rollout.gif")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the example to verify it works end-to-end**

Run: `zymera_lab/.venv/bin/python zymera_experiments/00_random_rollout.py`
Expected: prints `random-policy eval: {...}` and `wrote random_rollout.gif`; the
gif file appears. (Confirms `zymera_experiments → zymera` works and `train.evaluate`
+ the gif renderer are reachable from outside the package.)

> **Check the viz API first:** confirm the gif-render entry name in
> `zymera_lab/zymera/viz/__init__.py`. If it exports `render` (not `render_gif`),
> change the example's `zymera.viz.render_gif(...)` call accordingly.

- [ ] **Step 4: Commit**

```bash
git -C zymera_lab add -A    # nothing — experiments are outside the repo
cd zymera_experiments && git init -q 2>/dev/null; cd ..
# experiments folder is a plain workspace folder (optionally its own repo later);
# no commit into zymera_lab. Leave the generated gif untracked.
echo "zymera_experiments/ created (separate from the lab repo)"
```

---

### Task 5: Rewrite `CLAUDE.md` + `README.md` to the new design

The current `zymera_lab/CLAUDE.md` and `README.md` still describe the superseded
kymera design (the `zymera.lab` two-layer, "successor to zymera", "migration in
progress"). Rewrite both to the new structure: one `zymera` package = simulator +
`nets` + `train`; experiments separate in `zymera_experiments/`; build clean.

**Files:**
- Modify: `zymera_lab/CLAUDE.md` (full rewrite of "What this is", build-state, and the agent/lab sections)
- Modify: `zymera_lab/README.md` (intro + structure)

- [ ] **Step 1: Rewrite `zymera_lab/CLAUDE.md`**

Replace the stale framing with: `zymera` = JAX simulator + lab (`nets.py` building
blocks, `train.py` trainers/utils); `sensor.py` host visibility; experiments live
in `../zymera_experiments/` and import the package (one-way). Keep the accurate
parts (the five sim components, env/step contract, reward-term zoo, doctrine, the
216-test + golden-parity note). Remove: the `zymera.lab` subpackage claim, the
"successor to zymera" line, the "migration in progress / dual-maintenance" section,
and the movement-only/env-mediated-comms "hard stance" (the contract now has P3
seams for hierarchy + learned comms — note them as planned, not forbidden).

- [ ] **Step 2: Rewrite `zymera_lab/README.md`**

Intro: "`zymera` — a JAX-native toolkit for designing agent mechanisms and learning
stacks on communication-constrained swarm missions." Structure block mirroring the
spec (`zymera/` package with sim + `nets` + `train`; `zymera_experiments/` separate).
Install: `pip install -e ".[dev,viz]"`; test: `pytest tests/ -q`.

- [ ] **Step 3: Verify the docs reference no removed concepts**

Run: `grep -nE "kymera|zymera\.lab|successor to zymera|migration in progress" zymera_lab/CLAUDE.md zymera_lab/README.md`
Expected: no matches.

- [ ] **Step 4: Commit**

```bash
git -C zymera_lab add CLAUDE.md README.md
git -C zymera_lab commit -m "docs: rewrite CLAUDE.md + README to the new zymera-lab design (P1)"
```

---

### Task 6: Full green gate

- [ ] **Step 1: Run the entire suite**

Run: `zymera_lab/.venv/bin/python -m pytest zymera_lab/tests -q`
Expected: `222 passed` (the original 216 + sensor 2 + nets 2 + train 1 + any). The
golden-file parity gate is unaffected by the new modules.

- [ ] **Step 2: Confirm the public surface**

Run: `zymera_lab/.venv/bin/python -c "import zymera; from zymera import nets, train, sensor; print('lab modules OK')"`
Expected: `lab modules OK`

---

## Notes for P2 / P3 (out of scope here)

- **P2 (grow):** add real trainers (`train.ppo`, `train.es`, `train.supervised`) and blocks (graph belief, role-switcher, certainty-field exploration), reimplemented clean from the archived `zymera_env` reference, driven by the first real experiments; pull in `[lab]` deps (equinox, optax) then.
- **P3 (extend the contract):** hierarchy via lab low-level controllers (policy emits a goal → controller emits movement; no sim change); then learned comms via a scoped `zymera.comms` extension (sim change, re-baseline-gated).
