# zymera

A JAX-native toolkit for **designing agent mechanisms and learning stacks** on
communication-constrained cooperative (and adversarial) swarm missions. One
package, `zymera`: a multi-agent grid **simulator** plus a **lab** — `nets.py`
(composable agent building blocks) and `train.py` (trainers + shared utils). You
write experiments *separately* and import it.

## Structure

```
zymera/      simulator: env · worldgen · dynamics · comms · obs · missions · metrics · rollout · viz · sensor
             + nets.py (agent building blocks) + train.py (trainers + utils)
tests/
```

Experiments live in `../zymera_experiments/` (a separate sibling folder) and
import `zymera`. The dependency is one-way: `zymera_experiments → zymera`; the
simulator never imports `nets`/`train`. Proven blocks/trainers graduate from an
experiment into `nets.py`/`train.py` with a test.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,viz]"      # core is headless; [viz] adds matplotlib/pillow
pytest tests/ -q                 # full suite, incl. the golden-file parity gate
```

`[lab]` (equinox, optax) is declared for the trainers that arrive next (P2).

## Quick start

```python
import jax, zymera
from zymera import train, viz

env = zymera.make("comm-coverage", grid=12, n_agents=4)
print(train.evaluate(env, zymera.random_policy, n_steps=40, n_episodes=8,
                     key=jax.random.PRNGKey(0)))

traj = zymera.rollout(env, zymera.random_policy, 40, jax.random.PRNGKey(1), keep="all")
viz.render_gif(traj["world"], "rollout.gif", comm_radius=5)
```

Design: `docs/specs/2026-06-25-zymera-lab-design.md`. Scaffold plan:
`docs/plans/2026-06-25-zymera-lab-p1-scaffold.md`.
