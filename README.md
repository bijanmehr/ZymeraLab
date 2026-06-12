# kymera

JAX-native multi-agent grid **simulator library** for communication-constrained
swarm research. Like `gymnax`, but multi-agent — you write the training loop;
kymera gives you the env, the rollout primitive, the metrics, and the viz.

Successor to `zymera` (frozen at `../zymera_env/` as the parity reference),
rebuilt around **composition**: an env is five swappable components — worldgen,
dynamics, comms, obs, mission — on one orchestrator. A new research idea is a
new component, usually a plain function in your experiment script, never an
edit to the simulator.

> **Parity:** with the same PRNGKey and policy, kymera reproduces zymera v0
> bit-for-bit (walls, spawns, actions, positions, gossip belief maps,
> observations) and per-step rewards to 1e-5 — gated in `tests/test_parity.py`
> against golden trajectories dumped from the live v0 install.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[viz,dev]"      # core is headless; [viz] adds matplotlib/pillow
pytest tests/ -q                 # full suite incl. the v0 parity gate
```

## Five-line quick start

```python
import jax, kymera
from kymera import viz

env  = kymera.make("comm-coverage", grid=16, n_agents=4, comm_r=5)
traj = kymera.rollout(env, kymera.random_policy, 70, jax.random.PRNGKey(0),
                      keep="all", collect=("reward_terms",))
viz.make_report(traj, "report.html", env=env)
```

## Docs

Open **`docs/index.html`** in a browser — overview, public API, component map,
doctrine. The env walkthrough (recipes → World → composing → custom reward
terms → red-within-blue groups → custom envs → viz → memory profiles) is
**`docs/tutorial-env.html`**; every code block there was executed against this
build. Architecture rationale: `docs/specs/2026-06-11-kymera-design.md`.

## Public API (14 top-level names + 7 subnamespaces)

| name | what |
| --- | --- |
| `make` / `make_from` / `register_env` / `list_envs` | recipe registry; `env.spec()` round-trips |
| `Env` / `GridEnv` | gym-style base; the component orchestrator |
| `World` / `Body` / `ActionId` / `N_ACTIONS` / `ACTION_DELTAS` | the frozen state/action contract |
| `RewardTerm` | named, weighted, composable reward component |
| `rollout(env, policy, n, key, keep=, collect=)` | `lax.scan` rollout, vmap-friendly |
| `random_policy` | the example policy |

Subnamespaces: `kymera.worldgen` · `dynamics` · `comms` · `obs` · `missions` ·
`missions_terms` · `metrics` · `viz` (opt-in extra).

## Env contract

```python
obs, state = env.reset(key)
obs, state, reward, done, info = env.step(state, action, key)   # action: (N,) int32
```

`state` is an immutable pytree (jit/vmap/scan-safe). `info` has a fixed keyset:
`explored`, `step_count`, `seen_by`, `comm_graph` (delivered edges),
`reward_terms` (unweighted per-term `(N,)`), `metrics`.

## Status — v0.1.0

**Done:** core library + orchestrator (230+ tests, full v0 parity), recipes
`empty` / `comm-coverage`, grouped missions with k-of-N assignment, term
library (coverage/connectivity/collision/overlap/cohesion/degree/PBRS/soft-CBF),
viz (gif, comm overlay with potential-vs-delivered edges, self-contained HTML
report with per-term curves, mission annotations).

**Deferred (next milestones):** `kymera.lab` — unified PPO trainer
(flags-not-forks), `evaluate` with the sample-π doctrine + lie-proofing, run
provenance (`experiments/index.jsonl`, git fingerprints) — plus the trainer
shadow run vs v0, isometric renderer + keyboard teleop, channel bandwidth,
jamming, energy mechanics, traced-knob sweeps.

## Doctrine (enforced, not advisory)

Eval samples π, never argmax · multi-seed via vmap, never Python loops ·
fixed-length scan episodes; scan-over-agents conflict resolution · the reward
term **set** is static (never gate on `w == 0`) · connectivity reads potential
topology unless a term opts into delivered edges · the reset/step key-split
order is frozen.
