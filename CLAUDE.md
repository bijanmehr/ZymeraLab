# CLAUDE.md — zymera_lab

Guidance for Claude Code working **inside `zymera_lab/`**. Workspace overview: `../CLAUDE.md`.
Experiments live in `../zymera_experiments/` (separate). Archived reference: `../zymera_env/`.

## What this is

`zymera_lab/` is one Python package, **`zymera`** — a JAX-native toolkit for **designing agent mechanisms
and learning stacks** on communication-constrained cooperative (and adversarial) swarm missions. Three
flat layers, all in the `zymera` package:

- **simulator** — the grid env + components (`env`, `worldgen`, `dynamics`, `comms`, `obs`, `missions`,
  `metrics`, `rollout`, `viz`, `sensor`).
- **`nets.py`** — composable agent **building blocks** (the parts you wire into a policy).
- **`train.py`** — **trainers** + shared training utilities.

**Experiments are separate.** They live in `../zymera_experiments/`, import `zymera`, and write their own
learning stacks. The dependency is **one-way**: `zymera_experiments → zymera`; the simulator never imports
`nets`/`train`. Proven blocks/trainers graduate from an experiment into `nets.py`/`train.py` with a test.

The lab is built **clean**: `nets.py`/`train.py` start minimal and grow as experiments need them,
referencing the archived `../zymera_env/` for algorithms — no bulk-copy of the old engine/sprawl.

## Build state

- **Simulator:** built, **green** (golden-file parity gate included). Recipes: `comm-coverage`, `empty`.
- **Lab seeds (P1):** `nets.py` (one block — `mlp_init`/`mlp_apply`), `train.py` (`evaluate`),
  `sensor.py` (radius visibility). These are seeds; grow them as experiments need them.
- Reward-term zoo, grouping/adversarial substrate (`GroupedMission`/`RandomKofN`), potential-vs-delivered
  comms: implemented.
- **Tests:** `pytest tests/ -q` → **222 passed** (216 simulator + sensor/nets/train).

## Setup & commands

Python ≥ 3.10; venv `zymera_lab/.venv`.

```bash
source .venv/bin/activate
pip install -e ".[dev,viz]"          # core + tests + plotting. ([lab] = equinox/optax, for P2 trainers.)
pytest tests/ -q                      # 222 tests incl. the golden-file parity gate
python -c "import zymera; print(zymera.list_envs())"   # ['comm-coverage', 'empty']
# run an experiment (from the sibling folder):
.venv/bin/python ../zymera_experiments/00_random_rollout.py
```

## The agent contract

A policy is a **callable convention**, not a base class:

```
policy(obs, state, key) -> (action, state)
```

- `nets.py` provides the parts; an experiment wires them into a `policy`. `zymera.random_policy` is the
  reference stateless policy; `rollout(env, policy, n_steps, key)` runs episodes (`vmap` over key for seeds).
- `state` carries per-agent recurrent/memory (e.g. a belief); `()` for stateless policies.
- `action` today is movement `(N,) int32` (`STAY/UP/DOWN/LEFT/RIGHT`, `N_ACTIONS = 5`). The contract keeps
  **planned seams** (P3, not yet built): **hierarchy** — a policy emits a goal, a lab low-level controller
  turns it into movement (no sim change); **learned comms** — policies emit messages, a scoped `comms`
  extension transports them (sim change, re-baseline-gated).

`train.py` holds **independent trainers** (ppo/es/supervised arrive in P2) + shared utils (`evaluate` so
far), with **no forced common interface**. An experiment writes its learning stack by importing a trainer
+ `nets`.

## Public API (simulator)

```
make · make_from · list_envs · register_env · Env · GridEnv · World · Body ·
ActionId · ACTION_DELTAS · N_ACTIONS · RewardTerm · rollout · random_policy
subnamespaces:  worldgen · dynamics · comms · obs · missions · missions_terms · metrics · viz
lab modules:    nets · train · sensor
```

```python
import jax, zymera
from zymera import nets, train, viz
env = zymera.make("comm-coverage", grid=16, n_agents=4)
env = zymera.make("comm-coverage", grid=16, n_agents=4,
                  comm=zymera.comms.GossipChannel(zymera.comms.DiskTopology(5), dropout=0.1),
                  terms=[("coverage", 1.0), ("capped_giant", 1.5, dict(cap=3))])
env2 = env.replace(terms=[...])                 # cheap variation
spec = env.spec(); env3 = zymera.make_from(spec)   # round-trips to a plain dict (YAML/JSON-able)
```

## Architecture — the simulator components

Every component is a `dataclass(frozen=True)` of hashable Python values with pure-JAX methods; jitted code
closes over `env` (components are trace-time constants). Files in `zymera/`:

- **`worldgen.py`** — `Terrain` (`OpenTerrain`, `RandomWalls(n)`, `MapFile`, `Rooms`) + `Spawn`
  (`ScatterSpawn`, `ClusterSpawn(radius)`, `FixedSpawn`).
- **`dynamics.py`** — `GridDynamics(collision=…)` with `CollisionRule` (`NoCollision`, `SequentialClaim`
  — `lax.scan` over agents). `targets(world)` is the single source of truth for valid next-cells;
  `action_mask(world)`; `sequential_masked_sample(...)` — collision-free action sampling, agent order
  pinned 0..N-1.
- **`comms.py`** — `Topology` (`DiskTopology(radius, metric="chebyshev")`) + channels (`GossipChannel`
  with `delay`/`dropout`/`bandwidth`, `NullChannel`). **Potential vs delivered** edges are distinct:
  `Topology.adjacency` = who *could* talk; `World.comm_graph` = what was *delivered*. Reward/metrics read
  potential topology unless they opt into delivered; both are logged.
- **`obs.py`** — `ObsBuilder` (`VectorObs`, `GridObs(channels=…, sense_r=…, central=…)`). Channels are
  named functions in `CHANNEL_FNS` (`known`, `own_pos`, `known_walls`, `neighbors`, `local_frontier`,
  `team_explored`, `all_pos`, `walls`); add your own with `obs.register_channel(name, fn)`. `central=…` is
  the CTDE critic view. `requires` declares needed `StepCtx` fields so unused computation never compiles.
- **`missions.py`** — reward-as-data. `RewardTerm(name, weight, fn, requires)`; `Mission(terms=…)`;
  drawable `annotations` (`Point`/`Path`/`Region`). **Grouping / adversarial:** `Assignment` protocol
  (`FixedAssignment`, `RandomKofN(k, group=1)`) + `GroupedMission(assignment, missions)` routes per-group
  objectives over a shared world, metrics namespaced `g0/…`, `g1/…`.

Plus **`metrics.py`** (`StepCtx` — `derive(prev, world, requires, topology)`; pairwise dist, adjacency,
reach, giant component, coverage, redundancy, dist-to-frontier…), **`rollout.py`**
(`rollout(env, policy, n_steps, key, keep=, collect=)`, `random_policy`), **`viz/`** (`render_gif`,
`render_comm_gif`, `make_report`; `render_gif` takes `traj["world"]`), **`sensor.py`** (host radius
visibility — `visible_cells(pos, wall, radius)`).

## Env & step contract

State is two pytrees (`World` holds `Body`):

```
Body:  position (N,2) i32 · energy (N,) f32
World: body · explored (H,W) i32 visit counts · seen_by (N,H,W) bool (own knowledge) · wall (H,W) ·
       comm_graph (N,N) bool DELIVERED edges · step_count · channel (channel pytree) · mission (pytree) ·
       group (N,) i32 (assigned at reset)
```

`step(state, action, key)` order (frozen, parity rides on it): dynamics→explored→own knowledge→
channel.deliver→metrics.derive→mission.update(NPCs)→mission.reward→mission.done→obs. Key split is a
documented contract: `reset_key,scan_key = split(key)`; per step `k,action_key,step_key = split(k,3)`;
group assignment `fold_in(key,1)`, mission init `fold_in(key,2)`. `info` keys are fixed:
`{explored, step_count, seen_by, comm_graph, reward_terms, metrics}`.

## Reward-term zoo (`missions_terms.py`, implemented)

`new_coverage`, `reach_fraction`, `capped_giant(cap)` (connectivity), `collision_count`,
`same_step_overlap`, `cohesion_leash(leash, comm_r)`, `degree_floor(floor, comm_r)`,
`pbrs(phi, gamma)` with `phi_nearest_frontier` / `phi_field_mean`, `cbf_conn(...)`, `cbf_coll(...)`.
Terms are **unweighted** `(N,)` fns logged separately so analysis can re-weight post-hoc. Add a term
locally in an experiment file; if it proves out across >1 experiment, graduate it here with a test.

## Doctrine & conventions (enforced by tests)

- Eval = **sample π**, ε-free; never argmax. Multi-seed via `vmap`, never a Python loop. Fixed-length
  scan episodes; `lax.scan`-over-agents for sequential conflict resolution.
- Components are frozen/hashable static objects; **no Python branching on traced values**; `requires`-
  gating decides what compiles. Config objects never cross the jit boundary.
- The **term SET is static**; weights may become traced later — never gate on `w == 0`.
- Top-level simulator `__all__` stays small; components live in subnamespaces.
- Reset/step key-split order is frozen; connectivity terms read potential topology unless opting into
  delivered (both logged).

## Where to read more

- **Design:** `docs/specs/2026-06-25-zymera-lab-design.md`. **Scaffold plan:**
  `docs/plans/2026-06-25-zymera-lab-p1-scaffold.md`.
- Older `docs/specs/2026-06-11-*` / `docs/plans/2026-06-12-*` describe the original simulator design and
  are partly superseded by the consolidation spec above.
