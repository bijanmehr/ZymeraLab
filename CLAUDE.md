# CLAUDE.md — kymera

This file provides guidance to Claude Code (claude.ai/code) when working **inside `kymera/`**.
Workspace overview: `../CLAUDE.md`. The codebase kymera supersedes: `../zymera_env/CLAUDE.md`.

## What this is

`kymera` (after *chimera* — parts composed) is a JAX-native multi-agent grid **simulator library** for
cooperative, communication-constrained, and adversarial swarm research. It is the **successor to
`zymera`**, rebuilt around composition: *a new research idea is a new component written in an experiment
script, never an edit to the simulator.* Its own git repo. It is the **destination for new experiments**
(migration in progress).

Design intent: **one installable library, two layers** — `kymera` (the simulator) + `kymera.lab` (the
trainer / eval / run-provenance machinery). Experiments stay downstream: ~20-line files that import
kymera, define ideas locally, and "graduate" proven ones into the library. This is the deliberate
antidote to `zymera_env/examples/` sprawl.

## Build state — read this first

All **216 tests pass** (15 s), including a **bit-parity gate against zymera v0**. But kymera is a
*simulator you cannot train in yet*:

| Layer | State |
|------|-------|
| Simulator: `env`, `worldgen`, `dynamics`, `comms`, `obs`, `missions`(+`missions_terms`), `metrics`, `rollout` | ✅ built, parity-tested |
| Recipes | ✅ `empty`, `comm-coverage` registered |
| Reward-term zoo (`missions_terms.py`) | ✅ implemented (see below) |
| Grouping / adversarial substrate (`GroupedMission`, `FixedAssignment`, `RandomKofN`, potential-vs-delivered comms) | ✅ implemented |
| `viz` | ◑ flat `render` + `report` only (no iso / comm-overlay / annotations / teleop yet) |
| **`kymera.lab`** (PPO trainer + eval + provenance) | ❌ **not built** — the gate for any learned experiment |
| **`sensor.py`** (host occlusion sensor) | ❌ not built (in-env partial obs via `GridObs(sense_r=…)` works) |

**Implication:** you can compose/register envs, missions, groups, adversaries and run `rollout` under
random/scripted policies *today*; you cannot train a policy with the library until `kymera.lab` exists
(or you hand-roll a loop on `kymera.rollout`).

## Setup & commands

Python ≥ 3.10; venv at `kymera/.venv` (present). No console scripts yet.

```bash
source .venv/bin/activate
pip install -e ".[dev,viz]"          # core + tests + plotting. ".[lab]" extra exists for when lab lands.
pytest tests/ -q                      # 216 tests incl. the v0 parity gate
python -c "import kymera; print(kymera.list_envs())"   # ['comm-coverage', 'empty']
```

Deps: jax, jaxlib, chex, numpy. Extras: `[viz]` (matplotlib, pillow), `[lab]` (equinox, optax, msgspec,
pyyaml — **declared, layer not built**), `[dev]` (pytest).

## Public API

~13 top-level names + subnamespaces (the cap is a design rule — components stay behind subnamespaces):

```
make · make_from · list_envs · register_env · Env · GridEnv · World · Body ·
ActionId · ACTION_DELTAS · N_ACTIONS · RewardTerm · rollout · random_policy
subnamespaces:  worldgen · dynamics · comms · obs · missions · missions_terms · metrics  (+ viz, lab[future])
```

```python
import kymera as ky
env = ky.make("comm-coverage", grid=16, n_agents=4)                 # recipe + good defaults
env = ky.make("comm-coverage", grid=16, n_agents=4,
              comm=ky.comms.GossipChannel(ky.comms.DiskTopology(5), dropout=0.1),
              terms=[("coverage", 1.0), ("capped_giant", 1.5, dict(cap=3))])
env2 = env.replace(terms=[...])          # cheap variation
spec = env.spec(); env3 = ky.make_from(spec)   # round-trips to plain dict (YAML/JSON-able)
```

## Architecture — the five components

Every component is a `dataclass(frozen=True)` of hashable Python values with pure-JAX methods; jitted
code closes over `env` (components are trace-time constants). Files in `kymera/`:

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
  `team_explored`, `all_pos`, `walls`); add your own with `obs.register_channel(name, fn)` from an
  experiment file. `central=…` is the CTDE critic view. `requires` declares needed `StepCtx` fields so
  unused computation never compiles.
- **`missions.py`** — reward-as-data. `RewardTerm(name, weight, fn, requires)`; `Mission(terms=…)`;
  drawable `annotations` (`Point`/`Path`/`Region`). **Grouping / adversarial:** `Assignment` protocol
  (`FixedAssignment`, `RandomKofN(k, group=1)`) + `GroupedMission(assignment, missions)` routes per-group
  objectives over a shared world, metrics namespaced `g0/…`, `g1/…`. Action contract stays `(N,) int32`.

Plus **`metrics.py`** (`StepCtx` — `derive(prev, world, requires, topology)`; pairwise dist, adjacency,
reach, giant component, coverage, redundancy, dist-to-frontier…), **`rollout.py`**
(`rollout(env, policy, n_steps, key, keep=, collect=)`, `random_policy`), **`viz/`** (`render`, `report`).

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

## Agent design (the constraints that bound the research space)

In kymera an agent is three parts; only the brain is missing — **and two hard stances define what agents
you can build**:

- **Senses (input) — built.** A policy is `(obs, key) -> (N,) int32`; `obs` is composed from named
  channels (above), with a separate CTDE central view. New perception = a registered channel fn.
- **Body (output) — built but frozen.** Actions are a fixed `IntEnum` `STAY/UP/DOWN/LEFT/RIGHT`
  (`N_ACTIONS = 5`), **movement only, forever**. `dynamics` turns 5-way logits into collision-free
  actions. ⇒ Goal/hierarchical agents are expressible only as **policy-internal** hierarchy (a high-level
  goal selector feeding a low-level move head); there are **no env-level macro-actions or goal-actions**.
- **Brain (policy network) — NOT built.** kymera has no actor/critic/encoder/attention — only
  `random_policy`. Learned architectures belong in `kymera.lab.nets` (unbuilt); the reference nets live
  in `../zymera_env/examples/lib/` and `../zymera_env/swarm_explore/{policy,gcrn}.py`.
- **Communication is env-mediated; policies never emit messages.** No learned messaging / emergent
  protocols by design — comm is a channel gossiping env-owned belief maps. The flip side (good for the
  threat model): *attacks on* comms (message manipulation), obs spoofing, and action injection are
  modeled env-side over the `group` machinery, not as policy outputs.

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
- Top-level `__all__` ≤ ~13; components live in subnamespaces (a design rule, not an accident).
- Reset/step key-split order is frozen; connectivity terms read potential topology unless opting into
  delivered (both logged).

## Migration status & where to read more

- **The gate for new experiments is `kymera.lab`** (design "step 7": `config`/`runio` → `ppo` →
  `nets` → `eval`), accepted via a **shadow run** reproducing a known v0 result within seed noise. The
  `sensor.py` + a rewritten `swarm_explore/core.py` separately unlock the belief/relay re-dos.
- **Authoritative docs:** `docs/specs/2026-06-11-kymera-design.md` (full architecture, doctrine,
  parity strategy, migration steps 0–9) and `docs/plans/2026-06-12-kymera-implementation.md`. Also
  `README.md` and `docs/tutorial-env.html`.
- Per kymera's **dual-maintenance rule**: once lab's shadow run passes, `zymera_env/` freezes and all new
  work lands here. Until then, treat this as the forward target, not yet the active stack.
