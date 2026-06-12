# Kymera — design spec

**Date:** 2026-06-11
**Status:** approved direction; implementation starts as a parallel build (see §9)
**Supersedes:** the `zymera` v0 library at `../zymera_env/` (frozen, never edited; archived at cutover)

Kymera (after *chimera* — a creature composed of different parts) is a JAX-native
multi-agent grid **simulator library** for cooperative, communication-constrained,
and eventually adversarial swarm research. It is the successor to `zymera`,
rebuilt around composition so that a new research idea is a new *component*
written in an experiment script — never an edit to the simulator.

---

## 1. Principles

1. **It's a simulator; it should make our life easier.** One line builds a
   working env; full composition is available when needed; everything
   serializes; observability (per-term rewards, canonical metrics, drawable
   state) is built in, not bolted on.
2. **Experiments import the library and stay downstream.** The library ships
   the env *and* the reusable lab machinery (trainer, eval, run provenance).
   Experiment scripts are ~20 lines, live in a non-packaged `experiments/`
   folder in this repo, and define new ideas locally (reward terms, obs
   channels, nets). Proven ideas **graduate** into the library; failed ones die
   in the experiment file. This is the antidote to the `examples/` sprawl.
3. **JAX purity is non-negotiable.** Everything jit/vmap/`lax.scan`-safe; fixed
   shapes; no Python branching on traced values; trace-time gating on static
   config. Components are frozen, hashable, closure-captured static objects —
   the exact pattern the v0 trainers already prove works.
4. **Env-mediated communication only.** Actions are `(N,) int32` movement,
   forever. Communication is the env transporting env-owned knowledge over a
   topology with delivery effects (delay, dropout, bandwidth, jamming later).
   Policies never emit messages.
5. **Inherited doctrine** (from RedWithinBlue + the zymera journal):
   eval = sample π (never argmax); multi-seed via vmap, never a Python loop;
   fixed-length scan episodes; `lax.scan`-over-agents for sequential conflict
   resolution.

## 2. Decision log (user-confirmed)

| Decision | Choice |
|---|---|
| Name | `kymera` (PyPI free as of 2026-06-11; no macOS case-collision with the `Zymera/` vault) |
| Comm model | env-mediated only; action contract `(N,) int32` permanently |
| Library boundary | **one installable library, two layers**: `kymera` (simulator) + `kymera.lab` (trainer/eval/provenance); experiments downstream |
| Migration | **parallel build** in this sibling folder; `zymera_env/` is never touched; cutover after parity + shadow run |
| Roadmap extensions | new mission types · richer worlds/dynamics (maps, rooms, energy) · adversarial k-of-N red agents. Per-agent heterogeneous physics explicitly *not* prioritized |
| Architecture | component composition (panel winner, 3/3 judges) + grafts: `Assignment`/`RandomKofN`, eval lie-proofing, unweighted per-term logging, step-stamp bandwidth, traced-knobs-later, mission `annotations()` viz hook |

## 3. Library architecture

### 3.0 The static-object rule (JAX contract)

Every component is a `dataclasses.dataclass(frozen=True)` holding only Python
ints/floats/strs/tuples/components (hashable), with pure-JAX methods. Canonical
usage is **closure capture**: jitted code closes over `env`; components are
trace-time constants. Components that conceptually hold arrays (`MapFile`,
fixed waypoints) store nested tuples and materialize arrays inside methods.
Protocols are `typing.Protocol` (duck typing), never runtime-dispatched ABCs.

### 3.1 Components

```
kymera/
  env.py        # Env base, GridEnv orchestrator, registry, recipes, ActionId, ACTION_DELTAS
  worldgen.py   # Terrain + Spawn
  dynamics.py   # movement, collision, action masks, masked sampling
  comms.py      # Topology + channels
  obs.py        # ObsBuilder + named channel registry
  missions.py   # Mission, RewardTerm, GroupedMission, Assignment (+ missions/terms.py as it grows)
  metrics.py    # canonical metric functions + StepCtx
  rollout.py    # scan rollout + batch/seed helpers + field filtering
  sensor.py     # host-side numpy occlusion sensor (analysis tool, unchanged from v0)
  viz/          # render_gif, comm overlay, iso, report, teleop, annotation renderer
  lab/          # §5 — trainer, eval, runio, configs
```

**WorldGen** — un-bakes `reset`:

```python
class Terrain(Protocol):
    def walls(self, key, h: int, w: int) -> Array:            # (H,W) bool
# OpenTerrain() · RandomWalls(n_obstacles) · MapFile.load(path) · Rooms(rooms, door_w=1)

class Spawn(Protocol):
    def positions(self, key, wall, n_agents: int) -> Array:   # (N,2) i32, distinct, free
# ScatterSpawn() · ClusterSpawn(radius)  [v0 _cluster_spawn verbatim] · FixedSpawn(cells)
```

**Dynamics** — kills the three-place collision split:

```python
class CollisionRule(Protocol):
    def resolve(self, old_pos, proposed) -> tuple[Array, Array]:  # (new_pos (N,2), blocked (N,) bool)
# NoCollision() · SequentialClaim()   [lax.scan over agents; doctrine]

@dataclass(frozen=True)
class GridDynamics:
    collision: CollisionRule = NoCollision()
    def targets(self, world) -> Array       # (N,A,2) clipped+wall-reverted — THE single source of truth
    def action_mask(self, world) -> Array   # (N,A) bool physical validity; STAY always True
    def step(self, world, action) -> tuple[Body, Array]   # body', blocked

def sequential_masked_sample(logits, targets, init_pos, key) -> (actions, masked_logp)
def sequential_masked_logp_ent(logits, targets, init_pos, actions) -> (logp, entropy)
```

The trainer's collision-free sampler consumes `env.dynamics.targets(state)` —
the v0 `_frontier_core._targets` duplicate dies. `blocked` flows into `StepCtx`
so the collision reward term keeps emitting learning signal when env-level
blocking is on (v0's `hard_collision` silently erased attempts).
**Invariant (tested):** masked sampling never proposes a move
`SequentialClaim.resolve` would revert; agent order 0..N-1 is pinned forever.

**Comms** — the gossip payload lives in the channel (this is the detail the
panel verified bit-for-bit against `comm_coverage.py:297-299`: delivery is each
neighbour's *previous cumulative shared map* over *new-position* adjacency —
that's what makes information flood multi-hop):

```python
class Topology(Protocol):
    def adjacency(self, world) -> Array      # (N,N) bool, symmetric, diag True
# DiskTopology(radius, metric="chebyshev")

@chex.dataclass(frozen=True)
class ChannelState:
    shared: Array                            # (N,H,W) bool — post-gossip belief
    buffer: Array                            # (delay,N,H,W) bool ring buffer

@dataclass(frozen=True)
class GossipChannel:
    topology: Topology
    delay: int = 1                           # delay=1, dropout=0 ≡ v0 gossip exactly
    dropout: float = 0.0                     # per-edge Bernoulli per step (off-diagonal only)
    bandwidth: int | None = None             # top-k most-recent cells; needs i32 step-stamped
                                             # explored maps (incl. relayed cells) + lax.top_k, static k
    def init(self, world, outbox0) -> ChannelState
    def deliver(self, world, outbox, st, key) -> (incoming, ChannelState, delivered_adj)
# NullChannel()  — adjacency = eye; for non-comm missions
```

`World.comm_graph` = **delivered** edges (wired, no longer a placeholder).
v1 scope: payloads are OR-reducible bool grids only — say so in the docstring;
a payload type zoo is the fork-drift failure mode this design exists to prevent.

**Rule, decided now, before the dropout era:** reward terms and connectivity
metrics read **potential topology** adjacency unless a term explicitly opts
into delivered edges; `StepCtx` carries both and both are logged. This prevents
a second re-baseline trauma when comms become stochastic.

**ObsBuilder** — named composable channels:

```python
class ObsBuilder(Protocol):
    obs_spec: tuple[int, ...]; central_spec: tuple[int, ...] | None
    requires: frozenset[str]
    def agent_obs(self, world, ctx) -> Array      # (N, *obs_spec)
    def central_obs(self, world, ctx) -> Array    # CTDE critic view
# VectorObs() [v0 (N,3)] · GridObs(channels=(...), sense_r=1, central=(...))
CHANNEL_FNS: dict[str, Callable]  # "known","own_pos","known_walls","neighbors","local_frontier",
                                  # "team_explored","all_pos","walls","energy"… custom fns register here
```

`env.central_obs(state)` is a real method — the free-function `global_state` dies.

**metrics.py + StepCtx** — compute once, read everywhere (replaces the blocks
triplicated across v0's env reward, two trainer cores, and `graded_eval`):

```python
pairwise_dist · adjacency · reach · connected · giant_component · collisions
coverage_fraction · redundancy · dist_to_frontier ...

@chex.dataclass(frozen=True)
class StepCtx:        # fields are None at Python time unless requested; structure is a
    ...               # function of env config ONLY (never of data) — vmap/scan-safe
def derive(prev, world, requires: frozenset[str], topology) -> StepCtx
```

Python-time `if "reach" in requires:` gating — un-requested machinery (e.g. the
CBF λ₂ eigendecomposition) never compiles. `requires` = union of obs/mission/term
declarations, fixed at env `__init__`.

**Missions** — reward as data; per-group objectives:

```python
TermFn = Callable[[prev_world, world, action, ctx], Array]      # -> (N,) UNWEIGHTED

@dataclass(frozen=True)
class RewardTerm:
    name: str; weight: float; fn: TermFn; requires: frozenset[str] = frozenset()

class MissionProtocol(Protocol):
    terms: tuple[RewardTerm, ...]; requires: frozenset[str]
    def init_state(self, key, world) -> Any                     # FIXED pytree structure
    def update(self, prev, world, ctx, mstate, key) -> Any      # scripted NPCs (VIP, intruder)
    def done(self, world, ctx, mstate) -> Array                 # (N,) bool
    def reward(self, prev, world, action, ctx, mstate) -> (Array, dict[str, Array])
        # (Σ w_i·term_i  (N,),  {name: UNWEIGHTED (N,)})  → info["reward_terms"]
    def metrics(self, world, ctx, mstate) -> dict[str, Array]   # mission-owned success metrics
    def annotations(self, world, mstate) -> tuple[Annotation, ...]   # §4 — drawable primitives
```

`missions/terms.py` ships the proven zoo: `new_coverage`, `reach_fraction`,
`capped_giant(cap)`, `collision_count`, `same_step_overlap`,
`cohesion_leash(leash, comm_r)`, `degree_floor(floor, comm_r)`,
`pbrs(phi, gamma)` (+ `phi_nearest_frontier`, `phi_field_mean`),
`cbf_conn(...)`, `cbf_coll(...)`. PBRS via the combinator → policy-invariant by
construction. **Unweighted** logging lets analysis re-weight post-hoc without
re-running.

**Groups (adversarial roadmap)** — grafted `Assignment` so k-of-N membership
can randomize per reset without retracing:

```python
class Assignment(Protocol):
    def assign(self, key, n_agents) -> Array     # (N,) i32 group ids
# FixedAssignment(groups) · RandomKofN(k, group=1)

@dataclass(frozen=True)
class GroupedMission:
    assignment: Assignment
    missions: tuple[MissionProtocol, ...]        # index = group id
    # reward routed by where(group_id == g, r_g, 0); each group's terms computed
    # over all N (fixed shape) then masked. Metrics namespaced "g0/...", "g1/...".
    # Action contract unchanged: every agent emits an int.
```

**Mandates (from the judge panel):** (a) implement one cheap second mission
(chase *or* patrol) as the protocol's falsification test before building the
mission library in earnest; (b) test a 2-group case before any red training
exists — coverage-counting-red-visits and union-graph connectivity are
silently-wrong-by-default and must be decided per group explicitly.

### 3.2 World v2 and the step pipeline

```python
@chex.dataclass(frozen=True)
class Body:   position: Array               # (N,2) i32; energy: (N,) f32 (zeros until used)
@chex.dataclass(frozen=True)
class World:
    body: Body
    explored: Array                          # (H,W) i32 visit counts; .visited = explored > 0
    seen_by: Array                           # (N,H,W) bool — per-agent OWN knowledge
    wall: Array                              # (H,W) bool
    comm_graph: Array                        # (N,N) bool — DELIVERED edges (wired)
    step_count: Array
    channel: Any                             # channel-owned pytree (() for NullChannel)
    mission: Any                             # mission-owned pytree (() default)
    group: Array                             # (N,) i32 — assigned at reset
```

`step(state, action, key)` orchestration order:

```
1 dynamics.step      → body', blocked
2 visit counts       → explored'
3 own knowledge      → seen_by' = seen_by | footprint(body', cover_r) & ~wall
4 channel.deliver    → incoming, channel', delivered  (comm_graph := delivered)
5 metrics.derive     → ctx   (once; Python-time requires-gating)
6 mission.update     → mstate'  (NPCs, waypoints; gets its own key)
7 mission.reward     → reward (N,), unweighted terms dict
8 mission.done       → done (N,)
9 obs.agent_obs      → obs
info = {explored, step_count, seen_by, comm_graph, reward_terms, metrics}   # fixed keysets
```

`cover_r` (coverage footprint) stays core grid physics, not a component.
**The reset/step key-split order is a frozen, documented contract** — parity
and reproducibility ride on it.

### 3.3 Construction ergonomics

```python
import kymera as ky
env = ky.make("comm-coverage", grid=16, n_agents=4)            # recipe, good defaults
env = ky.make("comm-coverage", grid=16, n_agents=4,            # same recipe, overridden
              comm=ky.comms.GossipChannel(ky.comms.DiskTopology(5), dropout=0.1),
              collision="block",        # env-level: "none" | "block" (SequentialClaim);
                                        # collision-free SAMPLING is the trainer's PPOCfg.mask_collisions
              terms=[("coverage", 1.0), ("capped_giant", 1.5, dict(cap=3))])
env2 = env.replace(terms=[...])                                # cheap variation
spec = env.spec()                                              # → plain dict (YAML/JSON-able)
env3 = ky.make_from(spec)                                      # round-trip
print(env)                                                     # full composition, human-readable
```

Term shorthand: tuples `(name, weight[, params])` resolve against
`missions/terms.py`; full `RewardTerm(...)` objects (including locally-defined
fns) pass through unchanged. Constructors validate composition at Python time
(term `requires` available, mission-state structure stable, shape coherence)
with human error messages — fail before the trace, not inside it.

Public API: ~13 top-level names (`make`, `make_from`, `list_envs`,
`register_env`, `Env`, `GridEnv`, `World`, `Body`, `ActionId`, `N_ACTIONS`,
`RewardTerm`, `rollout`, `random_policy`) + subnamespaces
(`worldgen`, `dynamics`, `comms`, `obs`, `missions`, `metrics`, `viz`, `lab`).
The ~30 component names stay behind the subnamespaces; top-level `__all__`
holds at ~13. That cap is a design rule, not an accident.

## 4. Viz data contract

Viz must never depend on what a *training* rollout happened to store.

**Per-frame sources** (all in `World` / `info`):

| Viz product | Source |
|---|---|
| flat / iso gif | `body.position`, `wall`, `explored` |
| fog & belief views | `seen_by` (own) + `channel.shared` (post-gossip) → belief-vs-ground-truth rendering |
| comm overlay | `comm_graph` (delivered, solid) + `StepCtx` potential adjacency (thin) — honest under dropout |
| report curves | `info["reward_terms"]` (unweighted per-term) + `info["metrics"]` |
| red/blue coloring | `World.group` |
| collision debugging | `blocked` (N,) — flash reverted moves |
| mission overlays | `mission.annotations(world, mstate)` |

**Annotations** — missions stay drawable without viz knowing their internals:

```python
Annotation = Point(pos, tag) | Path(cells, tag) | Region(mask, tag)
```

`kymera.viz` renders annotation primitives generically (VIP marker, patrol
route, intruder, jammed region). New mission ⇒ automatically visualizable.

**Trajectory policy:**
- Training rollouts are lean by default: `rollout(..., keep=...)` drops the
  `(delay,N,H,W)` channel buffer and belief maps (HBM-load-bearing for
  multi-seed runs). `info` stacking is opt-in via `collect=("reward_terms",)`.
- **Re-simulation is the viz doctrine:** the sim is deterministic given
  (env spec, checkpoint, key), and all three live in run provenance (§5). So
  `viz.make_report(run_id)` re-rolls an episode with `keep="all"` at render
  time. Training trajectories can be deleted; every run stays renderable.
- `viz` keeps accepting both stacked pytrees (scan) and Python lists of worlds
  (teleop), as in v0.

## 5. `kymera.lab` — the machinery every experiment needs

```
kymera/lab/
  config.py    # msgspec.Struct: EnvCfg/PPOCfg/StopCfg/PhaseCfg/RunCfg; load/dump/override
  ppo.py       # the ONE trainer: fit(env, model, cfg, *, init_from, run)
  nets.py      # graduated reference models (Encoder, FrontierCommAttnAC, …) + registry
  eval.py      # evaluate(env, policy, n=32, masked=...) → EvalReport
  runio.py     # lab.run(...) context manager, run ids, index.jsonl, git fingerprint
```

- **One trainer, flags not forks.** `PPOCfg(value_norm=…, mask_collisions=…,
  value_clip=…, adv_norm=…, lr_anneal=…)` are trace-time branches in one
  `make_train`; `iters` is a config field (the `core.ITERS =` mutation pattern
  is dead). Masking consumes `env.dynamics.targets`. Curricula chain via
  `init_from` — warm-start is an argument, not a script.
- **Multi-seed via vmap** (doctrine): chunked `jit(vmap(scan))`,
  best-checkpoint tracked **inside the carry** per seed; `metrics.npz` gains a
  leading seed axis. Known semantic change: aggregate early-stop ≠ v0's
  per-run stop that protected late bloomers — record `core_version` in the
  index so eras stay comparable.
- **Metrics from trajectory shapes and env attributes only** — the
  `cc.GRID`/`cc.N_AGENTS` class of wrong-metrics bug is structurally
  impossible.
- **`lab.evaluate` bakes the doctrine once**: sample π, masked-iff-trained-
  masked, all metrics via `kymera.metrics`, plus lie-proofing asserts —
  deserialized params ≠ fresh init, and the checkpoint's recorded env-spec
  hash matches the eval env.
- **Provenance as a context manager:**

```python
with lab.run("curriculum-16x16", cfg) as run:    # creates experiments/runs/<id>/,
    ...                                           # dumps spec+config, git sha (+ patch.diff
                                                  # if dirty), appends experiments/index.jsonl
```

  `run_id = "20260611-curriculum-16x16-007"` (+ config digest). The index is
  append-only and never pruned; each line carries the full inline config, git
  fingerprint, per-seed bests, status, and a `pruned` flag once artifacts are
  deleted. The journal cites run ids; citations survive artifact deletion.
- Config objects never cross the jit boundary — `build_env`/`make_train`
  unpack them into static components at Python time.

## 6. Experiments model

Experiments live in `experiments/` in this repo — **not packaged**, imported
nothing-from; they import kymera. One git history pins experiment code and
library state together (the index's git sha resolves both).

```python
# experiments/curriculum_16.py  (~the whole file)
import kymera as ky
from kymera import lab

def frontier_pull(prev, world, action, ctx):           # idea born here;
    return -ctx.dist_to_frontier                        # graduates if it wins

env1 = ky.make("comm-coverage", grid=16, n_agents=4,
               terms=[("coverage", 1.0), ("pbrs_frontier", 1.0),
                      ky.RewardTerm("pull", 0.5, frontier_pull,
                                    requires=frozenset({"dist_to_frontier"}))])
env2 = env1.replace(terms=[("coverage", 1.0), ("capped_giant", 1.5, dict(cap=3))])

with lab.run("curriculum-16x16") as run:
    m1 = lab.ppo.fit(env1, model="frontier-comm-attn", cfg=lab.PPOCfg(iters=120), run=run)
    m2 = lab.ppo.fit(env2, init_from=m1, cfg=lab.PPOCfg(iters=100, mask_collisions=True), run=run)
    print(lab.evaluate(env2, m2, n=32))
```

**Graduation rule:** reward terms, obs channels, nets, missions are defined
locally in experiment files; once a thing proves out across >1 experiment, it
moves into the library with a test. Once superseded, the experiment file is
deleted — the index keeps the record.

## 7. comm-coverage re-expression (the parity target)

```python
def comm_coverage(grid=16, n_agents=4, comm_r=5, cover_r=0, sense_r=1,
                  n_obstacles=0, spawn_radius=2, collision="none",
                  delay=1, dropout=0.0, bandwidth=None, terms=DEFAULT_TERMS):
    return GridEnv(
        grid_h=grid, grid_w=grid, n_agents=n_agents, cover_r=cover_r,
        terrain=RandomWalls(n_obstacles) if n_obstacles else OpenTerrain(),
        spawn=ClusterSpawn(spawn_radius) if spawn_radius is not None else ScatterSpawn(),
        dynamics=GridDynamics(collision=_collision(collision)),
        channel=GossipChannel(DiskTopology(comm_r), delay, dropout, bandwidth),
        obs=GridObs(("known","own_pos","known_walls","neighbors","local_frontier"),
                    sense_r=sense_r, central=("team_explored","all_pos","walls")),
        mission=Mission(terms=_resolve(terms)))
ky.register_env("comm-coverage", comm_coverage)
```

State mapping from v0 `CommState`: `pos → body.position`,
`explored_by → seen_by`, `shared → channel.shared`,
`team_explored →` derived `visited`, `wall`/`step_count` same.

## 8. Doctrine (rules the code must enforce, with tests)

1. Eval = sample π, ε-free; never argmax.
2. Multi-seed via vmap; fixed-length scan episodes; scan-over-agents conflicts.
3. **Term SET is static; weights may become traced later; never gate on
   `w == 0`** (the v0 λ₂ gate is on weight values — this trap goes live the
   moment a weight is traced).
4. Connectivity terms/metrics read potential topology unless explicitly
   opting into delivered; both logged.
5. Reset/step key-split order is a frozen contract.
6. Mission `init_state`/`update` pytree structures must be identical —
   structure-equality test runs for every registered env.
7. Tracer regression test: jit-trace step/reset with abstract values to catch
   Python-branch-on-traced-value regressions structurally.
8. Top-level `__all__` ≤ ~13 names; components live in subnamespaces.

## 9. Migration: parallel build, zero edits to `zymera_env/`

The new name dissolves the package collision: `pip install -e .` (kymera) and
`pip install -e ../zymera_env` (zymera v0) coexist in **kymera's own venv** —
live A/B parity tests import both. `zymera_env/`'s venv and campaign are
untouched; optionally `git init` it once (zero code change) to snapshot.

**Parity strategy:**
- **Env: bit-parity.** Same PRNGKey → identical walls/spawns; same action
  sequence → identical positions, beliefs, per-step rewards (fp tolerance on
  sums). `GossipChannel(delay=1, dropout=0)` reproduces
  `(adj & prev_shared).any(1)` exactly. Golden trajectories also dumped to
  `tests/golden/` as insurance.
- **Trainer: statistical parity.** Chunked-vmap RNG differs by construction;
  gate = a **shadow run** reproducing a known v0 result (10×10 curriculum,
  ~71% cov / 0 collisions) within seed noise.

**Build order** (each step lands with tests, repo stays green):

| Step | What | Gate |
|---|---|---|
| 0 | scaffold, pyproject (`kymera*` only packaged), this doc = commit zero | — |
| 1 | `metrics.py` | values match v0's triplicated blocks |
| 2 | `dynamics` + masked sampling | claim-order property test |
| 3 | `comms` | gossip bit-parity vs live v0 |
| 4 | `worldgen` + `obs` | spawn/obs bit-parity (mirror v0 key order) |
| 5 | `missions` + terms + `GridEnv` + recipe | **full env parity gate** |
| 6 | `viz` port (+ annotations) | renders a re-simulated v0-config episode |
| 7 | `lab`: runio/config first, then trainer, nets ported verbatim | old `.eqx` checkpoints load |
| 8 | **shadow run** | curriculum result within seed noise |
| 9 | cutover: new experiments here; `zymera_env/` archived read-only; journal copied forward with a "kymera era" marker | — |

**Dual-maintenance rule:** the day the shadow run passes, the old stack is
frozen — in-flight experiments finish there, every new idea lands here only.

## 10. Risks

- **HBM blow-up via scan-stacked state** — the `(delay,N,H,W)` ring buffer over
  T×envs×seeds. Mitigation: lean rollout default + re-simulation doctrine (§4).
- **Component/jit friction** — env objects must stay trace-time constants;
  curriculum phases rebuilding envs = one retrace per phase (fine); a per-iter
  env rebuild would be a compile storm (watch for it).
- **Parity-gate fragility** — trajectory equality rides on key-split and op
  order (cluster-spawn `top_k` tie-breaks on uniform noise); budget time, use
  tolerance on fp reward sums, never chase bitwise float equality.
- **GossipChannel payload creep** — keep v1 to bool grids; a payload zoo forks
  the channel, recreating the drift this design kills.
- **Early-stop semantics under multi-seed vmap** — aggregate stop can cut a
  late-blooming seed (the w_conn=5 precedent); index records `core_version`.
- **Dual-maintenance window** — the freeze rule (§9) is what keeps the port
  honest; if old-stack feature work continues past the shadow run, kymera rots
  half-done.
- **API creep** — resist re-exporting components at top level; the 13-name cap
  is the line.

## 11. Open questions (deliberately deferred)

1. **Traced knobs** (functional-core graft): promote value-only knobs (reward
   weights, dropout, radii as broadcast-compares) into a small traced pytree —
   *eval first* — for one-compile dose-response/robustness sweeps. Additive;
   decide after cutover.
2. **Energy model** (Body.energy is reserved): drain/recharge rules, an energy
   mission term, an obs channel — design when the roadmap reaches it.
3. **Jamming**: a region mask zeroing edges crossing it — same `deliver`
   signature; design with the first adversarial-comms experiment.
4. **Report tooling**: how much of `_assemble_report*.py` graduates into
   `viz.make_report` vs stays experiment-local.
