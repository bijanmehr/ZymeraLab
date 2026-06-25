# Zymera Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the zymera simulator library + lab layer per the approved spec (`docs/specs/2026-06-11-zymera-design.md`), with bit-parity against zymera v0, and ship webpage-style docs with an env tutorial.

**Architecture:** Component-composition simulator (`zymera`: env/worldgen/dynamics/comms/obs/missions/metrics/rollout/viz) + `zymera.lab` (config/runio/eval/nets/ppo). Parallel build — `../zymera_env/` is **never modified**; parity via golden files dumped read-only from v0.

**Tech Stack:** JAX, chex (core); equinox/optax/msgspec/pyyaml (lab extra); matplotlib/pillow (viz extra); pytest.

---

## Frozen contracts (every task conforms to these)

### World schema (`zymera/env.py`)

```python
@chex.dataclass(frozen=True)
class Body:
    position: chex.Array          # (N,2) int32
    energy:   chex.Array          # (N,)  float32 — zeros until the energy roadmap lands

@chex.dataclass(frozen=True)
class World:
    body:       Body
    explored:   chex.Array        # (H,W) int32 visit counts
    seen_by:    chex.Array        # (N,H,W) bool — per-agent OWN sensed/covered cells
    wall:       chex.Array        # (H,W) bool
    comm_graph: chex.Array        # (N,N) bool — DELIVERED edges this step
    step_count: chex.Array        # () int32
    channel:    Any               # channel-owned pytree; () for NullChannel
    mission:    Any               # mission-owned pytree; () default
    group:      chex.Array        # (N,) int32 — assigned at reset
    # properties: grid_h, grid_w, n_agents, visited (explored>0), covered (seen_by.any(0))
```

**Coverage doctrine:** the team-coverage metric and `newly_covered` read
`World.covered == seen_by.any(0)` (union of cover footprints). With
`cover_r=0` this equals `visited`. `explored` visit counts exist for heatmaps.

### Key protocol (bit-parity with v0; FROZEN)

```python
# GridEnv.reset(key):
wkey, skey = jax.random.split(key)            # exactly as v0 comm_coverage.reset
wall = terrain.walls(wkey, H, W)
pos  = spawn.positions(skey, wall, N)         # ClusterSpawn splits skey -> (akey, ckey) internally, as v0
group = assignment.assign(jax.random.fold_in(key, 1), N)   # default FixedAssignment uses no key
mstate = mission.init_state(jax.random.fold_in(key, 2), world)  # coverage mission uses no key
# GridEnv.step(state, action, key):
k_chan, k_mis = jax.random.split(key)
# rollout(env, policy, n_steps, key): identical to v0 zymera.rollout —
reset_key, scan_key = jax.random.split(key); per step: k, action_key, step_key = jax.random.split(k, 3)
```

### Step pipeline order (spec §3.2)

dynamics.step → explored counts → seen_by |= footprint(cover_r) → channel.deliver
(outbox = seen_by', comm_graph := delivered) → metrics.derive(ctx) →
mission.update → mission.reward (Σ weighted; unweighted dict → info) →
mission.done → obs.agent_obs → info{explored, step_count, seen_by, comm_graph,
reward_terms, metrics}.

### Gossip parity (the load-bearing equivalence)

v0 (`comm_coverage.py:297-299`): `adj = adjacency(new_pos, comm_r)` (diag True);
`received = (adj[:,:,None,None] & prev_shared[None]).any(1)`; `shared = own | received`.
Zymera `GossipChannel(delay=1, dropout=0).deliver`: ring buffer holds previous
`shared`; `delivered_payload = (adj & buffer.head).any(1)`; `shared' = outbox | delivered_payload`;
push `shared'`. Diag-True self-loop ⇒ monotone. **Must match bit-for-bit.**

---

### Task 0: Scaffold + venv

**Files:** Create `pyproject.toml`, `zymera/__init__.py` (stub), `README.md` (stub), `.gitignore`, `tests/conftest.py`.

- [ ] `pyproject.toml`: name `zymera` 0.1.0, `requires-python>=3.10`; core deps `jax>=0.4, jaxlib>=0.4, chex>=0.1, numpy`; extras `viz=[matplotlib,pillow]`, `lab=[equinox>=0.11,optax>=0.1,msgspec>=0.18,pyyaml>=6.0]`, `dev=[pytest>=8.0]`; `packages.find include=["zymera*"]`.
- [ ] `.gitignore`: `.venv/`, `__pycache__/`, `*.egg-info/`, `experiments/runs/`, `*.gif`, `*.png` (keep `experiments/index.jsonl`, `tests/golden/*.npz`, `docs/`).
- [ ] `python3 -m venv .venv && .venv/bin/pip install -e ".[viz,lab,dev]"` — run in background.
- [ ] `tests/conftest.py`: force matplotlib Agg; fixtures `key` (PRNGKey(0)), `small_world`.
- [ ] Commit: `chore: scaffold package, venv, test harness`.

### Task 1: Golden reference data from v0 (zero-touch)

**Files:** Create `/tmp/dump_golden.py` (NOT in either repo's tree), output `tests/golden/{commcov_default.npz, commcov_traj.npz, empty_env.npz, cluster_spawn.npz}`.

- [ ] Script imports v0 via `sys.path` pointing at `../zymera_env` + `examples/`; uses `zymera_env/.venv/bin/python` (read-only execution; campaign venv unmodified).
- [ ] Dump, for keys PRNGKey(0/1/2): comm-coverage default cfg (16×16, N=4, comm_r=5, vis_r=0, sense_r=1, spawn_radius=2, weights 1/2/4) — wall, spawn pos, and a 70-step `zymera.rollout(env, random_policy, ...)` trajectory: positions (T+1,N,2), shared (T+1,N,H,W), explored_by, obs (T+1,N,5,H,W), reward (T,N), action (T,N). Also `empty-v0` 8×8 N=4 spawn+positions, and a standalone `_cluster_spawn` cell set.
- [ ] Commit goldens: `test: golden reference trajectories from zymera v0`.

### Task 2: Core contract — `zymera/env.py` (schema half) + `zymera/rollout.py`

**Files:** Create `zymera/env.py` (ActionId, ACTION_DELTAS, Body, World, Env base, registry, `_info_dict`), `zymera/rollout.py`, `tests/test_state.py`, `tests/test_rollout.py`.

- [ ] ActionId/deltas identical to v0 (STAY/N/E/S/W, append-only doctrine).
- [ ] `rollout(env, policy, n_steps, key, *, keep="lean", collect=())` — v0 key protocol; `keep="lean"` drops `channel`/`mission` slots from the stacked World (tree_map replace with `()`); `keep="all"` stacks everything; `collect` stacks chosen info keys.
- [ ] Tests: World pytree round-trips jit/vmap; registry duplicate-raises; rollout shapes (T+1 state side, T action side); vmap over keys.
- [ ] Run `pytest tests/test_state.py tests/test_rollout.py -q` → PASS (rollout test uses a trivial inline env). Commit.

### Task 3: `zymera/metrics.py` (+ StepCtx)

**Files:** Create `zymera/metrics.py`, `tests/test_metrics.py`.

- [ ] Functions: `pairwise_dist, adjacency, reach (log2 squaring), connected, giant_component, collisions, coverage_fraction, redundancy, dist_to_frontier, field_mean_dist` — semantics identical to v0 `cc._*` (verify vs hand-computed 3-agent cases + golden positions).
- [ ] `StepCtx` frozen chex.dataclass; fields None at Python time unless requested: `dist, adj, delivered, reach, collisions, blocked, newly_covered, overlap, dist_to_frontier, field_dist`. `derive(prev, world, requires, topology, *, delivered, blocked)` with Python-time gating; structure = f(env config) only.
- [ ] Test: ctx pytree structure identical across two different worlds (same requires); reach matches v0 closure on golden positions. Commit.

### Task 4: `zymera/worldgen.py`

**Files:** Create `zymera/worldgen.py`, `tests/test_worldgen.py`.

- [ ] `Terrain`: `OpenTerrain`, `RandomWalls(n)` (v0 `_random_wall` verbatim), `MapFile(cells)` (+`.load(path)`, `.from_string(...)` with `#`=wall), `Rooms(rooms, door_w)` (deterministic partition walls + doors, key-driven door placement).
- [ ] `Spawn`: `ScatterSpawn` (v0 weighted-choice), `ClusterSpawn(radius)` (v0 `_cluster_spawn` verbatim incl. internal `split(skey)` and `top_k` overflow), `FixedSpawn(cells)`.
- [ ] **Parity gate:** walls(PRNGKey golden) == golden wall; ClusterSpawn cells == golden spawn cells; ScatterSpawn == empty-env golden. Spawns distinct & free for all terrains. Commit.

### Task 5: `zymera/dynamics.py`

**Files:** Create `zymera/dynamics.py`, `tests/test_dynamics.py`.

- [ ] `GridDynamics(collision).targets(world) -> (N,A,2)` (clip + wall-revert, = v0 `_targets`); `action_mask(world) -> (N,A)`; `step(world, action) -> (Body', blocked)`.
- [ ] `NoCollision` (blocked = all-False); `SequentialClaim` (v0 `_resolve_collisions` scan + `blocked` output where reverted).
- [ ] `sequential_masked_sample(logits, targets, init_pos, key)`, `sequential_masked_logp_ent(...)` — port v0 `_frontier_core` versions onto `targets`.
- [ ] Tests: movement parity vs golden positions given golden actions (NoCollision path); property test over 200 random worlds: masked-sampled actions stepped through `SequentialClaim.resolve` produce `blocked.all() == False`; STAY always valid; agent order 0..N-1 documented + asserted. Commit.

### Task 6: `zymera/comms.py`

**Files:** Create `zymera/comms.py`, `tests/test_comms.py`.

- [ ] `Topology` protocol; `DiskTopology(radius, metric="chebyshev")` (diag True, symmetric).
- [ ] `ChannelState{shared (N,H,W) bool, buffer (delay,N,H,W) bool, stamps Optional}`; `GossipChannel(topology, delay=1, dropout=0.0, bandwidth=None)` with `.init(world, outbox0)` / `.deliver(world, outbox, st, key) -> (incoming, st', delivered_adj)`; dropout = per-edge symmetric Bernoulli, **off-diagonal only**; bandwidth = int32 step-stamp maps (stamps update on receipt too) + `lax.top_k` static k.
- [ ] `NullChannel` (adjacency = eye; incoming = outbox).
- [ ] **Parity gate:** replay golden trajectory positions through GossipChannel(delay=1, dropout=0); `shared` matches golden `shared` at every step bit-for-bit. delay=2 sanity (3-agent relay chain arrives one step later); dropout=1.0 ⇒ delivered = eye only; bandwidth fixed-shape under jit. Commit.

### Task 7: `zymera/obs.py`

**Files:** Create `zymera/obs.py`, `tests/test_obs.py`.

- [ ] `CHANNEL_FNS` registry: `known` (channel.shared), `own_pos`, `known_walls`, `neighbors` (potential adj offdiag), `local_frontier` (sense_r window of ~shared), `team_explored` (covered), `all_pos`, `walls`; custom fns registrable.
- [ ] `VectorObs` (v0 (N,3): pos + coverage fraction); `GridObs(channels, sense_r=1, central=(...))` with `agent_obs`/`central_obs`, `obs_spec`/`central_spec`/`requires`.
- [ ] **Parity gate:** GridObs on golden worlds == golden obs (N,5,H,W). Commit.

### Task 8: `zymera/missions.py` + `zymera/missions_terms.py`

**Files:** Create `zymera/missions.py`, `zymera/missions_terms.py` (flat module; spec allows growth into a package later), `tests/test_missions.py`.

- [ ] `RewardTerm(name, weight, fn, requires)`; `Mission(terms, max_steps=None)` with `init_state/update/done/reward/metrics/annotations` (defaults: `()` state, identity update, timeout-only done, sum-of-terms reward returning unweighted dict).
- [ ] Terms (all `(prev, world, action, ctx) -> (N,) UNWEIGHTED`): `new_coverage` (ctx.newly_covered counts), `reach_fraction`, `capped_giant(cap)`, `collision_count`, `same_step_overlap`, `cohesion_leash(leash, comm_r)`, `degree_floor(floor, comm_r)`, `pbrs(phi, gamma)` + `phi_nearest_frontier`/`phi_field_mean`, `cbf_conn(alpha, eps, sharp, comm_r)`, `cbf_coll(alpha, dmin)` (port v0 `examples/cbf.py` math).
- [ ] `Annotation` primitives: `Point(pos, tag)`, `Path(cells, tag)`, `Region(mask, tag)` (plain frozen dataclasses).
- [ ] `Assignment`: `FixedAssignment(groups)`, `RandomKofN(k, group=1)`; `GroupedMission(assignment, missions)` — where-mask routing, namespaced terms/metrics `g{i}/...`, done = own group's.
- [ ] Tests: term values vs golden reward decomposition (cov/conn/coll on golden trajectory steps); PBRS telescopes to ~γ-discounted potential difference; GroupedMission 2-group case: rewards route by mask, group metrics differ, RandomKofN assigns exactly k. Commit.

### Task 9: `GridEnv` orchestrator + recipes (env.py completed)

**Files:** Modify `zymera/env.py`; create `tests/test_env.py`, `tests/test_parity.py`.

- [ ] `GridEnv(grid_h, grid_w, n_agents, cover_r=0, terrain, spawn, dynamics, channel, obs, mission, assignment=FixedAssignment(all-0))` — step pipeline per frozen contract; Python-time validation (requires available in StepCtx, mission-state structure equality reset-vs-step, term name uniqueness) with human messages.
- [ ] Recipes: `register_env("empty", ...)` (open grid + VectorObs + no terms) and `register_env("comm-coverage", ...)` per spec §7 (term shorthand resolution `("name", w[, params])`).
- [ ] `make(name, **kw)`; `make_from(spec)`; `env.spec()` (recipe-built: `{"recipe": name, **kwargs}`); `env.replace(**kw)` (recipe-built rebuild; direct-construct raises with message); `repr(env)` prints composition.
- [ ] **FULL PARITY GATE:** `pytest tests/test_parity.py -q` — same PRNGKey + `random_policy` rollout on `make("comm-coverage")` reproduces golden walls, spawns, positions (exact), shared (exact), obs (exact), rewards (atol 1e-5). Plus: jit/vmap selftest, gossip-superset invariant, mission-structure test for every registered env (doctrine #6).
- [ ] Commit: `feat: GridEnv orchestrator + recipes; full v0 parity`.

### Task 10: `zymera/viz/`

**Files:** Create `zymera/viz/{__init__.py, render.py, iso.py, report.py, live.py, annotate.py}`, `tests/test_viz.py`.

- [ ] Port v0 `zymera/viz/*` (read from `../zymera_env/zymera/viz/`, adapt field names: `pos→body.position`, fog from `seen_by`/`channel.shared`); lazy matplotlib imports; `iter_worlds` duck-typing (stacked pytree | list).
- [ ] `render_gif` colors agents by `world.group`; comm overlay draws potential edges thin + `comm_graph` (delivered) solid.
- [ ] `annotate.py`: render `Point/Path/Region` from `mission.annotations`; report gains per-term reward curves when trajectory carries `info["reward_terms"]`.
- [ ] `make_report(traj_or_env, path)` — self-contained HTML (base64), as v0.
- [ ] Tests (Agg): draw_frame artist counts, gif bytes written, report HTML contains embedded image + term-curve section. Commit.

### Task 11: `zymera/lab/`

**Files:** Create `zymera/lab/{__init__.py, config.py, runio.py, eval.py, nets.py, ppo.py}`, `tests/test_lab.py`.

- [ ] `config.py`: msgspec Structs `EnvCfg/PPOCfg/StopCfg/PhaseCfg/RunCfg` per spec §5; `build_env(EnvCfg)->Env`; YAML/JSON load/dump; `override(cfg, "ppo.lr=1e-4")`.
- [ ] `runio.py`: `run_id = "<YYYYMMDD>-<name>-<counter>"`; `git_fingerprint()` (sha, branch, dirty + `patch.diff`); `lab.run(name, cfg=None)` context manager → `experiments/runs/<id>/`, append-only `experiments/index.jsonl` (open + close records, `pruned` flag tool); `RunDir.save_metrics/save_checkpoint/finalize`.
- [ ] `nets.py`: port `Encoder` (size-agnostic resize-to-4×4), CTDE ActorCritic, `FrontierCommAttnAC` from v0 examples (read-only; verbatim math so v0 `.eqx` checkpoints deserialize).
- [ ] `ppo.py`: ONE trainer — `make_train(env, model, cfg)` with trace-time flags `value_norm` (v0 `_frontier_core` ValueNorm), `mask_collisions` (sampler + loss via `env.dynamics.targets`), `value_clip/adv_norm/lr_anneal`; `gae`; `fit(env, model_or_name, cfg, *, init_from=None, run=None, seed=1, seeds=1, stop=StopCfg())` — seeds==1: Python loop + v0 early-stop semantics; seeds>1: `eqx.filter_vmap` chunked scan, in-carry per-seed best, aggregate stop between chunks; metrics from trajectory shapes + env attrs ONLY; per-term means logged via `collect=("reward_terms",)`.
- [ ] `eval.py`: `evaluate(env, model, *, n=32, masked, key)` → `EvalReport`; sample-π; lie-proofing (loaded-params ≠ fresh-init when template given; env-spec-hash match when loading from a run).
- [ ] Tests: config round-trip + override; runio creates dir/index, id increments, dirty fingerprint writes patch; smoke `fit` 3 iters on 6×6/N=2 env (loss finite, metrics keys), masked smoke (0 collisions in rollout); evaluate returns report, lie-proof assert fires on fresh params. Commit.

### Task 12: Webpage docs + README

**Files:** Create `docs/index.html`, `docs/tutorial-env.html`, `docs/tutorial-lab.html`, `docs/style.css`; rewrite `README.md`.

- [ ] Use the frontend-design skill; clean readable doc site (sidebar nav, code blocks, dark-friendly), self-contained (no CDN).
- [ ] `index.html`: what zymera is, install, 5-line quick start, public-API table, component map, doctrine list, links.
- [ ] `tutorial-env.html` (**the user-requested env tutorial**): make/recipes → reading World/info → rollout + vmap seeds → composing components (terrain/spawn/collision/channel) → writing a custom RewardTerm in your experiment file → groups/red-team → custom env registration → viz (gif/report/teleop, annotations).
- [ ] `tutorial-lab.html`: PPOCfg/fit, curriculum via init_from, lab.run provenance, evaluate doctrine.
- [ ] README: positioning, install, quick start, API table, status, link to docs + spec. Every code snippet in docs must be executed once (smoke) before inclusion.
- [ ] Commit: `docs: webpage docs + env/lab tutorials + README`.

### Task 13: Final verification + tag

- [ ] `pytest tests/ -q` → all green; run demo: `make("comm-coverage")` → rollout → `viz.render_gif` + `viz.make_report` artifacts render.
- [ ] Verify `import zymera` pulls no matplotlib/equinox (headless core check).
- [ ] `git tag v0.1.0`. Summary of deferred items (shadow run = trainer statistical parity on the 10×10 curriculum; traced knobs; Rooms polish) recorded in README status.

## Deferred (explicitly out of this build)

- **Shadow run** (lab trainer reproducing the v0 10×10 curriculum result within seed noise) — hours of compute; run after this lands.
- Traced-knobs eval sweeps; jamming; energy mechanics (Body.energy reserved); map editor.
