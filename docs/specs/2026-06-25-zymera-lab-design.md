# Zymera Lab — design

**Date:** 2026-06-25
**Status:** structure approved (brainstorm); implementation plan next.
**Context:** consolidates `kymera` + `zymera_env` into one codebase. `kymera` is renamed to the `zymera`
package; `zymera_env` is **archived** (tag `archive/pre-zymera-lab-fold`) as reference — *not* merged.

## Goal

`zymera_lab` is a JAX-native research toolkit for **designing different agent mechanisms and learning
stacks** on communication-constrained cooperative (and adversarial) swarm missions. It is the reusable,
tested library; **experiments are written separately**, downstream, and import it.

## Structure

```
Project.Zymera/
  zymera_lab/            the lab (renamed kymera) — reusable + tested
    zymera/              ONE package: simulator + building blocks + trainers
      env.py · worldgen.py · dynamics.py · comms.py · obs.py · missions.py ·
      missions_terms.py · metrics.py · rollout.py · viz/ · sensor.py        # simulator
      nets.py            # composable agent building blocks (encoders, belief, heads, attention, aggregators)
      train.py           # ready-to-use trainers (ppo / es / supervised) + shared rollout/eval/provenance utils
    tests/
  zymera_experiments/    separate sibling folder — import zymera, write learning stacks here
  zymera_env/            ARCHIVED (tagged) reference only — not copied, not active
```

- **No `zymera.lab` subpackage** — the repo folder already signals "lab"; `nets`/`train` are flat modules
  in the `zymera` package, next to the simulator.
- **Dependency rule (one-way):** `zymera_experiments → zymera` (sim + `nets` + `train`). The package never
  imports an experiment; the simulator modules never import `nets`/`train`.

## The agent contract

A policy is a **callable convention**, not a base class:

```
policy(obs, state, key) -> (action, state)
```

- `nets.py` provides composable parts; an experiment wires them into a `policy`.
- `state` carries per-agent recurrent/memory (e.g. a graph belief); `()` for stateless policies.
- `action` today is movement `(N,) int32`. The contract keeps **seams** for two extensions (built later,
  P3):
  - **hierarchy** — a policy emits a goal; a lab-provided low-level controller turns it into movement
    (no simulator change).
  - **learned comms** — policies emit messages; a scoped `zymera.comms` extension transports them over the
    topology (simulator change, re-baseline-gated).

## Learning

`train.py` ships **independent trainers** — `ppo`, `es`, `supervised` — over shared utilities (rollout,
eval, run-provenance), with **no forced common interface**. An **experiment writes its learning stack** by
importing a trainer + `nets` and composing them (wire blocks → policy, pick trainer, set the mission +
hyperparams, run). Proven pieces graduate back into `nets.py` / `train.py`.

## Build approach — clean, not copied

- The lab is built **fresh**. `nets.py` / `train.py` start minimal and gain a block or trainer **only when
  an experiment needs it**, referencing the archived `zymera_env` for the algorithm.
- Old experiment scripts, the `examples/` sprawl, and the overnight harness are **not** brought over.
- `zymera_env` stays archived (`archive/pre-zymera-lab-fold`) as a read-only reference.

## Phasing (fold-then-extend)

- **P1 — scaffold & green:** add minimal `zymera/nets.py`, `zymera/train.py`, `zymera/sensor.py`; create
  `zymera_experiments/`; archive `zymera_env`; rename folder `kymera/ → zymera_lab/`. The existing 216
  tests stay green.
- **P2 — grow:** add blocks/trainers (graph belief, ES role-switcher, PPO heads, certainty-field
  exploration) clean, driven by the first experiments; validate against archived results.
- **P3 — extend the contract:** hierarchy (lab low-level controllers), then learned comms (`zymera.comms`
  extension) — each only when an experiment needs it.

## Testing

- Keep the existing suite green (216 tests); the env parity gate runs off golden `.npz` (unchanged by the
  rename).
- New `nets` / `train` pieces get focused unit tests as they are added.
- Experiments are **not** part of the lab's test suite.

## Naming / git

- Package `zymera`; repo folder `zymera_lab/` (kymera renamed). Work on branch `zymera-lab-consolidation`.
- `zymera_env` is tagged `archive/pre-zymera-lab-fold` and left in place, read-only.
