"""zymera — JAX-native multi-agent grid simulator.

Headless core: ``import zymera`` pulls no plotting or training libraries.
Drawing lives in :mod:`zymera.viz`; training machinery in :mod:`zymera.lab`
(both are opt-in extras). Components live in the subnamespaces
(``zymera.worldgen`` / ``dynamics`` / ``comms`` / ``obs`` / ``missions`` /
``missions_terms`` / ``metrics``).
"""

# NOTE: .env must be imported FIRST — it drives the package load order.
# Component modules back-import names from zymera.env (e.g. ACTION_DELTAS);
# env.py defines those above its own component imports, so the cycle resolves
# only when env.py is the entry point.
from .env import (
    ActionId,
    ACTION_DELTAS,
    Body,
    Env,
    GridEnv,
    N_ACTIONS,
    World,
    list_envs,
    make,
    make_from,
    register_env,
)
from . import comms, dynamics, metrics, missions, missions_terms, obs, worldgen
from .missions import RewardTerm
from .rollout import random_policy, rollout

__version__ = "0.1.0"

__all__ = [
    "ActionId",
    "ACTION_DELTAS",
    "Body",
    "Env",
    "GridEnv",
    "N_ACTIONS",
    "RewardTerm",
    "World",
    "list_envs",
    "make",
    "make_from",
    "register_env",
    "random_policy",
    "rollout",
    # subnamespaces
    "comms", "dynamics", "metrics", "missions", "missions_terms", "obs", "worldgen",
]
