"""kymera — JAX-native multi-agent grid simulator.

Headless core: ``import kymera`` pulls no plotting or training libraries.
Drawing lives in :mod:`kymera.viz`; training machinery in :mod:`kymera.lab`
(both are opt-in extras).
"""

from .env import (
    ActionId,
    ACTION_DELTAS,
    Body,
    Env,
    N_ACTIONS,
    World,
    list_envs,
    make,
    make_from,
    register_env,
)
from .rollout import random_policy, rollout

__version__ = "0.1.0"

__all__ = [
    "ActionId",
    "ACTION_DELTAS",
    "Body",
    "Env",
    "N_ACTIONS",
    "World",
    "list_envs",
    "make",
    "make_from",
    "register_env",
    "random_policy",
    "rollout",
    "__version__",
]
