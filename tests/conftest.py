"""Shared fixtures. Forces a headless matplotlib backend before any test import."""

import matplotlib

matplotlib.use("Agg", force=True)

import jax
import pytest


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)
