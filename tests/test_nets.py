import jax
import jax.numpy as jnp
from zymera import nets


def test_mlp_shapes_and_determinism():
    key = jax.random.PRNGKey(0)
    params = nets.mlp_init(key, sizes=(4, 8, 2))
    x = jnp.ones((3, 4))                       # batch of 3, in-dim 4
    y = nets.mlp_apply(params, x)
    assert y.shape == (3, 2)
    y2 = nets.mlp_apply(params, x)
    assert jnp.allclose(y, y2)                 # pure function


def test_mlp_jittable():
    key = jax.random.PRNGKey(1)
    params = nets.mlp_init(key, sizes=(2, 4, 1))
    f = jax.jit(lambda p, x: nets.mlp_apply(p, x))
    out = f(params, jnp.zeros((1, 2)))
    assert out.shape == (1, 1)
