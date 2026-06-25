import jax
import zymera
from zymera import train


def test_evaluate_random_policy_returns_metrics():
    env = zymera.make("comm-coverage", grid=8, n_agents=2)
    report = train.evaluate(env, zymera.random_policy, n_steps=16,
                            n_episodes=4, key=jax.random.PRNGKey(0))
    assert set(report).issuperset({"return_mean", "return_std", "n_episodes"})
    assert report["n_episodes"] == 4
    assert report["return_mean"] == report["return_mean"]   # not NaN
