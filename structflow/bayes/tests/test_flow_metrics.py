"""Tests for ``bayes.metrics.analyze_flow_dynamics``.

``analyze_flow_dynamics`` is a diagnostic utility for flow-based distributions.
It is written against the *dict-based* velocity API (``velocity_fn(x_dict, t)``),
so it must be exercised with a :class:`StaticConditionalDistribution` whose
velocity function consumes and returns dictionaries of per-block values.
"""

import jax
import jax.numpy as jnp

from structflow.bayes.metrics import analyze_flow_dynamics
from structflow.bayes.distribution import Gaussian
from structflow.pgms.structure import BlockTriangularStructure
from structflow.conditional_flows.static.dist import StaticConditionalDistribution
from structflow.conditional_flows.static.flow import StaticConditionalFlow
from structflow.conditional_flows.tests.numerical_utils import (
    ExactGaussianStaticVelocity,
    create_exact_velocity_functions,
)

N_STEPS = 10


def _build_static_dist():
    """Build a simple two-block (x -> y) static flow with exact Gaussian velocities."""
    structure = BlockTriangularStructure(
        block_order=["x", "y"],
        dependencies={"x": [], "y": ["x"]},
        dims={"x": 1, "y": 1},
    )

    # Targets: N(0, I) -> N(mean, diag(var)) per block. Non-trivial means/vars
    # give the velocity field a non-zero Jacobian so the stiffness diagnostic is
    # meaningful.
    target_params = {
        "x": {"mean": jnp.array([1.0]), "var": jnp.array([0.5])},
        "y": {"mean": jnp.array([-1.0]), "var": jnp.array([2.0])},
    }
    v_fns = create_exact_velocity_functions(structure, target_params)
    velocity_fn = ExactGaussianStaticVelocity(structure, v_fns)
    flow = StaticConditionalFlow(velocity_fn, n_steps=N_STEPS)

    base_distributions = {
        "x": Gaussian(mean=jnp.zeros((1,)), cov=jnp.eye(1)),
        "y": Gaussian(mean=jnp.zeros((1,)), cov=jnp.eye(1)),
    }
    return StaticConditionalDistribution(flow, base_distributions)


def test_flow_metrics():
    key = jax.random.PRNGKey(0)
    dist = _build_static_dist()

    metrics = analyze_flow_dynamics(dist, key, num_samples=100)

    # The diagnostic returns per-timestep velocity/stiffness traces plus a
    # scalar path-straightness summary.
    assert "avg_velocity" in metrics
    assert "avg_stiffness" in metrics
    assert "path_straightness" in metrics

    # One value per integration node (n_steps + 1).
    assert len(metrics["avg_velocity"]) == N_STEPS + 1
    assert len(metrics["avg_stiffness"]) == N_STEPS + 1

    # All diagnostics should be finite and non-negative.
    assert jnp.all(jnp.isfinite(metrics["avg_velocity"]))
    assert jnp.all(metrics["avg_stiffness"] >= 0.0)
    assert jnp.isfinite(metrics["path_straightness"])


if __name__ == "__main__":
    test_flow_metrics()
    print("Test passed!")
