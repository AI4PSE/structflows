import pytest
import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
from scipy.stats import ks_2samp

from flow.flow import Flow
from structflow.bayes.distribution import Gaussian, FlowDistribution

key = random.PRNGKey(42)
d = 2
n_steps = 200
n_samples = 2000

# --- HELPER MODULES ---
class FuncVelocity(eqx.Module):
    """Wraps a raw function to make it a valid Equinox Module (PyTree)."""
    func: callable = eqx.field(static=True)

    def __call__(self, x, t):
        return self.func(x, t)

class LinearVelocity(eqx.Module):
    """Specific module for the change of variables test."""
    scale: float # or jax.Array

    def __call__(self, x, t):
        return self.scale * x

# --- DEFINITIONS ---

# A simple constant velocity field for testing
c_vec = jnp.array([2.0, -3.0])
# We wrap the lambda in our new Class
v_const = FuncVelocity(lambda x, t: c_vec)

# The original velocity functions for the sanity checks
def _v1_fn(x, t):
    return jnp.sin(x) + jnp.cos(10 * t)

def _v2_fn(x, t):
    return x * t + 5.0 * (1.0 - t)

# Wrap them!
v1 = FuncVelocity(_v1_fn)
v2 = FuncVelocity(_v2_fn)


@pytest.fixture
def setup_flow():
    """Pytest fixture to set up the flow and distributions for tests."""
    base_dist = Gaussian(dim=d)
    target_dist = Gaussian(mean=c_vec, cov=jnp.eye(d))

    # v_const is now an eqx.Module, so Flow accepts it happily
    flow = Flow(v_const, n_steps=n_steps, integrator_name='rk4')

    flow_dist = FlowDistribution(flow, base_dist)
    return base_dist, target_dist, flow, flow_dist

def test_log_prob_of_flow_distribution(setup_flow):
    _, target_dist, _, flow_dist = setup_flow
    
    test_points = target_dist.b_sample(key, n_samples)
    log_prob_flow = flow_dist.b_log_prob(test_points)
    log_prob_analytical = target_dist.b_log_prob(test_points)

    assert jnp.allclose(log_prob_flow, log_prob_analytical, atol=1e-4)

def test_sampling_from_flow_distribution(setup_flow):
    _, target_dist, _, flow_dist = setup_flow

    samples_flow = flow_dist.b_sample(key, n_samples)
    samples_analytical = target_dist.b_sample(random.split(key)[1], n_samples)

    for i in range(d):
        ks_statistic, p_value = ks_2samp(samples_flow[:, i], samples_analytical[:, i])
        assert p_value > 0.05, f"KS test failed for dimension {i} with p-value={p_value}"

@pytest.mark.parametrize("v, atol", [(v1, 1e-3), (v2, 1e-6)])
def test_forward_and_backward(v, atol):
    x = random.normal(key, (d,))
    f = Flow(v, n_steps=n_steps, integrator_name='rk4')
    y = f.forward(x)
    x_recon = f.backward(y)
    assert jnp.allclose(x, x_recon, atol=atol)

@pytest.mark.parametrize("v, atol", [(v1, 1e-3), (v2, 1e-3)])
def test_forward_and_backward_trajectory(v, atol):
    x = random.normal(key, (d,))
    f = Flow(v, n_steps=n_steps, integrator_name='rk4')
    xts = f.forward_trajectory(x)
    xts_rev = f.backward_trajectory(xts[-1])
    xts_ = jnp.flip(xts_rev, axis=0)
    assert jnp.allclose(xts, xts_, atol=atol)

@pytest.mark.parametrize("v, atol", [(v1, 1e-4), (v2, 1e-4)])
def test_forward_and_forward_trajectory(v, atol):
    x = random.normal(key, (d,))
    f = Flow(v, n_steps=n_steps, integrator_name='rk4')
    y = f.forward(x)
    yts = f.forward_trajectory(x)
    assert jnp.allclose(y, yts[-1], atol=atol)

def test_prob_log_prob_consistency():
    x0 = random.normal(key, (d,))
    p0 = jnp.array(0.5)
    log_p0 = jnp.log(p0)

    flow = Flow(v1, n_steps=n_steps)

    # Forward
    p1, y1 = flow.push_forward(p0, x0)
    log_p1, y2 = flow.push_forward_log_prob(log_p0, x0)

    assert jnp.allclose(y1, y2)
    assert jnp.allclose(jnp.log(p1), log_p1, atol=1e-5)

    # Backward
    p0_recon, x0_recon1 = flow.push_backward(p1, y1)
    log_p0_recon, x0_recon2 = flow.push_backward_log_prob(log_p1, y1)

    assert jnp.allclose(x0_recon1, x0_recon2)
    assert jnp.allclose(jnp.log(p0_recon), log_p0_recon, atol=1e-5)

def test_batching_consistency():
    n_batch = 10
    xs = random.normal(key, (n_batch, d))
    log_ps = random.normal(key, (n_batch,))

    flow = Flow(v2, n_steps=n_steps)

    ys_batch = flow.b_forward(xs)
    ys_manual = jnp.stack([flow.forward(x) for x in xs])
    assert jnp.allclose(ys_batch, ys_manual)

    log_p_batch, y_log_p_batch = flow.b_push_forward_log_prob(log_ps, xs)
    # Note: explicit map/zip is needed because push_forward_log_prob is unbatched
    log_p_manual, y_log_p_manual = map(jnp.stack, zip(*[flow.push_forward_log_prob(lp, x) for lp, x in zip(log_ps, xs)]))
    
    assert jnp.allclose(log_p_batch, log_p_manual)
    assert jnp.allclose(y_log_p_batch, y_log_p_manual)

def test_change_of_variables_correctness():
    """
    Tests the change of variables formula.
    UPDATED: We now use LinearVelocity module instead of passing params separately.
    """
    d_local = 3
    a = 0.5
    n_steps_local = 100

    # 1. Define the linear velocity field MODULE
    # The parameter 'a' is now stored INSIDE the module
    v_linear = LinearVelocity(scale=a)
    
    # No more velocity_params argument here!
    flow = Flow(v_linear, n_steps=n_steps_local, integrator_name='rk4')

    # 2. Define starting point
    x0 = random.normal(key, (d_local,))
    base_dist = Gaussian(dim=d_local)
    log_p_x0 = base_dist.log_prob(x0)

    # 3. Analytical result
    analytical_log_det_jacobian_change = -(a * d_local)

    # 4. Flow calculation
    log_p_x1, _ = flow.push_forward_log_prob(log_p_x0, x0)

    # 5. Assert
    assert jnp.allclose(log_p_x1, log_p_x0 + analytical_log_det_jacobian_change, atol=1e-3)

@pytest.mark.parametrize("v, integrator, atol", [
    (v1, 'rk4', 1e-6), 
    (v2, 'rk4', 1e-6),
    (v1, 'euler', 1e-2), 
    (v2, 'euler', 1e-2)
])
def test_forward_and_backward_integrators(v, integrator, atol):
    """ Renamed slightly to avoid name collision with previous test """
    x = random.normal(key, (d,))
    f = Flow(v, n_steps=n_steps, integrator_name=integrator)
    y = f.forward(x)
    x_recon = f.backward(y)
    assert jnp.isclose(x, x_recon, atol=atol).all()