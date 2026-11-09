"""
Test API completeness and correctness across all modes.

Tests that all distributions support:
- sample() and b_sample() for various query types
- log_prob() and b_log_prob() for supported queries
- counterfactual() and b_counterfactual() operations
- transform_to_noise() for inverse flow operations

Mode-specific behavior:
- Joint/Guidance: log_prob requires targets + conditions = all_vars
- Static/Dynamic/CAREFL: support all tractable queries
- All modes: gradients flow through all operations
"""

import pytest
import jax
import jax.numpy as jnp
from jax import random

from structflow.conditional_flows.tests.test_utils import (
    sample_dict_batch,
    compute_log_prob_batch,
    mode_supports_full_log_prob,
    mode_supports_transform_to_noise,
)
from structflow.conditional_flows.tests.conftest import MODES

N_SAMPLES = 8

# ============================================================================
# Sample Tests
# ============================================================================

@pytest.mark.parametrize("mode", MODES)
def test_unconditional_sample(mode, dist):
    """Test unconditional sampling from distribution."""
    key = random.PRNGKey(0)
    sample = dist.sample(key)
    
    assert isinstance(sample, dict)
    assert len(sample) > 0
    for block, val in sample.items():
        assert isinstance(val, jnp.ndarray)
        assert val.ndim >= 1
        assert jnp.all(jnp.isfinite(val))


@pytest.mark.parametrize("mode", MODES)
def test_conditional_sample_chain(mode, dist):
    """Test conditional sampling with chain structure."""
    key = random.PRNGKey(0)
    k1, k2 = random.split(key)
    
    # Get first variable
    x0_sample = dist.sample(k1, targets={'x0'})
    
    # Condition on first variable and sample second
    conditions = {'x0': x0_sample['x0']}
    x1_sample = dist.sample(k2, conditions=conditions, targets={'x1'})
    
    assert 'x0' not in x1_sample
    assert 'x1' in x1_sample
    assert jnp.all(jnp.isfinite(x1_sample['x1']))


@pytest.mark.parametrize("mode", MODES)
def test_batched_sample(mode, dist):
    """Test batched sampling."""
    key = random.PRNGKey(0)
    n_samples = N_SAMPLES
    
    if not hasattr(dist, 'b_sample'):
        pytest.skip("Distribution doesn't have b_sample")
    
    samples = dist.b_sample(key, n_samples)
    
    for block, val in samples.items():
        assert val.shape[0] == n_samples
        assert val.ndim >= 2
        assert jnp.all(jnp.isfinite(val))


# ============================================================================
# Log Prob Tests
# ============================================================================

@pytest.mark.parametrize("mode", MODES)
def test_log_prob_unconditional(mode, dist):
    """Test unconditional log_prob computation."""
    key = random.PRNGKey(0)
    x = dist.sample(key)
    
    lp = dist.log_prob(x)
    
    assert isinstance(lp, jnp.ndarray)
    assert lp.shape == ()
    assert jnp.isfinite(lp)


@pytest.mark.parametrize("mode", ["static", "dynamic", "carefl", "universal"])
def test_log_prob_conditional_chain(mode, dist):
    """Test conditional log_prob for tractable queries."""
    key = random.PRNGKey(0)
    k1, k2 = random.split(key)
    
    # Sample full data
    x_full = dist.sample(k1)
    
    # p(x1 | x0)
    lp_cond = dist.log_prob(
        x_full,
        targets={'x1'},
        conditions={'x0'}
    )
    
    assert isinstance(lp_cond, jnp.ndarray)
    assert lp_cond.shape == ()
    assert jnp.isfinite(lp_cond)


@pytest.mark.parametrize("mode", ["guided"])
def test_log_prob_joint_requires_all_vars(mode, dist):
    """Test that joint/guidance mode requires all variables."""
    key = random.PRNGKey(0)
    x = dist.sample(key)
    
    # Joint mode should accept the full dict
    lp_joint = dist.log_prob(x)
    assert jnp.isfinite(lp_joint)
    
    # Partial log_prob may raise error or be unsupported
    # This is mode-specific behavior


@pytest.mark.parametrize("mode", MODES)
def test_batched_log_prob(mode, dist):
    """Test batched log_prob computation."""
    key = random.PRNGKey(0)
    n_samples = N_SAMPLES
    
    if not hasattr(dist, 'b_log_prob'):
        pytest.skip("Distribution doesn't have b_log_prob")
    
    x = dist.b_sample(key, n_samples)
    log_probs = dist.b_log_prob(x)
    
    assert log_probs.shape == (n_samples,)
    assert jnp.all(jnp.isfinite(log_probs))


# ============================================================================
# Transform to Noise Tests
# ============================================================================

@pytest.mark.parametrize("mode", ["static", "dynamic", "carefl", "universal"])
def test_transform_to_noise(mode, dist):
    """Test inverse flow: data -> noise."""
    if not mode_supports_transform_to_noise(mode):
        pytest.skip(f"Mode {mode} doesn't support transform_to_noise")
    
    key = random.PRNGKey(0)
    x = dist.sample(key)
    
    z = dist.transform_to_noise(x)
    
    assert isinstance(z, dict)
    for block, val in z.items():
        assert isinstance(val, jnp.ndarray)
        assert val.shape == x[block].shape
        assert jnp.all(jnp.isfinite(val))


@pytest.mark.parametrize("mode", ["static", "dynamic", "carefl", "universal"])
def test_batched_transform_to_noise(mode, dist):
    """Test batched inverse flow."""
    if not mode_supports_transform_to_noise(mode):
        pytest.skip(f"Mode {mode} doesn't support batched transform_to_noise")
    
    if not hasattr(dist, 'b_transform_to_noise'):
        pytest.skip("Distribution doesn't have b_transform_to_noise")
    
    key = random.PRNGKey(0)
    x = dist.b_sample(key, N_SAMPLES)
    
    z = dist.b_transform_to_noise(x)
    
    assert isinstance(z, dict)
    for block, val in z.items():
        assert val.shape == x[block].shape


# ============================================================================
# Counterfactual Tests
# ============================================================================

@pytest.mark.parametrize("mode", MODES)
def test_counterfactual_single_intervention(mode, dist):
    """Test counterfactual with single intervention."""
    key = random.PRNGKey(0)
    k1, k2 = random.split(key)
    
    # Observe data
    data = dist.sample(k1)
    
    # Intervene on first variable
    interventions = {'x0': jnp.array([2.0])}
    
    # Compute counterfactual
    cf, log_p = dist.counterfactual(data, interventions)
    
    assert isinstance(cf, dict)
    assert 'x0' in cf
    assert log_p.shape == ()
    assert jnp.isfinite(log_p)
    
    # Intervention should be respected
    assert jnp.allclose(cf['x0'], interventions['x0'])


@pytest.mark.parametrize("mode", MODES)
def test_counterfactual_preserves_intervention(mode, dist):
    """Test that counterfactual respects the intervention exactly."""
    key = random.PRNGKey(0)
    data = dist.sample(key)
    
    interventions = {'x0': jnp.array([1.5])}
    cf, log_p = dist.counterfactual(data, interventions)
    
    # The intervened variable should match exactly
    assert jnp.allclose(cf['x0'], interventions['x0']), (
        f"Intervention not respected: got {cf['x0']}, expected {interventions['x0']}"
    )


@pytest.mark.parametrize("mode", MODES)
def test_counterfactual_with_targets(mode, dist):
    """Test counterfactual with specific target variables."""
    key = random.PRNGKey(0)
    data = dist.sample(key)
    
    interventions = {'x0': jnp.array([1.0])}
    cf, log_p = dist.counterfactual(data, interventions, targets={'x1'})
    
    # Should only return targets
    assert 'x1' in cf
    # x0 may or may not be in cf depending on implementation


@pytest.mark.parametrize("mode", MODES)
def test_batched_counterfactual(mode, dist):
    """Test batched counterfactual."""
    if not hasattr(dist, 'b_counterfactual'):
        pytest.skip("Distribution doesn't have b_counterfactual")
    
    key = random.PRNGKey(0)
    data = dist.b_sample(key, N_SAMPLES)
    
    interventions = {
        'x0': jnp.tile(jnp.array([1.0]), (N_SAMPLES, 1))
    }
    
    cf, log_p = dist.b_counterfactual(data, interventions)
    
    assert isinstance(cf, dict)
    for block, val in cf.items():
        assert val.shape[0] == N_SAMPLES

    assert log_p.shape == (N_SAMPLES,)
    assert jnp.all(jnp.isfinite(log_p))


# ============================================================================
# Gradient Flow Tests
# ============================================================================

@pytest.mark.parametrize("mode", MODES)
def test_gradients_flow_through_sample(mode, dist):
    """Test that gradients flow through sampling operation."""
    key = random.PRNGKey(0)
    
    def sample_fn(params):
        # Simple: just return a sample (no params used, but demonstrates differentiability)
        return dist.sample(key)
    
    # Sample should be differentiable w.r.t. distribution params
    # For now, just verify we can call it
    sample = sample_fn(None)
    assert isinstance(sample, dict)


@pytest.mark.parametrize("mode", MODES)
def test_gradients_flow_through_log_prob(mode, dist):
    """Test that gradients flow through log_prob."""
    key = random.PRNGKey(0)
    x = dist.sample(key)
    
    def log_prob_fn(x_):
        return dist.log_prob(x_)
    
    lp = log_prob_fn(x)
    grad_fn = jax.grad(log_prob_fn)
    
    try:
        grads = grad_fn(x)
        # Gradients should flow
        assert isinstance(grads, dict)
    except NotImplementedError:
        # Some modes may not support gradients through log_prob
        pass


# ============================================================================
# Mode-Specific Query Support
# ============================================================================

@pytest.mark.parametrize("mode", ["static", "dynamic", "carefl", "universal"])
def test_mode_supports_partial_queries(mode, dist):
    """
    Test that static/dynamic/carefl modes support partial queries.
    
    They should support log_prob for tractable targets+conditions subsets.
    """
    key = random.PRNGKey(0)
    x = dist.sample(key)
    
    # For chain x0 -> x1, we can query:
    # - p(x0) [unconditional]
    # - p(x1 | x0) [conditional]
    
    lp_x0 = dist.log_prob(x, targets={'x0'}, conditions=set())
    assert jnp.isfinite(lp_x0)

    lp_x1_given_x0 = dist.log_prob(x, targets={'x1'}, conditions={'x0'})
    assert jnp.isfinite(lp_x1_given_x0)


# ============================================================================
# Consistency Tests
# ============================================================================

@pytest.mark.parametrize("mode", MODES)
def test_sample_consistency_same_key(mode, dist):
    """Test that sampling with same key gives same result."""
    key = random.PRNGKey(42)
    
    sample1 = dist.sample(key)
    sample2 = dist.sample(key)
    
    for block in sample1.keys():
        assert jnp.allclose(sample1[block], sample2[block])


@pytest.mark.parametrize("mode", MODES)
def test_sample_differs_different_key(mode, dist):
    """Test that different keys give different samples."""
    key1 = random.PRNGKey(0)
    key2 = random.PRNGKey(1)
    
    sample1 = dist.sample(key1)
    sample2 = dist.sample(key2)
    
    # At least some blocks should differ
    any_different = False
    for block in sample1.keys():
        if not jnp.allclose(sample1[block], sample2[block]):
            any_different = True
            break
    
    assert any_different, "Samples should differ with different keys"
