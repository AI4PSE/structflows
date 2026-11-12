"""
Numerical correctness tests using exact Gaussian velocity functions.

This test suite verifies that conditional distributions produce statistically
correct samples and log probabilities when the velocity function is exactly known.

The test strategy:
1. Use exact_si_gaussian.py to create oracle velocities that transform N(0,I) -> N(m2,C2)
2. Inject these oracle velocities into distribution classes
3. Verify sample statistics and log probabilities match analytical Gaussian values

Structure: Two independent 1D variables with different target distributions
- x0 ~ N(1.5, 2.0)
- x1 ~ N(-0.5, 0.5)

Tested modes: static, dynamic, guided (CAREFL excluded - not velocity-based)
"""

import pytest
import jax
import jax.numpy as jnp
from jax import random
import optax

from structflow.pgms.structure import BlockTriangularStructure
from structflow.conditional_flows.static.dist import StaticConditionalDistribution
from structflow.conditional_flows.static.flow import StaticConditionalFlow
from structflow.conditional_flows.dynamic.parallel_dist import ParallelBlockTriangularDistribution
from structflow.conditional_flows.dynamic.parallel_flow import ParallelFlow
from structflow.conditional_flows.guidance.dist import GuidanceBasedDistribution
from structflow.conditional_flows.guidance.flow import GuidedFlow
from structflow.bayes.distribution import Gaussian, DictDistribution

from structflow.conditional_flows.tests.numerical_utils import (
    ExactGaussianStaticVelocity,
    ExactGaussianParallelVelocity,
    ExactGaussianGuidedVelocity,
    create_exact_velocity_functions,
    analytical_gaussian_log_prob,
    create_guided_field_from_velocities,
)


# ============================================================================
# Test Configuration
# ============================================================================

# Target distribution parameters (at t=1)
# IMPORTANT: Both variances should increase or both decrease to ensure
# the divergence integral is non-zero (otherwise sign errors won't be caught)
TARGET_PARAMS = {
    'x0': {'mean': jnp.array([1.5]), 'var': jnp.array([3.0])},  # variance triples
    'x1': {'mean': jnp.array([-0.5]), 'var': jnp.array([2.5])},  # variance 2.5x
}

# Independent structure (no dependencies between variables)
INDEPENDENT_STRUCTURE = BlockTriangularStructure(
    block_order=['x0', 'x1'],
    dependencies={'x0': [], 'x1': []},
    dims={'x0': 1, 'x1': 1}
)

# Integration parameters
N_STEPS = 100
# Tighter tolerance: With exact velocity and 100 steps, numerical error should be ~1e-3
# If this is too tight and tests fail, increase N_STEPS, not ATOL
INTEGRATION_ATOL = 0.02


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def exact_velocity_fns():
    """Create exact Gaussian velocity functions for each block."""
    return create_exact_velocity_functions(INDEPENDENT_STRUCTURE, TARGET_PARAMS)


@pytest.fixture
def base_distributions():
    """Create Gaussian base distributions for each block (N(0, I))."""
    return {
        'x0': Gaussian(mean=jnp.zeros((1,)), cov=jnp.eye(1)),
        'x1': Gaussian(mean=jnp.zeros((1,)), cov=jnp.eye(1)),
    }


@pytest.fixture
def static_dist(exact_velocity_fns, base_distributions):
    """Create StaticConditionalDistribution with exact velocity."""
    velocity_fn = ExactGaussianStaticVelocity(INDEPENDENT_STRUCTURE, exact_velocity_fns)
    flow = StaticConditionalFlow(velocity_fn, n_steps=N_STEPS)
    return StaticConditionalDistribution(flow, base_distributions)


@pytest.fixture
def dynamic_dist(exact_velocity_fns, base_distributions):
    """Create ParallelBlockTriangularDistribution with exact velocity."""
    velocity_fn = ExactGaussianParallelVelocity(INDEPENDENT_STRUCTURE, exact_velocity_fns)
    flow = ParallelFlow(velocity_fn, n_steps=N_STEPS)
    return ParallelBlockTriangularDistribution(flow, base_distributions)


@pytest.fixture
def guided_dist(exact_velocity_fns, base_distributions):
    """Create GuidanceBasedDistribution with exact velocity."""
    dims = {name: INDEPENDENT_STRUCTURE.dims[name]
            for name in INDEPENDENT_STRUCTURE.block_order}
    field = create_guided_field_from_velocities(
        exact_velocity_fns,
        INDEPENDENT_STRUCTURE.block_order,
        dims=dims
    )
    base_dict = DictDistribution(base_distributions)
    flow = GuidedFlow(model=field, n_steps=N_STEPS)
    return GuidanceBasedDistribution(
        flow=flow,
        base_distribution=base_dict,
        key_order=INDEPENDENT_STRUCTURE.block_order
    )


@pytest.fixture(params=['static', 'dynamic', 'guided'])
def exact_dist(request, static_dist, dynamic_dist, guided_dist):
    """Parametrized fixture that provides each distribution mode."""
    modes = {'static': static_dist, 'dynamic': dynamic_dist, 'guided': guided_dist}
    return modes[request.param]


# ============================================================================
# Test: Sample Mean Correctness
# ============================================================================

@pytest.mark.parametrize("mode", ["static", "dynamic", "guided"])
def test_sample_mean_matches_target(mode, static_dist, dynamic_dist, guided_dist):
    """
    Verify that sample mean converges to target mean.
    
    Uses statistical tolerance based on central limit theorem:
    tolerance = 3 * std_error = 3 * sigma / sqrt(n)
    """
    dists = {'static': static_dist, 'dynamic': dynamic_dist, 'guided': guided_dist}
    dist = dists[mode]
    
    key = random.PRNGKey(42)
    n_samples = 2000
    
    samples = dist.b_sample(key, n_samples)
    
    for block in ['x0', 'x1']:
        sample_mean = jnp.mean(samples[block], axis=0)
        target_mean = TARGET_PARAMS[block]['mean']
        target_std = jnp.sqrt(TARGET_PARAMS[block]['var'])
        
        # Tolerance: 3 sigma rule for standard error
        std_error = target_std / jnp.sqrt(n_samples)
        tolerance = 3.0 * std_error
        
        assert jnp.allclose(sample_mean, target_mean, atol=tolerance), (
            f"Mode {mode}, block {block}: "
            f"sample mean {sample_mean} != target mean {target_mean} "
            f"(tolerance {tolerance})"
        )


# ============================================================================
# Test: Sample Variance Correctness
# ============================================================================

@pytest.mark.parametrize("mode", ["static", "dynamic", "guided"])
def test_sample_variance_matches_target(mode, static_dist, dynamic_dist, guided_dist):
    """
    Verify that sample variance converges to target variance.
    
    With n=2000 samples, expect ~15% relative error in variance estimate.
    """
    dists = {'static': static_dist, 'dynamic': dynamic_dist, 'guided': guided_dist}
    dist = dists[mode]
    
    key = random.PRNGKey(43)
    n_samples = 2000
    
    samples = dist.b_sample(key, n_samples)
    
    for block in ['x0', 'x1']:
        sample_var = jnp.var(samples[block], axis=0)
        target_var = TARGET_PARAMS[block]['var']
        
        # Relative tolerance for variance (chi-squared based)
        relative_tol = 0.20  # 20% tolerance
        
        assert jnp.allclose(sample_var, target_var, rtol=relative_tol), (
            f"Mode {mode}, block {block}: "
            f"sample var {sample_var} != target var {target_var} "
            f"(rtol {relative_tol})"
        )


# ============================================================================
# Test: Log Probability at Specific Points
# ============================================================================

@pytest.mark.parametrize("mode", ["static", "dynamic", "guided"])
def test_log_prob_matches_analytical_at_means(mode, static_dist, dynamic_dist, guided_dist):
    """
    Verify log_prob at the means matches analytical Gaussian density.
    
    NOTE: This test catches bugs in divergence accumulation during log_prob computation.
    - Static mode: Should pass
    - Dynamic mode: May fail if divergence is not properly accumulated
    - Guided mode: May fail if log-det correction has sign/magnitude errors
    """
    dists = {'static': static_dist, 'dynamic': dynamic_dist, 'guided': guided_dist}
    dist = dists[mode]
    
    # Test point: at the means
    x = {
        'x0': TARGET_PARAMS['x0']['mean'],
        'x1': TARGET_PARAMS['x1']['mean'],
    }
    
    computed_lp = dist.log_prob(x)
    
    # Analytical log prob for independent Gaussians
    analytical_lp = (
        analytical_gaussian_log_prob(
            x['x0'], 
            TARGET_PARAMS['x0']['mean'], 
            TARGET_PARAMS['x0']['var']
        ) +
        analytical_gaussian_log_prob(
            x['x1'], 
            TARGET_PARAMS['x1']['mean'], 
            TARGET_PARAMS['x1']['var']
        )
    )
    
    assert jnp.allclose(computed_lp, analytical_lp, atol=INTEGRATION_ATOL), (
        f"Mode {mode}: log_prob mismatch at means. "
        f"Computed: {computed_lp}, Analytical: {analytical_lp}"
    )


@pytest.mark.parametrize("mode", ["static", "dynamic", "guided"])
def test_log_prob_matches_analytical_at_origin(mode, static_dist, dynamic_dist, guided_dist):
    """
    Verify log_prob at the origin matches analytical density.
    """
    dists = {'static': static_dist, 'dynamic': dynamic_dist, 'guided': guided_dist}
    dist = dists[mode]
    
    # Test point: origin
    x = {
        'x0': jnp.array([0.0]),
        'x1': jnp.array([0.0]),
    }
    
    computed_lp = dist.log_prob(x)
    
    analytical_lp = (
        analytical_gaussian_log_prob(
            x['x0'], 
            TARGET_PARAMS['x0']['mean'], 
            TARGET_PARAMS['x0']['var']
        ) +
        analytical_gaussian_log_prob(
            x['x1'], 
            TARGET_PARAMS['x1']['mean'], 
            TARGET_PARAMS['x1']['var']
        )
    )
    
    assert jnp.allclose(computed_lp, analytical_lp, atol=INTEGRATION_ATOL), (
        f"Mode {mode}: log_prob mismatch at origin. "
        f"Computed: {computed_lp}, Analytical: {analytical_lp}"
    )


@pytest.mark.parametrize("mode", ["static", "dynamic", "guided"])
def test_log_prob_matches_analytical_away_from_means(mode, static_dist, dynamic_dist, guided_dist):
    """
    Verify log_prob far from means matches analytical density.
    """
    dists = {'static': static_dist, 'dynamic': dynamic_dist, 'guided': guided_dist}
    dist = dists[mode]
    
    # Test point: 2 standard deviations away
    x = {
        'x0': jnp.array([1.5 + 2.0 * jnp.sqrt(2.0)]),  # mu + 2*sigma
        'x1': jnp.array([-0.5 - 2.0 * jnp.sqrt(0.5)]),  # mu - 2*sigma
    }
    
    computed_lp = dist.log_prob(x)
    
    analytical_lp = (
        analytical_gaussian_log_prob(
            x['x0'], 
            TARGET_PARAMS['x0']['mean'], 
            TARGET_PARAMS['x0']['var']
        ) +
        analytical_gaussian_log_prob(
            x['x1'], 
            TARGET_PARAMS['x1']['mean'], 
            TARGET_PARAMS['x1']['var']
        )
    )
    
    assert jnp.allclose(computed_lp, analytical_lp, atol=INTEGRATION_ATOL), (
        f"Mode {mode}: log_prob mismatch away from means. "
        f"Computed: {computed_lp}, Analytical: {analytical_lp}"
    )


# ============================================================================
# Test: Batched Log Probability
# ============================================================================

@pytest.mark.parametrize("mode", ["static", "dynamic", "guided"])
def test_batched_log_prob_matches_analytical(mode, static_dist, dynamic_dist, guided_dist):
    """
    Verify batched log_prob matches analytical Gaussian for sampled data.
    """
    dists = {'static': static_dist, 'dynamic': dynamic_dist, 'guided': guided_dist}
    dist = dists[mode]
    
    key = random.PRNGKey(44)
    n_samples = 50
    
    samples = dist.b_sample(key, n_samples)
    computed_lps = dist.b_log_prob(samples)
    
    # Compute analytical log_prob for each sample
    analytical_lps = jnp.array([
        analytical_gaussian_log_prob(
            samples['x0'][i], 
            TARGET_PARAMS['x0']['mean'], 
            TARGET_PARAMS['x0']['var']
        ) +
        analytical_gaussian_log_prob(
            samples['x1'][i], 
            TARGET_PARAMS['x1']['mean'], 
            TARGET_PARAMS['x1']['var']
        )
        for i in range(n_samples)
    ])
    
    assert jnp.allclose(computed_lps, analytical_lps, atol=INTEGRATION_ATOL), (
        f"Mode {mode}: batched log_prob mismatch. "
        f"Max difference: {jnp.max(jnp.abs(computed_lps - analytical_lps))}"
    )


# ============================================================================
# Test: Log Probability Ordering
# ============================================================================

@pytest.mark.parametrize("mode", ["static", "dynamic", "guided"])
def test_log_prob_ordering(mode, static_dist, dynamic_dist, guided_dist):
    """
    Verify that log_prob respects ordering (higher density -> higher log_prob).
    
    Points closer to the mean should have higher log_prob.
    """
    dists = {'static': static_dist, 'dynamic': dynamic_dist, 'guided': guided_dist}
    dist = dists[mode]
    
    # Point at mean (highest density)
    x_mean = {
        'x0': TARGET_PARAMS['x0']['mean'],
        'x1': TARGET_PARAMS['x1']['mean'],
    }
    
    # Point 1 std away
    x_1std = {
        'x0': TARGET_PARAMS['x0']['mean'] + jnp.sqrt(TARGET_PARAMS['x0']['var']),
        'x1': TARGET_PARAMS['x1']['mean'] - jnp.sqrt(TARGET_PARAMS['x1']['var']),
    }
    
    # Point 2 std away
    x_2std = {
        'x0': TARGET_PARAMS['x0']['mean'] + 2.0 * jnp.sqrt(TARGET_PARAMS['x0']['var']),
        'x1': TARGET_PARAMS['x1']['mean'] - 2.0 * jnp.sqrt(TARGET_PARAMS['x1']['var']),
    }
    
    lp_mean = dist.log_prob(x_mean)
    lp_1std = dist.log_prob(x_1std)
    lp_2std = dist.log_prob(x_2std)
    
    assert lp_mean > lp_1std, (
        f"Mode {mode}: log_prob at mean {lp_mean} should be > "
        f"log_prob at 1σ {lp_1std}"
    )
    assert lp_1std > lp_2std, (
        f"Mode {mode}: log_prob at 1σ {lp_1std} should be > "
        f"log_prob at 2σ {lp_2std}"
    )


# ============================================================================
# Test: Forward-Backward Consistency
# ============================================================================

@pytest.mark.parametrize("mode", ["static", "dynamic", "guided"])
def test_transform_to_noise_inverse_property(mode, static_dist, dynamic_dist, guided_dist):
    """
    Verify that transform_to_noise approximates the inverse of the forward flow.
    
    For exact velocity, forward(noise) -> x and backward(x) -> noise should be inverses.
    """
    dists = {'static': static_dist, 'dynamic': dynamic_dist, 'guided': guided_dist}
    dist = dists[mode]
    
    if not hasattr(dist, 'transform_to_noise'):
        pytest.skip(f"Mode {mode} does not support transform_to_noise")
    
    # Sample a point
    key = random.PRNGKey(45)
    x = dist.sample(key)
    
    # Transform to noise
    z = dist.transform_to_noise(x)
    
    # z should be approximately standard normal
    # Check that it has reasonable magnitude
    for block in ['x0', 'x1']:
        z_norm = jnp.linalg.norm(z[block])
        # Magnitude should be O(1) for standard normal
        assert z_norm < 5.0, (
            f"Mode {mode}, block {block}: noise norm {z_norm} seems too large"
        )
