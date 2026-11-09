import jax
import jax.numpy as jnp
import pytest
from structflow.bayes.distribution import Gaussian, ProcessDistribution
from structflow.bayes.metrics import compute_swd_jax, compute_sinkhorn_jax, compare_distributions, compute_kl_jax

def test_swd_identity():
    key = jax.random.PRNGKey(0)
    X = jax.random.normal(key, (100, 2))
    # Comparison with itself should be close to zero
    swd = compute_swd_jax(key, X, X)
    assert swd < 0.1

def test_swd_sensitivity():
    key = jax.random.PRNGKey(1)
    X = jax.random.normal(key, (100, 2))
    Y = jax.random.normal(key, (100, 2)) + 5.0
    swd_shifted = compute_swd_jax(key, X, Y)
    swd_identity = compute_swd_jax(key, X, X)
    # compute_swd_jax returns a value normalised by sqrt(N * D) (see its
    # docstring), so the absolute magnitude is small; the meaningful,
    # scale-free property is that a shifted distribution is clearly
    # separated from itself.
    assert swd_shifted > swd_identity
    assert swd_shifted > 0.1

def test_sinkhorn_identity():
    X = jnp.array([[0.0, 0.0], [1.0, 1.0]])
    cost = compute_sinkhorn_jax(X, X)
    # Entropic cost at identity is small (depending on epsilon)
    assert cost < 0.2

def test_kl_divergence():
    key = jax.random.PRNGKey(42)
    p = Gaussian(dim=2)
    # Slightly shifted Gaussian
    q = Gaussian(mean=jnp.array([0.5, 0.5]), cov=jnp.eye(2))
    
    kl = compute_kl_jax(p, q, key, num_samples=5000)
    
    # Analytic KL for shifted N(0, I) vs N(mu, I) is 0.5 * ||mu||^2
    # 0.5 * (0.5^2 + 0.5^2) = 0.5 * 0.5 = 0.25
    assert jnp.abs(kl - 0.25) < 0.05

def test_compare_distributions_identity():
    key = jax.random.PRNGKey(2)
    p = Gaussian(dim=2)
    q = Gaussian(dim=2)
    
    metrics = compare_distributions(p, q, key, num_samples=500)
    
    assert "SWD" in metrics
    assert "Test_NLL" in metrics
    assert "KL" in metrics
    
    assert metrics["SWD"] < 0.5
    assert jnp.abs(metrics["KL"]) < 0.1

def test_compare_distributions_intractable():
    key = jax.random.PRNGKey(3)
    p = Gaussian(dim=2)
    
    # Mock a process distribution (log_prob raises NotImplementedError)
    class MockProcess(ProcessDistribution):
        def __init__(self, base):
            super().__init__(None, base)
        def sample(self, key):
            return self.base_distribution.sample(key)

    q = MockProcess(p)
    
    metrics = compare_distributions(p, q, key, num_samples=200)
    
    # SWD should be valid
    assert metrics["SWD"] < 0.5
    
    # Others should be inf (as set in results dict)
    assert jnp.isinf(metrics["Test_NLL"])
    assert jnp.isinf(metrics["KL"])

def test_swd_scaling():
    key = jax.random.PRNGKey(42)
    # Test SWD with larger dimensions and samples
    n, d, p = 1000, 50, 100
    X = jax.random.normal(key, (n, d))
    Y = jax.random.normal(key, (n, d)) + 1.0
    swd_shifted = compute_swd_jax(key, X, Y, num_projections=p)
    swd_identity = compute_swd_jax(key, X, X, num_projections=p)
    # The normalisation (sqrt(N * D)) makes the absolute value small in high
    # dimensions; assert the scale-free property instead, and that the result
    # stays finite as dimensionality/sample count grow.
    assert jnp.isfinite(swd_shifted)
    assert swd_shifted > swd_identity

def test_compare_distributions_scaling():
    key = jax.random.PRNGKey(43)
    dim = 20
    p = Gaussian(dim=dim)
    q = Gaussian(mean=jnp.ones(dim)*0.1, dim=dim)
    
    metrics = compare_distributions(p, q, key, num_samples=1000, num_projections=50)
    assert "SWD" in metrics
    assert not jnp.isnan(metrics["SWD"])
