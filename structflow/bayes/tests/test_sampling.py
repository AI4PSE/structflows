import pytest
import jax
import jax.numpy as jnp
from structflow.bayes.distribution import Gaussian
from structflow.bayes.sampling import SimulationBasedPosterior, MCMCPosterior

def test_simulation_based_posterior():
    key = jax.random.PRNGKey(0)
    dim = 1
    # Prior: N(0, 1)
    prior = Gaussian(dim=dim)
    
    # Likelihood: N(theta, 0.1^2)
    # y_obs = 3.0
    y_observed = jnp.array([3.0])
    sigma_lik = 0.1
    
    def log_likelihood_fn(theta, y):
        # theta is (dim,)
        return jax.scipy.stats.norm.logpdf(y, loc=theta, scale=sigma_lik).sum()

    sbp = SimulationBasedPosterior(prior, log_likelihood_fn, y_observed, proposal_factor=1000)
    
    samples = sbp.b_sample(key, 1000)
    
    # Check if samples are closer to 3.0 than 0.0
    mean_sample = jnp.mean(samples)
    # Analytical posterior mean: (y/sigma^2 + 0/1) / (1/sigma^2 + 1) = (3/0.01) / (100 + 1) = 300 / 101 approx 2.97
    assert jnp.abs(mean_sample - 2.97) < 0.2

def test_mcmc_posterior():
    key = jax.random.PRNGKey(42)
    dim = 1
    prior = Gaussian(dim=dim)
    
    y_observed = jnp.array([3.0])
    sigma_lik = 0.1
    
    def log_likelihood_fn(theta, y):
        # theta is (dim,)
        return jax.scipy.stats.norm.logpdf(y, loc=theta, scale=sigma_lik).sum()

    mcmc_post = MCMCPosterior(prior, log_likelihood_fn, y_observed, warmup_steps=500)
    
    # MCMCPosterior.b_sample returns a batch of samples
    samples = mcmc_post.b_sample(key, 1000)
    
    mean_sample = jnp.mean(samples)
    # Analytical posterior mean approx 2.97
    assert jnp.abs(mean_sample - 2.97) < 0.1
    
    # Analytical variance: 1 / (1/sigma^2 + 1) = 1/101 approx 0.0099
    # std approx 0.1
    std_sample = jnp.std(samples)
    assert jnp.abs(std_sample - 0.1) < 0.05
