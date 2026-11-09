import jax
import jax.numpy as jnp
from jax import vmap, jit
import equinox as eqx
import blackjax
from typing import Dict, List, Set, Optional, Union, Callable, Any

from structflow.bayes.distribution import Distribution

def run_mcmc(log_prob_fn: Callable, 
             initial_position: Union[Dict[str, jnp.ndarray], jnp.ndarray], 
             key: jax.Array, 
             num_samples: int = 2000, 
             warmup_steps: int = 1000):
    """
    Utility for running MCMC sampling using BlackJAX NUTS.
    
    Args:
        log_prob_fn: Function that takes (params) and returns log-probability.
        initial_position: Starting point for the chains.
        key: JAX random key.
        num_samples: Number of samples to draw.
        warmup_steps: Number of warmup/adaptation steps.
        
    Returns:
        Samples from the posterior.
    """
    warmup = blackjax.window_adaptation(blackjax.nuts, log_prob_fn)
    adaptation_results, _ = warmup.run(key, initial_position, num_steps=warmup_steps)
    state = adaptation_results.state
    kernel = blackjax.nuts(log_prob_fn, **adaptation_results.parameters).step
    
    def inference_loop(state, key):
        state, _ = kernel(key, state)
        return state, state

    keys = jax.random.split(key, num_samples)
    _, states = jax.lax.scan(inference_loop, state, keys)
    
    return states.position

class SimulationBasedPosterior(Distribution):
    """
    Posterior distribution using Sampling Importance Resampling (SIR).
    Approximates the posterior by weighting prior samples by their likelihood.
    """
    prior: Distribution
    log_likelihood_fn: Callable = eqx.field(static=True)
    y: jnp.ndarray
    proposal_factor: int = eqx.field(static=True)

    def __init__(self, prior_distribution: Distribution, log_likelihood_fn: Callable, y_observed: jnp.ndarray, proposal_factor: int = 100):
        self.prior = prior_distribution
        self.log_likelihood_fn = log_likelihood_fn
        self.y = y_observed
        self.proposal_factor = proposal_factor

    def sample(self, key: jax.Array):
        return self.b_sample(key, 1)[0]
    
    def log_prob(self, x):
        raise NotImplementedError("SimulationBasedPosterior does not have a tractable log_prob.")

    @eqx.filter_jit
    def b_sample(self, key: jax.Array, n_samples: int):
        num_proposals = n_samples * self.proposal_factor
        key_prior, key_categorical = jax.random.split(key)

        prior_samples = self.prior.b_sample(key_prior, num_proposals)
        log_likelihoods = vmap(self.log_likelihood_fn, in_axes=(0, None))(prior_samples, self.y)

        # Handle potential NaNs/Infs in likelihood
        log_likelihoods = jnp.where(jnp.isnan(log_likelihoods), -jnp.inf, log_likelihoods)

        selected_indices = jax.random.categorical(key_categorical, log_likelihoods, shape=(n_samples,))
        
        return jax.tree_util.tree_map(lambda x: x[selected_indices], prior_samples)

class MCMCPosterior(Distribution):
    """
    Posterior distribution produced via MCMC sampling (NUTS).
    """
    prior: Distribution
    log_likelihood_fn: Callable = eqx.field(static=True)
    y: jnp.ndarray
    warmup_steps: int = eqx.field(static=True)

    def __init__(self, prior_distribution: Distribution, log_likelihood_fn: Callable, y_observed: jnp.ndarray, warmup_steps: int = 1000):
        self.prior = prior_distribution
        self.log_likelihood_fn = log_likelihood_fn
        self.y = y_observed
        self.warmup_steps = warmup_steps

    def sample(self, key: jax.Array):
        return self.b_sample(key, 1)[0]

    def log_prob(self, x):
        raise NotImplementedError("MCMCPosterior log_prob is not explicitly available.")

    @eqx.filter_jit
    def b_sample(self, key: jax.Array, n_samples: int):
        def log_prob_fn(params):
            lp_prior = self.prior.log_prob(params)
            lp_lik = self.log_likelihood_fn(params, self.y)
            return lp_prior + lp_lik

        key_init, key_mcmc = jax.random.split(key)
        initial_position = self.prior.sample(key_init)
        
        samples = run_mcmc(log_prob_fn, initial_position, key_mcmc, n_samples, self.warmup_steps)
        return samples
