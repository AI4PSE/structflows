import abc
import jax
import jax.numpy as jnp
from jax import vmap, random, jit
from jax.scipy.stats import multivariate_normal
from jax.scipy.special import logsumexp
import equinox as eqx
from typing import Optional, Any, Dict

class Distribution(eqx.Module, abc.ABC):
    """Abstract base class for probability distributions."""

    @abc.abstractmethod
    def sample(self, key):
        """Samples from the distribution."""
        raise NotImplementedError

    @abc.abstractmethod
    def log_prob(self, x):
        """Computes the log probability of a sample."""
        raise NotImplementedError

    def prob(self, x):
        return jnp.exp(self.log_prob(x))

    # --- Batched / JIT-compiled Entry Points ---
    @eqx.filter_jit
    def b_prob(self, x):
        return vmap(self.prob)(x)
    
    @eqx.filter_jit
    def b_log_prob(self, x):
        return vmap(self.log_prob)(x)

    # @eqx.filter_jit
    def b_sample(self, key, n_samples: int):
        keys = random.split(key, n_samples)
        return vmap(self.sample)(keys)

    def entropy(self, key, n_samples: int = 1000):
        samples = self.b_sample(key, n_samples)
        log_probs = self.b_log_prob(samples)
        return -jnp.mean(log_probs)

class Gaussian(Distribution):
    """A standard multivariate Gaussian distribution N(0, I)."""
    mean: jnp.ndarray
    cov: jnp.ndarray
    dim: int = eqx.field(static=True)

    def __init__(self, dim: Optional[int] = None, mean: Optional[jnp.ndarray] = None, cov: Optional[jnp.ndarray] = None):
        if mean is not None:
            self.mean = jnp.asarray(mean)
            self.dim = self.mean.shape[0]
            if cov is not None:
                self.cov = jnp.asarray(cov)
            else:
                self.cov = jnp.eye(self.dim)
        else:
            self.dim = dim
            self.mean = jnp.zeros(dim)
            self.cov = jnp.eye(dim)

    def sample(self, key):
        return jax.random.multivariate_normal(key, self.mean, self.cov)

    def log_prob(self, x):
        return multivariate_normal.logpdf(x, self.mean, self.cov)

    @property
    def entropy_analytic(self):
        return 0.5 * self.dim * (1.0 + jnp.log(2 * jnp.pi)) + 0.5 * jnp.linalg.slogdet(self.cov)[1]


# Helper function stays the same, just ensured it's functional
def gmm_log_p(means, covs, log_weights, x):
    x = jnp.atleast_2d(x)
    log_probs_all_components = jax.vmap(
        lambda point: multivariate_normal.logpdf(point, means, covs)
    )(x)
    weighted_log_probs = log_probs_all_components + log_weights
    log_p = logsumexp(weighted_log_probs, axis=1)
    return jnp.squeeze(log_p)

class GaussianMixture(Distribution):
    """A Gaussian Mixture Model (GMM) distribution."""
    weights: jnp.ndarray
    means: jnp.ndarray
    covs: jnp.ndarray
    log_weights: jnp.ndarray
    n_components: int = eqx.field(static=True)
    dim: int = eqx.field(static=True)

    def __init__(self, weights, means, covs):
        self.n_components, self.dim = means.shape
        self.weights = weights
        self.means = means
        self.covs = covs
        self.log_weights = jnp.log(weights)

    def sample(self, key):
        key_choice, key_sample = jax.random.split(key)
        component_index = jax.random.choice(
            key_choice, self.n_components, p=self.weights
        )
        chosen_mean = self.means[component_index]
        chosen_cov = self.covs[component_index]
        return jax.random.multivariate_normal(key_sample, chosen_mean, chosen_cov)

    def log_prob(self, x):
        return gmm_log_p(self.means, self.covs, self.log_weights, x)

class FlowDistribution(Distribution):
    """A distribution defined by a deterministic Flow transformation."""
    flow: Any
    base_distribution: Distribution

    def sample(self, key):
        x0 = self.base_distribution.sample(key)
        y = self.flow.forward(x0)
        return y

    def log_prob(self, x1):
        x0 = self.flow.backward(x1)
        log_p_x0 = self.base_distribution.log_prob(x0)
        # Assuming flow.push_forward_log_prob returns (log_prob, other_stuff)
        log_p_x1, _ = self.flow.push_forward_log_prob(log_p_x0, x0)
        return log_p_x1

class ProcessDistribution(Distribution):
    process: Any
    base_distribution: Distribution

    def sample(self, key):
        key1, key2 = jax.random.split(key)
        x0 = self.base_distribution.sample(key1)
        y = self.process.forward(x0, key2)[0]
        return y

    def log_prob(self, x):
        raise NotImplementedError("Intractable.")


class DictDistribution(Distribution):
    """
    A set of distributions over a dictionary of variables.
    """
    base_distributions: Dict[str, Distribution]

    @property
    def dim_dict(self):
        return {k: d.dim for k, d in self.base_distributions.items()}

    def sample(self, key):
        keys = jax.random.split(key, len(self.base_distributions))
        return {
            k: dist.sample(subkey) 
            for (k, dist), subkey in zip(self.base_distributions.items(), keys)
        }

    def log_prob(self, x):
        log_probs = [dist.log_prob(x[k]) for k, dist in self.base_distributions.items()]
        return jnp.sum(jnp.stack(log_probs))

    def marginal_log_prob(self, x, k):
        """Log density of the single variable k (the variables are independent)."""
        return self.base_distributions[k].log_prob(x[k])
    

class IsotropicLaplace(Distribution):
    r"""
    Simple Isotropic Laplace distribution: p(z) = \prod 0.5 * exp(-|z|)
    Used as the prior for CAREFL to ensure identifiability.
    """
    dim: int = eqx.field(static=True)

    def __init__(self, dim):
        self.dim = dim

    def log_prob(self, z: jnp.ndarray) -> jnp.ndarray:
        # log(0.5 * exp(-|x|)) = log(0.5) - |x|
        return jnp.sum(jnp.log(0.5) - jnp.abs(z))

    def sample(self, key: jax.Array) -> jnp.ndarray:
        return random.laplace(key, shape=(self.dim,))

