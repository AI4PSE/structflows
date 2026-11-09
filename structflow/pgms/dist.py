import jax
import jax.numpy as jnp
from jax import vmap, random
from typing import Dict, List, Set, Optional, Union
import equinox as eqx
from structflow.bayes.distribution import Distribution
from structflow.conditional_flows.utils import DictNormaliser

class ConditionalDistribution(Distribution, eqx.Module):
    """
    Abstract base class for conditional distributions.
    
    Provides a standard API for:
    - log_prob(x, targets, conditions)
    - sample(key, targets, conditions)
    - Batched versions of the above (b_log_prob, b_sample)
    """
    
    def sample(self, 
               key: jax.Array, 
               conditions: Optional[Dict[str, jnp.ndarray]] = None,
               targets: Optional[Union[Set[str], List[str]]] = None) -> Dict[str, jnp.ndarray]:
        """
        Sample from p(targets | conditions).
        """
        raise NotImplementedError

    def log_prob(self, 
                 x: Dict[str, jnp.ndarray], 
                 targets: Optional[Union[Set[str], List[str]]] = None, 
                 conditions: Optional[Union[Set[str], List[str]]] = None) -> jnp.ndarray:
        """
        Compute log p(targets | conditions) for a single example.
        """
        raise NotImplementedError

    def transform_to_noise(self, x: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """
        Maps Real Data -> Noise for a single example.
        """
        raise NotImplementedError

    def counterfactual(self, 
                       data: Dict[str, jnp.ndarray], 
                       interventions: Dict[str, jnp.ndarray],
                       targets: Optional[Set[str]] = None) -> Dict[str, jnp.ndarray]:
        """
        Perform counterfactual query: p(targets | data, do(interventions)) for a single example.
        """
        raise NotImplementedError

    def _validate_targets_conditions(self, targets, conditions, all_vars):
        """Helper to validate targets and conditions."""
        if conditions is not None:
            if isinstance(conditions, dict):
                cond_keys = set(conditions.keys())
            elif isinstance(conditions, (set, list, tuple, frozenset)):
                cond_keys = set(conditions)
            else:
                raise TypeError(f"conditions must be a dict or set/list of keys, got {type(conditions)}")
            
            unknown_conds = cond_keys - all_vars
            if unknown_conds:
                raise ValueError(f"Unknown blocks in conditions: {unknown_conds}")
        else:
            cond_keys = set()
        
        if targets is None:
            # Default: Sample everything not in conditions
            target_set = all_vars - cond_keys
        else:
            target_set = set(targets)
            unknown_targets = target_set - all_vars
            if unknown_targets:
                raise ValueError(f"Unknown blocks in targets: {unknown_targets}")
            
            # Check for overlap - a variable cannot be both target and condition
            overlap = target_set & cond_keys
            if overlap:
                raise ValueError(
                    f"Targets and conditions must be disjoint. "
                    f"Variable(s) appear in both: {overlap}. "
                    f"A variable cannot be both sampled (target) and observed (condition)."
                )
        return target_set, cond_keys

    # --- Batched API ---

    @eqx.filter_jit
    def b_sample(self, key, n_samples, conditions=None, targets=None):
        """
        Batched sampling.
        Assumes conditions are static over the batch (i.e. same condition for all samples).
        """
        keys = random.split(key, n_samples)
        # We map over keys only. conditions and targets are static.
        return vmap(lambda k: self.sample(k, conditions, targets))(keys)

    @eqx.filter_jit
    def b_log_prob(self, x, targets=None, conditions=None):
        """
        Batched log probability.
        Assumes targets and conditions structure is static.
        x is batched.
        """
        fn = lambda _x: self.log_prob(_x, targets, conditions)
        return vmap(fn)(x)

    @eqx.filter_jit
    def b_transform_to_noise(self, x: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """
        Batched transform to noise.
        """
        return vmap(self.transform_to_noise)(x)

    @eqx.filter_jit
    def b_counterfactual(self, 
                         data: Dict[str, jnp.ndarray], 
                         interventions: Dict[str, jnp.ndarray],
                         targets: Optional[Set[str]] = None) -> Dict[str, jnp.ndarray]:
        """
        Batched counterfactual query.
        """
        return vmap(lambda d, i: self.counterfactual(d, i, targets=targets))(data, interventions)


class NormalisedDistribution(ConditionalDistribution):
    """
    A distribution that normalises its inputs before passing them to an inner distribution.
    """
    dist: ConditionalDistribution # The inner distribution (built from normalised data)
    normaliser: DictNormaliser

    def log_prob(self, 
                 x: Dict[str, jnp.ndarray], 
                 targets: Optional[Union[Set[str], frozenset, List[str]]] = None, 
                 conditions: Optional[Union[Set[str], frozenset, List[str]]] = None) -> jnp.ndarray:
        
        # 1. Normalise inputs
        # DictNormaliser automatically handles that 'x' contains both targets and conditions
        x_norm = self.normaliser.normalise(x)
        
        # 2. Compute inner log prob
        lp_norm = self.dist.log_prob(x_norm, targets, conditions)
        
        # 3. Add Jacobian correction for TARGETS only
        return lp_norm + self.normaliser.log_det_jacobian(targets)

    def sample(self, 
               key: jax.Array, 
               conditions: Optional[Dict[str, jnp.ndarray]] = None,
               targets: Optional[Union[Set[str], frozenset]] = None) -> Dict[str, jnp.ndarray]:
        
        # 1. Normalise conditions (if any)
        cond_norm = None
        if conditions is not None:
            cond_norm = self.normaliser.normalise(conditions)
            
        # 2. Sample from inner flow
        samples_norm = self.dist.sample(key, targets=targets, conditions=cond_norm)
        
        # 3. Unnormalise outputs
        return self.normaliser.unnormalise(samples_norm)
    
    def transform_to_noise(self, x: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """
        Maps Real Data -> Noise for a single example. 
        """
        x_norm = self.normaliser.normalise(x)
        # Run Flow Backward (in Normalised Space)
        z = self.dist.transform_to_noise(x_norm)
        # Return z (It is N(0,1), no unnormalisation needed)
        return z

    def counterfactual(self, 
                       data: Dict[str, jnp.ndarray], 
                       interventions: Dict[str, jnp.ndarray],
                       targets: Optional[Set[str]] = None) -> Dict[str, jnp.ndarray]:
        # 1. Normalise inputs
        data_norm = self.normaliser.normalise(data)
        interventions_norm = self.normaliser.normalise(interventions)
        
        # 2. Compute inner counterfactual
        cf_norm = self.dist.counterfactual(data_norm, interventions_norm, targets=targets)
        
        # 3. Unnormalise outputs
        return self.normaliser.unnormalise(cf_norm)
