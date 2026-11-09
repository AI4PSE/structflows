import jax
import jax.numpy as jnp
from jax import vmap, random
from typing import Dict, List, Set, Optional, Union
import equinox as eqx
from structflow.bayes.distribution import Distribution
from structflow.conditional_flows.utils import DictNormaliser
from structflow.pgms.structural_pgm import StructuralPGM

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
               targets: Optional[Set[str]] = None) -> Dict[str, jnp.ndarray]:
        """
        Sample from p(targets | conditions).
        """
        raise NotImplementedError

    def log_prob(self,
                 x: Dict[str, jnp.ndarray],
                 targets: Optional[Set[str]] = None,
                 conditions: Optional[Set[str]] = None) -> jnp.ndarray:
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
                       targets: Optional[Set[str]] = None,
                       compute_log_prob: bool = True) -> Union[Dict[str, jnp.ndarray], tuple]:
        """
        Perform counterfactual query: p(targets | data, do(interventions)) for a single example.
        
        Returns:
            Either Dict[str, jnp.ndarray] of samples (legacy API) or
            tuple of (Dict[str, jnp.ndarray], jnp.ndarray) of (samples, log_prob) (new API).
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
                         targets: Optional[Set[str]] = None,
                         **kwargs) -> Union[Dict[str, jnp.ndarray], tuple]:
        """
        Batched counterfactual query.
        Returns:
            - Dict if counterfactual returns only samples (legacy API)
            - tuple of (Dict, Array) if counterfactual returns (samples, log_probs) (new API)
        """
        def _cf_wrapper(d, i):
            result = self.counterfactual(d, i, targets=targets)
            return result
        
        result = vmap(_cf_wrapper)(data, interventions)
        
        # result is either:
        #   - Dict[str, Array] where each Array has leading batch dimension
        #   - tuple of (Dict[str, Array], Array) where both have leading batch dimension
        return result


class StructuredPGMDistribution(ConditionalDistribution):
    """
    A ConditionalDistribution that wraps a StructuralPGM.
    
    This allows any StructuralPGM to be used as a ConditionalDistribution,
    enabling it to be used in flows, normalisers, and other wrappers.
    """
    pgm: StructuralPGM
    
    def sample(self,
               key: jax.Array,
               conditions: Optional[Dict[str, jnp.ndarray]] = None,
               targets: Optional[Set[str]] = None) -> Dict[str, jnp.ndarray]:
        """
        Sample from p(targets | conditions) using the PGM's sampling logic.
        """
        return self.pgm.sample(key, conditions=conditions, targets=targets)

    def log_prob(self,
                 x: Dict[str, jnp.ndarray],
                 targets: Optional[Set[str]] = None,
                 conditions: Optional[Set[str]] = None) -> jnp.ndarray:
        """
        Compute log p(targets | conditions) using the PGM's log_prob logic.
        """
        return self.pgm.log_prob(x, targets=targets, conditions=conditions)

    def transform_to_noise(self, x: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """
        Inverse flow: Maps Data -> Noise.
        Computes z_j = f_j^{-1}(x_j, pa_j).
        """
        noises = {}
        for node in self.pgm.topo_order:
            parents_flat = self.pgm._get_parents_flat(node, x)
            x_val = jnp.atleast_1d(x[node])
            noise, _ = self.pgm.mechanisms[node].inverse(parents_flat, x_val)
            noises[node] = noise
        return noises

    def counterfactual(self,
                       data: Dict[str, jnp.ndarray],
                       interventions: Dict[str, jnp.ndarray],
                       targets: Optional[Set[str]] = None) -> tuple:
        """
        Performs a counterfactual query using the PGM's logic.
        """
        return self.pgm.counterfactual(data, interventions, targets=targets)


class NormalisedDistribution(ConditionalDistribution):
    """
    A distribution that normalises its inputs before passing them to an inner distribution.
    """
    dist: ConditionalDistribution # The inner distribution (built from normalised data)
    normaliser: DictNormaliser

    def log_prob(self,
                 x: Dict[str, jnp.ndarray],
                 targets: Optional[Set[str]] = None,
                 conditions: Optional[Set[str]] = None) -> jnp.ndarray:
        
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
               targets: Optional[Set[str]] = None) -> Dict[str, jnp.ndarray]:
        
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
                       targets: Optional[Set[str]] = None,
                       compute_log_prob: bool = True) -> Union[Dict[str, jnp.ndarray], tuple]:
        """
        Perform counterfactual query with optional log probability.
        Returns both sample and log probability if inner distribution supports new API.
        
        The log probability includes corrections for normalization:
        - log_det_jacobian ONLY for non-intervened targets
        
        Note: Intervened variables are fixed by do() and don't contribute to the probability.
        The inner distribution's counterfactual() only computes log_prob for non-intervened
        targets, so we must match that by only applying Jacobian correction to those same
        variables.
        """
        # 1. Normalise inputs
        data_norm = self.normaliser.normalise(data)
        interventions_norm = self.normaliser.normalise(interventions)
        
        # 2. Compute inner counterfactual
        if compute_log_prob:
            cf_norm, log_prob_norm = self.dist.counterfactual(data_norm, interventions_norm, targets=targets)
        else:
            # Samples-only path (supported by static dists): skips the O(D^2)
            # divergence accumulation, log_prob comes back as NaN.
            cf_norm, log_prob_norm = self.dist.counterfactual(
                data_norm, interventions_norm, targets=targets, compute_log_prob=False)
         
        # 3. Unnormalise outputs
        cf = self.normaliser.unnormalise(cf_norm)
        
        # 4. Add log_det Jacobian corrections ONLY for non-intervened targets.
        # The inner distribution only computes probability for non-intervened
        # targets, so we must match that by only applying the Jacobian correction
        # to those same variables. When `targets` is None it defaults to *all*
        # variables, so we must resolve it explicitly (rather than passing None
        # to log_det_jacobian, which would sum over the intervened variables too
        # and reintroduce the mismatch this correction exists to avoid).
        interv_keys = set(interventions.keys())
        all_vars = set(self.normaliser.normalisers.keys())
        target_set = set(targets) if targets is not None else all_vars
        non_intervened_targets = target_set - interv_keys

        log_det_cf = (self.normaliser.log_det_jacobian(non_intervened_targets)
                      if non_intervened_targets else 0.0)

        log_prob = log_prob_norm + log_det_cf
        return cf, log_prob
