import jax
import jax.numpy as jnp
from typing import Dict, List, Set, Optional, Union, Tuple
import equinox as eqx
import blackjax
from structflow.pgms.dist import ConditionalDistribution


# --- MCMC Helper ---
def run_nuts(log_prob_fn, initial_position, key,
             num_samples=1000,
             num_steps_adaptation=250,
             max_num_doublings=7,
             target_accept_prob=0.7,
             thin_factor=1):
    """
    Run NUTS sampler with hyperparameters for cost/accuracy trade-off.
    
    Args:
        log_prob_fn: Log probability function to sample from.
        initial_position: Starting position for the sampler (dict).
        key: JAX random key.
        num_samples: Total number of samples to draw (default=1000).
                     ACCURACY: Higher = better estimates. COST: Linear increase.
        num_steps_adaptation: Number of warmup/adaptation steps (default=250).
                             ACCURACY: Higher = better adapted sampler. COST: Linear increase.
        max_num_doublings: Max trajectory length doubling iterations in NUTS (default=7).
                          ACCURACY: Higher = more careful exploration. COST: Exponential ~2^doublings.
        target_accept_prob: Target acceptance probability during adaptation (default=0.7).
                           ACCURACY: Higher = smaller steps, better exploration. COST: More gradients/evals.
        thin_factor: Keep every Nth sample, discard others (default=1, no thinning).
                    ACCURACY: Higher = reduces autocorrelation. COST: Produces fewer independent samples.
    
    Returns:
        Dictionary of samples with shape (num_effective_samples, *var_dims).
    
    Cost/Accuracy Examples:
        - Quick pilot: num_steps_adaptation=100, num_samples=200, max_num_doublings=8
        - Balanced: num_steps_adaptation=250, num_samples=1000, max_num_doublings=7 (default)
        - Accurate: num_steps_adaptation=1000, num_samples=5000, max_num_doublings=15
        - Ultra-fast: num_samples=100, num_steps_adaptation=100, thin_factor=1
    """
    # Adaptation phase with window adaptation
    window = blackjax.window_adaptation(blackjax.nuts, log_prob_fn)
    (state, parameters), _ = window.run(key, initial_position, num_steps=num_steps_adaptation)
    
    # Override max_num_doublings if provided (controls trajectory length)
    parameters = dict(parameters)
    if 'max_num_doublings' in parameters:
        parameters['max_num_doublings'] = max_num_doublings
    
    # Override target_accept_prob if supported by the sampler
    if 'target_accept_prob' in parameters:
        parameters['target_accept_prob'] = target_accept_prob
    
    def inference_loop(state, rng_key):
        state, _ = blackjax.nuts(log_prob_fn, **parameters).step(rng_key, state)
        return state, state.position

    # Sample collection with thinning applied post-hoc
    total_samples = num_samples * thin_factor  # Generate more, then thin
    keys = jax.random.split(key, total_samples)
    _, all_samples = jax.lax.scan(inference_loop, state, keys)
    
    # Apply thinning: keep every thin_factor-th sample
    if thin_factor > 1:
        # Select every thin_factor-th sample
        indices = jnp.arange(0, total_samples, thin_factor)
        thinned_samples = jax.tree_util.tree_map(
            lambda x: x[indices] if eqx.is_array(x) else x,
            all_samples
        )
        return thinned_samples
    
    return all_samples


class BasePGM(ConditionalDistribution, eqx.Module):
    """
    Standard base class for Probabilistic Graphical Model benchmarks.
    Inherits from ConditionalDistribution to provide a standard interface for
    sampling and log_prob, and eqx.Module for JAX compatibility.
    """
    
    def sample(self, 
               key: jax.Array, 
               conditions: Optional[Dict[str, jnp.ndarray]] = None,
               targets: Optional[Union[Set[str], List[str]]] = None) -> Dict[str, jnp.ndarray]:
        """
        Sample from p(targets | conditions).
        If targets is None, samples all variables.
        If conditions is None, samples from the joint.
        """
        raise NotImplementedError
        
    def log_prob(self, 
                 data: Dict[str, jnp.ndarray], 
                 targets: Optional[Union[Set[str], List[str]]] = None,
                 conditions: Optional[Union[Set[str], List[str]]] = None) -> jnp.ndarray:
        """
        Compute log p(targets | conditions).
        If targets and conditions are both None, computes joint log_prob.
        """
        raise NotImplementedError
        
    def get_pgm_structure(self) -> Dict[str, List[str]]:
        """
        Returns the topological structure of the PGM.
        Map of node_name -> [parent_names].
        """
        raise NotImplementedError

    def get_inference_targets_and_conditions(self) -> (Set[str], Set[str]):
        """
        Returns a default set of targets and conditions for this PGM.
        Usually represents a typical inference task.
        """
        raise NotImplementedError

    @property
    def variable_dims(self) -> Dict[str, int]:
        """
        Returns a dictionary mapping variable names to their event dimensions.
        """
        raise NotImplementedError

    def get_supported_distributions(self) -> List[Dict[str, frozenset[str]]]:
        """
        Returns a list of distributions supported by this PGM.
        Each entry is a dict with 'targets' and 'conditions' keys.
        """
        # Default: Joint and Default Posteriors
        dists = []
        
        # Joint
        all_vars = frozenset(self.variable_dims.keys())
        dists.append({"targets": all_vars, "conditions": frozenset()})
        
        # Inference
        try:
            t, c = self.get_inference_targets_and_conditions()
            dists.append({"targets": frozenset(t), "conditions": frozenset(c)})
        except NotImplementedError:
            pass
            
        return dists

def format_dist_str(targets: Union[Set[str], frozenset[str]], 
                    conditions: Optional[Union[Set[str], frozenset[str]]] = None) -> str:
    """
    Utility for turning targets/conditions into a human readable string p(targets | conditions).
    """
    t_list = sorted(list(targets))
    t_str = ", ".join(t_list)
    if not conditions:
        return f"p({t_str})"
    c_list = sorted(list(conditions))
    c_str = ", ".join(c_list)
    return f"p({t_str} | {c_str})"
