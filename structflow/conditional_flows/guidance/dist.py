import jax
import jax.numpy as jnp
from jax import lax, random
from typing import Dict, List, Set, Optional, Union, Callable, Any
import equinox as eqx

from structflow.bayes.distribution import Distribution
from structflow.conditional_flows.dist import ConditionalDistribution
from structflow.conditional_flows.guidance.flow import GuidedFlow
from flow.utils import divergence_jvp


def _partial_divergence_jvp(fn, x, t, target_indices):
    """
    Compute partial divergence (trace of Jacobian) for only the specified indices.
    
    For conditional log probability p(targets | conditions), we only need
    the divergence contribution from target dimensions, as conditions are fixed.
    
    Args:
        fn: Velocity function v(x, t) -> v
        x: Full state vector
        t: Time
        target_indices: Array of indices for which to compute divergence
    
    Returns:
        Sum of d(v_i)/d(x_i) for i in target_indices
    """
    def single_partial_div(idx):
        # Create tangent vector with 1 at position idx, 0 elsewhere
        tangent = jnp.zeros_like(x).at[idx].set(1.0)
        # JVP gives (v(x,t), J @ tangent) where J is the Jacobian
        _, jvp_val = jax.jvp(lambda _x: fn(_x, t), (x,), (tangent,))
        # jvp_val[idx] = sum_j (dv_idx/dx_j * tangent_j) = dv_idx/dx_idx
        return jvp_val[idx]
    
    # Sum over target indices
    return jnp.sum(jax.vmap(single_partial_div)(target_indices))

class GuidanceBasedDistribution(ConditionalDistribution, eqx.Module):
    """
    A distribution that uses reconstruction guidance to support arbitrary 
    conditional queries p(targets | conditions) from a joint distribution.
    
    Wraps a GuidedFlow to perform integration and adds dictionary handling.
    """
    flow: GuidedFlow
    base_distribution: Distribution
    key_order: Optional[List[str]] = eqx.field(static=True, default=None)

    def _flatten(self, x: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        if self.key_order is None:
            return x # Assume dict model if no key_order provided
        try:
            return jnp.concatenate([x[k] for k in self.key_order], axis=-1)
        except KeyError as e:
            # BUG here???
            raise ValueError(f"GuidanceBasedDistribution.log_prob requires all joint variables for backward integration. Missing: {e}")

    def _unflatten(self, x_flat: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        if self.key_order is None:
            return x_flat
        
        dims = self.base_distribution.dim_dict
        x_dict = {}
        start = 0
        for k in self.key_order:
            d = dims[k] if isinstance(dims[k], int) else dims[k][0]
            if isinstance(d, tuple): d = d[0]
            
            x_dict[k] = x_flat[..., start:start+d]
            start += d
        return x_dict

    def sample(self,
               key: jax.Array,
               conditions: Optional[Dict[str, jnp.ndarray]] = None,
               targets: Optional[Union[Set[str], List[str]]] = None) -> Dict[str, jnp.ndarray]:
        """
        Sample from p(targets | conditions) using reconstruction guidance.
        """
        all_vars = set(self.base_distribution.dim_dict.keys()) if hasattr(self.base_distribution, 'dim_dict') else set()
        target_set, cond_keys = self._validate_targets_conditions(targets, conditions, all_vars) if all_vars else (set(targets) if targets else set(), set(conditions.keys()) if conditions else set())

        # Sample base noise
        x0_dict = self.base_distribution.sample(key)
        x0 = self._flatten(x0_dict)

        # Prepare flow with unflatten capability
        flow = eqx.tree_at(lambda f: f.unflatten_fn, self.flow, self._unflatten)
        
        # Integrate
        # Pass conditions directly as dict (or empty dict)
        cond_dict = conditions if conditions else {}
        x1_out = flow.forward(x0, conditions=cond_dict)
        x1 = self._unflatten(x1_out)
        
        # Filter targets
        if targets is not None:
            return {k: x1[k] for k in target_set if k in x1}
        return x1

    def log_prob(self,
                 x: Dict[str, jnp.ndarray],
                 targets: Optional[Union[Set[str], List[str]]] = None,
                 conditions: Optional[Union[Set[str], List[str]]] = None) -> jnp.ndarray:
        """
        Evaluate density p(targets | conditions) via guided ODE integration.
        """
        all_vars = set(self.base_distribution.dim_dict.keys()) if hasattr(self.base_distribution, 'dim_dict') else set(x.keys())
        
        # Extract condition values
        if conditions is not None:
            cond_dict = {k: x[k] for k in conditions if k in x}
        else:
            cond_dict = {}

        target_set, cond_keys = self._validate_targets_conditions(targets, conditions, all_vars)

        # Joint model check: log_prob requires full state
        if self.key_order is not None:
            provided = target_set | cond_keys
            required = set(self.key_order)
            if not required.issubset(provided):
                missing = required - provided
                raise NotImplementedError(
                    f"GuidanceBasedDistribution (joint mode) requires all variables to evaluate log_prob. "
                    f"Missing: {missing}. Marginal density evaluation is not supported by unstructured joint flows."
                )

        # 1. Setup Guided Flow Components
        flow = eqx.tree_at(lambda f: f.unflatten_fn, self.flow, self._unflatten)
        bounded_vel = flow.get_velocity_fn(cond_dict)

        x_in = self._flatten(x)

        # 2. Integrate Backward
        traj_bwd = flow.trajectory_backward(x_in, conditions=cond_dict)
        
        x0_vec = traj_bwd[-1]
        x0 = self._unflatten(x0_vec)

        # 3. Base Log Prob, over the target blocks only (the divergence below is too);
        # the condition blocks' noise is not part of p(targets | conditions).
        log_p_base = sum(self.base_distribution.marginal_log_prob(x0, k)
                         for k in self.key_order if k in target_set)

        # 4. Divergence Accumulation - only for target dimensions
        # Compute target indices in the flattened vector
        dims = self.base_distribution.dim_dict
        target_indices = []
        start = 0
        for k in self.key_order:
            d = dims[k] if isinstance(dims[k], int) else dims[k][0]
            if isinstance(d, tuple): d = d[0]
            if k in target_set:
                target_indices.extend(range(start, start + d))
            start += d
        target_indices = jnp.array(target_indices)
        
        ts = jnp.linspace(1, 0, self.flow.n_steps + 1)
        dt = ts[1] - ts[0]
        
        def div_accumulator(i, acc):
            t = ts[i]
            xt = traj_bwd[i]
            # Only compute divergence for target dimensions
            # div = sum of d(v_i)/d(x_i) for i in target_indices
            div_val = _partial_divergence_jvp(bounded_vel, xt, t, target_indices)
            return acc + div_val * dt

        delta_log_p = lax.fori_loop(0, self.flow.n_steps, div_accumulator, 0.0)
        
        return log_p_base + delta_log_p

    def transform_to_noise(self, x: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """
        Maps factual data at t=1 back to noise space at t=0 using unconditional flow.
        """
        x_vec = self._flatten(x)
        # For abduction, we assume the data came from the joint distribution
        flow = eqx.tree_at(lambda f: f.unflatten_fn, self.flow, self._unflatten)
        z_vec = flow.backward(x_vec, conditions={})
        return self._unflatten(z_vec)

    def counterfactual(self,
                       data: Dict[str, jnp.ndarray],
                       interventions: Dict[str, jnp.ndarray],
                       targets: Optional[Union[Set[str], List[str]]] = None) -> tuple:
        """
        Approximate counterfactual for guidance-based flows.
        Returns both the sample and its log probability.
        
        1. Abduction: Map factual data back to noise space (unconditionally).
        2. Prediction: Integrate forward from that noise with guidance for interventions.
        
        Returns:
            (sample_dict, log_prob) where sample_dict is the counterfactual sample
            and log_prob is the log probability of the sample.
        """
        all_vars = set(self.base_distribution.dim_dict.keys())
        intervened_keys = set(interventions.keys())

        provided = set(data.keys()) | set(interventions.keys())
        required = all_vars
        if not required.issubset(provided):
            missing = required - provided
            raise NotImplementedError(
                f"GuidanceBasedDistribution (joint mode) requires all variables to evaluate log_prob. "
                f"Missing: {missing}. Marginal density evaluation is not supported by unstructured joint flows."
            )
        
        # Non-intervened targets: these are the blocks whose density we compute
        non_intervened_targets = all_vars - intervened_keys

        # Determine target blocks for log_prob computation
        if targets is None:
            target_set = non_intervened_targets
        else:
            target_set = set(targets)
        
        # Verify we are asking for all non-intervened variables (no marginals).
        # The density below is *always* the joint over `non_intervened_targets`
        # (see the use of `non_intervened_targets` when building target_indices),
        # so requesting a strict subset would silently return the joint density
        # while a wrapper (e.g. NormalisedDistribution) applies a Jacobian
        # correction for only the requested subset — a mismatch that previously
        # surfaced as negative KL. Fail loudly instead of returning a wrong value.
        if target_set != non_intervened_targets:
            raise NotImplementedError(
                f"GuidanceBasedDistribution (joint mode) computes the counterfactual density "
                f"jointly over all non-intervened variables {sorted(non_intervened_targets)}. "
                f"Requested targets {sorted(target_set)} differ from this set; marginal "
                f"counterfactual density (a subset, or targets overlapping interventions) is "
                f"not supported by unstructured joint flows."
            )

        # 1. Abduction
        x0_dict = self.transform_to_noise(data)
        x0_vec = self._flatten(x0_dict)
        
        # 2. Prediction - get trajectory for divergence accumulation
        flow = eqx.tree_at(lambda f: f.unflatten_fn, self.flow, self._unflatten)
        bounded_vel = flow.get_velocity_fn(interventions)
        traj_fwd = flow.trajectory_forward(x0_vec, conditions=interventions)
        x1_cf_vec = traj_fwd[-1]  # t=1 is the last element (traj goes from t=0 to t=1)
        x1_cf_dict = self._unflatten(x1_cf_vec)
        
        # 3. Compute log probability - only for non-intervened target dimensions
        # Compute target indices in the flattened vector
        dims = self.base_distribution.dim_dict
        target_indices = []
        start = 0
        for k in self.key_order:
            d = dims[k] if isinstance(dims[k], int) else dims[k][0]
            if isinstance(d, tuple): d = d[0]
            if k in non_intervened_targets:
                target_indices.extend(range(start, start + d))
            start += d
        target_indices = jnp.array(target_indices)
        
        ts = jnp.linspace(0, 1, self.flow.n_steps + 1)
        dt = ts[1] - ts[0]  # dt is positive for forward
        
        def div_accumulator(i, acc):
            t = ts[i]
            xt = traj_fwd[i]
            # Only compute divergence for non-intervened target dimensions
            div_val = _partial_divergence_jvp(bounded_vel, xt, t, target_indices)
            return acc + div_val * dt
        
        delta_log_p = lax.fori_loop(0, self.flow.n_steps, div_accumulator, 0.0)
        
        # 4. Base Log Prob (at t=0) - only for non-intervened targets
        log_p_base = 0.0
        for k in non_intervened_targets:
            log_p_base += self.base_distribution.marginal_log_prob(x0_dict, k)
        
        # For forward integration: log p(x_1) = log p(x_0) - ∫₀¹ div(v) dt
        log_prob = log_p_base - delta_log_p
        
        # 5. Finalize
        full_res = {**x1_cf_dict, **interventions}
        
        if targets is not None:
            sample_dict = {k: full_res[k] for k in target_set if k in full_res}
        else:
            sample_dict = full_res
            
        return sample_dict, log_prob
