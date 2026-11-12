import jax
import jax.numpy as jnp
from jax import random, lax, vmap
import equinox as eqx
from typing import Dict, List, Optional, Union, Set
from structflow.bayes.distribution import Distribution
from structflow.conditional_flows.dist import ConditionalDistribution
from flow.utils import divergence_jvp
from structflow.conditional_flows.dynamic.parallel_flow import ParallelFlow
from structflow.conditional_flows.dynamic.block_extractor import BlockBasedVelocityFn

def clip_by_norm(x, max_norm, axis=-1):
    """Clips the norm of a tensor x to max_norm."""
    norm = jnp.linalg.norm(x, axis=axis, keepdims=True)
    # Avoid division by zero
    factor = jnp.minimum(1.0, max_norm / (norm + 1e-6))
    return x * factor

class ParallelBlockTriangularDistribution(ConditionalDistribution, eqx.Module):
    """
    Simplified version of ParallelBlockTriangularDistribution.
    Instead of frozen trajectories, it uses backward integration to t=0 noise.
    """
    flow: ParallelFlow
    base_distributions: Dict[str, Distribution]
    
    def __init__(self, flow: ParallelFlow, base_distributions: Dict[str, Distribution]):
        self.flow = flow
        self.base_distributions = base_distributions
        
    def _flatten(self, data_dict: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Helper to flatten dict to vector in correct order."""
        structure = self.flow.velocity_fn.structure
        flat_list = []
        for block in structure.block_order:
            val = data_dict.get(block)
            if val is None:
                dim = structure.dims[block]
                val = jnp.zeros((dim,))
            flat_list.append(val.ravel())
        return jnp.concatenate(flat_list)

    def _unflatten(self, flat_vector: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Helper to unflatten vector to dict."""
        structure = self.flow.velocity_fn.structure
        result = {}
        start = 0
        for block in structure.block_order:
            dim = structure.dims[block]
            result[block] = flat_vector[start:start+dim]
            start += dim
        return result

    def sample(self,
               key: jax.Array,
               conditions: Optional[Dict[str, jnp.ndarray]] = None,
               targets: Optional[Union[Set[str], List[str]]] = None) -> Dict[str, jnp.ndarray]:
        
        structure = self.flow.velocity_fn.structure
        all_blocks_set = set(structure.block_order)
        
        target_set, cond_keys = self._validate_targets_conditions(targets, conditions, all_blocks_set)
        
        if not structure.is_ancestral_query(target_set, cond_keys):
             raise NotImplementedError("Intractable query: conditions depend on targets.")

        # 1. Handle Conditions: Backward integrate to get noise at t=0
        if conditions:
            # Prepare state at t=1 with conditions and zeros elsewhere
            # Due to block triangularity, zeros for targets don't affect condition noise
            x1_cond_dict = {b: conditions[b] for b in cond_keys}
            for b in structure.block_order:
                if b not in x1_cond_dict:
                    x1_cond_dict[b] = jnp.zeros((structure.dims[b],))
            
            z_cond_dict = self.flow.backward(x1_cond_dict)
            # Noise for condition blocks
            z_dict = {b: z_cond_dict[b] for b in cond_keys}
        else:
            z_dict = {}

        # 2. Sample noise for other required blocks (targets + non-condition ancestors)
        simulation_blocks = structure.get_required_simulation_blocks(target_set, cond_keys)
        # Filter out blocks already in z_dict (conditions)
        to_sample = [b for b in simulation_blocks if b not in z_dict]
        keys = random.split(key, len(to_sample))
        
        for i, block in enumerate(to_sample):
            z_dict[block] = self.base_distributions[block].sample(keys[i])
        
        # 3. Forward integrate from t=0 noise to get t=1 values
        # Initialize full x0 vector with zeros for blocks not in z_dict
        k, sk = random.split(key)
        for b in structure.block_order:
            if b not in z_dict:
                # raise ValueError(f'Missing var {b}')
                # z_dict[b] = jnp.zeros((structure.dims[b],))
                z_dict[b] = self.base_distributions[b].sample(sk)
                k, sk = random.split(k)
        
        x1_dict = self.flow.forward(z_dict)
        
        return {t: x1_dict[t] for t in target_set}

    @eqx.filter_jit
    def b_sample(self,
                 key: jax.Array,
                 n_samples: int,
                 conditions: Optional[Dict[str, jnp.ndarray]] = None,
                 targets: Optional[Union[Set[str], List[str]]] = None) -> Dict[str, jnp.ndarray]:
        """
        Optimized batched sampling for conditional queries.
        
        The key optimization: when conditions are provided, the backward integration
        to find condition noise only needs to happen ONCE, not once per sample.
        Then we batch the forward integration over different target noise samples.
        """
        structure = self.flow.velocity_fn.structure
        all_blocks_set = set(structure.block_order)
        
        target_set, cond_keys = self._validate_targets_conditions(targets, conditions, all_blocks_set)
        
        if not structure.is_ancestral_query(target_set, cond_keys):
            raise NotImplementedError("Intractable query: conditions depend on targets.")
        
        # 1. Handle Conditions: Backward integrate ONCE to get noise at t=0
        if conditions:
            x1_cond_dict = {b: conditions[b] for b in cond_keys}
            for b in structure.block_order:
                if b not in x1_cond_dict:
                    x1_cond_dict[b] = jnp.zeros((structure.dims[b],))
            
            z_cond_dict = self.flow.backward(x1_cond_dict)  # Only once!
            z_cond = {b: z_cond_dict[b] for b in cond_keys}
        else:
            z_cond = {}
        
        # 2. Determine which blocks need fresh noise samples
        simulation_blocks = structure.get_required_simulation_blocks(target_set, cond_keys)
        to_sample = [b for b in simulation_blocks if b not in z_cond]
        
        # 3. Sample noise for all n_samples at once
        keys = random.split(key, len(to_sample) + 1)
        main_key = keys[0]
        block_keys = keys[1:]
        
        # z_batch will have shape (n_samples, dim) for each block
        z_batch = {}
        for i, block in enumerate(to_sample):
            sample_keys = random.split(block_keys[i], n_samples)
            z_batch[block] = vmap(self.base_distributions[block].sample)(sample_keys)
        
        # Add condition noise (broadcast to n_samples)
        for b in z_cond:
            z_batch[b] = jnp.broadcast_to(z_cond[b][None, :], (n_samples, z_cond[b].shape[0]))
        
        # Fill in any remaining blocks with zeros or samples
        fill_keys = random.split(main_key, len(structure.block_order))
        for i, b in enumerate(structure.block_order):
            if b not in z_batch:
                sample_keys = random.split(fill_keys[i], n_samples)
                z_batch[b] = vmap(self.base_distributions[b].sample)(sample_keys)
        
        # 4. Batch forward integration
        # Restructure z_batch from {block: (n_samples, dim)} to list of {block: (dim,)}
        def get_sample_dict(i):
            return {b: z_batch[b][i] for b in structure.block_order}
        
        x1_batch = vmap(self.flow.forward)(
            {b: z_batch[b] for b in structure.block_order}
        )
        
        return {t: x1_batch[t] for t in target_set}

    def log_prob(self,
                 x: Dict[str, jnp.ndarray],
                 targets: Optional[Union[Set[str], List[str]]] = None,
                 conditions: Optional[Union[Set[str], List[str]]] = None) -> jnp.ndarray:
        
        structure = self.flow.velocity_fn.structure
        all_vars = set(structure.block_order)

        if conditions is None: conditions_keys = set()
        else: conditions_keys = set(conditions)
        
        if targets is None:
            target_set = all_vars - conditions_keys
        else:
            target_set = set(targets)
            
        target_set, cond_keys = self._validate_targets_conditions(target_set, conditions, all_vars)
        
        if not structure.is_tractable_query(target_set, cond_keys):
             raise NotImplementedError("Intractable log_prob.")

        # 1. Backward integrate full state to get noise and full trajectory
        # The input x may only contain targets + conditions. Fill missing blocks with zeros.
        # This is safe because is_tractable_query already validated that all ancestors
        # of targets are in targets or conditions.
        x_full = dict(x)
        for b in structure.block_order:
            if b not in x_full:
                x_full[b] = jnp.zeros((structure.dims[b],))
        
        xts_dict = self.flow.trajectory_backward(x_full)
        ts = jnp.linspace(1, 0, self.flow.n_steps + 1)
        dt = ts[1] - ts[0] # dt is negative
        
        # 2. Accumulate divergence only for target blocks
        block_net = BlockBasedVelocityFn.from_parallel(self.flow.velocity_fn)
        
        def div_accumulator(i, acc):
            t = ts[i]
            # Construct dictionary state at time step i
            x_dict_t = {b: xts_dict[b][i] for b in xts_dict}
            
            step_div = 0.0
            for block in target_set:
                # Capture block by value to avoid closure issues
                v_block_fn = lambda xb, blk=block, ctx=x_dict_t, time=t: \
                    block_net.get_block_velocity(blk)(xb, time, ctx)
                
                div_val = divergence_jvp(v_block_fn, argnums=0)(x_dict_t[block])
                step_div += div_val
            
            return acc + step_div * dt

        log_det = lax.fori_loop(0, self.flow.n_steps, div_accumulator, 0.0)
        
        # 3. Base Log Prob
        z_dict = {b: xts_dict[b][-1] for b in xts_dict}
        log_p_base = 0.0
        for block in target_set:
            log_p_base += self.base_distributions[block].log_prob(z_dict[block])
        
        # Sign correction: dt < 0 for backward integration, so -sign(dt) = +1
        # Formula: log p(x_1) = log p(x_0) - ∫₀¹ div(v) dt
        # When integrating backward: log_det ≈ -∫₀¹ div(v) dt, so we add it
        sign_dt = jnp.sign(dt)
        return log_p_base - sign_dt * log_det

    def transform_to_noise(self, x: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        return self.flow.backward(x)

    def counterfactual(self, data: Dict[str, jnp.ndarray], interventions: Dict[str, jnp.ndarray], targets: Optional[Union[Set[str], List[str]]] = None) -> tuple:
        """
        Counterfactual inference using the simplified abducted-noise approach.
        Returns both the sample and its log probability.
        
        Returns:
            (sample_dict, log_prob) where sample_dict is the counterfactual sample
            and log_prob is the log probability p(targets | do(interventions), factual_data).
        """
        structure = self.flow.velocity_fn.structure
        intervened_keys = set(interventions.keys())
        all_blocks = set(structure.block_order)
        
        # Determine target blocks for log_prob computation
        if targets is None:
            target_set = all_blocks
        else:
            target_set = set(targets)
        
        # Non-intervened targets: these are the blocks whose density we compute
        non_intervened_targets = target_set - intervened_keys
        
        # 1. Abduction: Map from observed data at t=1 back to noise space z at t=0
        z_dict = self.transform_to_noise(data)
        
        # 2. Action: Find noise for intervened variables that would produce intervention values at t=1
        if intervened_keys:
            x1_interv_dict = {b: interventions[b] for b in intervened_keys}
            for b in structure.block_order:
                if b not in x1_interv_dict:
                    x1_interv_dict[b] = jnp.zeros((structure.dims[b],))
            
            z_interv_dict = self.flow.backward(x1_interv_dict)
            
            # Replace abducted noise with intervention noise for intervened variables
            for b in intervened_keys:
                z_dict[b] = z_interv_dict[b]
        
        # 3. Prediction: Forward integrate from modified noise with trajectory tracking
        # Ensure z_dict has all blocks (fill missing with zeros from base distribution)
        z_dict_full = dict(z_dict)
        for b in structure.block_order:
            if b not in z_dict_full:
                z_dict_full[b] = self.base_distributions[b].sample(jax.random.PRNGKey(0))
        
        xts_dict = self.flow.trajectory_forward(z_dict_full)
        
        x1_cf_dict = {b: xts_dict[b][-1] for b in xts_dict}

        # 4. Hard-set intervened variables at t=1
        for b in intervened_keys:
            x1_cf_dict[b] = interventions[b]
        
        # 5. Compute log probability by accumulating divergence during forward integration
        # Only for non-intervened target blocks (intervened blocks are fixed, not random)
        ts = jnp.linspace(0, 1, self.flow.n_steps + 1)
        dt = ts[1] - ts[0]  # dt is positive for forward
        
        block_net = BlockBasedVelocityFn.from_parallel(self.flow.velocity_fn)
        
        def div_accumulator(i, acc):
            t = ts[i]
            x_dict_t = {b: xts_dict[b][i] for b in xts_dict}
            
            step_div = 0.0
            for block in non_intervened_targets:
                v_block_fn = lambda xb, blk=block, ctx=x_dict_t, time=t: \
                    block_net.get_block_velocity(blk)(xb, time, ctx)
                
                div_val = divergence_jvp(v_block_fn, argnums=0)(x_dict_t[block])
                step_div += div_val
            
            return acc + step_div * dt
        
        log_det = lax.fori_loop(0, self.flow.n_steps, div_accumulator, 0.0)
        
        # 6. Base Log Prob (at t=0) for non-intervened targets
        log_p_base = 0.0
        for block in non_intervened_targets:
            log_p_base += self.base_distributions[block].log_prob(z_dict[block])
        
        # Sign correction: dt > 0 for forward integration, so -sign(dt) = -1
        # Formula: log p(x_1) = log p(x_0) - ∫₀¹ div(v) dt
        # When integrating forward: log_det ≈ +∫₀¹ div(v) dt, so we subtract it
        sign_dt = jnp.sign(dt)
        log_prob = log_p_base - sign_dt * log_det
        
        # 7. Filter targets and return
        sample_dict = x1_cf_dict if targets is None else {t: x1_cf_dict[t] for t in target_set if t in x1_cf_dict}
        
        return sample_dict, log_prob
