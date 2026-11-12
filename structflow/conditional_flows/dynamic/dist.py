"""
DynamicDistribution: Conditional distribution using weight clipping and trajectory interpolation.
"""

import jax
import jax.numpy as jnp
from jax import random, lax, vmap
import equinox as eqx
from typing import Dict, List, Optional, Union, Set
from structflow.bayes.distribution import Distribution
from structflow.conditional_flows.dist import ConditionalDistribution
from flow.utils import divergence_jvp
from structflow.conditional_flows.dynamic.flow import DynamicFlow, ClampedFullVelocityFn
from structflow.conditional_flows.dynamic.subset_velocity import (
    clip_tail,
    SubsetVelocityFn,
)


class DynamicDistribution(ConditionalDistribution, eqx.Module):
    """
    Conditional distribution for dynamic conditioning (the default implementation).
    
    Uses weight clipping to remove unused variables and trajectory interpolation
    for conditions, enabling efficient and exact conditional sampling.
    """
    flow: DynamicFlow
    base_distributions: Dict[str, Distribution]
    
    def __init__(self, flow: DynamicFlow, base_distributions: Dict[str, Distribution]):
        """
        Initialize DynamicDistribution.
        
        Args:
            flow: DynamicFlow instance
            base_distributions: Dictionary mapping block names to base Distribution objects
        """
        self.flow = flow
        self.base_distributions = base_distributions
    
    def sample(self,
               key: jax.Array,
               conditions: Optional[Dict[str, jnp.ndarray]] = None,
               targets: Optional[Union[Set[str], List[str]]] = None) -> Dict[str, jnp.ndarray]:
        """
        Sample from p(targets | conditions).
        
        Uses trajectory interpolation for conditions and only integrates targets.
        """
        structure = self.flow.velocity_fn.structure
        all_blocks_set = set(structure.block_order)
        
        target_set, cond_keys = self._validate_targets_conditions(targets, conditions, all_blocks_set)
        
        if not structure.is_ancestral_query(target_set, cond_keys):
            raise NotImplementedError("Intractable query: conditions depend on targets.")
        
        # Get conditions dict if provided
        cond_dict = conditions if isinstance(conditions, dict) else None
        
        # Determine what needs to be simulated
        simulation_blocks = structure.get_required_simulation_blocks(target_set, cond_keys)
        
        # Sample base noise for simulation blocks
        keys = random.split(key, len(simulation_blocks))
        x0_sim = {}
        for i, block in enumerate(simulation_blocks):
            x0_sim[block] = self.base_distributions[block].sample(keys[i])
        
        # Forward integrate
        x1_dict = self.flow.forward(x0_sim, active_blocks=simulation_blocks, conditions=cond_dict)
        
        return {t: x1_dict[t] for t in target_set}
    
    @eqx.filter_jit
    def b_sample(self,
                 key: jax.Array,
                 n_samples: int,
                 conditions: Optional[Dict[str, jnp.ndarray]] = None,
                 targets: Optional[Union[Set[str], List[str]]] = None) -> Dict[str, jnp.ndarray]:
        """
        Batched sampling.
        """
        structure = self.flow.velocity_fn.structure
        all_blocks_set = set(structure.block_order)
        
        target_set, cond_keys = self._validate_targets_conditions(targets, conditions, all_blocks_set)
        
        if not structure.is_ancestral_query(target_set, cond_keys):
            raise NotImplementedError("Intractable query: conditions depend on targets.")
        
        # Determine simulation blocks
        simulation_blocks = structure.get_required_simulation_blocks(target_set, cond_keys)
        to_sample = [b for b in simulation_blocks]
        
        # Sample noise for all n_samples
        keys = random.split(key, len(to_sample) + 1)
        main_key = keys[0]
        block_keys = keys[1:]
        
        z_batch = {}
        for i, block in enumerate(to_sample):
            sample_keys = random.split(block_keys[i], n_samples)
            z_batch[block] = vmap(self.base_distributions[block].sample)(sample_keys)
        
        # Batch forward integration
        x1_batch = vmap(lambda z_dict: self.flow.forward(
            z_dict, 
            active_blocks=simulation_blocks,
            conditions=conditions
        ))({b: z_batch[b] for b in simulation_blocks})
        
        return {t: x1_batch[t] for t in target_set}
    
    def log_prob(self,
                 x: Dict[str, jnp.ndarray],
                 targets: Optional[Union[Set[str], List[str]]] = None,
                 conditions: Optional[Union[Set[str], List[str]]] = None) -> jnp.ndarray:
        """
        Compute log p(targets | conditions).
        """
        structure = self.flow.velocity_fn.structure
        all_vars = set(structure.block_order)
        
        if conditions is None:
            conditions_keys = set()
        else:
            conditions_keys = set(conditions)
        
        if targets is None:
            target_set = all_vars - conditions_keys
        else:
            target_set = set(targets)
        
        target_set, cond_keys = self._validate_targets_conditions(target_set, conditions, all_vars)
        
        if not structure.is_tractable_query(target_set, cond_keys):
            raise NotImplementedError("Intractable log_prob.")
        
        # Backward integrate to get noise
        x_full = dict(x)
        for b in structure.block_order:
            if b not in x_full:
                x_full[b] = jnp.zeros((structure.dims[b],))
        
        xts_dict = self.flow.trajectory_backward(x_full)
        ts = jnp.linspace(1, 0, self.flow.n_steps + 1)
        dt = ts[1] - ts[0]
        
        # Divergence over target blocks of the clamped full velocity
        target_list = [b for b in structure.block_order if b in target_set]
        cond_block_names = [b for b in structure.block_order if b in cond_keys]

        # Expand keep_set to the ancestral closure of (targets ∪ conditions) so
        # that clip_tail never encounters a parent of a kept block in the remove set.
        keep_set = structure.get_ancestral_closure(target_set | cond_keys)
        keep_blocks = [b for b in structure.block_order if b in keep_set]

        # All non-target blocks in keep_set (explicit conditions + implicit ancestors)
        # must supply their trajectory to the SubsetVelocityFn.
        expanded_cond_block_names = [b for b in structure.block_order
                                     if b in keep_set and b not in target_set]

        fwd_ts = jnp.linspace(0, 1, self.flow.n_steps + 1)
        condition_trajs = {
            b: jnp.flip(xts_dict[b], axis=0) for b in expanded_cond_block_names
        }

        # The divergence must be of the velocity the sampler integrates (the field's
        # velocity on the clamped full state), not of the clipped raw network, which
        # is the eta1 denoiser of OneSidedField_eta1, not a velocity.
        subset_vfn = ClampedFullVelocityFn(
            velocity_fn=self.flow.velocity_fn,
            condition_trajs=condition_trajs,
            traj_ts=fwd_ts,
            cond_block_names=expanded_cond_block_names,
            active_block_names=target_list,
        )
        
        def div_accumulator(i, acc):
            t = ts[i]
            x_active = jnp.concatenate([xts_dict[b][i] for b in target_list])
            
            div_val = divergence_jvp(
                lambda xa: subset_vfn(xa, t), argnums=0
            )(x_active)
            
            return acc + div_val * dt
        
        log_det = lax.fori_loop(0, self.flow.n_steps, div_accumulator, 0.0)
        
        # Base Log Prob
        z_dict = {b: xts_dict[b][-1] for b in xts_dict}
        log_p_base = 0.0
        for block in target_set:
            log_p_base += self.base_distributions[block].log_prob(z_dict[block])
        
        # Sign correction
        sign_dt = jnp.sign(dt)
        return log_p_base - sign_dt * log_det
    
    def transform_to_noise(self, x: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """
        Map from data to noise space.
        """
        return self.flow.backward(x)
    
    def counterfactual(self,
                       data: Dict[str, jnp.ndarray],
                       interventions: Dict[str, jnp.ndarray],
                       targets: Optional[Union[Set[str], List[str]]] = None) -> tuple:
        """
        Perform counterfactual inference with trajectory interpolation.
        
        Returns:
            (sample_dict, log_prob)
        """
        structure = self.flow.velocity_fn.structure
        intervened_keys = set(interventions.keys())
        all_blocks = set(structure.block_order)
        
        if targets is None:
            target_set = all_blocks
        else:
            target_set = set(targets)
        
        # Non-intervened targets for probability computation
        non_intervened_targets = target_set - intervened_keys
        
        # 1. Abduction: Map observed data to noise
        z_dict = self.transform_to_noise(data)

        # 2.-3. Action and prediction: integrate the non-intervened targets and their
        # non-intervened ancestors forward from the abducted noise. The intervened blocks'
        # trajectories are built from their values and their ancestors' counterfactual
        # trajectories (DynamicFlow._get_condition_trajectories with the abducted noise).
        # Prepare conditions with interventions
        cond_dict = dict(interventions)

        # Determine what to simulate: non-intervened targets plus any implicit
        # ancestors of those targets (ancestors that are not interventions themselves).
        non_interv_targets_set = target_set - intervened_keys
        implicit_ancestor_set = (
            structure.get_ancestral_closure(non_interv_targets_set)
            - non_interv_targets_set
            - intervened_keys
        )
        simulation_blocks = [
            b for b in structure.block_order
            if b in non_interv_targets_set | implicit_ancestor_set
        ]
        x0_sim = {b: z_dict[b] for b in simulation_blocks if b in z_dict}
        
        xts_dict = self.flow.trajectory_forward(
            x0_sim,
            active_blocks=simulation_blocks,
            conditions=cond_dict
        )
        
        x1_cf_dict = {}
        for b in simulation_blocks:
            if b in xts_dict:
                x1_cf_dict[b] = xts_dict[b][-1]
        
        # Hard-set intervened variables
        for b in intervened_keys:
            x1_cf_dict[b] = interventions[b]
        
        # 4. Compute log probability
        ts = jnp.linspace(0, 1, self.flow.n_steps + 1)
        dt = ts[1] - ts[0]
        
        # Divergence over non-intervened targets of the clamped full velocity.
        # Expand keep_set to the full ancestral closure so that clip_tail does
        # not raise when a parent of a kept block would otherwise be removed.
        active_blocks = [b for b in structure.block_order if b in non_intervened_targets]
        interv_block_names = [b for b in structure.block_order if b in intervened_keys]

        keep_set = structure.get_ancestral_closure(
            set(active_blocks) | set(interv_block_names)
        )
        keep_blocks = [b for b in structure.block_order if b in keep_set]

        # All non-active blocks in keep_set supply their trajectory as conditions.
        expanded_cond_block_names = [
            b for b in structure.block_order
            if b in keep_set and b not in set(active_blocks)
        ]

        # Trajectories for explicit intervention blocks
        condition_trajs = (
            self.flow._get_condition_trajectories(interventions, noise=x0_sim)
            if interv_block_names else {}
        )
        # Trajectories for implicit ancestor blocks (simulated above)
        for b in implicit_ancestor_set:
            if b in xts_dict:
                condition_trajs[b] = xts_dict[b]
        
        # Velocity the sampler integrates (see log_prob), not the clipped raw network.
        subset_vfn = ClampedFullVelocityFn(
            velocity_fn=self.flow.velocity_fn,
            condition_trajs=condition_trajs,
            traj_ts=ts,
            cond_block_names=expanded_cond_block_names,
            active_block_names=active_blocks,
        )
        
        def div_accumulator(i, acc):
            t = ts[i]
            x_active = jnp.concatenate([xts_dict[b][i] for b in active_blocks])
            
            div_val = divergence_jvp(
                lambda xa: subset_vfn(xa, t), argnums=0
            )(x_active)
            
            return acc + div_val * dt
        
        log_det = lax.fori_loop(0, self.flow.n_steps, div_accumulator, 0.0)
        
        # Base log prob for non-intervened targets
        log_p_base = 0.0
        for block in non_intervened_targets:
            if block in z_dict:
                log_p_base += self.base_distributions[block].log_prob(z_dict[block])
        
        sign_dt = jnp.sign(dt)
        log_prob = log_p_base - sign_dt * log_det
        
        # Filter targets
        sample_dict = x1_cf_dict if targets is None else {t: x1_cf_dict[t] for t in target_set if t in x1_cf_dict}
        
        return sample_dict, log_prob
