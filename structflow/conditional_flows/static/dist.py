import jax
import jax.numpy as jnp
from jax import lax, jit, random
from typing import Dict, List, Callable, Optional, Union, Set
import equinox as eqx

from structflow.bayes.distribution import Distribution
from flow.utils import divergence_jvp
from structflow.conditional_flows.static.flow import StaticConditionalFlow
from structflow.conditional_flows.utils import DictNormaliser
from structflow.conditional_flows.static.net import StructuredVelocityFn
from structflow.conditional_flows.dist import ConditionalDistribution
from structflow.conditional_flows.counterfactual import CounterfactualMixin

class StaticConditionalDistribution(CounterfactualMixin, ConditionalDistribution, eqx.Module):
    """
    Distribution defined by a set of conditional flows with static conditioning.
    
    Implements a joint distribution p(x1, ..., xn) where each variable xi can depend
    on previous variables through a static conditional flow:
    xi = Flow(zi; parents(xi))
    
    Attributes:
        flow: StaticConditionalFlow defining the transformation.
        base_distributions: Dictionary mapping block names to their base distributions.
    """
    flow: StaticConditionalFlow
    base_distributions: Dict[str, Distribution]

    def __init__(self, flow: StaticConditionalFlow, base_distributions: Dict[str, Distribution]):
        if not isinstance(flow, StaticConditionalFlow):
            raise TypeError(f"flow must be a StaticConditionalFlow, got {type(flow)}")
        if not isinstance(base_distributions, dict):
            raise TypeError(f"base_distributions must be a dict, got {type(base_distributions)}")
        
        structure_blocks = set(flow.velocity_fn.structure.block_order)
        base_blocks = set(base_distributions.keys())
        
        if structure_blocks != base_blocks:
            missing = structure_blocks - base_blocks
            extra = base_blocks - structure_blocks
            raise ValueError(f"Base distributions mismatch. Missing: {missing}, Extra: {extra}")
        
        self.flow = flow
        self.base_distributions = base_distributions

    def sample(self, 
               key: jax.Array, 
               conditions: Optional[Dict[str, jnp.ndarray]] = None,
               targets: Optional[Union[Set[str], List[str]]] = None,
               interventions: Optional[Dict[str, jnp.ndarray]] = None) -> Dict[str, jnp.ndarray]:
        """
        Sample from the distribution.

        ``interventions`` applies do(block = value): the block's incoming edges
        are deleted and its value is clamped, so it acts as a known value for
        everything downstream while its own parents become irrelevant to it.
        This differs from ``conditions``, which leaves the mechanism intact and
        therefore propagates information *up* the arrow -- conditioning on a
        child changes belief about its parent, intervening on it does not.

        The practical consequence is that a query which is intractable as a
        conditioning query can be perfectly tractable as an intervention: no
        inversion is required, because surgery removes exactly the edges that
        would have needed inverting. One fitted model can therefore answer both
        observational and interventional queries, without compiling and fitting
        a second model on the mutilated graph.
        """
        structure = self.flow.velocity_fn.structure
        all_blocks = set(structure.block_order)

        interventions = interventions or {}
        if interventions:
            unknown = set(interventions) - all_blocks
            if unknown:
                raise ValueError(f"Unknown blocks to intervene on: {sorted(unknown)}")
            overlap = set(interventions) & set(conditions or {})
            if overlap:
                raise ValueError(
                    f"Blocks both conditioned on and intervened on: {sorted(overlap)}. "
                    f"Observing a variable and setting it are different queries; "
                    f"pick one.")
            # graph surgery for the tractability check and for working out what
            # still needs simulating; the fitted blocks themselves are unchanged
            structure = structure.mutilated(set(interventions))
            conditions = dict(conditions or {}, **interventions)

        target_set, cond_keys = self._validate_targets_conditions(targets, conditions, all_blocks)

        if not structure.is_ancestral_query(target_set, cond_keys):
             raise NotImplementedError("Intractable query: conditions depend on targets (requires inversion).")

        # Determine what needs to be simulated
        # The flow handles the autoregressive sequence, we just need to tell it what to simulate.
        targets_to_simulate = target_set
        simulation_blocks = structure.get_required_simulation_blocks(targets_to_simulate, cond_keys)

        # is_ancestral_query only asks whether a condition is a descendant of a
        # TARGET. That is too weak. Intermediate blocks are simulated from their
        # priors, so if a condition is a descendant of any SIMULATED block, the
        # information that condition carries about it is silently discarded and
        # the samples answer a different question. In a Markovian MDP,
        # conditioning on the actions then returns the INTERVENTIONAL answer
        # rather than the observational one -- measured out by a factor of three
        # on a 4-step linear-Gaussian chain. Refuse rather than answer the wrong
        # query.
        for _b in simulation_blocks:
            _offenders = cond_keys & structure.get_all_descendants(_b)
            if _offenders:
                raise NotImplementedError(
                    f"Intractable query: condition(s) {sorted(_offenders)} are "
                    f"descendants of simulated block {_b!r}, so ancestral "
                    f"sampling would ignore what they say about it. Use the "
                    f"inference compilation, or pass interventions={{...}} if "
                    f"do() was meant rather than observation.")
        
        # Sample base noise for all blocks to be simulated
        keys = random.split(key, len(simulation_blocks))
        x0_sim = {}
        for i, block in enumerate(simulation_blocks):
            x0_sim[block] = self.base_distributions[block].sample(keys[i])
            
        # Run Flow Forward (Autoregressive Logic handled by Flow)
        # Note: we pass 'conditions' which provides initial known values.
        # The flow will augment this internally as it moves through the sequence.
        result_traj = self.flow.trajectory_forward(x0_sim, simulation_blocks, conditions)
        
        # Extract and format output
        final_output = {}
        
        # Include sampled targets
        for t in target_set:
            if t in result_traj:
                final_output[t] = result_traj[t][-1]
            # elif t in cond_keys and conditions is not None:
            #     final_output[t] = conditions[t]
                
        return final_output

    def log_prob(self, 
                 x: Dict[str, jnp.ndarray], 
                 targets: Optional[Union[Set[str], List[str]]] = None, 
                 conditions: Optional[Union[Set[str], List[str]]] = None) -> jnp.ndarray:
        """
        Compute log p(targets | conditions).
        """
        all_vars = set(self.flow.velocity_fn.structure.block_order)
        
        if conditions is None: conditions_keys = set()
        else: conditions_keys = set(conditions)
        
        if targets is None:
            targets_arg = [k for k in all_vars if k not in conditions_keys]
        else:
            targets_arg = targets
            
        target_set, cond_keys = self._validate_targets_conditions(targets_arg, conditions, all_vars)
        
        if not self.flow.velocity_fn.structure.is_tractable_query(target_set, cond_keys):
            raise NotImplementedError("Intractable log_prob query.")

        target_list = list(target_set)
        
        # 1. Integrate Backward
        # We pass 'x' as conditions. The flow.trajectory_backward will combine 
        # conditions + active_state to ensure correct static conditioning.
        # Actually flow.trajectory_backward(x1, active, conditions) automatically adds x1 to conditions.
        # So we can just pass the conditions dict (if any) or None.
        # BUT: x contains the values for 'active_blocks' (targets).
        # We need to pass x as x1 argument.
        
        # Prepare x1 input (values at t=1)
        # We can pass full 'x' dict, the flow will select active blocks.
        x1_input = x
        
        # We also pass 'conditions' (the observed values at t=1).
        # Actually x contains EVERYTHING (targets + conditions).
        # So passing x as x1 is sufficient if we trust flow to use x1 as conditions.
        # Flow.trajectory_backward does: combined = {**conditions, **x1}.
        # So if we pass x as x1, combined will have all x values.
        # If we pass explicit 'conditions' dict (if it exists separate from x), 
        # it might double up but that's fine.
        # 'conditions' argument to log_prob is meant to be a set of keys if 'x' has everything.
        # OR 'conditions' could be a dict of values?
        # My type hint says Union[Set, List]. 
        # But _validate handles dict too?
        # If 'conditions' is passed as keys, we assume values are in 'x'.
        # If passed as dict, values are there.
        
        # Let's extract explicit values for conditions if we want to be clean, 
        # but passing x (which has everything) as conditions is cleaner for the "all known" trick.
        
        # So: pass x as 'conditions' to be safe?
        # trajectory_backward(x, active, conditions=x).
        # combined = {**x, **x} = x.
        # Then _solve_subset uses {**state, **x}.
        # This is correct.
        
        traj_bwd = self.flow.trajectory_backward(x, target_list, conditions=x)
        
        # 2. Base Log Prob (t=0)
        log_p_base = 0.0
        x0_extracted = {k: traj_bwd[k][-1] for k in target_list}
        
        for k in target_list:
            log_p_base += self.base_distributions[k].log_prob(x0_extracted[k])
            
        # 3. Divergence Accumulation
        ts = jnp.linspace(1, 0, self.flow.n_steps + 1)
        dt = ts[1] - ts[0]
        
        def div_accumulator(i, acc):
            t = ts[i]
            state = {k: traj_bwd[k][i] for k in target_list}
            full_context = {**state, **x}
            
            step_div = 0.0
            for block in target_list:
                block_fn = self.flow.velocity_fn.blocks[block]
                val = state[block]
                
                parent_names = self.flow.velocity_fn.structure.dependencies.get(block, [])
                parents = [full_context[p] for p in parent_names]
                
                t_arr = jnp.atleast_1d(t)
                div_val = divergence_jvp(
                    lambda _x: block_fn(t_arr, _x, *parents),
                    argnums=0
                )(val)
                step_div += div_val
            return acc + step_div * dt

        delta_log_p = lax.fori_loop(0, self.flow.n_steps, div_accumulator, 0.0)
        
        return log_p_base + delta_log_p
        
    def transform_to_noise(self, x: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """
        Maps Real Data -> Noise for a single example.
        """
        return self.flow.backward(x)

    # --- Counterfactual backbone hooks (graph logic lives in CounterfactualMixin) ---

    @property
    def _cf_structure(self):
        return self.flow.velocity_fn.structure

    def _abduct(self, data: Dict[str, jnp.ndarray],
                blocks: Set[str]) -> Dict[str, jnp.ndarray]:
        """Recover exogenous noise for `blocks` by integrating the flow backward."""
        noise_traj = self.flow.trajectory_backward(
            data,
            active_blocks=list(blocks),
            conditions=data
        )
        return {k: v[-1] for k, v in noise_traj.items()}

    def _predict(self, x0_sim: Dict[str, jnp.ndarray],
                 conditions: Dict[str, jnp.ndarray],
                 simulation_blocks: List[str]) -> tuple:
        """Simulate `simulation_blocks` forward from noise. Returns the final
        values plus the full trajectory (needed for divergence accumulation)."""
        result_traj = self.flow.trajectory_forward(
            x0_sim,
            active_blocks=simulation_blocks,
            conditions=conditions
        )
        results = {}
        for k in simulation_blocks:
            results[k] = result_traj[k][-1]
        return results, result_traj

    def _predict_log_prob(self, x0_sim: Dict[str, jnp.ndarray],
                          simulation_blocks: List[str],
                          results: Dict[str, jnp.ndarray],
                          aux) -> jnp.ndarray:
        """Compute log probability by accumulating divergence during forward
        integration. The exact divergence is O(D^2) per step and OOMs on
        image-sized blocks; samples-only callers use compute_log_prob=False
        (handled in the mixin) and never reach this."""
        result_traj = aux

        ts = jnp.linspace(0, 1, self.flow.n_steps + 1)
        dt = ts[1] - ts[0]  # dt is positive for forward
        
        def div_accumulator(i, acc):
            t = ts[i]
            # Get state at this time step from trajectories
            state = {}
            for k in simulation_blocks:
                state[k] = result_traj[k][i]
            
            full_context = {**state, **results}
            
            step_div = 0.0
            for block in simulation_blocks:
                block_fn = self.flow.velocity_fn.blocks[block]
                val = state[block]
                
                parent_names = self.flow.velocity_fn.structure.dependencies.get(block, [])
                parents = [full_context[p] for p in parent_names]
                
                t_arr = jnp.atleast_1d(t)
                div_val = divergence_jvp(
                    lambda _x: block_fn(t_arr, _x, *parents),
                    argnums=0
                )(val)
                step_div += div_val
            return acc + step_div * dt

        delta_log_p = lax.fori_loop(0, self.flow.n_steps, div_accumulator, 0.0)

        # Base Log Prob (at t=0)
        log_p_base = 0.0
        for k in simulation_blocks:
            log_p_base += self.base_distributions[k].log_prob(x0_sim[k])

        # For forward integration: log p(x_1) = log p(x_0) - ∫₀¹ div(v) dt
        return log_p_base - delta_log_p
