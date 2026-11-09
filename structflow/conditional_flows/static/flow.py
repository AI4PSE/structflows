import jax
import jax.numpy as jnp
from jax import lax, jit, random, vmap
from jax.flatten_util import ravel_pytree
from typing import Dict, List
import equinox as eqx

from flow.flow import Flow
from flow.integrators import get_integrator
from structflow.conditional_flows.flow import ConditionalFlow
from structflow.conditional_flows.static.net import StructuredVelocityFn

class PartialVelocityFn(eqx.Module):
    """
    Wrapper for velocity function that fixes conditions.
    Used to adapt StructuredVelocityFn or Block to an unconditional interface v(x, t).
    """
    velocity_fn: StructuredVelocityFn
    conditions: Dict[str, jnp.ndarray]
    active_blocks: List[str]
    
    def __call__(self, x_state, t):
        # If active_blocks has only 1 item and x_state is array, we wrap it
        if len(self.active_blocks) == 1 and isinstance(x_state, jnp.ndarray):
             # When using Flow for single block, x_state is array.
             # We need to dict-ify it.
             block = self.active_blocks[0]
             state_dict = {block: x_state}
        else:
             state_dict = x_state

        # Combine state with static conditions
        full_context = dict(self.conditions)
        full_context.update(state_dict)

        velocities = self.velocity_fn(full_context, t, active_blocks=self.active_blocks)
        
        if len(self.active_blocks) == 1 and isinstance(x_state, jnp.ndarray):
            return velocities[self.active_blocks[0]]
        else:
            return velocities

class SingleBlockVelocityFn(eqx.Module):
    """
    Fast path for integrating one block: holds the block's (already unstacked)
    module and its fixed parent values. Equivalent to
    PartialVelocityFn(velocity_fn, conditions, [block]) but avoids re-gathering
    the stacked group parameters (tree_map(x[idx], ...)) at every ODE step —
    that per-NFE parameter copy dominates sampling cost for large blocks.
    """
    block: eqx.Module
    parents: tuple

    def __call__(self, x_state, t):
        t_arr = jnp.atleast_1d(jnp.asarray(t))
        # Integrator/trajectory code may pass (1, ..., dim) states; the old
        # grouped path's vmap-over-stack absorbed such axes. Blocks expect
        # (dim,). Leading axes must all be singletons (reshape errors if not).
        orig_shape = x_state.shape
        x = x_state.reshape(orig_shape[-1])
        out = self.block(t_arr, x, *self.parents)
        return out.reshape(orig_shape)


class StaticConditionalFlow(ConditionalFlow):
    """
    Flow for block-structured variables with static conditional dependencies.
    """
    velocity_fn: StructuredVelocityFn

    def __init__(self, velocity_fn: StructuredVelocityFn, n_steps: int, integrator_name: str = 'rk4'):
        super().__init__(n_steps, integrator_name)
        if not hasattr(velocity_fn, 'structure'):
            raise TypeError(f"velocity_fn must have 'structure' attribute, got {type(velocity_fn)}")
        
        self.velocity_fn = velocity_fn

    def trajectory_forward(self,
                           x0: Dict[str, jnp.ndarray],
                           active_blocks: List[str],
                           conditions: Dict[str, jnp.ndarray] = None,
                           **kwargs) -> Dict[str, jnp.ndarray]:
        """
        Compute forward trajectories with autoregressive sequencing.
        """
        if conditions is None: conditions = {}
        
        # Sort topological
        sorted_active = [b for b in self.velocity_fn.structure.block_order if b in active_blocks]
        if len(sorted_active) != len(active_blocks):
             raise ValueError("active_blocks validation failed")

        current_conditions = dict(conditions)
        final_trajectories = {}

        blocks_unstacked = self.velocity_fn.blocks  # one gather, outside the ODE loops
        for block in sorted_active:
            # Single-block velocity with conditions fixed in closure:
            # v_b(x_b, t). Unstacked once above — not per ODE step.
            parent_names = self.velocity_fn.structure.dependencies.get(block, [])
            v_part = SingleBlockVelocityFn(
                blocks_unstacked[block],
                tuple(current_conditions[p] for p in parent_names))

            # Create Flow
            # Note: Flow expects 'velocity' module.
            flow = Flow(v_part, self.n_steps, self.integrator_name)
            
            # Solve
            # Input is just the array for this block
            traj_array = flow.forward_trajectory(x0[block])
            
            final_trajectories[block] = traj_array
            # use the generated output as the conditions for the next block
            current_conditions[block] = traj_array[-1]
            
        return final_trajectories

    def trajectory_backward(self,
                            x1: Dict[str, jnp.ndarray],
                            active_blocks: List[str],
                            conditions: Dict[str, jnp.ndarray] = None,
                            **kwargs) -> Dict[str, jnp.ndarray]:
        """
        Compute backward trajectories. Each block integrated independently using static context.
        """
        if conditions is None: conditions = {}
        
        # Combined conditions: all t=1 values known
        combined_conditions = {**conditions, **x1}
        
        final_trajectories = {}
        blocks_unstacked = self.velocity_fn.blocks  # one gather, outside the ODE loops
        for block in active_blocks:
            parent_names = self.velocity_fn.structure.dependencies.get(block, [])
            v_part = SingleBlockVelocityFn(
                blocks_unstacked[block],
                tuple(combined_conditions[p] for p in parent_names))
            flow = Flow(v_part, self.n_steps, self.integrator_name)
            # Input is just the array for this block at t=1
            traj_dict = flow.backward_trajectory(x1[block])
            final_trajectories[block] = traj_dict
            
        return final_trajectories

    def forward(self, x0: Dict[str, jnp.ndarray], active_blocks: List[str] = None, conditions: Dict[str, jnp.ndarray] = None, **kwargs) -> Dict[str, jnp.ndarray]:
        if active_blocks is None:
            active_blocks = list(x0.keys())
        return super().forward(x0, active_blocks=active_blocks, conditions=conditions, **kwargs)

    def backward(self, x1: Dict[str, jnp.ndarray], active_blocks: List[str] = None, conditions: Dict[str, jnp.ndarray] = None, **kwargs) -> Dict[str, jnp.ndarray]:
        if active_blocks is None:
            active_blocks = list(x1.keys())
        return super().backward(x1, active_blocks=active_blocks, conditions=conditions, **kwargs)

    # --- Batched Entry Points ---

    @eqx.filter_jit
    def b_trajectory_forward(self, x0, active_blocks, conditions=None):
        fn = lambda _x0, _c: self.trajectory_forward(_x0, active_blocks, _c)
        return vmap(fn)(x0, conditions)

    @eqx.filter_jit
    def b_trajectory_backward(self, x1, active_blocks, conditions=None):
        fn = lambda _x1, _c: self.trajectory_backward(_x1, active_blocks, _c)
        return vmap(fn)(x1, conditions)

    @eqx.filter_jit
    def b_backward(self, 
                   x1: Dict[str, jnp.ndarray], 
                   active_blocks: List[str], 
                   conditions: Dict[str, jnp.ndarray] = None) -> Dict[str, jnp.ndarray]:
        traj_dict = self.b_trajectory_backward(x1, active_blocks, conditions)
        return {k: v[:, -1, :] for k, v in traj_dict.items()}
