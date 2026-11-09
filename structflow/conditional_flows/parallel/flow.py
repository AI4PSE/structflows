import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Dict, List, Optional, Union, Set
from flow.flow import Flow
from structflow.conditional_flows.flow import ConditionalFlow
from structflow.conditional_flows.parallel.net import ParallelStructuredFn

class ParallelFlow(ConditionalFlow):
    """
    ODE Flow wrapper for ParallelStructuredFn.
    Operates on flattened state vectors but supports Dictionary I/O.
    """
    velocity_fn: ParallelStructuredFn

    def __init__(self, velocity_fn: ParallelStructuredFn, n_steps: int = 10, integrator_name: str = 'rk4', **kwargs):
        super().__init__(n_steps, integrator_name)
        self.velocity_fn = velocity_fn
        
    def _flatten(self, x: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Flatten full dictionary containing all blocks in correct order."""
        return jnp.concatenate([x[b] for b in self.velocity_fn.structure.block_order])

    def _flatten_subset(self, data: Dict[str, jnp.ndarray], blocks: List[str]) -> jnp.ndarray:
        """Flatten dictionary with only specific blocks present, filling missing with zeros."""
        flat_list = []
        for b in blocks:
            if b in data:
                flat_list.append(data[b].ravel())
            else:
                dim = self.velocity_fn.structure.dims[b]
                flat_list.append(jnp.zeros((dim,)))
        return jnp.concatenate(flat_list)

    def _unflatten_traj(self, traj: jnp.ndarray, active_blocks: List[str]) -> Dict[str, jnp.ndarray]:
        """Unflatten trajectory for specific active blocks."""
        result = {}
        for block in active_blocks:
            start, end = self.velocity_fn.block_offsets[block]
            result[block] = traj[..., start:end]
        return result

    def _get_active_blocks(self, x: Union[Dict, jnp.ndarray]) -> List[str]:
        """Determine active blocks from input dictionary."""
        if isinstance(x, dict):
            return [b for b in self.velocity_fn.structure.block_order if b in x]
        else:
            return self.velocity_fn.structure.block_order

    def trajectory_forward(self, x0: Union[Dict, jnp.ndarray], active_blocks: List[str] = None, conditions=None, **kwargs):
        """
        Computes the forward trajectory from x0.
        """
        # Determine active blocks if not provided
        if active_blocks is None:
            if isinstance(x0, dict):
                active_blocks = self._get_active_blocks(x0)
            else:
                active_blocks = self.velocity_fn.structure.block_order

        # Flatten if needed
        if isinstance(x0, dict):
            flat_x0 = self._flatten_subset(x0, self.velocity_fn.structure.block_order)
        else:
            flat_x0 = x0

        # Run flow
        flow = Flow(self.velocity_fn, self.n_steps, self.integrator_name)
        traj = flow.forward_trajectory(flat_x0)
        
        return self._unflatten_traj(traj, active_blocks)

    def trajectory_backward(self, x1: Union[Dict, jnp.ndarray], active_blocks: List[str] = None, conditions=None, **kwargs):
        """
        Computes the backward trajectory from x1.
        """
        # Determine active blocks if not provided
        if active_blocks is None:
            if isinstance(x1, dict):
                active_blocks = self._get_active_blocks(x1)
            else:
                active_blocks = self.velocity_fn.structure.block_order

        if isinstance(x1, dict):
            flat_x1 = self._flatten_subset(x1, self.velocity_fn.structure.block_order)
        else:
            flat_x1 = x1

        flow = Flow(self.velocity_fn, self.n_steps, self.integrator_name)
        traj = flow.backward_trajectory(flat_x1)
        
        return self._unflatten_traj(traj, active_blocks)
